import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import yaml

from src.utils.ppl_utils import load_ppl_model, compute_ppl


def _mean(xs) -> float:
    # 支持 list / np.ndarray
    if isinstance(xs, np.ndarray):
        if xs.size == 0:
            return 0.0
        return float(xs.mean())
    # list 等可迭代
    xs = list(xs)
    return sum(xs) / len(xs) if len(xs) > 0 else 0.0


def _std(xs) -> float:
    if isinstance(xs, np.ndarray):
        if xs.size == 0:
            return 0.0
        return float(xs.std())
    xs = list(xs)
    if len(xs) == 0:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ======================= 0. 模型名称推断 =======================

def infer_model_name(score_path: Path, judged_path: Path) -> str:
    """
    从 score_with_D / qwen_judged 的 JSONL 中尽量推断模型名称。
    依次尝试字段: mllm_name / model_name / model / target_model。
    推断失败则返回 'unknown_model'。
    """
    def _scan_one(p: Path) -> str | None:
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    return None
                for k in ["mllm_name", "model_name", "model", "target_model"]:
                    v = obj.get(k)
                    if isinstance(v, str) and v:
                        # 清理为文件夹安全名
                        safe = "".join(
                            c if c.isalnum() or c in "-._" else "_"
                            for c in v
                        )
                        return safe or None
                return None
        return None

    for p in [score_path, judged_path]:
        name = _scan_one(p)
        if name:
            return name
    return "unknown_model"


# ======================= 1. 读取 / 对齐数据 =======================

def load_risk_and_D(path: Path) -> Dict[str, Dict[str, float]]:
    """
    针对 inconsistency_with_D_qwen.jsonl 的专用 loader.

    文件格式：每行一个 (id, mode)，形如：
      {
        "id": 0,
        "mode": "txt_img" | "txt_only" | "img_only" | "none",
        "cross_modal_D": 0.8655,          # 仅 txt_img 行有
        "qwen_judge": {
          "toxicity": 0.0~100.0,
          "is_attack_success": ...,
          "is_refusal": ...,
          ...
        },
        ...
      }

    这里聚合到 id 粒度，返回：
      data[id] = {
        "R_tv":  float in [0,1],   # R(text+image)
        "R_t0":  float in [0,1],   # R(text only)
        "R_0v":  float in [0,1],   # R(image only)
        "D":     cross-modal inconsistency (from txt_img 行的 cross_modal_D)
      }
    """

    data: Dict[str, Dict[str, float]] = {}
    n_lines = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)

            cid = str(obj.get("id"))
            mode = obj.get("mode")
            if cid is None or mode is None:
                continue

            # 为该 id 初始化默认结构
            rec = data.setdefault(
                cid,
                {
                    "R_tv": 0.0,   # txt_img
                    "R_t0": 0.0,   # txt_only
                    "R_0v": 0.0,   # img_only
                    "D": 0.0,      # 不一致性
                },
            )

            # 从 qwen_judge 中取 toxicity 作为风险分数
            judge = obj.get("qwen_judge") or {}
            tox = float(judge.get("toxicity", 0.0))
            risk = tox / 100.0  # 归一化到 [0,1]

            if mode == "txt_img":
                rec["R_tv"] = risk
                # 不一致性 D：来自 cross_modal_D（如果有）
                if "cross_modal_D" in obj:
                    rec["D"] = float(obj["cross_modal_D"])
            elif mode == "txt_only":
                rec["R_t0"] = risk
            elif mode == "img_only":
                rec["R_0v"] = risk
            else:
                # "none" 或其他 mode，目前不参与 R 的协同计算
                pass

    print(
        f"[load_risk_and_D] loaded {len(data)} ids from {path} "
        f"(parsed {n_lines} lines)"
    )
    return data


def load_qwen_judge(
    judged_path: Path,
    ppl_model=None,
    ppl_tokenizer=None,
    ppl_device=None,
    ppl_max_length: int = 512,
) -> Dict[str, Dict[str, Any]]:
    """
    读取 qwen_judge 的结果，对每个 id 聚合四个 mode 的 ASR / refusal / toxicity，
    并可选地为每一条 output 计算 PPL。

    返回:
      g[id] = {
        "per_mode": {
          "txt_img": {"tox":0-100, "asr":0/1, "refusal":0/1, "ppl":float|None},
          "txt_only": {...},
          "img_only": {...},
          "none": {...},
        }
      }
    """
    grouped: Dict[str, Dict[str, Any]] = {}
    n_lines = 0

    with judged_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            obj = json.loads(line)
            cid = str(obj.get("id"))
            mode = obj.get("mode")
            if cid is None or mode is None:
                continue

            jq = obj.get("qwen_judge") or {}
            tox = float(jq.get("toxicity", 0.0))
            asr = bool(jq.get("is_attack_success", False))
            refusal = bool(jq.get("is_refusal", False))

            # 计算 PPL（可选）
            output_text = obj.get("output", "") or ""
            ppl_val = None
            if (
                ppl_model is not None
                and ppl_tokenizer is not None
                and ppl_device is not None
                and output_text
            ):
                try:
                    ppl_val = compute_ppl(
                        model=ppl_model,
                        tokenizer=ppl_tokenizer,
                        text=output_text,
                        device=ppl_device,
                        max_length=ppl_max_length,
                    )
                except Exception:
                    # PPL 失败不致命，直接忽略
                    ppl_val = None

            grouped.setdefault(cid, {}).setdefault("per_mode", {})[mode] = {
                "tox": tox,
                "asr": int(asr),
                "refusal": int(refusal),
                "ppl": ppl_val,
            }

    print(
        f"[load_qwen_judge] loaded {len(grouped)} ids from {judged_path} "
        f"(parsed {n_lines} lines)"
    )
    return grouped


# ======================= 2. 构造特征 S, I 及标签 + PPL =======================

def build_samples(
    riskD: Dict[str, Dict[str, Any]],
    judge: Dict[str, Dict[str, Any]],
    risk_label_thr: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    构建:
      - X: 特征 [S, I, S*I]
      - y_asr: 是否 attack success (txt_img)
      - y_risk: 是否 R_tv >= risk_label_thr
      - y_refusal: txt_img 是否拒答
      - kept_ids: 为了后续可选调试
      - ppl_txtimg: 对应 txt_img 输出的 PPL (float, 可为 NaN 表示缺失)
    """

    X_list: List[List[float]] = []
    y_asr_list: List[int] = []
    y_risk_list: List[int] = []
    y_refusal_list: List[int] = []
    ppl_list: List[float] = []
    kept_ids: List[str] = []

    # 仅对两边都存在 id 的样本做分析
    common_ids = sorted(set(riskD.keys()) & set(judge.keys()))
    print(f"[build_samples] intersect ids = {len(common_ids)}")

    # 归一化 D(x)
    Ds = [riskD[cid]["D"] for cid in common_ids]
    D_min, D_max = min(Ds), max(Ds)
    D_range = max(D_max - D_min, 1e-6)

    for cid in common_ids:
        r = riskD[cid]
        j = judge[cid]["per_mode"]

        # 如果缺 txt_img，跳过
        if "txt_img" not in j:
            continue

        R_tv = float(r["R_tv"])
        R_t0 = float(r["R_t0"])
        R_0v = float(r["R_0v"])

        # 协同强度 S = max(0, R_tv - max(R_t0, R_0v))
        base_single = max(R_t0, R_0v)
        S = max(0.0, R_tv - base_single)

        # 归一化不一致度 I \in [0,1]
        D_raw = r["D"]
        I = (D_raw - D_min) / D_range

        # 交互项
        SI = S * I

        # 标签
        asr_tv = int(j["txt_img"]["asr"])
        risk_tv = int(R_tv >= risk_label_thr)
        refusal_tv = int(j["txt_img"]["refusal"])

        # txt_img 的 PPL（可选）
        ppl_raw = j["txt_img"].get("ppl", None)
        if isinstance(ppl_raw, (int, float)) and ppl_raw > 0.0:
            ppl_val = float(ppl_raw)
        else:
            ppl_val = float("nan")

        X_list.append([S, I, SI])
        y_asr_list.append(asr_tv)
        y_risk_list.append(risk_tv)
        y_refusal_list.append(refusal_tv)
        ppl_list.append(ppl_val)
        kept_ids.append(cid)

    X = np.asarray(X_list, dtype=float)
    y_asr = np.asarray(y_asr_list, dtype=int)
    y_risk = np.asarray(y_risk_list, dtype=int)
    y_refusal = np.asarray(y_refusal_list, dtype=int)
    ppl_txtimg = np.asarray(ppl_list, dtype=float)

    print(f"[build_samples] final samples = {len(kept_ids)}")

    if len(kept_ids) > 0:
        S = X[:, 0]
        I = X[:, 1]
        print(
            "[build_samples] S stats: "
            f"min={S.min():.4f}, max={S.max():.4f}, mean={S.mean():.4f}, std={S.std():.4f}"
        )
        print(
            "[build_samples] I stats: "
            f"min={I.min():.4f}, max={I.max():.4f}, mean={I.mean():.4f}, std={I.std():.4f}"
        )
        print(
            "[build_samples] y_asr: "
            f"pos={y_asr.sum()}, neg={len(y_asr) - y_asr.sum()}"
        )
        print(
            "[build_samples] y_risk: "
            f"pos={y_risk.sum()}, neg={len(y_risk) - y_risk.sum()}"
        )
        print(
            "[build_samples] y_refusal: "
            f"pos={y_refusal.sum()}, neg={len(y_refusal) - y_refusal.sum()}"
        )
        # PPL 简要统计（仅提示，完整放到 summary）
        mask_ppl = np.isfinite(ppl_txtimg) & (ppl_txtimg > 0)
        if mask_ppl.any():
            vals = ppl_txtimg[mask_ppl]
            print(
                "[build_samples] PPL(txt_img) stats: "
                f"n={len(vals)}, mean={vals.mean():.2f}, std={vals.std():.2f}, "
                f"min={vals.min():.2f}, max={vals.max():.2f}"
            )
        else:
            print("[build_samples] PPL(txt_img): no valid values")

    return X, y_asr, y_risk, y_refusal, kept_ids, ppl_txtimg


# ======================= 3. Logistic 回归 + 基础可视化 =======================

def fit_logreg(X: np.ndarray, y: np.ndarray, name: str) -> LogisticRegression | None:
    """如果标签只有一个类别，直接跳过并返回 None。"""
    uniq = np.unique(y)
    if len(uniq) < 2:
        print(f"[logreg-{name}] only one class {uniq[0]}, skip logistic regression.")
        return None

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=200,
    )
    clf.fit(X, y)
    return clf


def summarize_logreg(name: str, clf: LogisticRegression):
    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]
    print(f"[logreg-{name}] intercept = {intercept:.4f}")
    print(
        f"[logreg-{name}] coef (S, I, S*I) = "
        f"{coefs[0]:.4f}, {coefs[1]:.4f}, {coefs[2]:.4f}"
    )


def plot_surface(
    X: np.ndarray,
    clf: LogisticRegression,
    out_path: Path,
    title: str,
    grid_size: int = 50,
):
    """
    画出在 (S, I) 平面上的概率曲面（S*I 自动纳入特征）。
    """
    S = X[:, 0]
    I = X[:, 1]

    S_min, S_max = float(S.min()), float(S.max())
    I_min, I_max = float(I.min()), float(I.max())

    # 给一点 padding
    S_pad = 0.05 * (S_max - S_min + 1e-6)
    I_pad = 0.05 * (I_max - I_min + 1e-6)

    s_grid = np.linspace(S_min - S_pad, S_max + S_pad, grid_size)
    i_grid = np.linspace(I_min - I_pad, I_max + I_pad, grid_size)
    SS, II = np.meshgrid(s_grid, i_grid)

    SI = SS * II
    grid_X = np.stack([SS.ravel(), II.ravel(), SI.ravel()], axis=1)
    prob = clf.predict_proba(grid_X)[:, 1].reshape(SS.shape)

    plt.figure(figsize=(7, 5))
    cp = plt.contourf(SS, II, prob, levels=20)
    plt.colorbar(cp, label="Predicted probability")
    plt.xlabel("Synergy strength S")
    plt.ylabel("Inconsistency I")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


# ======================= 4. 新增：散点 & 1D 切片 =======================

def plot_scatter_SI(
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    title: str,
    label_name: str,
):
    """
    (S, I) 的散点图，按标签 y 上色。y 为 0/1。
    """
    S = X[:, 0]
    I = X[:, 1]

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    plt.figure(figsize=(6, 5))
    plt.scatter(S[neg_mask], I[neg_mask], alpha=0.4, label=f"{label_name}=0")
    plt.scatter(S[pos_mask], I[pos_mask], alpha=0.8, label=f"{label_name}=1", marker="x")
    plt.xlabel("Synergy strength S")
    plt.ylabel("Inconsistency I")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_slices_over_S(
    X: np.ndarray,
    clf: LogisticRegression,
    out_path: Path,
    title: str,
    fixed_I_list: List[float] | None = None,
):
    """
    固定若干 I 值，画 P(y=1 | S, I=I0) 随 S 的变化曲线。
    """
    if fixed_I_list is None:
        fixed_I_list = [0.0, 0.5, 1.0]

    S = X[:, 0]
    S_min, S_max = float(S.min()), float(S.max())
    S_pad = 0.05 * (S_max - S_min + 1e-6)
    s_grid = np.linspace(S_min - S_pad, S_max + S_pad, 200)

    plt.figure(figsize=(7, 5))

    for I0 in fixed_I_list:
        I_vec = np.full_like(s_grid, fill_value=I0, dtype=float)
        SI_vec = s_grid * I_vec
        grid_X = np.stack([s_grid, I_vec, SI_vec], axis=1)
        prob = clf.predict_proba(grid_X)[:, 1]
        plt.plot(s_grid, prob, label=f"I = {I0:.2f}")

    plt.xlabel("Synergy strength S")
    plt.ylabel("Predicted probability")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


# ======================= 5. J-score 相关 =======================

def compute_J(X: np.ndarray, lam: float = 1.0) -> np.ndarray:
    S = X[:, 0]
    I = X[:, 1]
    J = S * (1.0 + lam * I)
    return J


def plot_J_hist(
    X: np.ndarray,
    out_path: Path,
    lam: float = 1.0,
):
    """
    J = S * (1 + lam * I) 的分布直方图。
    """
    J = compute_J(X, lam=lam)

    plt.figure(figsize=(6, 4))
    plt.hist(J, bins=40, density=True)
    plt.xlabel(f"J = S * (1 + {lam} * I)")
    plt.ylabel("density")
    plt.title("Distribution of joint amplification score J")
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_bucket_bars(
    X: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    n_bins: int = 3,
    title: str = "ASR vs J(x) bucket",
    lam: float = 1.0,
):
    """
    把 J 分成 n_bins 桶，画出每个桶中的 y=1 率（例如 ASR 率）。
    """
    J = compute_J(X, lam=lam)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(J, qs)

    pos_rates: List[float] = []
    labels: List[str] = []

    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == 0:
            mask = (J >= lo) & (J <= hi)
        else:
            mask = (J > lo) & (J <= hi)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            pos_rates.append(0.0)
            labels.append(f"bin{b}")
            continue
        rate = float(y[idx].mean())
        pos_rates.append(rate)
        labels.append(f"{lo:.3f}–{hi:.3f}\n(n={len(idx)})")

    plt.figure(figsize=(7, 4))
    x = np.arange(n_bins)
    plt.bar(x, pos_rates)
    plt.xticks(x, labels)
    plt.ylabel("Positive rate")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


# ======================= 6. 新增：PPL 相关图 =======================

def plot_ppl_histogram(
    ppl: np.ndarray,
    out_path: Path,
    prefix: str = "txtimg"
):

    mask = np.isfinite(ppl) & (ppl > 0)
    if not mask.any():
        print("[plot] skip PPL histogram: no valid PPL")
        return

    vals = ppl[mask]

    # (1) log10 直方图
    log_vals = np.log10(vals)

    plt.figure(figsize=(6, 4))
    plt.hist(log_vals, bins=40, density=True, alpha=0.85)
    plt.xlabel("log10(PPL)")
    plt.ylabel("Density")
    plt.title(f"log-scale PPL distribution ({prefix})")
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_ppl_vs_J_scatter(
    X: np.ndarray,
    ppl: np.ndarray,
    out_path: Path,
    lam: float = 1.0,
    title: str = "PPL vs joint amplification J(x)",
):
    mask = np.isfinite(ppl) & (ppl > 0)
    if not mask.any():
        print("[plot] skip PPL vs J scatter: no valid PPL")
        return

    J = compute_J(X, lam=lam)
    Jv = J[mask]
    Pv = ppl[mask]

    plt.figure(figsize=(6, 4))
    plt.scatter(Jv, Pv, s=18, alpha=0.5)
    plt.xlabel(f"J = S * (1 + {lam} * I)")
    plt.ylabel("Perplexity (PPL)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_ppl_vs_label_boxplot(
    ppl: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    label_name: str = "ASR",
    title: str | None = None,
):
    """
    按标签 y 划分 PPL 分布，画 boxplot。
    """
    mask = np.isfinite(ppl) & (ppl > 0)
    if not mask.any():
        print(f"[plot] skip PPL vs {label_name} boxplot: no valid PPL")
        return

    ppl = ppl[mask]
    y = y[mask]

    vals0 = ppl[y == 0]
    vals1 = ppl[y == 1]

    data = []
    labels = []
    if len(vals0) > 0:
        data.append(vals0)
        labels.append(f"{label_name}=0")
    if len(vals1) > 0:
        data.append(vals1)
        labels.append(f"{label_name}=1")

    if not data:
        print(f"[plot] skip PPL vs {label_name} boxplot: no data after split")
        return

    if title is None:
        title = f"PPL vs {label_name} (txt+img)"

    plt.figure(figsize=(5, 4))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.ylabel("Perplexity (PPL)")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


# ======================= 7. 主流程 =======================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--score_with_D",
        required=True,
        help="JSONL with R and D(x), e.g. inconsistency_with_D_qwen.jsonl",
    )
    ap.add_argument(
        "--qwen_judged",
        required=True,
        help="JSONL with qwen judge results, e.g. inconsistency_with_D_qwen.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="root directory to save summary and plots; "
             "actual path will be out_dir/<model_name>/joint/...",
    )
    ap.add_argument(
        "--risk_label_thr",
        type=float,
        default=0.8,
        help="threshold on R_tv to define high-risk label",
    )
    args = ap.parse_args()

    score_path = Path(args.score_with_D)
    judged_path = Path(args.qwen_judged)

    # 统一绘图风格
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (5.5, 4.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # 推断模型名称，并构造模型专属目录：out_dir_root / model_name / joint
    out_root = Path(args.out_dir)
    model_name = infer_model_name(score_path, judged_path)
    model_dir = out_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[main] resolved model name: {model_name}")
    print(f"[main] metrics dir: {model_dir}")

    # 子目录：与图表类型对齐
    surfaces_dir = model_dir / "surfaces"
    slices_dir = model_dir / "slices"
    scatter_dir = model_dir / "scatter"
    J_dir = model_dir / "J"
    ppl_dir = model_dir / "ppl"
    for d in [surfaces_dir, slices_dir, scatter_dir, J_dir, ppl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 尝试加载本地 PPL 模型（例如 Qwen3-0.6B）
    ppl_model = ppl_tokenizer = ppl_device = None
    try:
        with open("configs/models.yaml", "r", encoding="utf-8") as f:
            models_cfg = yaml.safe_load(f)
        with open("configs/runtime.yaml", "r", encoding="utf-8") as f:
            runtime_cfg = yaml.safe_load(f)
        ppl_model, ppl_tokenizer, ppl_device = load_ppl_model(
            models_cfg=models_cfg,
            model_name="qwen25_0_5b_ppl",  # 请在 models.yaml 中配置这一项
            runtime_cfg=runtime_cfg,
        )
        print("[PPL] loaded local PPL model qwen25_0_5b_ppl.")
    except Exception as e:
        print(
            f"[PPL] WARNING: failed to load PPL model; PPL metrics will be limited. "
            f"Error: {e}"
        )

    riskD = load_risk_and_D(score_path)
    judge = load_qwen_judge(
        judged_path,
        ppl_model=ppl_model,
        ppl_tokenizer=ppl_tokenizer,
        ppl_device=ppl_device,
        ppl_max_length=512,
    )

    X, y_asr, y_risk, y_refusal, kept_ids, ppl_txtimg = build_samples(
        riskD, judge, risk_label_thr=args.risk_label_thr
    )

    # ---------- 1) ASR 回归 ----------
    clf_asr = fit_logreg(X, y_asr, name="asr")
    if clf_asr is not None:
        summarize_logreg("asr", clf_asr)
        plot_surface(
            X,
            clf_asr,
            out_path=surfaces_dir / "surface_asr_SI.png",
            title="Predicted ASR prob over (S, I)",
        )
        plot_slices_over_S(
            X,
            clf_asr,
            out_path=slices_dir / "slices_ASR_over_S.png",
            title="Predicted ASR prob vs S for fixed I",
        )

    # ---------- 2) 高风险回归 ----------
    clf_risk = fit_logreg(X, y_risk, name="risk")
    if clf_risk is not None:
        summarize_logreg("risk", clf_risk)
        plot_surface(
            X,
            clf_risk,
            out_path=surfaces_dir / "surface_risk_SI.png",
            title=f"Predicted P(R_txt_img >= {args.risk_label_thr}) over (S, I)",
        )
        plot_slices_over_S(
            X,
            clf_risk,
            out_path=slices_dir / "slices_risk_over_S.png",
            title=f"Predicted P(R_txt_img >= {args.risk_label_thr}) vs S for fixed I",
        )

    # ---------- 3) Refusal 回归 ----------
    clf_refusal = fit_logreg(X, y_refusal, name="refusal")
    if clf_refusal is not None:
        summarize_logreg("refusal", clf_refusal)
        plot_surface(
            X,
            clf_refusal,
            out_path=surfaces_dir / "surface_refusal_SI.png",
            title="Predicted refusal prob over (S, I)",
        )
        plot_slices_over_S(
            X,
            clf_refusal,
            out_path=slices_dir / "slices_refusal_over_S.png",
            title="Predicted refusal prob vs S for fixed I",
        )

    # ---------- 4) 散点图（真实标签分布） ----------
    plot_scatter_SI(
        X,
        y_asr,
        out_path=scatter_dir / "scatter_SI_ASR.png",
        title="ASR labels over (S, I)",
        label_name="ASR",
    )
    plot_scatter_SI(
        X,
        y_risk,
        out_path=scatter_dir / "scatter_SI_risk.png",
        title=f"High-risk (R_txt_img >= {args.risk_label_thr}) labels over (S, I)",
        label_name="HighRisk",
    )
    plot_scatter_SI(
        X,
        y_refusal,
        out_path=scatter_dir / "scatter_SI_refusal.png",
        title="Refusal labels over (S, I)",
        label_name="Refusal",
    )

    # ---------- 5) J(x) 直方图 + bucket 柱状图（以 ASR 为例） ----------
    plot_J_hist(
        X,
        out_path=J_dir / "hist_J.png",
        lam=1.0,
    )
    plot_bucket_bars(
        X,
        y_asr,
        out_path=J_dir / "bars_bucket_ASR.png",
        title="ASR rate by J(x) bucket",
        lam=1.0,
    )

    # ---------- 6) PPL 相关图（joint 视角） ----------
    plot_ppl_histogram(
        ppl_txtimg,
        out_path=ppl_dir / "ppl_hist_log_txtimg.png",
    )
    plot_ppl_vs_J_scatter(
        X,
        ppl_txtimg,
        out_path=ppl_dir / "ppl_vs_J_scatter.png",
        lam=1.0,
        title="PPL vs joint amplification J(x) (txt+img)",
    )
    plot_ppl_vs_label_boxplot(
        ppl_txtimg,
        y_asr,
        out_path=ppl_dir / "ppl_vs_ASR_boxplot_txtimg.png",
        label_name="ASR",
        title="PPL vs Attack Success (txt+img)",
    )

    # ---------- 7) 写 summary ----------
    summary: Dict[str, Any] = {
        "model_name": model_name,
        "n_samples": int(len(kept_ids)),
        "risk_label_thr": float(args.risk_label_thr),
    }
    if clf_asr is not None:
        summary["coef_asr"] = {
            "intercept": float(clf_asr.intercept_[0]),
            "S": float(clf_asr.coef_[0][0]),
            "I": float(clf_asr.coef_[0][1]),
            "SxI": float(clf_asr.coef_[0][2]),
        }
    if clf_risk is not None:
        summary["coef_risk"] = {
            "intercept": float(clf_risk.intercept_[0]),
            "S": float(clf_risk.coef_[0][0]),
            "I": float(clf_risk.coef_[0][1]),
            "SxI": float(clf_risk.coef_[0][2]),
        }
    if clf_refusal is not None:
        summary["coef_refusal"] = {
            "intercept": float(clf_refusal.intercept_[0]),
            "S": float(clf_refusal.coef_[0][0]),
            "I": float(clf_refusal.coef_[0][1]),
            "SxI": float(clf_refusal.coef_[0][2]),
        }

    # PPL 统计写入 summary
    mask_ppl = np.isfinite(ppl_txtimg) & (ppl_txtimg > 0)
    ppl_stats: Dict[str, Any] = {
        "has_ppl": bool(mask_ppl.any()),
    }
    if mask_ppl.any():
        vals = ppl_txtimg[mask_ppl]
        ppl_stats.update(
            {
                "n": int(len(vals)),
                "mean": float(_mean(vals)),
                "std": float(_std(vals)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )

        # 按 ASR 标签分组
        ppl_asr0 = vals[y_asr[mask_ppl] == 0]
        ppl_asr1 = vals[y_asr[mask_ppl] == 1]
        ppl_stats["by_ASR"] = {
            "ASR=0": {
                "n": int(len(ppl_asr0)),
                "mean": float(_mean(ppl_asr0)) if len(ppl_asr0) > 0 else 0.0,
                "std": float(_std(ppl_asr0)) if len(ppl_asr0) > 0 else 0.0,
            },
            "ASR=1": {
                "n": int(len(ppl_asr1)),
                "mean": float(_mean(ppl_asr1)) if len(ppl_asr1) > 0 else 0.0,
                "std": float(_std(ppl_asr1)) if len(ppl_asr1) > 0 else 0.0,
            },
        }

    summary["ppl_stats_txtimg"] = ppl_stats

    summary_path = model_dir / "joint_logreg_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summary] written to {summary_path}")


if __name__ == "__main__":
    main()
