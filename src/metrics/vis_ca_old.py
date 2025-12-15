
# scripts/vis_llava15_cross_attn.py

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import yaml
from tqdm import tqdm
from src.models.registry import build_model
from src.models.base import MLLM  # 仅用于类型标注，可选


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xm = x.mean()
    ym = y.mean()
    vx = x - xm
    vy = y - ym
    num = float((vx * vy).sum())
    den = float(np.sqrt((vx * vx).sum() * (vy * vy).sum()))
    return num / den if den > 0 else float("nan")


def _extract_asr(obj: Dict[str, Any]) -> Optional[float]:
    """
    尝试从样本中抽取 ASR（0/1 或概率），兼容多种字段命名。
    若找不到则返回 None。
    """
    obj = obj["qwen_judge"]

    # 直接数值型字段
    for key in ("asr", "attack_success", "is_attack_success"):
        if key in obj:
            v = obj[key]
            try:
                return float(v)
            except Exception:
                pass

    # 可能是 dict，按模式区分
    for key in ("asr_by_mode", "metrics", "attack"):
        if key in obj and isinstance(obj[key], dict):
            d = obj[key]
            for subk in ("txt_img", "txt+img", "both"):
                if subk in d:
                    try:
                        return float(d[subk])
                    except Exception:
                        continue

    # 特定命名
    if "txt_img_asr" in obj:
        try:
            return float(obj["txt_img_asr"])
        except Exception:
            pass

    return None


def load_samples_with_D(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "cross_modal_D" not in obj:
                continue
            if "image" not in obj or "prompt" not in obj:
                continue
            samples.append(obj)
    return samples


# ----------------------------------------------------------------------
# 绘图函数：mean 曲线
# ----------------------------------------------------------------------


def _plot_layer_curves(
    stats_by_bucket: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}

    # t->i
    plt.figure(figsize=(6.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        y = stats.get("mean_t2i", [])
        if not y:
            continue
        x = list(range(len(y)))
        plt.plot(x, y, label=f"{bname} D (t->i)", color=colors.get(bname, None))
    plt.xlabel("Layer index")
    plt.ylabel("Mean cross-attn (text -> image)")
    plt.title("LLaVA-1.5 cross-attn (t->i) vs layer\nfor low/mid/high D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_cross_attn_t2i_layers.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # i->t
    plt.figure(figsize=(6.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        y = stats.get("mean_i2t", [])
        if not y:
            continue
        x = list(range(len(y)))
        plt.plot(x, y, label=f"{bname} D (i->t)", color=colors.get(bname, None))
    plt.xlabel("Layer index")
    plt.ylabel("Mean cross-attn (image -> text)")
    plt.title("LLaVA-1.5 cross-attn (i->t) vs layer\nfor low/mid/high D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_cross_attn_i2t_layers.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


# ----------------------------------------------------------------------
# 绘图函数：layer-wise std 曲线（跨样本）
# ----------------------------------------------------------------------


def _plot_layer_std_curves(
    stats_by_bucket: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}

    # t->i std
    plt.figure(figsize=(6.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        y = stats.get("std_t2i", [])
        if not y:
            continue
        x = list(range(len(y)))
        plt.plot(x, y, label=f"{bname} D (t->i std)", color=colors.get(bname, None))
    plt.xlabel("Layer index")
    plt.ylabel("Std of cross-attn (text -> image)")
    plt.title("LLaVA-1.5 cross-attn std (t->i) vs layer\nfor low/mid/high D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_cross_attn_t2i_layers_std.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # i->t std
    plt.figure(figsize=(6.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        y = stats.get("std_i2t", [])
        if not y:
            continue
        x = list(range(len(y)))
        plt.plot(x, y, label=f"{bname} D (i->t std)", color=colors.get(bname, None))
    plt.xlabel("Layer index")
    plt.ylabel("Std of cross-attn (image -> text)")
    plt.title("LLaVA-1.5 cross-attn std (i->t) vs layer\nfor low/mid/high D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_cross_attn_i2t_layers_std.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


# ----------------------------------------------------------------------
# 绘图函数：D vs mean-attn / D vs std 散点
# ----------------------------------------------------------------------


def _plot_scatter_D_vs_attn(
    D_all: List[float],
    mean_i2t_all: List[float],
    mean_t2i_all: List[float],
    bucket_ids: List[str],
    out_dir: Path,
) -> None:
    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}

    # i->t
    plt.figure(figsize=(6.0, 4.0))
    for bname in ("low", "mid", "high"):
        xs = [d for d, b in zip(D_all, bucket_ids) if b == bname]
        ys = [m for m, b in zip(mean_i2t_all, bucket_ids) if b == bname]
        if not xs:
            continue
        plt.scatter(xs, ys, label=bname, alpha=0.8, s=24, color=colors.get(bname, None))
    plt.xlabel("cross_modal_D")
    plt.ylabel("mean cross-attn (i->t)")
    plt.title("D(x) vs mean cross-attn (image -> text)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_D_vs_mean_i2t_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # t->i
    plt.figure(figsize=(6.0, 4.0))
    for bname in ("low", "mid", "high"):
        xs = [d for d, b in zip(D_all, bucket_ids) if b == bname]
        ys = [m for m, b in zip(mean_t2i_all, bucket_ids) if b == bname]
        if not xs:
            continue
        plt.scatter(xs, ys, label=bname, alpha=0.8, s=24, color=colors.get(bname, None))
    plt.xlabel("cross_modal_D")
    plt.ylabel("mean cross-attn (t->i)")
    plt.title("D(x) vs mean cross-attn (text -> image)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_D_vs_mean_t2i_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


def _plot_scatter_D_vs_std(
    D_all: List[float],
    layer_std_i2t_all: List[float],
    layer_std_t2i_all: List[float],
    bucket_ids: List[str],
    out_dir: Path,
) -> None:
    """散点：D vs per-sample layer-std（i->t / t->i）"""
    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}

    # i->t std
    plt.figure(figsize=(6.0, 4.0))
    for bname in ("low", "mid", "high"):
        xs = [d for d, b in zip(D_all, bucket_ids) if b == bname]
        ys = [s for s, b in zip(layer_std_i2t_all, bucket_ids) if b == bname]
        if not xs:
            continue
        plt.scatter(xs, ys, label=bname, alpha=0.8, s=24, color=colors.get(bname, None))
    plt.xlabel("cross_modal_D")
    plt.ylabel("layer-wise std of cross-attn (i->t)")
    plt.title("D(x) vs layer-wise std (image -> text)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_D_vs_layer_std_i2t_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # t->i std
    plt.figure(figsize=(6.0, 4.0))
    for bname in ("low", "mid", "high"):
        xs = [d for d, b in zip(D_all, bucket_ids) if b == bname]
        ys = [s for s, b in zip(layer_std_t2i_all, bucket_ids) if b == bname]
        if not xs:
            continue
        plt.scatter(xs, ys, label=bname, alpha=0.8, s=24, color=colors.get(bname, None))
    plt.xlabel("cross_modal_D")
    plt.ylabel("layer-wise std of cross-attn (t->i)")
    plt.title("D(x) vs layer-wise std (text -> image)")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_D_vs_layer_std_t2i_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


# ----------------------------------------------------------------------
# 绘图函数：mean / layer-std 直方图
# ----------------------------------------------------------------------


def _plot_histograms(
    stats_by_bucket: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    colors = {"low": "tab:blue", "mid": "tab:orange", "high": "tab:green"}

    # i->t mean
    plt.figure(figsize=(7.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        data = stats.get("all_mean_i2t", [])
        if not data:
            continue
        plt.hist(
            data,
            bins=20,
            alpha=0.5,
            density=True,
            label=bname,
            color=colors.get(bname, None),
        )
    plt.xlabel("mean cross-attn (i->t)")
    plt.ylabel("Density")
    plt.title("Distribution of mean cross-attn (i->t) by D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_mean_i2t_hist_by_bucket.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # t->i mean
    plt.figure(figsize=(7.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        data = stats.get("all_mean_t2i", [])
        if not data:
            continue
        plt.hist(
            data,
            bins=20,
            alpha=0.5,
            density=True,
            label=bname,
            color=colors.get(bname, None),
        )
    plt.xlabel("mean cross-attn (t->i)")
    plt.ylabel("Density")
    plt.title("Distribution of mean cross-attn (t->i) by D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_mean_t2i_hist_by_bucket.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # -------- layer-wise std 直方图（per-sample） --------
    # i->t layer std
    plt.figure(figsize=(7.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        data = stats.get("layer_std_i2t_list", [])
        if not data:
            continue
        plt.hist(
            data,
            bins=20,
            alpha=0.5,
            density=True,
            label=bname,
            color=colors.get(bname, None),
        )
    plt.xlabel("per-sample layer-wise std (i->t)")
    plt.ylabel("Density")
    plt.title("Distribution of layer-wise std (i->t) by D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_layer_std_i2t_hist_by_bucket.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # t->i layer std
    plt.figure(figsize=(7.0, 4.0))
    for bname, stats in stats_by_bucket.items():
        data = stats.get("layer_std_t2i_list", [])
        if not data:
            continue
        plt.hist(
            data,
            bins=20,
            alpha=0.5,
            density=True,
            label=bname,
            color=colors.get(bname, None),
        )
    plt.xlabel("per-sample layer-wise std (t->i)")
    plt.ylabel("Density")
    plt.title("Distribution of layer-wise std (t->i) by D buckets")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path = out_dir / "llava15_layer_std_t2i_hist_by_bucket.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


# ----------------------------------------------------------------------
# 绘图函数：基于 (D, attn) -> ASR 的热图 / 曲面
# ----------------------------------------------------------------------


def _plot_D_attn_asr_surface(
    Ds: List[float],
    attn_vals: List[float],
    asr_vals: List[Optional[float]],
    out_dir: Path,
    direction: str = "i2t",
) -> None:
    """
    基于 (D, attn) -> ASR 的网格，画 2D heatmap + 3D surface。
    direction: "i2t" 或 "t2i"，仅用于文件名和标题。
    """
    # 过滤掉无 ASR 的样本
    Ds_arr = np.asarray([d for d, a in zip(Ds, asr_vals) if a is not None], dtype=float)
    A_arr = np.asarray([v for v, a in zip(attn_vals, asr_vals) if a is not None], dtype=float)
    Z_arr = np.asarray([a for a in asr_vals if a is not None], dtype=float)

    if Ds_arr.size < 10:
        print("[warn] not enough samples with ASR to build surface, skip")
        return

    n_bins_D = 6
    n_bins_A = 6
    D_edges = np.linspace(Ds_arr.min(), Ds_arr.max(), n_bins_D + 1)
    A_edges = np.linspace(A_arr.min(), A_arr.max(), n_bins_A + 1)

    D_idx = np.digitize(Ds_arr, D_edges) - 1
    A_idx = np.digitize(A_arr, A_edges) - 1

    Z = np.full((n_bins_D, n_bins_A), np.nan, dtype=float)

    for i in range(n_bins_D):
        for j in range(n_bins_A):
            mask = (D_idx == i) & (A_idx == j)
            if mask.sum() >= 3:
                Z[i, j] = float(Z_arr[mask].mean())

    D_centers = 0.5 * (D_edges[:-1] + D_edges[1:])
    A_centers = 0.5 * (A_edges[:-1] + A_edges[1:])
    DD, AA = np.meshgrid(D_centers, A_centers, indexing="ij")

    # 2D heatmap
    plt.figure(figsize=(6.0, 4.0))
    im = plt.pcolormesh(
        DD,
        AA,
        Z,
        cmap="viridis",
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    plt.xlabel("cross_modal_D")
    ylabel = (
        "mean cross-attn (i->t)"
        if direction == "i2t"
        else "mean cross-attn (t->i)"
    )
    plt.ylabel(ylabel)
    plt.title(f"ASR heatmap over (D, {direction})")
    plt.colorbar(im, label="ASR")
    plt.tight_layout()
    out_path = out_dir / f"llava15_D_{direction}_ASR_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")

    # 3D surface
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.0, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(DD, AA, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("cross_modal_D")
    ax.set_ylabel(ylabel)
    ax.set_zlabel("ASR")
    ax.set_title(f"ASR surface over (D, {direction})")
    plt.tight_layout()
    out_path = out_dir / f"llava15_D_{direction}_ASR_surface.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[plot] saved {out_path}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw",
        type=str,
        default="outputs/metrics/inconsistency_with_D_qwen.jsonl",
        help="jsonl with cross_modal_D / image / prompt / (optional) ASR",
    )
    ap.add_argument(
        "--models-cfg",
        type=str,
        default="configs/models.yaml",
        help="YAML config for models",
    )
    ap.add_argument(
        "--mllm-name",
        type=str,
        default="llava15_7b",
        help="model name key in models.yaml",
    )
    ap.add_argument(
        "--k-per-bucket",
        type=int,
        default=80,
        help="number of samples per D bucket (low/mid/high)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    with_D_path = Path(args.raw)
    models_cfg_path = Path(args.models_cfg)
    mllm_name = args.mllm_name
    k = args.k_per_bucket

    out_dir = Path("outputs/metrics/vis_cross_attn/mini")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = out_dir / mllm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载样本
    samples = load_samples_with_D(with_D_path)
    print(f"[load] loaded {len(samples)} samples with D from {with_D_path}")
    if not samples:
        return

    # 按 D 排序
    samples_sorted = sorted(samples, key=lambda s: float(s["cross_modal_D"]))
    n = len(samples_sorted)

    low_samples = samples_sorted[: min(k, n)]
    mid_center = n // 2
    mid_half = min(k // 2, mid_center)
    mid_samples = samples_sorted[mid_center - mid_half : mid_center + (k - mid_half)]
    high_samples = samples_sorted[max(0, n - k) :]

    buckets: Dict[str, List[Dict[str, Any]]] = {
        "low": low_samples,
        "mid": mid_samples,
        "high": high_samples,
    }

    # 2. 构建 Llava wrapper
    with models_cfg_path.open("r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    cfg = models_cfg[mllm_name]
    repo_id = cfg.get("repo_id") or cfg.get("repo_dir") or cfg.get("path")

    runtime_cfg = {
        "device": "cuda:0",
        "repr_layer": -1,  # 使用最后一层 hidden_state
    }
    print(f"[model] building {mllm_name}Wrapper from {repo_id}")
    mllm: MLLM = build_model(
        name=mllm_name,
        models_cfg=models_cfg,
        runtime_cfg=runtime_cfg,
    )

    # 3. 收集 per-layer 和 per-sample 统计
    stats_by_bucket: Dict[str, Dict[str, Any]] = {
        "low": {
            "all_t2i_layers": [],
            "all_i2t_layers": [],
            "all_mean_t2i": [],
            "all_mean_i2t": [],
            "layer_std_t2i_list": [],
            "layer_std_i2t_list": [],
        },
        "mid": {
            "all_t2i_layers": [],
            "all_i2t_layers": [],
            "all_mean_t2i": [],
            "all_mean_i2t": [],
            "layer_std_t2i_list": [],
            "layer_std_i2t_list": [],
        },
        "high": {
            "all_t2i_layers": [],
            "all_i2t_layers": [],
            "all_mean_t2i": [],
            "all_mean_i2t": [],
            "layer_std_t2i_list": [],
            "layer_std_i2t_list": [],
        },
    }

    D_all: List[float] = []
    mean_i2t_all: List[float] = []
    mean_t2i_all: List[float] = []
    asr_all: List[Optional[float]] = []
    bucket_ids: List[str] = []
    layer_std_i2t_all: List[float] = []
    layer_std_t2i_all: List[float] = []

    for bname, blist in buckets.items():
        if not blist:
            print(f"[warn] bucket {bname} is empty, skip")
            continue

        print(f"[bucket] {bname}: {len(blist)} samples")
        for obj in tqdm(blist, desc=f"bucket {bname}", leave=False):
            img_path = obj["image"]
            prompt = obj["prompt"]
            D_val = float(obj["cross_modal_D"])
            asr_val = _extract_asr(obj)

            img = Image.open(img_path).convert("RGB")

            _, layer_stats = mllm.encode_modalities_with_trace(
                image=img,
                prompt=prompt,
                gen_cfg=None,
            )

            if not layer_stats:
                continue

            t2i_vals = [s.t2i for s in layer_stats]
            i2t_vals = [s.i2t for s in layer_stats]

            stats_by_bucket[bname]["all_t2i_layers"].append(t2i_vals)
            stats_by_bucket[bname]["all_i2t_layers"].append(i2t_vals)

            mean_t2i = _mean(t2i_vals)
            mean_i2t = _mean(i2t_vals)
            stats_by_bucket[bname]["all_mean_t2i"].append(mean_t2i)
            stats_by_bucket[bname]["all_mean_i2t"].append(mean_i2t)

            # per-sample layer-wise std
            layer_std_t2i = float(np.std(np.asarray(t2i_vals, dtype=float)))
            layer_std_i2t = float(np.std(np.asarray(i2t_vals, dtype=float)))
            stats_by_bucket[bname]["layer_std_t2i_list"].append(layer_std_t2i)
            stats_by_bucket[bname]["layer_std_i2t_list"].append(layer_std_i2t)

            D_all.append(D_val)
            mean_t2i_all.append(mean_t2i)
            mean_i2t_all.append(mean_i2t)
            asr_all.append(asr_val)
            bucket_ids.append(bname)
            layer_std_t2i_all.append(layer_std_t2i)
            layer_std_i2t_all.append(layer_std_i2t)

    # 对每个 bucket 计算 layer-wise 平均曲线 + layer-wise 跨样本 std
    for bname, stats in stats_by_bucket.items():
        all_t2i_layers: List[List[float]] = stats["all_t2i_layers"]
        all_i2t_layers: List[List[float]] = stats["all_i2t_layers"]
        if not all_t2i_layers:
            print(f"[warn] bucket {bname} has no layer stats, skip")
            stats["mean_t2i"] = []
            stats["mean_i2t"] = []
            stats["std_t2i"] = []
            stats["std_i2t"] = []
            continue

        L = min(len(x) for x in all_t2i_layers)
        all_t2i_layers = [x[:L] for x in all_t2i_layers]
        all_i2t_layers = [x[:L] for x in all_i2t_layers]

        mean_t2i: List[float] = []
        mean_i2t: List[float] = []
        std_t2i: List[float] = []
        std_i2t: List[float] = []

        for l in range(L):
            t_vals = np.asarray([x[l] for x in all_t2i_layers], dtype=float)
            i_vals = np.asarray([x[l] for x in all_i2t_layers], dtype=float)
            mean_t2i.append(float(t_vals.mean()))
            mean_i2t.append(float(i_vals.mean()))
            std_t2i.append(float(t_vals.std()))
            std_i2t.append(float(i_vals.std()))

        stats["mean_t2i"] = mean_t2i
        stats["mean_i2t"] = mean_i2t
        stats["std_t2i"] = std_t2i
        stats["std_i2t"] = std_i2t

    # 4. 相关系数（mean）
    corr_t2i = _pearson_corr(D_all, mean_t2i_all)
    corr_i2t = _pearson_corr(D_all, mean_i2t_all)
    print(
        f"[corr] Pearson corr(D, mean_t2i) = {corr_t2i:.4f} "
        "(higher D -> weaker t->i ?)"
    )
    print(
        f"[corr] Pearson corr(D, mean_i2t) = {corr_i2t:.4f} "
        "(higher D -> weaker i->t ?)"
    )

    # 5. 绘图：layer-wise mean / std 曲线
    _plot_layer_curves(stats_by_bucket, out_dir)
    _plot_layer_std_curves(stats_by_bucket, out_dir)

    # 6. 绘图：D vs mean-attn / D vs layer-std 散点
    _plot_scatter_D_vs_attn(D_all, mean_i2t_all, mean_t2i_all, bucket_ids, out_dir)
    _plot_scatter_D_vs_std(
        D_all, layer_std_i2t_all, layer_std_t2i_all, bucket_ids, out_dir
    )

    # 7. 绘图：mean & layer-std 直方图
    _plot_histograms(stats_by_bucket, out_dir)

    # 8. 绘图：基于 (D, mean-attn) -> ASR 的热图 / 曲面（i->t & t->i）
    _plot_D_attn_asr_surface(D_all, mean_i2t_all, asr_all, out_dir, direction="i2t")
    _plot_D_attn_asr_surface(D_all, mean_t2i_all, asr_all, out_dir, direction="t2i")


if __name__ == "__main__":
    main()