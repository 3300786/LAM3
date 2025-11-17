import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def load_records(path: Path) -> List[Dict[str, Any]]:
    """
    读取 scores.jsonl，过滤掉缺失模态的样本，并预先计算：
      - phi = Rt_v - Rt_0 - R0_v + R0_0
      - delta_txt = Rt_v - Rt_0
      - ratio_txt = Rt_v / (Rt_0 + eps)
    """
    required_modes = {"txt_img", "txt_only", "img_only", "none"}
    records = []
    n_raw = 0
    n_skipped = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            obj = json.loads(line)

            Rm = obj.get("R", {})
            if not required_modes.issubset(Rm.keys()):
                n_skipped += 1
                continue

            Rt_v = float(Rm["txt_img"])
            Rt_0 = float(Rm["txt_only"])
            R0_v = float(Rm["img_only"])
            R0_0 = float(Rm["none"])

            phi = Rt_v - Rt_0 - R0_v + R0_0
            delta_txt = Rt_v - Rt_0
            eps = 1e-6
            ratio_txt = Rt_v / (Rt_0 + eps)

            records.append(
                {
                    "id": obj.get("id"),
                    "Rt_v": Rt_v,
                    "Rt_0": Rt_0,
                    "R0_v": R0_v,
                    "R0_0": R0_0,
                    "phi": phi,
                    "delta_txt": delta_txt,
                    "ratio_txt": ratio_txt,
                }
            )

    print(
        f"[compute-phi] raw={n_raw}, kept={len(records)}, "
        f"skipped_missing_modes={n_skipped}"
    )
    return records


def sweep_thresholds(
    records: List[Dict[str, Any]],
    thresholds: List[float],
) -> Dict[str, List[float]]:
    """
    对一组 toxicity threshold 进行扫描，计算协同相关曲线：
      - frac_synergy_like(thr):
          P(Rt_v >= thr and Rt_0 < thr)
      - frac_txtimg_only(thr):
          P(Rt_v >= thr and Rt_0 < thr and R0_v < thr)
      - mean_phi_synergy_like(thr):
          E[phi | Rt_v >= thr and Rt_0 < thr and R0_v < thr]
      - frac_boost_at_thr(thr):
          P(Rt_v > Rt_0 and Rt_v >= thr)
    """
    n_total = len(records)

    frac_synergy_like = []
    frac_txtimg_only = []
    mean_phi_synergy_like = []
    frac_boost_at_thr = []

    for thr in thresholds:
        n_synergy_like = 0
        n_txtimg_only = 0
        n_boost_at_thr = 0
        phi_synergy_samples: List[float] = []

        for r in records:
            Rt_v = r["Rt_v"]
            Rt_0 = r["Rt_0"]
            R0_v = r["R0_v"]
            phi = r["phi"]

            txtimg_toxic = Rt_v >= thr
            txtonly_toxic = Rt_0 >= thr
            imgonly_toxic = R0_v >= thr

            if Rt_v > Rt_0 and Rt_v >= thr:
                n_boost_at_thr += 1

            # 协同-like：txt+img toxic, txt-only 不 toxic
            if txtimg_toxic and (not txtonly_toxic):
                n_synergy_like += 1

            # 更严格：只有 txt+img toxic，其它单模态都不过阈值
            if txtimg_toxic and (not txtonly_toxic) and (not imgonly_toxic):
                n_txtimg_only += 1
                phi_synergy_samples.append(phi)

        frac_synergy_like.append(n_synergy_like / n_total if n_total else 0.0)
        frac_txtimg_only.append(n_txtimg_only / n_total if n_total else 0.0)
        mean_phi_synergy_like.append(_mean(phi_synergy_samples))
        frac_boost_at_thr.append(n_boost_at_thr / n_total if n_total else 0.0)

    curves = {
        "thresholds": thresholds,
        "frac_synergy_like": frac_synergy_like,
        "frac_txtimg_only": frac_txtimg_only,
        "mean_phi_synergy_like": mean_phi_synergy_like,
        "frac_boost_at_thr": frac_boost_at_thr,
    }
    return curves


def plot_hist(
    data: List[float],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 40,
    value_range=None,
    density: bool = True,
) -> None:
    if not data:
        print(f"[plot] no data for {title}, skip")
        return

    plt.figure()
    if value_range is not None:
        plt.hist(data, bins=bins, range=value_range, density=density)
    else:
        plt.hist(data, bins=bins, density=density)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density" if density else "count")
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_synergy_vs_threshold(curves: Dict[str, List[float]], out_path: Path) -> None:
    thresholds = curves["thresholds"]
    frac_synergy_like = curves["frac_synergy_like"]
    frac_txtimg_only = curves["frac_txtimg_only"]
    frac_boost_at_thr = curves["frac_boost_at_thr"]

    plt.figure()
    plt.plot(thresholds, frac_synergy_like, label="frac_synergy_like")
    plt.plot(thresholds, frac_txtimg_only, label="frac_txtimg_only")
    plt.plot(thresholds, frac_boost_at_thr, label="frac_boost_at_thr")

    plt.xlabel("tox_threshold (on R in [0,1])")
    plt.ylabel("fraction")
    plt.title("Synergy-related fractions vs tox_threshold")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_phi_synergy_vs_threshold(curves: Dict[str, List[float]], out_path: Path) -> None:
    thresholds = curves["thresholds"]
    mean_phi_synergy_like = curves["mean_phi_synergy_like"]

    plt.figure()
    plt.plot(thresholds, mean_phi_synergy_like, label="mean_phi_synergy_like")

    plt.xlabel("tox_threshold (on R in [0,1])")
    plt.ylabel("mean phi on synergy-like set")
    plt.title("Mean phi on synergy-like set vs tox_threshold")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inp",
        required=True,
        help="input JSONL (per-sample scores, e.g., synergy_jbv28k_mini_scores.jsonl)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="output directory for summary JSON and plots",
    )
    ap.add_argument(
        "--thr_min",
        type=float,
        default=0.0,
        help="min tox_threshold (inclusive), in [0,1]",
    )
    ap.add_argument(
        "--thr_max",
        type=float,
        default=0.5,
        help="max tox_threshold (inclusive), in [0,1]",
    )
    ap.add_argument(
        "--thr_step",
        type=float,
        default=0.01,
        help="step size for tox_threshold sweep",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取样本并计算 phi / delta_txt / ratio_txt
    records = load_records(in_path)
    if not records:
        print("[compute-phi] no valid records, exit")
        return

    phi_list = [r["phi"] for r in records]
    delta_txt_list = [r["delta_txt"] for r in records]
    ratio_txt_list = [r["ratio_txt"] for r in records]

    # 裁剪 ratio，避免极端长尾影响可视化
    clipped_ratio_list = [min(x, 5.0) for x in ratio_txt_list]

    # 全局 boost 率：P(Rt_v > Rt_0)
    n_total = len(records)
    n_boost = sum(1 for r in records if r["Rt_v"] > r["Rt_0"])
    boost_rate = n_boost / n_total if n_total else 0.0

    # 2) 阈值扫描
    thresholds = []
    t = args.thr_min
    while t <= args.thr_max + 1e-8:
        thresholds.append(round(t, 4))
        t += args.thr_step

    curves = sweep_thresholds(records, thresholds)

    # 3) 汇总全局统计 + 曲线数据到 JSON
    summary = {
        "num_samples": n_total,
        "phi_global": {
            "mean": _mean(phi_list),
            "std": _std(phi_list),
            "min": min(phi_list),
            "max": max(phi_list),
        },
        "delta_txt_global": {
            "mean": _mean(delta_txt_list),
            "std": _std(delta_txt_list),
            "min": min(delta_txt_list),
            "max": max(delta_txt_list),
        },
        "ratio_txt_global": {
            "mean": _mean(ratio_txt_list),
            "std": _std(ratio_txt_list),
            "min": min(ratio_txt_list),
            "max": max(ratio_txt_list),
        },
        "boost_rate": boost_rate,
        "threshold_curves": curves,
    }

    summary_path = out_dir / "phi_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[compute-phi] summary written to {summary_path}")

    # 4) 作图

    # 4.1 phi 全局分布
    plot_hist(
        phi_list,
        title="Distribution of phi",
        xlabel="phi",
        out_path=out_dir / "phi_hist.png",
        bins=40,
        density=True,
    )

    # 4.2 delta_txt = Rt_v - Rt_0 分布
    plot_hist(
        delta_txt_list,
        title="Distribution of delta_txt (Rt_v - Rt_0)",
        xlabel="delta_txt",
        out_path=out_dir / "delta_txt_hist.png",
        bins=40,
        density=True,
    )

    # 4.3 ratio_txt = Rt_v / (Rt_0 + eps) 分布（裁剪）
    plot_hist(
        clipped_ratio_list,
        title="Distribution of ratio_txt (clipped to <=5)",
        xlabel="ratio_txt",
        out_path=out_dir / "ratio_txt_hist.png",
        bins=40,
        density=True,
        value_range=(0.0, 5.0),
    )

    # 4.4 协同相关比例 vs 阈值
    plot_synergy_vs_threshold(curves, out_dir / "synergy_vs_threshold.png")

    # 4.5 协同集合上的平均 phi vs 阈值
    plot_phi_synergy_vs_threshold(curves, out_dir / "phi_synergy_vs_threshold.png")

    # 5) 在终端打印几项关键数字，方便快速观察
    print("\n[compute-phi] ==== Global stats ====")
    print(
        f"  phi: mean={summary['phi_global']['mean']:.4f}, "
        f"std={summary['phi_global']['std']:.4f}, "
        f"min={summary['phi_global']['min']:.4f}, "
        f"max={summary['phi_global']['max']:.4f}"
    )
    print(
        f"  delta_txt: mean={summary['delta_txt_global']['mean']:.4f}, "
        f"std={summary['delta_txt_global']['std']:.4f}, "
        f"min={summary['delta_txt_global']['min']:.4f}, "
        f"max={summary['delta_txt_global']['max']:.4f}"
    )
    print(
        f"  ratio_txt: mean={summary['ratio_txt_global']['mean']:.4f}, "
        f"std={summary['ratio_txt_global']['std']:.4f}, "
        f"min={summary['ratio_txt_global']['min']:.4f}, "
        f"max={summary['ratio_txt_global']['max']:.4f}"
    )
    print(f"  boost_rate (Rt_v > Rt_0): {summary['boost_rate']:.4f}")


if __name__ == "__main__":
    main()
