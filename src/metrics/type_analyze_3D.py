#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter
import csv
import math

import matplotlib.pyplot as plt
import numpy as np

# 尝试导入 sklearn（用于 logistic 回归），如果没有则退化为只做条件概率表
try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------
# 基础 I/O
# ---------------------------------------------------------

def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ---------------------------------------------------------
# 将单模型 judge 结果 -> 每个样本的 8-type 标签 + Toxicity
# ---------------------------------------------------------

TYPE_MAPPING = {
    (0, 0, 0): "Type-0",
    (1, 0, 0): "Type-1",
    (0, 1, 0): "Type-2",
    (1, 1, 0): "Type-3",
    (0, 0, 1): "Type-4",
    (1, 0, 1): "Type-5",
    (0, 1, 1): "Type-6",
    (1, 1, 1): "Type-7",
}


def classify_type(yA, yB, yAB):
    """
    yA: txt_only   (None / bool)
    yB: img_only   (None / bool)
    yAB: txt_img   (None / bool)
    """
    yA = 0 if yA is None else int(bool(yA))
    yB = 0 if yB is None else int(bool(yB))
    yAB = 0 if yAB is None else int(bool(yAB))
    triplet = (yA, yB, yAB)
    t = TYPE_MAPPING.get(triplet, None)
    if t is None:
        raise ValueError(f"Unexpected triplet: {triplet}")
    return t, triplet


def build_type_table(records):
    """
    records: list[dict] from a single model's jsonl.

    返回:
      - type_info: dict[id] = {
            "type": str,
            "triplet": (yA, yB, yAB),
            "yA": 0/1,
            "yB": 0/1,
            "yAB": 0/1,
            "toxA": float or None,
            "toxB": float or None,
            "toxAB": float or None,
            "raw_succ_modes": {...},
            "raw_tox_modes": {...},
        }
    """
    modes = ["txt_only", "img_only", "txt_img", "none"]

    bucket_succ = defaultdict(lambda: {m: None for m in modes})
    bucket_tox = defaultdict(lambda: {m: None for m in modes})

    for row in records:
        sid = str(row["id"])
        mode = row["mode"]  # txt_only / img_only / txt_img / none
        succ = row["qwen_judge"]["is_attack_success"]
        tox = row["qwen_judge"]["toxicity"]  # 0~100 float
        bucket_succ[sid][mode] = succ
        bucket_tox[sid][mode] = tox

    type_info = {}
    for sid in bucket_succ.keys():
        succ_modes = bucket_succ[sid]
        tox_modes = bucket_tox[sid]

        yA = succ_modes["txt_only"]
        yB = succ_modes["img_only"]
        yAB = succ_modes["txt_img"]

        t, triplet = classify_type(yA, yB, yAB)

        type_info[sid] = {
            "type": t,
            "triplet": triplet,
            "yA": triplet[0],
            "yB": triplet[1],
            "yAB": triplet[2],
            "toxA": tox_modes["txt_only"],
            "toxB": tox_modes["img_only"],
            "toxAB": tox_modes["txt_img"],
            "raw_succ_modes": succ_modes,
            "raw_tox_modes": tox_modes,
        }

    return type_info


# ---------------------------------------------------------
# 统计 & 可视化：单模型
# ---------------------------------------------------------

ALL_TYPES = [f"Type-{i}" for i in range(8)]


def summarize_single_model(name, type_info):
    """
    打印单模型的 Type 分布 + Toxicity 基本统计。
    """
    counter = Counter(v["type"] for v in type_info.values())
    total = sum(counter.values())
    print(f"\n=== Model: {name} ===")
    print(f"Total samples: {total}")
    for t in ALL_TYPES:
        c = counter.get(t, 0)
        print(f"{t}: {c} ({c/total:.3f})")

    # 折叠到 coarse group（与之前一致）
    abs_success = counter["Type-4"]
    maintain = counter["Type-5"] + counter["Type-6"]
    abs_fail = counter["Type-1"] + counter["Type-2"] + counter["Type-3"]
    unk_minus = counter["Type-0"]
    unk_plus = counter["Type-7"]
    print(f"Absolutely Success: {abs_success}")
    print(f"Maintain: {maintain}")
    print(f"Absolutely Fail: {abs_fail}")
    print(f"Unknown- (all fail): {unk_minus}")
    print(f"Unknown+ (all success): {unk_plus}")

    # ---- Toxicity 统计：按模式 txt_only / img_only / txt_img ----
    tox_A = []
    tox_B = []
    tox_AB = []
    for v in type_info.values():
        if v["toxA"] is not None:
            tox_A.append(v["toxA"])
        if v["toxB"] is not None:
            tox_B.append(v["toxB"])
        if v["toxAB"] is not None:
            tox_AB.append(v["toxAB"])

    def _summ_tox(xs, label):
        if not xs:
            print(f"{label}: no toxicity data.")
            return
        arr = np.asarray(xs, dtype=float)
        print(f"{label}: mean={arr.mean():.2f}, std={arr.std(ddof=0):.2f}, "
              f"min={arr.min():.1f}, max={arr.max():.1f}, n={len(arr)}")

    print(f"\n[Toxicity stats for {name}] (unconditional, all samples)")
    _summ_tox(tox_A, "  txt_only (A)")
    _summ_tox(tox_B, "  img_only (B)")
    _summ_tox(tox_AB, "  txt_img  (AB)")


def plot_single_model(name, type_info, outdir: Path):
    counter = Counter(v["type"] for v in type_info.values())
    counts = [counter.get(t, 0) for t in ALL_TYPES]

    x = np.arange(len(ALL_TYPES))
    plt.figure()
    plt.bar(x, counts)
    plt.xticks(x, ALL_TYPES, rotation=45)
    plt.ylabel("Count")
    plt.title(f"Type distribution - {name}")
    plt.tight_layout()
    out_path = outdir / f"{name}_type_distribution.png"
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------
# 统计 & 可视化：跨模型一致性 + 距离
# ---------------------------------------------------------

def build_pairwise_matrix(type_info_A, type_info_B):
    """
    type_info_A/B: dict[id] -> {..., "type": "Type-k", ...}
    返回 8x8 矩阵, mat[i,j] = count(Type-i in A, Type-j in B)
    只统计两个模型都包含的 id。
    """
    common_ids = sorted(set(type_info_A.keys()) & set(type_info_B.keys()))
    mat = np.zeros((8, 8), dtype=int)
    for sid in common_ids:
        tA = type_info_A[sid]["type"]
        tB = type_info_B[sid]["type"]
        i = int(tA.split("-")[1])
        j = int(tB.split("-")[1])
        mat[i, j] += 1
    return mat, common_ids


def hamming_distance(tri_a, tri_b):
    """
    tri_a, tri_b: (yA, yB, yAB) in {0,1}^3
    返回 0~3 的汉明距离
    """
    return sum(1 for x, y in zip(tri_a, tri_b) if x != y)


def summarize_pairwise(nameA, type_info_A, nameB, type_info_B):
    mat, common_ids = build_pairwise_matrix(type_info_A, type_info_B)
    n = len(common_ids)
    print(f"\n=== Pairwise: {nameA} vs {nameB} ===")
    print(f"Common sample count: {n}")

    # Overall exact-type agreement
    agree = sum(1 for sid in common_ids if type_info_A[sid]["type"] == type_info_B[sid]["type"])
    print(f"Exact Type agreement: {agree} ({agree/n:.3f})")

    # 排除 Type-0 再算一次
    non0_ids = [sid for sid in common_ids
                if type_info_A[sid]["type"] != "Type-0"
                or type_info_B[sid]["type"] != "Type-0"]
    if non0_ids:
        agree_non0 = sum(1 for sid in non0_ids
                         if type_info_A[sid]["type"] == type_info_B[sid]["type"])
        print(f"Agreement (excluding all-Type-0): {agree_non0}/{len(non0_ids)} "
              f"({agree_non0/len(non0_ids):.3f})")

    print("Type×Type contingency (rows: A, cols: B):")
    for i in range(8):
        row_counts = " ".join(f"{mat[i, j]:3d}" for j in range(8))
        print(f"Type-{i}: {row_counts}")


def compute_distance_stats(nameA, type_info_A, nameB, type_info_B):
    """
    计算两个模型之间的汉明距离分布:
      d in {0,1,2,3}
    以及平均距离。
    """
    common_ids = sorted(set(type_info_A.keys()) & set(type_info_B.keys()))
    dist_counter = Counter()
    d_values = []

    for sid in common_ids:
        tri_a = type_info_A[sid]["triplet"]
        tri_b = type_info_B[sid]["triplet"]
        d = hamming_distance(tri_a, tri_b)
        dist_counter[d] += 1
        d_values.append(d)

    n = len(common_ids)
    print(f"\n--- Hamming distance stats: {nameA} vs {nameB} ---")
    for d in range(4):
        c = dist_counter.get(d, 0)
        print(f"d={d}: {c} ({c / n:.3f})")
    avg_d = sum(d_values) / n if n > 0 else 0.0
    print(f"Average distance: {avg_d:.3f}")

    return dist_counter, d_values


def plot_pairwise_heatmap(nameA, type_info_A, nameB, type_info_B, outdir: Path,
                          normalize="row"):
    """
    normalize:
      - None: raw counts
      - "row": 每行归一化（条件在 A 的 Type）
      - "col": 每列归一化
    """
    mat, _ = build_pairwise_matrix(type_info_A, type_info_B)
    mat = mat.astype(float)

    if normalize == "row":
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mat_norm = mat / row_sums
        suffix = "_rownorm"
    elif normalize == "col":
        col_sums = mat.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        mat_norm = mat / col_sums
        suffix = "_colnorm"
    else:
        mat_norm = mat
        suffix = ""

    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat_norm, origin="lower", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(8), ALL_TYPES, rotation=45)
    plt.yticks(np.arange(8), ALL_TYPES)
    plt.xlabel(nameB)
    plt.ylabel(nameA)
    plt.title(f"Type×Type ({nameA} vs {nameB}){suffix}")
    plt.tight_layout()
    out_path = outdir / f"{nameA}_vs_{nameB}_heatmap{suffix}.png"
    plt.savefig(out_path)
    plt.close()


def plot_distance_hist(nameA, nameB, dist_counter, outdir: Path):
    """
    画两模型间汉明距离分布的柱状图。
    """
    counts = [dist_counter.get(d, 0) for d in range(4)]
    x = np.arange(4)
    plt.figure()
    plt.bar(x, counts)
    plt.xticks(x, [f"d={d}" for d in range(4)])
    plt.ylabel("Count")
    plt.title(f"Hamming distance distribution: {nameA} vs {nameB}")
    plt.tight_layout()
    out_path = outdir / f"{nameA}_vs_{nameB}_distance_hist.png"
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------
# CSV 导出
# ---------------------------------------------------------

def export_csv_per_sample_wide(
    info_llava, info_idefics, info_llama, outdir: Path
):
    """
    宽表：每行一个 sample id，列为各模型的 type/triplet。
    """
    all_ids = sorted(
        set(info_llava.keys()) | set(info_idefics.keys()) | set(info_llama.keys())
    )
    out_path = outdir / "types_per_sample_wide.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "type_llava", "triplet_llava",
            "type_idefics", "triplet_idefics",
            "type_llama", "triplet_llama",
        ])
        for sid in all_ids:
            row_llava = info_llava.get(sid)
            row_idefics = info_idefics.get(sid)
            row_llama = info_llama.get(sid)

            def safe_type_triplet(row):
                if row is None:
                    return "", ""
                return row["type"], str(row["triplet"])

            t_l, tri_l = safe_type_triplet(row_llava)
            t_i, tri_i = safe_type_triplet(row_idefics)
            t_m, tri_m = safe_type_triplet(row_llama)

            writer.writerow([sid, t_l, tri_l, t_i, tri_i, t_m, tri_m])

    print(f"[CSV] Per-sample wide view saved to: {out_path}")


def export_csv_per_sample_long(
    info_llava, info_idefics, info_llama, outdir: Path
):
    """
    长表：每行一个 (sample, model)，列为 id, model, type, yA, yB, yAB, toxA, toxB, toxAB。
    """
    out_path = outdir / "types_per_sample_long.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "model", "type", "yA", "yB", "yAB", "toxA", "toxB", "toxAB"])

        def dump(model_name, info_dict):
            for sid, v in info_dict.items():
                writer.writerow([
                    sid,
                    model_name,
                    v["type"],
                    v["yA"],
                    v["yB"],
                    v["yAB"],
                    v["toxA"],
                    v["toxB"],
                    v["toxAB"],
                ])

        dump("llava15_7b", info_llava)
        dump("idefics2_8b", info_idefics)
        dump("llama3.2_11b", info_llama)

    print(f"[CSV] Per-(sample, model) long view saved to: {out_path}")


def export_pairwise_distance_csv(
    nameA, infoA, nameB, infoB, outdir: Path
):
    """
    为每对模型输出一个 CSV:
      id, model_a, type_a, triplet_a, model_b, type_b, triplet_b, distance
    """
    common_ids = sorted(set(infoA.keys()) & set(infoB.keys()))
    out_path = outdir / f"{nameA}_vs_{nameB}_distance_per_sample.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "model_a", "type_a", "triplet_a",
            "model_b", "type_b", "triplet_b",
            "distance",
        ])
        for sid in common_ids:
            va = infoA[sid]
            vb = infoB[sid]
            d = hamming_distance(va["triplet"], vb["triplet"])
            writer.writerow([
                sid,
                nameA, va["type"], str(va["triplet"]),
                nameB, vb["type"], str(vb["triplet"]),
                d,
            ])
    print(f"[CSV] Pairwise distance per-sample saved to: {out_path}")


# ---------------------------------------------------------
# 单模态一致性分析（仍然基于 is_attack_success）
# ---------------------------------------------------------

def phi_coefficient(x, y):
    n11 = sum((x[i] == 1 and y[i] == 1) for i in range(len(x)))
    n00 = sum((x[i] == 0 and y[i] == 0) for i in range(len(x)))
    n10 = sum((x[i] == 1 and y[i] == 0) for i in range(len(x)))
    n01 = sum((x[i] == 0 and y[i] == 1) for i in range(len(x)))

    num = (n11 * n00 - n10 * n01)
    den = math.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
    return num / den if den > 0 else 0.0


def analyze_single_modality_similarity(nameA, infoA, nameB, infoB):
    """
    比较两个模型在 A(txt_only), B(img_only) 上的一致性。
    输出：
      - unconditional agreement
      - conditional agreement (|B=0 或 |A=0)
      - positive/negative consistency
      - phi correlation
    """
    common_ids = sorted(set(infoA.keys()) & set(infoB.keys()))

    # Extract A and B bits
    A_A = [infoA[sid]["yA"] for sid in common_ids]
    A_B = [infoB[sid]["yA"] for sid in common_ids]
    B_A = [infoA[sid]["yB"] for sid in common_ids]
    B_B = [infoB[sid]["yB"] for sid in common_ids]

    def agreement(v1, v2):
        return sum(x == y for x, y in zip(v1, v2)) / len(v1)

    # ----- A 模态 -----
    agreeA = agreement(A_A, A_B)

    # conditional: B=0
    cond_ids = [i for i, sid in enumerate(common_ids)
                if infoA[sid]["yB"] == 0 and infoB[sid]["yB"] == 0]

    if cond_ids:
        agreeA_cond = sum(A_A[i] == A_B[i] for i in cond_ids) / len(cond_ids)
    else:
        agreeA_cond = None

    # positive / negative agreement
    posA = sum((A_A[i] == 1) and (A_B[i] == 1) for i in range(len(common_ids)))
    negA = sum((A_A[i] == 0) and (A_B[i] == 0) for i in range(len(common_ids)))

    phiA = phi_coefficient(A_A, A_B)

    print(f"\n=== Single-modality similarity: {nameA} vs {nameB} ===")

    print(f"[A modality] unconditional agreement: {agreeA:.3f}")
    if agreeA_cond is not None:
        print(f"[A modality] conditional agreement (B=0): {agreeA_cond:.3f}")
    else:
        print(f"[A modality] conditional agreement (B=0): N/A")

    print(f"[A modality] both positive: {posA}")
    print(f"[A modality] both negative: {negA}")
    print(f"[A modality] φ coefficient: {phiA:.3f}")

    # ----- B 模态 (mirror) -----
    agreeB = agreement(B_A, B_B)

    cond_ids = [i for i, sid in enumerate(common_ids)
                if infoA[sid]["yA"] == 0 and infoB[sid]["yA"] == 0]
    if cond_ids:
        agreeB_cond = sum(B_A[i] == B_B[i] for i in cond_ids) / len(cond_ids)
    else:
        agreeB_cond = None

    posB = sum((B_A[i] == 1) and (B_B[i] == 1) for i in range(len(common_ids)))
    negB = sum((B_A[i] == 0) and (B_B[i] == 0) for i in range(len(common_ids)))
    phiB = phi_coefficient(B_A, B_B)

    print(f"\n[B modality] unconditional agreement: {agreeB:.3f}")
    if agreeB_cond is not None:
        print(f"[B modality] conditional agreement (A=0): {agreeB_cond:.3f}")
    else:
        print(f"[B modality] conditional agreement (A=0): N/A")

    print(f"[B modality] both positive: {posB}")
    print(f"[B modality] both negative: {negB}")
    print(f"[B modality] φ coefficient: {phiB:.3f}")

    print("----------------------------------------------------")


# ---------------------------------------------------------
# d=1 翻转分析
# ---------------------------------------------------------

def analyze_d1_bit_flips(nameA, infoA, nameB, infoB, outdir: Path):
    """
    对于所有 d=1 的样本，统计是 A / B / AB 哪一位在两个模型间发生翻转。
    并导出一个 CSV 方便人工检查。
    """
    common_ids = sorted(set(infoA.keys()) & set(infoB.keys()))
    cnt_A = 0
    cnt_B = 0
    cnt_AB = 0
    total_d1 = 0

    out_path = outdir / f"{nameA}_vs_{nameB}_d1_bit_flips.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "model_a", "triplet_a",
            "model_b", "triplet_b",
            "flip_A", "flip_B", "flip_AB",
        ])

        for sid in common_ids:
            tri_a = infoA[sid]["triplet"]  # (yA, yB, yAB)
            tri_b = infoB[sid]["triplet"]
            d = hamming_distance(tri_a, tri_b)
            if d != 1:
                continue
            total_d1 += 1
            flip_A = int(tri_a[0] != tri_b[0])
            flip_B = int(tri_a[1] != tri_b[1])
            flip_AB = int(tri_a[2] != tri_b[2])

            cnt_A += flip_A
            cnt_B += flip_B
            cnt_AB += flip_AB

            writer.writerow([
                sid,
                nameA, str(tri_a),
                nameB, str(tri_b),
                flip_A, flip_B, flip_AB,
            ])

    print(f"\n=== d=1 bit-flip analysis: {nameA} vs {nameB} ===")
    print(f"Total d=1 samples: {total_d1}")
    if total_d1 > 0:
        print(f"A-bit flips   : {cnt_A} ({cnt_A/total_d1:.3f})")
        print(f"B-bit flips   : {cnt_B} ({cnt_B/total_d1:.3f})")
        print(f"AB-bit flips  : {cnt_AB} ({cnt_AB/total_d1:.3f})")
    else:
        print("No d=1 samples found.")
    print(f"[CSV] d=1 bit-flip detail saved to: {out_path}")


# ---------------------------------------------------------
# AB = f(A,B) 逻辑结构分析（ASR 维度）
# ---------------------------------------------------------

def analyze_logistic_ab(model_name: str, info: dict, outdir: Path):
    """
    对单个模型做：
      1) 经验条件概率表：P(AB=1 | A,B), 对于(A,B)∈{0,1}×{0,1}
      2) 若 sklearn 可用，做 logistic regression:
           AB = sigma(w0 + w1*A + w2*B + w3*A*B)
    并输出到终端和 CSV。
    """
    # 准备数据
    ids = sorted(info.keys())
    X_list = []
    y_list = []
    for sid in ids:
        v = info[sid]
        A = v["yA"]
        B = v["yB"]
        AB = v["yAB"]
        X_list.append([A, B, A * B])
        y_list.append(AB)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    # 1) 条件概率表
    cond_counts = np.zeros((2, 2), dtype=int)  # count(A,B)
    cond_pos = np.zeros((2, 2), dtype=int)     # count(AB=1 | A,B)

    for A, B, AB in zip(X[:, 0], X[:, 1], y):
        a = int(A)
        b = int(B)
        cond_counts[a, b] += 1
        if AB == 1:
            cond_pos[a, b] += 1

    print(f"\n=== AB conditional structure for model: {model_name} ===")
    print("P(AB=1 | A,B) and counts:")
    for a in [0, 1]:
        for b in [0, 1]:
            c = cond_counts[a, b]
            p = cond_pos[a, b] / c if c > 0 else 0.0
            print(f"A={a}, B={b}: P={p:.3f}, count={c}")

    # 输出 CSV
    out_csv = outdir / f"{model_name}_AB_conditional_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["A", "B", "count", "AB_pos", "P(AB=1|A,B)"])
        for a in [0, 1]:
            for b in [0, 1]:
                c = cond_counts[a, b]
                pos = cond_pos[a, b]
                p = pos / c if c > 0 else 0.0
                writer.writerow([a, b, c, pos, p])
    print(f"[CSV] AB conditional table saved to: {out_csv}")

    # 2) logistic 回归（若 sklearn 可用）
    if HAS_SKLEARN:
        clf = LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000
        )
        clf.fit(X, y)
        w0 = clf.intercept_[0]
        w1, w2, w3 = clf.coef_[0]
        print("\n[Logistic Regression] AB ~ A + B + A*B")
        print(f"logit(AB=1) = {w0:.3f} + {w1:.3f}*A + {w2:.3f}*B + {w3:.3f}*(A*B)")
        # 保存到文件
        coef_csv = outdir / f"{model_name}_AB_logistic_coef.csv"
        with coef_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["intercept", "w_A", "w_B", "w_AxB"])
            writer.writerow([w0, w1, w2, w3])
        print(f"[CSV] Logistic regression coefficients saved to: {coef_csv}")
    else:
        print("\n[Logistic Regression] sklearn not available, skipped.")


# ---------------------------------------------------------
# 新增：AB = f(A,B) 在 Toxicity 维度上的结构
# ---------------------------------------------------------

def analyze_toxicity_ab(model_name: str, info: dict, outdir: Path):
    """
    对单个模型，从 Toxicity 角度分析协同结构：
      - 按 (A,B) 四象限统计 mean_toxA, mean_toxB, mean_toxAB（不区分触发）
      - 额外统计全局的：
          mean toxA | A=1
          mean toxB | B=1
          mean toxAB | AB=1
      - 输出表格 + CSV: *_AB_toxicity_conditional_table.csv
    """
    ids = sorted(info.keys())

    # 结构：dict[(A,B)] -> {"cnt":..., "toxA":[], "toxB":[], "toxAB":[]}
    stats = {
        (0, 0): {"cnt": 0, "toxA": [], "toxB": [], "toxAB": []},
        (0, 1): {"cnt": 0, "toxA": [], "toxB": [], "toxAB": []},
        (1, 0): {"cnt": 0, "toxA": [], "toxB": [], "toxAB": []},
        (1, 1): {"cnt": 0, "toxA": [], "toxB": [], "toxAB": []},
    }

    # 全局：ASR=1 下的 tox
    succ_toxA = []
    succ_toxB = []
    succ_toxAB = []

    for sid in ids:
        v = info[sid]
        A = int(v["yA"])
        B = int(v["yB"])
        key = (A, B)
        d = stats[key]
        d["cnt"] += 1
        if v["toxA"] is not None:
            d["toxA"].append(v["toxA"])
        if v["toxB"] is not None:
            d["toxB"].append(v["toxB"])
        if v["toxAB"] is not None:
            d["toxAB"].append(v["toxAB"])

        # 全局“成功条件下”的 Toxicity 收集
        if A == 1 and v["toxA"] is not None:
            succ_toxA.append(v["toxA"])
        if B == 1 and v["toxB"] is not None:
            succ_toxB.append(v["toxB"])
        if v["yAB"] == 1 and v["toxAB"] is not None:
            succ_toxAB.append(v["toxAB"])

    print(f"\n=== AB toxicity structure for model: {model_name} ===")
    print("Mean Toxicity (txt_only, img_only, txt_img) conditioned on (A,B):")
    for A in [0, 1]:
        for B in [0, 1]:
            d = stats[(A, B)]
            cnt = d["cnt"]

            def _mean_or_nan(xs):
                if not xs:
                    return float("nan")
                return float(np.mean(np.asarray(xs, dtype=float)))

            mA = _mean_or_nan(d["toxA"])
            mB = _mean_or_nan(d["toxB"])
            mAB = _mean_or_nan(d["toxAB"])
            print(
                f"A={A}, B={B}: count={cnt}, "
                f"mean_toxA={mA:.2f}, mean_toxB={mB:.2f}, mean_toxAB={mAB:.2f}"
            )

    # 全局：ASR=1 条件下的 Toxicity
    def _summ_success(xs, label):
        if not xs:
            print(f"{label}: no successful samples.")
            return
        arr = np.asarray(xs, dtype=float)
        print(f"{label}: mean={arr.mean():.2f}, std={arr.std(ddof=0):.2f}, "
              f"min={arr.min():.1f}, max={arr.max():.1f}, n={len(arr)}")

    print("\n[Success-conditioned Toxicity (only ASR=1 samples)]")
    _summ_success(succ_toxA, "  txt_only (tox | A=1)")
    _summ_success(succ_toxB, "  img_only (tox | B=1)")
    _summ_success(succ_toxAB, "  txt_img  (tox | AB=1)")

    # 输出 CSV
    out_csv = outdir / f"{model_name}_AB_toxicity_conditional_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "A", "B", "count",
            "mean_toxA_all", "mean_toxB_all", "mean_toxAB_all",
            "n_toxA_all", "n_toxB_all", "n_toxAB_all",
        ])
        for A in [0, 1]:
            for B in [0, 1]:
                d = stats[(A, B)]
                cnt = d["cnt"]

                def _safe_mean(xs):
                    if not xs:
                        return ""
                    return float(np.mean(np.asarray(xs, dtype=float)))

                mA = _safe_mean(d["toxA"])
                mB = _safe_mean(d["toxB"])
                mAB = _safe_mean(d["toxAB"])
                writer.writerow([
                    A, B, cnt,
                    mA, mB, mAB,
                    len(d["toxA"]), len(d["toxB"]), len(d["toxAB"]),
                ])

        # 额外在 CSV 尾部附上全局 ASR=1 下的统计汇总
        def _safe_global(xs):
            if not xs:
                return "", "", ""
            arr = np.asarray(xs, dtype=float)
            return float(arr.mean()), float(arr.std(ddof=0)), len(arr)

        gA_mean, gA_std, gA_n = _safe_global(succ_toxA)
        gB_mean, gB_std, gB_n = _safe_global(succ_toxB)
        gAB_mean, gAB_std, gAB_n = _safe_global(succ_toxAB)

        writer.writerow([])
        writer.writerow(["global_success_cond_A", "", "",
                         gA_mean, "", "", gA_n, "", ""])
        writer.writerow(["global_success_cond_B", "", "",
                         "", gB_mean, "", "", gB_n, ""])
        writer.writerow(["global_success_cond_AB", "", "",
                         "", "", gAB_mean, "", "", gAB_n])

    print(f"[CSV] AB toxicity conditional table saved to: {out_csv}")


# ---------------------------------------------------------
# 主函数
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava", required=True, help="jsonl for llava15_7b judge results")
    parser.add_argument("--idefics", required=True, help="jsonl for idefics2_8b judge results")
    parser.add_argument("--llama", required=True, help="jsonl for llama3.2_11b judge results")
    parser.add_argument("--outdir", required=True, help="output directory for figures and stats")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 逐模型读取 + 8-type 编码 + Toxicity
    rec_llava = load_jsonl(Path(args.llava))
    rec_idefics = load_jsonl(Path(args.idefics))
    rec_llama = load_jsonl(Path(args.llama))

    info_llava = build_type_table(rec_llava)
    info_idefics = build_type_table(rec_idefics)
    info_llama = build_type_table(rec_llama)

    # 2) 单模型统计 + 柱状图
    summarize_single_model("llava15_7b", info_llava)
    summarize_single_model("idefics2_8b", info_idefics)
    summarize_single_model("llama3.2_11b", info_llama)

    plot_single_model("llava15_7b", info_llava, outdir)
    plot_single_model("idefics2_8b", info_idefics, outdir)
    plot_single_model("llama3.2_11b", info_llama, outdir)

    # 3) 跨模型 pairwise 一致性 + 8×8 热力图 + 距离分布 + d=1 bit flip
    pairs = [
        ("llava15_7b", info_llava, "idefics2_8b", info_idefics),
        ("llava15_7b", info_llava, "llama3.2_11b", info_llama),
        ("idefics2_8b", info_idefics, "llama3.2_11b", info_llama),
    ]

    for (na, ia, nb, ib) in pairs:
        summarize_pairwise(na, ia, nb, ib)
        dist_counter, _ = compute_distance_stats(na, ia, nb, ib)
        plot_pairwise_heatmap(na, ia, nb, ib, outdir, normalize="row")
        plot_pairwise_heatmap(na, ia, nb, ib, outdir, normalize=None)
        plot_distance_hist(na, nb, dist_counter, outdir)
        export_pairwise_distance_csv(na, ia, nb, ib, outdir)
        analyze_d1_bit_flips(na, ia, nb, ib, outdir)

    # 4) 单模态一致性分析（ASR）
    for (na, ia, nb, ib) in pairs:
        analyze_single_modality_similarity(na, ia, nb, ib)

    # 5) 导出 CSV，便于进一步考察
    export_csv_per_sample_wide(info_llava, info_idefics, info_llama, outdir)
    export_csv_per_sample_long(info_llava, info_idefics, info_llama, outdir)

    # 6) 每个模型的 AB 逻辑结构分析（ASR）
    analyze_logistic_ab("llava15_7b", info_llava, outdir)
    analyze_logistic_ab("idefics2_8b", info_idefics, outdir)
    analyze_logistic_ab("llama3.2_11b", info_llama, outdir)

    # 7) 每个模型的 AB Toxicity 结构分析（包括 ASR=1 条件下的 Toxicity）
    analyze_toxicity_ab("llava15_7b", info_llava, outdir)
    analyze_toxicity_ab("idefics2_8b", info_idefics, outdir)
    analyze_toxicity_ab("llama3.2_11b", info_llama, outdir)


if __name__ == "__main__":
    main()
