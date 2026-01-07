import json
import os
import argparse
import pandas as pd
from collections import defaultdict, Counter
import numpy as np


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def analyze_statistics(input_file, output_dir):
    print(f"[Stats] Loading data from {input_file}...")
    records = load_data(input_file)

    # 初始化统计容器
    # stats[group] = { ... }
    group_stats = defaultdict(lambda: {
        "total": 0,
        "jailbreak_count": 0,
        "axis_I": Counter(),
        "axis_II": Counter(),
        "axis_III": Counter(),
        "behaviors": Counter()
    })

    # 遍历数据
    for rec in records:
        group = rec.get("group", "unknown")
        # 兼容 judge_result 在顶层或嵌套的情况
        judge = rec.get("judge_result", {})
        if not judge:
            # 尝试从顶层读取 (兼容旧格式)
            if "axis_I" in rec:
                judge = rec
            else:
                continue

        # 提取关键指标
        is_jb = judge.get("is_jailbreak_success", False)
        axis_i = judge.get("axis_I", "N/A")
        axis_a = judge.get("axis_II", "N/A")
        axis_h = judge.get("axis_III", "N/A")
        behavior = judge.get("behavior_type", "UNKNOWN")

        # 更新统计
        s = group_stats[group]
        s["total"] += 1
        if is_jb:
            s["jailbreak_count"] += 1

        s["axis_I"][axis_i] += 1
        s["axis_II"][axis_a] += 1
        s["axis_III"][axis_h] += 1
        s["behaviors"][behavior] += 1

    # ==========================================
    # 1. 生成 ASR 概览表 (Attack Success Rate)
    # ==========================================
    asr_data = []
    for group in sorted(group_stats.keys()):
        s = group_stats[group]
        total = s["total"]
        if total == 0: continue

        asr = (s["jailbreak_count"] / total) * 100
        asr_data.append({
            "Group": group,
            "Total": total,
            "Jailbroken": s["jailbreak_count"],
            "ASR (%)": f"{asr:.2f}%"
        })

    df_asr = pd.DataFrame(asr_data)

    print("\n" + "=" * 50)
    print(" 📊 ASR SUMMARY (Attack Success Rate)")
    print("=" * 50)
    if not df_asr.empty:
        print(df_asr.to_markdown(index=False))
    else:
        print("No data found.")

    # ==========================================
    # 2. 生成三轴分布表 (3-Axis Distribution)
    # ==========================================
    # 我们将计算每个轴各等级的百分比
    axis_data = []

    # 定义轴的顺序
    i_keys = ["I0", "I1", "I2", "I3"]
    a_keys = ["A0", "A1", "A2", "A3", "A4"]
    h_keys = ["H0", "H1", "H2", "H3"]

    for group in sorted(group_stats.keys()):
        s = group_stats[group]
        total = s["total"]
        if total == 0: continue

        row = {"Group": group}

        # Helper to calc percent
        def calc_pct(counter, keys):
            for k in keys:
                cnt = counter.get(k, 0)
                row[k] = f"{(cnt / total) * 100:.1f}%"

        calc_pct(s["axis_I"], i_keys)
        calc_pct(s["axis_II"], a_keys)
        calc_pct(s["axis_III"], h_keys)

        axis_data.append(row)

    df_axis = pd.DataFrame(axis_data)
    # 重新排列列顺序
    cols = ["Group"] + i_keys + a_keys + h_keys
    # 确保列存在 (防止某些 key 数据中完全没出现)
    existing_cols = [c for c in cols if c in df_axis.columns]
    df_axis = df_axis[existing_cols]

    print("\n" + "=" * 80)
    print(" 📐 3-AXIS DISTRIBUTION (Severity / Alignment / Harmfulness)")
    print("=" * 80)
    if not df_axis.empty:
        print(df_axis.to_markdown(index=False))

    # ==========================================
    # 3. 生成行为细分表 (Behavior Breakdown)
    # ==========================================
    beh_data = []
    # 预定义行为顺序
    beh_keys = [
        "R0_REFUSAL", "R1_SAFE_REDIRECT", "M0_MISINTERPRETATION",
        "S0_BENIGN_COMPLIANCE", "C0_UNSAFE_DEPICTION",
        "J2_THEORETICAL_COMPLIANCE", "J1_PARTIAL_COMPLIANCE", "J0_ACTIONABLE_COMPLIANCE"
    ]

    for group in sorted(group_stats.keys()):
        s = group_stats[group]
        total = s["total"]
        if total == 0: continue

        row = {"Group": group}
        for k in beh_keys:
            cnt = s["behaviors"].get(k, 0)
            # row[k] = cnt  # 这里用绝对数量可能更直观，或者用百分比
            row[k] = f"{(cnt/total)*100:.1f}%"

        beh_data.append(row)

    df_beh = pd.DataFrame(beh_data)

    print("\n" + "=" * 80)
    print(" 🎭 BEHAVIOR BREAKDOWN (Raw Counts)")
    print("=" * 80)
    if not df_beh.empty:
        print(df_beh.to_markdown(index=False))

    # ==========================================
    # 4. 保存文件
    # ==========================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if not df_asr.empty: df_asr.to_csv(os.path.join(output_dir, "stats_asr.csv"), index=False)
        if not df_axis.empty: df_axis.to_csv(os.path.join(output_dir, "stats_3axis.csv"), index=False)
        if not df_beh.empty: df_beh.to_csv(os.path.join(output_dir, "stats_behaviors.csv"), index=False)
        print(f"\n[Stats] CSV reports saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to final_judged.jsonl")
    parser.add_argument("--out_dir", type=str, default="Refusal/outputs/stats")
    args = parser.parse_args()

    analyze_statistics(args.input, args.out_dir)