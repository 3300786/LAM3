# src/utils/check_synergy_qwen_completeness.py

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Tuple

# FILE = Path("outputs/metrics/synergy_jbv28k_synergy_qwen_judge.jsonl")
FILE = Path("outputs/logs/synergy_jbv28k_raw_rebuild.jsonl")
TOTAL_ID = 28000
VALID_MODES = {"txt_img", "txt_only", "img_only", "none"}


def normalize_id(raw: Any):
    """把 id 规范化为 int；如果无法转换，返回 None。"""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            return int(s)
    return None


def main():
    print(f"[load] reading: {FILE}")
    if not FILE.exists():
        raise FileNotFoundError(f"{FILE} does not exist")

    records = []
    with FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                print("[warn] JSON decode failed:", e)

    print(f"[load] total loaded: {len(records)}")

    # 统计
    id_stats = defaultdict(int)                # 每个 id 出现次数（所有 mode 合并）
    pair_stats = defaultdict(int)              # 每个 (id, mode) 组合出现次数
    mode_stats = defaultdict(int)              # 有效 mode 计数

    invalid_ids = []                           # 记录真正非法 id
    invalid_modes = []                         # 非法 mode 记录 (raw_id, mode)

    for rec in records:
        raw_id = rec.get("id")
        mode = rec.get("mode")

        nid = normalize_id(raw_id)
        if nid is None or nid < 0 or nid >= TOTAL_ID:
            invalid_ids.append(raw_id)
            continue

        if mode not in VALID_MODES:
            invalid_modes.append((raw_id, mode))
            continue

        id_stats[nid] += 1
        pair_stats[(nid, mode)] += 1
        mode_stats[mode] += 1

    # ---- 按 id 维度的覆盖情况 ----
    all_ids = set(range(TOTAL_ID))
    present_ids = set(id_stats.keys())
    missing_ids = sorted(all_ids - present_ids)
    extra_ids = sorted(present_ids - all_ids)   # 理论上应为空
    dup_ids = [i for i, c in id_stats.items() if c > 4]  # 出现次数 >4（超过四种模式的理论上限）

    # ---- 按 (id, mode) 组合的覆盖情况 ----
    expected_pairs = TOTAL_ID * len(VALID_MODES)
    present_pairs = set(pair_stats.keys())  # 已出现的 (id, mode)
    all_pairs = {(i, m) for i in range(TOTAL_ID) for m in VALID_MODES}

    missing_pairs = sorted(all_pairs - present_pairs)
    dup_pairs = [(p, c) for p, c in pair_stats.items() if c > 1]

    # ---- 输出报告 ----
    print("\n============== completeness report ==============")

    print(f"[ids] expected id range: 0 ~ {TOTAL_ID-1}")
    print(f"[ids] ids with any valid record: {len(present_ids)}")
    print(f"[ids] missing ids (no valid record at all): {len(missing_ids)}")
    if missing_ids:
        print("  missing ids (first 20):", missing_ids[:20], "...")

    print(f"[ids] extra ids (out of expected range but passed normalize): {len(extra_ids)}")
    if extra_ids:
        print("  extra ids (first 20):", extra_ids[:20], "...")

    print(f"[ids] ids with >4 valid records (possible heavy duplication across modes): {len(dup_ids)}")
    if dup_ids:
        print("  example dup ids (first 20):", dup_ids[:20], "...")

    print(f"[ids] truly invalid ids (cannot normalize or out of range): {len(invalid_ids)}")
    if invalid_ids:
        print("  example invalid ids (first 20):", invalid_ids[:20], "...")

    print("\n[modes] valid mode distribution (after filtering invalid ids/modes):")
    for m in sorted(VALID_MODES):
        print(f"  {m:10s}: {mode_stats[m]}")

    print("\n[pairs] expected (id, mode) pairs:", expected_pairs)
    print(f"[pairs] present unique (id, mode) pairs: {len(present_pairs)}")
    print(f"[pairs] missing (id, mode) pairs: {len(missing_pairs)}")
    if missing_pairs:
        print("  missing pairs (first 20):", missing_pairs[:20], "...")

    print(f"[pairs] duplicated (id, mode) pairs (count > 1): {len(dup_pairs)}")
    if dup_pairs:
        print("  example dup pairs (first 10):", dup_pairs[:10], "...")

    print("\n============== end report ==============")


if __name__ == "__main__":
    main()
