# src/utils/rebuild_synergy_raw_from_qwen_judge.py

import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Tuple

JUDGE_FILE = Path("outputs/metrics/synergy_jbv28k_synergy_qwen_judge.jsonl")

TOTAL_ID = 28000
VALID_MODES = {"txt_img", "txt_only", "img_only", "none"}

RAW_REBUILD_FILE = Path("outputs/logs/synergy_jbv28k_raw_rebuild.jsonl")
MISSING_PAIRS_FILE = Path("outputs/logs/synergy_jbv28k_missing_pairs.jsonl")


def normalize_id(raw: Any):
    """把 id 规范化为 int；如果无法转换，返回 None。"""
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if s.isdigit():
            return int(s)
    return None


def pick_first(rec: Dict[str, Any], keys):
    """按顺序在 rec 里找第一个存在的 key。"""
    for k in keys:
        if k in rec:
            return rec[k]
    return None


def extract_model(rec: Dict[str, Any]) -> Any:
    # 你可以在这里按实际 schema 再加 key
    return pick_first(
        rec,
        ["model", "model_name", "llm_name", "backend_model"]
    )


def extract_prompt(rec: Dict[str, Any]) -> Any:
    # 原始攻击指令 / 用户请求
    return pick_first(
        rec,
        ["prompt", "user_request", "orig_request", "original_request",
         "query", "attack_prompt", "input"]
    )


def extract_image(rec: Dict[str, Any]) -> Any:
    # 图像路径
    return pick_first(
        rec,
        ["image", "image_path", "img_path", "image_file"]
    )


def extract_output(rec: Dict[str, Any]) -> Any:
    # 模型输出文本
    return pick_first(
        rec,
        ["output", "model_response", "response", "model_output", "answer"]
    )


def main():
    print(f"[load] reading judge file: {JUDGE_FILE}")
    if not JUDGE_FILE.exists():
        raise FileNotFoundError(f"{JUDGE_FILE} does not exist")

    # 1) 读取 judge 文件
    records = []
    with JUDGE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                print("[warn] JSON decode failed:", e)

    print(f"[load] total judge records loaded: {len(records)}")

    # 2) 规范化并过滤
    pair_stats: Dict[Tuple[int, str], int] = defaultdict(int)
    # 每个 (id, mode) 只保留一条记录（最后一条）
    valid_records: Dict[Tuple[int, str], Dict[str, Any]] = {}

    invalid_ids = 0
    invalid_modes = 0

    for rec in records:
        raw_id = rec.get("id")
        mode = rec.get("mode")

        nid = normalize_id(raw_id)
        if nid is None or nid < 0 or nid >= TOTAL_ID:
            invalid_ids += 1
            continue

        if mode not in VALID_MODES:
            invalid_modes += 1
            continue

        key = (nid, mode)
        pair_stats[key] += 1
        valid_records[key] = rec  # 同一 (id, mode) 多次时，以最后一条为准

    print(f"[filter] invalid ids:   {invalid_ids}")
    print(f"[filter] invalid modes: {invalid_modes}")
    print(f"[filter] valid (id, mode) pairs: {len(valid_records)}")

    # 3) 计算缺失 (id, mode)
    all_pairs = {(i, m) for i in range(TOTAL_ID) for m in VALID_MODES}
    present_pairs = set(valid_records.keys())
    missing_pairs = sorted(all_pairs - present_pairs)

    print(f"[pairs] expected total pairs: {len(all_pairs)}")
    print(f"[pairs] present pairs:        {len(present_pairs)}")
    print(f"[pairs] missing pairs:        {len(missing_pairs)}")
    if missing_pairs:
        print("  example missing pairs (first 20):", missing_pairs[:20], "...")

    # 4) 写出重建的 raw 文件（目标格式）
    RAW_REBUILD_FILE.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    with RAW_REBUILD_FILE.open("w", encoding="utf-8") as fout:
        for (nid, mode), rec in valid_records.items():
            raw_rec = {
                # 完全对齐你现在 raw 的 schema
                "id": nid,
                "mode": mode,
                "model": extract_model(rec),
                "prompt": extract_prompt(rec),
                "image": extract_image(rec),
                "output": extract_output(rec),
            }
            fout.write(json.dumps(raw_rec, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"[write] rebuilt raw written to: {RAW_REBUILD_FILE}")
    print(f"[write] lines written: {num_written}")

    # 5) 写出缺失的 (id, mode) 列表
    with MISSING_PAIRS_FILE.open("w", encoding="utf-8") as fout:
        for nid, mode in missing_pairs:
            fout.write(json.dumps({"id": nid, "mode": mode}) + "\n")

    print(f"[write] missing (id, mode) pairs written to: {MISSING_PAIRS_FILE}")
    print("[done] rebuild finished.")


if __name__ == "__main__":
    main()
