#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import defaultdict


ALLOWED_MODES = {"txt_img", "txt_only", "img_only", "none"}


# ------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------
def parse_id(raw):
    """把 obj['id'] 解析成 int，如果失败返回 None。"""
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


# ------------------------------------------------------------
# check 模式：只做体检，不改文件
# ------------------------------------------------------------
def cmd_check(input_path: str, expected_ids: int = 28000):
    path = Path(input_path)
    total_lines = 0
    bad_json_lines = []          # JSON 解析失败
    bad_idmode_lines = []        # id/mode 异常
    key_counts = defaultdict(int)  # (id, mode) -> count
    id_set = set()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            total_lines += 1
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_json_lines.append(lineno)
                continue

            id_raw = obj.get("id")
            mode_raw = obj.get("mode")
            id_int = parse_id(id_raw)
            mode_str = str(mode_raw) if mode_raw is not None else None

            if id_int is None or mode_str not in ALLOWED_MODES:
                bad_idmode_lines.append((lineno, id_raw, mode_raw))
                continue

            key = (id_int, mode_str)
            key_counts[key] += 1
            id_set.add(id_int)

    unique_keys = len(key_counts)
    dup_keys = {k: c for k, c in key_counts.items() if c > 1}

    # 统计缺失的 (id, mode)
    missing_keys = []
    for i in range(expected_ids):
        for m in ALLOWED_MODES:
            if (i, m) not in key_counts:
                missing_keys.append((i, m))

    print("========== SUMMARY ==========")
    print(f"File: {path}")
    print(f"Total lines      : {total_lines}")
    print(f"Unique (id,mode) : {unique_keys}")
    print(f"Expected combos  : {expected_ids} ids × {len(ALLOWED_MODES)} modes = {expected_ids * 4}")
    print()
    print(f"JSON parse errors: {len(bad_json_lines)}")
    print(f"Bad id/mode lines: {len(bad_idmode_lines)}")
    print(f"Duplicate keys   : {len(dup_keys)} (id,mode with count>1)")
    print(f"Missing combos   : {len(missing_keys)}")
    print()

    if bad_json_lines:
        print("---- JSON parse error lines (first 20) ----")
        for ln in bad_json_lines[:20]:
            print(f"  line {ln}")
        if len(bad_json_lines) > 20:
            print(f"  ... (+{len(bad_json_lines) - 20} more)")
        print()

    if bad_idmode_lines:
        print("---- Bad id/mode lines (first 20) ----")
        for ln, rid, rmode in bad_idmode_lines[:20]:
            print(f"  line {ln}: id={rid!r}, mode={rmode!r}")
        if len(bad_idmode_lines) > 20:
            print(f"  ... (+{len(bad_idmode_lines) - 20} more)")
        print()

    if dup_keys:
        print("---- Duplicate (id, mode) keys (first 20) ----")
        for (rid, rmode), cnt in list(dup_keys.items())[:20]:
            print(f"  id={rid}, mode={rmode}, count={cnt}")
        if len(dup_keys) > 20:
            print(f"  ... (+{len(dup_keys) - 20} more)")
        print()

    if missing_keys:
        print("---- Missing (id, mode) combos (first 20) ----")
        for rid, rmode in missing_keys[:20]:
            print(f"  id={rid}, mode={rmode}")
        if len(missing_keys) > 20:
            print(f"  ... (+{len(missing_keys) - 20} more)")
        print()

    print("========== END SUMMARY ==========")


# ------------------------------------------------------------
# clean 模式：按规则清洗并输出 *.clean.jsonl
# ------------------------------------------------------------
def cmd_clean(input_path: str, output_path: str | None, expected_ids: int = 28000):
    in_path = Path(input_path)
    if output_path is None:
        out_path = in_path.with_suffix(".clean.jsonl")
    else:
        out_path = Path(output_path)

    seen_keys = set()  # (id, mode)
    total_lines = 0
    written = 0
    skipped_json = 0
    skipped_idmode = 0
    skipped_dup = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for lineno, line in enumerate(fin, start=1):
            total_lines += 1
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped_json += 1
                continue

            id_int = parse_id(obj.get("id"))
            mode_str = str(obj.get("mode")) if obj.get("mode") is not None else None

            # 丢掉非法 id/mode
            if id_int is None or mode_str not in ALLOWED_MODES:
                skipped_idmode += 1
                continue

            key = (id_int, mode_str)
            if key in seen_keys:
                skipped_dup += 1
                continue

            seen_keys.add(key)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1

    print("========== CLEAN SUMMARY ==========")
    print(f"Input file      : {in_path}")
    print(f"Output file     : {out_path}")
    print(f"Total lines read: {total_lines}")
    print(f"Lines written   : {written}")
    print(f"Skipped (JSON error) : {skipped_json}")
    print(f"Skipped (id/mode bad): {skipped_idmode}")
    print(f"Skipped (duplicate)  : {skipped_dup}")
    print(f"Expected combos      : {expected_ids * 4}")
    print("========== END CLEAN SUMMARY ==========")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Sanity check & clean JailbreakV-28K judge jsonl files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    p_check = subparsers.add_parser("check", help="Check a jsonl file (no modification).")
    p_check.add_argument("--input", required=True, help="input jsonl file")
    p_check.add_argument("--expected-ids", type=int, default=28000,
                         help="expected number of unique ids (default: 28000)")

    # clean
    p_clean = subparsers.add_parser("clean", help="Clean a jsonl file, remove bad/dup lines.")
    p_clean.add_argument("--input", required=True, help="input jsonl file")
    p_clean.add_argument("--output", help="output jsonl file (default: *.clean.jsonl)")
    p_clean.add_argument("--expected-ids", type=int, default=28000,
                         help="expected number of unique ids (for summary only)")

    args = parser.parse_args()

    if args.command == "check":
        cmd_check(args.input, expected_ids=args.expected_ids)
    elif args.command == "clean":
        cmd_clean(args.input, args.output, expected_ids=args.expected_ids)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()

