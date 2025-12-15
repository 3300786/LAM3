#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# 认为这些路径/标记代表“无图像输入”
NULL_IMAGE_TOKENS = {
    "",
    None,
}
# 只要路径里包含 null_image 也视为“无图像”
NULL_IMAGE_SUBSTR = "null_image"


def has_text(prompt):
    """判断是否存在文本输入."""
    if prompt is None:
        return False
    if isinstance(prompt, str):
        return prompt.strip() != ""
    return False


def has_image(image_path):
    """判断是否存在图像输入（排除 null_image.png 等占位符）."""
    if image_path in NULL_IMAGE_TOKENS:
        return False
    if not isinstance(image_path, str):
        return False
    if NULL_IMAGE_SUBSTR in image_path:
        # 例如 data/synergy_jbv28k/null_image.png
        return False
    return True


# ------------------------------------------------------------
# 1) 检查各 mode 的输入是否符合期望
# ------------------------------------------------------------
def cmd_check(input_path: str):
    path = Path(input_path)
    total = 0
    mode_counts = Counter()

    # 统计：mode -> 违例类型 -> 数量
    # 违例类型：
    #   missing_text / missing_image / extra_text / extra_image / both_missing / both_present_for_none
    violations = defaultdict(Counter)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] JSON parse error at line {line_no}")
                continue

            mode = obj.get("mode")
            prompt = obj.get("prompt")
            image = obj.get("image")

            t = has_text(prompt)
            im = has_image(image)

            total += 1
            mode_counts[mode] += 1

            if mode == "txt_img":
                # 需要 text=True, image=True
                if not t and not im:
                    violations[mode]["missing_both"] += 1
                elif not t and im:
                    violations[mode]["missing_text"] += 1
                elif t and not im:
                    violations[mode]["missing_image"] += 1
            elif mode == "txt_only":
                # 需要 text=True, image=False
                if not t and not im:
                    violations[mode]["missing_text"] += 1
                elif not t and im:
                    violations[mode]["missing_text_but_has_image"] += 1
                elif t and im:
                    violations[mode]["extra_image"] += 1
            elif mode == "img_only":
                # 需要 text=False, image=True
                if not im and not t:
                    violations[mode]["missing_image"] += 1
                elif not im and t:
                    violations[mode]["missing_image_but_has_text"] += 1
                elif im and t:
                    violations[mode]["extra_text"] += 1
            elif mode == "none":
                # 需要 text=False, image=False
                if t and im:
                    violations[mode]["both_present"] += 1
                elif t and not im:
                    violations[mode]["extra_text"] += 1
                elif not t and im:
                    violations[mode]["extra_image"] += 1
            else:
                violations["UNKNOWN_MODE"]["count"] += 1

    print("========== MODE CONSISTENCY CHECK ==========")
    print(f"File: {path}")
    print(f"Total valid lines: {total}")
    print("\n-- Count by mode --")
    for m, c in mode_counts.items():
        print(f"  {m:8s}: {c}")

    print("\n-- Violations by mode --")
    if not violations:
        print("  No violations detected.")
    else:
        for m, vc in violations.items():
            print(f"  Mode={m}:")
            for vtype, cnt in vc.items():
                print(f"    {vtype:30s}: {cnt}")
    print("========== END CHECK ==========")


# ------------------------------------------------------------
# 2) 重新计算 A,B,AB 的 3-bit 编码，并统计 000–111 分布
#    A: txt_only, B: img_only, AB: txt_img
# ------------------------------------------------------------
def cmd_encode(input_path: str, output_path: str | None):
    path = Path(input_path)
    if output_path is not None:
        out_path = Path(output_path)
    else:
        out_path = path.with_suffix(".encoded.jsonl")

    # 按 id 聚合四个 mode 的 attack success
    # id -> {mode -> is_attack_success}
    per_id = defaultdict(dict)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] JSON parse error at line {line_no}")
                continue

            id_ = str(obj.get("id"))
            mode = obj.get("mode")
            judge = obj.get("qwen_judge", {})
            succ = bool(judge.get("is_attack_success", False))

            if mode not in ("txt_img", "txt_only", "img_only", "none"):
                continue

            per_id[id_][mode] = succ

    # 为每条记录写入 attack_code 字段，并统计分布
    code_counter = Counter()
    incomplete_ids = 0

    # 先全部读入，再写出（方便我们只 pass 一次）
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append(obj)

    for id_, modes in per_id.items():
        # 检查是否缺失某些 mode
        has_txt = "txt_only" in modes
        has_img = "img_only" in modes
        has_ab = "txt_img" in modes

        if not (has_txt and has_img and has_ab):
            incomplete_ids += 1

        A = 1 if modes.get("txt_only", False) else 0
        B = 1 if modes.get("img_only", False) else 0
        AB = 1 if modes.get("txt_img", False) else 0
        code = f"{A}{B}{AB}"
        code_counter[code] += 1

        # 把 code 暂存回去：id -> code
        modes["attack_code"] = code

    # 将 attack_code 写回每一条记录（如果所在 id 有 code）
    with out_path.open("w", encoding="utf-8") as fout:
        for obj in records:
            id_ = str(obj.get("id"))
            mdict = per_id.get(id_)
            if mdict and "attack_code" in mdict:
                obj["attack_code"] = mdict["attack_code"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("========== ATTACK CODE STATS ==========")
    print(f"File in : {path}")
    print(f"File out: {out_path}")
    print(f"Total unique ids: {len(per_id)}")
    print(f"Ids with missing modes(txt_only/img_only/txt_img): {incomplete_ids}")
    print("\n-- attack_code distribution (A,B,AB) --")
    for code in sorted(code_counter.keys()):
        print(f"  {code}: {code_counter[code]}")
    print("========================================")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Check mode/prompt/image consistency and recompute A,B,AB attack codes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check
    p_check = subparsers.add_parser("check", help="Check mode vs actual inputs.")
    p_check.add_argument("--input", required=True, help="input jsonl file")

    # encode
    p_encode = subparsers.add_parser("encode", help="Recompute A,B,AB attack codes.")
    p_encode.add_argument("--input", required=True, help="input jsonl file")
    p_encode.add_argument("--output", help="output jsonl file (default: *.encoded.jsonl)")

    args = parser.parse_args()

    if args.command == "check":
        cmd_check(args.input)
    elif args.command == "encode":
        cmd_encode(args.input, args.output)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
