#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

# 认为这些情况等价于“没有图像”
NULL_IMAGE_TOKENS = {"", None}
NULL_IMAGE_SUBSTR = "null_image"   # 例如 data/.../null_image.png


def has_real_image(image_path):
    """返回 True 表示有真实图像输入，False 表示等价于无图像."""
    if image_path in NULL_IMAGE_TOKENS:
        return False
    if not isinstance(image_path, str):
        return False
    if NULL_IMAGE_SUBSTR in image_path:
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Drop items whose mode requires image but image is null."
    )
    parser.add_argument("--input", required=True, help="input jsonl file")
    parser.add_argument(
        "--output",
        help="output jsonl file (default: add .imgclean.jsonl suffix)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".imgclean.jsonl")

    total = 0
    kept = 0
    dropped = 0
    json_err = 0

    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                json_err += 1
                # JSON 本身坏了，这里直接跳过（也可以按需写出）
                continue

            mode = obj.get("mode")
            image = obj.get("image")

            # 需要图像的 mode
            needs_image = mode in ("txt_img", "img_only")

            if needs_image and not has_real_image(image):
                # 这种情况：mode 需要图像，但图像是 null → 删除
                dropped += 1
                continue

            kept += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("========== DROP NULL-IMAGE ITEMS ==========")
    print(f"Input file  : {in_path}")
    print(f"Output file : {out_path}")
    print(f"Total lines : {total}")
    print(f"Kept        : {kept}")
    print(f"Dropped     : {dropped}  (mode in ['txt_img','img_only'] but image is null)")
    print(f"JSON errors : {json_err}")
    print("===========================================")


if __name__ == "__main__":
    main()
