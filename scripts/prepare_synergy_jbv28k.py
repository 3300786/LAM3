# scripts/prepare_synergy_jbv28k.py
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

from datasets import load_dataset
from tqdm import tqdm

from src.utils.runtime import load_yaml


PROMPT_CANDIDATES = [
    "jailbreak_query", "redteam_query",   # 先放在最前面
    "prompt", "attack_prompt", "jailbreak_prompt",
    "adv_prompt", "jailbreak_text", "text"
]
IMAGE_CANDIDATES = [
    "image_path", "img_path", "image", "image_name", "img_name"
]


def guess_columns(example: Dict[str, Any]) -> (str, str | None):
    """自动从第一条样本里推断文本和图片字段名."""
    prompt_key = None
    image_key = None

    for k in PROMPT_CANDIDATES:
        if k in example and isinstance(example[k], str):
            prompt_key = k
            break

    for k in IMAGE_CANDIDATES:
        if k in example:
            image_key = k
            break

    if prompt_key is None:
        raise ValueError(f"Cannot find prompt column in example keys={list(example.keys())}")

    # image_key 允许为 None（纯文本攻击）
    return prompt_key, image_key


def resolve_image_path(raw_img_val, img_root: str) -> str | None:
    """
    将原始 image 字段统一成本地路径：
      - 如果是字符串，认为是相对路径，直接拼到 img_root
      - 如果是 dict 且包含 'path'，用其 path 字段
      - 其他情况返回 None
    """
    if raw_img_val is None:
        return None

    if isinstance(raw_img_val, str):
        if raw_img_val.strip() == "":
            return None
        return os.path.join(img_root, raw_img_val)

    if isinstance(raw_img_val, dict):
        # HF 的 image feature 通常有 {'path': xxx, 'bytes': ...}
        path = raw_img_val.get("path")
        if path:
            # 如果已经是绝对路径就直接用
            if os.path.isabs(path):
                return path
            return os.path.join(img_root, path)

    # 兜底
    return None


def build_meta_from_original(cfg_path: str):
    cfg = load_yaml(cfg_path)
    data_cfg = cfg["data"]["synergy_jbv28k"]

    root = data_cfg["original_root"]          # e.g. "data/JailBreakV_28K_raw"
    meta_out = data_cfg["meta_path"]          # e.g. "data/jbv28k_synergy_meta.jsonl"
    max_samples = int(data_cfg.get("max_samples", 0) or 0)

    # 猜测主要标注文件：优先 json，然后 csv
    # 你也可以直接在 yaml 里指定 data_file
    data_file = data_cfg.get("data_file")
    if not data_file:
        # 默认尝试这两个文件名
        for cand in ["JailBreakV_28K.json", "JailBreakV_28K.csv"]:
            cand_path = os.path.join(root, cand)
            if os.path.isfile(cand_path):
                data_file = cand_path
                break
    if not data_file or not os.path.isfile(data_file):
        raise FileNotFoundError(
            f"Cannot find data_file; please set data.synergy_jbv28k.data_file in configs "
            f"or place JailBreakV_28K.json/csv under {root}"
        )

    ext = os.path.splitext(data_file)[1].lower()
    if ext == ".json":
        ds = load_dataset("json", data_files={"train": data_file})["train"]
    elif ext == ".csv":
        ds = load_dataset("csv", data_files={"train": data_file})["train"]
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")

    # 自动探测文本列和图像列
    first = ds[0]
    cfg_prompt = data_cfg.get("prompt_key")
    cfg_image = data_cfg.get("image_key")

    if cfg_prompt is not None:
        prompt_key = cfg_prompt
    else:
        prompt_key, _ = guess_columns(first)

    if cfg_image is not None:
        image_key = cfg_image
    else:
        # image 列可以不存在，所以单独再跑一次自动探测
        _, image_key = guess_columns(first)

    print(f"[meta] prompt_key = {prompt_key}, image_key = {image_key}")
    # 图像根目录：优先 yaml 里给的 image_root，否则用 root 本身
    img_root = data_cfg.get("image_root", root)

    Path(os.path.dirname(meta_out)).mkdir(parents=True, exist_ok=True)
    num = len(ds) if max_samples <= 0 else min(max_samples, len(ds))
    print(f"[meta] total rows = {len(ds)}, using {num} samples")

    with open(meta_out, "w", encoding="utf-8") as f:
        for i in tqdm(range(num), desc="build_meta"):
            ex = ds[i]
            text_attack = ex[prompt_key]
            raw_img = ex[image_key] if image_key is not None else None
            img_path = resolve_image_path(raw_img, img_root)

            rec = {
                "id": i,
                "text_attack": text_attack,
                "image_path": img_path,      # 允许为 None，后续会自动 fallback 到空白图
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[meta] wrote {num} records to {meta_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to configs/synergy_jbv28k.yaml")
    args = ap.parse_args()
    build_meta_from_original(args.cfg)
