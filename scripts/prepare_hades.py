# scripts/prepare_hades.py
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import json

ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录
OUT_ROOT = ROOT / "data" / "HADES"
IMG_DIR = OUT_ROOT / "images"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # 1) 加载 HF 数据集（默认 test split）
    ds = load_dataset("Monosail/HADES", split="test")

    # 2) 只保留 step == 5 的样本（750 条）
    ds = ds.filter(lambda ex: ex["step"] == 5)

    print(f"[hades] total step=5 samples: {len(ds)} (expected 750)")

    records = []

    for ex in tqdm(ds, desc="saving images"):
        # HF 字段名：image, id, scenario, keywords, step, category, behavior / instruction
        img: Image.Image = ex["image"]
        sample_id = ex["id"]           # e.g. Animal_000001_step5
        scenario = ex["scenario"]      # e.g. Animal
        keywords = ex["keywords"]      # e.g. beat
        category = ex["category"]      # e.g. behavior / question type

        # 文本字段有的版本叫 behavior，有的叫 instruction，这里兼容一下
        text = ex.get("behavior", None) or ex.get("instruction", None)
        if text is None:
            raise ValueError("No behavior/instruction field found in HADES sample")

        # 保存图片到本地
        img_fname = f"{sample_id}.png"
        img_path = IMG_DIR / img_fname
        img.save(img_path)

        # 构造一条通用样本记录（后面给 run_synergy 用）
        rec = {
            "id": f"hades_{sample_id}",
            "dataset": "HADES",
            "scenario": scenario,
            "keywords": keywords,
            "category": category,
            "step": int(ex["step"]),
            # LAM3 通用字段
            "mode": "img_and_txt",          # 和之前 txt_only / img_only 一致风格
            "prompt": text,                 # 给 MLLM 的文本指令
            "image": str(img_path),         # 本地图片路径（绝对 / 相对均可）
        }
        records.append(rec)

    # 3) 保存一个 meta.jsonl，格式对齐你之前 synergy 流程的 raw 输入
    out_jsonl = OUT_ROOT / "hades_750_meta.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[hades] saved meta to {out_jsonl}")
    print(f"[hades] images saved to {IMG_DIR}")

if __name__ == "__main__":
    main()
