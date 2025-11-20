# scripts/run_synergy_jbv28k.py
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import yaml
from PIL import Image
from tqdm import tqdm

from src.utils.runtime import load_yaml as load_yaml_runtime, set_seed, GenCfg
from src.models.registry import build_model


# ----------------------------------------------------
# Config helpers
# ----------------------------------------------------
def load_cfg(cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_meta(path, max_samples=None):
    # interpret 0 / None / negative as no limit
    if max_samples is None or max_samples <= 0:
        limit = None
    else:
        limit = max_samples

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)



# ----------------------------------------------------
# Null image
# ----------------------------------------------------
def ensure_null_image(dataset_root: Path) -> str:
    """
    为 txt_only 与 none 模式生成空白占位图。
    返回绝对路径 str。
    """
    null_path = dataset_root / "null_image.png"
    if not null_path.is_file():
        null_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        img.save(null_path)
        print(f"[run] created null image at {null_path}")
    return str(null_path)


# ----------------------------------------------------
# Query builder
# ----------------------------------------------------
def build_query(
    case: Dict, mode: str, dataset_root: Path, null_image_path: str
) -> Tuple[str, Optional[str]]:
    """
    返回 prompt 与 image_path 字符串（与 smoke_test 对齐）。
    - text_attack: 文本攻击提示
    - image_path: 可能是相对路径（相对于 dataset_root），也可能是 "data/..." 这种仓库内全路径
    """
    text_attack = case.get("text_attack", "") or ""
    image_rel = case.get("image_path", None)

    img_abs = None
    if image_rel:
        image_rel = str(image_rel)
        p = Path(image_rel)

        # 情况1：绝对路径，直接用
        if p.is_absolute():
            ipath = p
        # 情况2：已经是仓库内的完整相对路径，如 "data/JailBreakV_28K/xxx"
        elif image_rel.startswith("data/"):
            ipath = Path(image_rel)



        # 情况3：纯相对路径，如 "llm_transfer_attack/xxx.png"
        else:
            ipath = dataset_root / image_rel

        # 只在文件真实存在时才使用该路径
        if ipath.is_file():
            img_abs = str(ipath)

    if mode == "txt_img":
        prompt = text_attack
        image = img_abs if img_abs is not None else null_image_path
    elif mode == "txt_only":
        prompt = text_attack
        image = null_image_path
    elif mode == "img_only":
        prompt = ""
        image = img_abs if img_abs is not None else null_image_path
    elif mode == "none":
        prompt = ""
        image = null_image_path
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # print(prompt, image)
    return prompt, image




# ----------------------------------------------------
# Model loader (smoke_test 风格)
# ----------------------------------------------------
def build_model_and_gen(m_cfg: Dict):
    models_cfg_path = m_cfg.get("models_cfg_path", "configs/models.yaml")
    runtime_cfg_path = m_cfg.get("runtime_cfg_path", "configs/runtime.yaml")

    models = load_yaml_runtime(models_cfg_path)
    runtime = load_yaml_runtime(runtime_cfg_path)

    set_seed(runtime.get("seed", 42))

    model_name = m_cfg["name"]
    print("[run] model name:", model_name)
    model = build_model(model_name, models, runtime)

    gen = GenCfg(
        max_new_tokens=m_cfg.get("max_new_tokens", runtime.get("max_new_tokens", 64)),
        min_new_tokens=m_cfg.get("min_new_tokens", runtime.get("min_new_tokens", 0)),
        do_sample=m_cfg.get("do_sample", runtime.get("do_sample", False)),
        temperature=m_cfg.get("temperature", runtime.get("temperature", 0.0)),
        top_p=m_cfg.get("top_p", runtime.get("top_p", 1.0)),
    )
    return model, gen


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True,
                        help="configs/synergy_jbv28k.yaml")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    d_cfg = cfg["data"]["synergy_jbv28k"]
    e_cfg = cfg["eval"]
    m_cfg = cfg["model"]
    l_cfg = cfg["log"]

    meta_path = Path(d_cfg["meta_path"])
    dataset_root = meta_path.parent

    modes = e_cfg["modes"]
    max_samples = e_cfg.get("max_samples", None)

    raw_log_path = Path(l_cfg["raw_log_path"])
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 占位图
    null_image_path = ensure_null_image(dataset_root)

    # 模型

    model, gen_cfg = build_model_and_gen(m_cfg)

    # 为迭代器计数：构造 list 才能 tqdm
    cases = list(iter_meta(meta_path, max_samples=max_samples))

    print(f"[run] total samples = {len(cases)}")

    # 主循环（带进度条）
    with raw_log_path.open("w", encoding="utf-8") as fout:
        for case in tqdm(cases, desc="Processing samples", ncols=100):
            cid = case["id"]
            for mode in modes:
                prompt, img = build_query(case, mode, dataset_root, null_image_path)
                text = model.generate(img, prompt, gen_cfg)

                log = {
                    "id": cid,
                    "mode": mode,
                    "model": m_cfg["name"],
                    "prompt": prompt,
                    "image": img,
                    "output": text,
                }
                fout.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[run] raw logs saved to {raw_log_path}")


if __name__ == "__main__":
    main()
