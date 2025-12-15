# scripts/run_synergy_hades.py
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set

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


def load_completed_pairs(log_path: Path) -> Set[Tuple[str, str]]:
    """
    从已有的 raw.jsonl 中加载已经完成的 (id, mode)。
    若文件不存在，则返回空集合。
    """
    completed: Set[Tuple[str, str]] = set()
    if not log_path.is_file():
        return completed

    n_lines = 0
    n_ok = 0
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("id", None)
            mode = obj.get("mode", None)
            if cid is None or mode is None:
                continue
            completed.add((str(cid), str(mode)))
            n_ok += 1

    print(
        f"[resume] loaded {n_ok} completed (id, mode) pairs "
        f"from {log_path} (lines={n_lines})"
    )
    return completed


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
# Image resolver + Query builder
# ----------------------------------------------------
def resolve_image_abs(case: Dict, dataset_root: Path) -> Optional[str]:
    """
    将 case 中的图像路径解析为绝对路径（若文件存在），否则返回 None。

    HADES meta 示例：
    {
      "id": "hades_Animal_000001_step5",
      "dataset": "HADES",
      "scenario": "Animal",
      "keywords": "beat",
      "category": "behavior",
      "step": 5,
      "mode": "img_and_txt",
      "prompt": "...",
      "image": "/data2/.../HADES/images/Animal_000001_step5.png"
    }

    这里兼容两种字段：
    - 优先使用 "image_path"
    - 若无，则使用 "image"
    """
    image_rel = case.get("image_path", None)
    if not image_rel:
        image_rel = case.get("image", None)
    if not image_rel:
        return None

    image_rel = str(image_rel)
    p = Path(image_rel)

    # 情况1：绝对路径，直接用
    if p.is_absolute():
        ipath = p
    # 情况2：已经是仓库内的完整相对路径，如 "data/HADES/xxx"
    elif image_rel.startswith("data/"):
        ipath = Path(image_rel)
    # 情况3：纯相对路径，如 "images/xxx.png"
    else:
        ipath = dataset_root / image_rel

    return str(ipath) if ipath.is_file() else None


def build_query(
    case: Dict,
    mode: str,
    img_abs: Optional[str],
    null_image_path: str,
) -> Tuple[str, str]:
    """
    返回 prompt 与 image_path 字符串（与 smoke_test 对齐）。

    HADES 中：
    - 文本字段为 "prompt"
    JBV-28K 中：
    - 文本字段为 "text_attack"

    这里统一成 text_attack 优先，缺失时回退到 prompt。
    """
    text_attack = case.get("text_attack", None)
    if text_attack is None:
        text_attack = case.get("prompt", "") or ""
    else:
        text_attack = text_attack or ""

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
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="configs/synergy_hades.yaml",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="总 shard 数，用于多卡切片。默认为 1（不切片）。",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="当前进程的 shard 索引（0-based）。仅当 num_shards > 1 时生效。",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    # 注意：这里和 jbv28k 的 key 不同
    d_cfg = cfg["data"]["hades"]
    e_cfg = cfg["eval"]
    m_cfg = cfg["model"]
    l_cfg = cfg["log"]

    meta_path = Path(d_cfg["meta_path"])
    dataset_root = meta_path.parent

    modes = e_cfg["modes"]
    max_samples = e_cfg.get("max_samples", None)
    batch_size: int = e_cfg.get("batch_size", 4)  # 可在 yaml 中配置

    num_shards: int = max(1, int(getattr(args, "num_shards", 1)))
    shard_idx: int = int(getattr(args, "shard_idx", 0))
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx={shard_idx} out of range for num_shards={num_shards}")

    raw_log_path = Path(l_cfg["raw_log_path"])
    raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    # 占位图
    null_image_path = ensure_null_image(dataset_root)

    # 读取已完成的 (id, mode)，用于 resume
    completed_pairs = load_completed_pairs(raw_log_path)

    # 模型
    model, gen_cfg = build_model_and_gen(m_cfg)

    # 为迭代器计数：构造 list 才能 tqdm
    cases = list(iter_meta(meta_path, max_samples=max_samples))
    print(f"[run] total samples = {len(cases)}")
    print(f"[run] modes = {modes}")
    print(f"[run] batch_size = {batch_size}")
    print(f"[run] num_shards = {num_shards}, shard_idx = {shard_idx}")

    # 1) 基于 meta 构造所有“待完成任务”（过滤掉已完成的 id+mode）
    #    每个任务为 (cid, mode, prompt, img)
    tasks: List[Tuple[str, str, str, str]] = []
    for case in cases:
        cid = str(case["id"])
        img_abs = resolve_image_abs(case, dataset_root)
        for mode in modes:
            key = (cid, str(mode))
            if key in completed_pairs:
                continue
            prompt, img = build_query(case, str(mode), img_abs, null_image_path)
            tasks.append((cid, str(mode), prompt, img))

    total_tasks = len(tasks)
    print(
        f"[run] pending tasks after resume filter = {total_tasks} "
        f"(skipped {len(completed_pairs)} completed pairs)"
    )

    if total_tasks == 0:
        print("[run] nothing to do, all (id, mode) pairs already completed.")
        return

    # 2) 在“待完成任务”上做 shard 切分
    if num_shards > 1:
        shard_tasks = [
            task for idx, task in enumerate(tasks) if idx % num_shards == shard_idx
        ]
    else:
        shard_tasks = tasks

    print(
        f"[run] shard {shard_idx}/{num_shards}: "
        f"{len(shard_tasks)} tasks to process."
    )

    if not shard_tasks:
        print("[run] this shard has no tasks to process. exit.")
        return

    # 如果 wrapper 支持 generate_batch 则优先使用，否则退回单样本 generate
    has_generate_batch = callable(getattr(model, "generate_batch", None))

    # 主循环（对 shard_tasks 进行 batch 推理）
    buffer: List[Tuple[str, str, str, str]] = []  # (cid, mode, prompt, img)

    # 以 append 模式写入，避免覆盖已有日志，实现 resume
    with raw_log_path.open("a", encoding="utf-8") as fout:
        for cid, mode, prompt, img in tqdm(
            shard_tasks, desc="Processing tasks", ncols=100
        ):
            buffer.append((cid, mode, prompt, img))

            if len(buffer) >= batch_size:
                cids = [x[0] for x in buffer]
                modes_b = [x[1] for x in buffer]
                prompts_b = [x[2] for x in buffer]
                imgs_b = [x[3] for x in buffer]

                if has_generate_batch:
                    texts = model.generate_batch(imgs_b, prompts_b, gen_cfg)
                else:
                    texts = [
                        model.generate(img, prompt, gen_cfg)
                        for img, prompt in zip(imgs_b, prompts_b)
                    ]

                for cid_i, mode_i, prompt_i, img_i, text_i in zip(
                    cids, modes_b, prompts_b, imgs_b, texts
                ):
                    log = {
                        "id": cid_i,
                        "mode": mode_i,
                        "model": m_cfg["name"],
                        "prompt": prompt_i,
                        "image": img_i,
                        "output": text_i,
                        # 下面这些字段在 HADES meta 中是可选的，若不存在则为 None
                        "dataset": "HADES",
                        "scenario": next((c.get("scenario") for c in cases if str(c["id"]) == cid_i), None),
                        "keywords": next((c.get("keywords") for c in cases if str(c["id"]) == cid_i), None),
                        "category": next((c.get("category") for c in cases if str(c["id"]) == cid_i), None),
                    }
                    fout.write(json.dumps(log, ensure_ascii=False) + "\n")

                buffer.clear()

        # 收尾：处理最后不足一个 batch 的样本
        if buffer:
            cids = [x[0] for x in buffer]
            modes_b = [x[1] for x in buffer]
            prompts_b = [x[2] for x in buffer]
            imgs_b = [x[3] for x in buffer]

            if has_generate_batch:
                texts = model.generate_batch(imgs_b, prompts_b, gen_cfg)
            else:
                texts = [
                    model.generate(img, prompt, gen_cfg)
                    for img, prompt in zip(imgs_b, prompts_b)
                ]

            for cid_i, mode_i, prompt_i, img_i, text_i in zip(
                cids, modes_b, prompts_b, imgs_b, texts
            ):
                log = {
                    "id": cid_i,
                    "mode": mode_i,
                    "model": m_cfg["name"],
                    "prompt": prompt_i,
                    "image": img_i,
                    "output": text_i,
                    "dataset": "HADES",
                    "scenario": next((c.get("scenario") for c in cases if str(c["id"]) == cid_i), None),
                    "keywords": next((c.get("keywords") for c in cases if str(c["id"]) == cid_i), None),
                    "category": next((c.get("category") for c in cases if str(c["id"]) == cid_i), None),
                }
                fout.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"[run] raw logs (including previous + new) saved to {raw_log_path}")


if __name__ == "__main__":
    main()
