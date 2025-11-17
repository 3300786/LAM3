# src/metrics/collect_attn_stats.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np
import torch

from src.models.registry import build_model
from src.utils.runtime import GenCfg


RAW_LOG_FIELDS = {
    "id": "id",
    "mode": "mode",
    "prompt": "prompt",      # 文本字段名称（自行对齐）
    "image_path": "image",   # 图像路径字段名称（自行对齐）
}


def load_raw_logs(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    读取 raw synergy 日志，按 (id, mode) 组织。
    期望每行类似：
      {"id": "...", "mode": "txt_img"/"txt_only"/..., "prompt": "...", "image": "path/to/img"}
    """
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj[RAW_LOG_FIELDS["id"]]
            mode = obj[RAW_LOG_FIELDS["mode"]]

            grouped.setdefault(cid, {})[mode] = obj
    return grouped


def load_image(image_path: str):
    """
    根据你的项目实际实现：返回可被模型 wrapper 接受的 image 对象。
    可以是 PIL.Image / ndarray / tensor 等。
    这里给一个占位实现。
    """
    from PIL import Image
    return Image.open(image_path).convert("RGB")


def build_safe_token_set() -> Set[str]:
    """
    定义一组 safety 相关 token。实际可根据 tokenizer 的分词结果微调。
    """
    return {
        "safe", "safely", "harmless", "cannot", "can", "not",
        "sorry", "responsible", "unsafe", "dangerous", "avoid",
        "illegal", "harm", "harmful",
    }


def compute_safe_attention_sum(
    cross_attn: torch.Tensor,
    text_tokens: List[str],
    safe_tokens: Set[str],
) -> float:
    """
    cross_attn: (L_txt, L_img)，我们把每个 text token 的对整张图像的注意力求和，
    然后在 safety token 上再求和，得到一个标量。
    """
    # sum over image dimension -> shape (L_txt,)
    attn_txt = cross_attn.sum(dim=-1)  # (L_txt,)
    safe_indices = [i for i, tok in enumerate(text_tokens) if tok.lower() in safe_tokens]
    if not safe_indices:
        return 0.0
    safe_vals = attn_txt[safe_indices]
    return float(safe_vals.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw synergy JSONL")
    ap.add_argument("--model_name", required=True, help="model name in models config / registry")
    ap.add_argument("--models_cfg", required=True, help="YAML for models (传给 build_model)")
    ap.add_argument("--out", required=True, help="output attn_stats.jsonl")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) 构建模型
    models_cfg_path = Path(args.models_cfg)
    import yaml
    with models_cfg_path.open("r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    model = build_model(args.model_name, models_cfg, runtime=None)  # 根据你现有 registry 调整
    model_device = args.device

    # 2) 加载 raw logs
    grouped = load_raw_logs(raw_path)
    print(f"[attn] loaded {len(grouped)} sample groups from {raw_path}")

    safe_tokens = build_safe_token_set()
    gen_cfg = GenCfg()  # 如需要可根据项目实际传参

    n_samples = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for cid, modes in grouped.items():
            if "txt_img" not in modes or "txt_only" not in modes:
                continue

            log_tv = modes["txt_img"]
            log_t0 = modes["txt_only"]

            prompt = log_tv[RAW_LOG_FIELDS["prompt"]]
            img_path = log_tv[RAW_LOG_FIELDS["image_path"]]

            # 2.1 txt+img
            image = load_image(img_path)
            trace_tv = model.generate_with_trace(image, prompt, gen_cfg)
            cross_attn_tv = trace_tv.get("cross_attn", None)
            tokens_tv = trace_tv.get("text_tokens", None)

            # 2.2 txt-only（无图，或传入一个全灰图）
            trace_t0 = model.generate_with_trace(None, prompt, gen_cfg)
            cross_attn_t0 = trace_t0.get("cross_attn", None)
            tokens_t0 = trace_t0.get("text_tokens", None)

            # 如果没提供 cross_attn，跳过
            if cross_attn_tv is None or cross_attn_t0 is None:
                continue

            # 确保是 tensor
            if not isinstance(cross_attn_tv, torch.Tensor):
                cross_attn_tv = torch.as_tensor(cross_attn_tv)
            if not isinstance(cross_attn_t0, torch.Tensor):
                cross_attn_t0 = torch.as_tensor(cross_attn_t0)

            # 对齐长度（实际可根据你的实现调整）
            L = min(cross_attn_tv.shape[0], cross_attn_t0.shape[0])
            cross_attn_tv = cross_attn_tv[:L]
            cross_attn_t0 = cross_attn_t0[:L]

            # 3) 计算 ΔA（L2范数）
            delta_attn = cross_attn_tv - cross_attn_t0
            delta_attn_norm = float(torch.norm(delta_attn).item())

            # 4) 计算 ΔS（safety token 注意力差）
            tokens = tokens_tv if tokens_tv is not None else tokens_t0
            if tokens is None:
                delta_safe_attn = 0.0
            else:
                S_tv = compute_safe_attention_sum(cross_attn_tv, tokens, safe_tokens)
                S_t0 = compute_safe_attention_sum(cross_attn_t0, tokens, safe_tokens)
                delta_safe_attn = S_tv - S_t0

            rec = {
                "id": cid,
                "delta_attn_norm": delta_attn_norm,
                "delta_safe_attn": delta_safe_attn,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_samples += 1

    print(f"[attn] wrote {n_samples} attn records to {out_path}")


if __name__ == "__main__":
    main()
