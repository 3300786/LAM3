# src/utils/ppl_utils.py
"""
Lightweight PPL computation utilities using a small decoder-only LLM (e.g., Qwen-0.5B).

This module provides:
    - load_ppl_model(): load a HF causal LM based on models.yaml config
    - compute_ppl(): compute perplexity for a given text

PPL 模型不是 MLLM，不走 registry 的 wrapper。
只是简单地按 models.yaml 里的路径加载一个轻量级 causal LM。
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------------------------------------------
# Global cache: 避免在指标计算中重复加载模型（会很慢）
# -------------------------------------------------------------
_PPL_TOKENIZER = None
_PPL_MODEL = None
_PPL_DEVICE = None


# -------------------------------------------------------------
# 加载 PPL 模型
#   models_cfg: configs/models.yaml 解析后的 dict
#   model_name: e.g., "qwen25_0_5b_ppl"
#   runtime_cfg: 用来读取 device 类型
# -------------------------------------------------------------
def load_ppl_model(
    models_cfg: Dict[str, Any],
    model_name: str,
    runtime_cfg: Dict[str, Any],
) -> Tuple[Any, Any, torch.device]:
    global _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE

    if _PPL_MODEL is not None:
        return _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE

    if model_name not in models_cfg:
        raise ValueError(f"[PPL] model '{model_name}' not found in models.yaml")

    model_cfg = models_cfg[model_name]
    repo_id = model_cfg.get("repo_id")
    revision = model_cfg.get("revision", "main")
    if repo_id is None:
        raise ValueError(f"[PPL] models.yaml missing repo_id for '{model_name}'")

    # device
    device_str = runtime_cfg.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _PPL_DEVICE = device

    print(f"[PPL] Loading PPL model from {repo_id} (revision={revision}) on {device}...")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        revision=revision,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=revision,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()

    _PPL_MODEL = model
    _PPL_TOKENIZER = tokenizer

    return _PPL_MODEL, _PPL_TOKENIZER, _PPL_DEVICE


# -------------------------------------------------------------
# 计算 perplexity
# -------------------------------------------------------------
@torch.no_grad()
def compute_ppl(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> float:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 典型自回归 LM loss
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )

    nll = out.loss.item()
    ppl = math.exp(nll)
    return ppl
