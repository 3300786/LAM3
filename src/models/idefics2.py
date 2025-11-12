# src/models/idefics2.py
from typing import Optional, Any, Dict
from dataclasses import dataclass
from contextlib import suppress
from PIL import Image
import torch

from transformers import (
    Idefics2Processor,
    Idefics2ForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from src.utils.runtime import GenCfg

# ---- dtype 解析 ----
_DTYPE_ALIASES = {
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "fp16": "float16",  "float16": "float16",
    "fp32": "float32",  "float32": "float32",
}
def _parse_dtype(name: Optional[str]):
    key = (name or "bfloat16").lower()
    real = _DTYPE_ALIASES.get(key, key)
    return getattr(torch, real, torch.bfloat16)

# ---- bnb 4bit 配置 ----
def _bnb4bit_cfg(runtime_cfg: dict) -> BitsAndBytesConfig:
    q = runtime_cfg.get("quantization", {}) or {}
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_parse_dtype(q.get("compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=q.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("use_double_quant", True),
    )

# ---- 递归搬运到同一设备 ----
def _move_to_device(obj: Any, device: str):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)
    return obj


class Idefics2Wrapper:
    def __init__(self, repo_id: str, runtime_cfg: dict):
        self._device = str(runtime_cfg.get("device", "cuda:0"))
        img_short = int(runtime_cfg.get("image_short_edge", 336))
        attn_impl = runtime_cfg.get("attention_impl", "sdpa")

        if torch.cuda.is_available() and self._device.startswith("cuda:"):
            torch.cuda.set_device(int(self._device.split(":")[-1]))
        torch.backends.cuda.matmul.allow_tf32 = True
        with suppress(Exception):
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
            if attn_impl == "sdpa":
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)

        quant = runtime_cfg.get("quantization", {}) or {}
        use_q = bool(quant.get("enabled", True))
        if use_q:
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                repo_id,
                low_cpu_mem_usage=True,
                quantization_config=_bnb4bit_cfg(runtime_cfg),
                device_map=None,   # 让 HF 正常把模块放到 cuda:0
            )
        else:
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                repo_id,
                low_cpu_mem_usage=True,
                torch_dtype=_parse_dtype(runtime_cfg.get("precision", "bf16")),
            )
            if torch.cuda.is_available():
                self.model = self.model.to(self._device)

        # 用 Idefics2Processor（官方文档建议）
        self.processor = Idefics2Processor.from_pretrained(repo_id)

        # 控制短边
        ip = getattr(self.processor, "image_processor", None)
        if ip and hasattr(ip, "size") and isinstance(ip.size, dict) and "shortest_edge" in ip.size:
            ip.size["shortest_edge"] = img_short

        with suppress(Exception):
            if attn_impl == "sdpa":
                self.model.config.attn_implementation = "sdpa"

        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_path: str, prompt: str, gen: GenCfg) -> str:
        image = Image.open(image_path).convert("RGB")

        # 官方消息格式 + 让模板插入 <image> 与 <end_of_utterance>
        messages = [{
            "role": "user",
            "content": [
                {"type": "text",  "text": prompt},
                {"type": "image"},
            ],
        }]

        # 返回字符串文本，其中已包含 <image> 与 <end_of_utterance>\nAssistant:
        chat_str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,   # 关键
            tokenize=False
        )

        # 一次性构造对齐好的输入
        inputs = self.processor(images=[image], text=[chat_str], return_tensors="pt")
        # 不要动 position_ids，避免对齐被破坏
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._device, non_blocking=True)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=bool(gen.do_sample),
        )
        if gen.do_sample:
            gen_kwargs.update(temperature=float(gen.temperature), top_p=float(gen.top_p))

        # 只解码新增 token，避免把 User/模板串进输出
        prompt_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, **gen_kwargs)
        gen_ids = out[:, prompt_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # 可选：若模板里含有 "Assistant:" 前缀，做一次清洗
        if text.startswith("Assistant:"):
            text = text[len("Assistant:"):].lstrip()
        return text
