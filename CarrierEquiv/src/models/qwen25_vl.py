# src/models/qwen25_vl.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List

import torch
from PIL import Image
from transformers import AutoProcessor

from src.models.base import MLLM
from src.utils.runtime import GenCfg

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
except Exception as e:
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:
    process_vision_info = None  # type: ignore


def _map_precision(p: str) -> torch.dtype:
    p = (p or "bf16").lower()
    if p in ("bf16", "bfloat16"):
        return torch.bfloat16
    if p in ("fp16", "float16"):
        return torch.float16
    if p in ("fp32", "float32"):
        return torch.float32
    return torch.bfloat16


def _maybe_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class Qwen25VLWrapper(MLLM):
    """
    Qwen2.5-VL wrapper matching LAM3 interface:
      __init__(repo_id: str, runtime_cfg: dict)
      generate(image, prompt, gen_cfg: GenCfg) -> str

    Notes:
    - First phase focuses on R2 (single image). Multi-image native is optional later.
    - Attention/hidden-states tracing can be added via generate_with_trace override later.
    """

    def __init__(self, repo_id: str, runtime_cfg: dict):
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError(
                "Qwen2_5_VLForConditionalGeneration not found. "
                "You likely need a compatible transformers version (often from source) for Qwen2.5-VL."
            )
        if process_vision_info is None:
            raise ImportError(
                "qwen_vl_utils.process_vision_info not found. Please install qwen-vl-utils."
            )

        self.repo_id = repo_id
        self.runtime_cfg = runtime_cfg

        precision = runtime_cfg.get("precision", "bf16")
        torch_dtype = _map_precision(precision)

        device_map = runtime_cfg.get("device_map", "auto")
        attn_impl = runtime_cfg.get("attn_implementation", "auto")

        # Quantization: keep minimal pass-through
        quant = runtime_cfg.get("quantization", {}) or {}
        quant_enabled = bool(quant.get("enabled", False))

        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # If you later unify bnb configs, add them here.
        # For now: quant_enabled only toggles whether you pass extra kwargs.
        if quant_enabled:
            # Example of a conservative bnb config mapping (optional, extend as needed)
            # Many Qwen-VL users just rely on device_map + dtype initially.
            # Leave blank unless your env already uses bnb kwargs elsewhere.
            pass

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

        self.model.eval()

    @staticmethod
    def _resolve_dtype(p: str) -> torch.dtype:
        p = (p or "bf16").lower().strip()
        if p in ("bf16", "bfloat16"):
            return torch.bfloat16
        if p in ("fp16", "float16"):
            return torch.float16
        if p in ("fp32", "float32"):
            return torch.float32
        return torch.bfloat16

    @torch.no_grad()
    def generate(
            self,
            image: Any,
            prompt: str,
            gen_cfg: GenCfg,
            *,
            system_prompt: Optional[str] = None,
            mixed_order: Optional[str] = None,  # "image_first" | "text_first"
            debug_return_prompt: bool = False,  # audit helper
    ) -> str | Tuple[str, str]:
        """
        Carrier-Equiv friendly generate():
        - system prompt is injected as system-role (control variable)
        - user content is the treatment variable:
            text-only:  user=[text]
            image-only: user=[image]
            mixed:      user=[image,text] OR [text,image] controlled by mixed_order
        """

        if image is not None and not isinstance(image, Image.Image):
            raise TypeError(f"Qwen25VLWrapper expects PIL.Image.Image or None, got: {type(image)}")

        max_new_tokens = int(getattr(gen_cfg, "max_new_tokens", 256))
        min_new_tokens = int(getattr(gen_cfg, "min_new_tokens", 0))
        do_sample = bool(getattr(gen_cfg, "do_sample", False))
        temperature = float(getattr(gen_cfg, "temperature", 1.0))
        top_p = float(getattr(gen_cfg, "top_p", 1.0))
        seed = getattr(gen_cfg, "seed", None)

        # ---- system prompt (control variable) ----
        if system_prompt is None:
            system_prompt = self.runtime_cfg.get("global_system_prompt", None)

        # ---- mixed order resolution ----
        # priority: explicit arg > gen_cfg.mixed_order > runtime_cfg["mixed_order"] > "text_first"
        if mixed_order is None:
            mixed_order = getattr(gen_cfg, "mixed_order", None)
        if mixed_order is None:
            mixed_order = self.runtime_cfg.get("mixed_order", None)
        if mixed_order is None:
            mixed_order = "text_first"
        mixed_order = str(mixed_order).lower().strip()
        if mixed_order not in ("image_first", "text_first"):
            raise ValueError(f"mixed_order must be 'image_first' or 'text_first', got: {mixed_order}")

        # ---- messages ----
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": str(system_prompt)}]}
            )

        prompt = "" if prompt is None else str(prompt)
        prompt_is_empty = (prompt.strip() == "")

        if image is None:
            # text-only
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
        else:
            if prompt_is_empty:
                # image-only
                messages.append({"role": "user", "content": [{"type": "image", "image": image}]})
            else:
                # mixed (order controlled)
                if mixed_order == "image_first":
                    content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
                else:
                    content = [{"type": "text", "text": prompt}, {"type": "image", "image": image}]
                messages.append({"role": "user", "content": content})

        # ---- build inputs ----
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        if not image_inputs:
            image_inputs = None
        if not video_inputs:
            video_inputs = None

        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device = self.model.device
        inputs = _maybe_to_device(inputs, device)

        # ---- deterministic generation: use per-call generator (avoid global seed pollution) ----
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,  # expect 0 in your new setup
            do_sample=do_sample,  # expect False in your new setup
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            generator=generator,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[0][prompt_len:]
        out_text = self.processor.decode(gen_ids, skip_special_tokens=True).strip()

        if debug_return_prompt:
            return out_text, chat_text
        return out_text
