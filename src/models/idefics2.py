# src/models/idefics2.py
from typing import Optional, Any, Dict, List, Tuple
from dataclasses import dataclass
from contextlib import suppress
from PIL import Image
import torch
from src.models.base import MLLM

from transformers import (
    Idefics2Processor,
    Idefics2ForConditionalGeneration,
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


@dataclass
class CrossAttnLayerStat:
    layer: int
    t2i: float  # text -> image attention strength
    i2t: float  # image -> text attention strength


class Idefics2Wrapper(MLLM):
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
                device_map="auto",   # 让 HF 正常把模块放到 cuda:0
            )
        else:
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                repo_id,
                low_cpu_mem_usage=True,
                dtype=_parse_dtype(runtime_cfg.get("precision", "bf16")),
            )
            if torch.cuda.is_available():
                self.model = self.model.to(self._device)
        # self.model.set_attn_implementation("eager")

        # 用 Idefics2Processor（官方文档建议）
        self.processor = Idefics2Processor.from_pretrained(repo_id)

        # 控制短边
        ip = getattr(self.processor, "image_processor", None)
        if ip and hasattr(ip, "size") and isinstance(ip.size, dict) and "shortest_edge" in ip.size:
            ip.size["shortest_edge"] = img_short

        with suppress(Exception):
            # 注意：如果需要 output_attentions，建议在 runtime_cfg 里设置 attention_impl="eager"
            self.model.config.attn_implementation = attn_impl

        self.model.eval()

    # ------------------------------------------------------------------
    # 生成接口
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 内部：构造 text / image mask
    # ------------------------------------------------------------------
    def _build_text_image_masks(
        self,
        input_ids: torch.Tensor,           # (seq_len,)
        attention_mask: Optional[torch.Tensor],  # (seq_len,) 或 None
        image_token_id: Optional[int],
        pad_id: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          text_mask:  (seq_len,) bool
          image_mask: (seq_len,) bool
        """
        seq_len = input_ids.shape[0]
        if attention_mask is not None:
            attn = attention_mask.bool()
        else:
            attn = torch.ones(seq_len, dtype=torch.bool, device=input_ids.device)

        text_mask = attn.clone()
        if pad_id is not None:
            text_mask &= (input_ids != pad_id)
        if image_token_id is not None:
            text_mask &= (input_ids != image_token_id)

        if image_token_id is not None:
            image_mask = (input_ids == image_token_id) & attn
        else:
            image_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # 安全兜底：若 text_mask 为空，则退化为所有非 pad token
        if not text_mask.any():
            text_mask = attn

        return text_mask, image_mask

    # ------------------------------------------------------------------
    # 内部：encode + (可选) trace
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _encode_modalities_internal(
        self,
        image: Any,
        prompt: str,
        need_trace: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[CrossAttnLayerStat]]:
        # 1) 处理图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2) 构造 chat 模板
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]
        chat_str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 3) processor → BatchEncoding；统一搬到 self._device
        inputs = self.processor(images=[image], text=[chat_str], return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        else:
            inputs = _move_to_device(inputs, self._device)

        # 4) 前向
        outputs = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            output_attentions=need_trace,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # tuple(len=L): (1, seq_len, hidden)
        if isinstance(hidden_states, (tuple, list)):
            last_hidden = hidden_states[-1]  # (1, seq_len, hidden)
        else:
            last_hidden = hidden_states

        # 5) 基于 input_ids / attention_mask 划分 text / image token
        input_ids = inputs["input_ids"][0]  # (seq_len,)
        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask[0]

        tokenizer = self.processor.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)

        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            with suppress(Exception):
                image_token_id = tokenizer.convert_tokens_to_ids("<image>")

        text_mask, image_mask = self._build_text_image_masks(
            input_ids=input_ids,
            attention_mask=attn_mask,
            image_token_id=image_token_id,
            pad_id=pad_id,
        )

        # 6) 计算 e_text / e_image
        if text_mask.any():
            text_tokens = last_hidden[0, text_mask, :]  # (n_text, hidden)
            e_text = text_tokens.mean(dim=0)
        else:
            e_text = last_hidden[0].mean(dim=0)

        if image_mask.any():
            image_tokens = last_hidden[0, image_mask, :]  # (n_image, hidden)
            e_image = image_tokens.mean(dim=0)
        else:
            e_image = last_hidden[0].mean(dim=0)

        feats = {
            "text": e_text,
            "image": e_image,
        }

        layer_stats: List[CrossAttnLayerStat] = []

        # 7) 如需 trace，统计每层跨模态 self-attn
        if need_trace and hasattr(outputs, "attentions") and outputs.attentions is not None:
            attn_list = list(outputs.attentions)  # tuple(L) of (B, H, S, S)
            # (seq,) -> (1, 1, S, 1) / (1, 1, 1, S)
            t_query = text_mask.view(1, 1, -1, 1)
            i_query = image_mask.view(1, 1, -1, 1)
            t_key = text_mask.view(1, 1, 1, -1)
            i_key = image_mask.view(1, 1, 1, -1)

            for layer_idx, att in enumerate(attn_list):
                # att: (1, n_heads, S, S)
                # Text -> Image
                t2i_mask = t_query * i_key
                denom_t2i = t2i_mask.sum().item()
                if denom_t2i > 0:
                    t2i_vals = att * t2i_mask
                    t2i_mean = t2i_vals.sum().item() / denom_t2i
                else:
                    t2i_mean = 0.0

                # Image -> Text
                i2t_mask = i_query * t_key
                denom_i2t = i2t_mask.sum().item()
                if denom_i2t > 0:
                    i2t_vals = att * i2t_mask
                    i2t_mean = i2t_vals.sum().item() / denom_i2t
                else:
                    i2t_mean = 0.0

                layer_stats.append(
                    CrossAttnLayerStat(
                        layer=layer_idx,
                        t2i=float(t2i_mean),
                        i2t=float(i2t_mean),
                    )
                )

        return feats, layer_stats

    # ------------------------------------------------------------------
    # 公共接口：encode_modalities / encode_modalities_with_trace
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode_modalities(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, _ = self._encode_modalities_internal(
            image=image,
            prompt=prompt,
            need_trace=False,
        )
        return feats

    @torch.inference_mode()
    def encode_modalities_with_trace(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg | None = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[CrossAttnLayerStat]]:
        """
        返回:
          - feats: 与 encode_modalities 相同的 text/image 表征
          - layer_stats: 每层 text->image / image->text self-attn 平均强度
        """
        feats, layer_stats = self._encode_modalities_internal(
            image=image,
            prompt=prompt,
            need_trace=True,
        )
        return feats, layer_stats
