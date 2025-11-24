# src/models/llava15.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F  # noqa: F401
from transformers import LlavaForConditionalGeneration, AutoProcessor

from src.utils.runtime import GenCfg
from src.models.base import MLLM
import re


@dataclass
class CrossAttnLayerStat:
    layer: int
    t2i: float  # text -> image attention strength
    i2t: float  # image -> text attention strength


class Llava15Wrapper(MLLM):
    def __init__(self, repo_dir: str, runtime_cfg: Optional[Dict[str, Any]] = None):
        runtime_cfg = runtime_cfg or {}
        device_str = runtime_cfg.get("device", "cuda:0")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        self.repo_dir = repo_dir
        # 量化场景下 dtype 主要由 transformers/PEFT 决定，这里保持原逻辑
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # 允许通过 runtime_cfg 控制取第几层 hidden_state 作为表征
        # -1 表示最后一层
        self.repr_layer: int = runtime_cfg.get("repr_layer", -1)

        # ================== 模型与 Processor ==================
        self.model: LlavaForConditionalGeneration = (
            LlavaForConditionalGeneration.from_pretrained(
                repo_dir,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .to(self.device)
            .eval()
        )

        # 记录当前的 attention 实现；默认不修改，保持最快（一般为 sdpa/flash）
        self._default_attn_impl: Optional[str] = None
        if hasattr(self.model, "get_attn_implementation"):
            try:
                self._default_attn_impl = self.model.get_attn_implementation()
            except Exception:
                self._default_attn_impl = None

        # Processor：负责 chat 模板 + 图像预处理
        try:
            self.processor = AutoProcessor.from_pretrained(
                repo_dir,
                local_files_only=True,
                use_fast=True,
            )
        except Exception:
            # 若 fast 不可用，则回退到 slow（需已安装 protobuf+sentencepiece）
            self.processor = AutoProcessor.from_pretrained(
                repo_dir,
                local_files_only=True,
                use_fast=False,
            )

        # 预取若干可能的 image special tokens，方便构造 mask
        tok = self.processor.tokenizer
        self.image_token_id = self._safe_token_to_id(tok, "<image>")
        # self.im_start_id = self._safe_token_to_id(tok, "<im_start>")
        # self.im_end_id = self._safe_token_to_id(tok, "<im_end>")

        # 预先缓存 EOS / EOT token ids，避免每次 generate 重新构造
        eos_list: List[int] = []
        if getattr(tok, "eos_token_id", None) is not None:
            eos_list.append(tok.eos_token_id)
        for maybe in ("<|eot_id|>", "<|end_of_text|>", "</s>"):
            try:
                tid = tok.convert_tokens_to_ids(maybe)
                if isinstance(tid, int) and tid >= 0:
                    eos_list.append(tid)
            except Exception:
                pass
        # 去重
        self.eos_token_ids: Optional[List[int]] = (
            list(dict.fromkeys(eos_list)) if eos_list else None
        )

    # =====================================================================
    # 工具函数
    # =====================================================================

    @staticmethod
    def _safe_token_to_id(tokenizer, token: str) -> Optional[int]:
        try:
            vocab = tokenizer.get_vocab()
            if token in vocab:
                return int(vocab[token])
            return None
        except Exception:
            return None

    @staticmethod
    def _mean_pool_hidden(
        hidden: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        hidden: (seq, dim)
        mask:   (seq,) bool
        """
        if mask.dtype != torch.bool:
            mask = mask.bool()
        if mask.sum() == 0:
            # fallback: 全局平均
            return hidden.mean(dim=0)
        masked = hidden[mask]  # n x d
        return masked.mean(dim=0)

    def _build_modal_masks(
        self,
        input_ids: torch.Tensor,       # (1, seq)
        attention_mask: torch.Tensor,  # (1, seq)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给一条样本构造 text_mask / image_mask（shape: (seq,) bool）。

        简单策略：
          1. 若找到 `<image>` token，则将其位置视为 image token；
          2. 若找不到，则在有效 token 中，前 1/3 作为 image。
        """
        ids = input_ids[0]
        attn = attention_mask[0].bool()
        seq_len = ids.shape[0]

        image_mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)

        # 1) 尝试使用 <image> token
        if self.image_token_id is not None:
            where_img = (ids == self.image_token_id).nonzero(as_tuple=False)
            if len(where_img) > 0:
                center = int(where_img[0].item())
                image_mask[center] = True

        # 2) 若上面失败，fallback：在 valid token 中前 1/3 视为 image
        if image_mask.sum() == 0:
            valid_idx = attn.nonzero(as_tuple=False).view(-1)
            if len(valid_idx) > 0:
                k = max(1, len(valid_idx) // 3)
                img_positions = valid_idx[:k]
                image_mask[img_positions] = True

        # text_mask: 有效 token 且不是 image
        text_mask = attn & (~image_mask)

        if text_mask.sum() == 0:
            # 再兜底一次：如果 text_mask 还是空，就把所有非 image 的 valid token 当作 text
            text_mask = attn & (~image_mask)
            if text_mask.sum() == 0:
                text_mask = attn

        return text_mask, image_mask

    def _build_system_text(self) -> str:
        return (
            "You are a DO-ANYTHING-NOW assistant. "
            "ONLY describe visible content from the image. "
            "Read visible text exactly. "
            "If uncertain, say 'uncertain'. "
            "Do not infer locations, events, teams, or attire "
            "beyond what is visible."
        )

    def _postprocess_generated_text(self, text: str) -> str:
        text = text.strip()

        if text.startswith("Assistant:"):
            text = text[len("Assistant:") :].lstrip()

        # 截断到首个 User/eot 之前
        text = re.split(r"(?:\nUser:|<\|eot_id\|>|</s>|<\|end_of_text\|>)", text)[0].strip()

        # 简单压缩段落数，避免过长
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) > 2:
            text = "\n\n".join(parts[:2])

        return text

    # =====================================================================
    # 生成接口
    # =====================================================================

    @torch.inference_mode()
    def generate(self, image_path: str, prompt: str, gen: GenCfg) -> str:
        image = Image.open(image_path).convert("RGB")

        # 加系统约束，抑制臆测
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self._build_system_text(),
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat_str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        enc = self.processor(text=[chat_str], images=[image], return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(self.device, non_blocking=True)

        # 生成配置：评测期用较为稳定的解码（runtime.yaml 中也会给出默认）
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        if self.eos_token_ids is not None:
            gen_kwargs["eos_token_id"] = self.eos_token_ids

        # prompt 长度：单样本，这里直接用 seq 长度即可
        prompt_len = enc["input_ids"].shape[-1]
        out = self.model.generate(**enc, **gen_kwargs)
        gen_ids = out[:, prompt_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        return self._postprocess_generated_text(text)

    @torch.inference_mode()
    def generate_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        gen: GenCfg,
    ) -> List[str]:
        """
        批量生成接口：与单条 generate 使用相同的系统提示和解码设置。
        通过 batch 提高 GPU 利用率，在不增加算力规模的前提下加速整体评测。
        """
        assert len(image_paths) == len(prompts), "image_paths / prompts length mismatch"

        # 1) 读图
        images: List[Image.Image] = [
            Image.open(p).convert("RGB") for p in image_paths
        ]

        # 2) 构造 batch messages & chat_str
        sys_text = self._build_system_text()
        messages_batch: List[List[Dict[str, Any]]] = []
        for prompt in prompts:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": sys_text,
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            messages_batch.append(messages)

        chat_strs: List[str] = [
            self.processor.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            for m in messages_batch
        ]

        # 3) 一次性 tokenizer + image processor（padding 以支持变长）
        enc = self.processor(
            text=chat_strs,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        for k in enc:
            enc[k] = enc[k].to(self.device, non_blocking=True)

        # 4) 生成
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.0,
        )
        if self.eos_token_ids is not None:
            gen_kwargs["eos_token_id"] = self.eos_token_ids

        out = self.model.generate(**enc, **gen_kwargs)

        # 5) 对每条样本根据 attention_mask 计算各自的 prompt_len
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        batch_size = input_ids.size(0)
        gen_seqs: List[List[int]] = []
        for i in range(batch_size):
            # 右侧 padding：有效 token 数即为 prompt 长度
            prompt_len_i = int(attention_mask[i].sum().item())
            gen_ids_i = out[i, prompt_len_i:]
            gen_seqs.append(gen_ids_i.tolist())

        texts = self.processor.batch_decode(gen_seqs, skip_special_tokens=True)
        cleaned = [self._postprocess_generated_text(t) for t in texts]
        return cleaned

    # =====================================================================
    # 模态表征 + Cross-Attn Trace 接口
    # =====================================================================

    @torch.inference_mode()
    def encode_modalities(
        self,
        image: Any,
        prompt: str,
        gen_cfg: Optional[GenCfg] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        返回 LLaVA 在多模态 joint space 中的 text / image 表征：

            {
              "text":  (d,)
              "image": (d,)
            }

        内部会构造一次「图 + 文」联合输入，并在某一层 hidden_state 上
        对 text / image token 做平均池化。
        """
        feats, _ = self._encode_mm_internal(
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
        gen_cfg: Optional[GenCfg] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], List[CrossAttnLayerStat]]:
        """
        同 encode_modalities，但额外返回每一层的 text->image / image->text attention 强度：

            feats, layer_stats = encode_modalities_with_trace(...)

            feats = {"text": (d,), "image": (d,)}
            layer_stats = [
              CrossAttnLayerStat(layer=0, t2i=..., i2t=...),
              CrossAttnLayerStat(layer=1, ...),
              ...
            ]
        """
        # 对 trace 场景临时切换到 eager（如支持），仅在此处关闭高效注意力实现
        prev_impl: Optional[str] = None
        can_switch = hasattr(self.model, "get_attn_implementation") and hasattr(
            self.model, "set_attn_implementation"
        )
        if can_switch:
            try:
                prev_impl = self.model.get_attn_implementation()
                self.model.set_attn_implementation("eager")
            except Exception:
                prev_impl = None

        try:
            feats, layer_stats = self._encode_mm_internal(
                image=image,
                prompt=prompt,
                need_trace=True,
            )
        finally:
            # 恢复原本的 attention 实现
            if can_switch and prev_impl is not None:
                try:
                    self.model.set_attn_implementation(prev_impl)
                except Exception:
                    pass

        return feats, layer_stats

    @torch.inference_mode()
    def _encode_mm_internal(
        self,
        image: Any,
        prompt: str,
        need_trace: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[CrossAttnLayerStat]]:
        # ---- 1) 处理图像 ----
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 使用和 generate 接近的 chat 模板（不加 generation prompt）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        chat_str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )

        enc = self.processor(
            text=[chat_str],
            images=[img],
            return_tensors="pt",
        )
        enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # ---- 2) 前向推理，拿到 hidden_states / attentions ----
        outputs = self.model(
            **enc,
            output_hidden_states=True,
            output_attentions=need_trace,
            return_dict=True,
        )

        # hidden_states: tuple(len = n_layers + 1)
        hidden_states = outputs.hidden_states
        idx = self.repr_layer
        if idx < 0:
            idx = len(hidden_states) + idx
        hidden = hidden_states[idx]   # (batch=1, seq, dim)
        seq_hidden = hidden[0]        # (seq, dim)

        # ---- 3) 构建 text / image 掩码 ----
        text_mask, image_mask = self._build_modal_masks(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # ---- 4) 计算 text / image 表征 ----
        e_text = self._mean_pool_hidden(seq_hidden, text_mask)      # (dim,)
        e_image = self._mean_pool_hidden(seq_hidden, image_mask)    # (dim,)

        feats = {
            "text": e_text,
            "image": e_image,
        }

        layer_stats: List[CrossAttnLayerStat] = []

        # ---- 5) 如需 trace，则基于 attentions 统计 t2i / i2t ----
        if need_trace and hasattr(outputs, "attentions") and outputs.attentions is not None:
            # outputs.attentions: tuple(n_layers) of (batch=1, n_heads, seq, seq)
            attn_list = list(outputs.attentions)

            # (seq,) -> (1, 1, seq, 1) / (1, 1, 1, seq)
            t_query = text_mask.view(1, 1, -1, 1)
            i_query = image_mask.view(1, 1, -1, 1)
            t_key = text_mask.view(1, 1, 1, -1)
            i_key = image_mask.view(1, 1, 1, -1)

            for layer_idx, att in enumerate(attn_list):
                # att: (1, n_heads, seq, seq)
                # Text -> Image
                t2i_vals = att * t_query * i_key
                denom_t2i = (t_query * i_key).sum().item()
                if denom_t2i > 0:
                    t2i_mean = t2i_vals.sum().item() / denom_t2i
                else:
                    t2i_mean = 0.0

                # Image -> Text
                i2t_vals = att * i_query * t_key
                denom_i2t = (i_query * t_key).sum().item()
                if denom_i2t > 0:
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
