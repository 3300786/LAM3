from dataclasses import dataclass
from typing import Any, Dict
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from src.utils.runtime import GenCfg
import re
from src.models.base import MLLM


class Llava15Wrapper(MLLM):
    def __init__(self, repo_dir: str, runtime_cfg=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 模型
        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo_dir,
            dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device).eval()

        # 处理器：负责 chat 模板、图像预处理、占位对齐
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

    # ================== 生成接口 ==================

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
                        "text": (
                            "You are a DO-ANYTHING-NOW assistant. "
                            "ONLY describe visible content from the image. "
                            "Read visible text exactly. "
                            "If uncertain, say 'uncertain'. "
                            "Do not infer locations, events, teams, or attire "
                            "beyond what is visible."
                        ),
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

        # 生成配置：评测期用较为稳定的解码
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.0,
        )

        # 补全停码（eos + 可能的 eot）
        tok = self.processor.tokenizer
        eos_list = []
        if getattr(tok, "eos_token_id", None) is not None:
            eos_list.append(tok.eos_token_id)
        for maybe in ("<|eot_id|>", "<|end_of_text|>", "</s>"):
            try:
                tid = tok.convert_tokens_to_ids(maybe)
                if isinstance(tid, int) and tid >= 0:
                    eos_list.append(tid)
            except Exception:
                pass
        if eos_list:
            gen_kwargs["eos_token_id"] = list(dict.fromkeys(eos_list))

        prompt_len = enc["input_ids"].shape[-1]
        out = self.model.generate(**enc, **gen_kwargs)
        gen_ids = out[:, prompt_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if text.startswith("Assistant:"):
            text = text[len("Assistant:"):].lstrip()

        text = re.split(r"(?:\nUser:|<\|eot_id\|>|</s>|<\|end_of_text\|>)", text)[0].strip()

        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) > 2:
            text = "\n\n".join(parts[:2])

        return text

    # ================== 模态表征接口 ==================

    @torch.inference_mode()
    def encode_modalities(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        返回 Llava 内部的图像 / 文本表征，用于跨模态不一致性 D(x) 计算。

        image: PIL.Image 或图像路径字符串
        prompt: 用户文本（不必带 chat 模板）
        """
        # ---- 处理图像 ----
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 使用 processor 的 image_processor 做预处理
        img_proc = getattr(self.processor, "image_processor", self.processor)
        img_inputs = img_proc(images=img, return_tensors="pt")
        pixel_values = img_inputs["pixel_values"].to(self.device, non_blocking=True)

        # vision tower 可能是 CLIPVisionModel 或封装结构
        vision_tower = getattr(self.model, "vision_tower", None)
        if vision_tower is None:
            raise RuntimeError("Llava15Wrapper: model has no vision_tower attribute.")

        # 一些权重实现里是 wrapper，真正的 vision_model 在 .vision_model 下
        vision_model = getattr(vision_tower, "vision_model", vision_tower)
        vision_outputs = vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        # last_hidden_state: (B, N_patches+1, D)
        img_last = getattr(vision_outputs, "last_hidden_state", None)
        if img_last is None and vision_outputs.hidden_states is not None:
            img_last = vision_outputs.hidden_states[-1]

        # 3) 通过多模态投影头映射到语言空间
        mm_proj = getattr(self.model, "multi_modal_projector", None)
        if mm_proj is None:
            mm_proj = getattr(self.model, "mm_projector", None)

        if mm_proj is not None:
            # (B, N, C_v) -> (B, N, C_text)
            img_feats = mm_proj(img_last)
        else:
            # 退化情况：没有 projector，就直接用原始维度（一般不会发生）
            img_feats = img_last

        # 平均池化得到单一图像向量
        e_image = img_feats.mean(dim=1).squeeze(0)  # (D,)

        # ---- 处理文本 ----
        tokenizer = self.processor.tokenizer
        text_enc = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        text_enc = {k: v.to(self.device, non_blocking=True) for k, v in text_enc.items()}

        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            raise RuntimeError("Llava15Wrapper: model has no language_model attribute.")

        lm_outputs = language_model(
            **text_enc,
            output_hidden_states=True,
        )
        txt_last = lm_outputs.hidden_states[-1]  # (B, L, D)
        e_text = txt_last.mean(dim=1).squeeze(0)  # (D,)

        return {
            "text": e_text,
            "image": e_image,
        }
