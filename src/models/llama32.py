from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from src.models.base import MLLM   # 和你现有代码保持一致
from src.utils.runtime import GenCfg


class Llama32VisionWrapper(MLLM):
    def __init__(self, repo_dir: str, runtime_cfg=None):
        # 你项目里如果有统一的 device 逻辑，可以替换成 get_device()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = (
            torch.bfloat16 if self.device.type == "cuda" else torch.float32
        )

        # 注意：不要手动设置 flash_attention_2，Mllama 目前不支持，会报错:contentReference[oaicite:4]{index=4}
        self.model = MllamaForConditionalGeneration.from_pretrained(
            repo_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=None if self.device.type == "cpu" else {"": self.device},
        ).eval()

        # 处理器：chat template + 图像预处理
        self.processor = AutoProcessor.from_pretrained(
            repo_dir,
            use_fast=True,
        )

        self.runtime_cfg = runtime_cfg or {}

    def _build_messages(self, prompt: str, has_image: bool) -> List[Dict[str, Any]]:
        """
        和 Llava / IDEFICS 对齐的 chat 格式：
        - has_image=True: user 的 content 里包含一张 image + 一段 text
        - has_image=False: 只有 text
        具体的“txt_only / img_only / txt_img / none” 切换，仍由上游 pipeline 控制：
          - img_only: 上游传入 img!=None, prompt 是你构造的图像任务指令；
          - none: img=None, prompt 是纯文本占位任务。
        """
        if has_image:
            content = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        else:
            content = [
                {"type": "text", "text": prompt},
            ]
        return [{"role": "user", "content": content}]

    def _extract_assistant_text(self, text: str) -> str:
        """
        Llama 3.2 系列的 chat 模板会在 decode 后包含完整对话，
        一般形如:
          <|start_header_id|>user...<|end_header_id|>...<|eot_id|>...
          <|start_header_id|>assistant<|end_header_id|> ...
        这里简单做一次 split，和你现有 wrapper 的逻辑保持风格一致。
        """
        sep = "<|start_header_id|>assistant<|end_header_id|>"
        if sep in text:
            text = text.split(sep, 1)[1]
        text = text.replace("<|eot_id|>", "")
        return text.strip()

    @torch.inference_mode()
    def generate(
        self,
        img: Optional[Image.Image],
        prompt: str,
        cfg: GenCfg,
    ) -> str:
        has_image = img is not None
        messages = self._build_messages(prompt, has_image=has_image)

        chat_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if has_image:
            inputs = self.processor(
                images=img,
                text=chat_text,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=chat_text,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        max_new_tokens = getattr(cfg, "max_new_tokens", 256)
        temperature = getattr(cfg, "temperature", 0.7)
        top_p = getattr(cfg, "top_p", 0.9)
        do_sample = getattr(cfg, "do_sample", False)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        full_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=False,  # 先不跳，方便抽取 assistant 段
        )

        return self._extract_assistant_text(full_text)

    @torch.inference_mode()
    def encode_modalities(
            self,
            image: Any,
            prompt: str,
            gen_cfg: GenCfg | None = None,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 1) 读图像
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type for encode_modalities: {type(image)}")

        # 2) 构造与 generate 同风格的 chat 模板
        messages = self._build_messages(prompt, has_image=True)
        chat_str = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        # 3) processor → BatchEncoding；搬到 model.device
        proc_inputs = self.processor(
            images=[img],
            text=[chat_str],
            return_tensors="pt",
        )

        device = getattr(self.model, "device", self.device)
        if hasattr(proc_inputs, "to"):
            inputs = proc_inputs.to(device)
        else:
            inputs = {k: v.to(device) for k, v in proc_inputs.items()}

        # 4) 前向：拿所有 hidden_states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states  # tuple(len=L): (1, seq_len, hidden)

        if isinstance(hidden_states, (tuple, list)):
            L = len(hidden_states)
            idx = max(0, L - 1)
            last_hidden = hidden_states[idx]  # (1, seq_len, hidden)
        else:
            last_hidden = hidden_states

        # 5) 基于 input_ids 划分 text / image token
        input_ids = inputs["input_ids"][0]  # (seq_len,)

        tokenizer = self.processor.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", None)

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            try:
                image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
            except Exception:
                image_token_id = None

        text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if pad_id is not None:
            text_mask &= (input_ids != pad_id)
        if image_token_id is not None:
            text_mask &= (input_ids != image_token_id)

        if image_token_id is not None:
            image_mask = (input_ids == image_token_id)
        else:
            image_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # 6) 计算 e_text / e_image
        if text_mask.any():
            text_tokens = last_hidden[0, text_mask, :]  # (n_text, hidden)
            e_text = text_tokens.mean(dim=0)  # (hidden,)
        else:
            e_text = last_hidden[0].mean(dim=0)

        if image_mask.any():
            image_tokens = last_hidden[0, image_mask, :]  # (n_image, hidden)
            e_image = image_tokens.mean(dim=0)  # (hidden,)
        else:
            e_image = last_hidden[0].mean(dim=0)

        return {
            "text": e_text,
            "image": e_image,
        }
