# src/models/clip_cross.py

from typing import Any, Dict, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.models.base import MLLM
from src.utils.runtime import GenCfg


class ClipCrossEncoderWrapper(MLLM):
    """
    仅用于计算跨模态表征的 CLIP 封装：
      - encode_modalities(image, prompt) -> {"text": e_t, "image": e_i}
    不支持 generate()（如果被调用就直接报错）。
    """

    def __init__(self, repo_dir: str, runtime_cfg: Optional[Dict[str, Any]] = None):
        runtime_cfg = runtime_cfg or {}
        device_str = runtime_cfg.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = torch.device(device_str)

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # 加载 CLIP 模型和处理器
        self.model = CLIPModel.from_pretrained(
            repo_dir,
            torch_dtype=torch_dtype,
        ).to(self.device).eval()

        self.processor = CLIPProcessor.from_pretrained(repo_dir)

    @torch.inference_mode()
    def encode_modalities(
            self,
            image,
            prompt: str,
            gen_cfg=None,
            **kwargs,
    ):
        """
        使用 CLIP 作为跨模态编码器：
          - text: 使用 text encoder 得到 text_embeds 或倒数第 L 层平均
          - image: 使用 vision encoder 得到 image_embeds
        返回:
          {
            "text":  (hidden_dim,),
            "image": (hidden_dim,),
          }
        注意：这里不做归一化，归一化交给上层 D(x) 逻辑。
        """
        # 1) 读图像
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 2) 确定 CLIP 文本最大长度（一般是 77）
        #   - 优先从 text_config 取 max_position_embeddings
        #   - fallback 到 77，避免意外
        text_cfg = getattr(self.model, "text_model", None)
        max_txt_len = 77
        try:
            if text_cfg is not None and hasattr(text_cfg.config, "max_position_embeddings"):
                max_txt_len = int(text_cfg.config.max_position_embeddings)
            elif hasattr(self.model.config, "text_config") and hasattr(
                    self.model.config.text_config, "max_position_embeddings"
            ):
                max_txt_len = int(self.model.config.text_config.max_position_embeddings)
        except Exception:
            max_txt_len = 77

        # 3) processor 编码，并显式开启 truncation + max_length
        enc = self.processor(
            text=[prompt],
            images=[img],
            return_tensors="pt",
            padding="max_length",  # 统一到 max_txt_len
            truncation=True,
            max_length=max_txt_len,
        )

        # 4) 手动再裁一遍，确保不会超过 max_txt_len
        input_ids = enc["input_ids"][:, :max_txt_len].to(self.device, non_blocking=True)
        attn_mask = enc["attention_mask"][:, :max_txt_len].to(self.device, non_blocking=True)
        pixel_values = enc["pixel_values"].to(self.device, non_blocking=True)

        # 5) 前向：直接用 CLIPModel
        #    - 一般 CLIPModel.forward 会返回 text_embeds / image_embeds
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )

        # 6) 取 text / image 的全局嵌入
        #    - 标准 CLIPModel 有 text_embeds / image_embeds
        if hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
            e_text = outputs.text_embeds[0]  # (D,)
        else:
            # 回退：用最后一层 hidden_state 的 [EOS] 或平均
            text_hidden = outputs.text_model_output.last_hidden_state  # (1, L, D)
            # 取 mask 有效位置的平均
            mask = attn_mask[0].bool()
            e_text = text_hidden[0, mask, :].mean(dim=0)

        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            e_image = outputs.image_embeds[0]  # (D,)
        else:
            # 回退：用 vision encoder 的 CLS token 或平均
            vision_hidden = outputs.vision_model_output.last_hidden_state  # (1, N, D)
            e_image = vision_hidden[0].mean(dim=0)

        return {
            "text": e_text,  # 不要在这里做 F.normalize，外层已经做了
            "image": e_image,
        }

    @torch.inference_mode()
    def generate(
        self,
        img: Optional[Image.Image],
        prompt: str,
        cfg: GenCfg,
    ) -> str:
        """
        这里只是满足 MLLM 接口，不打算用 CLIP 做生成。
        调用时直接报错，避免误用。
        """
        raise NotImplementedError(
            "ClipCrossEncoderWrapper does not support text generation; "
            "it is only used for encode_modalities()."
        )
