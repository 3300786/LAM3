from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch, os
from src.utils.runtime import device_kwargs, GenCfg

class Llava15Wrapper:
    def __init__(self, repo_id:str, runtime_cfg:dict):
        self.processor = AutoProcessor.from_pretrained(repo_id, use_fast=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(repo_id, **device_kwargs(runtime_cfg))
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_path:str, prompt:str, gen:GenCfg):
        image = Image.open(image_path).convert("RGB")
        # LLaVA-HF 期望对话式提示，最小可行模板：
        conv = f"USER: {prompt}\nASSISTANT:"
        inputs = self.processor(text=conv, images=image, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        # 去掉历史 prompt 前缀，保留助手段落
        return text.split("ASSISTANT:")[-1].strip()
