from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from PIL import Image
import torch
from src.utils.runtime import device_kwargs, GenCfg

class Idefics2Wrapper:
    def __init__(self, repo_id:str, runtime_cfg:dict):
        self.processor = AutoProcessor.from_pretrained(repo_id, use_fast=True)
        self.model = Idefics2ForConditionalGeneration.from_pretrained(repo_id, **device_kwargs(runtime_cfg))
        self.model.eval()

    @torch.inference_mode()
    def generate(self, image_path:str, prompt:str, gen:GenCfg):
        image = Image.open(image_path).convert("RGB")
        # Idefics2 输入为多模态序列：列表形式的对话 turn
        msgs = [[{"type":"image"}, {"type":"text","text":prompt}]]
        inputs = self.processor(msgs, [image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            top_p=gen.top_p
        )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        return text.strip()
