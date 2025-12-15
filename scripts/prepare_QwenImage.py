import os
from PIL import Image
import torch

from diffusers import QwenImageEditPipeline

# scripts/prepare_QwenImage.py

pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    force_download=True,          # 强制重新下载
    #resume_download=False,        # 不续传（避免拼接错误文件）
    #use_auth_token=False,         # 如果不需要 HF token
    #trust_remote_code=True        # Qwen 通常需要
)
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image = Image.open("data/mini_bench/girl.png").convert("RGB")
prompt = "Remove the jacket worn by the girl in the picture."
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))
