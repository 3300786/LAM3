import os

import torch
from diffusers import ZImagePipeline

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

# prompt = "一张中景手机自拍照片拍摄了一位留着长黑发的年轻东亚女子在灯光明亮的浴室内对着镜子自拍。她的头微微倾斜，嘴唇嘟起做亲吻状，非常可爱俏皮。她右手拿着一部深灰色智能手机，遮住了部分脸，后置摄像头镜头对着镜子。"
import os

while True:
    print("input prompt (end with ###, type exit to quit):")
    lines = []
    while True:
        line = input()
        if line.strip() == "exit":
            exit(0)
        if line.strip() == "###":
            break
        lines.append(line)

    prompt = "\n".join(lines)

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    x = 1
    os.makedirs("outputs/Zimage", exist_ok=True)
    while True:
        path = f"outputs/Zimage/{x}.png"
        if not os.path.exists(path):
            image.save(path)
            print(f"Saved to {path}")
            break
        x += 1

