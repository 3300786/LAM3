#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Optional, List, Dict, Any
from transformers import LlavaForConditionalGeneration


# ==========================
# Victim MLLM 封装（保持不变）
# ==========================

class VictimClient:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[victim] loading LLaVA from {model_name_or_path} on {self.device} ...")

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            use_fast=True,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

        print("[victim] loaded.")

    def _build_chat_prompt(
        self,
        user_text: str,
        with_image: bool,
    ) -> str:
        """
        对 LLaVA 来说，正确的做法是：
        - 使用 processor.apply_chat_template
        - 如果有图像，conv 里加入一个 {"type": "image"} 的 content
        """

        if with_image:
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": user_text,
                }
            ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # 让模型知道要开始生成
        )
        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: int = 256,
    ) -> str:

        chat_prompt = self._build_chat_prompt(prompt, with_image=(image is not None))

        if image is None:
            inputs = self.processor(
                text=chat_prompt,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.processor(
                text=[chat_prompt],
                images=[image],
                return_tensors="pt",
            ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        # LLaVA 用的是 tokenizer
        text = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )
        return text.strip()

# ==========================
# Z-Image-Turbo 封装（改为支持 CPU + lazy init）
# ==========================

class ZImageClient:
    def __init__(self, device: str = "cpu", out_dir: str = "outputs/Zimage"):
        """
        默认放在 CPU 上运行，避免和 victim 抢显存。
        device 可以是 'cpu' 或 'cuda'；如果你真的想用 GPU，可以显式传 --zimage_device cuda，
        但这时要注意显存。
        """
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[z-image] loading Z-Image-Turbo on {self.device} ...")
        from diffusers import ZImagePipeline

        self.pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=False,
        ).to(self.device)

        # 如果你显存非常紧张，可以在这里启用 CPU offload:
        # if self.device.type == "cuda":
        #     self.pipe.enable_model_cpu_offload()

        print("[z-image] loaded.")

    @torch.no_grad()
    def generate_image(self, prompt: str, seed: int = 42,
                       height: int = 1024, width: int = 1024,
                       steps: int = 9) -> tuple[Image.Image, Path]:

        # 注意：这里 generator 的 device 要和 pipe 的 device 一致
        gen_device = self.device.type
        generator = torch.Generator(device=gen_device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=0.0,  # Turbo 模型要求 0
            generator=generator,
        ).images[0]

        filename = f"zimage_seed{seed}_h{height}_w{width}.png"
        out_path = self.out_dir / filename
        image.save(out_path)
        print(f"[z-image] saved to: {out_path}")

        return image, out_path


# ==========================
# 交互式攻击小流水线（ZImage lazy 初始化）
# ==========================

def interactive_loop(
    victim: VictimClient,
    zimage_device: str = "cpu",
    zimage_out_dir: str = "outputs/Zimage",
):
    """
    - victim 始终保持在 GPU（或你指定的 device）上
    - ZImageClient 在第一次需要图像攻击时才初始化，默认在 CPU 上
    """
    print("\n========== LAM3 / Z-Image Turbo Attack Test ==========")
    print("指令说明：")
    print("  - 在任何输入处键入 'quit' 退出整个程序")
    print("  - 在图像生成阶段输入 'done' 结束当前攻击的图像尝试，回到新的攻击轮次\n")

    zimg: Optional[ZImageClient] = None  # lazy init

    attack_idx = 0
    while True:
        attack_idx += 1
        print(f"\n===== Attack #{attack_idx} =====")

        # ---------- 1) 纯文本越狱尝试 ----------
        text_prompt = input("[attack] 输入纯文本 prompt（或 'quit' 退出）：\n> ").strip()
        if text_prompt.lower() == "quit":
            print("[exit] 用户退出。")
            break

        print("[victim] 纯文本模式推理中...")
        text_only_resp = victim.generate(text_prompt, image=None)
        print("\n[victim][text-only] 回复：")
        print("-" * 60)
        print(text_only_resp)
        print("-" * 60)

        # ---------- 2) 是否进行图像+文本攻击 ----------
        use_image = input("\n是否继续尝试 图像+文本 攻击？(y/n, 'quit' 退出)： ").strip().lower()
        if use_image == "quit":
            print("[exit] 用户退出。")
            break
        if use_image not in ("y", "yes"):
            continue  # 下一次攻击轮次

        # ---------- 3) lazily 初始化 Z-Image-Turbo ----------
        if zimg is None:
            zimg = ZImageClient(device=zimage_device, out_dir=zimage_out_dir)

        # ---------- 4) 在当前攻击下多次生成图像并测试 ----------
        img_try_idx = 0
        while True:
            img_prompt = input(
                "\n[img] 输入文生图 prompt（'done' 结束本轮图像攻击，'quit' 退出程序）：\n> "
            ).strip()
            if img_prompt.lower() == "quit":
                print("[exit] 用户退出。")
                return
            if img_prompt.lower() == "done":
                print("[attack] 本轮图像攻击结束，进入下一次攻击。")
                break

            img_try_idx += 1

            try:
                print(f"[z-image] 正在生成图像 #{img_try_idx} ...")
                image, img_path = zimg.generate_image(
                    prompt=img_prompt,
                    seed=42 + img_try_idx,
                    height=1024,
                    width=1024,
                    steps=9,
                )

                print(f"[victim] 图文模式推理中（图像：{img_path}）...")
                resp = victim.generate(text_prompt, image=image)
                print("\n[victim][image+text] 回复：")
                print("-" * 60)
                print(resp)
                print("-" * 60)
            except Exception as e:
                print(f"[error] 图像流水线或 victim 推理出错：{e}")
                continue


# ==========================
# main
# ==========================

def main():
    parser = argparse.ArgumentParser(description="LAM3 Z-Image Turbo + Victim MLLM test pipeline")
    parser.add_argument(
        "--victim_model",
        type=str,
        required=True,
        help="Victim MLLM 的 HuggingFace 名称或本地路径，例如 'liuhaotian/llava-v1.5-7b' 或 './checkpoints/llava15'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="victim 模型运行设备，例如 'cuda' 或 'cpu'",
    )
    parser.add_argument(
        "--zimage_device",
        type=str,
        default="cpu",
        help="Z-Image 的运行设备，默认 'cpu'，避免和 victim 抢显存。也可显式设为 'cuda'，但要自己保证显存。",
    )
    parser.add_argument(
        "--zimage_out",
        type=str,
        default="outputs/Zimage",
        help="Z-Image 生成图片的保存目录",
    )

    args = parser.parse_args()

    # 只初始化 victim（在 GPU）
    victim = VictimClient(args.victim_model, device=args.device)

    # ZImageClient 不在这里初始化，而是在 interactive_loop 里按需 lazy init
    interactive_loop(
        victim=victim,
        zimage_device=args.zimage_device,
        zimage_out_dir=args.zimage_out,
    )


if __name__ == "__main__":
    main()
