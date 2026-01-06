#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态交互式终端对话脚本（Qwen2.5-VL-7B-Instruct）

用法示例：
  python -m scripts.interactive_qwen2_5_vl_cli \
    --model_path /models/Qwen2.5-VL-7B-Instruct

输入格式：
  1) 纯文本：
       User: 你好，简单自我介绍一下。
  2) 图像 + 文本：
       User: img=/path/to/image.png 这张图里发生了什么？
     若只给图不写问题：
       User: img=/path/to/image.png
     会自动使用默认问题："Describe this image."

特殊命令：
  /exit 或 /quit 结束对话
  /clear 清空对话历史
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


# ===================== 参数解析 =====================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Interactive CLI chat with Qwen2.5-VL (multimodal)")
    ap.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen2.5-VL-7B-Instruct",
        help="Local path to Qwen2.5-VL-7B-Instruct, e.g. /models/Qwen2.5-VL-7B-Instruct",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use: auto / cuda / cpu",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature; 0 表示基本确定性（会自动关闭采样）",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p for sampling",
    )
    ap.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt passed into chat template.",
    )
    return ap


# ===================== 模型加载 =====================

def load_qwen2_5_vl(
    model_path: str,
    device_str: str = "auto",
):
    model_path = str(Path(model_path).expanduser())
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path '{model_path}' is not a directory.")

    # 设备选择
    if device_str == "cpu":
        device = "cpu"
        device_map = {"": "cpu"}
    else:
        if torch.cuda.is_available():
            device = "cuda"
            # 多 GPU 时直接交给 transformers 的 device_map="auto"
            device_map = "auto" if device_str in ("auto", "cuda") else {"": "cuda"}
        else:
            device = "cpu"
            device_map = {"": "cpu"}
            print("[warn] CUDA is not available, falling back to CPU (very slow).")

    print(f"[load] loading Qwen2.5-VL from {model_path} on {device} ...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print("[load] model & processor ready.")
    return model, processor, device


# ===================== 输入解析 =====================

def parse_user_input_for_images(user_inp: str) -> tuple[Optional[List[str]], str]:
    """
    支持多行输入：
    - 第一行如果以 img= 或 image= 开头，则解析图像路径
    - 剩余所有内容（包括第一行后面的部分 + 后续行）作为 text_query
    - 如果没有图像，则整个输入作为纯文本
    """
    user_inp = user_inp.strip()
    if not user_inp:
        return None, ""

    lines = user_inp.splitlines()
    first_line = lines[0].strip()

    # 只检查第一行是否匹配 img= 或 image=
    m = re.match(r"^(img|image)\s*=\s*(.+?)\s*(?:\s+(.+))?$", first_line, flags=re.IGNORECASE)
    if not m:
        # 第一行不是 img= 格式 → 纯文本
        return None, user_inp

    img_part = m.group(2).strip()        # 路径部分（可能是逗号分隔）
    first_line_remainder = m.group(3) or ""  # 第一行路径后面的文本（可能有）

    # 提取图像路径（支持逗号分隔多张图）
    paths = [p.strip() for p in img_part.split(",") if p.strip()]
    uris: List[str] = []

    for p in paths:
        # 支持相对路径、~、绝对路径
        path = Path(p).expanduser().resolve()
        if not path.is_file():
            print(f"[warn] image path not found: {path}")
            continue
        uris.append("file://" + str(path))

    if not uris:
        print("[warn] no valid images loaded, fallback to text-only.")
        return None, user_inp

    # 构建完整的查询文本：
    # - 第一行路径后面的部分（如果有）
    # - 加上后续所有行
    text_parts = []
    if first_line_remainder:
        text_parts.append(first_line_remainder)
    if len(lines) > 1:
        text_parts.extend(lines[1:])

    text_query = "\n".join(text_parts).strip()

    # 如果用户只给了图片路径，没写问题 → 默认描述
    if not text_query:
        text_query = "Describe this image."

    return uris, text_query


# ===================== 单轮生成 =====================

def generate_from_conversation(
    model,
    processor,
    device: str,
    conversation: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    conversation: HF chat template 的 message 列表：
      [{"role": "...", "content": ...}, ...]
    """
    # 1) chat template
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) 视觉信息（images/videos）解析
    #    注意：process_vision_info 接收的是 batch of conversations
    messages_batch = [conversation]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    # 3) 编码为张量
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    # 4) 生成
    do_sample = temperature > 0.0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # 只保留新生成部分（去掉 prompt）
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_texts[0].strip()


# ===================== 主逻辑：交互循环 =====================

def main():
    parser = build_parser()
    args = parser.parse_args()

    model, processor, device = load_qwen2_5_vl(
        model_path=args.model_path,
        device_str=args.device,
    )

    # 对话历史（符合 Qwen chat template）
    # 结构：List[{"role": "...", "content": ...}]
    history_msgs: List[Dict[str, Any]] = []
    if args.system_prompt:
        history_msgs.append({"role": "system", "content": args.system_prompt})

    print("\n[interactive] Qwen2.5-VL multimodal chat started.")
    print("说明：")
    print("  - 纯文本：直接输入问题并回车。")
    print("  - 图像 + 文本：")
    print("      img=/path/to/image.png 这张图里发生了什么？")
    print("    多图：")
    print("      img=1.png,2.png 帮我比较两张图的区别。")
    print("  - 指令：")
    print("      /exit 或 /quit   结束对话")
    print("      /clear           清空对话历史\n")

    try:
        while True:
            try:
                def read_multiline_input() -> str:
                    print("User: (输入完后单独输入 END 提交)")
                    lines = []
                    while True:
                        try:
                            line = input()
                        except EOFError:
                            return "\n".join(lines)
                        if line.strip() == "END":
                            break
                        lines.append(line)
                    return "\n".join(lines)
                user_inp = read_multiline_input()
                user_inp = user_inp.strip()
                if not user_inp:
                    continue
            except EOFError:
                print("\n[interactive] EOF received, exiting.")
                break

            if not user_inp:
                continue

            low = user_inp.lower()
            if low in {"exit", "/exit", "quit", "/quit"}:
                print("[interactive] Bye.")
                break
            if low in {"/clear", "clear"}:
                history_msgs = []
                if args.system_prompt:
                    history_msgs.append({"role": "system", "content": args.system_prompt})
                print("[interactive] history cleared.\n")
                continue

            # 解析图像前缀
            image_uris, text_query = parse_user_input_for_images(user_inp)
            print("image:", image_uris)
            print("text:", text_query)
            if not text_query:
                text_query = "Describe this image." if image_uris else user_inp

            # 构造当前用户消息
            content: Any
            if image_uris:
                chunks = [{"type": "image", "image": uri} for uri in image_uris]
                chunks.append({"type": "text", "text": text_query})
                content = chunks
            else:
                # 纯文本也使用 chunk 形式，便于统一处理
                content = [{"type": "text", "text": text_query}]

            user_msg = {
                "role": "user",
                "content": content,
            }

            # 当前轮完整对话（历史 + 新的 user）
            conversation = history_msgs + [user_msg]

            try:
                answer = generate_from_conversation(
                    model=model,
                    processor=processor,
                    device=device,
                    conversation=conversation,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                print(f"[error] generation failed: {e}\n")
                continue

            # 打印回复
            print(f"Assistant: {answer}\n")

            # 更新历史
            history_msgs.append(user_msg)
            history_msgs.append({"role": "assistant", "content": answer})

    except KeyboardInterrupt:
        print("\n[interactive] KeyboardInterrupt, exiting.")


if __name__ == "__main__":
    main()
