from dataclasses import dataclass
from typing import Any
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from src.utils.runtime import GenCfg


class Llava15Wrapper:
    def __init__(self, repo_dir: str, runtime_cfg=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型
        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo_dir,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(self.device).eval()

        # 处理器：负责 chat 模板、图像预处理、占位对齐
        self.processor = AutoProcessor.from_pretrained(
            repo_dir,
            local_files_only=True,
        )

    @torch.inference_mode()
    def generate(self, image_path: str, prompt: str, gen: GenCfg) -> str:
        image = Image.open(image_path).convert("RGB")

        # 加系统约束，抑制臆测
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text":
                        ("You are a vision-language assistant. "
                         "ONLY describe visible content from the image. "
                         "Read visible text exactly. "
                         "If uncertain, say 'uncertain'. "
                         "Do not infer locations, events, teams, or attire beyond what is visible.")}
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

        # 生成配置：评测期用确定性解码
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=int(gen.max_new_tokens),
            do_sample=False,                     # 关键：关闭采样
            temperature=1.0, top_p=1.0,          # 无效占位，保持接口一致
            repetition_penalty=1.0,
        )

        # 补全停码（eos + 可能的 eot）
        tok = self.processor.tokenizer
        eos_list = []
        if getattr(tok, "eos_token_id", None) is not None:
            eos_list.append(tok.eos_token_id)
        # LLaVA 部分权重含有对话结束符，名称可能不同，尝试安全转换
        for maybe in ("<|eot_id|>", "<|end_of_text|>", "</s>"):
            try:
                tid = tok.convert_tokens_to_ids(maybe)
                if isinstance(tid, int) and tid >= 0:
                    eos_list.append(tid)
            except Exception:
                pass
        if eos_list:
            # transformers 支持 list[int]
            gen_kwargs["eos_token_id"] = list(dict.fromkeys(eos_list))

        prompt_len = enc["input_ids"].shape[-1]
        out = self.model.generate(**enc, **gen_kwargs)
        gen_ids = out[:, prompt_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # 轻量后处理：去模板残留，阻断“新一轮”
        if text.startswith("Assistant:"):
            text = text[len("Assistant:"):].lstrip()

        import re
        # 若模型续写出下一轮提示或再次出现特殊停符，做一次安全截断
        text = re.split(r"(?:\nUser:|<\|eot_id\|>|</s>|<\|end_of_text\|>)", text)[0].strip()

        # 可选：限制输出两段，避免跑题延展
        parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(parts) > 2:
            text = "\n\n".join(parts[:2])

        return text

