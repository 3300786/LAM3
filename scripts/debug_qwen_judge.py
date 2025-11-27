# scripts/debug_qwen_judge.py

import argparse
import json
import os
from typing import Any, Optional

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForVision2Seq

from src.metrics.eval_synergy_qwen_asr import (
    call_local_qwen_judge,
)


def load_local_qwen_judge(models_cfg: dict, judge_model_tag: str):
    """
    加载本地 Qwen 评分类模型（与 eval_synergy_qwen_asr.main 中逻辑一致，但简化）。
    """
    if models_cfg is None:
        raise RuntimeError(
            "[judge/local] models_cfg is None; cannot look up judge model. "
            "Check configs/models.yaml."
        )

    if judge_model_tag not in models_cfg:
        raise RuntimeError(
            f"[judge/local] key '{judge_model_tag}' not found in configs/models.yaml. "
            "Please add an entry like:\n"
            "  qwen25_vl_3b_judge:\n"
            "    repo_id: \"/data2/.../Qwen2.5-VL-3B-Instruct\""
        )

    judge_cfg = models_cfg[judge_model_tag]
    judge_path = judge_cfg.get("repo_id", None)
    if not judge_path:
        raise RuntimeError(
            f"[judge/local] 'repo_id' not specified for '{judge_model_tag}' "
            "in configs/models.yaml."
        )
    if not os.path.isdir(judge_path):
        print(
            f"[judge/local] WARNING: repo_id='{judge_path}' is not a directory "
            "(path check failed); will still try to load from it."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        judge_path,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForVision2Seq.from_pretrained(
        judge_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    print(
        f"[judge/local] loaded judge model '{judge_model_tag}' from {judge_path} "
        f"on {device}"
    )
    return model, tokenizer, device


def build_api_client(api_key: str, api_base: Optional[str] = None):
    """
    构造 Qwen API 的 OpenAI 兼容 client。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "[judge/api] openai package not installed. "
            "Run `pip install openai` first."
        )

    if api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
        print(f"[judge/api] Using OpenAI-compatible client with base_url={api_base}")
    else:
        client = OpenAI(api_key=api_key)
        print("[judge/api] Using OpenAI-compatible client with default base_url")

    return client


def repl_loop(
    backend: str,
    *,
    judge_model=None,
    judge_tokenizer=None,
    judge_device=None,
    api_client: Any = None,
    api_model: Optional[str] = None,
    max_new_tokens: int = 256,
):
    """
    简单命令行 REPL：
      - 粘贴 model response（多行，以单独一行 END 结束）
      - 调用 Qwen 评分类并打印 JSON
    """
    backend = backend.lower()
    print("\n================ Qwen Judge REPL ================")
    print(f"Backend     : {backend}")
    print(f"API model   : {api_model if backend == 'api' else 'N/A'}")
    print(f"Max new tok : {max_new_tokens}")
    print("Exit        : Ctrl+C / Ctrl+D / 空行回车")
    print("=================================================\n")

    while True:
        print("Paste model response below. End with a single line 'END'.")
        print("(直接回车 + END 退出)")

        resp_lines = []
        try:
            while True:
                line = input()
                if line.strip() == "END":
                    break
                resp_lines.append(line)
        except (EOFError, KeyboardInterrupt):
            print("\n[repl] exit.")
            break

        if not resp_lines:
            print("[repl] empty response, exit.")
            break

        model_output = "\n".join(resp_lines)

        print("\n[repl] Calling Qwen judge ...")
        # 这里 user_prompt 传空字符串即可，内部已经忽略
        judge = call_local_qwen_judge(
            user_prompt="",
            model_output=model_output,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            judge_device=judge_device,
            max_new_tokens=max_new_tokens,
        )

        print("\n===== Judge Result =====")
        print(json.dumps(judge, ensure_ascii=False, indent=2))
        print("========================\n")


def main():
    ap = argparse.ArgumentParser(
        description="Interactive debug tool for Qwen judge (local/API).",
    )
    ap.add_argument(
        "--judge_backend",
        choices=["local", "api"],
        default="local",
        help="backend for Qwen judge: 'local' (HF) or 'api' (OpenAI-compatible)",
    )
    ap.add_argument(
        "--judge_model_tag",
        default="qwen25_vl_3b_judge",
        help="(local backend) key in configs/models.yaml for judge model",
    )
    ap.add_argument(
        "--config_models",
        default="configs/models.yaml",
        help="path to models.yaml (for local judge)",
    )
    ap.add_argument(
        "--api_base",
        default=None,
        help="API base URL for Qwen judge (only used when --judge_backend api); "
             "if None, use default OpenAI base",
    )
    ap.add_argument(
        "--api_key",
        default=None,
        help=(
            "API key for Qwen judge (only used when --judge_backend api); "
            "if None, read from env QWEN_API_KEY or OPENAI_API_KEY"
        ),
    )
    ap.add_argument(
        "--api_model",
        default="qwen-max",
        help="API model name for Qwen judge (only used when --judge_backend api)",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="max_new_tokens for judge generation",
    )

    args = ap.parse_args()

    backend = args.judge_backend.lower()

    judge_model = judge_tokenizer = judge_device = None
    api_client = None
    api_model_name = None

    if backend == "local":
        try:
            with open(args.config_models, "r", encoding="utf-8") as f:
                models_cfg = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(
                f"[judge/local] Failed to load {args.config_models}: {e}"
            )

        judge_model, judge_tokenizer, judge_device = load_local_qwen_judge(
            models_cfg=models_cfg,
            judge_model_tag=args.judge_model_tag,
        )

    elif backend == "api":
        api_key = (
            args.api_key
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "[judge/api] API key not provided. Use --api_key or set "
                "QWEN_API_KEY / OPENAI_API_KEY."
            )
        api_client = build_api_client(api_key=api_key, api_base=args.api_base)
        api_model_name = args.api_model

    else:
        raise ValueError(f"Unknown judge_backend={backend}")

    repl_loop(
        backend=backend,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        judge_device=judge_device,
        api_client=api_client,
        api_model=api_model_name,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
