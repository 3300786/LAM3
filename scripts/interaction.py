#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal interactive CLI (Qwen2.5-VL-7B-Instruct) with paired inference:
- Run TEXT-ONLY and TEXT+IMAGE for each user turn
- User labels each output as: safe / unsafe / invalid
- After each turn, print label ratios (overall + per mode)

Usage:
  python -m scripts.paired_qwen25_vl_cli \
    --model_path /models/Qwen2.5-VL-7B-Instruct \
    --default_image /path/to/default.png \
    --log_jsonl outputs/paired_eval.jsonl

Commands:
  /exit, /quit  exit
  /clear        reset counters & optional session history (this script is stateless for model context)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


LABELS = ("safe", "unsafe", "invalid")
MODES = ("text_only", "text_img")


# ===================== Args =====================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Paired inference CLI for Qwen2.5-VL (text-only vs text+image)")
    ap.add_argument(
        "--model_path",
        type=str,
        default="/data2/wangjingsong/models/Qwen2.5-VL-7B-Instruct",
        help="Local path to Qwen2.5-VL-7B-Instruct",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device: auto / cuda / cpu",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature; 0 => deterministic (no sampling)",
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
        help="System prompt used in chat template",
    )
    ap.add_argument(
        "--default_image",
        type=str,
        default="/home/wangjingsong/workspace/LAM3/data/synergy_jbv28k/null_image.png",
        help="Default image path used when user input is empty. If empty, TEXT+IMAGE will be skipped unless user provides a path.",
    )
    ap.add_argument(
        "--log_jsonl",
        type=str,
        default="",
        help="Optional path to write per-turn records as JSONL",
    )
    ap.add_argument(
        "--print_inputs",
        action="store_true",
        help="Print resolved image uri and prompt each turn",
    )
    return ap


# ===================== Model loading =====================

def load_qwen2_5_vl(model_path: str, device_str: str = "auto"):
    model_path = str(Path(model_path).expanduser())
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"model_path '{model_path}' is not a directory.")

    if device_str == "cpu":
        device = "cpu"
        device_map = {"": "cpu"}
    else:
        if torch.cuda.is_available():
            device = "cuda"
            device_map = "auto" if device_str in ("auto", "cuda") else {"": "cuda"}
        else:
            device = "cpu"
            device_map = {"": "cpu"}
            print("[warn] CUDA not available; using CPU (slow).")

    print(f"[load] loading Qwen2.5-VL from {model_path} on {device} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("[load] model & processor ready.")
    return model, processor, device


# ===================== Helpers =====================

def _path_to_file_uri(p: str) -> Optional[str]:
    if not p:
        return None
    path = Path(p).expanduser().resolve()
    if not path.is_file():
        return None
    return "file://" + str(path)


def _build_conversation(system_prompt: str, text_query: str, image_uri: Optional[str]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    if image_uri:
        content = [{"type": "image", "image": image_uri}, {"type": "text", "text": text_query}]
    else:
        content = [{"type": "text", "text": text_query}]
    msgs.append({"role": "user", "content": content})
    return msgs


@torch.inference_mode()
def generate_once(
    model,
    processor,
    device: str,
    conversation: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    # process_vision_info expects a batch of conversations
    messages_batch = [conversation]
    image_inputs, video_inputs = process_vision_info(messages_batch)

    inputs = processor(
        text=[text],
        images=image_inputs,   # can be None
        videos=video_inputs,   # can be None
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    generated_ids = model.generate(**inputs, **gen_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return out


def _read_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "/exit"


def _read_multiline(prompt: str) -> str:
    print(prompt)
    print("  (finish with a single line: END)")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END" or line.strip() == "###":
            break
        lines.append(line)
    print("Input Over. Inferring ...")
    return "\n".join(lines).strip()


def _normalize_label(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    if s in ("s", "safe"):
        return "safe"
    if s in ("u", "unsafe"):
        return "unsafe"
    if s in ("i", "invalid"):
        return "invalid"
    return None


def _ratios(counts: Dict[str, int]) -> Dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {k: 0.0 for k in LABELS}
    return {k: counts.get(k, 0) / total for k in LABELS}


def _print_stats(overall: Dict[str, int], per_mode: Dict[str, Dict[str, int]]) -> None:
    o = _ratios(overall)
    print("[stats] overall:")
    print(f"  total={sum(overall.values())}  safe={o['safe']:.3f}  unsafe={o['unsafe']:.3f}  invalid={o['invalid']:.3f}")

    for mode in MODES:
        r = _ratios(per_mode[mode])
        print(f"[stats] {mode}:")
        print(f"  total={sum(per_mode[mode].values())}  safe={r['safe']:.3f}  unsafe={r['unsafe']:.3f}  invalid={r['invalid']:.3f}")


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ===================== Main loop =====================

def main():
    args = build_parser().parse_args()

    model, processor, device = load_qwen2_5_vl(args.model_path, args.device)

    default_image_uri = _path_to_file_uri(args.default_image) if args.default_image else None
    if args.default_image and not default_image_uri:
        print(f"[warn] --default_image not found: {args.default_image} (TEXT+IMAGE may be skipped unless user provides a valid path)")

    overall_counts = {k: 0 for k in LABELS}
    mode_counts = {m: {k: 0 for k in LABELS} for m in MODES}
    turn_id = 0

    print("\n[interactive] Paired inference started.")
    print("Flow per turn:")
    print("  1) input image path (empty => default path)")
    print("  2) input text prompt (multiline; end with END)")
    print("  3) run TEXT-ONLY and TEXT+IMAGE inference")
    print("  4) label each output: safe/unsafe/invalid (or s/u/i)")
    print("Commands: /exit /quit, /clear\n")

    while True:
        img_inp = _read_line("Image path (empty => default): ").strip()
        # img_inp = ""
        low = img_inp.lower()

        if low in ("/exit", "exit", "/quit", "quit"):
            print("[interactive] Bye.")
            break

        if low in ("/clear", "clear"):
            overall_counts = {k: 0 for k in LABELS}
            mode_counts = {m: {k: 0 for k in LABELS} for m in MODES}
            turn_id = 0
            print("[interactive] counters cleared.\n")
            continue

        # resolve image uri
        image_uri = None
        if img_inp and img_inp != "":
            image_uri = _path_to_file_uri(img_inp)
            if not image_uri:
                print(f"[warn] image path not found: {Path(img_inp).expanduser().resolve()}")
        else:
            image_uri = default_image_uri

        prompt = _read_multiline("Prompt:")
        if not prompt:
            print("[warn] empty prompt, skip.\n")
            continue

        if args.print_inputs:
            print(f"[debug] image_uri={image_uri}")
            print(f"[debug] prompt=\n{prompt}\n")

        turn_id += 1
        ts = int(time.time())

        # ---- Inference: text-only
        text_only_conv = _build_conversation(args.system_prompt, prompt, image_uri=None)
        try:
            out_text_only = generate_once(
                model=model,
                processor=processor,
                device=device,
                conversation=text_only_conv,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            out_text_only = f"[error] generation failed: {e}"

        # ---- Inference: text+image (if image exists)
        out_text_img = ""
        ran_text_img = False
        if image_uri:
            text_img_conv = _build_conversation(args.system_prompt, prompt, image_uri=image_uri)
            try:
                out_text_img = generate_once(
                    model=model,
                    processor=processor,
                    device=device,
                    conversation=text_img_conv,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                ran_text_img = True
            except Exception as e:
                out_text_img = f"[error] generation failed: {e}"
                ran_text_img = True
        else:
            out_text_img = "[skip] no image provided and no default image set."

        # ---- Display outputs
        print("\n" + "=" * 80)
        print(f"[turn {turn_id}] TEXT-ONLY output:")
        print(out_text_only if out_text_only else "(empty)")
        print("-" * 80)
        print(f"[turn {turn_id}] TEXT+IMAGE output:")
        print(out_text_img if out_text_img else "(empty)")
        print("=" * 80 + "\n")

        # ---- Labeling (per mode)
        # text-only label
        while True:
            lab = _normalize_label(_read_line("Label TEXT-ONLY (safe/unsafe/invalid | s/u/i): "))
            if lab:
                break
            print("[warn] invalid label. Use: safe/unsafe/invalid or s/u/i.")
        mode_counts["text_only"][lab] += 1
        overall_counts[lab] += 1
        label_text_only = lab

        # text+img label (only if we ran it; skip only occurs when no image)
        label_text_img = None
        if ran_text_img:
            while True:
                lab2 = _normalize_label(_read_line("Label TEXT+IMAGE (safe/unsafe/invalid | s/u/i): "))
                if lab2:
                    break
                print("[warn] invalid label. Use: safe/unsafe/invalid or s/u/i.")
            mode_counts["text_img"][lab2] += 1
            overall_counts[lab2] += 1
            label_text_img = lab2
        else:
            # did not run text_img; do not count
            pass

        # ---- Print ratios
        print()
        _print_stats(overall_counts, mode_counts)
        print()

        # ---- Optional logging
        rec = {
            "ts": ts,
            "turn_id": turn_id,
            "image_path_input": img_inp,
            "image_uri": image_uri,
            "prompt": prompt,
            "outputs": {
                "text_only": out_text_only,
                "text_img": out_text_img,
            },
            "labels": {
                "text_only": label_text_only,
                "text_img": label_text_img,
            },
            "counts": {
                "overall": dict(overall_counts),
                "text_only": dict(mode_counts["text_only"]),
                "text_img": dict(mode_counts["text_img"]),
            },
        }
        _append_jsonl(args.log_jsonl, rec)


if __name__ == "__main__":
    main()
