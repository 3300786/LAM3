#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从带 qwen_judge 字段的 jsonl 文件中还原各项统计指标（ASR、Refusal、Toxicity 等），
并直接打印到命令行。

用法：
    python summarize_judged_jsonl.py --in path/to/judged.jsonl
    # 可选：按 mode 和按 model 进一步拆分
    python summarize_judged_jsonl.py --in path/to/judged.jsonl --by-mode --by-model
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, list):
            if not x:
                return default
            x = x[0]
        return float(x)
    except Exception:
        return default


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * num / den


def summarize_file(
    path: Path,
    by_mode: bool = False,
    by_model: bool = False,
) -> None:
    """
    读取 jsonl 文件，假定每行形如：
    {
      "id": "...",
      "mode": "txt_only" / "img_only" / "txt_img",
      "model": "llava15_7b",
      ...
      "qwen_judge": {
        "is_refusal": true,
        "is_safe": true,
        "toxicity": 0.0,
        "is_attack_success": false,
        ...
      }
    }
    统计并打印整体 +（可选）按 mode / 按 model 的指标。
    """

    total_lines = 0
    valid_items = 0

    # 全局统计
    asr_list: List[float] = []
    refusal_list: List[float] = []
    safe_list: List[float] = []
    tox_list: List[float] = []

    # 按 mode / model 统计
    mode_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0,
        "asr": 0,
        "refusal": 0,
        "safe": 0,
        "tox_sum": 0.0,
    })
    model_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0,
        "asr": 0,
        "refusal": 0,
        "safe": 0,
        "tox_sum": 0.0,
    })

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            judge = obj.get("qwen_judge")
            if not isinstance(judge, dict):
                continue

            is_asr = bool(judge.get("is_attack_success", False))
            is_refusal = bool(judge.get("is_refusal", False))
            is_safe = bool(judge.get("is_safe", True))
            toxicity = _safe_float(judge.get("toxicity", 0.0), default=0.0)

            valid_items += 1
            asr_list.append(1.0 if is_asr else 0.0)
            refusal_list.append(1.0 if is_refusal else 0.0)
            safe_list.append(1.0 if is_safe else 0.0)
            tox_list.append(toxicity)

            mode = str(obj.get("mode", "unknown"))
            model_name = str(obj.get("model", "unknown"))

            if by_mode:
                ms = mode_stats[mode]
                ms["n"] += 1
                ms["asr"] += 1 if is_asr else 0
                ms["refusal"] += 1 if is_refusal else 0
                ms["safe"] += 1 if is_safe else 0
                ms["tox_sum"] += toxicity

            if by_model:
                gs = model_stats[model_name]
                gs["n"] += 1
                gs["asr"] += 1 if is_asr else 0
                gs["refusal"] += 1 if is_refusal else 0
                gs["safe"] += 1 if is_safe else 0
                gs["tox_sum"] += toxicity

    # ---------- 总体统计 ----------
    print("=" * 80)
    print(f"[summary] file: {path}")
    print(f"[summary] total lines         : {total_lines}")
    print(f"[summary] valid judged samples: {valid_items}")
    print("-" * 80)

    if valid_items == 0:
        print("[summary] no valid qwen_judge items found, nothing to summarize.")
        return

    asr_rate = _mean(asr_list)
    refusal_rate = _mean(refusal_list)
    safe_rate = _mean(safe_list)
    tox_mean = _mean(tox_list)

    print("Overall metrics:")
    print(f"  Attack Success Rate (ASR) : {asr_rate:.4f} ({_pct(int(asr_rate*valid_items), valid_items):.2f}%)")
    print(f"  Refusal Rate              : {refusal_rate:.4f} ({_pct(int(refusal_rate*valid_items), valid_items):.2f}%)")
    print(f"  Safe Rate                 : {safe_rate:.4f} ({_pct(int(safe_rate*valid_items), valid_items):.2f}%)")
    print(f"  Toxicity (mean, 0-100)    : {tox_mean:.2f}")
    print()

    # ---------- 按 mode 统计 ----------
    if by_mode and mode_stats:
        print("Per-mode metrics:")
        modes_sorted = sorted(mode_stats.items(), key=lambda kv: kv[0])
        print(f"{'mode':<12} {'N':>6} {'ASR%':>8} {'Refusal%':>10} {'Safe%':>8} {'Tox_mean':>10}")
        for mode, st in modes_sorted:
            n = st["n"]
            if n == 0:
                continue
            asr_p = _pct(st["asr"], n)
            ref_p = _pct(st["refusal"], n)
            safe_p = _pct(st["safe"], n)
            tox_m = st["tox_sum"] / n if n > 0 else 0.0
            print(f"{mode:<12} {n:>6d} {asr_p:>8.2f} {ref_p:>10.2f} {safe_p:>8.2f} {tox_m:>10.2f}")
        print()

    # ---------- 按 model 统计 ----------
    if by_model and model_stats:
        print("Per-model metrics:")
        models_sorted = sorted(model_stats.items(), key=lambda kv: kv[0])
        print(f"{'model':<20} {'N':>6} {'ASR%':>8} {'Refusal%':>10} {'Safe%':>8} {'Tox_mean':>10}")
        for model_name, st in models_sorted:
            n = st["n"]
            if n == 0:
                continue
            asr_p = _pct(st["asr"], n)
            ref_p = _pct(st["refusal"], n)
            safe_p = _pct(st["safe"], n)
            tox_m = st["tox_sum"] / n if n > 0 else 0.0
            print(f"{model_name:<20} {n:>6d} {asr_p:>8.2f} {ref_p:>10.2f} {safe_p:>8.2f} {tox_m:>10.2f}")
        print()

    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(
        description="Summarize ASR / Refusal / Toxicity etc. from a judged jsonl file."
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input jsonl file with qwen_judge field.",
    )
    ap.add_argument(
        "--by-mode",
        action="store_true",
        help="Also summarize metrics grouped by sample 'mode' field.",
    )
    ap.add_argument(
        "--by-model",
        action="store_true",
        help="Also summarize metrics grouped by sample 'model' field.",
    )
    args = ap.parse_args()

    path = Path(args.in_path)
    if not path.is_file():
        raise SystemExit(f"[error] input file not found: {path}")

    summarize_file(path, by_mode=args.by_mode, by_model=args.by_model)


if __name__ == "__main__":
    main()
