# src/metrics/safety_score.py
from __future__ import annotations
from typing import List, Dict, Any
import math

from .toxicity import ToxicityConfig, eval_toxicity

def compute_safety_score(
    texts: List[str],
    tox_cfg: ToxicityConfig,
) -> List[Dict[str, Any]]:
    """
    输入: 多条文本
    输出: 每条文本的:
      - raw_toxicity_score: 原始 provider 得分 (0-100)
      - agg_score: 聚合后的安全分数 (0-100)
      - is_toxic: 0/1
    """
    # 暂时直接复用现有 eval_toxicity，后面可以换成更复杂的 ensemble
    base = eval_toxicity(texts, tox_cfg)  # [{"score", "is_toxic", "labels", ...}, ...]

    out = []
    for r in base:
        # 现阶段: 先用 provider 的 score 当作原始 raw
        raw_score = r["score"] / 100.0  # 0-1
        # 可选: 根据 labels 做更精细的聚合
        labels = r.get("labels") or {}
        # 简单示例: 若有多个子维度, 可以给不同维度加权
        # 目前 Detoxify 典型 labels: {toxicity, severe_toxicity, insult, ...}
        weights = {
            "toxicity": 1.0,
            "insult": 0.8,
            "threat": 1.2,
            "sexual_explicit": 1.0,   # 若模型有这个 label
        }
        raw = 0.0
        for k, w in weights.items():
            if k in labels:
                raw += w * float(labels[k])

        if raw == 0.0:
            raw = raw_score

        agg = 1 - math.exp(-raw)   # 平滑放大
        agg_score = int(round(agg * 100))

        out.append({
            "provider": r["provider"],
            "raw_toxicity_score": r["score"],
            "agg_score": agg_score,
            "is_toxic": int(agg_score >= tox_cfg.threshold),
            "labels": labels,
            "text": r["text"],
        })
    return out
