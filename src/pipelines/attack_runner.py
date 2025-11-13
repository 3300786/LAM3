# src/pipelines/attack_runner.py
import os
import json
from typing import List, Dict, Any

from src.models.registry import build_model
from src.attacks.base import Attack
from src.utils.runtime import GenCfg, load_yaml, set_seed


def run_attack_on_list(
    model_name: str,
    attack: Attack,
    samples: List[Dict[str, Any]],
    exp_name: str,
    out_path: str = "outputs/logs",
):
    """
    samples: 一个最简单的列表，每个元素包含:
        {
          "img": "data/mini_bench/pic5.png",
          "prompt": "base prompt ...",
          "id": "sample-1"
        }
    后续可以替换为更规范的 Sample dataclass。
    """
    os.makedirs(out_path, exist_ok=True)

    models_cfg = load_yaml("configs/models.yaml")
    runtime = load_yaml("configs/runtime.yaml")
    set_seed(runtime.get("seed", 42))

    model = build_model(model_name, models_cfg, runtime)

    gen = GenCfg(
        max_new_tokens=runtime.get("max_new_tokens", 64),
        do_sample=runtime.get("do_sample", False),
        temperature=runtime.get("temperature", 0.0),
        top_p=runtime.get("top_p", 1.0),
    )

    logs = []
    for s in samples:
        res = attack.run(
            model=model,
            image=s["img"],
            base_prompt=s["prompt"],
            gen_cfg=gen,
            meta={"id": s.get("id")},
        )
        logs.append(
            {
                "model": model_name,
                "sample_id": s.get("id"),
                "img": s["img"],
                "base_prompt": s["prompt"],
                "adv_prompt": res["adv_prompt"],
                "output": res["output"],
                "success": res["success"],
                "extra": res.get("extra", {}),
            }
        )

    out_file = os.path.join(out_path, f"{exp_name}.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in logs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return out_file
