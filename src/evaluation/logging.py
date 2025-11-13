# src/evaluation/logging.py
import os, json
from typing import List, Dict

def save_attack_logs(
    logs: List[Dict],
    exp_name: str,
    out_dir: str
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{exp_name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for row in logs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
