# src/data/datasets.py
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Sample:
    sample_id: str
    image_path: Optional[str]
    base_prompt: str              # 原始安全 prompt 或用户 query
    target_behavior: str          # 我们希望诱导的“违规行为”描述，用于ASR判定
    meta: Dict = None             # 额外信息，如场景标签、任务类型等
