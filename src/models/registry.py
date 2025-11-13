# src/models/registry.py
from typing import Dict
from src.models.llava15 import Llava15Wrapper
from src.models.idefics2 import Idefics2Wrapper
from src.models.base import MLLM

WRAPPER_CLS: Dict[str, type] = {
    "llava15_7b": Llava15Wrapper,
    "idefics2_8b": Idefics2Wrapper,
}


def build_model(name: str, models_cfg: dict, runtime_cfg: dict) -> MLLM:
    """
    name: "llava15_7b" / "idefics2_8b"
    models_cfg: configs/models.yaml 读取出来的 dict
    runtime_cfg: configs/runtime.yaml 读取出来的 dict
    """
    if name not in WRAPPER_CLS:
        raise ValueError(f"Unknown model name: {name}")

    wrapper_cls = WRAPPER_CLS[name]
    model_cfg = models_cfg.get(name, {})

    # 假设 models.yaml 至少包含 repo_id 字段
    repo_id = model_cfg.get("repo_id")
    if repo_id is None:
        raise ValueError(f"models.yaml missing 'repo_id' for model '{name}'")

    # 这里严格匹配你的 __init__(repo_id, runtime_cfg)
    model: MLLM = wrapper_cls(repo_id, runtime_cfg)
    return model
