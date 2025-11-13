# src/attacks/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
from src.models.base import MLLM
from src.utils.runtime import GenCfg


class AttackResult(Dict[str, Any]):
    """
    约定字段:
    - adv_prompt: 攻击后的完整 prompt
    - output: 模型输出
    - success: bool, 是否攻击成功（ASR 会基于这个统计）
    - extra: 其他调试/分析信息
    """


class Attack(ABC):
    @abstractmethod
    def run(
        self,
        model: MLLM,
        image: Any,
        base_prompt: str,
        gen_cfg: GenCfg,
        meta: Dict[str, Any] | None = None,
    ) -> AttackResult:
        """
        给一个模型和原始输入，返回攻击结果。
        """
        raise NotImplementedError
