# src/models/base.py
from abc import ABC, abstractmethod
from typing import Any
from src.utils.runtime import GenCfg


class MLLM(ABC):
    @abstractmethod
    def generate(self, image: Any, prompt: str, gen_cfg: GenCfg) -> str:
        """
        给定图像和文本 prompt，返回模型生成的字符串结果。
        """
        raise NotImplementedError

    def generate_with_trace(self, image: Any, prompt: str, gen_cfg: GenCfg, **kwargs):
        """
        默认只包装 generate 的输出，后面做梯度攻击时，
        各个 Wrapper 可以重写这个方法，返回 cross-attn / logits 等。
        """
        output = self.generate(image, prompt, gen_cfg)
        return {"output": output}
