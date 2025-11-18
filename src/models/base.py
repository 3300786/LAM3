# src/models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
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
# ===== 新增：模态表征接口，用于 D(x) 等分析 =====
    def encode_modalities(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg | None = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        返回一个 dict，键为模态名（例如 "text", "image"），值为对应模态的向量表征。
        缺省实现抛错，具体模型（Idefics2 / LLaVA 等）自行重写。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encode_modalities()."
        )
