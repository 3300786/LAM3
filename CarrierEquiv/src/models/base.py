# src/models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from src.utils.runtime import GenCfg


class MLLM(ABC):
    @abstractmethod
    def generate(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg,
        *,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        给定图像 + 文本 prompt，返回模型生成的字符串结果。

        Parameters
        ----------
        image:
            PIL.Image.Image | None (取决于具体 wrapper 支持)
        prompt:
            user role 的文本内容（不要把 system 拼进去）
        gen_cfg:
            生成参数
        system_prompt:
            system role 的内容（可选）。用于在不改变“载体变量”的前提下定义统一任务。
        kwargs:
            预留扩展：比如 trace=True、return_logits=True、tools=... 等。
        """
        raise NotImplementedError

    def generate_with_trace(
        self,
        image: Any,
        prompt: str,
        gen_cfg: GenCfg,
        *,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        默认只包装 generate 的输出。
        wrapper 若需要 cross-attn / logits / hidden-states，可重写此方法。
        """
        output = self.generate(
            image=image,
            prompt=prompt,
            gen_cfg=gen_cfg,
            system_prompt=system_prompt,
            **kwargs,
        )
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
    def encode_modalities_with_trace(
        self,
        image: Any,
        prompt: str,
        gen_cfg: Optional[GenCfg] = None,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encode_modalities_with_trace()."
        )