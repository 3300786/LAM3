import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    LlavaForConditionalGeneration,
    AutoProcessor
)
from PIL import Image

# 尝试导入 Qwen 特有的工具，如果不存在则忽略（兼容非 Qwen 环境）
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class ModelAdapter:
    """
    Base Adapter class to unify API for different VLMs.
    """

    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def prepare_inputs(self, messages):
        """
        Return dict of inputs compatible with model(**inputs)
        """
        raise NotImplementedError

    def get_hidden_states(self, inputs):
        """
        Return list of hidden states (one per layer)
        """
        raise NotImplementedError

    def generate(self, inputs, max_new_tokens):
        raise NotImplementedError

    def decode(self, generated_ids, input_ids):
        raise NotImplementedError


class Qwen2_5_VLAdapter(ModelAdapter):
    def load(self):
        print(f"Loading Qwen2.5-VL from {self.model_path}...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        return self.model, self.processor

    def prepare_inputs(self, messages):
        # Qwen specific processing
        text_in = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_in],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        return inputs


class LlavaAdapter(ModelAdapter):
    def load(self):
        print(f"Loading LLaVA-1.5 from {self.model_path}...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,  # LLaVA 通常用 fp16
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        return self.model, self.processor

    def prepare_inputs(self, messages):
        # LLaVA specific processing
        # messages structure: [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]

        # 1. Extract Text and Images
        prompt_text = ""
        images = []

        # Simple flattener for single-turn LLaVA
        content = messages[0]['content']
        for item in content:
            if item['type'] == 'text':
                prompt_text += item['text']
            elif item['type'] == 'image':
                images.append(item['image'])

        # LLaVA prompt format: "USER: <image>\nPrompt\nASSISTANT:"
        if images:
            # LLaVA 1.5 expects <image> token explicitly
            prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        else:
            prompt = f"USER: {prompt_text}\nASSISTANT:"

        if images:
            inputs = self.processor(text=prompt, images=images[0], return_tensors="pt").to(self.device)
        else:
            # LLaVA might complain if no image is passed but usually handles text-only if config allows.
            # Ideally LLaVA is vision-encoder-decoder, inputs without images might need dummy pixel_values or model config change.
            # For simplicity, we assume generic processor handle.
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.device)

        return inputs


def get_adapter(model_name, model_path, device):
    if "Qwen" in model_name:
        return Qwen2_5_VLAdapter(model_path, device)
    elif "llava" in model_name.lower():
        return LlavaAdapter(model_path, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")