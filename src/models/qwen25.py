import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

class Qwen25VLWrapper:
    def __init__(self, repo_dir, runtime_cfg=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        quant = runtime_cfg.get("quantization", {}) if runtime_cfg else {}
        load_kwargs = {}

        if quant.get("enabled", False):
            load_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": quant.get("use_double_quant", True),
                "bnb_4bit_quant_type": quant.get("quant_type", "nf4"),
            })
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModelForVision2Seq.from_pretrained(
            repo_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **load_kwargs,
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(
            repo_dir,
            trust_remote_code=True,
        )

    def _prepare_inputs(self, image_path, text):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device, dtype=torch.bfloat16)
        return inputs

    @torch.no_grad()
    def generate(self, image_path, prompt, gen_cfg):
        inputs = self._prepare_inputs(image_path, prompt)
        out = self.model.generate(
            **inputs,
            max_new_tokens=gen_cfg.max_new_tokens,
            min_new_tokens=gen_cfg.min_new_tokens,
            do_sample=gen_cfg.do_sample,
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.decode(out[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(self, image_paths, prompts, gen_cfg):
        # 按 batch size 处理
        outs = []
        for img, txt in zip(image_paths, prompts):
            outs.append(self.generate(img, txt, gen_cfg))
        return outs
