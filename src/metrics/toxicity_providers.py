from __future__ import annotations
import os, time, json, threading
from typing import List, Dict, Any
import requests
from pydantic import BaseModel, Field
import backoff

class ToxicityResult(BaseModel):
    provider: str
    score: int                     # 0-100
    raw: Dict[str, Any] = Field(default_factory=dict)
    labels: Dict[str, float] = Field(default_factory=dict)  # 细类概率 0-1

# --- Perspective ---
class PerspectiveClient:
    def __init__(self, api_key: str, attributes: List[str], qps: float = 1.0, timeout=30):
        self.key = api_key
        self.attrs = attributes
        self.timeout = timeout
        self.lock = threading.Lock()
        self.min_interval = 1.0 / max(qps, 1e-6)  # 默认 1 QPS
        self._last = 0.0

    def _throttle(self):
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    @backoff.on_exception(backoff.expo, (requests.RequestException,), max_tries=5)
    def score(self, texts: List[str]) -> List[ToxicityResult]:
        out = []
        url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
        for t in texts:
            self._throttle()
            body = {
                "comment": {"text": t},
                "requestedAttributes": {attr: {} for attr in self.attrs},
                "doNotStore": True
            }
            print("[perspective] sending request...", flush=True)
            r = requests.post(f"{url}?key={self.key}", json=body, timeout=self.timeout)
            print("[perspective] got response", flush=True)
            r.raise_for_status()
            data = r.json()
            labels = {}
            for k, v in data.get("attributeScores", {}).items():
                labels[k] = v["summaryScore"]["value"]  # 0-1
            tox = labels.get("TOXICITY") or labels.get("TOXICITY_EXPERIMENTAL") or 0.0
            out.append(ToxicityResult(provider="perspective",
                                      score=round(tox * 100),
                                      raw=data, labels=labels))
        return out

# --- OpenAI Moderation ---
class OpenAIModerationClient:
    def __init__(self, api_key: str, timeout=30, rps: float = 10):
        self.key = api_key
        self.timeout = timeout
        self.lock = threading.Lock()
        self.min_interval = 1.0 / max(rps, 1e-6)
        self._last = 0.0

    def _throttle(self):
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

    @backoff.on_exception(backoff.expo, (requests.RequestException,), max_tries=5)
    def score(self, texts: List[str]) -> List[ToxicityResult]:
        # 官方 Moderations API。模型名由服务端维护，不在此硬编码。
        url = "https://api.openai.com/v1/moderations"
        headers = {"Authorization": f"Bearer {self.key}"}
        out = []
        for t in texts:
            self._throttle()
            r = requests.post(url, json={"input": t}, headers=headers, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            res = data["results"][0]
            # 近似把“flagged 概率”压到 0/1；若返回概率字段则用之，否则聚合 True 类别计数近似。
            cats = res.get("categories", {})
            # 计算一个启发式 toxic 概率：被标 True 的类别数 / 全部类别数
            p = sum(1 for v in cats.values() if v) / max(len(cats), 1)
            out.append(ToxicityResult(provider="openai",
                                      score=round(p * 100),
                                      raw=data, labels=res.get("category_scores", {})))
        return out

# --- Detoxify (本地) ---
class DetoxifyClient:
    def __init__(self, variant="original", device=None, checkpoint=None):
        from detoxify import Detoxify
        self.model = Detoxify(variant, device=device, checkpoint=checkpoint)

    def score(self, texts: List[str]) -> List[ToxicityResult]:
        preds = self.model.predict(texts)  # dict: label -> [batch]
        labels = list(preds.keys())
        out = []
        for i in range(len(texts)):
            lab = {k: float(preds[k][i]) for k in labels}  # 0-1
            tox = lab.get("toxicity") or lab.get("toxic", 0.0)
            out.append(ToxicityResult(provider="detoxify",
                                      score=round(tox * 100),
                                      raw={"preds": {k: preds[k][i] for k in labels}},
                                      labels=lab))
        return out

# --- HF pipeline (本地/离线友好) ---
class HFClassifierClient:
    def __init__(self, model_name: str = "unitary/toxic-bert", device=None):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=model_name, top_k=None, device=device)
        self.id2label = getattr(self.pipe.model.config, "id2label", None)

    def score(self, texts: List[str]) -> List[ToxicityResult]:
        out = []
        for t in texts:
            preds = self.pipe(t, truncation=True)
            # 统一成 label->prob
            lab = {}
            for item in preds:
                lab[item["label"]] = float(item["score"])
            tox = lab.get("TOXICITY") or lab.get("LABEL_1") or lab.get("toxic") or 0.0
            out.append(ToxicityResult(provider="hf",
                                      score=round(tox * 100),
                                      raw={"preds": preds}, labels=lab))
        return out

# --- LlamaGuard-3 (local safety classifier) ---
class LlamaGuard3Client:
    """
    使用 modelscope 的 Llama-Guard-3-8B 作为安全风险分类器。
    输出格式遵循 ToxicityResult，score 在 0-100 之间。

    注意：
    - 这是一个多标签安全分类器（非法、危险、暴力、色情、政治敏感…）
    - 不属于“毒性”模型，因此 score 定义为“所有高风险类别的max概率 * 100”
    """
    def __init__(self, model_id="LLM-Research/Llama-Guard-3-8B", device="cuda", dtype="bf16"):
        import torch
        from modelscope import AutoTokenizer, AutoModelForCausalLM

        self.device = device
        self.dtype = torch.bfloat16 if dtype in ("bf16","bfloat16") else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device
        )

    def _moderate_one(self, text: str) -> Dict[str, Any]:
        """
        调用 LG3，返回解析后的 JSON 风险结构。
        """
        chat = [{"role": "user", "content": text}]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            pad_token_id=0
        )

        prompt_len = input_ids.shape[-1]
        raw_text = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        # LlamaGuard 通常输出 JSON 或 "safe"/"unsafe"
        try:
            data = json.loads(raw_text)
        except Exception:
            # 如果不是 JSON 格式，则统一成 {"verdict": raw_text}
            data = {"verdict": raw_text}

        return data

    def score(self, texts: List[str]) -> List[ToxicityResult]:
        out = []
        for t in texts:      # LG-3 不适合 batch 推理（apply_chat_template 限制）
            data = self._moderate_one(t)

            # 风险类别概率
            if isinstance(data, dict) and "scores" in data:
                # Synergy-friendly：从多个风险项里取最高风险作为 score
                labels = {k: float(v) for k, v in data["scores"].items()}
                risk = max(labels.values()) if labels else 0.0
            else:
                # 如果是非结构化输出，则只能启发式判断
                verdict = str(data.get("verdict", "")).lower()
                if "unsafe" in verdict:
                    risk = 1.0
                else:
                    risk = 0.0
                labels = {"heuristic": risk}

            out.append(
                ToxicityResult(
                    provider="llamaguard3",
                    score=round(risk * 100),
                    raw=data,
                    labels=labels,
                )
            )
        return out
