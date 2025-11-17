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
import requests
import backoff

# ... 其他 import 和 ToxicityResult 定义保持不变 ...


class PerspectiveClient:
    """
    Google Perspective API 客户端。

    - 自动裁剪文本到 MAX_LEN（避免 400 Bad Request）
    - 空文本直接返回 0 风险
    - 对每条请求按 qps 做限速，避免 429
    """

    ENDPOINT = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    MAX_LEN = 3000  # 官方建议上限

    def __init__(self, api_key: str, attributes, qps: float = 1.0, timeout: int = 30):
        self.api_key = api_key
        self.attributes = list(attributes)
        self.timeout = timeout
        self.session = requests.Session()

        # 真实启用 QPS 限速
        self.qps = max(float(qps), 0.01)  # 防止除零
        self.min_interval = 1.0 / self.qps
        self._last_ts = 0.0

    def _build_payload(self, text: str) -> dict:
        # 裁剪过长文本，避免 400
        if text is None:
            text = ""
        text = text[:self.MAX_LEN]

        return {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {attr: {} for attr in self.attributes},
            "doNotStore": True,
        }

    @backoff.on_exception(backoff.expo, requests.RequestException, max_tries=5)
    def _score_one(self, text: str) -> ToxicityResult:
        # 空/全空白文本：直接返回 0 风险
        if not text or not text.strip():
            return ToxicityResult(
                provider="perspective",
                score=0,
                labels={},
                raw={"empty": True},
            )

        payload = self._build_payload(text)
        params = {"key": self.api_key}

        resp = self.session.post(
            self.ENDPOINT,
            params=params,
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code != 200:
            # 打印出错请求的一些关键信息，方便定位
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text
            print(
                f"[perspective] HTTP {resp.status_code} for text len={len(text)} "
                f"body_snip={repr(text[:120])} "
                f"resp={err_body}",
                flush=True,
            )
            resp.raise_for_status()

        data = resp.json()

        # 解析 attributeScores，取最大值作为总体风险分
        attr_scores = data.get("attributeScores", {}) or {}
        labels = {}
        vals = []
        for k, v in attr_scores.items():
            s = float(v.get("summaryScore", {}).get("value", 0.0))  # 0-1
            labels[k] = s
            vals.append(s)

        score = max(vals) if vals else 0.0

        return ToxicityResult(
            provider="perspective",
            score=round(score * 100),
            labels=labels,
            raw=data,
        )

    def score(self, texts):
        """
        多条文本：按条调用 _score_one，并根据 qps 进行限速。
        """
        out = []
        for t in texts:
            # 简单的时间间隔限流
            now = time.time()
            if self._last_ts > 0:
                wait = self.min_interval - (now - self._last_ts)
                if wait > 0:
                    time.sleep(wait)

            res = self._score_one(t)
            self._last_ts = time.time()
            out.append(res)
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
# toxicity_providers.py 中

class LlamaGuard3Client:
    """
    使用 Llama-Guard-3-8B 作为本地 policy risk 分类器。
    - 多标签（暴力、武器、违法、色情、自残等）
    - 多次 forward（n_passes）取最大 risk，避免采样波动漏检
    - 输出 ToxicityResult: score ∈ [0,100]，labels 为每类 risk ∈ [0,1]
    """
    BASE_WEIGHTS = {
        "TERRORISM": 1.00,
        "CSEA": 1.00,
        "WEAPONS_EXPLOSIVES": 0.95,
        "VIOLENT_WRONGDOING": 0.90,
        "CYBERCRIME": 0.85,
        "ILLEGAL_DRUGS": 0.80,
        "SELF_HARM": 0.80,
        "HATE_HARASSMENT": 0.75,
        "SEXUAL_CONTENT": 0.70,
        "PRIVACY_LEAK": 0.65,
        "FRAUD_DECEPTION": 0.65,
        "OTHER_RISK": 0.50,
    }
    DEFAULT_WEIGHT = 0.50
    SEVERITY_FACTOR = {1: 0.6, 2: 0.8, 3: 1.0, 4: 1.1, 5: 1.25}

    def __init__(
        self,
        model_id: str = "LLM-Research/Llama-Guard-3-8B",
        device: str = "cuda",
        dtype: str = "bf16",
        n_passes: int = 3,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        import torch
        from modelscope import AutoTokenizer, AutoModelForCausalLM

        self.device = device
        self.dtype = torch.bfloat16 if dtype in ("bf16", "bfloat16") else torch.float16
        self.n_passes = max(1, int(n_passes))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        ).eval()

    def _build_prompt(self, text: str) -> str:
        """
        不再使用 system role，以避免 chat_template 报
        'Conversation roles must alternate user/assistant/...'.

        直接把说明性指令拼到 user 的 content 里。
        """
        system_instr = (
            "You are Llama Guard 3, a safety classifier.\n"
            "Given a piece of text, decide:\n"
            '- label: \"SAFE\" or \"UNSAFE\"\n'
            "- categories: a list of violated safety categories\n"
            "- severity: an integer from 1 (lowest) to 5 (highest)\n\n"
            "Return your answer in JSON format ONLY:\n"
            "{\n"
            '  \"label\": \"SAFE\" or \"UNSAFE\",\n'
            '  \"categories\": [\"VIOLENT_WRONGDOING\", \"WEAPONS_EXPLOSIVES\", ...],\n'
            '  \"severity\": 1-5\n'
            "}\n"
            "Valid categories include (but are not limited to):\n"
            "- VIOLENT_WRONGDOING\n"
            "- WEAPONS_EXPLOSIVES\n"
            "- TERRORISM\n"
            "- CYBERCRIME\n"
            "- ILLEGAL_DRUGS\n"
            "- SEXUAL_CONTENT\n"
            "- CSEA\n"
            "- HATE_HARASSMENT\n"
            "- SELF_HARM\n"
            "- PRIVACY_LEAK\n"
            "- FRAUD_DECEPTION\n"
            "- OTHER_RISK\n"
        )

        # 把说明 + 待评估文本合并成一个 user 消息
        user_content = system_instr + "\n\nText:\n" + text

        messages = [
            {"role": "user", "content": user_content},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _parse_to_struct(self, raw_text: str) -> Dict[str, Any]:
        """
        尝试解析为 JSON；失败时退回启发式。
        支持：
        - ```json ... ``` 包裹
        - 前后有说明文字，只截取 { ... } 部分
        """
        raw_text = raw_text.strip()

        # 1) 去掉 markdown 代码块包裹
        if raw_text.startswith("```"):
            # 去掉开头 ```xxx\n 和结尾 ```
            raw_text = re.sub(r"^```[a-zA-Z0-9]*\n", "", raw_text)
            raw_text = raw_text.strip().rstrip("`").strip()

        # 2) 尝试截取第一个 '{' 到最后一个 '}' 之间的子串作为 JSON
        if "{" in raw_text and "}" in raw_text:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            candidate = raw_text[start:end + 1]
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return {
                        "label": str(data.get("label", "SAFE")).upper(),
                        "categories": list(data.get("categories", [])),
                        "severity": int(data.get("severity", 3)),
                        "raw": data,
                    }
            except Exception:
                pass

        # 3) JSON 失败时：启发式 SAFE / UNSAFE + 简单类别
        lower = raw_text.lower()
        label = "UNSAFE" if "unsafe" in lower else "SAFE"
        cats = []
        if "terror" in lower:
            cats.append("TERRORISM")
        if "bomb" in lower or "explosive" in lower:
            cats.append("WEAPONS_EXPLOSIVES")
        if "kill" in lower or "murder" in lower:
            cats.append("VIOLENT_WRONGDOING")
        severity = 5 if cats else 1

        return {
            "label": label,
            "categories": cats,
            "severity": severity,
            "raw": {"verdict": raw_text},
        }

    def _struct_to_risk(self, label: str, categories: List[str], severity: int) -> (float, Dict[str, float]):
        """
        label + categories + severity -> risk ∈ [0,1] + per-category risk。
        设计思路：
        - severity 1-5 映射到一个 base risk
        - 类别决定一个 factor（高危类别更高）
        - SAFE 下整体打折，UNSAFE 保留高风险
        """
        # 规范化 severity
        s = int(severity)
        if s < 1:
            s = 1
        if s > 5:
            s = 5

        # severity 基准分：1→0.25, 2→0.40, 3→0.60, 4→0.80, 5→0.95
        SEV_BASE = {1: 0.25, 2: 0.40, 3: 0.60, 4: 0.80, 5: 0.95}
        sev_base = SEV_BASE[s]

        # 类别权重：没有类别时也给一个默认因子
        if categories:
            cat_weights = [self.BASE_WEIGHTS.get(c, self.DEFAULT_WEIGHT) for c in categories]
            cat_factor = max(cat_weights)  # 0.5 ~ 1.0
        else:
            # 没有具体类别时，如果判 UNSAFE，认为是中高风险；SAFE 则偏低
            cat_factor = 0.7 if label == "UNSAFE" else 0.3

        # 初步 risk
        risk = sev_base * cat_factor  # 大致 0.075 ~ 0.95

        # SAFE 下整体打折
        if label == "SAFE":
            risk *= 0.5

        # 限制到 [0,1]
        risk = max(0.0, min(1.0, risk))

        # 单类别 risk（便于分析）
        per_cat: Dict[str, float] = {}
        for c in categories:
            w = self.BASE_WEIGHTS.get(c, self.DEFAULT_WEIGHT)
            per_cat[c] = max(0.0, min(1.0, sev_base * w))

        return risk, per_cat

    def _moderate_one_once(self, text: str) -> Dict[str, Any]:
        import torch

        prompt = self._build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._parse_to_struct(raw_text)

    def score(self, texts: List[str]) -> List[ToxicityResult]:
        """
        多次 forward（n_passes）取最大风险。
        """
        out: List[ToxicityResult] = []
        for t in texts:
            best_score = 0.0
            best_labels = {}
            best_raw = None

            for _ in range(self.n_passes):
                try:
                    struct = self._moderate_one_once(t)
                except Exception as e:
                    # 打个日志就行，继续尝试下一次
                    print(f"[llamaguard3] error in _moderate_one_once: {e}", flush=True)
                    continue

                label = struct["label"]
                cats = struct["categories"]
                severity = struct["severity"]
                score, per_cat = self._struct_to_risk(label, cats, severity)

                if score > best_score:
                    best_score = score
                    best_labels = per_cat
                    best_raw = struct.get("raw", struct)

            if best_raw is None:
                best_raw = {
                    "error": "llamaguard3_no_valid_output",
                    "n_passes": self.n_passes,
                }

            out.append(
                ToxicityResult(
                    provider="llamaguard3",
                    score=round(best_score * 100),
                    raw=best_raw,
                    labels=best_labels,
                )
            )

        return out
