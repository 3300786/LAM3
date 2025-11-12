from dataclasses import dataclass
import torch, yaml, random, numpy as np
from typing import Dict, Any

# ---- 生成配置（一次定义，全项目复用） ----
@dataclass
class GenCfg:
    max_new_tokens: int = 256
    min_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False  # 需要采样时置 True

# ---- 基础工具 ----
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---- 统一规范化 runtime 配置键 ----
def normalize_runtime(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化键位：
      - 兼容旧键：顶层 load_in_4bit -> quantization.enabled
      - 填默认值：compute_dtype/quant_type/use_double_quant
      - 兼容 device/device_map：若给定 device，则映射成 {"": device}
    """
    cfg = dict(cfg or {})
    q = dict(cfg.get("quantization") or {})

    # 旧键映射
    if "load_in_4bit" in cfg and "enabled" not in q:
        q["enabled"] = bool(cfg.pop("load_in_4bit"))

    # 默认量化开关与参数
    q.setdefault("enabled", False)
    q.setdefault("compute_dtype", str(cfg.get("precision", "bf16")))
    q.setdefault("quant_type", "nf4")
    q.setdefault("use_double_quant", True)
    cfg["quantization"] = q

    # 设备键对齐：优先 device，其次 device_map，默认 "auto"
    dev = cfg.get("device")
    if dev:
        cfg["device_map"] = {"": dev}
    cfg.setdefault("device_map", "auto")

    # 精度默认
    cfg.setdefault("precision", "bf16")

    return cfg

# ---- from_pretrained 的设备与精度参数（轻量、安全） ----
def device_kwargs(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回可直接传给 `from_pretrained` 的最小必要 kwargs。
    规则：
      - 非量化：设置 torch_dtype + device_map
      - 量化(bnb4bit)：不设置 torch_dtype（避免触发量化权重上的 .to）
        仅在 device_map 不是 'auto' 时设置 device_map（减少 accelerate 的干预）
      - BitsAndBytesConfig 交由各 wrapper 内部构造
    """
    cfg = normalize_runtime(runtime_cfg)
    out: Dict[str, Any] = {}

    q_enabled = bool(cfg["quantization"]["enabled"])
    precision = str(cfg.get("precision", "bf16")).lower()
    dtype = torch.bfloat16 if precision in ("bf16", "bfloat16") else torch.float16

    dm = cfg.get("device_map", "auto")

    if not q_enabled:
        out["torch_dtype"] = dtype
        out["device_map"] = dm
    else:
        # 量化路径：避免显式 dtype；仅在明确指定时传入 device_map
        if isinstance(dm, dict) or (isinstance(dm, str) and dm != "auto"):
            out["device_map"] = dm
        # 否则让各 wrapper 自己决定（更安全）

    # 统一加上低内存加载
    out["low_cpu_mem_usage"] = True
    return out
