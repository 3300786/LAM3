# scripts/attn_intra-inter.py
# Added: unit_mode = png / prompt_semantic / prompt_strict
# - png: unit_id = base_qid (same PNG)
# - prompt_semantic: unit_id = hash(t1 + t2) (semantic role)
# - prompt_strict: unit_id = hash(exact prompt text used by wrapper if available, else reconstructed)
#
# Stats are computed over unit_id instead of base_qid.
# Group slicing still supported: group_mode=all/within/both.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
except Exception:
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore


# ---------------------------
# Utils
# ---------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _get_group_sort_key(g: str) -> Tuple[int, str]:
    m = re.match(r"([A-Z])(\d+)", g)
    if m:
        letter = m.group(1)
        num = int(m.group(2))
        letter_rank = {"A": 0, "B": 1, "C": 2}.get(letter, 9)
        return (letter_rank * 100 + num, g)
    return (9999, g)


def _set_attn_eager_if_possible(model) -> None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    for k in ["attn_implementation", "_attn_implementation"]:
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, "eager")
            except Exception:
                pass
    if hasattr(cfg, "_attn_implementation_autoset"):
        try:
            setattr(cfg, "_attn_implementation_autoset", False)
        except Exception:
            pass
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    la, lb = a.shape[0], b.shape[0]
    if la != lb:
        L = max(la, lb)
        if la < L:
            a = np.pad(a, (0, L - la), constant_values=0.0)
        if lb < L:
            b = np.pad(b, (0, L - lb), constant_values=0.0)

    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        raise RuntimeError("JSON must be a list of objects.")
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_outcome_flags(obj: Dict[str, Any]) -> Dict[str, Any]:
    j = obj.get("qwen_judge") or {}
    is_asr = bool(j.get("is_attack_success", False)) and (not bool(j.get("echo_only", False)))
    is_refusal = bool(j.get("is_refusal", False))
    echo = bool(j.get("echo_only", False))
    return {"asr": int(is_asr), "refusal": int(is_refusal), "echo": int(echo)}


def _parse_groups(s: str) -> Optional[set[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return {x.strip() for x in s.split(",") if x.strip()}


def _parse_hw(s: str) -> Optional[Tuple[int, int]]:
    s = (s or "").strip().lower()
    if not s or s == "none":
        return None
    if "x" not in s:
        raise ValueError(f"--resample_hw expects like 16x16, got: {s}")
    a, b = s.split("x", 1)
    h, w = int(a), int(b)
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid resample_hw: {s}")
    return (h, w)


def _resample_grid_np(grid: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    g = torch.from_numpy(grid.astype(np.float32))[None, None, ...]  # [1,1,H,W]
    out = torch.nn.functional.interpolate(g, size=hw, mode="bilinear", align_corners=False)
    return out[0, 0].cpu().numpy()


def _norm_ws(s: str) -> str:
    # stable hashing: collapse whitespace, trim
    return re.sub(r"\s+", " ", (s or "").strip())


def _sha1_short(s: str, n: int = 12) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return h[:n]


def _build_prompt_semantic_key(it: Dict[str, Any]) -> str:
    # semantic role key: t1 + t2 (do NOT include group / mixed order)
    t1 = str(it.get("t1", it.get("t1_raw", "")) or "")
    t2 = str(it.get("t2", "") or "")
    s = _norm_ws(t1) + "\n" + _norm_ws(t2)
    return f"ps:{_sha1_short(s)}"


def _build_prompt_strict_key(it: Dict[str, Any]) -> str:
    # strict key aims to reflect the exact text passed as prompt_text (wrapper side).
    # Prefer user_prompt if non-empty; else reconstruct from protocol + t1/t2 + mixed_order.
    user_prompt = str(it.get("user_prompt", "") or "").strip()
    if user_prompt:
        s = user_prompt
    else:
        t1 = str(it.get("t1", it.get("t1_raw", "")) or "")
        t2 = str(it.get("t2", "") or "")
        sys = str((it.get("protocol") or {}).get("global_system_prompt", "") or "")
        mixed_order = it.get("mixed_order", None)
        # Keep reconstruction deterministic; do NOT guess wrapper-specific fixed guides.
        s = (
            "### SYSTEM ###\n"
            + _norm_ws(sys)
            + "\n\n### INSTRUCTION ###\n"
            + _norm_ws(t2)
            + "\n\n### CLUE ###\n"
            + _norm_ws(t1)
            + "\n\n### MIXED_ORDER ###\n"
            + _norm_ws("" if mixed_order is None else str(mixed_order))
        )
    return f"pt:{_sha1_short(_norm_ws(s))}"


def build_unit_id(it: Dict[str, Any], unit_mode: str) -> str:
    if unit_mode == "png":
        return str(it["base_qid"])
    if unit_mode == "prompt_semantic":
        return _build_prompt_semantic_key(it)
    if unit_mode == "prompt_strict":
        return _build_prompt_strict_key(it)
    raise ValueError(f"unknown unit_mode={unit_mode}")


# ---------------------------
# Attention extraction (KEEP STABLE)
# ---------------------------
def _infer_thw(image_grid_thw: Optional[torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    if image_grid_thw is None or not isinstance(image_grid_thw, torch.Tensor):
        return None
    t = image_grid_thw.detach().cpu()
    if t.ndim == 3:
        t = t[0]
    if t.ndim == 2 and t.shape[-1] == 3 and t.shape[0] >= 1:
        tt = int(t[0, 0].item())
        hh = int(t[0, 1].item())
        ww = int(t[0, 2].item())
        if tt > 0 and hh > 0 and ww > 0:
            return (tt, hh, ww)
    return None


def _get_first_existing_token_id(tokenizer, candidates: List[str]) -> Optional[int]:
    for s in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0:
                return tid
        except Exception:
            pass
    return None


def build_vision_mask_from_image_pad(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    pad_id = _get_first_existing_token_id(
        tokenizer,
        ["<|image_pad|>", "<image_pad>", "<|vision_pad|>", "<vision_pad>", "<|image|>"],
    )
    if pad_id is None:
        raise RuntimeError("Cannot find image pad token id. Tokenizer may use different vision placeholders.")
    ids = input_ids[0]
    vision_mask = (ids == pad_id)
    if not bool(vision_mask.any().item()):
        raise RuntimeError(f"pad_id={pad_id} found but no match in input_ids; vision placeholders differ.")
    return vision_mask


def build_text_mask_basic(input_ids: torch.Tensor, vision_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    ids = input_ids[0]
    text_mask = ~vision_mask
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int) and tid >= 0:
            text_mask = text_mask & (ids != tid)
    try:
        added = getattr(tokenizer, "additional_special_tokens", None)
        if isinstance(added, list):
            for t in added:
                try:
                    tid = tokenizer.convert_tokens_to_ids(t)
                    if isinstance(tid, int) and tid >= 0:
                        text_mask = text_mask & (ids != tid)
                except Exception:
                    pass
    except Exception:
        pass
    return text_mask


def _mask_span(mask_1d: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    return (int(idx.min().item()), int(idx.max().item()))


def attention_text_to_vision_fwd(
    A_ss: torch.Tensor,
    text_mask: torch.Tensor,
    vision_mask: torch.Tensor,
    agg: str = "max",
) -> torch.Tensor:
    q_idx = torch.nonzero(text_mask, as_tuple=False).squeeze(1)
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    if q_idx.numel() == 0 or k_idx.numel() == 0:
        raise RuntimeError("Empty text_mask or vision_mask.")
    sub = A_ss.index_select(0, q_idx).index_select(1, k_idx)
    if agg == "mean":
        return sub.mean(dim=0)
    if agg == "max":
        return sub.max(dim=0).values
    raise ValueError(f"Unknown agg={agg}")


def attention_gen1_q_to_vision(
    step_attn_layer: torch.Tensor,
    vision_mask: torch.Tensor,
) -> torch.Tensor:
    A = step_attn_layer[0]
    if A.ndim != 3:
        raise RuntimeError(f"Unexpected GEN1 layer attn ndim={A.ndim}, shape={tuple(A.shape)}")

    H, Q, K = A.shape
    q_idx = 0 if Q == 1 else (Q - 1)
    a_row = A.mean(dim=0)[q_idx]
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    vec = a_row.index_select(0, k_idx)
    vec = vec.float().clamp(min=0)
    vec = vec / (vec.sum() + 1e-6)
    return vec


def vec_to_grid(vec: np.ndarray, grid_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    vec = vec.astype(np.float32).reshape(-1)
    P = int(vec.shape[0])
    if grid_hw is not None:
        h, w = grid_hw
        G = h * w
        if G > 0:
            if P >= G:
                vv = vec[:G]
            else:
                vv = np.pad(vec, (0, G - P), constant_values=0.0)
            return vv.reshape(h, w)
    h = int(math.sqrt(P)) or 1
    w = int(math.ceil(P / h))
    G = h * w
    if P < G:
        vec = np.pad(vec, (0, G - P), constant_values=0.0)
    else:
        vec = vec[:G]
    return vec.reshape(h, w)


@torch.no_grad()
def extract_fwd_and_gen1(
    model,
    processor,
    device: str,
    image_path: Optional[str],
    prompt_text: str,
    layer_idx: int,
    agg_fwd: str,
    merge_size: int,
    skip_gen1: bool = False,
) -> Dict[str, Any]:
    tok = processor.tokenizer

    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], return_tensors="pt")

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    out = model(
        **inputs,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )
    attns = getattr(out, "attentions", None)
    if attns is None:
        raise RuntimeError("FWD: outputs.attentions is None")

    input_ids = inputs["input_ids"]
    if image_path is None:
        return {"has_image": False, "fwd_grid": None, "gen1_grid": None, "meta": {"seq_len": int(input_ids.shape[1])}}

    vision_mask = build_vision_mask_from_image_pad(input_ids, tok)
    text_mask = build_text_mask_basic(input_ids, vision_mask, tok)

    L = len(attns)
    li = layer_idx if layer_idx >= 0 else (L + layer_idx)
    if not (0 <= li < L):
        raise ValueError(f"Invalid layer_idx={layer_idx}, resolved={li}, total_layers={L}")

    A_ss = attns[li][0].mean(dim=0)
    fwd_vec = attention_text_to_vision_fwd(A_ss, text_mask=text_mask, vision_mask=vision_mask, agg=agg_fwd)
    fwd_vec = fwd_vec.float().clamp(min=0)
    fwd_vec = fwd_vec / (fwd_vec.sum() + 1e-6)

    thw = _infer_thw(inputs.get("image_grid_thw", None))
    if thw:
        _, h, w = thw
        grid_hw = (max(1, h // merge_size), max(1, w // merge_size))
    else:
        grid_hw = None

    gen1_vec = None
    gen1_meta: Dict[str, Any] = {}
    if not skip_gen1:
        try:
            gen_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
                use_cache=True,
            )
            g_attn = getattr(gen_out, "attentions", None)
            if g_attn is not None and isinstance(g_attn, (tuple, list)) and len(g_attn) >= 1:
                step0 = g_attn[0]
                if isinstance(step0, (tuple, list)) and len(step0) == L:
                    layer0 = step0[li]
                    gen1_vec = attention_gen1_q_to_vision(layer0, vision_mask=vision_mask)
                    gen1_meta = {"gen_attn_ok": True, "step0_layer_shape": list(layer0.shape)}
                else:
                    gen1_meta = {"gen_attn_ok": False, "reason": "unexpected step0 structure"}
            else:
                gen1_meta = {"gen_attn_ok": False, "reason": "gen_out.attentions is None/empty"}
        except Exception as e:
            gen1_meta = {"gen_attn_ok": False, "reason": f"exception: {type(e).__name__}: {e}"}
    else:
        gen1_meta = {"gen_attn_ok": False, "reason": "skip_gen1=True"}

    fwd_grid = vec_to_grid(fwd_vec.detach().cpu().numpy(), grid_hw=grid_hw)
    gen1_grid = None
    if gen1_vec is not None:
        gen1_grid = vec_to_grid(gen1_vec.detach().cpu().numpy(), grid_hw=grid_hw)

    meta = {
        "has_image": True,
        "seq_len": int(input_ids.shape[1]),
        "vision_span": list(_mask_span(vision_mask) or []),
        "text_span": list(_mask_span(text_mask) or []),
        "vision_token_count": int(vision_mask.sum().item()),
        "text_token_count": int(text_mask.sum().item()),
        "image_grid_thw": list(thw) if thw else None,
        "grid_hw": list(grid_hw) if grid_hw else None,
        "merge_size": merge_size,
        "layer_resolved": li,
        "agg_fwd": agg_fwd,
        "gen1_meta": gen1_meta,
    }
    return {"has_image": True, "fwd_grid": fwd_grid, "gen1_grid": gen1_grid, "meta": meta}


# ---------------------------
# Intra/Inter statistics (enhanced, unit-aware)
# ---------------------------
@dataclass
class SepStats:
    unit_id: str
    group: str  # "__ALL__" or actual group
    label_name: str
    n0: int
    n1: int
    mu00: float
    mu11: float
    mu01: float
    delta: float
    delta_weighted: float
    within_pairs: int
    between_pairs: int


def _pairwise_sum_count(X: List[np.ndarray]) -> Tuple[float, int]:
    if len(X) < 2:
        return 0.0, 0
    s = 0.0
    c = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            s += _cosine_flat(X[i], X[j])
            c += 1
    return float(s), int(c)


def _cross_sum_count(X0: List[np.ndarray], X1: List[np.ndarray]) -> Tuple[float, int]:
    if len(X0) == 0 or len(X1) == 0:
        return 0.0, 0
    s = 0.0
    c = 0
    for a in X0:
        for b in X1:
            s += _cosine_flat(a, b)
            c += 1
    return float(s), int(c)


def _maybe_balance(X0: List[np.ndarray], X1: List[np.ndarray], seed: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n0, n1 = len(X0), len(X1)
    if n0 == 0 or n1 == 0:
        return X0, X1
    m = min(n0, n1)
    if n0 == m and n1 == m:
        return X0, X1
    rng = np.random.RandomState(seed)
    if n0 > m:
        idx = rng.choice(n0, size=m, replace=False)
        X0 = [X0[i] for i in idx]
    if n1 > m:
        idx = rng.choice(n1, size=m, replace=False)
        X1 = [X1[i] for i in idx]
    return X0, X1


def compute_separability(
    rows: List[Dict[str, Any]],
    label_key: str,
    feat_key: str,
    group_mode: str,  # "all" or "within"
    delta_mode: str,  # "legacy" or "weighted"
    balance_within_unit: bool,
    seed: int,
) -> List[SepStats]:
    from collections import defaultdict

    mp: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    if group_mode == "all":
        for r in rows:
            mp[(str(r["unit_id"]), "__ALL__")].append(r)
    elif group_mode == "within":
        for r in rows:
            g = str(r.get("group", "") or "UNKNOWN")
            mp[(str(r["unit_id"]), g)].append(r)
    else:
        raise ValueError(f"unknown group_mode={group_mode}")

    out: List[SepStats] = []
    for (uid, g), xs in mp.items():
        X0: List[np.ndarray] = []
        X1: List[np.ndarray] = []
        for it in xs:
            feat = it.get(feat_key, None)
            if not isinstance(feat, np.ndarray):
                continue
            y = int(it[label_key])
            (X0 if y == 0 else X1).append(feat)

        if len(X0) + len(X1) < 2:
            continue
        if len(X0) == 0 or len(X1) == 0:
            continue  # no split -> cannot do intra/inter

        if balance_within_unit:
            X0, X1 = _maybe_balance(X0, X1, seed=seed + (hash(uid) % 1000003))

        s00, c00 = _pairwise_sum_count(X0)
        s11, c11 = _pairwise_sum_count(X1)
        mu00 = float(s00 / c00) if c00 > 0 else float("nan")
        mu11 = float(s11 / c11) if c11 > 0 else float("nan")

        s01, c01 = _cross_sum_count(X0, X1)
        if c01 <= 0:
            continue
        mu01 = float(s01 / c01)

        if (not math.isnan(mu00)) and (not math.isnan(mu11)):
            delta_legacy = (mu00 + mu11) / 2.0 - mu01
        else:
            delta_legacy = float("nan")

        sw = s00 + s11
        cw = c00 + c11
        delta_weighted = float((sw / cw) - mu01) if cw > 0 else float("nan")

        delta = delta_legacy if delta_mode == "legacy" else delta_weighted

        out.append(
            SepStats(
                unit_id=str(uid),
                group=str(g),
                label_name=str(label_key),
                n0=len(X0),
                n1=len(X1),
                mu00=mu00,
                mu11=mu11,
                mu01=mu01,
                delta=float(delta),
                delta_weighted=float(delta_weighted),
                within_pairs=int(cw),
                between_pairs=int(c01),
            )
        )
    return out


def summarize_stats(stats: List[SepStats]) -> Dict[str, Any]:
    deltas = np.array([s.delta for s in stats if not (math.isnan(s.delta) or math.isinf(s.delta))], dtype=np.float64)
    if deltas.size == 0:
        return {
            "num_units_with_split": 0,
            "delta_mean": float("nan"),
            "delta_median": float("nan"),
            "delta_std": float("nan"),
            "delta_min": float("nan"),
            "delta_max": float("nan"),
            "delta_pos_frac": float("nan"),
        }
    return {
        "num_units_with_split": int(deltas.size),
        "delta_mean": float(np.mean(deltas)),
        "delta_median": float(np.median(deltas)),
        "delta_std": float(np.std(deltas)),
        "delta_min": float(np.min(deltas)),
        "delta_max": float(np.max(deltas)),
        "delta_pos_frac": float(np.mean(deltas > 0)),
    }


# ---------------------------
# Visualization (PCA 2D + LDA 1D)
# ---------------------------
def _try_import_sklearn():
    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        return PCA, LinearDiscriminantAnalysis, StandardScaler
    except Exception:
        return None, None, None


def _save_pca_2d_plot(out_png: Path, X: np.ndarray, y: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=0.8)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _save_lda_1d_plot(out_png: Path, z: np.ndarray, y: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    z0 = z[y == 0]
    z1 = z[y == 1]
    plt.hist(z0, bins=40, alpha=0.6, density=True)
    plt.hist(z1, bins=40, alpha=0.6, density=True)
    plt.title(title)
    plt.xlabel("LDA-1")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _build_Xy(
    feat_rows: List[Dict[str, Any]],
    label_key: str,
    group_filter: Optional[str],
) -> Tuple[List[np.ndarray], List[int]]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for r in feat_rows:
        if group_filter is not None and str(r.get("group", "")) != group_filter:
            continue
        feat = r.get("feat", None)
        if not isinstance(feat, np.ndarray):
            continue
        X_list.append(feat.reshape(-1).astype(np.float32))
        y_list.append(int(r[label_key]))
    return X_list, y_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_json", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--target_model", default="qwen25_vl_7b")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_items", type=int, default=-1)
    ap.add_argument("--max_unit", type=int, default=-1, help="limit distinct units (after filtering), for debug")

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--agg_fwd", choices=["mean", "max"], default="max")
    ap.add_argument("--merge_size", type=int, default=2)
    ap.add_argument("--skip_gen1", action="store_true")

    ap.add_argument("--feat", default="gen1", choices=["gen1", "fwd"])
    ap.add_argument("--label", default="refusal", choices=["refusal", "asr"])

    # NEW: unit definition (png vs prompt)
    ap.add_argument("--unit_mode", default="png", choices=["png", "prompt_semantic", "prompt_strict"],
                    help="png: base_qid; prompt_semantic: hash(t1+t2); prompt_strict: hash(user_prompt or reconstructed)")

    # group controls
    ap.add_argument("--group_mode", default="all", choices=["all", "within", "both"],
                    help="all: per unit_id; within: (unit_id,group); both: run both")
    ap.add_argument("--groups", default="", help='comma-separated group filter, e.g. "B2,C1" (optional)')
    ap.add_argument("--drop_echo", action="store_true", help="drop echo_only=1 items before analysis")

    # metric controls
    ap.add_argument("--delta_mode", default="legacy", choices=["legacy", "weighted"])
    ap.add_argument("--balance_within_unit", action="store_true", help="downsample each class to min(n0,n1) per unit")
    ap.add_argument("--seed", type=int, default=0)

    # visualization controls
    ap.add_argument("--resample_hw", default="none", help='resample grid to fixed size for feature similarity, e.g. "16x16"')
    ap.add_argument("--plot_per_group", action="store_true")
    ap.add_argument("--min_plot_n", type=int, default=200)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    allowed_groups = _parse_groups(args.groups)
    resample_hw = _parse_hw(args.resample_hw)

    items = _load_json_or_jsonl(Path(args.judged_json))

    # Filter items (NOTE: attention here requires image_path != None)
    filtered: List[Dict[str, Any]] = []
    for it in items:
        if str(it.get("model", "")) != args.target_model:
            continue
        if it.get("qid") is None:
            continue
        if it.get("image_path") is None:
            continue  # vision attention not defined
        g = str(it.get("group", "") or "UNKNOWN")
        if allowed_groups is not None and g not in allowed_groups:
            continue
        flags = _extract_outcome_flags(it)
        if args.drop_echo and int(flags["echo"]) == 1:
            continue
        filtered.append(it)
        if args.max_items > 0 and len(filtered) >= args.max_items:
            break

    # Build unit ids early, optionally cap by max_unit
    for it in filtered:
        it["_unit_id"] = build_unit_id(it, args.unit_mode)

    if args.max_unit > 0:
        from collections import defaultdict
        mp = defaultdict(list)
        for it in filtered:
            mp[str(it["_unit_id"])].append(it)
        # keep most populated units
        unit_sorted = sorted([(len(v), k) for k, v in mp.items()], key=lambda x: (-x[0], x[1]))
        keep = set([k for _, k in unit_sorted[: args.max_unit]])
        filtered = [it for it in filtered if str(it["_unit_id"]) in keep]

    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in your transformers build.")

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device).eval()
    _set_attn_eager_if_possible(model)

    # Extract features
    feat_rows: List[Dict[str, Any]] = []
    cache: Dict[str, Dict[str, Any]] = {}

    pbar = tqdm(filtered, desc="extract", dynamic_ncols=True)
    for it in pbar:
        qid = str(it["qid"])
        if qid in cache:
            rec = cache[qid]
        else:
            # IMPORTANT: for prompt-based unit analysis we must use the actual prompt_text used by wrapper.
            # Prefer user_prompt, else fallback to "" (will reduce usefulness for strict).
            prompt_text = str(it.get("user_prompt", it.get("prompt", "")) or "")
            img_path = str(it.get("image_path"))
            try:
                rec = extract_fwd_and_gen1(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=img_path,
                    prompt_text=prompt_text,
                    layer_idx=args.layer,
                    agg_fwd=args.agg_fwd,
                    merge_size=args.merge_size,
                    skip_gen1=bool(args.skip_gen1),
                )
            except Exception as e:
                rec = {"has_image": True, "fwd_grid": None, "gen1_grid": None, "meta": {"error": f"{type(e).__name__}: {e}"}}
            cache[qid] = rec

        flags = _extract_outcome_flags(it)
        feat = rec.get("gen1_grid" if args.feat == "gen1" else "fwd_grid", None)
        if isinstance(feat, np.ndarray) and resample_hw is not None:
            feat = _resample_grid_np(feat, resample_hw)

        feat_rows.append(
            {
                "qid": qid,
                "unit_id": str(it["_unit_id"]),
                "base_qid": str(it.get("base_qid", "")),
                "group": str(it.get("group", "") or "UNKNOWN"),
                "image_path": str(it.get("image_path", "")),
                "prompt_semantic_key": _build_prompt_semantic_key(it),
                "prompt_strict_key": _build_prompt_strict_key(it),
                **flags,
                "feat": feat,
            }
        )

    # ---------------- stats: ALL / WITHIN ----------------
    stats_all: List[SepStats] = []
    stats_within: List[SepStats] = []

    if args.group_mode in ("all", "both"):
        stats_all = compute_separability(
            feat_rows,
            label_key=args.label,
            feat_key="feat",
            group_mode="all",
            delta_mode=args.delta_mode,
            balance_within_unit=bool(args.balance_within_unit),
            seed=int(args.seed),
        )
        with (out_dir / f"per_unit_stats_all_{args.unit_mode}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "group", "label", "n0", "n1", "mu00", "mu11", "mu01", "delta", "delta_weighted", "within_pairs", "between_pairs"])
            for s in stats_all:
                w.writerow([s.unit_id, s.group, s.label_name, s.n0, s.n1, s.mu00, s.mu11, s.mu01, s.delta, s.delta_weighted, s.within_pairs, s.between_pairs])

    if args.group_mode in ("within", "both"):
        stats_within = compute_separability(
            feat_rows,
            label_key=args.label,
            feat_key="feat",
            group_mode="within",
            delta_mode=args.delta_mode,
            balance_within_unit=bool(args.balance_within_unit),
            seed=int(args.seed),
        )
        with (out_dir / f"per_unit_stats_within_group_{args.unit_mode}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["unit_id", "group", "label", "n0", "n1", "mu00", "mu11", "mu01", "delta", "delta_weighted", "within_pairs", "between_pairs"])
            for s in stats_within:
                w.writerow([s.unit_id, s.group, s.label_name, s.n0, s.n1, s.mu00, s.mu11, s.mu01, s.delta, s.delta_weighted, s.within_pairs, s.between_pairs])

        # per-group summary
        from collections import defaultdict
        gp: Dict[str, List[SepStats]] = defaultdict(list)
        for s in stats_within:
            gp[s.group].append(s)

        groups_sorted = sorted(gp.keys(), key=_get_group_sort_key)
        with (out_dir / f"per_group_summary_{args.unit_mode}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "num_units_with_split", "delta_mean", "delta_median", "delta_std", "delta_min", "delta_max", "delta_pos_frac"])
            for g in groups_sorted:
                d = summarize_stats(gp[g])
                w.writerow([g, d["num_units_with_split"], d["delta_mean"], d["delta_median"], d["delta_std"], d["delta_min"], d["delta_max"], d["delta_pos_frac"]])

    # global summary json
    summary = {
        "args": vars(args),
        "num_items_used": len(feat_rows),
        "num_distinct_units": int(len({r["unit_id"] for r in feat_rows})),
        "summary_all": summarize_stats(stats_all) if stats_all else None,
        "summary_within_group": summarize_stats(stats_within) if stats_within else None,
    }
    (out_dir / "global_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # dump per-sample meta
    with (out_dir / "features_meta.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "unit_id", "base_qid", "group", "refusal", "asr", "echo", "image_path", "feat_shape", "prompt_semantic_key", "prompt_strict_key"])
        for r in feat_rows:
            feat = r.get("feat", None)
            shp = ""
            if isinstance(feat, np.ndarray):
                shp = "x".join(map(str, feat.shape))
            w.writerow([r["qid"], r["unit_id"], r["base_qid"], r["group"], r["refusal"], r["asr"], r["echo"], r["image_path"], shp, r["prompt_semantic_key"], r["prompt_strict_key"]])

    # ---------------- visualization ----------------
    PCA, LDA, Scaler = _try_import_sklearn()
    if PCA is not None:
        def _plot_subset(tag: str, group_name: Optional[str]) -> None:
            X_list, y_list = _build_Xy(feat_rows, args.label, group_filter=group_name)
            if len(X_list) < 10 or len(set(y_list)) < 2:
                return
            max_dim = max(x.shape[0] for x in X_list)
            X = np.stack([np.pad(x, (0, max_dim - x.shape[0]), constant_values=0.0) for x in X_list], axis=0)
            y = np.asarray(y_list, dtype=np.int64)

            scaler = Scaler()
            Xs = scaler.fit_transform(X)

            pca = PCA(n_components=2, random_state=0)
            X2 = pca.fit_transform(Xs)
            _save_pca_2d_plot(out_dir / f"pca_2d_{args.label}_{args.unit_mode}_{tag}.png", X2, y, f"PCA(2D) {args.feat}, {args.label}, unit={args.unit_mode}, {tag}")

            lda = LDA(n_components=1)
            z = lda.fit_transform(Xs, y).reshape(-1)
            _save_lda_1d_plot(out_dir / f"lda_1d_{args.label}_{args.unit_mode}_{tag}.png", z, y, f"LDA(1D) {args.feat}, {args.label}, unit={args.unit_mode}, {tag}")

        _plot_subset("ALL", None)

        if args.plot_per_group:
            groups = sorted({str(r.get("group", "")) for r in feat_rows}, key=_get_group_sort_key)
            for g in groups:
                _, yy = _build_Xy(feat_rows, args.label, group_filter=g)
                if len(yy) < args.min_plot_n or len(set(yy)) < 2:
                    continue
                _plot_subset(g, g)

    print(f"[ok] out_dir: {out_dir}")
    print("[ok] wrote: global_summary.json, features_meta.csv")
    if args.group_mode in ("all", "both"):
        print(f"[ok] wrote: per_unit_stats_all_{args.unit_mode}.csv")
    if args.group_mode in ("within", "both"):
        print(f"[ok] wrote: per_unit_stats_within_group_{args.unit_mode}.csv, per_group_summary_{args.unit_mode}.csv")
    print("[ok] plots saved if sklearn available.")


if __name__ == "__main__":
    main()
