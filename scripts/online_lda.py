# scripts/online_lda.py
# -*- coding: utf-8 -*-
"""
Cross-carrier online evaluation with two LDA bundles (trained on different carriers),
using a unified feature: FWD attention (text -> vision).

Key fix:
- Your dataset has true text-only items where image_path == null (e.g., group A2).
  FWD(text->vision) cannot be computed without vision tokens.
- Therefore, fit2 supports neutral placeholders:
    * text-only carrier: substitute a neutral image path (e.g., white.png)
    * img-only  carrier: substitute a neutral text (e.g., "OK") if prompt is empty

This makes "carrier" refer to semantic carrier in your dataset, while placeholders are
engineering scaffolds to satisfy model I/O.

Subcommands:
  fit2  : fit TEXT-bundle (trained on text-only carrier) and IMG-bundle (trained on img-only carrier)
  xeval : interactive cross-carrier eval (opposite-bundle prediction), accepts manual labels (-1/0/1)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
except Exception:
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore


# ---------------------------
# Basic utils
# ---------------------------
def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _get_first_existing_token_id(tokenizer, candidates: List[str]) -> Optional[int]:
    for s in candidates:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0:
                return tid
        except Exception:
            pass
    return None


def _mask_span(mask_1d: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    return (int(idx.min().item()), int(idx.max().item()))


# ---------------------------
# Judge flags + strict filtering
# ---------------------------
def _extract_qwen_judge(obj: Dict[str, Any]) -> Dict[str, Any]:
    j = obj.get("qwen_judge") or {}
    is_refusal = bool(j.get("is_refusal", False))
    echo = bool(j.get("echo_only", False))
    is_safe = bool(j.get("is_safe", False))
    is_asr = bool(j.get("is_attack_success", False)) and (not echo)
    tox = float(j.get("toxicity", 0.0) or 0.0)
    return {
        "is_refusal": int(is_refusal),
        "echo": int(echo),
        "is_safe": int(is_safe),
        "asr": int(is_asr),
        "toxicity": tox,
    }


def _strict_bin_from_flags(flags: Dict[str, Any]) -> Optional[int]:
    """
    strict_safe   -> 0
    strict_unsafe -> 1
    else -> None (drop)
    """
    is_refusal = int(flags["is_refusal"])
    echo = int(flags["echo"])
    is_safe = int(flags["is_safe"])
    if echo == 1:
        return None
    if is_refusal == 1 and is_safe == 1:
        return 0
    if is_refusal == 0 and is_safe == 0:
        return 1
    return None


# ---------------------------
# Carrier inference
# ---------------------------
def _infer_carrier(it: Dict[str, Any], carrier_field: str = "") -> str:
    """
    Returns one of: "text_only", "img_only", "mixed".
    If carrier_field exists, use it. Else infer from (image_path, user_prompt).
    """
    if carrier_field:
        v = it.get(carrier_field, None)
        if isinstance(v, str) and v.strip():
            s = v.strip().lower()
            if s in ("text_only", "txt_only", "text"):
                return "text_only"
            if s in ("img_only", "image_only", "image"):
                return "img_only"
            if s in ("mixed", "txt_img", "text_image"):
                return "mixed"

    img = it.get("image_path", None)
    prompt = str(it.get("user_prompt", it.get("prompt", "")) or "").strip()

    if img is None:
        return "text_only"
    if prompt == "":
        return "img_only"
    return "mixed"


# ---------------------------
# Attention extraction (FWD only)
# ---------------------------
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
def extract_fwd_grid(
    model,
    processor,
    device: str,
    image_path: Optional[str],
    prompt_text: str,
    layer_idx: int,
    agg_fwd: str,
    merge_size: int,
) -> Dict[str, Any]:
    tok = processor.tokenizer
    if image_path is None:
        return {"ok": False, "reason": "no_image", "fwd_grid": None, "meta": {}}

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], return_tensors="pt")

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
        return {"ok": False, "reason": "no_attentions", "fwd_grid": None, "meta": {}}

    input_ids = inputs["input_ids"]
    vision_mask = build_vision_mask_from_image_pad(input_ids, tok)
    text_mask = build_text_mask_basic(input_ids, vision_mask, tok)

    if int(text_mask.sum().item()) <= 0:
        return {"ok": False, "reason": "no_text_tokens", "fwd_grid": None, "meta": {}}

    L = len(attns)
    li = layer_idx if layer_idx >= 0 else (L + layer_idx)
    if not (0 <= li < L):
        return {"ok": False, "reason": f"bad_layer:{layer_idx}->{li}/{L}", "fwd_grid": None, "meta": {}}

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

    fwd_grid = vec_to_grid(fwd_vec.detach().cpu().numpy(), grid_hw=grid_hw)
    meta = {
        "seq_len": int(input_ids.shape[1]),
        "vision_span": list(_mask_span(vision_mask) or []),
        "text_span": list(_mask_span(text_mask) or []),
        "vision_token_count": int(vision_mask.sum().item()),
        "text_token_count": int(text_mask.sum().item()),
        "image_grid_thw": list(thw) if thw else None,
        "grid_hw": list(grid_hw) if grid_hw else None,
        "merge_size": int(merge_size),
        "layer_resolved": int(li),
        "agg_fwd": str(agg_fwd),
    }
    return {"ok": True, "reason": "ok", "fwd_grid": fwd_grid, "meta": meta}


# ---------------------------
# LDA bundle
# ---------------------------
@dataclass
class LDABundle:
    w: np.ndarray
    b: float
    mean: np.ndarray
    scale: np.ndarray
    max_dim: int
    tau: float
    meta: Dict[str, Any]


def _standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - mean) / (scale + 1e-12)


def _pad_to_dim(x: np.ndarray, D: int) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32)
    if x.shape[0] == D:
        return x
    if x.shape[0] > D:
        return x[:D]
    return np.pad(x, (0, D - x.shape[0]), constant_values=0.0)


def _try_import_sklearn():
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        return LinearDiscriminantAnalysis, StandardScaler
    except Exception:
        return None, None


def _fit_lda_bundle(
    X_list: List[np.ndarray],
    y_list: List[int],
    tau_mode: str,
    tau_q: float,
    meta: Dict[str, Any],
) -> LDABundle:
    LDA, Scaler = _try_import_sklearn()
    if LDA is None:
        raise RuntimeError("scikit-learn not available. Please install sklearn in this env.")

    max_dim = max(x.reshape(-1).shape[0] for x in X_list)
    X = np.stack([_pad_to_dim(x, max_dim) for x in X_list], axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    scaler = Scaler()
    Xs = scaler.fit_transform(X)

    lda = LDA(n_components=1)
    lda.fit(Xs, y)

    w = lda.coef_.reshape(-1).astype(np.float32)
    b = float(lda.intercept_.reshape(-1)[0])

    scores = (Xs @ w) + b
    if tau_mode == "quantile":
        tau = float(np.quantile(np.abs(scores), tau_q))
    elif tau_mode == "fixed":
        tau = float(meta.get("tau_fixed", 0.0))
    else:
        raise ValueError(f"unknown tau_mode={tau_mode}")

    return LDABundle(
        w=w,
        b=b,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        max_dim=int(max_dim),
        tau=float(tau),
        meta=meta,
    )


def save_bundle_npz(path: Path, bundle: LDABundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        w=bundle.w,
        b=np.asarray([bundle.b], dtype=np.float32),
        mean=bundle.mean,
        scale=bundle.scale,
        max_dim=np.asarray([bundle.max_dim], dtype=np.int64),
        tau=np.asarray([bundle.tau], dtype=np.float32),
        meta_json=np.asarray([json.dumps(bundle.meta, ensure_ascii=False)], dtype=object),
    )


def load_bundle_npz(path: Path) -> LDABundle:
    d = np.load(str(path), allow_pickle=True)
    meta = json.loads(str(d["meta_json"][0]))
    return LDABundle(
        w=d["w"].astype(np.float32),
        b=float(d["b"][0]),
        mean=d["mean"].astype(np.float32),
        scale=d["scale"].astype(np.float32),
        max_dim=int(d["max_dim"][0]),
        tau=float(d["tau"][0]),
        meta=meta,
    )


def predict_3way(bundle: LDABundle, feat_grid: np.ndarray) -> Tuple[float, int]:
    x = _pad_to_dim(feat_grid.reshape(-1), bundle.max_dim)
    xs = _standardize(x, bundle.mean, bundle.scale)
    score = float(xs @ bundle.w + bundle.b)
    tau = float(bundle.tau)
    if score >= tau:
        return score, +1
    if score <= -tau:
        return score, -1
    return score, 0


# ---------------------------
# FIT2: with explicit placeholders
# ---------------------------
def _filter_items_by_carrier(
    judged: List[Dict[str, Any]],
    *,
    target_model: str,
    only_perturb0: bool,
    drop_echo: bool,
    carrier_want: str,
    carrier_field: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in judged:
        if str(it.get("model", "")) != target_model:
            continue
        if only_perturb0:
            p = float(it.get("perturb_p", 0.0) or 0.0)
            if abs(p) > 1e-12:
                continue

        carrier = _infer_carrier(it, carrier_field=carrier_field)
        if carrier != carrier_want:
            continue

        flags = _extract_qwen_judge(it)
        if drop_echo and int(flags["echo"]) == 1:
            continue

        strict_bin = _strict_bin_from_flags(flags)
        if strict_bin is None:
            continue

        out.append(it)
    return out


def _resolve_placeholders_for_item(
    it: Dict[str, Any],
    *,
    carrier: str,
    text_only_placeholder_image: Optional[str],
    img_only_placeholder_text: str,
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """
    Returns: (image_path, prompt_text, placeholder_meta)
    """
    img = it.get("image_path", None)
    prompt = str(it.get("user_prompt", it.get("prompt", "")) or "")

    meta: Dict[str, Any] = {"carrier": carrier, "used_placeholder_image": False, "used_placeholder_text": False}

    if carrier == "text_only":
        # dataset has no image; must inject placeholder image to compute FWD
        if img is None:
            if not text_only_placeholder_image:
                return None, prompt, {**meta, "placeholder_error": "need_text_only_placeholder_image"}
            img = text_only_placeholder_image
            meta["used_placeholder_image"] = True

    if carrier == "img_only":
        # dataset has empty user_prompt; ensure minimal text so text_mask is non-empty
        if str(prompt).strip() == "":
            prompt = img_only_placeholder_text
            meta["used_placeholder_text"] = True

    # mixed: as is
    return (None if img is None else str(img)), str(prompt), meta


def cmd_fit2(args) -> None:
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in this transformers build.")

    judged = _load_json_or_jsonl(Path(args.judged_json))

    # placeholders sanity
    text_only_ph_img = args.text_only_placeholder_image.strip() if args.text_only_placeholder_image else ""
    if text_only_ph_img:
        if not Path(text_only_ph_img).exists():
            raise RuntimeError(f"--text_only_placeholder_image not found: {text_only_ph_img}")

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

    text_items = _filter_items_by_carrier(
        judged,
        target_model=args.target_model,
        only_perturb0=args.only_perturb0,
        drop_echo=args.drop_echo,
        carrier_want="text_only",
        carrier_field=args.carrier_field,
    )
    img_items = _filter_items_by_carrier(
        judged,
        target_model=args.target_model,
        only_perturb0=args.only_perturb0,
        drop_echo=args.drop_echo,
        carrier_want="img_only",
        carrier_field=args.carrier_field,
    )

    print(json.dumps(
        {
            "found": {
                "text_only_items_after_filters": len(text_items),
                "img_only_items_after_filters": len(img_items),
            },
            "placeholders": {
                "text_only_placeholder_image": (text_only_ph_img or None),
                "img_only_placeholder_text": args.img_only_placeholder_text,
            },
        },
        indent=2,
        ensure_ascii=False
    ))

    from tqdm import tqdm

    def extract_for_items(items: List[Dict[str, Any]], carrier: str) -> Tuple[List[np.ndarray], List[int], Dict[str, int], Dict[str, int]]:
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        cc = {0: 0, 1: 0}
        plc = {"used_placeholder_image": 0, "used_placeholder_text": 0, "skipped_no_image": 0, "skipped_no_text": 0}

        cache: Dict[str, Dict[str, Any]] = {}

        for it in tqdm(items, desc=f"extract[{carrier}]", unit="item", ncols=100):
            qid = str(it.get("qid", "")) or f"noqid_{len(X_list)}"

            img_path, prompt_text, ph_meta = _resolve_placeholders_for_item(
                it,
                carrier=carrier,
                text_only_placeholder_image=(text_only_ph_img or None),
                img_only_placeholder_text=args.img_only_placeholder_text,
            )
            if img_path is None:
                plc["skipped_no_image"] += 1
                continue

            if ph_meta.get("used_placeholder_image"):
                plc["used_placeholder_image"] += 1
            if ph_meta.get("used_placeholder_text"):
                plc["used_placeholder_text"] += 1

            cache_key = f"{qid}::{img_path}::{hash(prompt_text)}::{args.layer}::{args.agg_fwd}::{args.merge_size}"
            if cache_key in cache:
                rec = cache[cache_key]
            else:
                rec = extract_fwd_grid(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=img_path,
                    prompt_text=prompt_text,
                    layer_idx=args.layer,
                    agg_fwd=args.agg_fwd,
                    merge_size=args.merge_size,
                )
                cache[cache_key] = rec

            if not rec.get("ok", False):
                if rec.get("reason") == "no_text_tokens":
                    plc["skipped_no_text"] += 1
                continue

            fwd_grid = rec.get("fwd_grid", None)
            if not isinstance(fwd_grid, np.ndarray):
                continue

            flags = _extract_qwen_judge(it)
            y = _strict_bin_from_flags(flags)
            if y is None:
                continue

            X_list.append(fwd_grid)
            y_list.append(int(y))
            cc[int(y)] += 1

        return X_list, y_list, {"strict_safe(0)": cc[0], "strict_unsafe(1)": cc[1]}, plc

    # extract
    X_text, y_text, cc_text, plc_text = extract_for_items(text_items, "text_only")
    X_img, y_img, cc_img, plc_img = extract_for_items(img_items, "img_only")

    print(json.dumps(
        {
            "usable": {
                "text_only": {"n": len(X_text), "classes": sorted(list(set(y_text))), "class_counts": cc_text, "placeholders": plc_text},
                "img_only": {"n": len(X_img), "classes": sorted(list(set(y_img))), "class_counts": cc_img, "placeholders": plc_img},
            },
            "note": "If text_only n=0, you must provide a real placeholder image via --text_only_placeholder_image.",
        },
        indent=2,
        ensure_ascii=False
    ))

    if len(X_text) < 50 or len(set(y_text)) < 2:
        raise RuntimeError(f"TEXT bundle not enough usable features: n={len(X_text)}, classes={set(y_text)}")
    if len(X_img) < 50 or len(set(y_img)) < 2:
        raise RuntimeError(f"IMG bundle not enough usable features: n={len(X_img)}, classes={set(y_img)}")

    meta_common = {
        "feat": "fwd_text_to_vision",
        "label": "strict_bin",
        "model_id": str(args.model_id),
        "target_model": str(args.target_model),
        "layer": int(args.layer),
        "agg_fwd": str(args.agg_fwd),
        "merge_size": int(args.merge_size),
        "only_perturb0": bool(args.only_perturb0),
        "drop_echo": bool(args.drop_echo),
        "tau_mode": str(args.tau_mode),
        "tau_q": float(args.tau_q),
        "seed": int(args.seed),
        "placeholders": {
            "text_only_placeholder_image": (text_only_ph_img or None),
            "img_only_placeholder_text": args.img_only_placeholder_text,
        },
    }

    meta_text = dict(meta_common)
    meta_text.update({"train_carrier": "text_only", "num_train": int(len(X_text)), "class_counts": cc_text})
    meta_img = dict(meta_common)
    meta_img.update({"train_carrier": "img_only", "num_train": int(len(X_img)), "class_counts": cc_img})

    bundle_text = _fit_lda_bundle(X_text, y_text, tau_mode=args.tau_mode, tau_q=args.tau_q, meta=meta_text)
    bundle_img = _fit_lda_bundle(X_img, y_img, tau_mode=args.tau_mode, tau_q=args.tau_q, meta=meta_img)

    save_bundle_npz(Path(args.out_text_bundle), bundle_text)
    save_bundle_npz(Path(args.out_img_bundle), bundle_img)

    print(json.dumps(
        {
            "ok": True,
            "out_text_bundle": str(args.out_text_bundle),
            "out_img_bundle": str(args.out_img_bundle),
            "text_tau": bundle_text.tau,
            "img_tau": bundle_img.tau,
        },
        indent=2,
        ensure_ascii=False
    ))


# ---------------------------
# Xeval (interactive) — unchanged core idea
# ---------------------------
def _read_multiline(prompt: str) -> str:
    print(prompt)
    print("Enter lines, finish with a single line: END")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def _parse_input_mode(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    if s in ("text", "text_only", "txt", "a"):
        return "text_only"
    if s in ("img", "image", "img_only", "b"):
        return "img_only"
    if s in ("mixed", "txt_img", "ab"):
        return "mixed"
    return None


def cmd_xeval(args) -> None:
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in this transformers build.")

    bundle_text = load_bundle_npz(Path(args.text_bundle))
    bundle_img = load_bundle_npz(Path(args.img_bundle))

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

    n_used = 0
    n_correct = 0
    logs: List[Dict[str, Any]] = []

    while True:
        img_path = input("\n[round] image_path (or 'exit'): ").strip()
        if img_path.lower() == "exit":
            break
        if not img_path:
            print("[warn] empty image path; retry.")
            continue
        if not Path(img_path).exists():
            print(f"[warn] not found: {img_path}")
            continue

        text = _read_multiline("[round] text:")
        mode_raw = input("[round] input_mode (text_only/img_only/mixed): ").strip()
        mode = _parse_input_mode(mode_raw)
        if mode is None:
            print("[warn] invalid input_mode; retry.")
            continue

        # cross-carrier selection
        # text_only input -> use IMG-trained bundle; img_only input -> use TEXT-trained bundle
        if mode == "text_only":
            bun = bundle_img
            bun_name = "IMG_bundle(opposite)"
        elif mode == "img_only":
            bun = bundle_text
            bun_name = "TEXT_bundle(opposite)"
        else:
            bun = None
            bun_name = "BOTH"

        # compute fwd attention
        layer = int(bundle_text.meta.get("layer", -1))
        agg = str(bundle_text.meta.get("agg_fwd", "max"))
        merge_size = int(bundle_text.meta.get("merge_size", 2))
        rec = extract_fwd_grid(
            model=model,
            processor=processor,
            device=device,
            image_path=img_path,
            prompt_text=text,
            layer_idx=layer,
            agg_fwd=agg,
            merge_size=merge_size,
        )

        if not rec.get("ok", False) or not isinstance(rec.get("fwd_grid", None), np.ndarray):
            print(f"[pred] SKIP (cannot compute FWD): reason={rec.get('reason')}")
            logs.append({"ok": False, "reason": rec.get("reason"), "input_mode": mode, "image": img_path})
            continue

        feat = rec["fwd_grid"]

        if mode == "mixed":
            s1, p1 = predict_3way(bundle_text, feat)
            s2, p2 = predict_3way(bundle_img, feat)
            score = 0.5 * (s1 + s2)
            tau = float(max(bundle_text.tau, bundle_img.tau))
            if score >= tau:
                pred = +1
            elif score <= -tau:
                pred = -1
            else:
                pred = 0
            pred_msg = {
                "used_bundle": "BOTH(avg)",
                "text_bundle": {"score": float(s1), "pred": int(p1), "tau": float(bundle_text.tau)},
                "img_bundle": {"score": float(s2), "pred": int(p2), "tau": float(bundle_img.tau)},
                "avg": {"score": float(score), "pred": int(pred), "tau": float(tau)},
            }
        else:
            score, pred = predict_3way(bun, feat)
            pred_msg = {"used_bundle": bun_name, "score": float(score), "pred": int(pred), "tau": float(bun.tau)}

        print("[pred]", json.dumps(pred_msg, ensure_ascii=False))

        lab_raw = input("[label] manual label in {-1,0,1}, 9=skip: ").strip()
        if lab_raw == "9":
            logs.append({"ok": True, "skipped_label": True, "pred": int(pred), "input_mode": mode, "image": img_path})
            continue
        try:
            y_true = int(lab_raw)
        except Exception:
            print("[warn] invalid label; skip.")
            continue
        if y_true not in (-1, 0, 1):
            print("[warn] label must be -1/0/1; skip.")
            continue

        n_used += 1
        correct = int(y_true == int(pred))
        n_correct += correct
        acc = n_correct / n_used
        print(f"[acc] used={n_used}, correct={n_correct}, acc={acc:.4f}")

        logs.append(
            {
                "ok": True,
                "input_mode": mode,
                "image": img_path,
                "pred": int(pred),
                "true": int(y_true),
                "correct": bool(correct),
                "pred_detail": pred_msg,
                "attn_meta": rec.get("meta", {}),
            }
        )

    acc = (n_correct / n_used) if n_used > 0 else float("nan")
    report = {"ok": True, "used": n_used, "correct": n_correct, "accuracy": acc}
    print("\n[final]", json.dumps(report, indent=2, ensure_ascii=False))

    if args.save_log:
        p = Path(args.save_log)
        _safe_mkdir(p.parent)
        p.write_text(json.dumps({"report": report, "logs": logs}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[final] saved log to: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- fit2 ----
    ap_fit = sub.add_parser("fit2")
    ap_fit.add_argument("--judged_json", required=True)
    ap_fit.add_argument("--model_id", required=True)
    ap_fit.add_argument("--out_text_bundle", required=True)
    ap_fit.add_argument("--out_img_bundle", required=True)
    ap_fit.add_argument("--target_model", default="qwen25_vl_7b")

    ap_fit.add_argument("--device", default="cuda")
    ap_fit.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap_fit.add_argument("--layer", type=int, default=-1)
    ap_fit.add_argument("--agg_fwd", choices=["mean", "max"], default="max")
    ap_fit.add_argument("--merge_size", type=int, default=2)

    ap_fit.add_argument("--only_perturb0", action="store_true")
    ap_fit.add_argument("--drop_echo", action="store_true")

    ap_fit.add_argument("--tau_mode", default="quantile", choices=["quantile", "fixed"])
    ap_fit.add_argument("--tau_q", type=float, default=0.80)
    ap_fit.add_argument("--seed", type=int, default=0)

    ap_fit.add_argument("--carrier_field", default="", help="optional explicit carrier field in judged items")

    # NEW: placeholders
    ap_fit.add_argument(
        "--text_only_placeholder_image",
        default="",
        help="neutral image path used when carrier=text_only and image_path is null (required to compute FWD)",
    )
    ap_fit.add_argument(
        "--img_only_placeholder_text",
        default="OK",
        help="neutral text used when carrier=img_only but user_prompt is empty (to ensure text tokens exist)",
    )

    # ---- xeval ----
    ap_x = sub.add_parser("xeval")
    ap_x.add_argument("--text_bundle", required=True)
    ap_x.add_argument("--img_bundle", required=True)
    ap_x.add_argument("--model_id", required=True)
    ap_x.add_argument("--device", default="cuda")
    ap_x.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap_x.add_argument("--save_log", default="", help="optional path to save json log")

    args = ap.parse_args()
    if args.cmd == "fit2":
        cmd_fit2(args)
    elif args.cmd == "xeval":
        cmd_xeval(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
