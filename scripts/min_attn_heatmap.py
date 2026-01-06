# scripts/min_attn_heatmap_qwen25vl.py
# Minimal: (text + image) -> Qwen-2.5-VL-7B forward -> attention heatmap overlay
# Includes:
#  (1) FWD text->vision aggregation (mean/max over all text queries)
#  (2) GEN1 proxy: use the last input token's attention row (q = S-1),
#      which is the query used to predict the first generated token.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image
from matplotlib import cm
from transformers import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
except Exception:
    Qwen2_5_VLForConditionalGeneration = None  # type: ignore


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


def _infer_grid_thw(image_grid_thw: Optional[torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    if image_grid_thw is None or not isinstance(image_grid_thw, torch.Tensor):
        return None
    t = image_grid_thw.detach().cpu()
    if t.ndim == 3:  # [B,N,3]
        t = t[0]
    if t.ndim == 2 and t.shape[-1] == 3 and t.shape[0] >= 1:
        tt = int(t[0, 0].item())
        hh = int(t[0, 1].item())
        ww = int(t[0, 2].item())
        if tt > 0 and hh > 0 and ww > 0:
            return (tt, hh, ww)
    return None


def _get_merge_size(model) -> int:
    vc = getattr(model.config, "vision_config", None)
    for name in ["spatial_merge_size", "merge_size", "spatial_merge", "vision_merge_size"]:
        v = getattr(vc, name, None) if vc is not None else None
        if isinstance(v, int) and v > 0:
            return v
    for name in ["spatial_merge_size", "merge_size", "spatial_merge", "vision_merge_size"]:
        v = getattr(model.config, name, None)
        if isinstance(v, int) and v > 0:
            return v
    return 1


def grid_hw_from_thw(thw: Optional[Tuple[int, int, int]], merge_size: int) -> Optional[Tuple[int, int]]:
    if thw is None:
        return None
    _, h, w = thw
    ms = max(1, int(merge_size))
    return (max(1, h // ms), max(1, w // ms))


def vec_to_grid(vec: np.ndarray, grid_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    vec = vec.astype(np.float32)
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


def overlay_heatmap_on_image(img_rgb_u8: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    H, W = img_rgb_u8.shape[:2]
    heat_img = Image.fromarray((heat01 * 255.0).clip(0, 255).astype(np.uint8))
    heat_img = heat_img.resize((W, H), resample=Image.BICUBIC)
    heat01_resized = np.asarray(heat_img).astype(np.float32) / 255.0

    cmap = cm.get_cmap("inferno")
    heat_rgb = (cmap(heat01_resized)[..., :3] * 255.0).astype(np.uint8)

    overlay = (img_rgb_u8.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha)
    return overlay.clip(0, 255).astype(np.uint8)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = float(x.min()), float(x.max())
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1e-6
    return (x - vmin) / denom


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
        raise RuntimeError("Cannot find image pad token id.")

    ids_1d = input_ids[0]
    vision_mask = (ids_1d == pad_id)
    if not bool(vision_mask.any().item()):
        raise RuntimeError(f"Found pad_id={pad_id} but no positions match in input_ids.")
    return vision_mask


def build_text_mask_basic(input_ids: torch.Tensor, vision_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    ids_1d = input_ids[0]
    text_mask = ~vision_mask

    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int) and tid >= 0:
            special_ids.add(tid)
    for tid in special_ids:
        text_mask = text_mask & (ids_1d != tid)

    try:
        added = getattr(tokenizer, "additional_special_tokens", None)
        if isinstance(added, list) and added:
            add_ids = []
            for t in added:
                try:
                    i = tokenizer.convert_tokens_to_ids(t)
                    if isinstance(i, int) and i >= 0:
                        add_ids.append(i)
                except Exception:
                    pass
            for tid in set(add_ids):
                text_mask = text_mask & (ids_1d != tid)
    except Exception:
        pass

    return text_mask


def _mask_to_span(mask_1d: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    return (int(idx.min().item()), int(idx.max().item()))


def sanity_check_vision_mask(vision_mask: torch.Tensor, image_grid_thw: Optional[torch.Tensor]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    v_cnt = int(vision_mask.sum().item())
    out["vision_token_count"] = v_cnt

    thw = _infer_grid_thw(image_grid_thw)
    out["image_grid_thw"] = list(thw) if thw else None
    if thw is None:
        out["sanity_ok"] = True
        out["ratio_vs_grid"] = None
        return out

    t, h, w = thw
    expected = int(t * h * w)
    out["grid_token_estimate"] = expected
    ratio = v_cnt / max(1, expected)
    out["ratio_vs_grid"] = float(ratio)
    out["sanity_ok"] = bool(0.15 <= ratio <= 3.0)
    return out


def attention_text_to_vision(A: torch.Tensor, text_mask: torch.Tensor, vision_mask: torch.Tensor, agg: str) -> torch.Tensor:
    q_idx = torch.nonzero(text_mask, as_tuple=False).squeeze(1)
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    if q_idx.numel() == 0 or k_idx.numel() == 0:
        raise RuntimeError("Empty text_mask or vision_mask.")
    sub = A.index_select(0, q_idx).index_select(1, k_idx)  # [Q,P]
    if agg == "mean":
        return sub.mean(dim=0)
    if agg == "max":
        return sub.max(dim=0).values
    raise ValueError(f"Unknown agg={agg}")


def attention_query_to_vision(a_row: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    if k_idx.numel() == 0:
        raise RuntimeError("Empty vision_mask.")
    return a_row.index_select(0, k_idx)


def fix_len(vec: np.ndarray, P: int) -> np.ndarray:
    vec = vec.astype(np.float32)
    if vec.shape[0] == P:
        return vec
    if vec.shape[0] > P:
        return vec[:P]
    return np.pad(vec, (0, P - vec.shape[0]), constant_values=0.0)


def extract_last_query_row_from_attn_tensor(layer_attn: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    layer_attn typical shapes:
      forward: [B,H,S,S]
      some generate impl: [B,H,Q,S] with Q==S or Q==1
    Returns:
      a_row: [S] = mean_over_heads(layer_attn)[q_last]
    """
    dbg: Dict[str, Any] = {"layer_tensor_shape": list(layer_attn.shape)}
    if layer_attn.ndim != 4:
        raise ValueError(f"Expected 4D attn [B,H,Q,S], got shape={list(layer_attn.shape)}")

    t = layer_attn[0]          # [H,Q,S]
    t = t.mean(dim=0)          # [Q,S]
    Q = int(t.shape[0])
    q_last = Q - 1
    a_row = t[q_last]          # [S]
    dbg["Q"] = Q
    dbg["q_last"] = q_last
    dbg["a_row_shape"] = list(a_row.shape)
    return a_row, dbg


@torch.no_grad()
def run_once(
    model_id: str,
    image_path: str,
    text: str,
    base_out_dir: str,
    layer: int = -1,
    device: str = "cuda",
    agg: str = "mean",
    alpha: float = 0.45,
    try_generate: bool = False,
) -> Dict[str, Any]:
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in your transformers build.")

    image_path_obj = Path(image_path)
    outp = Path(base_out_dir) / image_path_obj.stem
    outp.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tok = processor.tokenizer

    torch_device = torch.device(device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(torch_device).eval()

    _set_attn_eager_if_possible(model)

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(torch_device)

    input_ids = inputs["input_ids"]
    S_in = int(input_ids.shape[1])

    vision_mask = build_vision_mask_from_image_pad(input_ids, tok).to(torch_device)
    text_mask = build_text_mask_basic(input_ids, vision_mask, tok).to(torch_device)

    v_span = _mask_to_span(vision_mask)
    t_span = _mask_to_span(text_mask)

    sanity = sanity_check_vision_mask(vision_mask, inputs.get("image_grid_thw", None))
    if not bool(sanity.get("sanity_ok", True)):
        raise RuntimeError(f"Vision mask sanity check failed: {json.dumps(sanity, ensure_ascii=False)}")

    forward_out = model(**inputs, use_cache=False, output_attentions=True, return_dict=True)
    attns_fwd = getattr(forward_out, "attentions", None)
    if attns_fwd is None:
        raise RuntimeError("Forward did not return attentions.")

    L = len(attns_fwd)
    layer_idx = layer if layer >= 0 else (L + layer)
    if not (0 <= layer_idx < L):
        raise ValueError(f"Invalid layer={layer}, resolved layer_idx={layer_idx}, total_layers={L}")

    thw = _infer_grid_thw(inputs.get("image_grid_thw", None))
    merge_size = _get_merge_size(model)
    grid_hw = grid_hw_from_thw(thw, merge_size=merge_size)

    img_rgb = np.asarray(image).astype(np.uint8)
    layer_str = f"layer{layer_idx:02d}"

    # ---------------- FWD (all text queries -> vision keys)
    A_fwd = attns_fwd[layer_idx][0].mean(dim=0)  # [S,S]
    vec_fwd = attention_text_to_vision(A_fwd, text_mask=text_mask, vision_mask=vision_mask, agg=agg)
    vec01_fwd = normalize01(vec_fwd.detach().float().cpu().numpy())
    grid_fwd = vec_to_grid(vec01_fwd, grid_hw=grid_hw)
    overlay_fwd = overlay_heatmap_on_image(img_rgb, grid_fwd, alpha=alpha)

    out_overlay_fwd = outp / f"{layer_str}_FWD_text2vision_{agg}_alpha{alpha}_overlay.png"
    Image.fromarray(overlay_fwd).save(out_overlay_fwd)
    out_raw_fwd = outp / f"{layer_str}_FWD_text2vision_{agg}_heatgrid.npy"
    np.save(out_raw_fwd, grid_fwd)

    # ---------------- GEN1 PROXY (q = last input token)
    # This corresponds to the query used to predict the next token (first generated token).
    a_row_gen1 = A_fwd[S_in - 1]  # [S]
    vec_gen1 = attention_query_to_vision(a_row_gen1, vision_mask=vision_mask).detach().float().cpu().numpy()
    P = int(vision_mask.sum().item())
    vec_gen1 = fix_len(vec_gen1, P)
    vec01_gen1 = normalize01(vec_gen1)
    grid_gen1 = vec_to_grid(vec01_gen1, grid_hw=grid_hw)
    overlay_gen1 = overlay_heatmap_on_image(img_rgb, grid_gen1, alpha=alpha)

    out_overlay_gen1 = outp / f"{layer_str}_GEN1proxy_lastq2vision_alpha{alpha}_overlay.png"
    Image.fromarray(overlay_gen1).save(out_overlay_gen1)
    out_raw_gen1 = outp / f"{layer_str}_GEN1proxy_lastq2vision_heatgrid.npy"
    np.save(out_raw_gen1, grid_gen1)

    # scalar: vision reliance mass for last query
    denom = float(a_row_gen1.sum().item()) + 1e-12
    gen1_vri = float(a_row_gen1[vision_mask].sum().item()) / denom

    gen_debug: Dict[str, Any] = {}
    gen_try_metrics: Dict[str, Any] = {}
    if try_generate:
        # Optional debugging: inspect generate-attentions structure.
        try:
            gen_out = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            att = getattr(gen_out, "attentions", None)
            gen_debug["att_type"] = str(type(att))
            gen_debug["num_steps"] = len(att) if isinstance(att, (tuple, list)) else None
            if isinstance(att, (tuple, list)) and len(att) >= 1:
                step0 = att[0]
                gen_debug["step0_type"] = str(type(step0))
                gen_debug["step0_len"] = len(step0) if isinstance(step0, (tuple, list)) else None
                if isinstance(step0, (tuple, list)) and 0 <= layer_idx < len(step0):
                    layer_t = step0[layer_idx]
                    if isinstance(layer_t, torch.Tensor):
                        a_row2, dbg2 = extract_last_query_row_from_attn_tensor(layer_t.to(torch_device))
                        v2 = attention_query_to_vision(a_row2, vision_mask).detach().float().cpu().numpy()
                        v2 = fix_len(v2, P)
                        gen_try_metrics["gen_attn_lastq_vri"] = float(a_row2[vision_mask].sum().item()) / (float(a_row2.sum().item()) + 1e-12)
                        gen_debug["parsed_layer"] = dbg2
                        # save this alt overlay too
                        v2_01 = normalize01(v2)
                        g2 = vec_to_grid(v2_01, grid_hw=grid_hw)
                        o2 = overlay_heatmap_on_image(img_rgb, g2, alpha=alpha)
                        p2 = outp / f"{layer_str}_GEN1_generate_lastq2vision_alpha{alpha}_overlay.png"
                        Image.fromarray(o2).save(p2)
                        gen_try_metrics["gen_overlay"] = str(p2)
        except Exception as e:
            gen_try_metrics["gen_error"] = f"{type(e).__name__}: {str(e)}"

    info: Dict[str, Any] = {
        "image_subdir": str(outp),
        "model_id": model_id,
        "device": str(torch_device),
        "dtype": str(dtype),
        "layer_idx": layer_idx,
        "seq_len_input": S_in,
        "vision_span": list(v_span) if v_span else None,
        "text_span": list(t_span) if t_span else None,
        "vision_token_count": int(vision_mask.sum().item()),
        "text_token_count": int(text_mask.sum().item()),
        "image_grid_thw": sanity.get("image_grid_thw", None),
        "grid_token_estimate": sanity.get("grid_token_estimate", None),
        "ratio_vs_grid": sanity.get("ratio_vs_grid", None),
        "merge_size_used": int(merge_size),
        "grid_hw_from_processor": list(grid_hw) if grid_hw else None,
        "fwd_overlay": str(out_overlay_fwd),
        "fwd_heatgrid_npy": str(out_raw_fwd),
        "gen1_overlay": str(out_overlay_gen1),
        "gen1_heatgrid_npy": str(out_raw_gen1),
        "agg_fwd": agg,
        "alpha": alpha,
        "gen1_metrics": {"gen1_vri": float(gen1_vri), "gen1_vec_len": int(P)},
        "gen_debug": gen_debug,
        "gen_try_metrics": gen_try_metrics,
    }

    (outp / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="/data2/wangjingsong/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out_dir", default="outputs/vis_attn_qwen25vl")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--try_generate", action="store_true", help="debug: also parse generate() attentions and save alt overlay")
    args = ap.parse_args()

    info = run_once(
        model_id=args.model_id,
        image_path=args.image,
        text=args.text,
        base_out_dir=args.out_dir,
        layer=args.layer,
        device=args.device,
        agg=args.agg,
        alpha=args.alpha,
        try_generate=args.try_generate,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
