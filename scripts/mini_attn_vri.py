# scripts/min_attn_heatmap_qwen25vl.py
# Minimal: (text + image) -> Qwen-2.5-VL-7B forward (no generation) -> attention heatmap overlay
#
# Key fixes vs your version:
# 1) Do NOT guess vision span by <|vision_start|> ...; instead build vision_mask from image_pad token positions.
# 2) Do NOT use a single q_idx (often special token). Aggregate attention from ALL text tokens -> vision tokens.
# 3) Sanity-check vision_token_count against image_grid_thw (approx).
# 4) Save extra debug: token counts, vision/text spans, top tokens around boundaries.

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
    # Best-effort: force eager attention so output_attentions works reliably
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

    # Also disable SDPA flash/mem-efficient where available
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _infer_grid_hw_from_image_grid_thw(image_grid_thw: Optional[torch.Tensor]) -> Optional[Tuple[int, int, int]]:
    """
    Return (t,h,w) from processor-provided image_grid_thw (best-effort).
    Qwen-VL processors often output image_grid_thw: [num_images, 3] = (t,h,w) or [B,N,3].
    """
    if image_grid_thw is None:
        return None
    t = image_grid_thw
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().cpu()

    if t.ndim == 3:  # [B, N, 3]
        t = t[0]
    if t.ndim == 2 and t.shape[-1] == 3 and t.shape[0] >= 1:
        tt = int(t[0, 0].item())
        hh = int(t[0, 1].item())
        ww = int(t[0, 2].item())
        if tt > 0 and hh > 0 and ww > 0:
            return (tt, hh, ww)
    return None


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

    # fallback near-square
    h = int(math.sqrt(P)) or 1
    w = int(math.ceil(P / h))
    G = h * w
    if P < G:
        vec = np.pad(vec, (0, G - P), constant_values=0.0)
    else:
        vec = vec[:G]
    return vec.reshape(h, w)


def overlay_heatmap_on_image(img_rgb_u8: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb_u8: [H,W,3] uint8
    heat01:     [Hg,Wg] float in [0,1] (will be resized to [H,W])
    """
    H, W = img_rgb_u8.shape[:2]
    heat_img = Image.fromarray((heat01 * 255.0).clip(0, 255).astype(np.uint8))
    heat_img = heat_img.resize((W, H), resample=Image.BICUBIC)
    heat01_resized = np.asarray(heat_img).astype(np.float32) / 255.0

    cmap = cm.get_cmap("inferno")
    heat_rgb = (cmap(heat01_resized)[..., :3] * 255.0).astype(np.uint8)

    overlay = (img_rgb_u8.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha)
    return overlay.clip(0, 255).astype(np.uint8)


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
    """
    Build vision mask from repeated image-pad placeholder tokens.
    Works for many Qwen2.5-VL variants where vision tokens are represented as repeated <|image_pad|>.
    """
    # common placeholders
    pad_id = _get_first_existing_token_id(
        tokenizer,
        ["<|image_pad|>", "<image_pad>", "<|vision_pad|>", "<vision_pad>", "<|image|>"],
    )
    if pad_id is None:
        raise RuntimeError("Cannot find image pad token id. Your tokenizer may use different vision placeholders.")

    ids_1d = input_ids[0]
    vision_mask = (ids_1d == pad_id)

    # If it's all-false, fail fast
    if not bool(vision_mask.any().item()):
        raise RuntimeError(
            f"Found pad_id={pad_id} but no positions match in input_ids. "
            "Your model may not use repeated <|image_pad|> placeholders."
        )
    return vision_mask  # [S] bool


def build_text_mask_basic(input_ids: torch.Tensor, vision_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Minimal text mask:
      - everything that's NOT vision token
      - exclude obvious special tokens (bos/eos/pad) if present
    """
    ids_1d = input_ids[0]
    text_mask = ~vision_mask

    # Remove pad/bos/eos if known
    special_ids = set()
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        tid = getattr(tokenizer, attr, None)
        if isinstance(tid, int) and tid >= 0:
            special_ids.add(tid)
    if special_ids:
        for tid in special_ids:
            text_mask = text_mask & (ids_1d != tid)

    # Also remove any "added special tokens" if tokenizer exposes them
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

    return text_mask  # [S] bool


def _mask_to_span(mask_1d: torch.Tensor) -> Optional[Tuple[int, int]]:
    idx = torch.nonzero(mask_1d, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return None
    return (int(idx.min().item()), int(idx.max().item()))


def sanity_check_vision_mask(
    vision_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """
    Compare vision token count with grid tokens ~ (t*h*w).
    Not necessarily equal (some implementations compress), but must be same order of magnitude.
    """
    out: Dict[str, Any] = {}
    v_cnt = int(vision_mask.sum().item())
    out["vision_token_count"] = v_cnt

    thw = _infer_grid_hw_from_image_grid_thw(image_grid_thw)
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
    ok = (0.2 <= ratio <= 2.0)
    out["sanity_ok"] = bool(ok)
    return out


def attention_text_to_vision(
    A: torch.Tensor,  # [S,S] head-avg
    text_mask: torch.Tensor,  # [S]
    vision_mask: torch.Tensor,  # [S]
    agg: str = "mean",
) -> torch.Tensor:
    """
    Compute vector over vision tokens:
      vec[p] = agg_{q in text_tokens} A[q, vision_tokens[p]]
    """
    q_idx = torch.nonzero(text_mask, as_tuple=False).squeeze(1)
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    if q_idx.numel() == 0 or k_idx.numel() == 0:
        raise RuntimeError("Empty text_mask or vision_mask.")

    sub = A.index_select(0, q_idx).index_select(1, k_idx)  # [Q,P]
    if agg == "mean":
        return sub.mean(dim=0)  # [P]
    if agg == "max":
        return sub.max(dim=0).values
    raise ValueError(f"Unknown agg={agg}")


def save_debug_tokens(
    out_dir: Path,
    tokenizer,
    input_ids: torch.Tensor,
    vision_span: Optional[Tuple[int, int]],
    text_span: Optional[Tuple[int, int]],
    window: int = 32,
) -> str:
    ids = input_ids[0].detach().cpu().tolist()
    S = len(ids)

    def decode_range(a: int, b: int) -> List[str]:
        a = max(0, min(a, S))
        b = max(0, min(b, S))
        toks = tokenizer.convert_ids_to_tokens(ids[a:b])
        return toks

    dbg: Dict[str, Any] = {"seq_len": S, "vision_span": vision_span, "text_span": text_span}
    if vision_span is not None:
        v0, v1 = vision_span
        dbg["vision_left_ctx"] = decode_range(v0 - window, v0 + window)
        dbg["vision_right_ctx"] = decode_range(v1 - window, v1 + window)
    if text_span is not None:
        t0, t1 = text_span
        dbg["text_left_ctx"] = decode_range(t0 - window, t0 + window)
        dbg["text_right_ctx"] = decode_range(t1 - window, t1 + window)

    p = out_dir / "debug_tokens.json"
    p.write_text(json.dumps(dbg, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


@torch.no_grad()
def run_once(
    model_id: str,
    image_path: str,
    text: str,
    base_out_dir: str,        # 改为 base_out_dir，只作为根目录
    layer: int = -1,
    device: str = "cuda",
    agg: str = "mean",
    alpha: float = 0.45,
) -> Dict[str, Any]:
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in your transformers build.")

    # ---------- 新增：根据输入图像路径创建专用子文件夹 ----------
    image_path_obj = Path(image_path)
    image_stem = image_path_obj.stem  # 文件名去掉扩展名
    outp = Path(base_out_dir) / image_stem
    outp.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # 可选：取消注释以强制更高分辨率
    # processor.image_processor.min_pixels = 256 * 28 * 28
    # processor.image_processor.max_pixels = 1280 * 28 * 28
    # processor.image_processor.size["shortest_edge"] = 384

    tok = processor.tokenizer

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device).eval()

    _set_attn_eager_if_possible(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    outputs = model(
        **inputs,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

    attns = getattr(outputs, "attentions", None)
    if attns is None:
        raise RuntimeError("Model did not return attentions. Try forcing eager attention or upgrading transformers.")

    input_ids = inputs["input_ids"]
    S = int(input_ids.shape[1])

    vision_mask = build_vision_mask_from_image_pad(input_ids, tok)
    text_mask = build_text_mask_basic(input_ids, vision_mask, tok)

    v_span = _mask_to_span(vision_mask)
    t_span = _mask_to_span(text_mask)

    sanity = sanity_check_vision_mask(vision_mask, inputs.get("image_grid_thw", None))
    if not bool(sanity.get("sanity_ok", True)):
        dbg_path = save_debug_tokens(outp, tok, input_ids, v_span, t_span)
        raise RuntimeError(
            f"Vision mask sanity check failed. Details: {json.dumps(sanity, ensure_ascii=False)}\n"
            f"Saved debug tokens to: {dbg_path}"
        )

    L = len(attns)
    layer_idx = layer if layer >= 0 else (L + layer)
    if not (0 <= layer_idx < L):
        raise ValueError(f"Invalid layer={layer}, resolved layer_idx={layer_idx}, total_layers={L}")

    A = attns[layer_idx][0].mean(dim=0)  # [S,S] head-averaged

    vec = attention_text_to_vision(A, text_mask=text_mask, vision_mask=vision_mask, agg=agg)
    vec = vec.detach().float().cpu().numpy()

    # 归一化
    vmin, vmax = vec.min(), vec.max()
    denom = (vmax - vmin) if (vmax - vmin) > 1e-6 else 1e-6
    vec01 = (vec - vmin) / denom

    # grid shape (merged)
    thw = _infer_grid_hw_from_image_grid_thw(inputs.get("image_grid_thw", None))
    MERGE_SIZE = 2
    if thw:
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        grid_hw = (h // MERGE_SIZE, w // MERGE_SIZE)
    else:
        grid_hw = None

    grid = vec_to_grid(vec01, grid_hw=grid_hw)

    img_rgb = np.asarray(image).astype(np.uint8)
    overlay = overlay_heatmap_on_image(img_rgb, grid, alpha=alpha)

    # ---------- 修改：保存路径使用 layer 编号，便于批量查看 ----------
    layer_str = f"layer{layer_idx:02d}"
    out_overlay = outp / f"{layer_str}_text2vision_{agg}_alpha{alpha}_overlay.png"
    Image.fromarray(overlay).save(out_overlay)

    out_raw = outp / f"{layer_str}_text2vision_{agg}_heatgrid.npy"
    np.save(out_raw, grid)
    # -----------------------------------------------------------------

    dbg_tokens_path = save_debug_tokens(outp, tok, input_ids, v_span, t_span)

    return {
        "image_subdir": str(outp),
        "out_overlay": str(out_overlay),
        "out_heatgrid_npy": str(out_raw),
        "debug_tokens_json": dbg_tokens_path,
        "layer_idx": layer_idx,
        "seq_len": S,
        "vision_span": list(v_span) if v_span else None,
        "text_span": list(t_span) if t_span else None,
        "vision_token_count": int(vision_mask.sum().item()),
        "text_token_count": int(text_mask.sum().item()),
        "image_grid_thw": sanity.get("image_grid_thw", None),
        "grid_token_estimate": sanity.get("grid_token_estimate", None),
        "ratio_vs_grid": sanity.get("ratio_vs_grid", None),
        "grid_hw_from_processor": list(grid_hw) if grid_hw else None,
        "grid_shape_used": list(grid.shape),
        "agg": agg,
        "alpha": alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="/data2/wangjingsong/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--text", required=True, help="Text prompt")
    ap.add_argument("--out_dir", default="outputs/vis_attn_qwen25vl", help="Root directory for all outputs")
    ap.add_argument("--layer", type=int, default=-1, help="Which layer to visualize; -1 = last layer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--agg", default="mean", choices=["mean", "max"])
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    info = run_once(
        model_id=args.model_id,
        image_path=args.image,
        text=args.text,
        base_out_dir=args.out_dir,   # 现在是根目录
        layer=args.layer,
        device=args.device,
        agg=args.agg,
        alpha=args.alpha,
    )
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()