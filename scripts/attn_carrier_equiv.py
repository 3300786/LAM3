# src/metrics/attn_carrier_equiv_qwen25vl.py
from __future__ import annotations

import argparse
import json
import math
import os
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
# Utils
# ---------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))


def _get_group_sort_key(g: str) -> Tuple[int, str]:
    # A1,A2,... B1... C1...
    import re
    m = re.match(r"([A-Z])(\d+)", g)
    if m:
        letter = m.group(1)
        num = int(m.group(2))
        letter_rank = {"A": 0, "B": 1, "C": 2}.get(letter, 9)
        return (letter_rank * 100 + num, g)
    return (9999, g)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def _infer_thw(image_grid_thw: Optional[torch.Tensor]) -> Optional[Tuple[int, int, int]]:
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
    # remove added specials
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
    A_ss: torch.Tensor,  # [S,S] head-avg
    text_mask: torch.Tensor,
    vision_mask: torch.Tensor,
    agg: str = "max",
) -> torch.Tensor:
    q_idx = torch.nonzero(text_mask, as_tuple=False).squeeze(1)
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    if q_idx.numel() == 0 or k_idx.numel() == 0:
        raise RuntimeError("Empty text_mask or vision_mask.")
    sub = A_ss.index_select(0, q_idx).index_select(1, k_idx)  # [Q,P]
    if agg == "mean":
        return sub.mean(dim=0)
    if agg == "max":
        return sub.max(dim=0).values
    raise ValueError(f"Unknown agg={agg}")


def attention_gen1_q_to_vision(
    step_attn_layer: torch.Tensor,  # [B,H,Q,K] OR [B,H,S,S] in some impls
    vision_mask: torch.Tensor,
) -> torch.Tensor:
    """
    返回 [P]，并做 sum-normalize（避免全灰/不稳定）。
    """
    A = step_attn_layer[0]  # [H,Q,K] or [H,S,S]
    # 兼容两种形态
    if A.ndim != 3:
        raise RuntimeError(f"Unexpected GEN1 layer attn ndim={A.ndim}, shape={tuple(A.shape)}")

    H, Q, K = A.shape
    if Q == 1:
        q_idx = 0
    else:
        q_idx = Q - 1  # 用最后一个 query（通常对应“生成前/生成位点”）

    a_row = A.mean(dim=0)[q_idx]  # [K]
    k_idx = torch.nonzero(vision_mask, as_tuple=False).squeeze(1)
    vec = a_row.index_select(0, k_idx)  # [P]

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
    # fallback near-square
    h = int(math.sqrt(P)) or 1
    w = int(math.ceil(P / h))
    G = h * w
    if P < G:
        vec = np.pad(vec, (0, G - P), constant_values=0.0)
    else:
        vec = vec[:G]
    return vec.reshape(h, w)


# ---------------------------
# Carrier-equiv record parsing
# ---------------------------
def _extract_outcome_flags(obj: Dict[str, Any]) -> Dict[str, Any]:
    j = obj.get("qwen_judge") or {}
    is_asr = bool(j.get("is_attack_success", False)) and (not bool(j.get("echo_only", False)))
    is_refusal = bool(j.get("is_refusal", False))
    echo = bool(j.get("echo_only", False))
    return {"asr": is_asr, "refusal": is_refusal, "echo": echo}


def _load_items(judged_jsonl: Path, target_model: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with judged_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("model", "")) != target_model:
                continue
            # 必要字段
            if obj.get("base_qid") is None or obj.get("group") is None:
                continue
            items.append(obj)
    return items


def _select_baseqids(items: List[Dict[str, Any]], max_base: int, seed: int = 0) -> List[str]:
    # 按 base_qid 聚合，优先选择 group 完整度更高的
    from collections import defaultdict
    mp = defaultdict(list)
    for it in items:
        mp[str(it["base_qid"])].append(it)
    base_list = []
    for b, xs in mp.items():
        groups = {str(x["group"]) for x in xs}
        base_list.append((len(groups), b))
    base_list.sort(key=lambda x: (-x[0], x[1]))
    chosen = [b for _, b in base_list[: max_base if max_base > 0 else len(base_list)]]
    return chosen


# ---------------------------
# Attention extractor (single item)
# ---------------------------
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
) -> Dict[str, Any]:
    tok = processor.tokenizer

    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], images=[image], return_tensors="pt")
    else:
        # 纯文本：为了统一，我们仍然走 chat_template，但不带 image
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[chat], return_tensors="pt")

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    # -------- FWD attentions --------
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
    vision_mask = None
    text_mask = None

    if image_path is not None:
        vision_mask = build_vision_mask_from_image_pad(input_ids, tok)
        text_mask = build_text_mask_basic(input_ids, vision_mask, tok)
    else:
        # 没有图：直接返回空（此脚本主要用于含图 groups）
        return {
            "has_image": False,
            "fwd_vec": None,
            "gen1_vec": None,
            "meta": {"seq_len": int(input_ids.shape[1])},
        }

    L = len(attns)
    li = layer_idx if layer_idx >= 0 else (L + layer_idx)
    if not (0 <= li < L):
        raise ValueError(f"Invalid layer_idx={layer_idx}, resolved={li}, total_layers={L}")

    A_ss = attns[li][0].mean(dim=0)  # [S,S]
    fwd_vec = attention_text_to_vision_fwd(A_ss, text_mask=text_mask, vision_mask=vision_mask, agg=agg_fwd)
    fwd_vec = fwd_vec.float().clamp(min=0)
    fwd_vec = fwd_vec / (fwd_vec.sum() + 1e-6)  # 统一用 sum-normalize，便于相似度

    # grid
    thw = _infer_thw(inputs.get("image_grid_thw", None))
    if thw:
        t, h, w = thw
        grid_hw = (max(1, h // merge_size), max(1, w // merge_size))
    else:
        grid_hw = None

    # -------- GEN1 attentions --------
    # 用 generate 取第一步 attentions（返回结构可能是 tuple(tuple(layer)... per step)
    gen1_vec = None
    gen1_meta = {}
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
        # 兼容：g_attn 可能是 tuple(steps)，每个 step 是 tuple(layers)
        if g_attn is not None and isinstance(g_attn, (tuple, list)) and len(g_attn) >= 1:
            step0 = g_attn[0]
            if isinstance(step0, (tuple, list)) and len(step0) == L:
                layer0 = step0[li]  # [B,H,Q,K]
                gen1_vec = attention_gen1_q_to_vision(layer0, vision_mask=vision_mask)
                gen1_meta = {"gen_attn_ok": True, "step0_layer_shape": list(layer0.shape)}
            else:
                gen1_meta = {"gen_attn_ok": False, "reason": f"unexpected step0 type/len: {type(step0)}"}
        else:
            gen1_meta = {"gen_attn_ok": False, "reason": "gen_out.attentions is None/empty"}
    except Exception as e:
        gen1_meta = {"gen_attn_ok": False, "reason": f"exception: {type(e).__name__}: {e}"}

    # pack
    fwd_np = fwd_vec.detach().cpu().numpy()
    fwd_grid = vec_to_grid(fwd_np, grid_hw=grid_hw)

    gen1_grid = None
    if gen1_vec is not None:
        gen1_np = gen1_vec.detach().cpu().numpy()
        gen1_grid = vec_to_grid(gen1_np, grid_hw=grid_hw)

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
# Main analysis
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged_jsonl", required=True)
    ap.add_argument("--model_id", required=True, help="Qwen2.5-VL-7B-Instruct local path or hf repo")
    ap.add_argument("--target_model", default="qwen25_vl_7b", help="filter item['model']")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--max_base", type=int, default=50, help="how many base_qid to process (debug)")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--agg_fwd", choices=["mean", "max"], default="max")
    ap.add_argument("--merge_size", type=int, default=2)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judged_jsonl = Path(args.judged_jsonl)
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    # load items
    items = _load_items(judged_jsonl, target_model=args.target_model)

    # choose base_qids
    base_qids = _select_baseqids(items, max_base=args.max_base, seed=args.seed)
    base_set = set(base_qids)

    # regroup
    from collections import defaultdict
    mp = defaultdict(list)
    for it in items:
        b = str(it["base_qid"])
        if b in base_set:
            mp[b].append(it)

    # load model once
    if Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError("Qwen2_5_VLForConditionalGeneration not available in your transformers build.")
    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device).eval()
    _set_attn_eager_if_possible(model)

    # cache per (qid) attention grids
    attn_cache: Dict[str, Dict[str, Any]] = {}

    # output tables
    pair_rows: List[Dict[str, Any]] = []

    # For summary: choose a reference group (default A2 if exists)
    ref_group = "A2"

    for b in sorted(base_qids):
        xs = mp.get(b, [])
        if not xs:
            continue
        xs.sort(key=lambda o: _get_group_sort_key(str(o["group"])))

        # extract grids per group for this base
        per_group: Dict[str, Dict[str, Any]] = {}
        for it in xs:
            qid = str(it["qid"])
            g = str(it["group"])
            img = it.get("image_path", None)
            prompt = str(it.get("user_prompt", it.get("prompt", "")) or "")

            # 只对含图的组做 attention（A组纯文本一般没有 vision tokens）
            if img is None:
                continue

            if qid in attn_cache:
                rec = attn_cache[qid]
            else:
                rec = extract_fwd_and_gen1(
                    model=model,
                    processor=processor,
                    device=device,
                    image_path=str(img),
                    prompt_text=prompt,
                    layer_idx=args.layer,
                    agg_fwd=args.agg_fwd,
                    merge_size=args.merge_size,
                )
                attn_cache[qid] = rec

            flags = _extract_outcome_flags(it)
            per_group[g] = {
                "qid": qid,
                "perturb_p": float(it.get("perturb_p", 0.0) or 0.0),
                "fwd_grid": rec.get("fwd_grid", None),
                "gen1_grid": rec.get("gen1_grid", None),
                "meta": rec.get("meta", {}),
                **flags,
            }

        groups = sorted(per_group.keys(), key=_get_group_sort_key)
        if len(groups) < 2:
            continue

        # build pairwise similarities
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                r1, r2 = per_group[g1], per_group[g2]

                fwd1, fwd2 = r1["fwd_grid"], r2["fwd_grid"]
                gen1_1, gen1_2 = r1["gen1_grid"], r2["gen1_grid"]

                sim_fwd = _cosine(fwd1, fwd2) if (fwd1 is not None and fwd2 is not None) else float("nan")
                sim_gen1 = _cosine(gen1_1, gen1_2) if (gen1_1 is not None and gen1_2 is not None) else float("nan")

                row = {
                    "base_qid": b,
                    "perturb_p": r1["perturb_p"],
                    "g1": g1,
                    "g2": g2,
                    "qid_g1": r1["qid"],
                    "qid_g2": r2["qid"],
                    "sim_fwd": sim_fwd,
                    "sim_gen1": sim_gen1,
                    "asr_g1": int(r1["asr"]),
                    "asr_g2": int(r2["asr"]),
                    "refusal_g1": int(r1["refusal"]),
                    "refusal_g2": int(r2["refusal"]),
                    "echo_g1": int(r1["echo"]),
                    "echo_g2": int(r2["echo"]),
                    "same_asr": int(int(r1["asr"]) == int(r2["asr"])),
                    "same_refusal": int(int(r1["refusal"]) == int(r2["refusal"])),
                    "meta_vision_cnt": r1["meta"].get("vision_token_count", None),
                    "meta_grid_hw": r1["meta"].get("grid_hw", None),
                    "meta_gen1_ok_g1": int(bool(r1["meta"].get("gen1_meta", {}).get("gen_attn_ok", False))),
                    "meta_gen1_ok_g2": int(bool(r2["meta"].get("gen1_meta", {}).get("gen_attn_ok", False))),
                }
                pair_rows.append(row)

    # write CSV (no pandas dependency)
    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        keys = list(rows[0].keys())
        with path.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                vals = []
                for k in keys:
                    v = r.get(k, "")
                    if isinstance(v, (list, dict)):
                        s = json.dumps(v, ensure_ascii=False)
                    else:
                        s = str(v)
                    # basic csv escape
                    if "," in s or '"' in s:
                        s = '"' + s.replace('"', '""') + '"'
                    vals.append(s)
                f.write(",".join(vals) + "\n")

    out_pairs = out_dir / "attn_pairs.csv"
    write_csv(out_pairs, pair_rows)

    # quick summaries
    # 1) by same_asr (成功一致 vs 不一致)
    def summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        def _stat(xs: List[float]) -> Dict[str, float]:
            xs = [x for x in xs if not (math.isnan(x) or math.isinf(x))]
            if not xs:
                return {"n": 0, "mean": float("nan"), "std": float("nan")}
            arr = np.asarray(xs, dtype=np.float32)
            return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std())}

        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            buckets.setdefault(str(r.get(key, "")), []).append(r)

        out = {}
        for bk, rs in buckets.items():
            out[bk] = {
                "sim_fwd": _stat([float(x["sim_fwd"]) for x in rs]),
                "sim_gen1": _stat([float(x["sim_gen1"]) for x in rs]),
                "count_pairs": len(rs),
            }
        return out

    summary = {
        "args": vars(args),
        "num_pairs": len(pair_rows),
        "by_same_asr": summarize(pair_rows, "same_asr"),
        "by_same_refusal": summarize(pair_rows, "same_refusal"),
    }
    (out_dir / "attn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] wrote: {out_pairs}")
    print(f"[ok] wrote: {out_dir / 'attn_summary.json'}")
    print("[tip] 下一步：你可以按 group 选定参考组(A2/C2等)，只比较 sim_to_ref，更直观。")


if __name__ == "__main__":
    main()
