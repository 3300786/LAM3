# src/metrics/attn_carrier_equiv_qwen25vl.py
# Carrier-Equiv attention observables for Qwen2.5-VL-7B
#
# What this script does (high-level):
# 1) Load judged jsonl (carrier-equiv results).
# 2) For each selected base_qid, extract per-group attention heatmaps (FWD text->vision, GEN1 q->vision).
#    NOTE: Attention extraction logic is kept the same as your stable version.
# 3) Aggregate similarities by (group_i, group_j) into group-pair matrices:
#    - all pairs
#    - same_refusal vs diff_refusal
#    - same_asr vs diff_asr
# 4) Also dump "key pairs" (B1-C2, C1-C2, C9-C10 by default) for focused comparisons.
# 5) Add tqdm progress bars.
#
# Outputs (in out_dir):
# - group_pair_stats.json               (all aggregated stats)
# - matrix_sim_fwd_all.csv              (group-pair mean sim_fwd)
# - matrix_sim_gen1_all.csv             (group-pair mean sim_gen1)
# - matrix_sim_fwd_same_refusal.csv     (subset)
# - matrix_sim_gen1_same_refusal.csv
# - matrix_sim_fwd_diff_refusal.csv
# - matrix_sim_gen1_diff_refusal.csv
# - matrix_sim_fwd_same_asr.csv
# - matrix_sim_gen1_same_asr.csv
# - matrix_sim_fwd_diff_asr.csv
# - matrix_sim_gen1_diff_asr.csv
# - key_pairs.csv                       (row-level for selected key group pairs)
# - attn_summary.json                   (quick summary counts)

from __future__ import annotations

import argparse
import csv
import json
import math
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
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
            if obj.get("base_qid") is None or obj.get("group") is None:
                continue
            items.append(obj)
    return items


def _select_baseqids(items: List[Dict[str, Any]], max_base: int) -> List[str]:
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
    skip_gen1: bool = False,
) -> Dict[str, Any]:
    """
    注意力计算逻辑保持不变（只做了最小的容错与 meta 记录）。
    """
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
    if image_path is None:
        return {"has_image": False, "fwd_grid": None, "gen1_grid": None, "meta": {"seq_len": int(input_ids.shape[1])}}

    vision_mask = build_vision_mask_from_image_pad(input_ids, tok)
    text_mask = build_text_mask_basic(input_ids, vision_mask, tok)

    L = len(attns)
    li = layer_idx if layer_idx >= 0 else (L + layer_idx)
    if not (0 <= li < L):
        raise ValueError(f"Invalid layer_idx={layer_idx}, resolved={li}, total_layers={L}")

    A_ss = attns[li][0].mean(dim=0)  # [S,S]
    fwd_vec = attention_text_to_vision_fwd(A_ss, text_mask=text_mask, vision_mask=vision_mask, agg=agg_fwd)
    fwd_vec = fwd_vec.float().clamp(min=0)
    fwd_vec = fwd_vec / (fwd_vec.sum() + 1e-6)  # sum-normalize

    # grid
    thw = _infer_thw(inputs.get("image_grid_thw", None))
    if thw:
        t, h, w = thw
        grid_hw = (max(1, h // merge_size), max(1, w // merge_size))
    else:
        grid_hw = None

    # -------- GEN1 attentions --------
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
            # g_attn: tuple(steps), each step: tuple(layers)
            if g_attn is not None and isinstance(g_attn, (tuple, list)) and len(g_attn) >= 1:
                step0 = g_attn[0]
                if isinstance(step0, (tuple, list)) and len(step0) == L:
                    layer0 = step0[li]  # [B,H,Q,K]
                    gen1_vec = attention_gen1_q_to_vision(layer0, vision_mask=vision_mask)
                    gen1_meta = {"gen_attn_ok": True, "step0_layer_shape": list(layer0.shape)}
                else:
                    gen1_meta = {"gen_attn_ok": False, "reason": f"unexpected step0 type/len: {type(step0)} / {getattr(step0, '__len__', lambda: -1)()}"}
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
# Aggregators
# ---------------------------
@dataclass
class OnlineStats:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0

    def add(self, x: float) -> None:
        if math.isnan(x) or math.isinf(x):
            return
        self.n += 1
        self.s += float(x)
        self.ss += float(x) * float(x)

    def to_dict(self) -> Dict[str, float]:
        if self.n <= 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan")}
        mean = self.s / self.n
        var = max(0.0, (self.ss / self.n) - mean * mean)
        return {"n": int(self.n), "mean": float(mean), "std": float(math.sqrt(var))}


def _pair_key(g1: str, g2: str) -> Tuple[str, str]:
    return (g1, g2) if _get_group_sort_key(g1) <= _get_group_sort_key(g2) else (g2, g1)


def _write_matrix_csv(path: Path, groups: List[str], mat: Dict[Tuple[str, str], float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group"] + groups)
        for gi in groups:
            row = [gi]
            for gj in groups:
                if gi == gj:
                    row.append("1.0")
                else:
                    k = _pair_key(gi, gj)
                    v = mat.get(k, float("nan"))
                    row.append(str(v))
            w.writerow(row)


def _write_key_pairs_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            out = []
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, (dict, list)):
                    out.append(json.dumps(v, ensure_ascii=False))
                else:
                    out.append(v)
            w.writerow(out)


# ---------------------------
# Main analysis
# ---------------------------
def main() -> None:
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
    ap.add_argument("--skip_gen1", action="store_true", help="only compute FWD (debug/perf)")

    # Focused group-pairs (do not change attention; only change grouping/statistics)
    ap.add_argument(
        "--key_pairs",
        default="B1:C2,C1:C2,C9:C10",
        help='comma-separated pairs like "B1:C2,C1:C2". only applied if both groups exist.',
    )

    args = ap.parse_args()

    judged_jsonl = Path(args.judged_jsonl)
    out_dir = Path(args.out_dir)
    _safe_mkdir(out_dir)

    # load items
    items = _load_items(judged_jsonl, target_model=args.target_model)

    # choose base_qids
    base_qids = _select_baseqids(items, max_base=args.max_base)
    base_set = set(base_qids)

    # regroup by base
    from collections import defaultdict

    mp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
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

    # cache per qid -> extracted grids
    attn_cache: Dict[str, Dict[str, Any]] = {}

    # online stats by group pair
    stats_all_fwd: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_all_gen1: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)

    stats_same_refusal_fwd: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_same_refusal_gen1: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_diff_refusal_fwd: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_diff_refusal_gen1: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)

    stats_same_asr_fwd: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_same_asr_gen1: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_diff_asr_fwd: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)
    stats_diff_asr_gen1: Dict[Tuple[str, str], OnlineStats] = defaultdict(OnlineStats)

    # counts / health
    total_pairs = 0
    used_pairs = 0
    gen1_ok_cnt = 0
    gen1_total_cnt = 0

    # key pairs rows
    key_pairs_spec = []
    for x in str(args.key_pairs).split(","):
        x = x.strip()
        if not x:
            continue
        if ":" not in x:
            continue
        a, b = x.split(":", 1)
        key_pairs_spec.append((_pair_key(a.strip(), b.strip())))
    key_rows: List[Dict[str, Any]] = []

    # collect global group set (image groups only) for matrices
    global_groups_set: set[str] = set()

    pbar = tqdm(sorted(base_qids), desc="base_qid", dynamic_ncols=True)
    for b in pbar:
        xs = mp.get(b, [])
        if not xs:
            continue
        xs.sort(key=lambda o: _get_group_sort_key(str(o["group"])))

        # per-group extracted features (image-only groups)
        per_group: Dict[str, Dict[str, Any]] = {}
        for it in xs:
            g = str(it["group"])
            img = it.get("image_path", None)
            if img is None:
                continue  # skip no-image groups (A*)
            img_path = str(img)
            qid = str(it["qid"])
            prompt = str(it.get("user_prompt", it.get("prompt", "")) or "")

            if qid in attn_cache:
                rec = attn_cache[qid]
            else:
                try:
                    rec = extract_fwd_and_gen1(
                        model=model,
                        processor=processor,
                        device=device,
                        image_path=img_path,
                        prompt_text=prompt,
                        layer_idx=args.layer,
                        agg_fwd=args.agg_fwd,
                        merge_size=args.merge_size,
                        skip_gen1=bool(args.skip_gen1),
                    )
                except Exception as e:
                    # 保守：不让单条样本的注意力异常干扰整体统计
                    rec = {
                        "has_image": True,
                        "fwd_grid": None,
                        "gen1_grid": None,
                        "meta": {"error": f"{type(e).__name__}: {e}"},
                    }
                attn_cache[qid] = rec

            flags = _extract_outcome_flags(it)
            per_group[g] = {
                "qid": qid,
                "fwd_grid": rec.get("fwd_grid", None),
                "gen1_grid": rec.get("gen1_grid", None),
                "meta": rec.get("meta", {}),
                **flags,
            }
            global_groups_set.add(g)

            # gen1 health
            gm = (rec.get("meta", {}) or {}).get("gen1_meta", {}) or {}
            gen1_total_cnt += 1
            if bool(gm.get("gen_attn_ok", False)):
                gen1_ok_cnt += 1

        groups = sorted(per_group.keys(), key=_get_group_sort_key)
        if len(groups) < 2:
            continue

        # pairwise aggregation within this base_qid
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                r1, r2 = per_group[g1], per_group[g2]
                k = _pair_key(g1, g2)

                total_pairs += 1

                f1, f2 = r1["fwd_grid"], r2["fwd_grid"]
                if f1 is None or f2 is None:
                    continue
                sim_fwd = _cosine(f1, f2)

                sim_gen1 = float("nan")
                if not bool(args.skip_gen1):
                    gg1, gg2 = r1["gen1_grid"], r2["gen1_grid"]
                    if gg1 is not None and gg2 is not None:
                        sim_gen1 = _cosine(gg1, gg2)

                used_pairs += 1

                # all
                stats_all_fwd[k].add(sim_fwd)
                stats_all_gen1[k].add(sim_gen1)

                # same/diff refusal
                same_refusal = (int(r1["refusal"]) == int(r2["refusal"]))
                if same_refusal:
                    stats_same_refusal_fwd[k].add(sim_fwd)
                    stats_same_refusal_gen1[k].add(sim_gen1)
                else:
                    stats_diff_refusal_fwd[k].add(sim_fwd)
                    stats_diff_refusal_gen1[k].add(sim_gen1)

                # same/diff asr
                same_asr = (int(r1["asr"]) == int(r2["asr"]))
                if same_asr:
                    stats_same_asr_fwd[k].add(sim_fwd)
                    stats_same_asr_gen1[k].add(sim_gen1)
                else:
                    stats_diff_asr_fwd[k].add(sim_fwd)
                    stats_diff_asr_gen1[k].add(sim_gen1)

                # key pairs dump
                if k in key_pairs_spec:
                    key_rows.append(
                        {
                            "base_qid": b,
                            "g1": k[0],
                            "g2": k[1],
                            "qid_g1": r1["qid"] if g1 == k[0] else r2["qid"],
                            "qid_g2": r2["qid"] if g2 == k[1] else r1["qid"],
                            "sim_fwd": sim_fwd,
                            "sim_gen1": sim_gen1,
                            "asr_g1": int(r1["asr"]) if g1 == k[0] else int(r2["asr"]),
                            "asr_g2": int(r2["asr"]) if g2 == k[1] else int(r1["asr"]),
                            "refusal_g1": int(r1["refusal"]) if g1 == k[0] else int(r2["refusal"]),
                            "refusal_g2": int(r2["refusal"]) if g2 == k[1] else int(r1["refusal"]),
                            "echo_g1": int(r1["echo"]) if g1 == k[0] else int(r2["echo"]),
                            "echo_g2": int(r2["echo"]) if g2 == k[1] else int(r1["echo"]),
                        }
                    )

        pbar.set_postfix(
            {
                "pairs": used_pairs,
                "gen1_ok_rate": (gen1_ok_cnt / max(1, gen1_total_cnt)),
            }
        )

    # finalize group ordering for matrices
    all_groups = sorted(global_groups_set, key=_get_group_sort_key)

    # build matrices (mean only)
    def _mean_matrix(stats_map: Dict[Tuple[str, str], OnlineStats]) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        for k, st in stats_map.items():
            d = st.to_dict()
            out[k] = float(d["mean"])
        return out

    # write matrix csvs
    _write_matrix_csv(out_dir / "matrix_sim_fwd_all.csv", all_groups, _mean_matrix(stats_all_fwd))
    _write_matrix_csv(out_dir / "matrix_sim_gen1_all.csv", all_groups, _mean_matrix(stats_all_gen1))

    _write_matrix_csv(out_dir / "matrix_sim_fwd_same_refusal.csv", all_groups, _mean_matrix(stats_same_refusal_fwd))
    _write_matrix_csv(out_dir / "matrix_sim_gen1_same_refusal.csv", all_groups, _mean_matrix(stats_same_refusal_gen1))
    _write_matrix_csv(out_dir / "matrix_sim_fwd_diff_refusal.csv", all_groups, _mean_matrix(stats_diff_refusal_fwd))
    _write_matrix_csv(out_dir / "matrix_sim_gen1_diff_refusal.csv", all_groups, _mean_matrix(stats_diff_refusal_gen1))

    _write_matrix_csv(out_dir / "matrix_sim_fwd_same_asr.csv", all_groups, _mean_matrix(stats_same_asr_fwd))
    _write_matrix_csv(out_dir / "matrix_sim_gen1_same_asr.csv", all_groups, _mean_matrix(stats_same_asr_gen1))
    _write_matrix_csv(out_dir / "matrix_sim_fwd_diff_asr.csv", all_groups, _mean_matrix(stats_diff_asr_fwd))
    _write_matrix_csv(out_dir / "matrix_sim_gen1_diff_asr.csv", all_groups, _mean_matrix(stats_diff_asr_gen1))

    # dump key pairs rows
    _write_key_pairs_csv(out_dir / "key_pairs.csv", key_rows)

    # dump detailed stats json
    def _stats_dict(stats_map: Dict[Tuple[str, str], OnlineStats]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for (g1, g2), st in stats_map.items():
            out[f"{g1}|{g2}"] = st.to_dict()
        return out

    group_pair_stats = {
        "groups": all_groups,
        "all": {"fwd": _stats_dict(stats_all_fwd), "gen1": _stats_dict(stats_all_gen1)},
        "same_refusal": {"fwd": _stats_dict(stats_same_refusal_fwd), "gen1": _stats_dict(stats_same_refusal_gen1)},
        "diff_refusal": {"fwd": _stats_dict(stats_diff_refusal_fwd), "gen1": _stats_dict(stats_diff_refusal_gen1)},
        "same_asr": {"fwd": _stats_dict(stats_same_asr_fwd), "gen1": _stats_dict(stats_same_asr_gen1)},
        "diff_asr": {"fwd": _stats_dict(stats_diff_asr_fwd), "gen1": _stats_dict(stats_diff_asr_gen1)},
    }
    (out_dir / "group_pair_stats.json").write_text(json.dumps(group_pair_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # quick summary
    summary = {
        "args": vars(args),
        "num_base": len(base_qids),
        "num_groups_global": len(all_groups),
        "total_pairs_considered": total_pairs,
        "pairs_with_fwd_used": used_pairs,
        "gen1_ok_rate_over_items": float(gen1_ok_cnt / max(1, gen1_total_cnt)),
        "key_pairs_spec": [f"{a}:{b}" for a, b in key_pairs_spec],
        "key_pairs_rows": len(key_rows),
    }
    (out_dir / "attn_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[ok] out_dir: {out_dir}")
    print(f"[ok] wrote matrices + group_pair_stats.json + key_pairs.csv + attn_summary.json")


if __name__ == "__main__":
    main()
