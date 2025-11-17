# src/metrics/build_synergy_subset.py
import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List


def load_grouped_asr(judged_in: Path) -> Dict[str, Dict[str, Any]]:
    """
    从 qwen_judge 结果中读取并按 id 聚合 asr 信息。

    返回:
      grouped[id]["per_mode"][mode]["asr"] = bool
    """
    grouped: Dict[str, Dict[str, Any]] = {}

    with judged_in.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = str(obj.get("id"))
            mode = obj.get("mode")
            judge = obj.get("qwen_judge") or {}
            asr = bool(judge.get("is_attack_success", False))

            grouped.setdefault(cid, {}).setdefault("per_mode", {})[mode] = {
                "asr": asr,
            }

    return grouped


def classify_ids(grouped: Dict[str, Dict[str, Any]]):
    """
    按 asr 模式把 id 分成三类:
      A: strict synergy
         asr(txt_img)=1, asr(txt_only)=0, asr(img_only)=0
      B: image-dominant synergy
         asr(txt_img)=1, asr(txt_only)=0, asr(img_only)=1
      C: text-dominant synergy
         asr(txt_img)=1, asr(txt_only)=1, asr(img_only)=0
      其他模式归为 others
    """
    A_ids: List[str] = []
    B_ids: List[str] = []
    C_ids: List[str] = []
    others: List[str] = []

    for cid, rec in grouped.items():
        per_mode = rec.get("per_mode", {})
        asr_tv = per_mode.get("txt_img", {}).get("asr", False)
        asr_t0 = per_mode.get("txt_only", {}).get("asr", False)
        asr_0v = per_mode.get("img_only", {}).get("asr", False)

        if asr_tv and (not asr_t0) and (not asr_0v):
            A_ids.append(cid)
        elif asr_tv and (not asr_t0) and asr_0v:
            B_ids.append(cid)
        elif asr_tv and asr_t0 and (not asr_0v):
            C_ids.append(cid)
        else:
            others.append(cid)

    return A_ids, B_ids, C_ids, others


def sample_ids(
    A_ids: List[str],
    B_ids: List[str],
    C_ids: List[str],
    max_per_type: int,
    seed: int = 42,
):
    """
    在 A/B/C 三类中各随机采样最多 max_per_type 个 id。
    返回 union 结果 selected_ids。
    """
    rng = random.Random(seed)

    def _sample(lst: List[str]) -> List[str]:
        if len(lst) <= max_per_type:
            return lst[:]
        lst_copy = lst[:]
        rng.shuffle(lst_copy)
        return lst_copy[:max_per_type]

    A_sel = _sample(A_ids)
    B_sel = _sample(B_ids)
    C_sel = _sample(C_ids)

    selected_ids = set(A_sel) | set(B_sel) | set(C_sel)

    print("[build-synergy-subset] class counts (before sampling):")
    print(f"  A(strict): {len(A_ids)}")
    print(f"  B(img-dom): {len(B_ids)}")
    print(f"  C(txt-dom): {len(C_ids)}")
    print("[build-synergy-subset] sampled ids:")
    print(f"  A(strict): {len(A_sel)}")
    print(f"  B(img-dom): {len(B_sel)}")
    print(f"  C(txt-dom): {len(C_sel)}")
    print(f"  total selected ids: {len(selected_ids)}")

    return selected_ids, A_sel, B_sel, C_sel


def filter_jsonl_by_ids(src: Path, dst: Path, keep_ids: set):
    """
    从 src(jsonl) 中筛选 id 在 keep_ids 的行，写入 dst。
    保留原始结构（包括 qwen_judge 等字段）。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    with src.open("r", encoding="utf-8") as fin, dst.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            cid = str(obj.get("id"))
            if cid in keep_ids:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"[build-synergy-subset] filter {src} -> {dst}")
    print(f"  lines in  = {n_in}")
    print(f"  lines out = {n_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_in",
        required=True,
        help="原始 raw jsonl, e.g. synergy_jbv28k_mini_raw.jsonl",
    )
    ap.add_argument(
        "--judged_in",
        required=True,
        help="带 qwen_judge 的 jsonl, e.g. synergy_jbv28k_mini_qwen_judge.jsonl",
    )
    ap.add_argument(
        "--raw_out",
        required=True,
        help="输出的重采样 raw jsonl",
    )
    ap.add_argument(
        "--judged_out",
        required=True,
        help="输出的重采样 judged jsonl",
    )
    ap.add_argument(
        "--max_per_type",
        type=int,
        default=80,
        help="每个协同类型(A/B/C)最多采样多少个 id",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    args = ap.parse_args()

    raw_in = Path(args.raw_in)
    judged_in = Path(args.judged_in)
    raw_out = Path(args.raw_out)
    judged_out = Path(args.judged_out)

    grouped = load_grouped_asr(judged_in)
    A_ids, B_ids, C_ids, others = classify_ids(grouped)
    selected_ids, A_sel, B_sel, C_sel = sample_ids(
        A_ids, B_ids, C_ids, max_per_type=args.max_per_type, seed=args.seed
    )

    # 过滤 raw / judged 文件
    filter_jsonl_by_ids(raw_in, raw_out, selected_ids)
    filter_jsonl_by_ids(judged_in, judged_out, selected_ids)

    print("[build-synergy-subset] done.")


if __name__ == "__main__":
    main()
