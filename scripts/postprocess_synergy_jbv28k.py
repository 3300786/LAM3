# scripts/postprocess_synergy_jbv28k.py
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

ALLOWED_MODES: List[str] = ["txt_img", "txt_only", "img_only", "none"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_path",
        type=str,
        default="outputs/logs/synergy_jbv28k_raw.jsonl",
        help="原始 raw 日志路径",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="outputs/logs/synergy_jbv28k_raw.sorted.jsonl",
        help="去重并排序后的输出路径",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=28000,
        help="样本数（id 范围为 [0, n_samples)）",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    n_samples = args.n_samples

    if not in_path.is_file():
        raise FileNotFoundError(f"in_path not found: {in_path}")

    print(f"[post] input  file: {in_path}")
    print(f"[post] output file: {out_path}")
    print(f"[post] expected ids: [0, {n_samples}), modes: {ALLOWED_MODES}")

    # (id, mode) -> record
    records: Dict[Tuple[int, str], Dict[str, Any]] = {}

    total_lines = 0
    parsed_lines = 0
    dup_count = 0
    skipped_bad_mode = 0
    skipped_bad_id = 0
    skipped_other = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped_other += 1
                continue

            parsed_lines += 1
            cid_raw = obj.get("id", None)
            mode_raw = obj.get("mode", None)

            if cid_raw is None or mode_raw is None:
                skipped_other += 1
                continue

            mode = str(mode_raw)
            if mode not in ALLOWED_MODES:
                skipped_bad_mode += 1
                continue

            try:
                cid_int = int(cid_raw)
            except Exception:
                skipped_bad_id += 1
                continue

            if not (0 <= cid_int < n_samples):
                skipped_bad_id += 1
                continue

            key = (cid_int, mode)
            if key in records:
                dup_count += 1
                # 保留第一个，后面的丢弃
                continue

            records[key] = obj

    print(f"[post] total lines read        : {total_lines}")
    print(f"[post] lines parsed as JSON   : {parsed_lines}")
    print(f"[post] duplicates (id, mode)  : {dup_count}")
    print(f"[post] skipped bad_mode       : {skipped_bad_mode}")
    print(f"[post] skipped bad_id/range   : {skipped_bad_id}")
    print(f"[post] skipped other/invalid  : {skipped_other}")
    print(f"[post] unique valid pairs kept: {len(records)}")

    # 检查缺失 / 多余
    expected_total = n_samples * len(ALLOWED_MODES)
    missing: List[Tuple[int, str]] = []
    for cid in range(n_samples):
        for mode in ALLOWED_MODES:
            if (cid, mode) not in records:
                missing.append((cid, mode))

    extra_pairs = len(records) - (expected_total - len(missing))

    print(f"[post] expected total pairs   : {expected_total}")
    print(f"[post] missing pairs         : {len(missing)}")
    print(f"[post] extra pairs (rough)   : {extra_pairs}")

    if missing:
        print("[post] ERROR: some (id, mode) pairs are missing. Examples (up to 20):")
        for cid, mode in missing[:20]:
            print(f"    missing: id={cid}, mode={mode}")
        print(
            "[post] Please检查 raw 日志是否完整；本脚本不会生成输出文件以避免使用不完整数据。"
        )
        raise SystemExit(1)

    # 若没有缺失，则按指定顺序写出：{0, txt_img}, {0, txt_only}, ...
    print("[post] all required (id, mode) pairs are present, writing sorted output...")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for cid in range(n_samples):
            for mode in ALLOWED_MODES:
                key = (cid, mode)
                rec = records.get(key)
                if rec is None:
                    # 理论上不会发生，因为刚刚已经检查过 missing
                    raise RuntimeError(f"internal error: missing record for {key}")
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[post] done. sorted file written to:", out_path)


if __name__ == "__main__":
    main()
