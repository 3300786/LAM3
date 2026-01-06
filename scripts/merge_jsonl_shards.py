from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, List, Tuple

def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="shard jsonl paths")
    ap.add_argument("--out", required=True, help="merged jsonl path")
    args = ap.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for ip in args.inputs:
        all_rows.extend(read_jsonl(Path(ip)))

    # dedup by (qid, group, model) to be safe
    seen = {}
    for r in all_rows:
        key = (r.get("qid"), r.get("group"), r.get("model"))
        seen[key] = r

    merged = list(seen.values())
    merged.sort(key=lambda r: (str(r.get("qid")), str(r.get("group")), str(r.get("model"))))

    out = Path(args.out)
    write_jsonl(out, merged)

    # quick sanity stats
    groups = {}
    for r in merged:
        groups[r["group"]] = groups.get(r["group"], 0) + 1

    print(f"[OK] inputs rows={len(all_rows)}  merged unique={len(merged)}  -> {out}")
    print("[STATS] group counts:", dict(sorted(groups.items())))

if __name__ == "__main__":
    main()
