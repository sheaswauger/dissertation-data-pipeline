#!/usr/bin/env python
from __future__ import annotations
import sys, json
from pathlib import Path
import pandas as pd

def main(inp: str, out: str) -> None:
    records = [json.loads(l) for l in Path(inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    # Naive flatten for demo; customize for your schema
    df = pd.json_normalize(records, max_level=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: normalize_json.py <in.jsonl> <out.parquet>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
