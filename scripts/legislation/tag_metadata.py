#!/usr/bin/env python
from __future__ import annotations
import sys, re
import pandas as pd
from pathlib import Path

THEME_RULES = {
    "school_safety": [r"school", r"safety", r"security", r"resource officer", r"SRO"],
    "firearm_restrictions": [r"firearm", r"gun", r"weapon", r"background check"],
    "mental_health": [r"mental health", r"counseling", r"behavioral"],
}

def tag_row(text: str) -> set[str]:
    tags = set()
    for theme, pats in THEME_RULES.items():
        for pat in pats:
            if re.search(pat, text, flags=re.I):
                tags.add(theme)
                break
    return tags

def main(inp: str, out: str) -> None:
    df = pd.read_parquet(inp)
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ("title", "summary", "text"))]
    basis = df[text_cols].fillna("").agg(" ".join, axis=1) if text_cols else pd.Series([""] * len(df))
    df["tags"] = basis.apply(lambda s: sorted(tag_row(s)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved tagged data to {out}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tag_metadata.py <in.parquet> <out.parquet>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
