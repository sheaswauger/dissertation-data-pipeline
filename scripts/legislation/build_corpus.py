#!/usr/bin/env python3
"""
build_corpus.py - Final pipeline stage.
Takes normalized parquet from normalize_corpus.py,
produces master analysis-ready corpus.

Usage:
    python build_corpus.py
    python build_corpus.py --input output/corpus/master_corpus_20260307.parquet
"""

import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv


FINAL_COLUMNS = [
    "bill_id", "state", "year", "bill_number", "title", "status",
    "last_action_date", "matched_keywords", "search_method", "is_amendment",
    "text_source", "strikethrough_detected", "strikethrough_limitation_flag",
    "raw_word_count", "normalized_word_count", "raw_text", "normalized_text",
]


def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except Exception:
        return 0


def find_latest(corpus_dir, pattern):
    files = sorted(corpus_dir.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {corpus_dir}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to normalized parquet file")
    parser.add_argument("--decisions", help="Path to normalization decisions CSV")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    load_dotenv(project_dir / ".env")
    corpus_dir = project_dir / "output" / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.input) if args.input else find_latest(corpus_dir, "master_corpus_2*.parquet")
    decisions_path = Path(args.decisions) if args.decisions else find_latest(corpus_dir, "normalization_decisions_*.csv")

    print(f"Input:     {parquet_path}")
    print(f"Decisions: {decisions_path}")

    df = pd.read_parquet(parquet_path)
    decisions_df = pd.read_csv(decisions_path)

    merge_cols = ["bill_id", "is_amendment", "strikethrough_detected",
                  "strikethrough_limitation_flag", "raw_word_count", "normalized_word_count"]
    df = df.merge(decisions_df[merge_cols], on="bill_id", how="left")

    # Normalize column names (handle varying casing from collector)
    col_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    df["state"] = df.get("State", df.get("state", "")).astype(str).str.strip().str.upper()
    df["bill_number"] = df.get("Bill Number", df.get("bill_number", "")).astype(str).str.strip()
    df["title"] = df.get("Title", df.get("title", "")).astype(str).str.strip()
    df["status"] = df.get("Status", df.get("status", "")).astype(str).str.strip()
    df["last_action_date"] = df.get("Last Action Date", df.get("last_action_date", "")).astype(str).str.strip()
    df["matched_keywords"] = df.get("Matched Keywords", df.get("matched_keywords", "")).astype(str).str.strip()
    df["search_method"] = df.get("Search Method", df.get("search_method", "unknown")).astype(str).str.strip()
    df["text_source"] = df.get("text_source", "unknown")
    df["bill_id"] = df["state"] + "_" + df["bill_number"]
    df["year"] = df["last_action_date"].apply(extract_year)

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    final = df[FINAL_COLUMNS].copy()
    final = final[final["state"].str.len() == 2]
    final = final[final["year"] >= 2010]
    final = final.drop_duplicates(subset=["bill_id"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full parquet (all columns including text)
    out_parquet = corpus_dir / f"master_corpus_final_{timestamp}.parquet"
    final.to_parquet(out_parquet, index=False)

    # Metadata-only CSV (no text blobs)
    meta_cols = [c for c in FINAL_COLUMNS if c not in ["raw_text", "normalized_text"]]
    out_csv = corpus_dir / f"master_corpus_final_{timestamp}.csv"
    final[meta_cols].to_csv(out_csv, index=False)

    # Dedoose-ready CSV
    dedoose_cols = ["bill_id", "state", "year", "bill_number", "title",
                    "status", "matched_keywords", "normalized_text"]
    dedoose_csv = corpus_dir / f"dedoose_sample_{timestamp}.csv"
    dedoose = final[dedoose_cols].dropna(subset=["normalized_text"])
    dedoose = dedoose[dedoose["normalized_text"].str.len() > 50]
    dedoose.to_csv(dedoose_csv, index=False)

    # Summary report
    summary_path = corpus_dir / f"corpus_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write(f"MASTER CORPUS SUMMARY\nGenerated: {datetime.now().isoformat()}\n\n")
        f.write(f"Total bills:           {len(final)}\n")
        f.write(f"Bills with text:       {(final['normalized_text'].str.len() > 50).sum()}\n")
        f.write(f"Year range:            {final['year'].min()} - {final['year'].max()}\n")
        f.write(f"States covered:        {final['state'].nunique()}\n")
        f.write(f"Amendment bills:       {final['is_amendment'].sum()}\n")
        f.write(f"Strikethrough removed: {final['strikethrough_detected'].sum()}\n")
        f.write(f"PDF flags:             {final['strikethrough_limitation_flag'].sum()}\n\n")
        f.write("BILLS BY YEAR\n" + "="*40 + "\n")
        for yr, ct in final["year"].value_counts().sort_index().items():
            f.write(f"  {yr}: {ct}\n")
        f.write("\nBILLS BY STATE (top 15)\n" + "="*40 + "\n")
        for st, ct in final["state"].value_counts().head(15).items():
            f.write(f"  {st}: {ct}\n")

    print(f"\n{'='*50}")
    print(f"Total bills:    {len(final)}")
    print(f"Bills w/ text:  {(final['normalized_text'].str.len() > 50).sum()}")
    print(f"States:         {final['state'].nunique()}")
    print(f"\nOutputs:")
    print(f"  {out_parquet}")
    print(f"  {out_csv}")
    print(f"  {dedoose_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
