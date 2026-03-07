#!/usr/bin/env python3
"""
normalize_corpus.py

Third stage of the corpus pipeline. Takes the enriched CSV from fetch_texts.py,
applies text normalization to each bill, and outputs:

1. A master parquet file with both raw_text and normalized_text columns
2. A decisions log CSV documenting every normalization decision made
3. A summary report for the data quality documentation

Normalization philosophy:
- raw_text is NEVER modified — always preserved exactly as fetched
- normalized_text is a parallel version for NLP processing only
- All decisions are logged for methodological transparency

Strikethrough handling (Option 2 per research design decision 2026-03-07):
- HTML bills: <s>, <strike>, <del> tags and their content are removed from
  normalized_text. Raw_text preserves them.
- PDF bills: strikethrough cannot be detected — flagged in decisions log
  as a known limitation.
- Amendment bills are flagged in corpus metadata.

Usage:
    python normalize_corpus.py
    python normalize_corpus.py --input output/corpus_with_texts_20260307.csv
    python normalize_corpus.py --limit 10
"""

import os
import sys
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# ─── Normalization decisions log schema ───────────────────────────────────────
DECISIONS_COLUMNS = [
    "bill_id",
    "state",
    "bill_number",
    "text_source",
    "is_amendment",
    "original_format",
    "strikethrough_detected",
    "strikethrough_chars_removed",
    "strikethrough_detection_method",
    "strikethrough_limitation_flag",
    "html_artifacts_removed",
    "raw_word_count",
    "normalized_word_count",
    "words_removed_pct",
    "normalization_timestamp",
    "notes",
]


def detect_amendment(title: str, raw_text: str) -> bool:
    """
    Heuristically detect if a bill is an amendment to existing law.
    Amendments typically contain struck text modifying existing statutes.
    """
    title_lower = str(title).lower()
    amendment_signals = [
        "amend", "amendment", "relating to", "concerning",
        "modifying", "revising", "repealing", "adding to"
    ]
    if any(signal in title_lower for signal in amendment_signals):
        return True

    text_lower = str(raw_text).lower()
    if "be it enacted" in text_lower and "amend" in text_lower[:500]:
        return True

    return False


def remove_strikethrough_html(html_text: str) -> Tuple[str, int, str]:
    """
    Remove strikethrough content from HTML bill text.
    Returns: (cleaned_text, chars_removed, method_used)
    """
    soup = BeautifulSoup(html_text, "lxml")

    chars_removed = 0
    method = "none_found"

    strikethrough_tags = soup.find_all(["s", "strike", "del"])

    if strikethrough_tags:
        for tag in strikethrough_tags:
            chars_removed += len(tag.get_text())
            tag.decompose()
        method = "html_tags_s_strike_del"

    css_struck = soup.find_all(
        style=re.compile(r"text-decoration\s*:\s*line-through", re.IGNORECASE)
    )
    if css_struck:
        for tag in css_struck:
            chars_removed += len(tag.get_text())
            tag.decompose()
        method = "css_line_through" if not strikethrough_tags else "html_tags_and_css"

    for script in soup(["script", "style"]):
        script.decompose()

    cleaned = soup.get_text(separator="\n", strip=True)
    return cleaned, chars_removed, method


def normalize_plain_text(text: str) -> str:
    """
    Light normalization for plain text bills (no strikethrough detection possible).
    """
    text = text.lower()
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', ' ', text)
    text = re.sub(r'\f', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text


def normalize_bill(
    row: pd.Series,
    plain_dir: Path,
) -> Tuple[str, str, dict]:
    """
    Normalize a single bill. Returns:
    - raw_text: full text, unmodified
    - normalized_text: cleaned parallel version
    - decision: dict of decisions made for the log
    """
    state = str(row.get("State", ""))
    bill_number = str(row.get("Bill Number", ""))
    title = str(row.get("Title", ""))
    plain_text_file = str(row.get("plain_text_file", ""))
    text_source = str(row.get("text_source", "unknown"))

    decision = {
        "bill_id": f"{state}_{bill_number}",
        "state": state,
        "bill_number": bill_number,
        "text_source": text_source,
        "is_amendment": False,
        "original_format": "unknown",
        "strikethrough_detected": False,
        "strikethrough_chars_removed": 0,
        "strikethrough_detection_method": "none",
        "strikethrough_limitation_flag": False,
        "html_artifacts_removed": False,
        "raw_word_count": 0,
        "normalized_word_count": 0,
        "words_removed_pct": 0.0,
        "normalization_timestamp": datetime.now().isoformat(),
        "notes": "",
    }

    raw_text = ""
    if plain_text_file:
        plain_path = plain_dir / plain_text_file
        if plain_path.exists():
            with open(plain_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()

    if not raw_text:
        decision["notes"] = "no_text_available"
        return "", "", decision

    decision["raw_word_count"] = len(raw_text.split())
    decision["is_amendment"] = detect_amendment(title, raw_text)

    raw_file = str(row.get("raw_text_file", ""))
    is_html = raw_file.endswith((".html", ".htm")) or "<html" in raw_text[:500].lower()
    is_pdf = raw_file.endswith(".pdf")

    if is_html or "<" in raw_text[:1000]:
        decision["original_format"] = "html"

        normalized, chars_removed, method = remove_strikethrough_html(raw_text)

        if chars_removed > 0:
            decision["strikethrough_detected"] = True
            decision["strikethrough_chars_removed"] = chars_removed
            decision["strikethrough_detection_method"] = method

        normalized = normalized.lower()
        normalized = re.sub(r'[ \t]+', ' ', normalized)
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        normalized = re.sub(r'^\s*\d+\s*$', '', normalized, flags=re.MULTILINE)
        normalized = normalized.strip()
        decision["html_artifacts_removed"] = True

    elif is_pdf:
        decision["original_format"] = "pdf"
        decision["strikethrough_limitation_flag"] = True
        decision["notes"] = (
            "PDF_strikethrough_undetectable: strikethrough text cannot be "
            "identified in PDF format. Deleted statutory text may be present "
            "in normalized_text. See data quality report section 3.2."
        )
        normalized = normalize_plain_text(raw_text)

    else:
        decision["original_format"] = "plain_text"
        decision["strikethrough_limitation_flag"] = True
        decision["notes"] = (
            "plain_text_strikethrough_undetectable: no formatting available "
            "to identify deleted text."
        )
        normalized = normalize_plain_text(raw_text)

    decision["normalized_word_count"] = len(normalized.split())
    raw_wc = decision["raw_word_count"]
    if raw_wc > 0:
        removed = raw_wc - decision["normalized_word_count"]
        decision["words_removed_pct"] = round((removed / raw_wc) * 100, 2)

    return raw_text, normalized, decision


def find_latest_corpus_csv(output_dir: Path) -> Path:
    csvs = sorted(output_dir.glob("corpus_with_texts_*.csv"), reverse=True)
    if not csvs:
        csvs = sorted(output_dir.glob("results_*.csv"), reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No corpus CSV found in {output_dir}")
    return csvs[0]


def main():
    parser = argparse.ArgumentParser(description="Normalize bill corpus text")
    parser.add_argument("--input", help="Path to corpus CSV")
    parser.add_argument("--limit", type=int, help="Process first N bills only")
    parser.add_argument("--force", action="store_true", help="Reprocess all bills")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    load_dotenv(project_dir / ".env")

    output_dir = project_dir / "output"
    plain_dir = output_dir / "bill_texts" / "plain"
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        input_csv = Path(args.input)
    else:
        input_csv = find_latest_corpus_csv(output_dir)

    print(f"Input:     {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} bills")

    if args.limit:
        df = df.head(args.limit)
        print(f"Limiting to {args.limit} bills")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_texts = []
    normalized_texts = []
    decisions = []

    print(f"\nNormalizing {len(df)} bills...\n")

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        state = row.get("State", "")
        bill_number = row.get("Bill Number", "")

        if i % 50 == 0 or i == 1:
            print(f"[{i}/{len(df)}] {state} {bill_number}")

        raw, normalized, decision = normalize_bill(row, plain_dir)
        raw_texts.append(raw)
        normalized_texts.append(normalized)
        decisions.append(decision)

    df["raw_text"] = raw_texts
    df["normalized_text"] = normalized_texts

    decisions_df = pd.DataFrame(decisions, columns=DECISIONS_COLUMNS)

    parquet_path = corpus_dir / f"master_corpus_{timestamp}.parquet"
    df.to_parquet(parquet_path, index=False)

    decisions_path = corpus_dir / f"normalization_decisions_{timestamp}.csv"
    decisions_df.to_csv(decisions_path, index=False)

    n_amendments = decisions_df["is_amendment"].sum()
    n_struck = decisions_df["strikethrough_detected"].sum()
    n_pdf_flag = decisions_df["strikethrough_limitation_flag"].sum()
    n_no_text = (decisions_df["notes"] == "no_text_available").sum()
    avg_removed = decisions_df["words_removed_pct"].mean()

    summary_path = corpus_dir / f"normalization_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("CORPUS NORMALIZATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Input: {input_csv}\n\n")
        f.write(f"Total bills processed:          {len(df)}\n")
        f.write(f"Bills with no text:             {n_no_text}\n")
        f.write(f"Bills identified as amendments: {n_amendments}\n")
        f.write(f"Bills with strikethrough found: {n_struck}\n")
        f.write(f"PDF/plain text limitation flag: {n_pdf_flag}\n")
        f.write(f"Avg words removed per bill:     {avg_removed:.1f}%\n\n")
        f.write("METHODOLOGICAL NOTE\n")
        f.write("-" * 60 + "\n")
        f.write(
            "Strikethrough text represents statutory deletions in amendment bills.\n"
            "Per research design decision (2026-03-07), strikethrough content is\n"
            "removed from normalized_text for HTML-format bills only.\n"
            "raw_text always preserves the original extracted content.\n\n"
            "For PDF and plain text bills, strikethrough cannot be detected.\n"
            "These bills are flagged in the decisions log under\n"
            "'strikethrough_limitation_flag'. See data quality report section 3.2\n"
            "for implications and sensitivity analysis recommendations.\n"
        )

    print(f"\n{'='*60}")
    print("NORMALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Bills processed:        {len(df)}")
    print(f"Amendments detected:    {n_amendments}")
    print(f"Strikethrough removed:  {n_struck} bills")
    print(f"PDF limitation flags:   {n_pdf_flag} bills")
    print(f"No text available:      {n_no_text} bills")
    print(f"Avg words removed:      {avg_removed:.1f}%")
    print(f"\nOutputs:")
    print(f"  Master corpus:        {parquet_path}")
    print(f"  Decisions log:        {decisions_path}")
    print(f"  Summary report:       {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
