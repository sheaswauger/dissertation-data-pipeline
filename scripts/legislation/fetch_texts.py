#!/usr/bin/env python3
import os, sys, time, base64, argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd

LEGISCAN_BASE_URL = "https://api.legiscan.com/"

def get_bill_text_by_doc_id(api_key, doc_id, rate_limit_delay=0.5):
    params = {"key": api_key, "op": "getBillText", "id": doc_id}
    time.sleep(rate_limit_delay)
    response = requests.get(LEGISCAN_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "OK":
        return None, None
    text_obj = data.get("text", {})
    doc_base64 = text_obj.get("doc", "")
    mime = text_obj.get("mime", "text/html")
    if not doc_base64:
        return None, None
    return base64.b64decode(doc_base64), mime

def extract_plain_text(raw_bytes, mime):
    if "html" in mime:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw_bytes.decode("utf-8", errors="ignore"), "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except Exception:
            return raw_bytes.decode("utf-8", errors="ignore")
    elif "pdf" in mime:
        try:
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""
    else:
        return raw_bytes.decode("utf-8", errors="ignore")

def get_bill_id_from_url(url):
    if not url:
        return 0
    parts = str(url).rstrip("/").split("/")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return 0

def find_latest_results_csv(output_dir):
    csvs = sorted(output_dir.glob("results_*.csv"), reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No results_*.csv found in {output_dir}")
    return csvs[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent.parent
    load_dotenv(project_dir / ".env")
    api_key = os.getenv("LEGISCAN_API_KEY")
    if not api_key:
        print("ERROR: LEGISCAN_API_KEY not set")
        sys.exit(1)
    rate_limit_delay = float(os.getenv("RATE_LIMIT_DELAY", 0.5))

    output_dir = project_dir / "output"
    texts_dir = output_dir / "bill_texts" / "raw"
    plain_dir = output_dir / "bill_texts" / "plain"
    texts_dir.mkdir(parents=True, exist_ok=True)
    plain_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(args.input) if args.input else find_latest_results_csv(output_dir)
    print(f"Input: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} bills")
    if args.limit:
        df = df.head(args.limit)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"fetch_texts_{timestamp}.log"
    stats = {"total": len(df), "fetched": 0, "skipped": 0, "failed": 0, "no_text": 0}
    df["raw_text_file"] = ""
    df["plain_text_file"] = ""
    df["text_source"] = ""

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        state = row.get("State", "")
        bill_number = row.get("Bill Number", "")
        text_url = row.get("LegiScan Text Page URL", "")
        print(f"[{i}/{len(df)}] {state} {bill_number}", end=" ")
        safe_bill = str(bill_number).replace("/", "_").replace(" ", "_")
        plain_file = plain_dir / f"{state}_{safe_bill}.txt"
        raw_file = texts_dir / f"{state}_{safe_bill}.raw"
        if plain_file.exists() and not args.force:
            print("already fetched, skipping")
            df.at[idx, "plain_text_file"] = str(plain_file.name)
            df.at[idx, "text_source"] = "cached"
            stats["skipped"] += 1
            continue
        doc_id = get_bill_id_from_url(text_url)
        if not doc_id:
            print("no doc ID, skipping")
            stats["no_text"] += 1
            continue
        try:
            raw_bytes, mime = get_bill_text_by_doc_id(api_key, doc_id, rate_limit_delay)
            if not raw_bytes:
                print("no text returned")
                stats["no_text"] += 1
                continue
            with open(raw_file, "wb") as f:
                f.write(raw_bytes)
            plain_text = extract_plain_text(raw_bytes, mime)
            with open(plain_file, "w", encoding="utf-8") as f:
                f.write(plain_text)
            df.at[idx, "raw_text_file"] = str(raw_file.name)
            df.at[idx, "plain_text_file"] = str(plain_file.name)
            df.at[idx, "text_source"] = "legiscan_api"
            print(f"fetched ({len(plain_text.split()):,} words)")
            stats["fetched"] += 1
        except Exception as e:
            print(f"ERROR: {e}")
            stats["failed"] += 1
            with open(log_path, "a") as log:
                log.write(f"{state} {bill_number} | {e}\n")

    enriched_csv = output_dir / f"corpus_with_texts_{timestamp}.csv"
    df.to_csv(enriched_csv, index=False)
    print(f"\nDone: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats['failed']} failed")
    print(f"Output: {enriched_csv}")

if __name__ == "__main__":
    main()
