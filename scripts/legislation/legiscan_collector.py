#!/usr/bin/env python3
"""
LegiScan Bill Collector
Collects school shooting-related bills from all 50 states.

API compliance: Uses dataset_hash values to avoid redundant downloads.
Per LegiScan policy, getDataset is only called when the hash has changed.
"""

import os
import sys
import json
import csv
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv

from legiscan_api import LegiScanAPI, DatasetHashCache
from text_processor import TextProcessor
from utils import ProgressTracker, ErrorLogger, setup_logging, load_keywords


class BillCollector:

    STATUS_ENROLLED = 3
    STATUS_PASSED = 4
    EVENT_CHAPTERED = 8

    STATE_IDS = {
        'AL': 1, 'AK': 2, 'AZ': 3, 'AR': 4, 'CA': 5, 'CO': 6, 'CT': 7,
        'DE': 8, 'FL': 9, 'GA': 10, 'HI': 11, 'ID': 12, 'IL': 13, 'IN': 14,
        'IA': 15, 'KS': 16, 'KY': 17, 'LA': 18, 'ME': 19, 'MD': 20, 'MA': 21,
        'MI': 22, 'MN': 23, 'MS': 24, 'MO': 25, 'MT': 26, 'NE': 27, 'NV': 28,
        'NH': 29, 'NJ': 30, 'NM': 31, 'NY': 32, 'NC': 33, 'ND': 34, 'OH': 35,
        'OK': 36, 'OR': 37, 'PA': 38, 'RI': 39, 'SC': 40, 'SD': 41, 'TN': 42,
        'TX': 43, 'UT': 44, 'VT': 45, 'VA': 46, 'WA': 47, 'WV': 48, 'WI': 49,
        'WY': 50
    }

    def __init__(
        self,
        api_key: str,
        keywords: List[str],
        start_year: int,
        end_year: int,
        output_dir: Path,
        cache_dir: Path,
        rate_limit_delay: float = 0.5
    ):
        self.api = LegiScanAPI(api_key, rate_limit_delay)
        self.text_processor = TextProcessor()
        self.keywords = keywords
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir

        self.datasets_cache_dir = cache_dir / "datasets"
        self.hash_cache_file = cache_dir / "dataset_hashes.json"
        self.html_dir = output_dir / "bill_texts" / "html"
        self.txt_dir = output_dir / "bill_texts" / "txt"
        self.raw_bills_dir = output_dir / "raw_responses" / "bills"
        self.logs_dir = Path("logs")

        for d in [
            self.datasets_cache_dir, self.html_dir, self.txt_dir,
            self.raw_bills_dir, self.logs_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.hash_cache = DatasetHashCache(self.hash_cache_file)
        self.progress_tracker = ProgressTracker(self.logs_dir / "processed_datasets.json")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.error_logger = ErrorLogger(self.logs_dir / f"errors_{timestamp}.txt")
        self.logger = setup_logging(self.logs_dir)

        self.csv_file = None
        self.csv_writer = None

    def is_final_status(self, bill: Dict) -> bool:
        status = bill.get('status', 0)
        if status in [self.STATUS_ENROLLED, self.STATUS_PASSED]:
            return True
        progress = bill.get('progress', [])
        if any(step.get('event') == self.EVENT_CHAPTERED for step in progress):
            return True
        return False

    def get_status_text(self, bill: Dict) -> str:
        status = bill.get('status', 0)
        progress = bill.get('progress', [])
        status_map = {
            0: 'N/A', 1: 'Introduced', 2: 'Engrossed',
            3: 'Enrolled', 4: 'Passed', 5: 'Vetoed', 6: 'Failed'
        }
        if any(step.get('event') == self.EVENT_CHAPTERED for step in progress):
            return 'Enacted'
        return status_map.get(status, f'Status {status}')

    def in_date_range(self, date_str: str) -> bool:
        if not date_str or date_str == '0000-00-00':
            return False
        try:
            year = int(date_str.split('-')[0])
            return self.start_year <= year <= self.end_year
        except Exception:
            return False

    def download_dataset_if_changed(self, dataset_info: Dict) -> Optional[Path]:
        """
        Core compliance method: checks dataset_hash before downloading.
        Only calls getDataset if the hash has changed since last run.
        """
        session_id = dataset_info['session_id']
        access_key = dataset_info['access_key']
        session_name = dataset_info['session_name']
        new_hash = dataset_info.get('dataset_hash', '')

        cache_file = self.datasets_cache_dir / f"session_{session_id}.zip"

        if cache_file.exists() and not self.hash_cache.has_changed(session_id, new_hash):
            self.logger.info(f"⏭️  Skipping {session_name} — hash unchanged")
            return cache_file

        if cache_file.exists():
            self.logger.info(f"🔄 Re-downloading {session_name} — hash changed")
        else:
            self.logger.info(f"⬇️  Downloading {session_name}")

        try:
            zip_bytes = self.api.get_dataset(session_id, access_key)
            with open(cache_file, 'wb') as f:
                f.write(zip_bytes)
            self.hash_cache.update(session_id, new_hash, session_name)
            self.logger.info(f"✅ Cached {session_name} ({len(zip_bytes):,} bytes)")
            return cache_file
        except Exception as e:
            self.logger.error(f"❌ Failed to download {session_name}: {e}")
            self.error_logger.log_dataset_failure(session_id, str(e))
            return None

    def process_bill(self, bill: Dict, zf: zipfile.ZipFile) -> Optional[Dict]:
        try:
            bill_id = bill.get('bill_id')
            state = bill.get('state', '')
            bill_number = bill.get('bill_number', '')
            title = bill.get('title', '')
            url = bill.get('url', '')
            last_action_date = bill.get('status_date', '')

            if not self.is_final_status(bill):
                return None
            if not self.in_date_range(last_action_date):
                return None

            status_text = self.get_status_text(bill)

            raw_bill_file = self.raw_bills_dir / f"bill_{bill_id}.json"
            with open(raw_bill_file, 'w') as f:
                json.dump({'bill': bill}, f, indent=2)

            texts = bill.get('texts', [])
            html_filename = ""
            txt_filename = ""
            text_url = texts[0].get('url', '') if texts else ''
            search_method = "metadata"
            search_text = ""

            if texts:
                sorted_texts = sorted(texts, key=lambda x: (
                    0 if x.get('mime_id') == 1 else
                    2 if x.get('mime_id') == 2 else 1
                ))
                best_text = sorted_texts[0]
                doc_id = best_text.get('doc_id')
                if doc_id:
                    text_files = [
                        f for f in zf.namelist()
                        if f'/text/{doc_id}.json' in f or f == f'text/{doc_id}.json'
                    ]
                    if text_files:
                        try:
                            with zf.open(text_files[0]) as f:
                                text_data = json.load(f)
                            text_obj = text_data.get('text', {})
                            doc_base64 = text_obj.get('doc', '')
                            mime_type = text_obj.get('mime', 'text/html')
                            if doc_base64:
                                doc_bytes, file_ext = self.text_processor.decode_bill_text(doc_base64, mime_type)
                                html_filename, txt_filename = self.text_processor.save_bill_text(
                                    doc_bytes, file_ext, state, bill_number,
                                    self.html_dir, self.txt_dir
                                )
                                search_text = self.text_processor.extract_text_from_bytes(doc_bytes, file_ext)
                                search_method = "full_text"
                        except Exception as e:
                            self.logger.warning(f"Could not extract text for {state} {bill_number}: {e}")

            if not search_text:
                search_parts = [title, bill.get('description', '')]
                for hist_item in bill.get('history', []):
                    action = hist_item.get('action', '')
                    if action:
                        search_parts.append(action)
                search_text = ' '.join(search_parts)
                search_method = "metadata"

            matched_keywords = self.text_processor.search_keywords(search_text, self.keywords)
            if not matched_keywords:
                return None

            return {
                'State': state,
                'Bill Number': bill_number,
                'Title': title,
                'LegiScan Bill Main Page URL': url,
                'LegiScan Text Page URL': text_url,
                'Status': status_text,
                'Last Action Date': last_action_date,
                'Matched Keywords': ', '.join(matched_keywords),
                'Search Method': search_method,
                'Bill Text HTML File': html_filename,
                'Bill Text TXT File': txt_filename,
            }

        except Exception as e:
            self.logger.error(f"Failed to process bill {bill.get('bill_id')}: {e}")
            self.error_logger.log_bill_failure(
                bill.get('bill_id', 0), bill.get('state', ''),
                bill.get('bill_number', ''), str(e)
            )
            return None

    def process_dataset(self, zip_path: Path, session_name: str) -> int:
        bills_found = 0
        stats = {'total_bills': 0, 'final_status_bills': 0, 'matched_bills': 0,
                 'full_text_matches': 0, 'metadata_matches': 0}
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                bill_files = [f for f in zf.namelist() if '/bill/' in f and f.endswith('.json')]
                stats['total_bills'] = len(bill_files)
                self.logger.info(f"Processing {len(bill_files)} bills from {session_name}")
                for i, bill_file in enumerate(bill_files, 1):
                    if i % 100 == 0:
                        self.logger.info(f"  Progress: {i}/{len(bill_files)}")
                    try:
                        with zf.open(bill_file) as f:
                            bill_data = json.load(f)
                        bill = bill_data.get('bill', {})
                        if self.is_final_status(bill):
                            stats['final_status_bills'] += 1
                        bill_result = self.process_bill(bill, zf)
                        if bill_result:
                            self.csv_writer.writerow(bill_result)
                            bills_found += 1
                            stats['matched_bills'] += 1
                            if bill_result['Search Method'] == 'full_text':
                                stats['full_text_matches'] += 1
                            else:
                                stats['metadata_matches'] += 1
                            self.logger.info(
                                f"  ✓ {bill_result['State']} {bill_result['Bill Number']} "
                                f"| {bill_result['Matched Keywords']} | {bill_result['Search Method']}"
                            )
                    except Exception as e:
                        self.logger.warning(f"Error processing {bill_file}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error reading dataset ZIP: {e}")
            raise

        self.logger.info(f"  Bills total: {stats['total_bills']} | "
                         f"Final status: {stats['final_status_bills']} | "
                         f"Matched: {stats['matched_bills']}")
        return bills_found

    def run(self, test_state: Optional[str] = None, test_year: Optional[int] = None, force: bool = False):
        self.logger.info("=" * 70)
        self.logger.info("LEGISCAN BILL COLLECTION")
        self.logger.info(f"Keywords: {len(self.keywords)} terms | Years: {self.start_year}-{self.end_year}")
        self.logger.info(f"Test mode: state={test_state or 'ALL'} year={test_year or 'ALL'}")
        self.logger.info("=" * 70)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f"results_{timestamp}.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            'State', 'Bill Number', 'Title', 'LegiScan Bill Main Page URL',
            'LegiScan Text Page URL', 'Status', 'Last Action Date',
            'Matched Keywords', 'Search Method', 'Bill Text HTML File', 'Bill Text TXT File'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

        all_datasets = []
        for year in range(self.start_year, self.end_year + 1):
            if test_year and year != test_year:
                continue
            self.logger.info(f"Fetching dataset list for {year}...")
            datasets = self.api.get_dataset_list(year=year)
            if test_state:
                test_state_id = self.STATE_IDS.get(test_state.upper())
                datasets = [d for d in datasets if d.get('state_id') == test_state_id]
            all_datasets.extend(datasets)

        self.logger.info(f"Found {len(all_datasets)} total datasets")

        total_bills_found = 0
        for i, dataset_info in enumerate(all_datasets, 1):
            session_id = dataset_info['session_id']
            session_name = dataset_info['session_name']
            self.logger.info(f"\n[{i}/{len(all_datasets)}] {session_name}")

            if not force and self.progress_tracker.is_processed(session_id):
                stored_hash = self.hash_cache.get_hash(session_id)
                current_hash = dataset_info.get('dataset_hash', '')
                if stored_hash == current_hash:
                    self.logger.info("  Skipping — already processed, hash unchanged")
                    continue

            zip_path = self.download_dataset_if_changed(dataset_info)
            if not zip_path:
                continue

            try:
                bills_found = self.process_dataset(zip_path, session_name)
                total_bills_found += bills_found
                self.progress_tracker.mark_processed(session_id, bills_found)
                self.logger.info(f"  Found {bills_found} bills (running total: {total_bills_found})")
            except Exception as e:
                self.logger.error(f"  Failed: {e}")
                self.error_logger.log_dataset_failure(session_id, str(e))
                continue

        self.csv_file.close()
        self.logger.info("\n" + "=" * 70)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info(f"Total bills found: {total_bills_found}")
        self.logger.info(f"Output: {csv_path}")
        self.logger.info(f"API calls used: {self.api.get_api_usage()}")
        self.logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='LegiScan Bill Collector')
    parser.add_argument('--test-state', help='Test with single state (e.g. CO)')
    parser.add_argument('--test-year', type=int, help='Test with single year (e.g. 2023)')
    parser.add_argument('--force', action='store_true', help='Force reprocess all datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')
    args = parser.parse_args()

    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)

    api_key = os.getenv('LEGISCAN_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        api_key = input("Enter your LegiScan API key: ").strip()
    if not api_key:
        print("Error: API key required")
        sys.exit(1)

    start_year = int(os.getenv('START_YEAR', 2010))
    end_year = int(os.getenv('END_YEAR', 2025))
    rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', 0.5))

    project_dir = Path(__file__).parent.parent.parent
    keywords_file = project_dir / 'input.txt'
    keywords = load_keywords(keywords_file)
    print(f"Loaded {len(keywords)} keywords")

    if args.dry_run:
        print(f"\nDRY RUN: {start_year}-{end_year} | state={args.test_state or 'ALL'} | year={args.test_year or 'ALL'}")
        sys.exit(0)

    output_dir = project_dir / 'output'
    cache_dir = output_dir / 'cache'
    output_dir.mkdir(parents=True, exist_ok=True)

    collector = BillCollector(
        api_key=api_key,
        keywords=keywords,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        cache_dir=cache_dir,
        rate_limit_delay=rate_limit_delay,
    )
    collector.run(
        test_state=args.test_state,
        test_year=args.test_year,
        force=args.force,
    )


if __name__ == '__main__':
    main()
