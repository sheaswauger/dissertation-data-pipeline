#!/usr/bin/env python3
"""
LegiScan Bill Collector
Main script for collecting school shooting-related bills from all 50 states.
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

# Import our modules
from legiscan_api import LegiScanAPI
from text_processor import TextProcessor
from utils import ProgressTracker, ErrorLogger, setup_logging, load_keywords


class BillCollector:
    """Main bill collection orchestrator"""

    # Status codes for final bills
    STATUS_ENROLLED = 3
    STATUS_PASSED = 4
    EVENT_CHAPTERED = 8

    # State abbreviation to ID mapping (partial list for common states)
    STATE_IDS = {
        'AL': 1, 'AK': 2, 'AZ': 3, 'AR': 4, 'CA': 5, 'CO': 6, 'CT': 7, 'DE': 8,
        'FL': 9, 'GA': 10, 'HI': 11, 'ID': 12, 'IL': 13, 'IN': 14, 'IA': 15,
        'KS': 16, 'KY': 17, 'LA': 18, 'ME': 19, 'MD': 20, 'MA': 21, 'MI': 22,
        'MN': 23, 'MS': 24, 'MO': 25, 'MT': 26, 'NE': 27, 'NV': 28, 'NH': 29,
        'NJ': 30, 'NM': 31, 'NY': 32, 'NC': 33, 'ND': 34, 'OH': 35, 'OK': 36,
        'OR': 37, 'PA': 38, 'RI': 39, 'SC': 40, 'SD': 41, 'TN': 42, 'TX': 43,
        'UT': 44, 'VT': 45, 'VA': 46, 'WA': 47, 'WV': 48, 'WI': 49, 'WY': 50
    }

    def __init__(
        self,
        api_key: str,
        keywords: List[str],
        start_year: int,
        end_year: int,
        output_dir: Path,
        rate_limit_delay: float = 0.5
    ):
        self.api = LegiScanAPI(api_key, rate_limit_delay)
        self.text_processor = TextProcessor()
        self.keywords = keywords
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = output_dir

        # Setup directories
        self.datasets_cache_dir = output_dir / "datasets_cache"
        self.html_dir = output_dir / "bill_texts" / "html"
        self.txt_dir = output_dir / "bill_texts" / "txt"
        self.raw_bills_dir = output_dir / "raw_responses" / "bills"
        self.raw_datasets_dir = output_dir / "raw_responses" / "datasets"
        self.logs_dir = Path("logs")

        for d in [self.datasets_cache_dir, self.html_dir, self.txt_dir,
                  self.raw_bills_dir, self.raw_datasets_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Setup tracking
        self.progress_tracker = ProgressTracker(self.logs_dir / "processed_datasets.json")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.error_logger = ErrorLogger(self.logs_dir / f"errors_{timestamp}.txt")

        # Results storage
        self.csv_file = None
        self.csv_writer = None

        self.logger = setup_logging(self.logs_dir)

    def is_final_status(self, bill: Dict) -> bool:
        """Check if bill has a final status (enrolled/passed/enacted)"""
        status = bill.get('status', 0)

        # Check status codes
        if status in [self.STATUS_ENROLLED, self.STATUS_PASSED]:
            return True

        # Check for chaptered/enacted in progress
        progress = bill.get('progress', [])
        if any(step.get('event') == self.EVENT_CHAPTERED for step in progress):
            return True

        return False

    def get_status_text(self, bill: Dict) -> str:
        """Get human-readable status text"""
        status = bill.get('status', 0)
        progress = bill.get('progress', [])

        status_map = {
            0: 'N/A',
            1: 'Introduced',
            2: 'Engrossed',
            3: 'Enrolled',
            4: 'Passed',
            5: 'Vetoed',
            6: 'Failed'
        }

        # Check for chaptered
        if any(step.get('event') == self.EVENT_CHAPTERED for step in progress):
            return 'Enacted'

        return status_map.get(status, f'Status {status}')

    def in_date_range(self, date_str: str) -> bool:
        """Check if date is in our target range"""
        if not date_str or date_str == '0000-00-00':
            return False

        try:
            year = int(date_str.split('-')[0])
            return self.start_year <= year <= self.end_year
        except:
            return False

    def download_dataset(self, dataset_info: Dict) -> Optional[Path]:
        """Download and cache a dataset"""
        session_id = dataset_info['session_id']
        access_key = dataset_info['access_key']
        session_name = dataset_info['session_name']

        # Check if already cached
        cache_file = self.datasets_cache_dir / f"session_{session_id}.zip"
        if cache_file.exists():
            self.logger.info(f"Using cached dataset: {session_name}")
            return cache_file

        self.logger.info(f"Downloading dataset: {session_name}")
        try:
            zip_bytes = self.api.get_dataset(session_id, access_key)

            # Save to cache
            with open(cache_file, 'wb') as f:
                f.write(zip_bytes)

            self.logger.info(f"Cached dataset ({len(zip_bytes)} bytes)")
            return cache_file

        except Exception as e:
            self.logger.error(f"Failed to download dataset {session_id}: {e}")
            self.error_logger.log_dataset_failure(session_id, str(e))
            return None

    def process_bill(self, bill: Dict, zf: zipfile.ZipFile) -> Optional[Dict]:
        """Process a single bill from dataset"""
        try:
            bill_id = bill.get('bill_id')
            state = bill.get('state', '')
            bill_number = bill.get('bill_number', '')
            title = bill.get('title', '')
            url = bill.get('url', '')
            last_action_date = bill.get('status_date', '')

            # Check final status
            if not self.is_final_status(bill):
                return None

            # Check date range
            if not self.in_date_range(last_action_date):
                return None

            # Get status text
            status_text = self.get_status_text(bill)

            # Save raw bill JSON
            raw_bill_file = self.raw_bills_dir / f"bill_{bill_id}.json"
            with open(raw_bill_file, 'w') as f:
                json.dump({'bill': bill}, f, indent=2)

            # HYBRID APPROACH: Try full text first, fall back to title/description
            texts = bill.get('texts', [])
            html_filename = ""
            txt_filename = ""
            text_url = texts[0].get('url', '') if texts else ''
            search_method = "metadata"  # Default fallback
            search_text = ""

            # Try to get full bill text
            if texts:
                sorted_texts = sorted(texts, key=lambda x: (
                    0 if x.get('mime_id') == 1 else  # HTML
                    2 if x.get('mime_id') == 2 else  # PDF
                    1  # Others
                ))

                best_text = sorted_texts[0]
                doc_id = best_text.get('doc_id')

                if doc_id:
                    # Try to extract text from dataset
                    text_files = [f for f in zf.namelist() if f'/text/{doc_id}.json' in f or f == f'text/{doc_id}.json']

                    if text_files:
                        text_file = text_files[0]
                        try:
                            with zf.open(text_file) as f:
                                text_data = json.load(f)
                                text_obj = text_data.get('text', {})
                                doc_base64 = text_obj.get('doc', '')
                                mime_type = text_obj.get('mime', 'text/html')

                                if doc_base64:
                                    # Decode and extract text
                                    doc_bytes, file_ext = self.text_processor.decode_bill_text(doc_base64, mime_type)

                                    # Save HTML + TXT versions
                                    html_filename, txt_filename = self.text_processor.save_bill_text(
                                        doc_bytes, file_ext, state, bill_number, self.html_dir, self.txt_dir
                                    )

                                    # Extract plain text for searching
                                    search_text = self.text_processor.extract_text_from_bytes(doc_bytes, file_ext)
                                    search_method = "full_text"
                        except Exception as e:
                            self.logger.warning(f"Could not extract text for {state} {bill_number}: {e}")

            # FALLBACK: If no full text, search title + description + history
            if not search_text:
                search_parts = [
                    title,
                    bill.get('description', ''),
                ]
                # Add history actions
                history = bill.get('history', [])
                for hist_item in history:
                    action = hist_item.get('action', '')
                    if action:
                        search_parts.append(action)

                search_text = ' '.join(search_parts)
                search_method = "metadata"

            # Search for keywords in the text (full or metadata)
            matched_keywords = self.text_processor.search_keywords(search_text, self.keywords)

            if not matched_keywords:
                # No keywords matched - skip this bill
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
                'Bill Text TXT File': txt_filename
            }

        except Exception as e:
            self.logger.error(f"Failed to process bill {bill.get('bill_id')}: {e}")
            self.error_logger.log_bill_failure(
                bill.get('bill_id', 0),
                bill.get('state', ''),
                bill.get('bill_number', ''),
                str(e)
            )
            return None

    def process_dataset(self, zip_path: Path, session_name: str) -> int:
        """Process a single dataset and return count of bills found"""
        bills_found = 0

        # Statistics tracking
        stats = {
            'total_bills': 0,
            'final_status_bills': 0,
            'matched_bills': 0,
            'full_text_matches': 0,
            'metadata_matches': 0
        }

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List all bill JSON files (handle both formats: bill/*.json and STATE/SESSION/bill/*.json)
                bill_files = [f for f in zf.namelist()
                              if '/bill/' in f and f.endswith('.json')]

                stats['total_bills'] = len(bill_files)
                self.logger.info(f"Processing {len(bill_files)} bills from {session_name}")

                for i, bill_file in enumerate(bill_files, 1):
                    if i % 100 == 0:
                        self.logger.info(f"  Progress: {i}/{len(bill_files)} bills checked")

                    try:
                        # Read bill JSON
                        with zf.open(bill_file) as f:
                            bill_data = json.load(f)
                            bill = bill_data.get('bill', {})

                            # Track if bill has final status
                            if self.is_final_status(bill):
                                stats['final_status_bills'] += 1

                            # Process bill
                            bill_result = self.process_bill(bill, zf)

                            if bill_result:
                                # Write to CSV
                                self.csv_writer.writerow(bill_result)
                                bills_found += 1
                                stats['matched_bills'] += 1

                                # Track search method
                                if bill_result['Search Method'] == 'full_text':
                                    stats['full_text_matches'] += 1
                                else:
                                    stats['metadata_matches'] += 1

                                self.logger.info(
                                    f"  ✓ Found: {bill_result['State']} {bill_result['Bill Number']} "
                                    f"(Keywords: {bill_result['Matched Keywords']}, Method: {bill_result['Search Method']})"
                                )

                    except Exception as e:
                        self.logger.warning(f"Error processing {bill_file}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"Error reading dataset ZIP: {e}")
            raise

        # Log summary statistics
        self.logger.info(f"  --- Dataset Summary ---")
        self.logger.info(f"  Total bills in dataset: {stats['total_bills']}")
        self.logger.info(f"  Bills with final status: {stats['final_status_bills']}")
        self.logger.info(f"  Bills matched keywords: {stats['matched_bills']}")
        if stats['matched_bills'] > 0:
            self.logger.info(f"    - Found via full text: {stats['full_text_matches']}")
            self.logger.info(f"    - Found via metadata: {stats['metadata_matches']}")
        if stats['final_status_bills'] > 0:
            match_rate = (stats['matched_bills'] / stats['final_status_bills']) * 100
            self.logger.info(f"  Match rate: {match_rate:.2f}% of final bills")

        return bills_found

    def run(self, test_state: Optional[str] = None, test_year: Optional[int] = None, force: bool = False):
        """Main execution method"""
        self.logger.info("=" * 80)
        self.logger.info("LEGISCAN BILL COLLECTION - BULK DATASET METHOD")
        self.logger.info("=" * 80)
        self.logger.info(f"Keywords: {', '.join(self.keywords)}")
        self.logger.info(f"Year range: {self.start_year}-{self.end_year}")
        self.logger.info(f"Test mode: {test_state or 'ALL'} - {test_year or 'ALL YEARS'}")
        self.logger.info("=" * 80)

        # Setup CSV output
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f"final_results_{timestamp}.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')

        fieldnames = [
            'State', 'Bill Number', 'Title',
            'LegiScan Bill Main Page URL', 'LegiScan Text Page URL',
            'Status', 'Last Action Date', 'Matched Keywords', 'Search Method',
            'Bill Text HTML File', 'Bill Text TXT File'
        ]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()

        # Get datasets
        self.logger.info("\nFetching dataset list...")
        all_datasets = []

        for year in range(self.start_year, self.end_year + 1):
            # Skip if test year specified
            if test_year and year != test_year:
                continue

            self.logger.info(f"  Checking year {year}...")
            datasets = self.api.get_dataset_list(year=year)

            # Filter by test state if specified
            if test_state:
                test_state_id = self.STATE_IDS.get(test_state.upper())
                datasets = [d for d in datasets if d.get('state_id') == test_state_id]

            all_datasets.extend(datasets)

        self.logger.info(f"\n✓ Found {len(all_datasets)} total datasets")

        # Process each dataset
        total_bills_found = 0
        datasets_processed = 0

        for i, dataset_info in enumerate(all_datasets, 1):
            session_id = dataset_info['session_id']
            session_name = dataset_info['session_name']

            self.logger.info(f"\n[{i}/{len(all_datasets)}] Processing: {session_name}")

            # Check if already processed (unless force mode)
            if not force and self.progress_tracker.is_processed(session_id):
                self.logger.info(f"  Skipping (already processed)")
                continue

            # Download dataset
            zip_path = self.download_dataset(dataset_info)
            if not zip_path:
                continue

            # Process dataset
            try:
                bills_found = self.process_dataset(zip_path, session_name)
                total_bills_found += bills_found
                datasets_processed += 1

                # Mark as processed
                self.progress_tracker.mark_processed(session_id, bills_found)

                self.logger.info(f"  ✓ Found {bills_found} matching bills (total: {total_bills_found})")

            except Exception as e:
                self.logger.error(f"  ✗ Failed to process dataset: {e}")
                self.error_logger.log_dataset_failure(session_id, str(e))
                continue

        # Close CSV
        self.csv_file.close()

        # Final summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COLLECTION COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Datasets processed: {datasets_processed}")
        self.logger.info(f"Total bills found: {total_bills_found}")
        self.logger.info(f"CSV output: {csv_path}")
        self.logger.info(f"HTML texts: {self.html_dir}")
        self.logger.info(f"TXT texts: {self.txt_dir}")
        self.logger.info(f"API calls used: {self.api.get_api_usage()}")
        self.logger.info(f"Errors: {self.error_logger.get_summary()}")
        self.logger.info("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LegiScan Bill Collector')
    parser.add_argument('--test-state', help='Test with a single state (e.g., CA)')
    parser.add_argument('--test-year', type=int, help='Test with a single year (e.g., 2023)')
    parser.add_argument('--force', action='store_true', help='Force reprocess already completed datasets')
    parser.add_argument('--dry-run', action='store_true', help='Show plan without executing')

    args = parser.parse_args()

    # Load environment variables
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

    # Get API key
    api_key = os.getenv('LEGISCAN_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        api_key = input("Enter your LegiScan API key: ").strip()
        if not api_key:
            print("Error: API key is required")
            sys.exit(1)

    # Get configuration
    start_year = int(os.getenv('START_YEAR', 2010))
    end_year = int(os.getenv('END_YEAR', 2025))
    rate_limit_delay = float(os.getenv('RATE_LIMIT_DELAY', 0.5))

    # Load keywords
    project_dir = Path(__file__).parent.parent
    keywords_file = project_dir / 'input.txt'
    keywords = load_keywords(keywords_file)

    print(f"Loaded {len(keywords)} keywords: {', '.join(keywords)}")

    if args.dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"Would collect bills from {start_year}-{end_year}")
        print(f"Test state: {args.test_state or 'ALL'}")
        print(f"Test year: {args.test_year or 'ALL'}")
        print(f"Force reprocess: {args.force}")
        sys.exit(0)

    # Run collector
    output_dir = project_dir / 'output'
    collector = BillCollector(
        api_key=api_key,
        keywords=keywords,
        start_year=start_year,
        end_year=end_year,
        output_dir=output_dir,
        rate_limit_delay=rate_limit_delay
    )

    collector.run(
        test_state=args.test_state,
        test_year=args.test_year,
        force=args.force
    )


if __name__ == '__main__':
    main()
