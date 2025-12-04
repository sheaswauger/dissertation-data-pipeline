#!/usr/bin/env python3
"""
LegiScan Data Enricher

Fills missing data in legiscan_combined.csv using LegiScan API (primary)
and Open States API (fallback).

Usage:
    python legiscan_enricher.py [--limit N]
"""

import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Check for required libraries
MISSING_LIBS = []
try:
    import pandas as pd
except ImportError:
    MISSING_LIBS.append("pandas")

try:
    import requests
except ImportError:
    MISSING_LIBS.append("requests")

if MISSING_LIBS:
    print(f"ERROR: Missing required libraries: {', '.join(MISSING_LIBS)}")
    print(f"\nInstall them with:")
    print(f"  pip install {' '.join(MISSING_LIBS)}")
    sys.exit(1)


# Constants
BASE_DIR = Path.home() / "Desktop" / "Legiscan Data"
INPUT_CSV = BASE_DIR / "legiscan_combined.csv"
OUTPUT_CSV = BASE_DIR / "legiscan_enriched.csv"
OUTPUT_XLSX = BASE_DIR / "legiscan_enriched.xlsx"
API_KEYS_FILE = BASE_DIR / "API keys.txt"
CACHE_FILE = BASE_DIR / "api_cache.json"

# API Configuration
LEGISCAN_BASE_URL = "https://api.legiscan.com/"
OPENSTATES_BASE_URL = "https://v3.openstates.org"
RATE_LIMIT_DELAY = 1.0  # seconds between LegiScan API calls
OPENSTATES_DELAY = 30.0  # seconds between OpenStates API calls (avoid rate limiting)
RATE_LIMIT_BACKOFF = 60.0  # seconds to wait on rate limit error

# Fields that can be enriched
ENRICHABLE_FIELDS = [
    "Title",
    "Legiscan Bill Main Page",
    "Legiscan Text Page",
    "Status",
    "Last Action Date"
]


class APICache:
    """Persistent cache for API responses."""

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.modified = False

    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        return {}

    def get(self, key: str) -> Optional[Dict]:
        """Get cached response."""
        return self.cache.get(key)

    def set(self, key: str, value: Dict):
        """Set cached response."""
        self.cache[key] = value
        self.modified = True

    def save(self):
        """Save cache to file."""
        if self.modified:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                print(f"\n✓ Cache saved to {self.cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")


def load_api_keys() -> Dict[str, str]:
    """Load API keys from file."""
    print(f"Loading API keys from: {API_KEYS_FILE}")

    if not API_KEYS_FILE.exists():
        print(f"ERROR: API keys file not found: {API_KEYS_FILE}")
        sys.exit(1)

    keys = {}
    with open(API_KEYS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                keys[key.strip()] = value.strip()

    if 'LEGISCAN_API_KEY' not in keys:
        print("ERROR: LEGISCAN_API_KEY not found in API keys file")
        sys.exit(1)

    if 'OPENSTATES_API_KEY' not in keys:
        print("Warning: OPENSTATES_API_KEY not found in API keys file")

    print(f"✓ Loaded {len(keys)} API key(s)\n")
    return keys


def normalize_bill_number(bill_number: str) -> List[str]:
    """
    Generate possible bill number format variations for API queries.

    Returns list of variations to try, in order of likelihood.
    """
    if not bill_number or pd.isna(bill_number):
        return []

    bill_num = str(bill_number).strip().upper()

    # Remove state prefix if present (e.g., "CA AB123" -> "AB123")
    parts = bill_num.split()
    if len(parts) > 1 and len(parts[0]) == 2:
        bill_num = ' '.join(parts[1:])

    variations = [bill_num]  # Original format

    # Try without spaces (e.g., "H.B. 123" -> "HB123")
    no_spaces = bill_num.replace(' ', '').replace('.', '')
    if no_spaces != bill_num:
        variations.append(no_spaces)

    # Try with space (e.g., "HB123" -> "HB 123")
    match = re.match(r'([A-Z]+)(\d+)', no_spaces)
    if match:
        with_space = f"{match.group(1)} {match.group(2)}"
        if with_space not in variations:
            variations.append(with_space)

    # Try with dots (e.g., "HB123" -> "H.B. 123")
    if match:
        prefix = match.group(1)
        number = match.group(2)
        if len(prefix) >= 2:
            dotted = '.'.join(prefix) + '. ' + number
            if dotted not in variations:
                variations.append(dotted)

    return variations


def query_legiscan(state: str, bill_number: str, api_key: str, cache: APICache) -> Optional[Dict]:
    """
    Query LegiScan API for bill information.

    Returns dict with bill data or None if not found.
    """
    if not state or pd.isna(state) or not bill_number or pd.isna(bill_number):
        return None

    state = str(state).strip().upper()
    bill_variations = normalize_bill_number(bill_number)

    for bill_var in bill_variations:
        cache_key = f"legiscan_{state}_{bill_var}"

        # Check cache first
        cached = cache.get(cache_key)
        if cached is not None:
            if cached.get('found'):
                return cached
            else:
                continue  # Try next variation

        # Query API
        try:
            params = {
                'key': api_key,
                'op': 'getBill',
                'state': state,
                'bill': bill_var
            }

            time.sleep(RATE_LIMIT_DELAY)
            response = requests.get(LEGISCAN_BASE_URL, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('status') == 'OK' and 'bill' in data:
                bill = data['bill']
                result = {
                    'found': True,
                    'api': 'legiscan',
                    'title': bill.get('title', ''),
                    'status': bill.get('status_desc', ''),
                    'last_action_date': bill.get('last_action_date', ''),
                    'bill_url': bill.get('url', ''),
                    'text_url': bill.get('text_url', ''),
                    'state_link': bill.get('state_link', '')
                }
                cache.set(cache_key, result)
                return result
            else:
                # Cache negative result
                cache.set(cache_key, {'found': False})

        except requests.exceptions.RequestException as e:
            print(f"  LegiScan API error for {state} {bill_var}: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error querying LegiScan for {state} {bill_var}: {e}")
            continue

    return None


def query_openstates(state: str, bill_number: str, api_key: str, cache: APICache) -> Optional[Dict]:
    """
    Query Open States API for bill information.

    Returns dict with bill data or None if not found.
    """
    if not api_key:
        return None

    if not state or pd.isna(state) or not bill_number or pd.isna(bill_number):
        return None

    # Map state names to abbreviations if needed
    state = str(state).strip().upper()
    if len(state) > 2:
        state = state[:2]  # Use first 2 chars if full name given

    bill_variations = normalize_bill_number(bill_number)

    for bill_var in bill_variations:
        cache_key = f"openstates_{state}_{bill_var}"

        # Check cache first
        cached = cache.get(cache_key)
        if cached is not None:
            if cached.get('found'):
                return cached
            else:
                continue  # Try next variation

        # Query API
        try:
            headers = {
                'X-API-KEY': api_key
            }

            # Search for bill
            url = f"{OPENSTATES_BASE_URL}/bills"
            params = {
                'jurisdiction': state,
                'identifier': bill_var
            }

            time.sleep(OPENSTATES_DELAY)
            response = requests.get(url, headers=headers, params=params, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                print(f"  Rate limited by OpenStates, waiting {RATE_LIMIT_BACKOFF}s...")
                time.sleep(RATE_LIMIT_BACKOFF)
                response = requests.get(url, headers=headers, params=params, timeout=30)

            response.raise_for_status()

            data = response.json()

            if data.get('results') and len(data['results']) > 0:
                bill = data['results'][0]

                # Get last action date
                last_action_date = ''
                if bill.get('actions'):
                    actions = sorted(bill['actions'], key=lambda x: x.get('date', ''), reverse=True)
                    if actions:
                        last_action_date = actions[0].get('date', '')

                # Get status
                status = bill.get('classification', [''])[0] if bill.get('classification') else ''

                result = {
                    'found': True,
                    'api': 'openstates',
                    'title': bill.get('title', ''),
                    'status': status,
                    'last_action_date': last_action_date,
                    'bill_url': f"https://openstates.org/{state}/bills/{bill.get('id', '')}" if bill.get('id') else '',
                    'text_url': '',  # Open States doesn't provide direct text URL
                    'openstates_url': bill.get('openstates_url', '')
                }
                cache.set(cache_key, result)
                return result
            else:
                # Cache negative result
                cache.set(cache_key, {'found': False})

        except requests.exceptions.RequestException as e:
            print(f"  OpenStates API error for {state} {bill_var}: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error querying OpenStates for {state} {bill_var}: {e}")
            continue

    return None


def needs_enrichment(row: pd.Series) -> bool:
    """Check if row has any missing or 'Not Found' data."""
    for field in ENRICHABLE_FIELDS:
        value = row.get(field, '')
        if pd.isna(value) or value == '' or str(value).strip().lower() == 'not found':
            return True
    return False


def enrich_row(row: pd.Series, api_keys: Dict[str, str], cache: APICache) -> Tuple[pd.Series, str, bool]:
    """
    Enrich a single row with missing data.

    Returns: (enriched_row, api_used, success)
    """
    if not needs_enrichment(row):
        return row, 'none', False

    state = row.get('State', '')
    bill_number = row.get('Bill Number', '')

    # Try LegiScan first
    legiscan_data = query_legiscan(state, bill_number, api_keys['LEGISCAN_API_KEY'], cache)

    if legiscan_data:
        # Fill missing fields from LegiScan
        enriched = row.copy()
        changed = False

        if pd.isna(enriched.get('Title')) or enriched.get('Title') == '' or str(enriched.get('Title')).strip().lower() == 'not found':
            if legiscan_data.get('title'):
                enriched['Title'] = legiscan_data['title']
                changed = True

        if pd.isna(enriched.get('Legiscan Bill Main Page')) or enriched.get('Legiscan Bill Main Page') == '':
            if legiscan_data.get('state_link'):
                enriched['Legiscan Bill Main Page'] = legiscan_data['state_link']
                changed = True
            elif legiscan_data.get('bill_url'):
                enriched['Legiscan Bill Main Page'] = legiscan_data['bill_url']
                changed = True

        if pd.isna(enriched.get('Legiscan Text Page')) or enriched.get('Legiscan Text Page') == '':
            if legiscan_data.get('text_url'):
                enriched['Legiscan Text Page'] = legiscan_data['text_url']
                changed = True

        if pd.isna(enriched.get('Status')) or enriched.get('Status') == '':
            if legiscan_data.get('status'):
                enriched['Status'] = legiscan_data['status']
                changed = True

        if pd.isna(enriched.get('Last Action Date')) or enriched.get('Last Action Date') == '':
            if legiscan_data.get('last_action_date'):
                enriched['Last Action Date'] = legiscan_data['last_action_date']
                changed = True

        return enriched, 'legiscan', changed

    # Try OpenStates as fallback (with 30s delay to avoid rate limiting)
    if 'OPENSTATES_API_KEY' in api_keys:
        openstates_data = query_openstates(state, bill_number, api_keys['OPENSTATES_API_KEY'], cache)
        if openstates_data:
            enriched = row.copy()
            changed = False

            if pd.isna(enriched.get('Title')) or enriched.get('Title') == '' or enriched.get('Title') == 'Not Found':
                if openstates_data.get('title'):
                    enriched['Title'] = openstates_data['title']
                    changed = True

            if pd.isna(enriched.get('Legiscan Bill Main Page')) or enriched.get('Legiscan Bill Main Page') == '':
                if openstates_data.get('bill_url'):
                    enriched['Legiscan Bill Main Page'] = openstates_data['bill_url']
                    changed = True

            if pd.isna(enriched.get('Legiscan Text Page')) or enriched.get('Legiscan Text Page') == '':
                if openstates_data.get('text_url'):
                    enriched['Legiscan Text Page'] = openstates_data['text_url']
                    changed = True

            if pd.isna(enriched.get('Status')) or enriched.get('Status') == '':
                if openstates_data.get('status'):
                    enriched['Status'] = openstates_data['status']
                    changed = True

            if pd.isna(enriched.get('Last Action Date')) or enriched.get('Last Action Date') == '':
                if openstates_data.get('last_action_date'):
                    enriched['Last Action Date'] = openstates_data['last_action_date']
                    changed = True

            return enriched, 'openstates', changed

    # No data found
    return row, 'none', False


def load_data() -> pd.DataFrame:
    """Load input CSV file."""
    print(f"Loading data from: {INPUT_CSV}\n")

    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"✓ Loaded {len(df)} rows\n")
    return df


def save_output(df: pd.DataFrame):
    """Save enriched data to CSV and Excel."""
    print(f"\nSaving output...")

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved CSV: {OUTPUT_CSV}")

    # Save Excel
    try:
        import openpyxl
        with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Bills')

            # Auto-adjust column widths
            worksheet = writer.sheets['Bills']
            for idx, col in enumerate(df.columns, 1):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                max_length = min(max_length, 50)
                worksheet.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = max_length + 2

        print(f"✓ Saved Excel: {OUTPUT_XLSX}")
    except ImportError:
        print(f"  Warning: openpyxl not installed, skipping Excel output")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("LEGISCAN DATA ENRICHER")
    print("="*80 + "\n")

    # Parse command line arguments
    limit = None
    if '--limit' in sys.argv:
        try:
            limit_idx = sys.argv.index('--limit')
            limit = int(sys.argv[limit_idx + 1])
            print(f"TEST MODE: Processing first {limit} rows needing enrichment\n")
        except (IndexError, ValueError):
            print("ERROR: --limit requires a numeric argument")
            sys.exit(1)

    # Load API keys
    api_keys = load_api_keys()

    # Load data
    df = load_data()

    # Initialize cache
    cache = APICache(CACHE_FILE)
    print(f"API cache: {len(cache.cache)} entries loaded\n")

    # Find rows needing enrichment
    needs_enrich = df.apply(needs_enrichment, axis=1)
    rows_to_process = df[needs_enrich].copy()

    print(f"Rows needing enrichment: {len(rows_to_process)} / {len(df)}")
    print(f"Rows with complete data: {len(df) - len(rows_to_process)}\n")

    if len(rows_to_process) == 0:
        print("No rows need enrichment. Exiting.")
        return

    # Apply limit if specified
    if limit:
        rows_to_process = rows_to_process.head(limit)
        print(f"Processing first {len(rows_to_process)} rows\n")

    # Process rows
    print("="*80)
    print("ENRICHING DATA")
    print("="*80 + "\n")

    stats = {
        'processed': 0,
        'enriched': 0,
        'via_legiscan': 0,
        'via_openstates': 0,
        'failed': 0
    }

    enriched_indices = []

    for idx, row in rows_to_process.iterrows():
        stats['processed'] += 1

        # Show progress
        if stats['processed'] % 10 == 0 or stats['processed'] == 1:
            print(f"[{stats['processed']}/{len(rows_to_process)}] Processing {row.get('State', '')} {row.get('Bill Number', '')}")

        enriched_row, api_used, changed = enrich_row(row, api_keys, cache)

        if changed:
            stats['enriched'] += 1
            enriched_indices.append(idx)
            df.loc[idx] = enriched_row

            if api_used == 'legiscan':
                stats['via_legiscan'] += 1
                print(f"  ✓ Enriched via LegiScan")
            elif api_used == 'openstates':
                stats['via_openstates'] += 1
                print(f"  ✓ Enriched via OpenStates")
        else:
            stats['failed'] += 1
            print(f"  ✗ No data found")

        # Periodically save cache
        if stats['processed'] % 50 == 0:
            cache.save()

    # Save final cache
    cache.save()

    # Save output
    save_output(df)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")
    print(f"Rows processed:       {stats['processed']}")
    print(f"Rows enriched:        {stats['enriched']}")
    print(f"  - via LegiScan:     {stats['via_legiscan']}")
    print(f"  - via OpenStates:   {stats['via_openstates']}")
    print(f"Rows still missing:   {stats['failed']}")
    print(f"\nSuccess rate:         {stats['enriched']/stats['processed']*100:.1f}%")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
