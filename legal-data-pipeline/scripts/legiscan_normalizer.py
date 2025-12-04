#!/usr/bin/env python3
"""
LegiScan Spreadsheet Normalizer

Processes LegiScan spreadsheets from ~/Desktop/Legiscan and combines them into
a single normalized dataset with consistent columns and formatting.

Output Schema:
    State | Bill Number | Title | Legiscan Bill Main Page | Legiscan Text Page |
    Status | Last Action Date | Bill ID

Usage:
    python legiscan_normalizer.py [--dry-run]
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import re

# Check for required libraries
MISSING_LIBS = []
try:
    import pandas as pd
except ImportError:
    MISSING_LIBS.append("pandas")

try:
    import openpyxl
except ImportError:
    MISSING_LIBS.append("openpyxl")

if MISSING_LIBS:
    print(f"ERROR: Missing required libraries: {', '.join(MISSING_LIBS)}")
    print(f"\nInstall them with:")
    print(f"  pip install {' '.join(MISSING_LIBS)}")
    sys.exit(1)


# Constants
INPUT_DIR = Path.home() / "Desktop" / "Legiscan"
OUTPUT_CSV = INPUT_DIR / "legiscan_combined.csv"
OUTPUT_XLSX = INPUT_DIR / "legiscan_combined.xlsx"

# Standard output columns
OUTPUT_COLUMNS = [
    "State",
    "Bill Number",
    "Title",
    "Legiscan Bill Main Page",
    "Legiscan Text Page",
    "Status",
    "Last Action Date",
    "Bill ID"
]

# Column mapping patterns (case-insensitive)
COLUMN_PATTERNS = {
    "State": ["state"],
    "Bill Number": ["bill", "number", "bill number", "bill no", "bill id", "billno"],
    "Title": ["title"],
    "Status": ["status"],
    "Last Action Date": ["last", "action", "date", "last action", "last date", "action date"],
    "Legiscan Text Page": ["text", "bill text", "text page", "text url"],
    "Legiscan Bill Main Page": ["page", "url", "link", "main page", "bill page", "bill url"]
}


def normalize_column_name(col: str) -> str:
    """Normalize column name for matching (lowercase, no extra whitespace)."""
    return re.sub(r'\s+', ' ', str(col).strip().lower())


def find_best_column_match(columns: List[str], patterns: List[str]) -> str:
    """
    Find the best matching column from a list based on patterns.

    Args:
        columns: List of column names from the dataframe
        patterns: List of pattern strings to match against

    Returns:
        Matched column name or empty string if no match
    """
    normalized_cols = {normalize_column_name(col): col for col in columns}

    # Try exact matches first
    for pattern in patterns:
        norm_pattern = normalize_column_name(pattern)
        if norm_pattern in normalized_cols:
            return normalized_cols[norm_pattern]

    # Try substring matches
    for pattern in patterns:
        norm_pattern = normalize_column_name(pattern)
        for norm_col, orig_col in normalized_cols.items():
            if all(word in norm_col for word in norm_pattern.split()):
                return orig_col

    return ""


def load_files(input_dir: Path) -> List[Tuple[str, pd.DataFrame, Dict[str, str]]]:
    """
    Load all CSV and Excel files from the input directory.

    Args:
        input_dir: Directory containing input files

    Returns:
        List of tuples: (filename, dataframe, column_mapping)
    """
    print(f"\n{'='*80}")
    print("STEP 1: LOADING FILES")
    print(f"{'='*80}\n")

    if not input_dir.exists():
        print(f"ERROR: Directory not found: {input_dir}")
        sys.exit(1)

    loaded_files = []
    file_patterns = ["*.csv", "*.xlsx", "*.xls"]

    for pattern in file_patterns:
        for file_path in sorted(input_dir.glob(pattern)):
            # Skip input.txt and any hidden files
            if file_path.name.startswith('.') or file_path.name == 'input.txt':
                continue

            print(f"Loading: {file_path.name}")

            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")

                # Map columns
                column_mapping = {}
                for std_col, patterns in COLUMN_PATTERNS.items():
                    matched_col = find_best_column_match(df.columns, patterns)
                    if matched_col:
                        column_mapping[matched_col] = std_col
                        print(f"  Mapped: '{matched_col}' → '{std_col}'")

                if column_mapping:
                    loaded_files.append((file_path.name, df, column_mapping))
                else:
                    print(f"  WARNING: No columns matched - skipping file")

                print()

            except Exception as e:
                print(f"  ERROR loading file: {e}\n")
                continue

    print(f"Successfully loaded {len(loaded_files)} file(s)\n")
    return loaded_files


def normalize_title(title: Any) -> str:
    """Convert title to title case."""
    if pd.isna(title) or title == "":
        return ""
    return str(title).strip().title()


def parse_date(date_value: Any) -> str:
    """
    Parse date value and convert to YYYY-MM-DD format.

    Args:
        date_value: Date value (string, datetime, or other)

    Returns:
        Date string in YYYY-MM-DD format or empty string
    """
    if pd.isna(date_value) or date_value == "":
        return ""

    # If already datetime
    if isinstance(date_value, (pd.Timestamp, datetime)):
        return date_value.strftime("%Y-%m-%d")

    date_str = str(date_value).strip()
    if not date_str:
        return ""

    # Try common date formats
    formats = [
        "%Y-%m-%d",      # 2020-01-15
        "%m/%d/%Y",      # 01/15/2020
        "%m/%d/%y",      # 01/15/20
        "%Y/%m/%d",      # 2020/01/15
        "%m-%d-%Y",      # 01-15-2020
        "%m-%d-%y",      # 01-15-20
        "%B %d, %Y",     # January 15, 2020
        "%b %d, %Y",     # Jan 15, 2020
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            # Handle 2-digit years (assume 2000s)
            if dt.year < 100:
                dt = dt.replace(year=dt.year + 2000)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    print(f"  WARNING: Could not parse date: '{date_str}'")
    return ""


def normalize_url(url: Any) -> str:
    """Normalize URL by stripping whitespace and converting to string."""
    if pd.isna(url) or url == "":
        return ""
    return str(url).strip()


def normalize_values(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize values in the dataframe.

    Args:
        df: Input dataframe
        column_mapping: Mapping of original columns to standard columns

    Returns:
        Normalized dataframe with standard columns
    """
    # Rename columns according to mapping
    df_norm = df.rename(columns=column_mapping)

    # Keep only mapped columns that exist
    existing_cols = [col for col in OUTPUT_COLUMNS[:-1] if col in df_norm.columns]
    df_norm = df_norm[existing_cols].copy()

    # Normalize each column
    if "Title" in df_norm.columns:
        df_norm["Title"] = df_norm["Title"].apply(normalize_title)

    if "Last Action Date" in df_norm.columns:
        df_norm["Last Action Date"] = df_norm["Last Action Date"].apply(parse_date)

    for url_col in ["Legiscan Bill Main Page", "Legiscan Text Page"]:
        if url_col in df_norm.columns:
            df_norm[url_col] = df_norm[url_col].apply(normalize_url)

    # Ensure all output columns exist (fill with empty strings if missing)
    for col in OUTPUT_COLUMNS[:-1]:  # Exclude Bill ID for now
        if col not in df_norm.columns:
            df_norm[col] = ""

    return df_norm[OUTPUT_COLUMNS[:-1]]  # Return without Bill ID


def merge_duplicates(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate rows based on State + Bill Number.
    Keep the latest date and merge data from all duplicates.

    Args:
        all_data: Combined dataframe from all files

    Returns:
        Deduplicated dataframe
    """
    print(f"\n{'='*80}")
    print("STEP 4: MERGING & DEDUPLICATING")
    print(f"{'='*80}\n")

    print(f"Total rows before deduplication: {len(all_data)}")

    # Create dedup key
    all_data['_dedup_key'] = (
        all_data['State'].astype(str).str.strip().str.upper() + '_' +
        all_data['Bill Number'].astype(str).str.strip().str.upper()
    )

    # Group by dedup key
    grouped = all_data.groupby('_dedup_key')
    duplicates = grouped.filter(lambda x: len(x) > 1)

    if len(duplicates) > 0:
        print(f"Found {len(duplicates)} duplicate rows in {duplicates['_dedup_key'].nunique()} bill(s)")
        print("\nDuplicate bills:")
        for key in duplicates['_dedup_key'].unique():
            dup_rows = duplicates[duplicates['_dedup_key'] == key]
            print(f"  {key}: {len(dup_rows)} copies")

            # Check for conflicts
            for col in ['Title', 'Status', 'Legiscan Bill Main Page', 'Legiscan Text Page']:
                unique_vals = dup_rows[col].dropna().unique()
                unique_vals = [v for v in unique_vals if v != ""]
                if len(unique_vals) > 1:
                    print(f"    CONFLICT in {col}: {unique_vals}")
    else:
        print("No duplicates found")

    # Merge duplicates: keep latest date, fill missing values
    merged_rows = []

    for key, group in grouped:
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
        else:
            # Sort by date (latest first), then by completeness
            group = group.copy()
            group['_date_sort'] = pd.to_datetime(group['Last Action Date'], errors='coerce')
            group['_completeness'] = group.notna().sum(axis=1)
            group = group.sort_values(['_date_sort', '_completeness'],
                                     ascending=[False, False])

            # Start with the row with latest date
            merged = group.iloc[0].copy()

            # Fill in missing values from other rows
            for col in OUTPUT_COLUMNS[:-1]:
                if pd.isna(merged[col]) or merged[col] == "":
                    for _, row in group.iterrows():
                        if not pd.isna(row[col]) and row[col] != "":
                            merged[col] = row[col]
                            break

            merged_rows.append(merged)

    result = pd.DataFrame(merged_rows)
    result = result.drop(columns=['_dedup_key'], errors='ignore')
    result = result.drop(columns=['_date_sort', '_completeness'], errors='ignore')

    print(f"\nTotal rows after deduplication: {len(result)}")
    print(f"Removed {len(all_data) - len(result)} duplicate row(s)\n")

    return result


def add_bill_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Bill ID column as State_BillNumber.

    Args:
        df: Input dataframe

    Returns:
        Dataframe with Bill ID column
    """
    df = df.copy()
    df['Bill ID'] = (
        df['State'].astype(str).str.strip().str.upper() + '_' +
        df['Bill Number'].astype(str).str.strip().str.upper()
    )
    return df[OUTPUT_COLUMNS]


def save_output(df: pd.DataFrame, dry_run: bool = False):
    """
    Save the normalized data to CSV and Excel files.

    Args:
        df: Normalized dataframe
        dry_run: If True, don't write files (just preview)
    """
    print(f"\n{'='*80}")
    print("STEP 5: SAVING OUTPUT")
    print(f"{'='*80}\n")

    if dry_run:
        print("DRY RUN MODE - Files will not be written\n")
        print(f"Preview of first 10 rows:")
        print(df.head(10).to_string())
        print(f"\n... and {len(df) - 10} more rows")
        return

    # Save CSV
    print(f"Writing CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved {len(df)} rows")

    # Save Excel with clickable URLs
    print(f"\nWriting Excel: {OUTPUT_XLSX}")
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Bills')

        # Auto-adjust column widths
        worksheet = writer.sheets['Bills']
        for idx, col in enumerate(df.columns, 1):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            # Cap at 50 characters
            max_length = min(max_length, 50)
            worksheet.column_dimensions[openpyxl.utils.get_column_letter(idx)].width = max_length + 2

    print(f"  ✓ Saved {len(df)} rows")
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_XLSX}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("LEGISCAN SPREADSHEET NORMALIZER")
    print("="*80)

    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("\nRunning in DRY-RUN mode (no files will be written)")

    # Step 1: Load all files
    loaded_files = load_files(INPUT_DIR)

    if not loaded_files:
        print("ERROR: No files loaded. Exiting.")
        sys.exit(1)

    # Step 2 & 3: Normalize columns and values
    print(f"{'='*80}")
    print("STEP 2 & 3: NORMALIZING COLUMNS & VALUES")
    print(f"{'='*80}\n")

    all_dataframes = []
    for filename, df, column_mapping in loaded_files:
        print(f"Normalizing: {filename}")
        df_norm = normalize_values(df, column_mapping)
        all_dataframes.append(df_norm)
        print(f"  ✓ Normalized {len(df_norm)} rows\n")

    # Combine all dataframes
    combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined total: {len(combined)} rows\n")

    # Step 4: Merge duplicates
    deduplicated = merge_duplicates(combined)

    # Add Bill ID
    final = add_bill_id(deduplicated)

    # Step 5: Save output
    save_output(final, dry_run=dry_run)

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
