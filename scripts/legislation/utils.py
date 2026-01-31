#!/usr/bin/env python3
"""
Utility Functions
Handles logging, progress tracking, and error recording.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ProgressTracker:
    """Track processed datasets for resumability"""

    def __init__(self, tracking_file: Path):
        self.tracking_file = tracking_file
        self.data = self._load()

    def _load(self) -> Dict:
        """Load tracking data from file"""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        """Save tracking data to file"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_processed(self, session_id: int) -> bool:
        """Check if a dataset has been processed"""
        return str(session_id) in self.data

    def mark_processed(self, session_id: int, bills_found: int):
        """Mark a dataset as processed"""
        self.data[str(session_id)] = {
            'processed_date': datetime.now().isoformat(),
            'bills_found': bills_found,
            'status': 'complete'
        }
        self._save()

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        total_processed = len(self.data)
        total_bills = sum(d.get('bills_found', 0) for d in self.data.values())
        return {
            'datasets_processed': total_processed,
            'total_bills_found': total_bills
        }


class ErrorLogger:
    """Log failed operations to a text file"""

    def __init__(self, error_file: Path):
        self.error_file = error_file
        self.errors = {
            'dataset_failures': [],
            'bill_failures': [],
            'text_extraction_failures': []
        }

    def log_dataset_failure(self, session_id: int, error: str):
        """Log a dataset download/processing failure"""
        self.errors['dataset_failures'].append({
            'session_id': session_id,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        self._save()

    def log_bill_failure(self, bill_id: int, state: str, bill_number: str, error: str):
        """Log a bill processing failure"""
        self.errors['bill_failures'].append({
            'bill_id': bill_id,
            'state': state,
            'bill_number': bill_number,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        self._save()

    def log_text_extraction_failure(self, doc_id: int, bill_ref: str, error: str):
        """Log a text extraction failure"""
        self.errors['text_extraction_failures'].append({
            'doc_id': doc_id,
            'bill_ref': bill_ref,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        self._save()

    def _save(self):
        """Save errors to file in human-readable format"""
        with open(self.error_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("LEGISCAN DATA COLLECTION - ERROR LOG\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Dataset failures
            f.write(f"DATASET DOWNLOAD/PROCESSING FAILURES: {len(self.errors['dataset_failures'])}\n")
            f.write("-" * 80 + "\n")
            for err in self.errors['dataset_failures']:
                f.write(f"Session ID: {err['session_id']}\n")
                f.write(f"Error: {err['error']}\n")
                f.write(f"Time: {err['timestamp']}\n\n")

            # Bill processing failures
            f.write(f"\nBILL PROCESSING FAILURES: {len(self.errors['bill_failures'])}\n")
            f.write("-" * 80 + "\n")
            for err in self.errors['bill_failures']:
                f.write(f"Bill ID: {err['bill_id']}\n")
                f.write(f"State: {err['state']}, Bill: {err['bill_number']}\n")
                f.write(f"Error: {err['error']}\n")
                f.write(f"Time: {err['timestamp']}\n\n")

            # Text extraction failures
            f.write(f"\nTEXT EXTRACTION FAILURES: {len(self.errors['text_extraction_failures'])}\n")
            f.write("-" * 80 + "\n")
            for err in self.errors['text_extraction_failures']:
                f.write(f"Doc ID: {err['doc_id']}\n")
                f.write(f"Bill: {err['bill_ref']}\n")
                f.write(f"Error: {err['error']}\n")
                f.write(f"Time: {err['timestamp']}\n\n")

    def get_summary(self) -> str:
        """Get a summary of errors"""
        return (
            f"Errors: {len(self.errors['dataset_failures'])} datasets, "
            f"{len(self.errors['bill_failures'])} bills, "
            f"{len(self.errors['text_extraction_failures'])} text extractions"
        )


def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"collection_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_keywords(keywords_file: Path) -> list:
    """Load keywords from file"""
    keywords = []
    with open(keywords_file, 'r') as f:
        for line in f:
            keyword = line.strip()
            if keyword and not keyword.startswith('#'):
                keywords.append(keyword)
    return keywords
