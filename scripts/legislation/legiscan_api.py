#!/usr/bin/env python3
"""
LegiScan API Client
Compliant with LegiScan API usage policy:
- Uses dataset_hash to avoid redundant downloads
- Only calls getDataset when hash has changed
- Caches all downloaded ZIPs locally
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional


LEGISCAN_BASE_URL = "https://api.legiscan.com/"


class DatasetHashCache:
    """
    Stores dataset_hash values locally to prevent redundant API downloads.
    LegiScan requires checking hashes before downloading datasets.
    """

    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.data, f, indent=2)

    def has_changed(self, session_id: int, new_hash: str) -> bool:
        """Returns True if the dataset has changed since last download."""
        stored_hash = self.data.get(str(session_id), {}).get("dataset_hash")
        return stored_hash != new_hash

    def update(self, session_id: int, dataset_hash: str, session_name: str):
        """Record the hash after a successful download."""
        self.data[str(session_id)] = {
            "dataset_hash": dataset_hash,
            "session_name": session_name,
        }
        self._save()

    def get_hash(self, session_id: int) -> Optional[str]:
        return self.data.get(str(session_id), {}).get("dataset_hash")


class LegiScanAPI:
    """
    LegiScan API client with compliant rate limiting and hash-based caching.
    """

    def __init__(self, api_key: str, rate_limit_delay: float = 0.5):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._api_call_count = 0

    def _get(self, params: Dict) -> Dict:
        """Make a GET request to the LegiScan API."""
        params["key"] = self.api_key
        time.sleep(self.rate_limit_delay)
        self._api_call_count += 1

        response = requests.get(LEGISCAN_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ERROR":
            raise ValueError(f"LegiScan API error: {data.get('alert', {}).get('message', 'Unknown error')}")

        return data

    def get_dataset_list(self, year: Optional[int] = None) -> List[Dict]:
        """
        Fetch list of available datasets with their hash values.
        This is a lightweight call — hash comparison happens here
        before any dataset download.
        """
        params = {"op": "getDatasetList"}
        if year:
            params["year"] = year

        data = self._get(params)
        datasets = data.get("datasetlist", [])

        # Filter out metadata entry
        return [d for d in datasets if isinstance(d, dict) and "session_id" in d]

    def get_dataset(self, session_id: int, access_key: str) -> bytes:
        """
        Download a dataset ZIP. Should only be called when hash has changed.
        """
        params = {
            "op": "getDataset",
            "id": session_id,
            "access_key": access_key,
        }
        params["key"] = self.api_key
        time.sleep(self.rate_limit_delay)
        self._api_call_count += 1

        response = requests.get(LEGISCAN_BASE_URL, params=params, timeout=120)
        response.raise_for_status()

        data = response.json()
        if data.get("status") == "ERROR":
            raise ValueError(f"LegiScan API error: {data.get('alert', {}).get('message', 'Unknown error')}")

        import base64
        zip_b64 = data.get("dataset", {}).get("zip", "")
        if not zip_b64:
            raise ValueError(f"No ZIP data returned for session {session_id}")

        return base64.b64decode(zip_b64)

    def get_api_usage(self) -> int:
        return self._api_call_count
