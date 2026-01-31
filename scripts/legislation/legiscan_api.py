#!/usr/bin/env python3
"""
LegiScan API Client Wrapper
Handles all API interactions with rate limiting and error handling.
"""

import requests
import time
import base64
from typing import Dict, List, Optional
import logging


class LegiScanAPI:
    """Wrapper for LegiScan API operations with rate limiting and retry logic"""

    BASE_URL = "https://api.legiscan.com/"

    def __init__(self, api_key: str, rate_limit_delay: float = 0.5):
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.api_call_count = 0
        self.logger = logging.getLogger(__name__)

    def _make_request(
        self,
        operation: str,
        params: Optional[Dict] = None,
        retries: int = 3
    ) -> Dict:
        """Make API request with rate limiting and retry logic"""
        if params is None:
            params = {}

        params['key'] = self.api_key
        params['op'] = operation

        for attempt in range(retries):
            try:
                # Rate limiting
                time.sleep(self.rate_limit_delay)

                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()

                # Track API usage
                self.api_call_count += 1

                if data.get('status') != 'OK':
                    error_msg = data.get('alert', {}).get('message', 'Unknown error')

                    # Empty result set is not really an error for some operations
                    if error_msg == 'Empty result set' and operation in ['getDatasetList']:
                        return {'status': 'OK', 'datasetlist': []}

                    raise Exception(f"API Error: {error_msg}")

                self.logger.debug(f"API call {self.api_call_count}: {operation} - Success")
                return data

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")

                if attempt < retries - 1:
                    # Exponential backoff
                    delay = 2 ** attempt
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Request failed after {retries} attempts: {e}")

    def get_dataset_list(self, state: Optional[str] = None, year: Optional[int] = None) -> List[Dict]:
        """Get list of available datasets"""
        params = {}
        if state:
            params['state'] = state
        if year:
            params['year'] = year

        try:
            data = self._make_request('getDatasetList', params)
            datasets = data.get('datasetlist', [])
            self.logger.info(f"Retrieved {len(datasets)} datasets for year={year}, state={state}")
            return datasets
        except Exception as e:
            self.logger.error(f"Failed to get dataset list: {e}")
            return []

    def get_dataset(self, session_id: int, access_key: str) -> bytes:
        """Download a dataset ZIP file"""
        params = {
            'id': session_id,
            'access_key': access_key
        }

        data = self._make_request('getDataset', params)
        dataset = data.get('dataset', {})
        zip_base64 = dataset.get('zip', '')

        if not zip_base64:
            raise Exception(f"No ZIP data in dataset response for session {session_id}")

        # Decode base64 ZIP
        zip_bytes = base64.b64decode(zip_base64)
        self.logger.info(f"Downloaded dataset for session {session_id} ({len(zip_bytes)} bytes)")
        return zip_bytes

    def get_api_usage(self) -> int:
        """Return the number of API calls made"""
        return self.api_call_count
