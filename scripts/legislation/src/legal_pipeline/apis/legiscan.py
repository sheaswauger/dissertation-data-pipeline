from __future__ import annotations
import requests, os
from typing import Dict, Any
from ..config import settings

BASE = "https://api.legiscan.com/"

def test_ping(api_key: str | None = None) -> Dict[str, Any]:
    key = api_key or settings.legiscan_api_key
    if not key:
        raise RuntimeError("LEGISCAN_API_KEY not configured")
    r = requests.get(BASE, params={"key": key, "op": "getSessionList", "state": "CO"}, timeout=30)
    r.raise_for_status()
    return r.json()
