from __future__ import annotations
import requests
from typing import Iterator, Dict, Any, Optional
from ..config import settings

BASE = "https://v3.openstates.org"

def iter_bills(state: str, session: str | int | None = None, limit: int = 50, api_key: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Yield bill search results from OpenStates API v3."""
    key = api_key or settings.openstates_api_key
    if not key:
        raise RuntimeError("OPENSTATES_API_KEY not configured")
    params = {
        "jurisdiction": state,
        "per_page": min(limit, 50),
        "page": 1,
    }
    if session:
        params["session"] = str(session)
    headers = {"X-API-KEY": key}
    fetched = 0
    while fetched < limit:
        r = requests.get(f"{BASE}/bills", params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            yield item
            fetched += 1
            if fetched >= limit:
                break
        params["page"] += 1
