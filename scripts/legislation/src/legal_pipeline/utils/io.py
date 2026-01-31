from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable, Any

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
