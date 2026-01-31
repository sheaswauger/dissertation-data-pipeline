from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
from .apis.openstates import iter_bills

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Retrieve legislative data (demo).")
    p.add_argument("--provider", choices=["openstates"], default="openstates")
    p.add_argument("--state", required=True, help="Two-letter state or full name (OpenStates jurisdiction)")
    p.add_argument("--session", help="Session identifier (string)")
    p.add_argument("--limit", type=int, default=10, help="Max records to fetch")
    p.add_argument("--out", type=Path, required=True, help="Output JSONL file")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    if args.provider == "openstates":
        for rec in iter_bills(state=args.state, session=args.session, limit=args.limit):
            args.out.write_text("", encoding="utf-8") if not args.out.exists() else None
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} records to {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
