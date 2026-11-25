"""Fetch a real, provenance-tracked long NSE history cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.nse_long_history import fetch_long_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("data_cache/nse_long_history_v1"))
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2026-08-22")
    args = parser.parse_args()

    if args.symbols_file:
        symbols = [
            line.strip() for line in args.symbols_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        symbols = [
            path.stem.removesuffix("_v5")
            for path in sorted(Path("data_cache").glob("*_v5.parquet"))
            if not path.stem.startswith("IDX_")
        ]
    manifest = fetch_long_history(
        symbols=symbols,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
    )
    ok = sum(entry.get("status") == "ok" for entry in manifest["entries"])
    failed = sum(entry.get("status") == "failed" for entry in manifest["entries"])
    print({"output_dir": str(args.output_dir), "ok": ok, "failed": failed})


if __name__ == "__main__":
    main()
