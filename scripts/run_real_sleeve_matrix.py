"""Run a real NSE causal sleeve and portfolio comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.security_master import PointInTimeSecurityMaster
from app.quant.sleeve_validation import run_real_sleeve_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--min-oos-days", type=int, default=504)
    parser.add_argument("--price-column", default="Close")
    parser.add_argument("--min-symbols", type=int, default=10)
    parser.add_argument("--security-master", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/backtest_reports"))
    args = parser.parse_args()

    master = (
        PointInTimeSecurityMaster.from_file(args.security_master)
        if args.security_master else None
    )
    result = run_real_sleeve_matrix(
        cache_dir=args.cache_dir,
        train_years=args.train_years,
        min_oos_days=args.min_oos_days,
        min_symbols=args.min_symbols,
        security_master=master,
        price_column=args.price_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"real_sleeve_matrix_{args.train_years}yr_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "train": [result.train_start, result.train_end],
        "test": [result.test_start, result.test_end],
        "universe_size": result.universe_size,
        "universe_status": result.universe_status,
        "sleeves": {
            name: item["test_metrics"] for name, item in result.sleeves.items()
        },
        "portfolios": {
            name: item["test_metrics"] for name, item in result.portfolios.items()
        },
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
