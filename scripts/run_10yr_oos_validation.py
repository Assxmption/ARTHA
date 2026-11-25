"""Run the real-data 10-year train / later OOS validation experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.historical_validation import run_ten_year_oos_validation
from app.data.security_master import PointInTimeSecurityMaster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--min-symbols", type=int, default=10)
    parser.add_argument("--min-oos-days", type=int, default=504)
    parser.add_argument("--price-column", default="Close")
    parser.add_argument(
        "--security-master", type=Path, default=None,
        help="Authoritative point-in-time security master (JSON or CSV).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/backtest_reports"))
    args = parser.parse_args()

    result = run_ten_year_oos_validation(
        cache_dir=args.cache_dir,
        train_years=args.train_years,
        min_symbols=args.min_symbols,
        min_oos_days=args.min_oos_days,
        price_column=args.price_column,
        security_master=(
            PointInTimeSecurityMaster.from_file(args.security_master)
            if args.security_master else None
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"real_data_{args.train_years}yr_oos_{stamp}.json"
    output.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "train": [result.train_start, result.train_end],
        "test": [result.test_start, result.test_end],
        "universe_size": result.universe_size,
        "universe_status": result.universe_status,
        "selected_config": result.to_dict()["selected_config"],
        "training_metrics": result.training_metrics,
        "test_metrics": result.test_metrics,
        "benchmark_metrics": result.benchmark_metrics,
        "validation": result.validation,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
