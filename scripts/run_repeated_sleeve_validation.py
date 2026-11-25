"""Run repeated real-data 10-year-training/OOS sleeve validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.quant.sleeve_validation import run_repeated_real_sleeve_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache/nse_long_history_v1"))
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--oos-days", type=int, default=504)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-symbols", type=int, default=10)
    parser.add_argument("--minimum-symbol-start", default="2008-01-01")
    parser.add_argument("--price-column", default="Adj Close")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/backtest_reports"))
    args = parser.parse_args()

    result = run_repeated_real_sleeve_validation(
        cache_dir=args.cache_dir,
        train_years=args.train_years,
        oos_days=args.oos_days,
        n_folds=args.folds,
        min_symbols=args.min_symbols,
        minimum_symbol_start=pd.Timestamp(args.minimum_symbol_start),
        price_column=args.price_column,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (
        f"real_repeated_sleeves_{args.train_years}yr_{args.folds}fold_"
        f"{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "universe_size": result.universe_size,
        "universe_status": result.universe_status,
        "aggregate_selected_metrics": result.aggregate_selected_metrics,
        "candidate_aggregate_metrics": result.candidate_aggregate_metrics,
        "selected_configs": [fold["selected_config"] for fold in result.folds],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
