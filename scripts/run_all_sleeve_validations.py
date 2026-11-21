"""
ARTHA — Run All Sleeve Validations (Expanded)
===============================================
Runs the full sleeve validation matrix + repeated walk-forward with an expanded
set of sleeve configs covering all strategy families:

  - Momentum (12m skip-1m, 12m skip-3m, 6m)
  - Mean Reversion (20-day, 10-day)
  - Trend Following (200-day, 100-day)
  - Reversal (1m, 3m, 5-day)
  - Factor-like (252-day long/short, 126-day long/short)

Each sleeve is run through the same causal train/OOS protocol with
next-session returns and predeclared parameters.

AGENTS.md Rule 1: All computation is deterministic — no LLM involvement.
AGENTS.md Rule 6: Nothing reaches a report unvalidated.
AGENTS.md Rule 8: Optimize for Sharpe/Sortino, never hardcode return targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.security_master import PointInTimeSecurityMaster
from app.quant.sleeve_validation import (
    RankSleeveConfig,
    run_real_sleeve_matrix,
    run_repeated_real_sleeve_validation,
)


# ── Expanded Sleeve Configurations ──────────────────────────────────────────────

EXPANDED_CONFIGS = [
    # Momentum family
    RankSleeveConfig("momentum_12m_skip_1m", lookback_days=252, skip_days=21,
                     rebalance_days=21),
    RankSleeveConfig("momentum_12m_skip_3m", lookback_days=252, skip_days=63,
                     rebalance_days=21),
    RankSleeveConfig("momentum_6m", lookback_days=126, skip_days=21,
                     rebalance_days=21),

    # Mean Reversion family (reverse_rank=True → buy losers/sell winners)
    RankSleeveConfig("mean_reversion_20d", lookback_days=20, skip_days=0,
                     rebalance_days=5, reverse_rank=True, top_fraction=0.20),
    RankSleeveConfig("mean_reversion_10d", lookback_days=10, skip_days=0,
                     rebalance_days=5, reverse_rank=True, top_fraction=0.20),

    # Trend Following family (long lookback → buy established trends)
    RankSleeveConfig("trend_following_200d", lookback_days=200, skip_days=0,
                     rebalance_days=21, top_fraction=0.20),
    RankSleeveConfig("trend_following_100d", lookback_days=100, skip_days=0,
                     rebalance_days=21, top_fraction=0.20),

    # Reversal family
    RankSleeveConfig("reversal_1m", lookback_days=21, skip_days=1,
                     rebalance_days=5, reverse_rank=True),
    RankSleeveConfig("reversal_3m", lookback_days=63, skip_days=5,
                     rebalance_days=21, reverse_rank=True),
    RankSleeveConfig("reversal_5d", lookback_days=5, skip_days=0,
                     rebalance_days=5, reverse_rank=True),

    # Factor-like (long/short, momentum-driven, capturing cross-sectional value)
    RankSleeveConfig("momentum_long_short_12m", lookback_days=252, skip_days=21,
                     rebalance_days=21, long_only=False),
    RankSleeveConfig("momentum_long_short_6m", lookback_days=126, skip_days=21,
                     rebalance_days=21, long_only=False),
]


def _summarize_metrics(metrics: dict) -> str:
    """One-line summary of key metrics."""
    return (
        f"Sharpe={metrics.get('sharpe_ratio', 0):.3f}  "
        f"Sortino={metrics.get('sortino_ratio', 0):.3f}  "
        f"MaxDD={metrics.get('max_drawdown', 0):.1%}  "
        f"CAGR={metrics.get('cagr', 0):.1%}  "
        f"Days={metrics.get('num_trading_days', 0)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run expanded sleeve validations (matrix + walk-forward).",
    )
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("data_cache/nse_long_history_v1"))
    parser.add_argument("--train-years", type=int, default=10)
    parser.add_argument("--min-oos-days", type=int, default=504)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-symbols", type=int, default=10)
    parser.add_argument("--minimum-symbol-start", default="2008-01-01")
    parser.add_argument("--price-column", default="Adj Close")
    parser.add_argument("--security-master", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("docs/backtest_reports"))
    parser.add_argument("--skip-matrix", action="store_true",
                        help="Skip single-holdout matrix; run only repeated folds.")
    parser.add_argument("--skip-repeated", action="store_true",
                        help="Skip repeated folds; run only single-holdout matrix.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = f"{datetime.now():%Y%m%d_%H%M%S}"

    master = (
        PointInTimeSecurityMaster.from_file(args.security_master)
        if args.security_master else None
    )

    # ── 1. Single-holdout Sleeve Matrix ─────────────────────────────────────
    if not args.skip_matrix:
        print("\n" + "=" * 72)
        print(" SLEEVE MATRIX — Single Train/OOS Holdout")
        print("=" * 72)
        try:
            matrix_result = run_real_sleeve_matrix(
                cache_dir=args.cache_dir,
                train_years=args.train_years,
                min_oos_days=args.min_oos_days,
                min_symbols=args.min_symbols,
                security_master=master,
                price_column=args.price_column,
                configs=EXPANDED_CONFIGS,
            )
            matrix_path = args.output_dir / (
                f"expanded_sleeve_matrix_{args.train_years}yr_{timestamp}.json"
            )
            matrix_path.write_text(
                json.dumps(matrix_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n📄 Report: {matrix_path}")
            print(f"   Universe: {matrix_result.universe_size} symbols "
                  f"({matrix_result.universe_status})")
            print(f"   Train: {matrix_result.train_start} → {matrix_result.train_end}")
            print(f"   OOS:   {matrix_result.test_start} → {matrix_result.test_end}")
            print("\n   Sleeve OOS Metrics:")
            for name, data in matrix_result.sleeves.items():
                tm = data.get("test_metrics", {})
                print(f"     {name:35s} {_summarize_metrics(tm)}")
            print("\n   Portfolio OOS Metrics:")
            for name, data in matrix_result.portfolios.items():
                pm = data.get("test_metrics", {})
                print(f"     {name:35s} {_summarize_metrics(pm)}")
        except Exception as exc:
            print(f"\n❌ Sleeve matrix failed: {exc}")

    # ── 2. Repeated Walk-Forward Folds ──────────────────────────────────────
    if not args.skip_repeated:
        print("\n" + "=" * 72)
        print(f" REPEATED WALK-FORWARD — {args.folds} Folds")
        print("=" * 72)
        try:
            repeated_result = run_repeated_real_sleeve_validation(
                cache_dir=args.cache_dir,
                train_years=args.train_years,
                oos_days=args.min_oos_days,
                n_folds=args.folds,
                min_symbols=args.min_symbols,
                minimum_symbol_start=pd.Timestamp(args.minimum_symbol_start),
                security_master=master,
                price_column=args.price_column,
                configs=EXPANDED_CONFIGS,
            )
            repeated_path = args.output_dir / (
                f"expanded_repeated_sleeves_{args.train_years}yr_"
                f"{args.folds}fold_{timestamp}.json"
            )
            repeated_path.write_text(
                json.dumps(repeated_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            print(f"\n📄 Report: {repeated_path}")
            print(f"   Universe: {repeated_result.universe_size} symbols "
                  f"({repeated_result.universe_status})")
            print(f"   Folds: {repeated_result.n_folds}")
            print(f"\n   Aggregate Selected OOS:")
            print(f"     {_summarize_metrics(repeated_result.aggregate_selected_metrics)}")
            print(f"\n   Per-candidate Aggregate OOS:")
            for name, metrics in repeated_result.candidate_aggregate_metrics.items():
                print(f"     {name:35s} {_summarize_metrics(metrics)}")
            print(f"\n   Fold-by-fold selections:")
            for fold in repeated_result.folds:
                print(f"     Fold {fold['fold']}: "
                      f"{fold['test_start']}→{fold['test_end']}  "
                      f"selected={fold['selected_config']}")
        except Exception as exc:
            print(f"\n❌ Repeated validation failed: {exc}")

    print("\n✅ All sleeve validations complete.")


if __name__ == "__main__":
    main()
