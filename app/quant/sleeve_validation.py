"""Causal real-data sleeve comparison for NSE research.

This is an experiment harness, not a promotion path. It evaluates a small,
predeclared set of deterministic cross-sectional sleeves on the same cached
price panel, freezes portfolio weights from the training period, and reports
the later OOS period. It deliberately carries the cache-only survivorship
warning until an authoritative security master is provided.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from app.data.security_master import PointInTimeSecurityMaster
from app.quant.backtest import MIN_OOS_TRADING_DAYS, compute_metrics
from app.quant.historical_validation import (
    HistoricalValidationError,
    SourceCoverage,
    load_real_nse_universe,
)


@dataclass(frozen=True)
class RankSleeveConfig:
    name: str
    lookback_days: int
    skip_days: int
    rebalance_days: int
    top_fraction: float = 0.20
    long_only: bool = True
    reverse_rank: bool = False
    gross_exposure: float = 1.0


@dataclass
class SleeveMatrixResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    universe_size: int
    universe_status: str
    source_files: list[SourceCoverage]
    sleeves: dict[str, dict]
    portfolios: dict[str, dict]
    train_correlations: dict[str, dict[str, float]]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepeatedSleeveResult:
    """Nested chronological fold result for the fixed sleeve candidates."""

    train_years: int
    oos_days: int
    n_folds: int
    universe_size: int
    universe_status: str
    folds: list[dict]
    aggregate_selected_metrics: dict
    candidate_aggregate_metrics: dict[str, dict]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def rank_sleeve_returns(
    prices: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: RankSleeveConfig,
    *,
    trading_index: Optional[pd.Index] = None,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    cost_bps: float = 15.0,
) -> pd.Series:
    """Build a causal rank sleeve using next-session returns.

    Scores use only prices through the decision date. Positions are held from
    the following available session until the next rebalance. Long/short
    sleeves use half the configured gross exposure on each leg; long-only
    sleeves use the full configured gross exposure.
    """
    if config.lookback_days <= 0 or config.rebalance_days <= 0:
        raise ValueError("lookback_days and rebalance_days must be positive")
    if not 0 < config.top_fraction <= 0.5:
        raise ValueError("top_fraction must be in (0, 0.5]")

    panel = pd.DataFrame(prices).sort_index()
    if trading_index is not None:
        panel = panel.reindex(pd.DatetimeIndex(trading_index))
    panel = panel.loc[:end]
    output_index = panel.index[(panel.index >= start) & (panel.index <= end)]
    output = pd.Series(0.0, index=output_index, dtype=float, name=config.name)
    if panel.empty or output.empty:
        return output

    returns = panel.pct_change()
    dates = panel.index
    first_date_position = int(
        np.searchsorted(dates.values, np.datetime64(start), side="left")
    )
    first_rebalance = max(config.lookback_days + config.skip_days, first_date_position)
    previous_long: set[str] = set()
    previous_short: set[str] = set()

    for decision_position in range(
        first_rebalance, len(dates), config.rebalance_days,
    ):
        decision_date = dates[decision_position]
        if decision_date > end:
            break
        score_end = decision_position - config.skip_days
        score_start = score_end - config.lookback_days
        if score_start < 0:
            continue

        score = (
            panel.iloc[score_end] / panel.iloc[score_start] - 1.0
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if security_master is not None:
            active = set(security_master.symbols_at(decision_date.date()))
            score = score[score.index.isin(active)]
        if len(score) < 5:
            continue

        ranked = score.sort_values(ascending=config.reverse_rank)
        n_per_leg = max(1, int(len(ranked) * config.top_fraction))
        long_symbols = set(ranked.index[:n_per_leg])
        short_symbols = (
            set() if config.long_only else set(ranked.index[-n_per_leg:])
        )
        # The book chosen at this close is held through the close of the next
        # decision date (inclusive).  Ending one session earlier left every
        # rebalance day flat (return 0), silently dropping 1/rebalance_days of
        # all sessions — 20% of days for a weekly sleeve.
        hold_end = min(
            decision_position + config.rebalance_days + 1, len(dates),
        )
        denominator = max(len(long_symbols) + len(short_symbols), 1)
        turnover = (
            len(long_symbols - previous_long)
            + len(previous_long - long_symbols)
            + len(short_symbols - previous_short)
            + len(previous_short - short_symbols)
        ) / denominator
        previous_long, previous_short = long_symbols, short_symbols

        for position in range(decision_position + 1, hold_end):
            current_date = dates[position]
            if current_date not in output.index:
                continue
            if security_master is not None:
                active = set(security_master.symbols_at(current_date.date()))
                day_long = long_symbols & active
                day_short = short_symbols & active
            else:
                day_long, day_short = long_symbols, short_symbols
            day_returns = returns.iloc[position]
            long_return = (
                day_returns[list(day_long)].dropna().mean() if day_long else 0.0
            )
            short_return = (
                day_returns[list(day_short)].dropna().mean() if day_short else 0.0
            )
            if config.long_only:
                value = config.gross_exposure * long_return
            else:
                leg_exposure = config.gross_exposure / 2.0
                value = leg_exposure * (long_return - short_return)
            if position == decision_position + 1:
                value -= (cost_bps / 10_000.0) * turnover
            output.loc[current_date] = float(value)

    return output


def _metrics(series: pd.Series) -> dict:
    metrics = asdict(compute_metrics(series.dropna()))
    metrics["num_trades"] = None
    metrics["trade_count_available"] = False
    metrics["return_observations"] = metrics["num_trading_days"]
    return metrics


def _windows(
    nifty: pd.Series,
    prices: dict[str, pd.Series],
    train_years: int,
    min_oos_days: int,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    common_start = max(series.index.min() for series in prices.values())
    common_end = min(series.index.max() for series in prices.values())
    index = nifty.loc[common_start:common_end].index
    if len(index) <= min_oos_days:
        raise HistoricalValidationError(
            f"Only {len(index)} common sessions available; need more than "
            f"{min_oos_days}"
        )
    test_start_position = len(index) - min_oos_days
    train_end = pd.Timestamp(index[test_start_position - 1]).normalize()
    test_start = pd.Timestamp(index[test_start_position]).normalize()
    train_start = (
        train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
    ).normalize()
    test_end = pd.Timestamp(index[-1]).normalize()
    return train_start, train_end, test_start, test_end, index


def _portfolio_weights(
    train_returns: pd.DataFrame,
    method: str,
) -> dict[str, float]:
    names = list(train_returns.columns)
    if not names:
        return {}
    if method == "equal_weight":
        return {name: 1.0 / len(names) for name in names}
    if method == "inverse_volatility":
        vol = train_returns.std().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        inverse = (1.0 / vol.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(0.0)
        if inverse.sum() <= 0:
            return {name: 1.0 / len(names) for name in names}
        weights = inverse / inverse.sum()
        return {name: float(weights[name]) for name in names}
    raise ValueError(f"Unknown portfolio weighting method: {method}")


def run_real_sleeve_matrix(
    cache_dir: Path = Path("data_cache"),
    train_years: int = 10,
    min_oos_days: int = MIN_OOS_TRADING_DAYS,
    min_symbols: int = 10,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    price_column: str = "Close",
    configs: Optional[Iterable[RankSleeveConfig]] = None,
) -> SleeveMatrixResult:
    """Evaluate predeclared sleeves and train-only portfolio combinations."""
    prices, coverage, nifty = load_real_nse_universe(
        cache_dir=cache_dir,
        train_years=train_years,
        min_symbols=min_symbols,
        security_master=security_master,
        min_oos_days=min_oos_days,
        price_column=price_column,
    )
    train_start, train_end, test_start, test_end, index = _windows(
        nifty, prices, train_years, min_oos_days,
    )
    configs = list(configs or [
        RankSleeveConfig("momentum_12m_skip_1m", 252, 21, 21),
        RankSleeveConfig("momentum_12m_skip_3m", 252, 63, 21),
        RankSleeveConfig("momentum_6m", 126, 21, 21),
        RankSleeveConfig("reversal_1m", 21, 1, 5, reverse_rank=True),
        RankSleeveConfig("reversal_3m", 63, 5, 21, reverse_rank=True),
        RankSleeveConfig("momentum_long_short", 252, 21, 21, long_only=False),
    ])

    train_returns: dict[str, pd.Series] = {}
    test_returns: dict[str, pd.Series] = {}
    sleeve_results: dict[str, dict] = {}
    train_index = index[index <= train_end]
    test_index = index[index >= test_start]
    for config in configs:
        train_stream = rank_sleeve_returns(
            prices, train_start, train_end, config,
            trading_index=train_index, security_master=security_master,
        )
        test_stream = rank_sleeve_returns(
            prices, test_start, test_end, config,
            trading_index=index, security_master=security_master,
        )
        train_returns[config.name] = train_stream
        test_returns[config.name] = test_stream
        sleeve_results[config.name] = {
            "config": asdict(config),
            "training_metrics": _metrics(train_stream),
            "test_metrics": _metrics(test_stream),
            "eligible_for_factstore": False,
            "rejection_reasons": [
                "chronological_holdout_not_walk_forward_evidence",
                "survivorship_biased_universe" if security_master is None else "",
            ],
        }
        sleeve_results[config.name]["rejection_reasons"] = [
            reason for reason in sleeve_results[config.name]["rejection_reasons"]
            if reason
        ]

    train_frame = pd.DataFrame(train_returns).reindex(train_index).fillna(0.0)
    test_frame = pd.DataFrame(test_returns).reindex(test_index).fillna(0.0)
    correlations = train_frame.corr().fillna(0.0)
    correlation_dict = {
        name: {other: float(correlations.loc[name, other]) for other in correlations.columns}
        for name in correlations.index
    }

    portfolios: dict[str, dict] = {}
    for method in ("equal_weight", "inverse_volatility"):
        weights = _portfolio_weights(train_frame, method)
        portfolio_train = sum(
            train_frame[name] * weight for name, weight in weights.items()
        )
        portfolio_test = sum(
            test_frame[name] * weight for name, weight in weights.items()
        )
        portfolios[method] = {
            "weights": weights,
            "training_metrics": _metrics(portfolio_train),
            "test_metrics": _metrics(portfolio_test),
            "eligible_for_factstore": False,
            "rejection_reasons": [
                "chronological_holdout_not_walk_forward_evidence",
                "survivorship_biased_universe" if security_master is None else "",
            ],
        }
        portfolios[method]["rejection_reasons"] = [
            reason for reason in portfolios[method]["rejection_reasons"] if reason
        ]

    manifest_hash = hashlib.sha256(
        json.dumps([asdict(item) for item in coverage], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return SleeveMatrixResult(
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        universe_size=len(prices),
        universe_status="point_in_time" if security_master else "survivorship_biased",
        source_files=coverage,
        sleeves=sleeve_results,
        portfolios=portfolios,
        train_correlations=correlation_dict,
        notes=[
            "All inputs are cached real NSE EOD files; no synthetic values were used.",
            "Each sleeve uses next-session returns and predeclared parameters.",
            "Portfolio weights are computed only from the training period.",
            f"Source manifest SHA-256: {manifest_hash}",
            "This matrix is diagnostic until repeated walk-forward folds and an authoritative universe pass.",
        ],
    )


def run_repeated_real_sleeve_validation(
    cache_dir: Path = Path("data_cache"),
    train_years: int = 10,
    oos_days: int = MIN_OOS_TRADING_DAYS,
    n_folds: int = 4,
    min_symbols: int = 10,
    minimum_symbol_start: Optional[pd.Timestamp] = None,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    price_column: str = "Close",
    configs: Optional[Iterable[RankSleeveConfig]] = None,
) -> RepeatedSleeveResult:
    """Run disjoint chronological 10-year-train/OOS folds.

    Configuration selection happens independently inside each training window;
    no final OOS period is inspected to choose a sleeve. Results remain
    diagnostic until the point-in-time universe and full validation evidence
    requirements are satisfied.
    """
    if train_years <= 0 or oos_days <= 0 or n_folds <= 0:
        raise ValueError("train_years, oos_days, and n_folds must be positive")
    prices, coverage, nifty = load_real_nse_universe(
        cache_dir=cache_dir,
        train_years=train_years,
        min_symbols=min_symbols,
        security_master=security_master,
        min_oos_days=oos_days,
        price_column=price_column,
        minimum_symbol_start=minimum_symbol_start,
    )
    common_start = max(series.index.min() for series in prices.values())
    common_end = min(series.index.max() for series in prices.values())
    index = nifty.loc[common_start:common_end].index
    required_sessions = n_folds * oos_days + int(train_years * 252)
    if len(index) < required_sessions:
        raise HistoricalValidationError(
            f"Need at least {required_sessions} common sessions for {n_folds} "
            f"folds; found {len(index)}"
        )

    candidate_list = list(configs or [
        RankSleeveConfig("momentum_12m_skip_1m", 252, 21, 21),
        RankSleeveConfig("momentum_12m_skip_3m", 252, 63, 21),
        RankSleeveConfig("momentum_6m", 126, 21, 21),
        RankSleeveConfig("reversal_1m", 21, 1, 5, reverse_rank=True),
        RankSleeveConfig("reversal_3m", 63, 5, 21, reverse_rank=True),
        RankSleeveConfig("momentum_long_short", 252, 21, 21, long_only=False),
    ])

    fold_specs: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    test_end_position = len(index)
    for _ in range(n_folds):
        test_start_position = test_end_position - oos_days
        if test_start_position <= 0:
            raise HistoricalValidationError("Insufficient sessions for fold boundary")
        train_end = pd.Timestamp(index[test_start_position - 1]).normalize()
        test_start = pd.Timestamp(index[test_start_position]).normalize()
        train_start = (
            train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
        ).normalize()
        test_end = pd.Timestamp(index[test_end_position - 1]).normalize()
        fold_specs.append((train_start, train_end, test_start, test_end))
        test_end_position = test_start_position
    fold_specs.reverse()

    folds: list[dict] = []
    selected_streams: list[pd.Series] = []
    candidate_streams: dict[str, list[pd.Series]] = {
        config.name: [] for config in candidate_list
    }
    for fold_number, (train_start, train_end, test_start, test_end) in enumerate(fold_specs):
        train_metrics: dict[str, dict] = {}
        test_streams: dict[str, pd.Series] = {}
        for config in candidate_list:
            train_stream = rank_sleeve_returns(
                prices, train_start, train_end, config,
                trading_index=index, security_master=security_master,
            )
            test_stream = rank_sleeve_returns(
                prices, test_start, test_end, config,
                trading_index=index, security_master=security_master,
            )
            train_metrics[config.name] = _metrics(train_stream)
            test_streams[config.name] = test_stream
            candidate_streams[config.name].append(test_stream)

        selected_name = max(
            train_metrics,
            key=lambda name: (
                train_metrics[name]["sharpe_ratio"],
                -train_metrics[name]["max_drawdown"],
            ),
        )
        selected_test = test_streams[selected_name]
        selected_streams.append(selected_test)
        folds.append({
            "fold": fold_number,
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "selected_config": selected_name,
            "training_metrics": train_metrics,
            "selected_test_metrics": _metrics(selected_test),
            "candidate_test_metrics": {
                name: _metrics(stream) for name, stream in test_streams.items()
            },
        })

    selected_aggregate = pd.concat(selected_streams).sort_index()
    candidate_aggregate_metrics = {
        name: _metrics(pd.concat(streams).sort_index())
        for name, streams in candidate_streams.items()
    }
    manifest_hash = hashlib.sha256(
        json.dumps([asdict(item) for item in coverage], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return RepeatedSleeveResult(
        train_years=train_years,
        oos_days=oos_days,
        n_folds=n_folds,
        universe_size=len(prices),
        universe_status="point_in_time" if security_master else "survivorship_biased",
        folds=folds,
        aggregate_selected_metrics=_metrics(selected_aggregate),
        candidate_aggregate_metrics=candidate_aggregate_metrics,
        notes=[
            "All values came from real cached provider data; no synthetic values were used.",
            "Each fold selected its configuration using only its own training window.",
            "OOS folds are disjoint and each contains exactly the requested observations.",
            f"Source manifest SHA-256: {manifest_hash}",
            "The cache-only universe is survivorship-biased until an authoritative security master is supplied.",
            "This is diagnostic research and cannot create validated Fact Store evidence.",
        ],
    )
