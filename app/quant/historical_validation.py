"""
Real-data historical validation with an explicit train/test boundary.

This module is intentionally separate from the older report-generation
scripts.  It refuses synthetic data, records source-file hashes, selects a
strategy configuration only on the training window, and evaluates that frozen
configuration on a later out-of-sample window.

The initial experiment is a cross-sectional momentum sleeve because it has a
fully deterministic signal and does not require fitting an HMM on future data.
The same boundary contract can be applied to the remaining sleeves after their
per-date fitting paths are audited.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from app.data.security_master import PointInTimeSecurityMaster, SecurityMasterError
from app.quant.backtest import (
    MIN_OOS_TRADING_DAYS,
    BacktestMetrics,
    compute_metrics,
)

logger = logging.getLogger("artha.quant.historical_validation")


class HistoricalValidationError(RuntimeError):
    """Raised when the real-data validation contract cannot be satisfied."""


@dataclass(frozen=True)
class MomentumConfig:
    lookback_days: int
    skip_days: int
    rebalance_days: int
    top_fraction: float = 0.20
    long_only: bool = True


@dataclass(frozen=True)
class SourceCoverage:
    symbol: str
    path: str
    sha256: str
    first_date: str
    last_date: str
    rows: int


@dataclass
class HistoricalValidationResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    universe_size: int
    universe_status: str
    security_master_sha256: Optional[str]
    source_files: list[SourceCoverage]
    selected_config: MomentumConfig
    training_metrics: dict
    test_metrics: dict
    benchmark_metrics: dict
    candidate_training_metrics: dict
    validation: dict
    notes: list[str]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["selected_config"] = asdict(self.selected_config)
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.DatetimeIndex):
        index = frame.index
    else:
        date_column = next(
            (column for column in frame.columns
             if str(column).lower() in {"date", "datetime", "timestamp"}),
            None,
        )
        if date_column is None:
            raise HistoricalValidationError("Price file has no date index or date column")
        index = pd.to_datetime(frame[date_column], errors="coerce")
    index = pd.DatetimeIndex(index).tz_localize(None)
    if index.isna().any():
        raise HistoricalValidationError("Price file contains invalid dates")
    return index.normalize()


def load_real_nse_universe(
    cache_dir: Path = Path("data_cache"),
    source_pattern: str = "*_v5.parquet",
    train_years: int = 10,
    min_symbols: int = 10,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    min_oos_days: int = MIN_OOS_TRADING_DAYS,
    price_column: str = "Close",
    minimum_symbol_start: Optional[pd.Timestamp] = None,
) -> tuple[dict[str, pd.Series], list[SourceCoverage], pd.Series]:
    """Load real EOD files, optionally keyed by a point-in-time master.

    Without ``security_master`` this preserves the original diagnostic mode,
    which selects complete-history files and is explicitly survivorship-biased.
    A supplied master disables that future-defined filtering and leaves
    membership decisions to each rebalance date.
    """
    files = sorted(cache_dir.glob(source_pattern))
    if not files:
        raise HistoricalValidationError(f"No real NSE files found in {cache_dir}")

    raw: dict[str, tuple[Path, pd.Series, pd.Timestamp, pd.Timestamp]] = {}
    for path in files:
        symbol = path.stem.removesuffix("_v5")
        if symbol.startswith("IDX_"):
            continue
        if security_master is not None:
            try:
                symbol = security_master.canonicalize(symbol)
            except SecurityMasterError as exc:
                raise HistoricalValidationError(str(exc)) from exc
        try:
            frame = pd.read_parquet(path)
            if price_column not in frame.columns:
                continue
            index = _date_index(frame)
            close = pd.to_numeric(frame[price_column], errors="coerce")
            series = pd.Series(close.to_numpy(), index=index, name=symbol)
            series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
            series = series[series > 0]
            if series.empty:
                continue
            if symbol in raw:
                raise HistoricalValidationError(
                    f"Multiple price files map to canonical security {symbol}"
                )
            raw[symbol] = (path, series, series.index.min(), series.index.max())
        except HistoricalValidationError:
            raise
        except Exception as exc:
            logger.warning("Skipping invalid source %s: %s", path, exc)

    index_path = cache_dir / "IDX_NSEI_v5.parquet"
    if not index_path.exists():
        raise HistoricalValidationError("Real NIFTY index file IDX_NSEI_v5.parquet is required")
    index_frame = pd.read_parquet(index_path)
    if "Close" not in index_frame.columns:
        raise HistoricalValidationError("NIFTY index has no Close column")
    index_dates = _date_index(index_frame)
    nifty = pd.Series(
        pd.to_numeric(index_frame["Close"], errors="coerce").to_numpy(),
        index=index_dates,
        name="NIFTY50",
    ).dropna()
    nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()

    if not raw or nifty.empty:
        raise HistoricalValidationError("Real NSE universe or NIFTY index is empty")

    provisional_end = min(
        min(item[3] for item in raw.values()),
        nifty.index.max(),
    )
    nifty = nifty.loc[:provisional_end]
    if len(nifty) <= min_oos_days:
        raise HistoricalValidationError(
            f"Only {len(nifty)} benchmark sessions available; need more than "
            f"{min_oos_days} OOS sessions"
        )
    oos_start_position = len(nifty) - min_oos_days
    train_end = nifty.index[oos_start_position - 1]
    requested_train_start = (
        train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
    ).normalize()

    if security_master is not None:
        common_start = nifty.index.min()
        common_end = nifty.index.max()
        eligible = {
            symbol: item for symbol, item in raw.items()
            if item[3] >= requested_train_start and item[2] <= common_end
        }
        if len(eligible) < min_symbols:
            raise HistoricalValidationError(
                f"Only {len(eligible)} master-mapped symbols overlap the index "
                f"window; need {min_symbols}"
            )
    else:
        # Diagnostic mode still has survivorship bias, but it no longer lets a
        # later IPO shorten the requested training window. The OOS boundary is
        # derived from the benchmark calendar and the minimum canonical gate.
        eligibility_start = minimum_symbol_start or requested_train_start
        eligible = {
            symbol: item
            for symbol, item in raw.items()
            if item[2] <= eligibility_start and item[3] >= provisional_end
        }
        if eligible:
            common_start = max(
                min(item[2] for item in eligible.values()),
                nifty.index.min(),
            )
            common_end = provisional_end
        else:
            common_start, common_end = pd.Timestamp("NaT"), pd.Timestamp("NaT")
        common_start = max(common_start, nifty.index.min())
        common_end = min(common_end, provisional_end)
    train_end = common_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
    if pd.isna(common_start) or pd.isna(common_end) or common_end <= train_end:
        raise HistoricalValidationError(
            f"Common real-data coverage is shorter than {train_years} training years "
            "plus a meaningful OOS window"
        )

    if security_master is None:
        # Re-apply the final common boundary so every selected series is
        # available for both the entire training and OOS windows.
        eligible = {
            symbol: item
            for symbol, item in eligible.items()
            if item[2] <= common_start and item[3] >= common_end
        }
        if len(eligible) < min_symbols:
            raise HistoricalValidationError(
                f"Only {len(eligible)} symbols have common coverage; need {min_symbols}"
            )

    panel: dict[str, pd.Series] = {}
    coverage: list[SourceCoverage] = []
    for symbol, (path, series, _, _) in sorted(eligible.items()):
        panel[symbol] = series.loc[common_start:common_end]
        coverage.append(SourceCoverage(
            symbol=symbol,
            path=str(path),
            sha256=_sha256(path),
            first_date=str(series.index.min().date()),
            last_date=str(series.index.max().date()),
            rows=len(series),
        ))

    nifty = nifty.loc[common_start:common_end]
    return panel, coverage, nifty


def cross_sectional_momentum(
    prices: dict[str, pd.Series],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: MomentumConfig,
    cost_bps: float = 15.0,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    trading_index: Optional[pd.Index] = None,
) -> pd.Series:
    """Generate a causal daily return stream between two dates."""
    panel = pd.DataFrame(prices).sort_index()
    if trading_index is not None:
        panel = panel.reindex(pd.DatetimeIndex(trading_index))
    panel = panel.loc[:end]
    if panel.empty:
        return pd.Series(dtype=float)

    returns = panel.pct_change()
    dates = panel.index
    output = pd.Series(0.0, index=dates[(dates >= start) & (dates <= end)], dtype=float)
    previous_long: set[str] = set()
    previous_short: set[str] = set()

    first_rebalance = max(
        config.lookback_days + config.skip_days,
        int(np.searchsorted(dates.values, np.datetime64(start), side="left")),
    )
    for t in range(first_rebalance, len(dates), config.rebalance_days):
        decision_date = dates[t]
        if decision_date > end:
            break
        score_end = t - config.skip_days
        score_start = score_end - config.lookback_days
        if score_start < 0:
            continue
        score = panel.iloc[score_end] / panel.iloc[score_start] - 1.0
        score = score.replace([np.inf, -np.inf], np.nan).dropna()
        if security_master is not None:
            active_symbols = set(security_master.symbols_at(decision_date.date()))
            score = score[score.index.isin(active_symbols)]
        if len(score) < 5:
            continue
        n = max(1, int(len(score) * config.top_fraction))
        ranked = score.sort_values(ascending=False)
        long_symbols = set(ranked.index[:n])
        short_symbols = set() if config.long_only else set(ranked.index[-n:])
        hold_end = min(t + config.rebalance_days, len(dates))
        turnover = (
            len(long_symbols - previous_long) + len(previous_long - long_symbols)
            + len(short_symbols - previous_short) + len(previous_short - short_symbols)
        ) / max(len(long_symbols) + len(short_symbols), 1)
        previous_long, previous_short = long_symbols, short_symbols

        for d in range(t + 1, hold_end):
            dt = dates[d]
            if dt < start or dt > end:
                continue
            day = returns.iloc[d]
            if security_master is not None:
                active_symbols = set(security_master.symbols_at(dt.date()))
                day_long_symbols = long_symbols & active_symbols
                day_short_symbols = short_symbols & active_symbols
            else:
                day_long_symbols = long_symbols
                day_short_symbols = short_symbols
            long_ret = day[list(day_long_symbols)].dropna().mean() if day_long_symbols else 0.0
            short_ret = day[list(day_short_symbols)].dropna().mean() if day_short_symbols else 0.0
            value = long_ret - short_ret if short_symbols else long_ret * 0.5
            if d == t + 1:
                value -= (cost_bps / 10_000.0) * turnover
            output.loc[dt] = float(value)

    return output


def _metrics(series: pd.Series) -> dict:
    metrics = asdict(compute_metrics(series.dropna()))
    # A return stream alone has no position-change ledger. The canonical
    # helper treats observations as pseudo-trades for legacy compatibility;
    # do not publish that number as executed trade count.
    metrics["num_trades"] = None
    metrics["trade_count_available"] = False
    metrics["return_observations"] = metrics["num_trading_days"]
    return metrics


def _validation_evidence(
    metrics: BacktestMetrics,
    *,
    method: str,
    identity_payload: dict,
    disqualifying_reasons: Iterable[str] = (),
) -> dict:
    """Apply the canonical gate and retain an explicit rejection reason.

    The historical runner currently produces one chronological holdout rather
    than the fold-based ``walk_forward_oos`` contract required for Fact Store
    promotion.  It therefore records a diagnostic decision, but can never
    manufacture validated signal evidence by setting ``validated=True``.
    """
    disqualifiers = tuple(disqualifying_reasons)
    passed_metrics_gate = metrics.validate()
    reasons: list[str] = []
    if metrics.sharpe_ratio < 1.0:
        reasons.append("oos_sharpe_below_1.0")
    if metrics.max_drawdown > 0.25:
        reasons.append("oos_drawdown_above_25pct")
    if metrics.num_trading_days < 504:
        reasons.append("oos_observations_below_504")
    if not passed_metrics_gate:
        # Keep the reason list defensive if the gate changes in one place but
        # this helper is not updated in the same commit.
        reasons.append("canonical_metrics_gate_failed")
    reasons.extend(disqualifiers)

    payload = {
        "method": method,
        "metrics": asdict(metrics),
        "identity": identity_payload,
    }
    validation_id = "holdout-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "validation_method": method,
        "validation_id": validation_id,
        "validation_oos_days": metrics.num_trading_days,
        "backtest_sharpe": float(metrics.sharpe_ratio),
        "backtest_sortino": float(metrics.sortino_ratio),
        "backtest_max_drawdown": float(metrics.max_drawdown),
        "metrics_gate_passed": bool(passed_metrics_gate),
        "data_quality_gate_passed": not bool(disqualifiers),
        "validated": False,
        "eligible_for_factstore": False,
        "rejection_reasons": sorted(set(reasons)) or [
            "chronological_holdout_is_not_walk_forward_evidence"
        ],
    }


def run_ten_year_oos_validation(
    cache_dir: Path = Path("data_cache"),
    train_years: int = 10,
    min_symbols: int = 10,
    candidates: Optional[Iterable[MomentumConfig]] = None,
    security_master: Optional[PointInTimeSecurityMaster] = None,
    min_oos_days: int = MIN_OOS_TRADING_DAYS,
    price_column: str = "Close",
) -> HistoricalValidationResult:
    """Select on a real 10-year train window, then freeze and test OOS."""
    prices, coverage, nifty = load_real_nse_universe(
        cache_dir=cache_dir,
        train_years=train_years,
        min_symbols=min_symbols,
        security_master=security_master,
        min_oos_days=min_oos_days,
        price_column=price_column,
    )
    common_start = max(series.index.min() for series in prices.values())
    common_end = min(series.index.max() for series in prices.values())
    available_index = nifty.loc[common_start:common_end].index
    if len(available_index) <= min_oos_days:
        raise HistoricalValidationError(
            f"Only {len(available_index)} common benchmark sessions available; "
            f"need more than {min_oos_days}"
        )
    oos_start_position = len(available_index) - min_oos_days
    test_start = pd.Timestamp(available_index[oos_start_position]).normalize()
    train_end = pd.Timestamp(available_index[oos_start_position - 1]).normalize()
    train_start = (
        train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
    ).normalize()
    test_end = pd.Timestamp(available_index[-1]).normalize()

    candidate_list = list(candidates or [
        MomentumConfig(126, 21, 21),
        MomentumConfig(252, 21, 21),
        MomentumConfig(378, 21, 21),
        MomentumConfig(252, 63, 21),
        MomentumConfig(252, 21, 63),
    ])
    train_prices = {symbol: series.loc[:train_end] for symbol, series in prices.items()}
    candidate_metrics = {}
    for config in candidate_list:
        train_stream = cross_sectional_momentum(
            train_prices, train_start, train_end, config,
            security_master=security_master,
            trading_index=nifty.index[nifty.index <= train_end],
        )
        candidate_metrics[json.dumps(asdict(config), sort_keys=True)] = _metrics(train_stream)

    # Selection is made only from train data. No OOS value is used here.
    selected = max(
        candidate_list,
        key=lambda config: (
            candidate_metrics[json.dumps(asdict(config), sort_keys=True)]["sharpe_ratio"],
            -candidate_metrics[json.dumps(asdict(config), sort_keys=True)]["max_drawdown"],
        ),
    )
    test_stream = cross_sectional_momentum(
        prices, test_start, test_end, selected,
        security_master=security_master,
        trading_index=nifty.index,
    )
    benchmark = nifty.pct_change().loc[test_start:test_end].dropna()
    test_metric_object = compute_metrics(test_stream.dropna())
    source_manifest_hash = hashlib.sha256(
        json.dumps([asdict(item) for item in coverage], sort_keys=True).encode("utf-8")
    ).hexdigest()
    validation = _validation_evidence(
        test_metric_object,
        method="chronological_holdout_oos",
        identity_payload={
            "train_start": str(train_start.date()),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "selected_config": asdict(selected),
            "source_manifest_sha256": source_manifest_hash,
        },
        disqualifying_reasons=(
            () if security_master is not None else ("survivorship_biased_universe",)
        ),
    )
    notes = [
        "All price inputs came from cached *_v5.parquet files; no synthetic values were used.",
        "Parameters were selected only on the training window and then frozen for OOS.",
        (
            "Point-in-time security master supplied; membership is evaluated at each rebalance date."
            if security_master is not None else
            "Universe is the set of cached symbols with complete common coverage; this is survivorship-biased and is reported explicitly."
        ),
        "Data ends at the latest common cached date; it is not a live-through-today result.",
        "This chronological holdout is diagnostic only and cannot create Fact Store validation evidence; the canonical walk-forward contract remains unmet.",
    ]
    return HistoricalValidationResult(
        train_start=str(train_start.date()),
        train_end=str(train_end.date()),
        test_start=str(test_start.date()),
        test_end=str(test_end.date()),
        universe_size=len(prices),
        universe_status=("point_in_time" if security_master is not None else "survivorship_biased"),
        security_master_sha256=(security_master.sha256 if security_master is not None else None),
        source_files=coverage,
        selected_config=selected,
        training_metrics=candidate_metrics[json.dumps(asdict(selected), sort_keys=True)],
        test_metrics=_metrics(test_stream),
        benchmark_metrics=_metrics(benchmark),
        candidate_training_metrics=candidate_metrics,
        validation=validation,
        notes=notes,
    )
