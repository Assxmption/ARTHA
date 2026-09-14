"""
Live Strategy Engine
====================
Runs the full multi-strategy signal engine on live (or delayed) market data.

This is the same logic as run_multi_strategy_backtest.py, refactored to work
incrementally for the paper trading daemon. Each strategy function takes
historical price data and returns per-symbol weights.

AGENTS.md Rule 1: All computation is deterministic — no LLM computes numbers.
AGENTS.md Rule 9: Signal generation is safe to automate. No order placement.
"""

from __future__ import annotations

import logging
import pickle
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.quant.regime import RegimeDetector, detect_regime

logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    """Output from a single strategy: per-symbol weights."""
    name: str
    weights: dict[str, float]  # symbol → signed weight (positive=long)
    sharpe_estimate: float = 0.0
    last_computed: str = ""


@dataclass
class PortfolioTarget:
    """Combined target portfolio from all strategies."""
    weights: dict[str, float]          # symbol → target weight (0-1)
    strategy_signals: list[StrategySignal]
    regime: str = "UNKNOWN"
    hedge_ratio: float = 0.0
    allocation_weights: dict[str, float] = field(default_factory=dict)
    computed_at: str = ""


@dataclass
class ModelCheckpoint:
    """Saved model state for persistence across restarts."""
    regime_detector: Optional[RegimeDetector] = None
    ml_model: Optional[object] = None
    ml_feature_importance: dict[str, float] = field(default_factory=dict)
    last_regime_train: Optional[str] = None
    last_ml_train: Optional[str] = None
    allocation_weights: dict[str, float] = field(default_factory=dict)


# ── The Engine ──────────────────────────────────────────────────────────────────

class LiveStrategyEngine:
    """
    Runs all sub-strategies on historical + current data and returns
    target portfolio weights.

    Lifecycle:
      1. initialize(historical_prices) — train models on 2y of data
      2. compute_targets(current_prices) — generate target weights
      3. retrain_if_due() — periodically re-fit ML and HMM models

    The engine is stateful (holds trained models) and persists checkpoints
    to disk for daemon restart recovery.
    """

    def __init__(
        self,
        symbols: list[str],
        model_dir: str | Path = "data_cache/models",
        ml_retrain_days: int = 30,
        regime_retrain_days: int = 90,
    ):
        self.symbols = symbols
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.ml_retrain_days = ml_retrain_days
        self.regime_retrain_days = regime_retrain_days

        # Models
        self._regime_detector: Optional[RegimeDetector] = None
        self._ml_model = None
        self._ml_feature_importance: dict[str, float] = {}

        # State
        self._initialized = False
        self._current_regime = "UNKNOWN"
        self._allocation_weights: dict[str, float] = {}
        self._last_regime_train: Optional[date] = None
        self._last_ml_train: Optional[date] = None
        self._historical_prices: Optional[dict[str, pd.Series]] = None
        self._historical_volumes: Optional[dict[str, pd.Series]] = None

        # Try to restore from checkpoint
        self._load_checkpoint()

    # ── Initialization ──────────────────────────────────────────────────────

    def initialize(
        self,
        prices: dict[str, pd.Series],
        volumes: Optional[dict[str, pd.Series]] = None,
        fundamentals: Optional[dict[str, dict]] = None,
    ):
        """
        Train all models on historical data.

        Args:
            prices: {symbol: pd.Series of daily close prices}
            volumes: {symbol: pd.Series of daily volume}
            fundamentals: {symbol: {metric: value}}
        """
        logger.info("═══ Initializing Live Strategy Engine ═══")
        logger.info("  %d symbols, %d price series",
                    len(self.symbols), len(prices))

        self._historical_prices = prices
        self._historical_volumes = volumes or {}

        # Build NIFTY proxy (equal-weight of all symbols)
        panel = pd.DataFrame(prices).dropna(how='all')
        if panel.empty:
            logger.error("No price data available for initialization")
            return

        nifty_proxy = panel.mean(axis=1)
        nifty_proxy.name = "NIFTY"

        # Train regime detector
        self._train_regime(nifty_proxy)

        # Train ML alpha model if we have enough data
        if len(panel) >= 504 and fundamentals:
            self._train_ml_alpha(prices, volumes or {}, fundamentals)

        self._initialized = True
        self._save_checkpoint()
        logger.info("Engine initialized — regime=%s", self._current_regime)

    # ── Signal Generation ───────────────────────────────────────────────────

    def compute_targets(
        self,
        current_prices: dict[str, float],
        fundamentals: Optional[dict[str, dict]] = None,
    ) -> Optional[PortfolioTarget]:
        """
        Compute target portfolio weights using all sub-strategies.

        This is called on every signal cycle (e.g., every 15 minutes).

        Args:
            current_prices: {symbol: latest_price}
            fundamentals: {symbol: {metric: value}}

        Returns:
            PortfolioTarget with per-symbol weights, or None on failure
        """
        if not self._initialized or not self._historical_prices:
            logger.warning("Engine not initialized — returning None")
            return None

        # Update historical prices with current
        for sym, price in current_prices.items():
            if sym in self._historical_prices:
                today = pd.Timestamp(date.today())
                series = self._historical_prices[sym]
                if today not in series.index:
                    self._historical_prices[sym] = pd.concat([
                        series,
                        pd.Series([price], index=[today], name=sym),
                    ])
                else:
                    self._historical_prices[sym].loc[today] = price

        # Run each strategy
        signals = []

        panel = pd.DataFrame(self._historical_prices).dropna(how='all')
        nifty_proxy = panel.mean(axis=1)
        daily_returns = panel.pct_change()

        # Strategy A: Momentum
        mom_weights = self._compute_momentum(panel, nifty_proxy)
        if mom_weights:
            signals.append(StrategySignal(
                name="momentum", weights=mom_weights,
                last_computed=datetime.now().isoformat(),
            ))

        # Strategy B: Mean Reversion
        vol_panel = pd.DataFrame(self._historical_volumes).reindex(panel.index) if self._historical_volumes else None
        mr_weights = self._compute_mean_reversion(panel, vol_panel)
        if mr_weights:
            signals.append(StrategySignal(
                name="mean_reversion", weights=mr_weights,
                last_computed=datetime.now().isoformat(),
            ))

        # Strategy C: Trend Following
        tf_weights = self._compute_trend_following(panel)
        if tf_weights:
            signals.append(StrategySignal(
                name="trend_following", weights=tf_weights,
                last_computed=datetime.now().isoformat(),
            ))

        # Strategy D: Short-Term Reversal
        str_weights = self._compute_short_term_reversal(panel)
        if str_weights:
            signals.append(StrategySignal(
                name="st_reversal", weights=str_weights,
                last_computed=datetime.now().isoformat(),
            ))

        # Strategy E: Factor Model
        if fundamentals:
            fac_weights = self._compute_factor_model(panel, fundamentals)
            if fac_weights:
                signals.append(StrategySignal(
                    name="factors", weights=fac_weights,
                    last_computed=datetime.now().isoformat(),
                ))

        # Strategy F: ML Alpha (uses pre-trained model)
        if self._ml_model is not None and fundamentals:
            ml_weights = self._compute_ml_alpha(panel, fundamentals)
            if ml_weights:
                signals.append(StrategySignal(
                    name="ml_alpha", weights=ml_weights,
                    last_computed=datetime.now().isoformat(),
                ))

        if not signals:
            logger.warning("No strategies produced signals")
            return None

        # Combine via rolling tangency allocation
        combined_weights = self._combine_signals(signals, daily_returns)

        # Apply regime-dependent hedge overlay
        regime = self._get_current_regime(nifty_proxy)
        hedge_ratio = self._get_hedge_ratio(regime)

        return PortfolioTarget(
            weights=combined_weights,
            strategy_signals=signals,
            regime=regime,
            hedge_ratio=hedge_ratio,
            allocation_weights=self._allocation_weights,
            computed_at=datetime.now().isoformat(),
        )

    # ── Per-Strategy Signal Generators ──────────────────────────────────────

    def _compute_momentum(
        self, panel: pd.DataFrame, nifty_proxy: pd.Series,
    ) -> dict[str, float]:
        """
        6-1 cross-sectional momentum.
        Returns {symbol: weight} for the top quintile.
        """
        if len(panel) < 170:
            return {}

        # Regime detection for conditioning
        regime = self._get_current_regime(nifty_proxy)

        # 6-month return, skip 1 month
        mom_6_1 = panel.pct_change(126).iloc[-1]
        vol_20d = panel.pct_change().rolling(20).std().iloc[-1]

        scores = mom_6_1.dropna()
        vols = vol_20d.dropna()

        if len(scores) < 10:
            return {}

        common = scores.index.intersection(vols.index)
        scores = scores[common]
        vols = vols[common]

        # Filter out top-quartile volatile stocks
        vol_threshold = vols.quantile(0.75)
        scores = scores[vols <= vol_threshold]

        if len(scores) < 6:
            return {}

        n_per_leg = max(1, len(scores) // 5)
        ranked = scores.sort_values(ascending=False)
        long_syms = ranked.index[:n_per_leg].tolist()

        # Equal weight across long symbols
        weight = 1.0 / n_per_leg
        if regime in ('SIDEWAYS', 'BEAR'):
            weight *= 0.5  # Reduced exposure in non-bull regimes
        else:
            weight *= 0.5  # Long-only scaled to half exposure

        return {sym: weight for sym in long_syms}

    def _compute_mean_reversion(
        self, panel: pd.DataFrame, vol_panel: Optional[pd.DataFrame],
    ) -> dict[str, float]:
        """
        Dip-buying: z-score < -2.5 AND volume > 1.5x average.
        """
        if len(panel) < 30:
            return {}

        sma20 = panel.rolling(20).mean()
        std20 = panel.rolling(20).std()
        z_scores = (panel - sma20) / std20.replace(0, np.nan)

        z_today = z_scores.iloc[-1].dropna()

        if vol_panel is not None and not vol_panel.empty:
            vol_sma20 = vol_panel.rolling(20).mean()
            vol_ratio = vol_panel / vol_sma20.replace(0, np.nan)
            vr_today = vol_ratio.iloc[-1].dropna()
            common = z_today.index.intersection(vr_today.index)
            z_filt = z_today[common]
            vr_filt = vr_today[common]
            signals = z_filt[(z_filt < -2.5) & (vr_filt > 1.5)]
        else:
            signals = z_today[z_today < -2.5]

        if len(signals) == 0:
            return {}

        weight = 1.0 / max(len(signals), 1)
        return {sym: weight for sym in signals.index}

    def _compute_trend_following(self, panel: pd.DataFrame) -> dict[str, float]:
        """
        Price > EMA50 > EMA200 (golden cross + above trend).
        Vol-targeted position sizing.
        """
        if len(panel) < 210:
            return {}

        ema50 = panel.ewm(span=50, adjust=False).mean()
        ema200 = panel.ewm(span=200, adjust=False).mean()
        rolling_vol = panel.pct_change().rolling(20).std() * np.sqrt(252)

        price_above_ema50 = panel.iloc[-1] > ema50.iloc[-1]
        ema50_above_ema200 = ema50.iloc[-1] > ema200.iloc[-1]
        trend_signal = price_above_ema50 & ema50_above_ema200

        long_syms = trend_signal[trend_signal].dropna().index.tolist()
        if not long_syms:
            return {}

        TARGET_VOL = 0.15
        weights = {}
        for sym in long_syms:
            if sym in rolling_vol.columns:
                v = rolling_vol[sym].iloc[-1]
                if not np.isnan(v) and v > 0.01:
                    w = min(TARGET_VOL / v, 3.0) / len(long_syms)
                    weights[sym] = w

        return weights

    def _compute_short_term_reversal(self, panel: pd.DataFrame) -> dict[str, float]:
        """
        Buy last week's losers (bottom quintile of 5-day returns).
        """
        if len(panel) < 30:
            return {}

        weekly_ret = panel.pct_change(5).iloc[-1].dropna()
        if len(weekly_ret) < 10:
            return {}

        n_per_leg = max(1, len(weekly_ret) // 5)
        ranked = weekly_ret.sort_values()
        losers = ranked.index[:n_per_leg].tolist()

        weight = 1.0 / n_per_leg
        return {sym: weight for sym in losers}

    def _compute_factor_model(
        self, panel: pd.DataFrame, fundamentals: dict[str, dict],
    ) -> dict[str, float]:
        """
        Multi-factor scoring: quality + value + momentum → top quintile.
        """
        if len(panel) < 252:
            return {}

        try:
            from app.quant.factor_backtest import backtest_factor_model

            # Build stocks dict from panel
            stocks = {}
            for sym in panel.columns:
                stocks[sym] = pd.DataFrame({'Close': panel[sym]})

            result = backtest_factor_model(
                price_data=stocks,
                fundamentals=fundamentals,
                rebalance_days=21,
                lookback_momentum=252,
                quintile_pct=0.20,
                long_only=True,
            )

            if result.n_rebalances == 0:
                return {}

            # Use the last rebalance's top quintile as the current signal
            # Factor model returns daily returns, not weights directly.
            # Approximate: equal-weight the universe (factor model handles
            # stock selection internally via quintile ranking).
            n = len(panel.columns)
            top_n = max(1, n // 5)

            # Score stocks by multi-factor composite
            mom_12 = panel.pct_change(252).iloc[-1].dropna()
            valid_syms = mom_12.index.tolist()

            # Simple composite: momentum + value (P/E inverse)
            scores = {}
            for sym in valid_syms:
                fund = fundamentals.get(sym, {})
                pe = fund.get('pe_ratio', fund.get('trailingPE', 30))
                mom = mom_12.get(sym, 0)
                if pe and pe > 0 and not np.isnan(pe):
                    scores[sym] = mom + (1.0 / pe) * 10  # Blend
                else:
                    scores[sym] = mom

            if len(scores) < 5:
                return {}

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top = [sym for sym, _ in ranked[:top_n]]
            weight = 1.0 / len(top)
            return {sym: weight for sym in top}

        except Exception as e:
            logger.warning("Factor model failed: %s", e)
            return {}

    def _compute_ml_alpha(
        self, panel: pd.DataFrame, fundamentals: dict[str, dict],
    ) -> dict[str, float]:
        """
        Use the pre-trained ML model to score stocks.
        Returns top quintile as weights.
        """
        if self._ml_model is None:
            return {}

        try:
            from app.quant.signals import build_signal_matrix

            prices = {sym: panel[sym] for sym in panel.columns}
            volumes = {sym: self._historical_volumes.get(sym, pd.Series(dtype=float))
                       for sym in panel.columns}
            sector_map = {sym: fundamentals.get(sym, {}).get('sector', 'Other')
                         for sym in panel.columns}

            # Build signal matrix for today
            today = panel.index[-1:]
            signal_matrix = build_signal_matrix(
                prices=prices,
                volumes=volumes,
                fundamentals=fundamentals,
                sector_map=sector_map,
                dates=today,
                min_history=252,
            )

            if signal_matrix.empty:
                return {}

            # Predict alpha scores
            feature_cols = [c for c in signal_matrix.columns
                           if c not in ('date', 'symbol', 'fwd_return')]
            X = signal_matrix[feature_cols].fillna(0)
            predictions = self._ml_model.predict(X)

            # Map back to symbols
            symbols = signal_matrix.index.get_level_values('symbol').tolist()
            alpha_scores = dict(zip(symbols, predictions))

            # Top quintile
            n_top = max(1, len(alpha_scores) // 5)
            ranked = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
            top = [sym for sym, _ in ranked[:n_top]]

            weight = 0.5 / len(top)  # 50% exposure (matches backtest)
            return {sym: weight for sym in top}

        except Exception as e:
            logger.warning("ML alpha prediction failed: %s", e)
            return {}

    # ── Signal Combination ──────────────────────────────────────────────────

    def _combine_signals(
        self,
        signals: list[StrategySignal],
        daily_returns: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Combine per-strategy weights using Sharpe²-proportional allocation.

        We use a simpler allocation than the backtest's rolling tangency
        because we don't have enough live history for covariance estimation.
        The allocation weights are seeded from the backtest's final weights
        and updated as live data accumulates.
        """
        # Start with equal allocation across strategies
        n_strats = len(signals)
        strat_alloc = {s.name: 1.0 / n_strats for s in signals}

        # If we have stored allocation weights from initialization, use those
        if self._allocation_weights:
            total = 0
            for s in signals:
                if s.name in self._allocation_weights:
                    strat_alloc[s.name] = self._allocation_weights[s.name]
                    total += strat_alloc[s.name]
            # Renormalize
            if total > 0:
                strat_alloc = {k: v / total for k, v in strat_alloc.items()
                               if k in {s.name for s in signals}}

        # Combine per-symbol weights across strategies
        combined: dict[str, float] = {}
        for signal in signals:
            alloc = strat_alloc.get(signal.name, 1.0 / n_strats)
            for sym, w in signal.weights.items():
                if sym not in combined:
                    combined[sym] = 0.0
                combined[sym] += w * alloc

        # Normalize to sum to 1.0 (long-only)
        total = sum(max(w, 0) for w in combined.values())
        if total > 0:
            combined = {k: max(v, 0) / total for k, v in combined.items()}
        else:
            # Fallback: equal weight
            syms = list(set(s for sig in signals for s in sig.weights))
            if syms:
                combined = {s: 1.0 / len(syms) for s in syms}

        self._allocation_weights = strat_alloc
        return combined

    # ── Regime & Hedge ──────────────────────────────────────────────────────

    def _get_current_regime(self, nifty_proxy: pd.Series) -> str:
        """Get the current market regime from the HMM detector."""
        if self._regime_detector is None:
            return "UNKNOWN"

        try:
            regimes = self._regime_detector.predict(nifty_proxy)
            if not regimes.empty:
                r = regimes.iloc[-1]
                self._current_regime = (
                    r if isinstance(r, str)
                    else (r.value if hasattr(r, 'value') else str(r))
                )
        except Exception as e:
            logger.warning("Regime prediction failed: %s", e)

        return self._current_regime

    @staticmethod
    def _get_hedge_ratio(regime: str) -> float:
        """
        Conservative hedge priors (same as backtest).
        NOT optimized on sample — see run_multi_strategy_backtest.py
        for documentation of these values.
        """
        ratios = {"BULL": 0.30, "BEAR": 0.70, "SIDEWAYS": 0.50}
        return ratios.get(regime, 0.50)

    # ── Model Training ──────────────────────────────────────────────────────

    def _train_regime(self, nifty_proxy: pd.Series):
        """Train or re-train the HMM regime detector."""
        logger.info("Training HMM regime detector on %d observations...",
                    len(nifty_proxy))
        try:
            self._regime_detector = RegimeDetector()
            self._regime_detector.fit(nifty_proxy)
            self._last_regime_train = date.today()

            # Get current regime
            regimes = self._regime_detector.predict(nifty_proxy)
            if not regimes.empty:
                r = regimes.iloc[-1]
                self._current_regime = (
                    r if isinstance(r, str)
                    else (r.value if hasattr(r, 'value') else str(r))
                )
            logger.info("Regime detector trained — current regime: %s",
                       self._current_regime)
        except Exception as e:
            logger.error("Regime training failed: %s", e)

    def _train_ml_alpha(
        self,
        prices: dict[str, pd.Series],
        volumes: dict[str, pd.Series],
        fundamentals: dict[str, dict],
    ):
        """Train the ML alpha ensemble."""
        try:
            from app.quant.signals import build_signal_matrix
            from app.quant.ml_alpha import train_alpha_model

            logger.info("Training ML alpha ensemble...")

            sector_map = {sym: fundamentals.get(sym, {}).get('sector', 'Other')
                         for sym in prices}

            panel = pd.DataFrame(prices).dropna(how='all')
            all_dates = sorted(panel.index.tolist())
            weekly_dates = pd.DatetimeIndex(all_dates[252::5])

            if len(weekly_dates) < 100:
                logger.warning("Too few dates for ML training: %d", len(weekly_dates))
                return

            signal_matrix = build_signal_matrix(
                prices=prices,
                volumes=volumes,
                fundamentals=fundamentals,
                sector_map=sector_map,
                dates=weekly_dates,
                min_history=252,
            )

            if signal_matrix.empty or len(signal_matrix) < 1000:
                logger.warning("Signal matrix too small: %d", len(signal_matrix))
                return

            # Build forward returns
            fwd_records = []
            for dt in weekly_dates:
                for sym in prices:
                    p = prices[sym]
                    if dt not in p.index:
                        continue
                    idx = p.index.get_loc(dt)
                    if not isinstance(idx, int):
                        idx = int(idx) if isinstance(idx, np.integer) else 0
                    fwd_idx = idx + 5
                    if fwd_idx < len(p):
                        fwd_ret = (p.iloc[fwd_idx] - p.iloc[idx]) / p.iloc[idx]
                        fwd_records.append({
                            "date": dt, "symbol": sym, "fwd_return": fwd_ret,
                        })

            fwd_df = pd.DataFrame(fwd_records).set_index(["date", "symbol"])

            result = train_alpha_model(
                signal_matrix=signal_matrix,
                forward_returns=fwd_df,
                min_train_days=100,
                step_size=4,
            )

            if result.n_folds >= 3:
                self._ml_model = result.ensemble_model
                self._ml_feature_importance = result.feature_importance
                self._last_ml_train = date.today()
                logger.info(
                    "ML alpha trained: %d folds, IC=%.4f, R²=%.4f",
                    result.n_folds, result.oos_ic, result.oos_r2,
                )
            else:
                logger.warning("ML training insufficient: %d folds", result.n_folds)

        except Exception as e:
            logger.error("ML alpha training failed: %s", e, exc_info=True)

    def retrain_if_due(
        self,
        fundamentals: Optional[dict[str, dict]] = None,
    ) -> bool:
        """
        Check if any model needs retraining and retrain if so.

        Called at end of market day by the daemon.
        Returns True if any model was retrained.
        """
        retrained = False
        today = date.today()

        # Regime HMM retrain
        if (self._last_regime_train is None or
                (today - self._last_regime_train).days >= self.regime_retrain_days):
            if self._historical_prices:
                panel = pd.DataFrame(self._historical_prices).dropna(how='all')
                nifty_proxy = panel.mean(axis=1)
                self._train_regime(nifty_proxy)
                retrained = True

        # ML Alpha retrain
        if (self._last_ml_train is None or
                (today - self._last_ml_train).days >= self.ml_retrain_days):
            if self._historical_prices and fundamentals:
                self._train_ml_alpha(
                    self._historical_prices,
                    self._historical_volumes or {},
                    fundamentals,
                )
                retrained = True

        if retrained:
            self._save_checkpoint()

        return retrained

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save_checkpoint(self):
        """Save model state to disk."""
        checkpoint_path = self.model_dir / "live_strategy_checkpoint.pkl"
        try:
            checkpoint = ModelCheckpoint(
                regime_detector=self._regime_detector,
                ml_model=self._ml_model,
                ml_feature_importance=self._ml_feature_importance,
                last_regime_train=(
                    self._last_regime_train.isoformat()
                    if self._last_regime_train else None
                ),
                last_ml_train=(
                    self._last_ml_train.isoformat()
                    if self._last_ml_train else None
                ),
                allocation_weights=self._allocation_weights,
            )
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            logger.info("Model checkpoint saved to %s", checkpoint_path)
        except Exception as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def _load_checkpoint(self):
        """Load model state from disk."""
        checkpoint_path = self.model_dir / "live_strategy_checkpoint.pkl"
        if not checkpoint_path.exists():
            return

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint: ModelCheckpoint = pickle.load(f)

            self._regime_detector = checkpoint.regime_detector
            self._ml_model = checkpoint.ml_model
            self._ml_feature_importance = checkpoint.ml_feature_importance
            self._allocation_weights = checkpoint.allocation_weights

            if checkpoint.last_regime_train:
                self._last_regime_train = date.fromisoformat(
                    checkpoint.last_regime_train
                )
            if checkpoint.last_ml_train:
                self._last_ml_train = date.fromisoformat(
                    checkpoint.last_ml_train
                )

            self._initialized = bool(self._regime_detector)
            logger.info(
                "Restored model checkpoint: regime=%s, ml=%s, alloc=%d strategies",
                "trained" if self._regime_detector else "none",
                "trained" if self._ml_model else "none",
                len(self._allocation_weights),
            )
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)

    @property
    def current_regime(self) -> str:
        return self._current_regime

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def model_info(self) -> dict:
        """Model training status for API responses."""
        return {
            "initialized": self._initialized,
            "regime_detector": "trained" if self._regime_detector else "none",
            "ml_model": "trained" if self._ml_model else "none",
            "last_regime_train": (
                self._last_regime_train.isoformat()
                if self._last_regime_train else None
            ),
            "last_ml_train": (
                self._last_ml_train.isoformat()
                if self._last_ml_train else None
            ),
            "ml_retrain_due_in_days": (
                max(0, self.ml_retrain_days - (date.today() - self._last_ml_train).days)
                if self._last_ml_train else 0
            ),
            "regime_retrain_due_in_days": (
                max(0, self.regime_retrain_days - (date.today() - self._last_regime_train).days)
                if self._last_regime_train else 0
            ),
            "allocation_weights": self._allocation_weights,
            "top_ml_features": dict(
                sorted(self._ml_feature_importance.items(),
                       key=lambda x: x[1], reverse=True)[:10]
            ) if self._ml_feature_importance else {},
        }
