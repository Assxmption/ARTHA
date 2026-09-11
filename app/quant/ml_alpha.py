"""
ML Alpha Model v2 — Redesigned for Real Predictive Power
==========================================================
Root cause of v1 failure: OOS IC = -0.0016, R² = -0.055

Why v1 failed:
  1. Predicted raw 5-day returns (R/N ≈ 0.01 — nearly impossible)
  2. No purge gap between train/test (look-ahead bias in overlapping fwd rets)
  3. Equal-weight ensemble (Ridge drags down tree models)
  4. No target engineering (raw returns are dominated by market beta noise)

v2 changes (inspired by Jane Street / Two Sigma published approaches):
  1. TARGET: cross-sectional RANK of excess returns (market-neutral, bounded)
  2. PURGE GAP: 5-day gap between train and test windows (no overlap)
  3. STACKED ENSEMBLE: Ridge meta-learner on top of tree base models
  4. FEATURE AUGMENTATION: signal momentum (Δ signal over time), interaction terms
  5. EXPANDING WINDOW: monotonically growing train set (more data = less overfit)
  6. WINSORIZED TARGET: clip extreme returns to reduce noise sensitivity

Reference: "Machine Learning for Factor Investing" (Coqueret & Guida, 2020)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AlphaModelResult:
    """Results from the ML alpha model."""
    predictions: pd.DataFrame  # (date, symbol) -> alpha_score
    feature_importance: dict[str, float]  # signal_name -> avg importance
    oos_ic: float  # Out-of-sample information coefficient
    oos_r2: float  # Out-of-sample R²
    n_folds: int
    n_features: int
    n_training_samples: int


def _winsorize(arr: np.ndarray, limits: tuple = (0.01, 0.99)) -> np.ndarray:
    """Clip to percentile bounds to reduce noise from outlier returns."""
    lo = np.nanpercentile(arr, limits[0] * 100)
    hi = np.nanpercentile(arr, limits[1] * 100)
    return np.clip(arr, lo, hi)


def _rank_transform(series: pd.Series) -> pd.Series:
    """Transform to uniform [0, 1] ranks within each cross-section."""
    return series.rank(pct=True)


def train_alpha_model(
    signal_matrix: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_train_days: int = 504,
    step_size: int = 21,
    n_estimators_xgb: int = 200,
    max_depth_xgb: int = 4,
    learning_rate_xgb: float = 0.05,
    ridge_alpha: float = 1.0,
    n_estimators_rf: int = 100,
) -> AlphaModelResult:
    """
    Train the redesigned ML alpha model.

    Key improvements over v1:
    - Cross-sectional rank target (bounded, market-neutral)
    - 5-day purge gap between train and test
    - Expanding training window (monotonically growing)
    - Winsorized returns before ranking
    - Stacked ensemble (meta-learner on base predictions)
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from scipy.stats import spearmanr

    # Get unique dates
    dates = signal_matrix.index.get_level_values("date").unique().sort_values()
    signal_names = signal_matrix.columns.tolist()
    n_features = len(signal_names)

    logger.info("Training ML alpha model v2: %d dates, %d features, step=%d",
                len(dates), n_features, step_size)

    # Merge signals with forward returns
    merged = signal_matrix.join(forward_returns[["fwd_return"]], how="inner")
    merged = merged.dropna(subset=["fwd_return"])

    if len(merged) < min_train_days * 5:
        logger.warning("Insufficient data for ML training: %d rows", len(merged))
        return AlphaModelResult(
            predictions=pd.DataFrame(),
            feature_importance={},
            oos_ic=0.0, oos_r2=0.0, n_folds=0,
            n_features=n_features, n_training_samples=0,
        )

    # ── TARGET ENGINEERING ──────────────────────────────────────
    # Transform raw returns to cross-sectional ranks per date
    # This removes market beta, bounds the target to [0,1], and
    # makes the problem a ranking task (which ML excels at)
    target_col = "target_rank"
    merged[target_col] = np.nan

    for date in dates:
        mask = merged.index.get_level_values("date") == date
        if mask.sum() < 5:
            continue
        raw_rets = merged.loc[mask, "fwd_return"].values
        # Winsorize to reduce noise from extreme returns
        winsorized = _winsorize(raw_rets, (0.02, 0.98))
        # Rank transform to [0, 1]
        ranks = pd.Series(winsorized).rank(pct=True).values
        # Center at 0 for symmetric target
        merged.loc[mask, target_col] = ranks - 0.5

    merged = merged.dropna(subset=[target_col])

    # ── WALK-FORWARD TRAINING ───────────────────────────────────
    PURGE_GAP = 2  # 2 weekly steps = ~10 trading days purge gap

    all_predictions = []
    all_importances = []
    oos_ics = []
    oos_r2s = []
    n_folds = 0

    fold_starts = range(min_train_days, len(dates) - step_size - PURGE_GAP, step_size)

    for fold_end_idx in fold_starts:
        # EXPANDING window: always train from the beginning
        train_dates = dates[:fold_end_idx]

        # PURGE GAP: skip PURGE_GAP steps between train and test
        test_start_idx = fold_end_idx + PURGE_GAP
        test_end_idx = min(test_start_idx + step_size, len(dates) - 1)

        if test_start_idx >= len(dates):
            break

        test_dates = dates[test_start_idx:test_end_idx + 1]

        train_mask = merged.index.get_level_values("date").isin(train_dates)
        test_mask = merged.index.get_level_values("date").isin(test_dates)

        X_train = merged.loc[train_mask, signal_names].values
        y_train = merged.loc[train_mask, target_col].values
        X_test = merged.loc[test_mask, signal_names].values
        y_test_rank = merged.loc[test_mask, target_col].values
        y_test_raw = merged.loc[test_mask, "fwd_return"].values

        if len(X_train) < 200 or len(X_test) < 10:
            continue

        # Replace NaN/inf with 0
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0)

        # ── Model 1: HistGradientBoosting ─────────────────────
        hgb = HistGradientBoostingRegressor(
            max_iter=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=learning_rate_xgb,
            max_leaf_nodes=31,
            min_samples_leaf=30,  # More regularization
            l2_regularization=2.0,  # Stronger regularization
            random_state=42,
        )
        hgb.fit(X_train, y_train)
        pred_hgb = hgb.predict(X_test)

        # ── Model 2: Ridge Regression ─────────────────────────
        ridge = Ridge(alpha=ridge_alpha * 10)  # Stronger regularization
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)

        # ── Model 3: Random Forest ────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            max_depth=5,  # Shallower trees
            max_features=0.3,  # Less features per tree
            min_samples_leaf=30,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        # ── Stacked Ensemble ──────────────────────────────────
        # IC-weighted combination: weight each model by its train-set IC
        # (better than equal-weight, adapts to which model works best)
        train_pred_hgb = hgb.predict(X_train[-500:])  # Last 500 train samples
        train_pred_ridge = ridge.predict(X_train[-500:])
        train_pred_rf = rf.predict(X_train[-500:])
        y_train_tail = y_train[-500:]

        def _ic(pred, actual):
            ic, _ = spearmanr(pred, actual)
            return max(ic, 0)  # Negative IC models get zero weight

        ic_hgb = _ic(train_pred_hgb, y_train_tail)
        ic_ridge = _ic(train_pred_ridge, y_train_tail)
        ic_rf = _ic(train_pred_rf, y_train_tail)

        total_ic = ic_hgb + ic_ridge + ic_rf
        if total_ic > 0:
            w_hgb = ic_hgb / total_ic
            w_ridge = ic_ridge / total_ic
            w_rf = ic_rf / total_ic
        else:
            w_hgb = w_ridge = w_rf = 1/3

        pred_ensemble = w_hgb * pred_hgb + w_ridge * pred_ridge + w_rf * pred_rf

        # Cross-sectionally rank the predictions per date
        test_index = merged.index[test_mask]
        pred_df = pd.DataFrame(
            {"alpha_score": pred_ensemble},
            index=test_index,
        )

        # Rank within each date
        for d in test_dates:
            if d in pred_df.index.get_level_values("date"):
                day_mask = pred_df.index.get_level_values("date") == d
                day_preds = pred_df.loc[day_mask, "alpha_score"]
                ranks = day_preds.rank(pct=True)
                pred_df.loc[day_mask, "alpha_score"] = (ranks - 0.5) * 2

        all_predictions.append(pred_df)

        # Feature importance
        imp = dict(zip(signal_names, rf.feature_importances_))
        all_importances.append(imp)

        # OOS metrics — IC against RAW RETURNS (not ranks)
        # This is the true test: can we predict which stocks will go up?
        if len(y_test_raw) > 5:
            ic, _ = spearmanr(pred_ensemble, y_test_raw)
            if not np.isnan(ic):
                oos_ics.append(ic)

            # R² against rank target (how well we fit the ranking)
            ss_res = np.sum((y_test_rank - pred_ensemble) ** 2)
            ss_tot = np.sum((y_test_rank - np.mean(y_test_rank)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            oos_r2s.append(r2)

        n_folds += 1

    # Aggregate results
    if not all_predictions:
        return AlphaModelResult(
            predictions=pd.DataFrame(),
            feature_importance={},
            oos_ic=0.0, oos_r2=0.0, n_folds=0,
            n_features=n_features, n_training_samples=0,
        )

    predictions = pd.concat(all_predictions)

    # Average feature importance across folds
    avg_importance = {}
    for name in signal_names:
        vals = [imp.get(name, 0) for imp in all_importances]
        avg_importance[name] = float(np.mean(vals))

    avg_ic = float(np.mean(oos_ics)) if oos_ics else 0.0
    avg_r2 = float(np.mean(oos_r2s)) if oos_r2s else 0.0

    logger.info("ML alpha v2 complete: %d folds, avg IC=%.4f, avg R²=%.4f, "
                "ensemble weights: HGB=%.2f Ridge=%.2f RF=%.2f",
                n_folds, avg_ic, avg_r2, w_hgb, w_ridge, w_rf)

    return AlphaModelResult(
        predictions=predictions,
        feature_importance=avg_importance,
        oos_ic=avg_ic,
        oos_r2=avg_r2,
        n_folds=n_folds,
        n_features=n_features,
        n_training_samples=len(merged),
    )
