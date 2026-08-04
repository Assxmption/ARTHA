"""
ML Alpha Model — Ensemble Predictor
=====================================
3-model ensemble that combines 50+ signals into a single alpha prediction
per stock per day using walk-forward cross-validation.

Models:
  1. XGBoost — captures non-linear signal interactions
  2. Ridge Regression — stable linear combination baseline
  3. Random Forest — bagged decorrelation

Walk-Forward Protocol (no data leakage):
  - Train on [0, T], predict on [T, T+step]
  - Slide T forward by step_size (21 trading days = 1 month)
  - Minimum 504 days (2 years) training window
  - Feature importance tracked at each fold

Output: alpha_score per stock per day (cross-sectionally ranked prediction
of next-day excess return).

Reference: Implementation Plan v5 §ML Alpha Model
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
    Train the 3-model ensemble using walk-forward cross-validation.

    Args:
        signal_matrix: DataFrame with MultiIndex (date, symbol), columns = signal names.
        forward_returns: DataFrame with MultiIndex (date, symbol), column 'fwd_return'.
        min_train_days: Minimum training window in trading days.
        step_size: Number of days to advance per fold.

    Returns:
        AlphaModelResult with predictions, feature importance, and metrics.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

    # Get unique dates
    dates = signal_matrix.index.get_level_values("date").unique().sort_values()
    signal_names = signal_matrix.columns.tolist()
    n_features = len(signal_names)

    logger.info("Training ML alpha model: %d dates, %d features, step=%d",
                len(dates), n_features, step_size)

    # Merge signals with forward returns
    merged = signal_matrix.join(forward_returns[["fwd_return"]], how="inner")
    merged = merged.dropna(subset=["fwd_return"])

    if len(merged) < min_train_days * 10:
        logger.warning("Insufficient data for ML training: %d rows", len(merged))
        return AlphaModelResult(
            predictions=pd.DataFrame(),
            feature_importance={},
            oos_ic=0.0,
            oos_r2=0.0,
            n_folds=0,
            n_features=n_features,
            n_training_samples=0,
        )

    # Walk-forward folds
    all_predictions = []
    all_importances = []
    oos_ics = []
    oos_r2s = []
    n_folds = 0

    fold_starts = range(min_train_days, len(dates) - step_size, step_size)

    for fold_end_idx in fold_starts:
        train_end_date = dates[fold_end_idx]
        test_start_date = dates[fold_end_idx]
        test_end_idx = min(fold_end_idx + step_size, len(dates) - 1)
        test_end_date = dates[test_end_idx]

        # Split
        train_dates = dates[:fold_end_idx]
        test_dates = dates[fold_end_idx:test_end_idx + 1]

        train_mask = merged.index.get_level_values("date").isin(train_dates)
        test_mask = merged.index.get_level_values("date").isin(test_dates)

        X_train = merged.loc[train_mask, signal_names].values
        y_train = merged.loc[train_mask, "fwd_return"].values
        X_test = merged.loc[test_mask, signal_names].values
        y_test = merged.loc[test_mask, "fwd_return"].values

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # Replace NaN/inf with 0
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0)

        # ── Model 1: HistGradientBoosting (sklearn native, no libomp) ──
        hgb = HistGradientBoostingRegressor(
            max_iter=n_estimators_xgb,
            max_depth=max_depth_xgb,
            learning_rate=learning_rate_xgb,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        )
        hgb.fit(X_train, y_train)
        pred_hgb = hgb.predict(X_test)

        # ── Model 2: Ridge Regression ─────────────────────────
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X_train, y_train)
        pred_ridge = ridge.predict(X_test)

        # ── Model 3: Random Forest ────────────────────────────
        rf = RandomForestRegressor(
            n_estimators=n_estimators_rf,
            max_depth=6,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)

        # ── Ensemble (equal weight) ───────────────────────────
        pred_ensemble = (pred_hgb + pred_ridge + pred_rf) / 3

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
                # Convert to cross-sectional z-score (rank-based)
                ranks = day_preds.rank(pct=True)
                pred_df.loc[day_mask, "alpha_score"] = (ranks - 0.5) * 2  # Scale to [-1, 1]

        all_predictions.append(pred_df)

        # Feature importance (Random Forest — reliable & available)
        imp = dict(zip(signal_names, rf.feature_importances_))
        all_importances.append(imp)

        # OOS metrics
        if len(y_test) > 5:
            # Information coefficient (rank correlation)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred_ensemble, y_test)
            if not np.isnan(ic):
                oos_ics.append(ic)

            # R²
            ss_res = np.sum((y_test - pred_ensemble) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            oos_r2s.append(r2)

        n_folds += 1

    # Aggregate results
    if not all_predictions:
        return AlphaModelResult(
            predictions=pd.DataFrame(),
            feature_importance={},
            oos_ic=0.0,
            oos_r2=0.0,
            n_folds=0,
            n_features=n_features,
            n_training_samples=0,
        )

    predictions = pd.concat(all_predictions)

    # Average feature importance across folds
    avg_importance = {}
    for name in signal_names:
        vals = [imp.get(name, 0) for imp in all_importances]
        avg_importance[name] = float(np.mean(vals))

    avg_ic = float(np.mean(oos_ics)) if oos_ics else 0.0
    avg_r2 = float(np.mean(oos_r2s)) if oos_r2s else 0.0

    logger.info("ML alpha training complete: %d folds, avg IC=%.4f, avg R²=%.4f",
                n_folds, avg_ic, avg_r2)

    return AlphaModelResult(
        predictions=predictions,
        feature_importance=avg_importance,
        oos_ic=avg_ic,
        oos_r2=avg_r2,
        n_folds=n_folds,
        n_features=n_features,
        n_training_samples=len(merged),
    )
