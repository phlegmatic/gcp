"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_series() -> pd.DataFrame:
    """A deterministic daily demand series with weekly seasonality + trend."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    trend = np.linspace(50, 120, len(dates))
    weekly = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = rng.normal(0, 3, len(dates))
    demand = trend + weekly + noise
    return pd.DataFrame({"ds": dates, "demand": demand})


@pytest.fixture()
def ensemble_bundle(synthetic_series):
    """A small, real weighted-ensemble bundle matching build_ensemble's format.

    Trains Ridge + RandomForest on the synthetic series so the serving code can
    be exercised end-to-end without any cloud dependency.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    from forecasting.models.features import build_features, feature_columns

    feats = build_features(synthetic_series)
    cols = feature_columns(feats)
    x, y = feats[cols], feats["demand"]

    ridge = Ridge(alpha=1.0).fit(x, y)
    rf = RandomForestRegressor(
        n_estimators=25, max_depth=6, n_jobs=1, random_state=42
    ).fit(x, y)

    return {
        "type": "weighted_ensemble",
        "members": [
            {"model_name": "ridge", "weight": 0.6, "estimator": ridge},
            {"model_name": "random_forest", "weight": 0.4, "estimator": rf},
        ],
        "feature_cols": cols,
        "metrics": {"rmse": 3.2, "mae": 2.5, "mape": 4.0},
    }
