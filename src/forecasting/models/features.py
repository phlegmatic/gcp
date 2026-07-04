"""Time-series feature engineering shared by training and inference.

Deliberately simple (lag + rolling + calendar features) so both the Ridge and
Random Forest models consume an identical feature matrix -- a prerequisite for
a fair weighted ensemble.
"""

from __future__ import annotations

import pandas as pd

LAGS: tuple[int, ...] = (1, 7, 14)
ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
TARGET_COL = "demand"
DATE_COL = "ds"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame with lag/rolling/calendar features.

    Expects columns [`ds` (datetime), `demand` (float)] sorted ascending.
    Rows with NaNs introduced by lags are dropped.
    """
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL).reset_index(drop=True)

    for lag in LAGS:
        out[f"lag_{lag}"] = out[TARGET_COL].shift(lag)
    for window in ROLLING_WINDOWS:
        out[f"roll_mean_{window}"] = out[TARGET_COL].shift(1).rolling(window).mean()
        out[f"roll_std_{window}"] = out[TARGET_COL].shift(1).rolling(window).std()

    out["dayofweek"] = out[DATE_COL].dt.dayofweek
    out["month"] = out[DATE_COL].dt.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    return out.dropna().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything except date + target)."""
    return [c for c in df.columns if c not in (DATE_COL, TARGET_COL)]
