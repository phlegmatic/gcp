"""Pure, framework-agnostic inference for the weighted-ensemble bundle.

Consumes the exact artifact produced by Pipeline 2's `build_ensemble` step:

    {
        "type": "weighted_ensemble",
        "members": [{"model_name", "weight", "estimator"}, ...],
        "feature_cols": [...],
        "metrics": {"rmse", "mae", "mape"},
    }

Reuses `forecasting.models.features.build_features` so serving features are
computed by the SAME code path as training -> no train/serve skew.

This module deliberately has NO web or heavy-cloud imports; the single optional
GCS read is isolated in `load_bundle` and imported lazily.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from forecasting.models.features import DATE_COL, TARGET_COL, build_features


@dataclass(frozen=True)
class LoadedModel:
    """In-memory representation of a deserialized ensemble bundle."""

    bundle: dict[str, Any]

    @property
    def feature_cols(self) -> list[str]:
        return list(self.bundle["feature_cols"])

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self.bundle.get("metrics", {}))

    @property
    def members(self) -> list[dict[str, Any]]:
        return list(self.bundle["members"])


def load_bundle_from_bytes(payload: bytes) -> LoadedModel:
    """Deserialize a pickled ensemble bundle from raw bytes."""
    bundle = pickle.loads(payload)
    _validate_bundle(bundle)
    return LoadedModel(bundle=bundle)


def load_bundle_from_gcs(uri: str) -> LoadedModel:
    """Download and deserialize the ensemble bundle from a gs:// URI.

    GCS import is lazy so unit tests (which use `load_bundle_from_bytes`) never
    require the cloud SDK or credentials.
    """
    from forecasting.utils.gcs import read_bytes_gcs

    return load_bundle_from_bytes(read_bytes_gcs(uri))


def _validate_bundle(bundle: dict[str, Any]) -> None:
    for key in ("members", "feature_cols"):
        if key not in bundle:
            raise ValueError(f"Malformed ensemble bundle: missing '{key}'.")
    if not bundle["members"]:
        raise ValueError("Ensemble bundle contains no members.")


def _ensemble_row_prediction(model: LoadedModel, x_row: pd.DataFrame) -> float:
    """Weighted sum of each member's single-row prediction."""
    total = 0.0
    for member in model.members:
        pred = float(member["estimator"].predict(x_row)[0])
        total += float(member["weight"]) * pred
    return total


def forecast(
    model: LoadedModel,
    history: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """Recursively forecast `horizon` future daily steps.

    The features are lags/rolling stats of `demand`, so a multi-step forecast
    must feed each prediction back into the history before computing the next
    step's features. This mirrors how the model would be used in production and
    avoids look-ahead leakage.

    Parameters
    ----------
    model:
        A `LoadedModel` (deserialized ensemble bundle).
    history:
        DataFrame with at least `ds` (date) and `demand` columns, chronologically
        ordered. Must be long enough to warm up the largest lag/rolling window.
    horizon:
        Number of future daily steps to predict (>= 1).

    Returns
    -------
    DataFrame with columns [`ds`, `demand`] for the forecasted future steps,
    where `demand` holds the predicted values.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if TARGET_COL not in history.columns or DATE_COL not in history.columns:
        raise ValueError(f"history must contain '{DATE_COL}' and '{TARGET_COL}'.")

    working = history[[DATE_COL, TARGET_COL]].copy()
    working[DATE_COL] = pd.to_datetime(working[DATE_COL])
    working = working.sort_values(DATE_COL).reset_index(drop=True)

    feature_cols = model.feature_cols
    out_dates: list[pd.Timestamp] = []
    out_preds: list[float] = []

    for _ in range(horizon):
        # Append a placeholder future row so build_features can compute calendar
        # features for the step being predicted; its lags come from real/prior
        # predicted demand already present in `working`.
        next_date = working[DATE_COL].iloc[-1] + pd.Timedelta(days=1)
        probe = pd.concat(
            [working, pd.DataFrame({DATE_COL: [next_date], TARGET_COL: [np.nan]})],
            ignore_index=True,
        )
        # build_features drops NaN rows (incl. our placeholder target), so we
        # recompute calendar features for the future row explicitly and pull
        # lag/rolling values from the fully-populated feature frame.
        feats = _features_for_next_step(probe, feature_cols)
        x_row = feats[feature_cols]
        yhat = _ensemble_row_prediction(model, x_row)

        out_dates.append(next_date)
        out_preds.append(yhat)
        # Feed the prediction back in for the next recursive step.
        working = pd.concat(
            [working, pd.DataFrame({DATE_COL: [next_date], TARGET_COL: [yhat]})],
            ignore_index=True,
        )

    return pd.DataFrame({DATE_COL: out_dates, TARGET_COL: out_preds})


def _features_for_next_step(
    probe: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    """Compute the single feature row for the last (future) date in `probe`.

    We fill the future target with the last known value ONLY so build_features'
    dropna does not discard the row; lag/rolling features intentionally use
    `shift(1)`/`shift(lag)` and therefore never read this placeholder for the
    row being predicted.
    """
    filled = probe.copy()
    last_known = filled[TARGET_COL].ffill().iloc[-1]
    filled[TARGET_COL] = filled[TARGET_COL].fillna(last_known)
    feats = build_features(filled)
    if feats.empty:
        raise ValueError(
            "Not enough history to compute features; provide a longer series "
            "(at least the largest rolling window + max lag observations)."
        )
    row = feats.iloc[[-1]].copy()
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        raise ValueError(f"Feature columns missing at inference time: {missing}")
    return row
