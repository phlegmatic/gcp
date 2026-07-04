"""Pure training + ensembling logic (framework-agnostic, unit-testable).

The KFP components in `forecasting/components` are thin wrappers that call into
this module. Keeping the ML here means we can test it without spinning up any
GCP resource.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from forecasting.utils.metrics import mae, mape, rmse

SUPPORTED_MODELS = ("ridge", "random_forest")


@dataclass
class TrainResult:
    model_name: str
    estimator: object
    metrics: dict[str, float]


def _make_estimator(model_name: str, seed: int = 42):
    if model_name == "ridge":
        return Ridge(alpha=1.0, random_state=seed)
    if model_name == "random_forest":
        # Small forest keeps memory + CPU-time within e2-standard-2 limits.
        return RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=1, random_state=seed
        )
    raise ValueError(f"Unknown model '{model_name}'. Supported: {SUPPORTED_MODELS}")


def train_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> TrainResult:
    """Fit a single model and compute validation metrics."""
    estimator = _make_estimator(model_name, seed=seed)
    estimator.fit(x_train, y_train)
    preds = estimator.predict(x_val)
    metrics = {
        "mae": mae(y_val, preds),
        "rmse": rmse(y_val, preds),
        "mape": mape(y_val, preds),
    }
    return TrainResult(model_name=model_name, estimator=estimator, metrics=metrics)


def inverse_error_weights(
    rmses: dict[str, float], eps: float = 1e-9
) -> dict[str, float]:
    """Compute normalized inverse-RMSE weights (lower error -> higher weight)."""
    inv = {name: 1.0 / (value + eps) for name, value in rmses.items()}
    total = sum(inv.values())
    return {name: value / total for name, value in inv.items()}


def ensemble_predict(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Weighted average of per-model prediction vectors."""
    names = list(predictions)
    stacked = np.vstack([np.asarray(predictions[n]).ravel() for n in names])
    weight_vec = np.array([weights[n] for n in names]).reshape(-1, 1)
    return (stacked * weight_vec).sum(axis=0)
