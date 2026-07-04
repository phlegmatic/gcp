"""Pure-numpy forecasting error metrics (no cloud deps -> trivially unit-tested)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("Empty arrays passed to metric.")
    return yt, yp


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error (%). `eps` guards against divide-by-zero."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)
