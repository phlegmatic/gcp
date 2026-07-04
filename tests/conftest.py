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
