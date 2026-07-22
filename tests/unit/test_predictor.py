"""Unit tests for the pure serving inference core (no cloud, no web)."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from forecasting.serving import predictor

pytestmark = pytest.mark.unit


def test_load_bundle_from_bytes_roundtrip(ensemble_bundle):
    payload = pickle.dumps(ensemble_bundle)
    model = predictor.load_bundle_from_bytes(payload)
    assert model.feature_cols == ensemble_bundle["feature_cols"]
    assert model.metrics["rmse"] == 3.2
    assert len(model.members) == 2


def test_load_bundle_rejects_malformed():
    with pytest.raises(ValueError):
        predictor.load_bundle_from_bytes(pickle.dumps({"members": []}))
    with pytest.raises(ValueError):
        predictor.load_bundle_from_bytes(pickle.dumps({"feature_cols": []}))


def test_forecast_single_step_shape(ensemble_bundle, synthetic_series):
    model = predictor.load_bundle_from_bytes(pickle.dumps(ensemble_bundle))
    out = predictor.forecast(model, synthetic_series, horizon=1)
    assert list(out.columns) == ["ds", "demand"]
    assert len(out) == 1
    assert np.isfinite(out["demand"].iloc[0])


def test_forecast_multi_step_is_recursive_and_dated(ensemble_bundle, synthetic_series):
    model = predictor.load_bundle_from_bytes(pickle.dumps(ensemble_bundle))
    out = predictor.forecast(model, synthetic_series, horizon=7)
    assert len(out) == 7
    # Dates are consecutive daily and start the day after the last observation.
    last = pd.to_datetime(synthetic_series["ds"]).max()
    expected = pd.date_range(last + pd.Timedelta(days=1), periods=7, freq="D")
    assert list(pd.to_datetime(out["ds"])) == list(expected)
    assert np.all(np.isfinite(out["demand"].to_numpy()))


def test_forecast_predictions_are_in_reasonable_range(
    ensemble_bundle, synthetic_series
):
    model = predictor.load_bundle_from_bytes(pickle.dumps(ensemble_bundle))
    out = predictor.forecast(model, synthetic_series, horizon=5)
    recent = synthetic_series["demand"].tail(30)
    # On a smooth series, the forecast should sit near the recent scale.
    assert out["demand"].min() > recent.min() - 40
    assert out["demand"].max() < recent.max() + 40


def test_forecast_rejects_bad_horizon(ensemble_bundle, synthetic_series):
    model = predictor.load_bundle_from_bytes(pickle.dumps(ensemble_bundle))
    with pytest.raises(ValueError):
        predictor.forecast(model, synthetic_series, horizon=0)


def test_forecast_requires_columns(ensemble_bundle):
    model = predictor.load_bundle_from_bytes(pickle.dumps(ensemble_bundle))
    bad = pd.DataFrame({"date": [1, 2], "value": [3, 4]})
    with pytest.raises(ValueError):
        predictor.forecast(model, bad, horizon=1)
