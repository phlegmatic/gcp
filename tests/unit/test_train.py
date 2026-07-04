"""Unit tests for feature engineering + training + ensembling logic."""

import numpy as np
import pytest

from forecasting.models.features import build_features, feature_columns
from forecasting.models.train import (
    ensemble_predict,
    inverse_error_weights,
    train_model,
)

pytestmark = pytest.mark.unit


def test_build_features_drops_nan_rows_and_adds_columns(synthetic_series):
    feats = build_features(synthetic_series)
    assert "lag_1" in feats.columns
    assert "roll_mean_7" in feats.columns
    assert "is_weekend" in feats.columns
    # No NaNs remain after dropna.
    assert not feats.isna().any().any()
    # Fewer rows than input because of lag/rolling warmup.
    assert len(feats) < len(synthetic_series)


def test_train_model_returns_reasonable_metrics(synthetic_series):
    feats = build_features(synthetic_series)
    cols = feature_columns(feats)
    split = int(len(feats) * 0.8)
    x_tr, y_tr = feats[cols].iloc[:split].to_numpy(), feats["demand"].iloc[:split]
    x_va, y_va = feats[cols].iloc[split:].to_numpy(), feats["demand"].iloc[split:]

    result = train_model("ridge", x_tr, y_tr.to_numpy(), x_va, y_va.to_numpy())
    assert result.model_name == "ridge"
    assert result.metrics["rmse"] >= 0
    # On a smooth synthetic series, error should be modest relative to scale.
    assert result.metrics["mape"] < 50


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        train_model(
            "xgboost", np.zeros((2, 1)), np.zeros(2), np.zeros((2, 1)), np.zeros(2)
        )


def test_inverse_error_weights_sum_to_one_and_favor_lower_rmse():
    weights = inverse_error_weights({"ridge": 2.0, "rf": 4.0})
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
    assert weights["ridge"] > weights["rf"]


def test_ensemble_predict_is_weighted_average():
    preds = {"a": np.array([0.0, 10.0]), "b": np.array([10.0, 0.0])}
    weights = {"a": 0.5, "b": 0.5}
    out = ensemble_predict(preds, weights)
    assert np.allclose(out, [5.0, 5.0])
