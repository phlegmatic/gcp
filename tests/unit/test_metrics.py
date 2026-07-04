"""Unit tests for pure metric functions (no cloud)."""

import numpy as np
import pytest

from forecasting.utils.metrics import mae, mape, rmse

pytestmark = pytest.mark.unit


def test_perfect_prediction_is_zero_error():
    y = [1.0, 2.0, 3.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape(y, y) == pytest.approx(0.0, abs=1e-6)


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = [0.0, 0.0, 0.0]
    y_pred = [0.0, 0.0, 9.0]
    assert rmse(y_true, y_pred) > mae(y_true, y_pred)


def test_mape_is_percentage():
    y_true = [100.0, 100.0]
    y_pred = [110.0, 90.0]
    assert mape(y_true, y_pred) == pytest.approx(10.0, rel=1e-3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_empty_raises():
    with pytest.raises(ValueError):
        rmse(np.array([]), np.array([]))
