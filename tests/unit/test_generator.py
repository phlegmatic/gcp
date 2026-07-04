"""Unit tests for the synthetic data generator + local runner parity."""

import pytest

from forecasting.data import (
    DriftConfig,
    SeriesConfig,
    generate_base_and_drift,
    generate_series,
    to_pipeline_frame,
)
from forecasting.data.generator import RAW_DATE_COL, RAW_TARGET_COL

pytestmark = pytest.mark.unit


def test_generate_series_schema_and_length():
    cfg = SeriesConfig(n_days=100, seed=1)
    df = generate_series(cfg)
    assert list(df.columns) == [RAW_DATE_COL, RAW_TARGET_COL]
    assert len(df) == 100
    assert (df[RAW_TARGET_COL] >= 0).all()  # demand clipped at 0


def test_generate_series_is_deterministic():
    cfg = SeriesConfig(n_days=50, seed=123)
    a = generate_series(cfg)
    b = generate_series(cfg)
    assert a.equals(b)


def test_base_and_drift_are_contiguous_and_shifted():
    base_cfg = SeriesConfig(n_days=120, base_level=80.0, seed=42)
    drift_cfg = DriftConfig(level_scale=1.5, seed=7)
    base_df, drift_df, full_df = generate_base_and_drift(base_cfg, drift_cfg)

    assert len(full_df) == len(base_df) + len(drift_df)
    # Drift window starts the day after base ends (contiguous).
    assert drift_df[RAW_DATE_COL].iloc[0] > base_df[RAW_DATE_COL].iloc[-1]
    # The drifted regime has a materially higher mean (level shock detectable).
    assert drift_df[RAW_TARGET_COL].mean() > base_df[RAW_TARGET_COL].mean()


def test_to_pipeline_frame_renames_columns():
    df = generate_series(SeriesConfig(n_days=10))
    renamed = to_pipeline_frame(df)
    assert list(renamed.columns) == ["ds", "demand"]
