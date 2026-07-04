"""Integration test: full local pipeline run on generated data (no cloud).

Verifies that the local runner -- which reuses the same `forecasting.models.*`
code as the deployed KFP pipelines -- executes end-to-end and produces a
serialized ensemble with sane metrics.
"""

import pickle

import pytest

from forecasting.data import (
    DriftConfig,
    SeriesConfig,
    generate_base_and_drift,
    to_pipeline_frame,
)
from forecasting.local_runner import run_end_to_end_local, summarize

pytestmark = pytest.mark.integration


def test_local_end_to_end_produces_ensemble(tmp_path):
    base_cfg = SeriesConfig(n_days=200, seed=42)
    drift_cfg = DriftConfig(level_scale=1.4, seed=7)
    base_df, drift_df, full_raw = generate_base_and_drift(base_cfg, drift_cfg)

    pipeline_df = to_pipeline_frame(full_raw)
    split_date = str(drift_df["sale_date"].iloc[0])

    result = run_end_to_end_local(
        raw_df=pipeline_df,
        split_date=split_date,
        val_fraction=0.2,
        out_dir=str(tmp_path),
    )

    # Both models trained + an ensemble produced.
    assert set(result.training.per_model_metrics) == {"ridge", "random_forest"}
    assert pytest.approx(sum(result.training.weights.values()), rel=1e-9) == 1.0
    assert result.training.ensemble_metrics["rmse"] >= 0

    # Model bundle is serialized and loadable with the expected structure.
    with open(result.training.model_path, "rb") as fh:
        bundle = pickle.load(fh)
    assert bundle["type"] == "weighted_ensemble"
    assert len(bundle["members"]) == 2
    assert bundle["feature_cols"]

    # Drift should be detectable between the base and drifted windows.
    summary = summarize(result)
    assert summary["drift"]["dataset_drift"] is not None
