"""Local, cloud-free end-to-end runner mirroring the deployed Vertex pipelines.

This module executes the SAME logical DAGs as the KFP pipelines, but using the
pure `forecasting.models.*` code paths and local filesystem artifacts instead of
BigQuery/GCS. The point: what you prototype in the notebook is functionally
identical to what runs serverlessly in production -- only the I/O backend differs.

Mapping
-------
Pipeline 1 (data_pipeline.py)             -> run_data_pipeline_local()
  run_dbt_transform (build features)          uses models.features.build_features
  extract_reference_and_current (BQ split)    pandas split on split_date
  detect_drift (Evidently -> GCS)             local Evidently report -> data/ dir

Pipeline 2 (training_pipeline.py)         -> run_training_pipeline_local()
  train_single_model x2 (parallel)            models.train.train_model (Ridge/RF)
  build_ensemble (fan-in, inverse-RMSE)       models.train.inverse_error_weights
                                              + ensemble_predict, serialized to disk
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from forecasting.models.features import build_features, feature_columns
from forecasting.models.train import (
    SUPPORTED_MODELS,
    ensemble_predict,
    inverse_error_weights,
    train_model,
)
from forecasting.utils.metrics import mae, mape, rmse


# ---------------------------------------------------------------------------
# Pipeline 1: data ingest, feature build, reference/current split, drift
# ---------------------------------------------------------------------------
@dataclass
class DataPipelineResult:
    features: pd.DataFrame
    reference: pd.DataFrame
    current: pd.DataFrame
    drift_summary: dict
    drift_report_path: str | None


def run_data_pipeline_local(
    raw_df: pd.DataFrame,
    split_date: str,
    out_dir: str | Path = "data/local_run",
    write_evidently_html: bool = True,
) -> DataPipelineResult:
    """Local equivalent of Pipeline 1.

    Parameters
    ----------
    raw_df : DataFrame with columns [`ds`, `demand`] (use
        `forecasting.data.to_pipeline_frame` on the generator's raw output).
    split_date : rows with ds < split_date form the drift *reference*; the rest
        are the *current* window (matches the BQ extract component).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step: build features (mirrors dbt mart `demand_features`).
    features = build_features(raw_df)
    features.to_parquet(out / "demand_features.parquet", index=False)

    # Step: reference/current split (mirrors extract_reference_and_current).
    split_ts = pd.to_datetime(split_date)
    reference = features[features["ds"] < split_ts].reset_index(drop=True)
    current = features[features["ds"] >= split_ts].reset_index(drop=True)

    # Step: drift detection (mirrors detect_drift). Evidently is optional so the
    # runner still works in a minimal env; falls back to a simple stats summary.
    drift_summary, report_path = _detect_drift_local(
        reference, current, out, write_evidently_html
    )

    return DataPipelineResult(
        features=features,
        reference=reference,
        current=current,
        drift_summary=drift_summary,
        drift_report_path=report_path,
    )


def _detect_drift_local(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out: Path,
    write_html: bool,
) -> tuple[dict, str | None]:
    """Run Evidently if available; otherwise a lightweight mean-shift summary."""
    numeric_cols = [c for c in reference.columns if c not in ("ds",)]
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference[numeric_cols],
            current_data=current[numeric_cols],
        )
        summary = report.as_dict()["metrics"][0]["result"]
        report_path: str | None = None
        if write_html:
            report_path = str(out / "drift_report.html")
            report.save_html(report_path)
        (out / "drift_summary.json").write_text(
            json.dumps(summary, default=str, indent=2)
        )
        return summary, report_path
    except ImportError:
        # Fallback: per-feature standardized mean shift + a crude drift flag.
        shifts = {}
        drifted = 0
        for col in numeric_cols:
            ref_mean, ref_std = reference[col].mean(), reference[col].std() + 1e-9
            cur_mean = current[col].mean()
            z = abs((cur_mean - ref_mean) / ref_std)
            shifts[col] = float(z)
            drifted += int(z > 2.0)
        summary = {
            "backend": "fallback_zscore",
            "number_of_drifted_columns": drifted,
            "dataset_drift": bool(drifted > 0),
            "per_feature_zscore": shifts,
        }
        (out / "drift_summary.json").write_text(json.dumps(summary, indent=2))
        return summary, None


# ---------------------------------------------------------------------------
# Pipeline 2: parallel training + inverse-RMSE weighted ensemble
# ---------------------------------------------------------------------------
@dataclass
class TrainingPipelineResult:
    per_model_metrics: dict[str, dict[str, float]]
    weights: dict[str, float]
    ensemble_metrics: dict[str, float]
    model_path: str


def run_training_pipeline_local(
    features: pd.DataFrame,
    val_fraction: float = 0.2,
    model_names: tuple[str, ...] = SUPPORTED_MODELS,
    out_dir: str | Path = "data/local_run",
    seed: int = 42,
) -> TrainingPipelineResult:
    """Local equivalent of Pipeline 2 (parallel train branches + fan-in ensemble).

    Runs the branches sequentially here (a single laptop core), but the LOGIC --
    independent per-model fits, then an inverse-RMSE weighted fan-in -- is
    identical to the parallel Vertex DAG.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = feature_columns(features)
    split = int(len(features) * (1 - val_fraction))
    x_train = features[cols].iloc[:split].to_numpy()
    y_train = features["demand"].iloc[:split].to_numpy()
    x_val = features[cols].iloc[split:].to_numpy()
    y_val = features["demand"].iloc[split:].to_numpy()

    # --- "Parallel" branches: one fit per model -----------------------------
    trained = {}
    per_model_metrics = {}
    val_predictions = {}
    for name in model_names:
        result = train_model(name, x_train, y_train, x_val, y_val, seed=seed)
        trained[name] = result.estimator
        per_model_metrics[name] = result.metrics
        val_predictions[name] = result.estimator.predict(x_val) # type: ignore[attr-defined]

    # --- Fan-in: inverse-RMSE weighted ensemble ------------------------------
    rmses = {name: m["rmse"] for name, m in per_model_metrics.items()}
    weights = inverse_error_weights(rmses)
    ens_pred = ensemble_predict(val_predictions, weights)
    ensemble_metrics = {
        "rmse": rmse(y_val, ens_pred),
        "mae": mae(y_val, ens_pred),
        "mape": mape(y_val, ens_pred),
    }

    bundle = {
        "type": "weighted_ensemble",
        "members": [
            {"model_name": n, "weight": weights[n], "estimator": trained[n]}
            for n in model_names
        ],
        "feature_cols": cols,
        "metrics": ensemble_metrics,
        "weights": weights,
    }
    model_path = out / "ensemble_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)

    return TrainingPipelineResult(
        per_model_metrics=per_model_metrics,
        weights=weights,
        ensemble_metrics=ensemble_metrics,
        model_path=str(model_path),
    )


@dataclass
class EndToEndResult:
    data: DataPipelineResult
    training: TrainingPipelineResult


def run_end_to_end_local(
    raw_df: pd.DataFrame,
    split_date: str,
    val_fraction: float = 0.2,
    out_dir: str | Path = "data/local_run",
) -> EndToEndResult:
    """Run Pipeline 1 then Pipeline 2 locally, exactly as production would chain."""
    data_result = run_data_pipeline_local(raw_df, split_date, out_dir=out_dir)
    training_result = run_training_pipeline_local(
        data_result.features, val_fraction=val_fraction, out_dir=out_dir
    )
    return EndToEndResult(data=data_result, training=training_result)


def summarize(result: EndToEndResult) -> dict:
    """Compact, JSON-serializable summary for logging/notebook display."""
    return {
        "drift": {
            "dataset_drift": result.data.drift_summary.get("dataset_drift"),
            "n_drifted_features": result.data.drift_summary.get(
                "number_of_drifted_columns"
            ),
        },
        "per_model_metrics": result.training.per_model_metrics,
        "ensemble_weights": result.training.weights,
        "ensemble_metrics": result.training.ensemble_metrics,
        "model_path": result.training.model_path,
    }
