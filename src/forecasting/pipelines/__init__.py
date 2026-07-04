"""KFP pipeline definitions. Import `PIPELINES` from `deploy_pipeline.py`."""

from forecasting.pipelines.data_pipeline import data_ingest_validation_pipeline
from forecasting.pipelines.training_pipeline import training_ensemble_pipeline

# Registry consumed by deployment/deploy_pipeline.py
PIPELINES = {
    "data": data_ingest_validation_pipeline,
    "training": training_ensemble_pipeline,
}

__all__ = [
    "PIPELINES",
    "data_ingest_validation_pipeline",
    "training_ensemble_pipeline",
]
