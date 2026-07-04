"""Centralized, environment-driven configuration.

Everything that could incur cost or that changes between environments MUST be
read from here (never hard-coded in components). This keeps the free-tier
guardrails in one auditable place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

# Load a local .env if present (no-op in CI where vars are injected directly).
load_dotenv()


def _require(key: str, default: str | None = None) -> str:
    value = os.getenv(key, default)
    if value is None:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and populate it."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable, validated settings resolved from the environment."""

    project_id: str = field(default_factory=lambda: _require("GCP_PROJECT_ID"))
    region: str = field(default_factory=lambda: _require("GCP_REGION", "us-central1"))

    gcs_bucket: str = field(default_factory=lambda: _require("GCS_BUCKET"))
    pipeline_root: str = field(default_factory=lambda: _require("GCS_PIPELINE_ROOT"))
    drift_prefix: str = field(default_factory=lambda: _require("GCS_DRIFT_PREFIX"))
    model_prefix: str = field(default_factory=lambda: _require("GCS_MODEL_PREFIX"))

    bq_dataset_raw: str = field(
        default_factory=lambda: _require("BQ_DATASET_RAW", "demand_raw")
    )
    bq_dataset_mart: str = field(
        default_factory=lambda: _require("BQ_DATASET_MART", "demand_mart")
    )
    bq_location: str = field(default_factory=lambda: _require("BQ_LOCATION", "US"))

    experiment_name: str = field(
        default_factory=lambda: _require("VERTEX_EXPERIMENT_NAME", "demand-forecasting")
    )
    vertex_sa_email: str = field(
        default_factory=lambda: os.getenv("VERTEX_SA_EMAIL", "")
    )
    tracking_backend: str = field(
        default_factory=lambda: os.getenv("TRACKING_BACKEND", "vertex")
    )
    mlflow_artifact_root: str = field(
        default_factory=lambda: os.getenv("MLFLOW_GCS_ARTIFACT_ROOT", "")
    )

    # ---- Free-tier guardrails (do NOT raise these without cost review) ------
    #: Vertex AI Pipelines steps default to e2-standard machines; we force the
    #: smallest allowed CPU machine to minimize vCPU-hour cost.
    default_machine_type: str = "e2-standard-2"
    #: Hard ceiling on BigQuery bytes billed per query (1 GB) to stay well under
    #: the 1 TB/month Sandbox free limit and abort runaway scans.
    bq_max_bytes_billed: int = 1_000_000_000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance."""
    return Settings()
