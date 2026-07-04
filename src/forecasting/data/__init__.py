"""Data generation subpackage (synthetic demand series with drift)."""

from forecasting.data.generator import (
    DriftConfig,
    SeriesConfig,
    generate_base_and_drift,
    generate_series,
    to_pipeline_frame,
)

__all__ = [
    "DriftConfig",
    "SeriesConfig",
    "generate_base_and_drift",
    "generate_series",
    "to_pipeline_frame",
]
