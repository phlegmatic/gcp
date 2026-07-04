"""Shared utilities (GCS, metrics, tracking). Import from here, do not duplicate.

Note: GCS helpers require `google-cloud-storage`. They are imported lazily so
that pure-Python consumers (metrics, local runner, notebook) do NOT need the
cloud SDK installed. Access them via `forecasting.utils.gcs` or the lazy
attributes below.
"""

from forecasting.utils.metrics import mae, mape, rmse

__all__ = [
    "read_json_gcs",
    "write_bytes_gcs",
    "write_json_gcs",
    "mae",
    "mape",
    "rmse",
]


def __getattr__(name: str):
    # Lazy re-export of GCS helpers so importing metrics doesn't require the
    # google-cloud-storage dependency (keeps local/notebook envs lightweight).
    if name in {"read_json_gcs", "write_bytes_gcs", "write_json_gcs"}:
        from forecasting.utils import gcs

        return getattr(gcs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
