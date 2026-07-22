"""Thin, dependency-light helpers for reading/writing GCS objects.

Kept tiny on purpose: KFP components install only `google-cloud-storage` when
they need these, avoiding heavy images that increase serverless cold-start
time and cost.
"""

from __future__ import annotations

import json
from typing import Any

from google.cloud import storage


def _split_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri!r}")
    without_scheme = uri[len("gs://") :]
    bucket, _, blob = without_scheme.partition("/")
    if not blob:
        raise ValueError(f"GCS URI must include an object path: {uri!r}")
    return bucket, blob


def write_bytes_gcs(
    uri: str, data: bytes, content_type: str = "application/octet-stream"
) -> str:
    """Upload raw bytes to a gs:// URI and return the URI."""
    bucket_name, blob_name = _split_gcs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return uri


def read_bytes_gcs(uri: str) -> bytes:
    """Download the raw bytes of an object at a gs:// URI."""
    bucket_name, blob_name = _split_gcs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.download_as_bytes()


def write_json_gcs(uri: str, obj: Any) -> str:
    """Serialize `obj` to JSON and upload to a gs:// URI."""
    payload = json.dumps(obj, indent=2, default=str).encode("utf-8")
    return write_bytes_gcs(uri, payload, content_type="application/json")


def read_json_gcs(uri: str) -> Any:
    """Download and JSON-parse an object at a gs:// URI."""
    bucket_name, blob_name = _split_gcs_uri(uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return json.loads(blob.download_as_bytes())
