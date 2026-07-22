"""Integration tests for the FastAPI serving app.

Marked `unit` because they use FastAPI's in-process TestClient and a locally
pickled model -- no real GCP resource is touched. They are grouped under
`integration/` because they exercise the HTTP layer end-to-end.
"""

from __future__ import annotations

import pickle

import pytest

pytestmark = pytest.mark.unit

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(tmp_path, ensemble_bundle, monkeypatch):
    """A TestClient wired to a locally-pickled model via SERVING_MODEL_URI."""
    model_path = tmp_path / "ensemble_model.pkl"
    model_path.write_bytes(pickle.dumps(ensemble_bundle))
    monkeypatch.setenv("SERVING_MODEL_URI", str(model_path))

    # Import after env is set; reset the module-level cache for isolation.
    from forecasting.serving import app as app_module

    app_module._MODEL = None
    return TestClient(app_module.app)


def _history_payload(synthetic_series, n=60):
    tail = synthetic_series.tail(n)
    return [
        {"ds": ts.date().isoformat(), "demand": float(v)}
        for ts, v in zip(tail["ds"], tail["demand"])
    ]


def test_healthz_does_not_require_model(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Demand Forecasting" in resp.text
    assert "<canvas" in resp.text


def test_metadata_returns_weights_and_metrics(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "weighted_ensemble"
    names = {m["model_name"] for m in body["members"]}
    assert names == {"ridge", "random_forest"}
    assert "rmse" in body["metrics"]


def test_predict_returns_forecast(client, synthetic_series):
    payload = {"history": _history_payload(synthetic_series), "horizon": 5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == 5
    assert len(body["forecast"]) == 5
    for point in body["forecast"]:
        assert "ds" in point and "demand" in point


def test_predict_rejects_empty_history(client):
    resp = client.post("/predict", json={"history": [], "horizon": 1})
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_horizon(client, synthetic_series):
    payload = {"history": _history_payload(synthetic_series), "horizon": 999}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
