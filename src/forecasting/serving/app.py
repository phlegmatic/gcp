"""FastAPI serving app: JSON prediction API + a minimal demo Web UI.

Designed for Cloud Run scale-to-zero:
- The model is loaded LAZILY into a process-level cache on first prediction, so
  `/healthz` returns instantly (fast liveness probe) and idle containers cost $0.
- Endpoints:
    GET  /            -> HTML demo UI (form + Chart.js line chart)
    POST /predict     -> JSON forecast API
    GET  /metadata    -> ensemble metrics + member weights
    GET  /healthz     -> liveness probe (no model load)

The model URI defaults to `SERVING_MODEL_URI` (see config/settings.py); a local
file path or gs:// URI both work, enabling local dev without GCS.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from forecasting.serving import predictor
from forecasting.serving.predictor import LoadedModel

app = FastAPI(
    title="Demand Forecasting API",
    version="0.1.0",
    description="Serverless serving endpoint for the weighted-ensemble model.",
)

# --- process-level model cache (thread-safe lazy singleton) -----------------
_MODEL: Optional[LoadedModel] = None
_MODEL_LOCK = threading.Lock()


def _model_uri() -> str:
    """Resolve the artifact URI, preferring the env var, then Settings."""
    uri = os.getenv("SERVING_MODEL_URI")
    if uri:
        return uri
    # Fall back to Settings only if env is unset; import lazily so importing the
    # app module never forces full cloud config resolution (helps local tests).
    from forecasting.config import get_settings

    settings = get_settings()
    return f"{settings.model_prefix}/ensemble_model.pkl"


def _load_model(uri: str) -> LoadedModel:
    """Load a bundle from a gs:// URI or a local file path."""
    if uri.startswith("gs://"):
        return predictor.load_bundle_from_gcs(uri)
    with open(uri, "rb") as fh:
        return predictor.load_bundle_from_bytes(fh.read())


def get_model() -> LoadedModel:
    """Return the cached model, loading it on first use (double-checked lock)."""
    global _MODEL
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = _load_model(_model_uri())
    return _MODEL


# --- request / response schemas ---------------------------------------------
class HistoryPoint(BaseModel):
    ds: str = Field(..., description="ISO date, e.g. '2024-01-31'.")
    demand: float = Field(..., description="Observed demand for that date.")


class PredictRequest(BaseModel):
    history: list[HistoryPoint] = Field(
        ..., description="Chronological observed demand series."
    )
    horizon: int = Field(1, ge=1, le=90, description="Future daily steps to forecast.")


class ForecastPoint(BaseModel):
    ds: str
    demand: float


class PredictResponse(BaseModel):
    forecast: list[ForecastPoint]
    horizon: int
    model_metrics: dict[str, float]


# --- endpoints ---------------------------------------------------------------
@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe. Intentionally does NOT trigger a model load."""
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    """Return the trained ensemble's metrics and member weights."""
    model = get_model()
    return {
        "type": model.bundle.get("type", "weighted_ensemble"),
        "feature_cols": model.feature_cols,
        "metrics": model.metrics,
        "members": [
            {"model_name": m["model_name"], "weight": float(m["weight"])}
            for m in model.members
        ],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Recursively forecast `horizon` future steps from the supplied history."""
    model = get_model()
    history = pd.DataFrame([p.model_dump() for p in req.history])
    if history.empty:
        raise HTTPException(status_code=422, detail="history must be non-empty.")
    try:
        result = predictor.forecast(model, history, horizon=req.horizon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    forecast_points = [
        ForecastPoint(ds=pd.Timestamp(r.ds).date().isoformat(), demand=float(r.demand))
        for r in result.itertuples(index=False)
    ]
    return PredictResponse(
        forecast=forecast_points,
        horizon=req.horizon,
        model_metrics=model.metrics,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the single-file demo UI."""
    return _INDEX_HTML


# --- minimal single-file UI (form + Chart.js) --------------------------------
_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Demand Forecasting Demo</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 900px;
           color: #1a1a1a; }
    h1 { font-size: 1.4rem; }
    textarea { width: 100%; height: 140px; font-family: monospace; font-size: .85rem; }
    label { display:block; margin: .6rem 0 .2rem; font-weight: 600; }
    button { margin-top: .8rem; padding: .5rem 1rem; font-size: 1rem; cursor: pointer;
             border: 0; border-radius: 6px; background: #1a73e8; color: #fff; }
    button:disabled { opacity: .6; cursor: default; }
    #meta { font-size: .85rem; color: #555; margin-top: 1rem; white-space: pre-wrap; }
    #err { color: #c5221f; margin-top: .5rem; }
  </style>
</head>
<body>
  <h1>Demand Forecasting &mdash; Ensemble Serving Demo</h1>
  <p>Paste a daily <code>ds,demand</code> history (CSV, one row per line), choose a
     horizon, and forecast. Backed by the weighted Ridge + RandomForest ensemble.</p>

  <label for="hist">History (CSV: ds,demand)</label>
  <textarea id="hist">2024-01-01,100
2024-01-02,102
2024-01-03,98
2024-01-04,105
2024-01-05,110</textarea>

  <label for="horizon">Horizon (days)</label>
  <input id="horizon" type="number" min="1" max="90" value="7" />

  <button id="go" onclick="run()">Forecast</button>
  <div id="err"></div>
  <div id="meta"></div>
  <canvas id="chart" height="140"></canvas>

  <script>
    let chart;
    function parseCsv(text) {
      return text.trim().split(/\\n+/).map(line => {
        const [ds, demand] = line.split(",");
        return { ds: ds.trim(), demand: parseFloat(demand) };
      });
    }
    async function run() {
      const btn = document.getElementById("go");
      const err = document.getElementById("err");
      err.textContent = "";
      btn.disabled = true; btn.textContent = "Forecasting...";
      try {
        const history = parseCsv(document.getElementById("hist").value);
        const horizon = parseInt(document.getElementById("horizon").value, 10);
        const res = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ history, horizon }),
        });
        if (!res.ok) {
          const e = await res.json();
          throw new Error(e.detail || res.statusText);
        }
        const data = await res.json();
        render(history, data.forecast);
        document.getElementById("meta").textContent =
          "Model metrics: " + JSON.stringify(data.model_metrics);
      } catch (e) {
        err.textContent = "Error: " + e.message;
      } finally {
        btn.disabled = false; btn.textContent = "Forecast";
      }
    }
    function render(history, forecast) {
      const labels = history.map(p => p.ds).concat(forecast.map(p => p.ds));
      const hist = history.map(p => p.demand).concat(forecast.map(() => null));
      const fc = history.map(() => null);
      if (history.length) fc[history.length - 1] = history[history.length - 1].demand;
      forecast.forEach(p => fc.push(p.demand));
      const cfg = {
        type: "line",
        data: { labels, datasets: [
          { label: "History", data: hist, borderColor: "#1a73e8", tension: .2 },
          { label: "Forecast", data: fc, borderColor: "#e37400",
            borderDash: [6, 4], tension: .2 },
        ]},
        options: { responsive: true, interaction: { intersect: false } },
      };
      if (chart) chart.destroy();
      chart = new Chart(document.getElementById("chart"), cfg);
    }
  </script>
</body>
</html>
"""
