# GCP Free-Tier Demand Forecasting (Serverless MLOps)

A production-grade, highly maintainable **time-series demand forecasting** system
engineered to run entirely within **Google Cloud Platform Free Tier / micro-billing**
limits. Pipelines are authored with the **Kubeflow Pipelines (`kfp`) SDK**, compiled
to YAML, and executed **serverlessly on Vertex AI Pipelines** — **no GKE cluster is
ever provisioned**.

Dependency management and task running use **[`uv`](https://docs.astral.sh/uv/)**
(no `pip`, no `make`, no manual virtualenvs).

---

## Why this architecture is (near) free

| Concern | Choice | Free-tier lever |
| --- | --- | --- |
| Orchestration | Vertex AI Pipelines (serverless) | Pay only per-step vCPU/GB-seconds; no idle cluster |
| Compute for triggers | GitHub Actions **or** Always-Free `e2-micro` VM | 2000 CI min/mo; 1 e2-micro always free |
| Data warehouse | BigQuery **Sandbox** | 10 GB storage + 1 TB query/mo, **no billing account** |
| Transforms | `dbt-core` + `dbt-bigquery` | Runs inside a slim KFP step |
| Artifacts / reports / models | Google Cloud Storage | 5 GB-months always free (us regions) |
| Experiment tracking | Vertex AI Experiments *or* file-based MLflow on GCS | Metadata store is free |
| Drift detection | `evidently` (fallback: `ydata-profiling`) | Runs in a slim step, report saved to GCS |
| Model serving | Cloud Run (scale-to-zero FastAPI) | Bills $0 while idle, autoscales on demand; no 24/7 Vertex Endpoint node |

Cost guardrails are centralized in `src/forecasting/config/settings.py`
(`bq_max_bytes_billed`, `default_machine_type`) and enforced in every component.

---

# Getting started — from a fresh instance to a successful notebook run

Everything below runs **100% locally with no GCP account, no cloud cost**. It
uses the exact same `src/forecasting` code that the deployed Vertex AI pipelines
use, so a successful local run proves the pipelines are wired correctly.

> **Assumption:** you have already installed a base **Python 3.10 or 3.11**
> (needed only to bootstrap `uv`; `uv` then manages the project's own pinned
> Python). Everything else is installed for you.

### Step 1 — Install `uv`

Linux / macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload PATH so `uv` is available
uv --version
```

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

### Step 2 — Get the code

```bash
git clone <your-repo-url> gcp-demand-forecasting
cd gcp-demand-forecasting
```

### Step 3 — Create the environment (one command)

This reads `.python-version` (pinned to 3.11), fetches that interpreter if
needed, creates `.venv`, and installs **all** dependencies from the committed
`uv.lock` — the runtime deps, the `dev` group, plus the `data` (Evidently/dbt)
and `notebook` (JupyterLab/matplotlib) extras:

```bash
uv sync --extra data --extra notebook
```

That's it — no `pip`, no manual `venv activate`. Every command below uses
`uv run`, which executes inside this environment automatically.

### Step 4 — Verify the install (fast, no cloud)

```bash
uv run pytest -m unit
```
You should see all unit tests pass (metrics, feature engineering, training,
ensembling, and the data generator).

### Step 5 — Generate a synthetic dataset (base + drift)

The generator produces data whose schema is identical to the production
BigQuery raw source, including an **injectable drifted regime** to exercise the
drift step:

```bash
uv run python scripts/generate_data.py --mode drift
```
This writes `data/raw/sales_base.csv`, `sales_drift.csv`, and `sales.csv` (the
full base+drift union the pipeline ingests). Tune the drift, e.g.:
```bash
uv run python scripts/generate_data.py --mode drift --level-scale 1.6 --noise-scale 3 --parquet
```

### Step 6 — Run the whole project end-to-end (still no cloud)

`src/forecasting/local_runner.py` runs the **same two DAGs** as production using
local files:

```bash
uv run python -c "from forecasting.data import SeriesConfig, DriftConfig, generate_base_and_drift, to_pipeline_frame; from forecasting.local_runner import run_end_to_end_local, summarize; import json; b,d,f=generate_base_and_drift(SeriesConfig(n_days=365), DriftConfig(level_scale=1.4)); r=run_end_to_end_local(to_pipeline_frame(f), split_date=str(d['sale_date'].iloc[0])); print(json.dumps(summarize(r), indent=2, default=str))"
```
You'll see the drift verdict, per-model metrics (Ridge + Random Forest),
inverse-RMSE ensemble weights, and the ensemble metrics.

### Step 7 — Launch the notebook and run it end-to-end (the goal)

```bash
uv run jupyter lab notebooks/local_end_to_end.ipynb
```
In JupyterLab choose **Run ▸ Run All Cells**. The notebook:
1. generates a base + drifted dataset,
2. plots the two regimes,
3. runs **Pipeline 1** (build features → reference/current split → Evidently drift report),
4. runs **Pipeline 2** (parallel Ridge + Random Forest → inverse-RMSE weighted ensemble),
5. loads the serialized ensemble and produces predictions.

A clean run prints `dataset_drift: True`, per-model + ensemble metrics, and
writes `data/local_run/ensemble_model.pkl` and `drift_report.html`.

> **Headless check (CI / no browser):** you can execute the whole notebook
> non-interactively and fail on any error:
> ```bash
> uv run jupyter nbconvert --to notebook --execute \
>     --ExecutePreprocessor.timeout=300 \
>     --output executed.ipynb notebooks/local_end_to_end.ipynb
> ```

You have now run the identical logic that deploys to Vertex AI — locally, for
free. Continue to the sections below when you're ready to run it serverlessly.

---

## Common tasks (uv replaces make)

| Task | Command |
| --- | --- |
| Create/refresh env (all extras) | `uv sync --extra data --extra notebook --extra tracking` |
| Add a dependency | `uv add <pkg>` (or `uv add --dev <pkg>` for tooling) |
| Unit tests (no cloud) | `uv run pytest -m unit` |
| Integration tests (compiles pipelines + local E2E) | `uv run pytest -m integration` |
| Full test suite + coverage | `uv run pytest` |
| Format | `uv run black src tests deployment && uv run isort src tests deployment` |
| Lint (black, isort, flake8, mypy, gitleaks) | `uv run pre-commit run --all-files` |
| Generate base dataset | `uv run python scripts/generate_data.py --mode base` |
| Generate drifted dataset | `uv run python scripts/generate_data.py --mode drift` |
| Launch notebook | `uv run jupyter lab notebooks/local_end_to_end.ipynb` |
| Compile pipelines (no cost) | `uv run python deployment/deploy_pipeline.py --pipeline data --compile-only` |
| Submit to Vertex AI | `uv run python deployment/deploy_pipeline.py --pipeline training --submit` |
| Serve the model locally (FastAPI + demo UI) | `SERVING_MODEL_URI=./ensemble_model.pkl uv run uvicorn forecasting.serving.app:app --reload` |
| Deploy serving to Cloud Run (dry-run) | `uv run python deployment/serving/deploy_service.py --dry-run` |
| Deploy serving to Cloud Run | `uv run python deployment/serving/deploy_service.py --deploy` |

Install pre-commit's git hook once: `uv run pre-commit install`.

---

## Repository layout

```
.
├── src/forecasting/
│   ├── config/            # Env-driven Settings + free-tier guardrails
│   ├── data/              # Synthetic dataset generator (base + drift)
│   ├── utils/             # GCS + metrics helpers (cloud-light, unit-tested)
│   ├── models/            # PURE sklearn logic: features, train, ensemble
│   ├── components/        # KFP @component definitions (thin wrappers)
│   ├── pipelines/         # KFP @pipeline DAGs (data + training)
│   ├── serving/           # Cloud Run inference: pure predictor + thin FastAPI app
│   └── local_runner.py    # Cloud-free E2E runner (same logic as pipelines)
├── deployment/
│   ├── deploy_pipeline.py # Compile KFP -> YAML -> submit to Vertex AI
│   ├── serving/           # Cloud Run inference: Dockerfile, cloudbuild, deploy_service.py
│   └── compiled/          # Generated pipeline specs (gitignored)
├── data/raw/              # Locally generated datasets (gitignored)
├── dbt/                   # dbt-bigquery project (staging + marts)
├── notebooks/
│   └── local_end_to_end.ipynb  # Prototype the whole project locally
├── scripts/               # generate_data.py + e2-micro cron trigger
├── tests/{unit,integration}
├── .github/workflows/     # ci.yml (compile+test) + deploy.yml (gated submit) + deploy-serving.yml (gated Cloud Run)
├── AI_AGENT_GUIDELINES.md # READ THIS before AI-assisted changes
├── .pre-commit-config.yaml
├── .python-version        # pinned interpreter for uv
├── uv.lock                # fully pinned, cross-platform lockfile (committed)
└── pyproject.toml
```

---

## The two pipelines

### Pipeline 1 — Data Ingest, Validation & Drift
`src/forecasting/pipelines/data_pipeline.py`

```
run_dbt_transform  ->  extract_reference_and_current  ->  detect_drift
   (dbt build)          (BQ split, byte-capped)          (Evidently -> GCS)
```
Outputs an HTML + JSON drift report to `GCS_DRIFT_PREFIX` and logs
`dataset_drift` / `n_drifted_features` as Vertex AI Pipeline metrics.

### Pipeline 2 — Parallel Training & Ensembling
`src/forecasting/pipelines/training_pipeline.py`

```
                 +--> train_single_model(ridge) ---------+
 load_data ------+                                         +--> build_ensemble --> GCS
                 +--> train_single_model(random_forest) --+
```
The two training branches run **in parallel** on Vertex AI. `build_ensemble` is a
**fan-in** step that waits for both, computes **inverse-RMSE weights**, evaluates
the weighted ensemble, and serializes the final model bundle to `GCS_MODEL_PREFIX`.

The local notebook/runner mirrors these exact DAGs — see
`src/forecasting/local_runner.py`.

---

# Deploying to Vertex AI (serverless, when you're ready)

The steps above never touch GCP. To run the same code serverlessly:

### Prerequisites

1. A GCP project (Free Tier eligible). **Do not** enable a paid billing account
   if you want to stay in BigQuery Sandbox.
2. `gcloud` authenticated: `gcloud auth application-default login`.
3. Enable APIs:
   ```bash
   gcloud services enable aiplatform.googleapis.com bigquery.googleapis.com storage.googleapis.com
   ```

### Configure + provision

```bash
cp .env.example .env                       # edit project id, bucket, datasets
gsutil mb -l us-central1 gs://$GCS_BUCKET  # one-time bucket
# Load raw data into BigQuery Sandbox dataset demand_raw.sales
#   (columns: sale_date DATE, units_sold NUMERIC). Hint:
uv run python scripts/generate_data.py --mode drift --bq-load-hint
```

### Compile (free) then submit (micro-billing)

```bash
# Compile to YAML only — zero cloud cost, good for CI:
uv run python deployment/deploy_pipeline.py --pipeline data --compile-only

# Submit a serverless run:
uv run python deployment/deploy_pipeline.py --pipeline data --submit      # Pipeline 1
uv run python deployment/deploy_pipeline.py --pipeline training --submit  # Pipeline 2
```

### Experiment tracking

Set `TRACKING_BACKEND` in `.env`:
- `vertex` — KFP `Metrics` artifacts appear automatically in the Vertex AI
  Experiments UI (free metadata store).
- `mlflow_gcs` — point MLflow at `MLFLOW_GCS_ARTIFACT_ROOT` (a GCS path) for a
  serverless, file-based tracking store with no server to run.

### Scheduling (no always-on infra)

**Option A — Vertex AI native schedule (recommended, zero VM):**
```bash
uv run python deployment/deploy_pipeline.py --pipeline data --schedule "0 6 * * 1"
```

**Option B — Always-Free e2-micro VM + cron:** see
`scripts/trigger_from_e2_micro.sh` (installs `uv`, then only *submits*; all ML
runs serverless).

**Option C — GitHub Actions:** trigger `.github/workflows/deploy.yml` manually
(`workflow_dispatch`) with `pipeline` and `mode` inputs. Uses Workload Identity
Federation (no JSON keys). CI (`ci.yml`) lints + unit-tests + compiles on every
push, all via `uv`.

---

# Serving the model on Cloud Run (scale-to-zero)

Once Pipeline 2 has written the ensemble bundle to `GCS_MODEL_PREFIX`
(`ensemble_model.pkl`), you can expose it as a REST API + demo UI. We use
**Cloud Run**, not a Vertex AI Endpoint: an Endpoint bills a node **24/7** even
at zero traffic, whereas Cloud Run scales to **zero** ($0 while idle), autoscales
on demand, and gives free HTTPS. The service is loosely coupled to training — it
only *reads* the model artifact from GCS.

The container:
- pins runtime deps to the **exact training versions** (avoids pickle skew),
- loads the model **lazily** (fast cold starts, `/healthz` needs no model),
- runs as a **least-privilege** service account (read-only on the models bucket).

See `PROJECT_OVERVIEW.md` (Model Serving) and
`deployment/serving/deploy_service.py` for the full trade-offs.

### Serve locally (no cloud)

```bash
uv sync --extra serving
# Point at a local bundle produced by the notebook/local runner:
SERVING_MODEL_URI=./data/local_run/ensemble_model.pkl \
  uv run uvicorn forecasting.serving.app:app --reload
# Open http://127.0.0.1:8000  (demo UI) or POST /predict {"horizon": 14}
```

Endpoints: `GET /healthz` (liveness), `GET /metadata` (metrics + member
weights), `POST /predict` (recursive forecast, horizon 1–90), `GET /` (demo UI).

### Deploy to Cloud Run

**Option A — CLI (`deploy_service.py`):**
```bash
# One-time infra (enable APIs, Artifact Registry repo, least-privilege SA):
# see the header of deployment/serving/deploy_service.py for the exact commands.

# Print the gcloud commands without running anything:
uv run python deployment/serving/deploy_service.py --dry-run

# Build + push the image only:
uv run python deployment/serving/deploy_service.py --build-only

# Build image (Cloud Build) + deploy the scale-to-zero service:
uv run python deployment/serving/deploy_service.py --deploy
```

**Option B — GitHub Actions (gated):** trigger
`.github/workflows/deploy-serving.yml` manually (`workflow_dispatch`) with a
`mode` input. Uses Workload Identity Federation (no JSON keys), same as
`deploy.yml`:
- `bootstrap` — one-time, idempotent: enable serving APIs, create the
  Artifact Registry `serving` repo, and the least-privilege `run-inference`
  service account with read-only access to the models bucket.
- `build-only` — build + push the image via Cloud Build.
- `deploy` — build + push, then deploy the Cloud Run service.

After deploy, fetch the URL:
```bash
gcloud run services describe demand-forecasting \
  --region=us-central1 --format='value(status.url)'
```

Cost guardrails (in `deploy_service.py`): `--min-instances=0` (scale to zero),
`--max-instances=2`, `--concurrency=80`, `--cpu=1`, `--memory=512Mi`.

---

## Development contract

Golden rule: **ML logic lives in `models/` (pure, testable); `components/` are
thin KFP wrappers; `local_runner.py` mirrors the pipeline DAGs.** See
`AI_AGENT_GUIDELINES.md` for the full contract before making changes.

---

## License

Apache-2.0. See `LICENSE`.
