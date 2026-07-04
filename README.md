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
│   └── local_runner.py    # Cloud-free E2E runner (same logic as pipelines)
├── deployment/
│   ├── deploy_pipeline.py # Compile KFP -> YAML -> submit to Vertex AI
│   └── compiled/          # Generated pipeline specs (gitignored)
├── data/raw/              # Locally generated datasets (gitignored)
├── dbt/                   # dbt-bigquery project (staging + marts)
├── notebooks/
│   └── local_end_to_end.ipynb  # Prototype the whole project locally
├── scripts/               # generate_data.py + e2-micro cron trigger
├── tests/{unit,integration}
├── .github/workflows/     # ci.yml (compile+test) + deploy.yml (gated submit)
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

## Development contract

Golden rule: **ML logic lives in `models/` (pure, testable); `components/` are
thin KFP wrappers; `local_runner.py` mirrors the pipeline DAGs.** See
`AI_AGENT_GUIDELINES.md` for the full contract before making changes.

---

## License

Apache-2.0. See `LICENSE`.
