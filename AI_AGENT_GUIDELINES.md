# AI Agent Guidelines

> **READ THIS FIRST.** This file is the binding contract for any AI coding agent
> (Cursor, Copilot, Claude, etc.) or human contributor working in this repo. It
> defines the architecture, the **non-negotiable GCP free-tier constraints**, and
> exactly how to add new pipeline components. Violating these rules can trigger
> real billing or break serverless execution.

A symlink `.cursorrules -> AI_AGENT_GUIDELINES.md` exposes these rules to Cursor.

---

## 1. What this project is

A time-series **demand forecasting** MLOps system that:
- Authors pipelines with the **`kfp` (Kubeflow Pipelines) SDK v2**.
- **Compiles pipelines to YAML** and runs them **serverlessly on Vertex AI
  Pipelines**. There is **NO GKE cluster** — never add one.
- Stays inside **GCP Free Tier / micro-billing**: BigQuery Sandbox, GCS
  always-free, Vertex AI serverless, and an Always-Free `e2-micro` VM (or GitHub
  Actions) purely to *trigger* runs.

---

## 2. Hard constraints (NEVER violate)

1. **No GKE, no Dataproc, no always-on services.** Orchestration is Vertex AI
   Pipelines (serverless) only. Triggers are `e2-micro` VM or GitHub Actions.
2. **BigQuery = Sandbox.** Never instruct enabling a paid billing account.
   Every BigQuery query MUST pass `maximum_bytes_billed` using
   `settings.bq_max_bytes_billed`. Never remove or raise this cap without an
   explicit human cost review.
3. **Smallest machines.** KFP tasks pin CPU/memory via
   `.set_cpu_limit("1").set_memory_limit("2G")`. Do not request GPUs, accelerators,
   or machine types larger than `e2-standard-2`.
4. **Slim component images.** Base image is `python:3.10-slim`. Each `@component`
   declares the **minimal** `packages_to_install`. Never add heavy libs (torch,
   tensorflow, xgboost) — the free-tier stack is scikit-learn only.
5. **No secrets in code.** All cloud config comes from env vars via
   `forecasting.config.get_settings()`. Never hard-code project ids, bucket names,
   or credentials. Never commit `.env` or JSON keys (gitleaks + pre-commit block
   this).
6. **GCS is the only artifact store.** Reports, models, and intermediate data go to
   the bucket in `.env`. Do not introduce new storage backends.
7. **Pin dependency versions.** Match versions already in `pyproject.toml` and
   the committed `uv.lock`. Manage deps with **`uv`** (`uv add`, `uv add --dev`,
   `uv lock`) — never hand-edit `uv.lock`, never use `pip install` in workflows.
   The pinned interpreter is in `.python-version`.

---

## 3. Architecture & where code goes

| Layer | Path | Rule |
| --- | --- | --- |
| Config | `src/forecasting/config/` | The ONLY place env vars are read. Add new settings here, typed + validated. |
| Data gen | `src/forecasting/data/` | Synthetic dataset generator (base + drift). Output schema MUST match the dbt raw source (`sale_date`, `units_sold`). |
| Utils | `src/forecasting/utils/` | Small, cloud-light helpers (GCS, metrics). GCS is imported lazily so pure consumers don't need the cloud SDK. Must be unit-testable. |
| **ML logic** | `src/forecasting/models/` | **PURE** functions/classes. NO `kfp` imports, NO cloud calls. All real ML lives here so it is unit-tested without GCP. |
| Local runner | `src/forecasting/local_runner.py` | Cloud-free mirror of the two pipeline DAGs (reuses `models/`). Keep it in lockstep with `pipelines/` so notebook == production. |
| KFP components | `src/forecasting/components/` | **Thin** `@component` wrappers that call into `models/`. Import heavy libs *inside* the function body. |
| Pipelines | `src/forecasting/pipelines/` | `@pipeline` DAGs wiring components. Register new pipelines in `pipelines/__init__.py::PIPELINES`. |
| Deployment | `deployment/deploy_pipeline.py` | Compile + submit. Add per-pipeline params in `_pipeline_parameters`. |
| Notebook | `notebooks/local_end_to_end.ipynb` | Prototyping surface. Must use `src` code only (never fork logic into the notebook). |
| dbt | `dbt/` | Transforms into BigQuery. Mart table `demand_features` is the training/drift input. |

**Golden rule:** if it is ML, it goes in `models/` and is unit-tested. Components
and the local runner are dumb glue. **If you change a pipeline's logic, update
`local_runner.py` to match** so notebook prototyping stays faithful to production.

---

## 4. How to write a NEW KFP component

Follow this template exactly:

```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Output

PY_IMAGE = "python:3.11-slim"

@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=["pandas==2.2.2", "pyarrow==16.1.0"],  # minimal + pinned
)
def my_new_step(
    input_data: Input[Dataset],
    some_param: str,
    output_data: Output[Dataset],
) -> str:
    """One-line summary. Explain inputs/outputs and any GCS/BQ side effects."""
    # 1) Import heavy libs INSIDE the function (keeps compile env light).
    import pandas as pd

    # 2) For BigQuery, ALWAYS cap bytes billed:
    #    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=...)
    df = pd.read_parquet(input_data.path)
    # ... logic (prefer calling into forecasting.models.* for real ML) ...
    df.to_parquet(output_data.path)
    return "gs://..."  # return URIs/paths, never in-memory objects
```

Rules:
- Exchange data ONLY via KFP `Input`/`Output` artifacts or JSON-serializable
  parameters. Never use module globals or pickle across steps except serialized
  model artifacts.
- Log metrics with `Output[Metrics].log_metric(...)` so they show in Vertex AI.
- Keep `packages_to_install` minimal and version-pinned.

### Wiring it into a pipeline

```python
@dsl.pipeline(name="...", description="...")
def my_pipeline(project_id: str, ...):
    task = my_new_step(input_data=prev.outputs["out"], some_param="x")
    task.set_display_name("my-new-step")
    task.set_cpu_limit("1").set_memory_limit("2G")   # REQUIRED cost guard
```

Then register in `src/forecasting/pipelines/__init__.py`:
```python
PIPELINES = {"data": ..., "training": ..., "my": my_pipeline}
```
and add its runtime params to `deployment/deploy_pipeline.py::_pipeline_parameters`.

### Parallel + fan-in pattern
Two tasks with no data dependency on each other run in parallel automatically.
A downstream task that consumes both their outputs is the fan-in (it implicitly
waits for both). See `training_pipeline.py` for the canonical example.

---

## 5. Testing contract

- **Every function in `models/` and `utils/` must have a unit test** in
  `tests/unit/` (marker `@pytest.mark.unit`, no cloud).
- **New pipelines must compile** — add/extend the parametrized compile test in
  `tests/integration/test_pipeline_compile.py` (marker `@pytest.mark.integration`).
- Run before committing: `uv run pytest -m unit && uv run pytest -m integration && uv run pre-commit run --all-files`.
- Do NOT write tests that hit live GCP by default; if you must, gate them behind
  `@pytest.mark.integration` and document the cost.

---

## 6. Code quality

- Formatting: **Black** (line length 88) + **isort** (`profile=black`).
- Linting: **flake8** (+ bugbear). Components may need `# noqa` only for the
  documented in-function-import pattern (E402 already ignored for `components/`).
- Types: **mypy** on `src/` (`--ignore-missing-imports`).
- All enforced by `.pre-commit-config.yaml`. Run `uv run pre-commit run --all-files`.
  Never disable hooks
  (`--no-verify`) to get around failures — fix the code.

---

## 7. Deployment & scheduling — what you may generate

- Compile only (safe, free): `deploy_pipeline.py --pipeline <k> --compile-only`.
- Submit a run (micro-billing): `--submit`.
- Recurring: `--schedule "<cron>"` (Vertex-native, preferred), or the `e2-micro`
  cron in `scripts/`, or the gated `deploy.yml` GitHub Action.
- Never generate code that force-pushes, deletes buckets/datasets, or removes the
  byte-billed cap.

---

## 8. Quick checklist before you propose a change

- [ ] ML logic in `models/`, component is a thin wrapper.
- [ ] `packages_to_install` minimal + version-pinned; base image `python:3.11-slim`.
- [ ] BigQuery calls use `settings.bq_max_bytes_billed`.
- [ ] Task has `set_cpu_limit`/`set_memory_limit`.
- [ ] No secrets/hard-coded project ids; config via `get_settings()`.
- [ ] Unit test added; pipeline still compiles; `uv run pre-commit run --all-files` passes.
- [ ] No GKE / GPU / paid-tier resource introduced.
