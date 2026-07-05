"""Pipeline 1 components: dbt transform, data extract, drift detection."""

import json
from pathlib import Path

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output

# Base image for lightweight python steps. Slim keeps cold-start + egress low.
PY_IMAGE = "python:3.11-slim"

# ---------------------------------------------------------------------------
# Embed the dbt project at COMPILE time.
#
# KFP components run on a stock slim image that does NOT contain the repo's
# `dbt/` directory, so `dbt build --project-dir /app/dbt` used to fail with
# "Path '/app/dbt' does not exist". Instead of building a custom container, we
# read every file under `dbt/` here (when the pipeline is compiled) and embed
# their contents. At runtime the component rehydrates the project into a temp
# dir and runs dbt against it. This keeps `dbt/` as the single source of truth
# (no hardcoded SQL) and stays within the no-container-build, free-tier design.
#
# IMPORTANT: `_DBT_PROJECT_FILES` is consumed by the *pipeline* function (see
# pipelines/data_pipeline.py), which threads it into the component as a runtime
# parameter value. It must NOT be used as a component parameter default: KFP
# serializes only the component's function body, so a module-level name
# referenced there would raise `NameError` on the remote worker.
# ---------------------------------------------------------------------------
_DBT_DIR = Path(__file__).resolve().parents[3] / "dbt"


def _load_dbt_project_files() -> str:
    """Read the dbt project tree into a JSON {relative_path: contents} map.

    Skips generated/vendored dirs and empty `.gitkeep` placeholders. Returns a
    JSON string so it can be passed as a serializable KFP parameter default.
    """
    files: dict[str, str] = {}
    if not _DBT_DIR.is_dir():
        return json.dumps(files)

    skip_dirs = {"target", "dbt_packages", "logs", "__pycache__"}
    for path in sorted(_DBT_DIR.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.relative_to(_DBT_DIR).parts):
            continue
        if path.name == ".gitkeep":
            continue
        rel = path.relative_to(_DBT_DIR).as_posix()
        files[rel] = path.read_text(encoding="utf-8")
    return json.dumps(files)


# Computed once at import/compile time and baked into the compiled pipeline spec.
_DBT_PROJECT_FILES = _load_dbt_project_files()


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=["dbt-bigquery==1.8.2"],
)
def run_dbt_transform(
    project_id: str,
    bq_dataset_mart: str,
    bq_location: str,
    dbt_project_files_json: str,
    dbt_target: str = "prod",
) -> str:
    """Run `dbt build` against BigQuery to materialize the mart table.

    The dbt project files are embedded (as a JSON map) at compile time and
    rehydrated to a temp directory here, so no custom container image or mount
    is required. dbt reads its BigQuery profile from the env vars set below.

    Returns the fully-qualified mart table id.
    """
    import json as _json
    import os
    import subprocess
    import tempfile

    os.environ["DBT_PROJECT_ID"] = project_id
    os.environ["DBT_DATASET"] = bq_dataset_mart
    os.environ["DBT_LOCATION"] = bq_location

    files = _json.loads(dbt_project_files_json)
    if not files:
        raise RuntimeError(
            "No embedded dbt project files were found. The pipeline was likely "
            "compiled without the `dbt/` directory present."
        )

    project_dir = tempfile.mkdtemp(prefix="dbt_project_")
    for rel_path, contents in files.items():
        dest = os.path.join(project_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as handle:
            handle.write(contents)

    # dbt looks for profiles.yml under --profiles-dir; the repo keeps it inside
    # the project dir, so point both at the rehydrated project directory.
    subprocess.run(
        [
            "dbt",
            "build",
            "--target",
            dbt_target,
            "--project-dir",
            project_dir,
            "--profiles-dir",
            project_dir,
        ],
        check=True,
    )
    return f"{project_id}.{bq_dataset_mart}.demand_features"


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery==3.25.0",
        "pandas==2.2.2",
        "pyarrow==16.1.0",
        "db-dtypes==1.2.0",
    ],
)
def extract_reference_and_current(
    project_id: str,
    mart_table: str,
    bq_max_bytes_billed: int,
    reference_out: Output[Dataset],
    current_out: Output[Dataset],
    split_date: str,
) -> None:
    """Split the mart into a reference (historical) and current window for drift.

    Uses a hard `maximum_bytes_billed` cap so a bad query can never blow the
    1 TB/month Sandbox limit.
    """
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=bq_max_bytes_billed)

    ref_sql = f"SELECT * FROM `{mart_table}` WHERE ds < '{split_date}'"
    cur_sql = f"SELECT * FROM `{mart_table}` WHERE ds >= '{split_date}'"

    ref_df = client.query(ref_sql, job_config=job_config).to_dataframe()
    cur_df = client.query(cur_sql, job_config=job_config).to_dataframe()

    # BigQuery returns DATE columns as the `db-dtypes` extension type
    # ("dbdate"). That extension dtype is only understood where `db-dtypes` is
    # installed; the downstream drift step does not install it, so writing it
    # into parquet would fail there with "data type 'dbdate' not understood".
    # Normalize the date column to a plain pandas datetime64 so the parquet is
    # portable across steps.
    if "ds" in ref_df.columns:
        ref_df["ds"] = pd.to_datetime(ref_df["ds"])
    if "ds" in cur_df.columns:
        cur_df["ds"] = pd.to_datetime(cur_df["ds"])

    ref_df.to_parquet(reference_out.path)
    cur_df.to_parquet(current_out.path)


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=[
        "evidently==0.4.30",
        # Evidently 0.4.30's np.corrcoef path breaks on NumPy 2.x with
        # "AttributeError: 'float' object has no attribute 'shape'". Pin to the
        # NumPy 1.x the project is tested against.
        "numpy==1.26.4",
        "pandas==2.2.2",
        "pyarrow==16.1.0",
        "google-cloud-storage==2.17.0",
    ],
)
def detect_drift(
    reference: Input[Dataset],
    current: Input[Dataset],
    drift_report_uri: str,
    drift_metrics: Output[Metrics],
) -> str:
    """Run Evidently data-drift and save an HTML + JSON report to GCS.

    Emits `dataset_drift` and `n_drifted_features` as KFP metrics so the run is
    visible in the Vertex AI Pipelines UI. Returns the GCS URI of the report.
    """
    import json

    import pandas as pd
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    from google.cloud import storage

    ref_df = pd.read_parquet(reference.path)
    cur_df = pd.read_parquet(current.path)

    # Evidently's DataDriftPreset runs np.corrcoef over the numeric columns.
    # BigQuery's mart returns some columns as pandas *nullable/extension* dtypes
    # (e.g. Int64 for EXTRACT(dayofweek/month), and nullable floats for lags).
    # Calling `.values` on those yields an object-dtype array, which makes
    # np.corrcoef fail with "'float' object has no attribute 'shape'". Coerce
    # every numeric column to plain float64 (and drop the date column, which is
    # not a drift feature) so numpy receives real float arrays.
    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[c for c in ("ds",) if c in df.columns])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df

    ref_df = _prep(ref_df)
    cur_df = _prep(cur_df)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)
    result = report.as_dict()

    summary = result["metrics"][0]["result"]
    dataset_drift = bool(summary.get("dataset_drift", False))
    n_drifted = int(summary.get("number_of_drifted_columns", 0))

    drift_metrics.log_metric("dataset_drift", float(dataset_drift))
    drift_metrics.log_metric("n_drifted_features", float(n_drifted))

    # Persist both HTML (human) and JSON (machine) reports to GCS.
    client = storage.Client()
    bucket_name = drift_report_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(drift_report_uri.replace("gs://", "").split("/")[1:])
    bucket = client.bucket(bucket_name)

    html_blob = f"{prefix}/drift_report.html"
    json_blob = f"{prefix}/drift_report.json"
    bucket.blob(html_blob).upload_from_string(
        report.get_html(), content_type="text/html"
    )
    bucket.blob(json_blob).upload_from_string(
        json.dumps(summary, default=str), content_type="application/json"
    )
    return f"gs://{bucket_name}/{html_blob}"
