"""Pipeline 1 components: dbt transform, data extract, drift detection."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output

# Base image for lightweight python steps. Slim keeps cold-start + egress low.
PY_IMAGE = "python:3.10-slim"


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=["dbt-bigquery==1.8.2"],
)
def run_dbt_transform(
    project_id: str,
    bq_dataset_mart: str,
    bq_location: str,
    dbt_target: str = "prod",
) -> str:
    """Run `dbt build` against BigQuery Sandbox to materialize the mart table.

    Returns the fully-qualified mart table id. The dbt project is expected to be
    baked into the image or mounted; in the free-tier flow we shell out to dbt
    which reads profiles from env vars set below.
    """
    import os
    import subprocess

    os.environ["DBT_PROJECT_ID"] = project_id
    os.environ["DBT_DATASET"] = bq_dataset_mart
    os.environ["DBT_LOCATION"] = bq_location

    subprocess.run(
        ["dbt", "build", "--target", dbt_target, "--project-dir", "/app/dbt"],
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
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=bq_max_bytes_billed)

    ref_sql = f"SELECT * FROM `{mart_table}` WHERE ds < '{split_date}'"
    cur_sql = f"SELECT * FROM `{mart_table}` WHERE ds >= '{split_date}'"

    ref_df = client.query(ref_sql, job_config=job_config).to_dataframe()
    cur_df = client.query(cur_sql, job_config=job_config).to_dataframe()

    ref_df.to_parquet(reference_out.path)
    cur_df.to_parquet(current_out.path)


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=[
        "evidently==0.4.30",
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
