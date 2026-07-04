"""Pipeline 1: Data Ingest, Validation & Drift.

DAG:  run_dbt_transform -> extract_reference_and_current -> detect_drift

Runs serverless on Vertex AI Pipelines. Every step uses the smallest slim
image and a hard BigQuery byte cap to stay inside BigQuery Sandbox + GCS free
tiers.
"""

from kfp import dsl

from forecasting.components.data_components import (
    detect_drift,
    extract_reference_and_current,
    run_dbt_transform,
)


@dsl.pipeline(
    name="demand-data-ingest-validation-drift",
    description="dbt transform -> BQ extract -> Evidently drift report to GCS.",
)
def data_ingest_validation_pipeline(
    project_id: str,
    bq_dataset_mart: str,
    bq_location: str,
    bq_max_bytes_billed: int,
    drift_report_uri: str,
    split_date: str,
    dbt_target: str = "prod",
) -> None:
    dbt_task = run_dbt_transform(
        project_id=project_id,
        bq_dataset_mart=bq_dataset_mart,
        bq_location=bq_location,
        dbt_target=dbt_target,
    )
    dbt_task.set_display_name("dbt-build-mart")
    # Cap resources to the smallest CPU footprint (cost control).
    dbt_task.set_cpu_limit("1").set_memory_limit("2G")

    extract_task = extract_reference_and_current(
        project_id=project_id,
        mart_table=dbt_task.output,
        bq_max_bytes_billed=bq_max_bytes_billed,
        split_date=split_date,
    )
    extract_task.set_display_name("extract-reference-current")
    extract_task.set_cpu_limit("1").set_memory_limit("2G")

    drift_task = detect_drift(
        reference=extract_task.outputs["reference_out"],
        current=extract_task.outputs["current_out"],
        drift_report_uri=drift_report_uri,
    )
    drift_task.set_display_name("evidently-drift-detection")
    drift_task.set_cpu_limit("1").set_memory_limit("2G")
