"""Pipeline 2: Parallel Training & Ensembling.

DAG:
                 +--> train_single_model(ridge) --------+
   load_data ----+                                       +--> build_ensemble
                 +--> train_single_model(random_forest)-+

The two training tasks have no dependency on each other, so Vertex AI runs them
in parallel. `build_ensemble` fans them in, waiting on both before executing.
"""

from kfp import dsl
from kfp.dsl import Dataset, Output

from forecasting.components.training_components import (
    build_ensemble,
    train_single_model,
)

PY_IMAGE = "python:3.11-slim"


@dsl.component(
    base_image=PY_IMAGE,
    packages_to_install=[
        "google-cloud-bigquery==3.25.0",
        "pandas==2.2.2",
        "numpy==1.26.4",
        "pyarrow==16.1.0",
        "db-dtypes==1.2.0",
    ],
)
def load_training_data(
    project_id: str,
    mart_table: str,
    bq_max_bytes_billed: int,
    training_data: Output[Dataset],
) -> None:
    """Pull the feature mart from BigQuery into a parquet Dataset artifact."""
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=bq_max_bytes_billed)
    df = client.query(
        f"SELECT * FROM `{mart_table}` ORDER BY ds", job_config=job_config
    ).to_dataframe()
    df.to_parquet(training_data.path)


@dsl.pipeline(
    name="demand-training-ensemble",
    description="Parallel Ridge + RandomForest training with weighted ensemble.",
)
def training_ensemble_pipeline(
    project_id: str,
    mart_table: str,
    bq_max_bytes_billed: int,
    model_output_uri: str,
    val_fraction: float = 0.2,
) -> None:
    load_task = load_training_data(
        project_id=project_id,
        mart_table=mart_table,
        bq_max_bytes_billed=bq_max_bytes_billed,
    )
    load_task.set_display_name("load-training-data")
    load_task.set_cpu_limit("1").set_memory_limit("2G")

    # --- Parallel branch A: Ridge -------------------------------------------
    ridge_task = train_single_model(
        training_data=load_task.outputs["training_data"],
        model_name="ridge",
        val_fraction=val_fraction,
    )
    ridge_task.set_display_name("train-ridge")
    ridge_task.set_cpu_limit("1").set_memory_limit("2G")

    # --- Parallel branch B: Random Forest -----------------------------------
    rf_task = train_single_model(
        training_data=load_task.outputs["training_data"],
        model_name="random_forest",
        val_fraction=val_fraction,
    )
    rf_task.set_display_name("train-random-forest")
    rf_task.set_cpu_limit("1").set_memory_limit("2G")

    # --- Fan-in: ensemble (implicitly waits for BOTH branches) --------------
    ensemble_task = build_ensemble(
        training_data=load_task.outputs["training_data"],
        model_a=ridge_task.outputs["trained_model"],
        model_b=rf_task.outputs["trained_model"],
        rmse_a=ridge_task.outputs["Output"],
        rmse_b=rf_task.outputs["Output"],
        val_fraction=val_fraction,
        model_output_uri=model_output_uri,
    )
    ensemble_task.set_display_name("build-weighted-ensemble")
    ensemble_task.set_cpu_limit("1").set_memory_limit("2G")
