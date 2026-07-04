#!/usr/bin/env python
"""Compile a KFP pipeline to YAML and submit it to Vertex AI Pipelines.

This is the single entry point used by both the e2-micro VM cron trigger and
the GitHub Actions workflow. It NEVER provisions a GKE cluster; Vertex AI
Pipelines runs the compiled spec serverlessly, which is the free-tier strategy.

Usage
-----
    # Compile only (no cost, safe to run in CI on every push):
    python deployment/deploy_pipeline.py --pipeline data --compile-only

    # Compile + submit a run to Vertex AI (incurs micro-billing on step vCPUs):
    python deployment/deploy_pipeline.py --pipeline training --submit

    # Schedule a recurring run (uses Vertex Pipelines Scheduler, no VM needed):
    python deployment/deploy_pipeline.py --pipeline data --schedule "0 6 * * 1"

Environment: all cloud settings come from `.env` / env vars via
`forecasting.config.get_settings()`. See `.env.example`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from kfp import compiler

from forecasting.config import get_settings
from forecasting.pipelines import PIPELINES

COMPILED_DIR = Path(__file__).parent / "compiled"


def _pipeline_parameters(pipeline_key: str, settings) -> dict:
    """Assemble the runtime parameters for each pipeline from settings."""
    if pipeline_key == "data":
        return {
            "project_id": settings.project_id,
            "bq_dataset_mart": settings.bq_dataset_mart,
            "bq_location": settings.bq_location,
            "bq_max_bytes_billed": settings.bq_max_bytes_billed,
            "drift_report_uri": settings.drift_prefix,
            # Reference/current split: everything before this date is "reference".
            "split_date": "2024-01-01",
            "dbt_target": "prod",
        }
    if pipeline_key == "training":
        return {
            "project_id": settings.project_id,
            "mart_table": f"{settings.project_id}."
            f"{settings.bq_dataset_mart}.demand_features",
            "bq_max_bytes_billed": settings.bq_max_bytes_billed,
            "model_output_uri": settings.model_prefix,
            "val_fraction": 0.2,
        }
    raise ValueError(f"Unknown pipeline key: {pipeline_key}")


def compile_pipeline(pipeline_key: str) -> Path:
    """Compile the selected pipeline to a versioned YAML spec. Returns its path."""
    if pipeline_key not in PIPELINES:
        raise ValueError(
            f"Unknown pipeline '{pipeline_key}'. Choices: {list(PIPELINES)}"
        )
    COMPILED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_path = COMPILED_DIR / f"{pipeline_key}_pipeline_{stamp}.yaml"

    compiler.Compiler().compile(
        pipeline_func=PIPELINES[pipeline_key],
        package_path=str(out_path),
    )
    # Also write a stable "latest" alias for CI artifact upload.
    latest = COMPILED_DIR / f"{pipeline_key}_pipeline_latest.yaml"
    latest.write_bytes(out_path.read_bytes())
    print(f"[compile] {pipeline_key} -> {out_path}")
    return out_path


def submit_run(pipeline_key: str, spec_path: Path, schedule: str | None) -> None:
    """Submit a one-off run OR create a recurring schedule on Vertex AI."""
    # Import here so `--compile-only` runs without cloud SDK auth.
    from google.cloud import aiplatform

    settings = get_settings()
    aiplatform.init(
        project=settings.project_id,
        location=settings.region,
        staging_bucket=settings.gcs_bucket,
        experiment=settings.experiment_name,
    )

    job = aiplatform.PipelineJob(
        display_name=f"{pipeline_key}-{datetime.utcnow():%Y%m%d-%H%M%S}",
        template_path=str(spec_path),
        pipeline_root=settings.pipeline_root,
        parameter_values=_pipeline_parameters(pipeline_key, settings),
        enable_caching=True,  # Free-tier saver: skip unchanged steps.
    )

    service_account = settings.vertex_sa_email or None

    if schedule:
        pipeline_job_schedule = job.create_schedule(
            display_name=f"{pipeline_key}-schedule",
            cron=schedule,
            service_account=service_account,
            max_concurrent_run_count=1,
        )
        print(
            f"[schedule] {pipeline_key} cron='{schedule}' -> "
            f"{pipeline_job_schedule.resource_name}"
        )
        return

    job.submit(service_account=service_account, experiment=settings.experiment_name)
    print(f"[submit] {pipeline_key} run submitted: {job.resource_name}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline",
        required=True,
        choices=list(PIPELINES),
        help="Which pipeline to compile/submit.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile to YAML only (zero cloud cost).",
    )
    action.add_argument(
        "--submit",
        action="store_true",
        help="Compile then submit a single run to Vertex AI Pipelines.",
    )
    parser.add_argument(
        "--schedule",
        metavar="CRON",
        default=None,
        help="Create a recurring Vertex AI schedule (implies --submit path).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = compile_pipeline(args.pipeline)

    if args.compile_only and not args.schedule:
        print("[done] compile-only mode; nothing submitted.")
        return 0

    submit_run(args.pipeline, spec_path, schedule=args.schedule)
    return 0


if __name__ == "__main__":
    sys.exit(main())
