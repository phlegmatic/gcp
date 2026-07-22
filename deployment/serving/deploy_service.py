#!/usr/bin/env python
"""Build + deploy the inference container to Cloud Run (scale-to-zero).

Why Cloud Run (not a Vertex AI Endpoint)?
-----------------------------------------
A Vertex AI Endpoint keeps at least one node running 24/7 and bills for it even
at zero traffic. For a sporadic demo on a $300 trial that contradicts the whole
free-tier ethos of this repo. Cloud Run scales to ZERO instances when idle
($0), autoscales on demand, gives free HTTPS, and hosts the API + demo UI in one
container. See PROJECT_OVERVIEW.md (Model Serving section) for the trade-offs.

This script shells out to `gcloud` (Cloud Build + Cloud Run) rather than adding
a new cloud SDK dependency, mirroring the repo's documented-CLI approach to
infra. Every step is also written out below as copy-pasteable commands.

Prerequisites (one-time, least-privilege service account)
---------------------------------------------------------
    PROJECT=$GCP_PROJECT_ID
    REGION=us-central1
    BUCKET=$GCS_BUCKET

    # 1. Enable APIs
    gcloud services enable run.googleapis.com \\
        artifactregistry.googleapis.com cloudbuild.googleapis.com

    # 2. Artifact Registry repo for the image
    gcloud artifacts repositories create serving \\
        --repository-format=docker --location=$REGION

    # 3. Dedicated runtime service account (read-only on the models bucket)
    gcloud iam service-accounts create run-inference \\
        --display-name="Cloud Run inference (read-only models)"
    gsutil iam ch \\
        serviceAccount:run-inference@$PROJECT.iam.gserviceaccount.com:\\
roles/storage.objectViewer \\
        gs://$BUCKET

Usage
-----
    # Build image (Cloud Build) + deploy service (scale-to-zero, public demo):
    python deployment/serving/deploy_service.py --deploy

    # Build + push image only:
    python deployment/serving/deploy_service.py --build-only

    # Print the equivalent gcloud commands without running anything:
    python deployment/serving/deploy_service.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from forecasting.config import get_settings

SERVICE_NAME = "demand-forecasting"
AR_REPO = "serving"
REPO_ROOT = Path(__file__).resolve().parents[2]
CLOUDBUILD = Path(__file__).parent / "cloudbuild.yaml"

# --- Free-tier / cost guardrails (the "seasoned" part) ----------------------
# min-instances=0  -> scale to zero, $0 while idle (the entire point).
# max-instances=2  -> cap runaway cost / blast radius.
# concurrency=80   -> sklearn predict is cheap; pack many requests per instance.
# cpu/memory       -> smallest billable size that fits the pickle comfortably.
RUN_FLAGS = [
    "--min-instances=0",
    "--max-instances=2",
    "--concurrency=80",
    "--cpu=1",
    "--memory=512Mi",
    "--timeout=60",
    "--allow-unauthenticated",  # public demo; see docs for the IAP/auth path.
]


def _image_uri(settings) -> str:
    return (
        f"{settings.region}-docker.pkg.dev/"
        f"{settings.project_id}/{AR_REPO}/{SERVICE_NAME}:latest"
    )


def _serving_sa(settings) -> str:
    return f"run-inference@{settings.project_id}.iam.gserviceaccount.com"


def _model_uri(settings) -> str:
    return settings.serving_model_uri or f"{settings.model_prefix}/ensemble_model.pkl"


def _build_cmd(settings) -> list[str]:
    return [
        "gcloud",
        "builds",
        "submit",
        str(REPO_ROOT),
        f"--config={CLOUDBUILD}",
        f"--substitutions=_IMAGE={_image_uri(settings)}",
        f"--project={settings.project_id}",
    ]


def _deploy_cmd(settings) -> list[str]:
    env_vars = ",".join(
        [
            f"SERVING_MODEL_URI={_model_uri(settings)}",
            f"GCP_PROJECT_ID={settings.project_id}",
            f"GCS_BUCKET={settings.gcs_bucket}",
            f"GCS_PIPELINE_ROOT={settings.pipeline_root}",
            f"GCS_DRIFT_PREFIX={settings.drift_prefix}",
            f"GCS_MODEL_PREFIX={settings.model_prefix}",
        ]
    )
    return [
        "gcloud",
        "run",
        "deploy",
        SERVICE_NAME,
        f"--image={_image_uri(settings)}",
        f"--region={settings.region}",
        f"--project={settings.project_id}",
        f"--service-account={_serving_sa(settings)}",
        f"--set-env-vars={env_vars}",
        *RUN_FLAGS,
    ]


def _run(cmd: list[str], dry_run: bool) -> None:
    print(f"[cmd] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--deploy", action="store_true", help="Build image + deploy to Cloud Run."
    )
    action.add_argument(
        "--build-only", action="store_true", help="Build + push image only."
    )
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud commands without executing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()

    print(f"[info] image  = {_image_uri(settings)}")
    print(f"[info] model  = {_model_uri(settings)}")
    print(f"[info] sa     = {_serving_sa(settings)}")

    dry = args.dry_run
    _run(_build_cmd(settings), dry)

    if args.build_only:
        print("[done] build-only; not deployed.")
        return 0

    _run(_deploy_cmd(settings), dry)
    print("[done] deployed. Fetch the URL with:")
    print(
        f"  gcloud run services describe {SERVICE_NAME} "
        f"--region={settings.region} --format='value(status.url)'"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
