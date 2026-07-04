#!/usr/bin/env bash
# ============================================================================
# e2-micro cron trigger for Vertex AI Pipelines (Always-Free VM strategy).
#
# The Always-Free e2-micro VM (us-central1/us-west1/us-east1) runs this script
# from crontab. It does NOT run any ML itself -- it only *submits* the pipeline
# to serverless Vertex AI Pipelines, then goes back to idle. This keeps the VM
# well within the free 1-vCPU burst budget.
#
# One-time VM setup (uv-based, no system Python management needed):
#   sudo apt-get update && sudo apt-get install -y git curl
#   curl -LsSf https://astral.sh/uv/install.sh | sh   # installs uv
#   git clone <repo> && cd <repo>
#   uv sync --extra tracking        # uv fetches the pinned Python + deps
#   cp .env.example .env && edit .env
#
# Crontab (weekly data pipeline Mon 06:00, training Mon 07:00):
#   0 6 * * 1 /home/user/repo/scripts/trigger_from_e2_micro.sh data   >> ~/pipe.log 2>&1
#   0 7 * * 1 /home/user/repo/scripts/trigger_from_e2_micro.sh training >> ~/pipe.log 2>&1
# ============================================================================
set -euo pipefail

PIPELINE="${1:-data}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Ensure uv is on PATH (default install location for the systemd/cron user).
export PATH="$HOME/.local/bin:$PATH"

echo "[$(date -u +%FT%TZ)] Submitting '${PIPELINE}' pipeline to Vertex AI..."
# `uv run` transparently syncs deps if needed and uses the project environment.
uv run python deployment/deploy_pipeline.py --pipeline "$PIPELINE" --submit
echo "[$(date -u +%FT%TZ)] Submission complete."
