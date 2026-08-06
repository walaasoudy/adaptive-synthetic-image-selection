#!/usr/bin/env bash
# Launch (or resume) Stage 1 SDXL LoRA training, resilient to RunPod pod restarts and SSH
# disconnects (docs/stage1_plan.md §10).
#
# On start: checks checkpoints/stage1_lora_sdxl/latest_run.json for an existing run_id, and that
# run's latest.json for the newest full-resumable checkpoint. If found, resumes from it;
# otherwise starts a fresh run. Runs under nohup so a dropped VS Code SSH session doesn't kill
# training; all output is appended to a log file on the persistent volume.
#
# Usage:
#   bash scripts/train/launch_resumable.sh [extra overrides passed through to train_lora_sdxl.py]
#
# Recommended before trusting a long run: kill this process deliberately once, then re-run this
# script and confirm it resumes from the same step (docs/stage1_plan.md §10).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PROJECT_ROOT:=$REPO_ROOT}"
export PROJECT_ROOT

CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints/stage1_lora_sdxl"
LOGS_DIR="${PROJECT_ROOT}/logs/stage1_lora_sdxl"
mkdir -p "$CHECKPOINTS_DIR" "$LOGS_DIR"

LATEST_RUN_FILE="${CHECKPOINTS_DIR}/latest_run.json"
RESUME_ARGS=()

if [[ -f "$LATEST_RUN_FILE" ]]; then
  RUN_ID=$(python3 -c "import json; print(json.load(open('${LATEST_RUN_FILE}'))['run_id'])")
  RUN_LATEST_JSON="${CHECKPOINTS_DIR}/${RUN_ID}/latest.json"
  if [[ -f "$RUN_LATEST_JSON" ]]; then
    CHECKPOINT_DIR=$(python3 -c "import json; print(json.load(open('${RUN_LATEST_JSON}'))['checkpoint_dir'])")
    echo "Found existing run ${RUN_ID}; resuming from ${CHECKPOINT_DIR}"
    RESUME_ARGS=(--resume-from "$CHECKPOINT_DIR" --run-id "$RUN_ID")
  else
    echo "Found run ${RUN_ID} but no checkpoint yet; continuing to write into it."
    RESUME_ARGS=(--run-id "$RUN_ID")
  fi
else
  echo "No prior run found; starting fresh."
fi

LOG_FILE="${LOGS_DIR}/train_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "Logging to ${LOG_FILE}"

cd "$PROJECT_ROOT"
nohup accelerate launch --config_file configs/accelerate_config.yaml \
  scripts/train/train_lora_sdxl.py "${RESUME_ARGS[@]}" "$@" \
  >> "$LOG_FILE" 2>&1 &

echo "Training launched in background, PID $!. Tail progress with:"
echo "  tail -f ${LOG_FILE}"
