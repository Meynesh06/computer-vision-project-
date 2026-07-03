#!/usr/bin/env bash
# scripts/gcp/04_deploy_and_run.sh
#
# Pushes the repo to an already-created VM (run 03_create_vm.sh first),
# then runs scripts/gcp/_remote_bootstrap.sh over SSH to install deps, pull
# data, sanity-test, and launch training + the cost-cap watchdog detached.
#
# Code transfer uses a local tarball + scp rather than `gcloud compute scp
# --recurse` directly: gcloud's scp wrapper has no --exclude support, and
# copying .venv/data/raw/outputs verbatim would be slow and pointless (the
# VM builds its own venv and pulls data from GCS separately). No git remote
# is configured for this project, so tarball+scp is the transfer path.
#
# Usage:
#   scripts/gcp/04_deploy_and_run.sh --name NAME --data-mode full|calibration \
#     [--zone ZONE] [--config CONFIG_PATH] [--fresh]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

NAME=""
ZONE="$GCP_ZONE"
CONFIG_PATH=""
FRESH_FLAG=""
DATA_MODE=""
REMOTE_DIR="tissue-classifier"

usage() {
  cat >&2 <<EOF
Usage: $0 --name NAME --data-mode full|calibration [--zone ZONE] [--config CONFIG_PATH] [--fresh]

--config defaults to configs/calibration.yaml for --data-mode calibration,
or configs/lora_finetune.yaml for --data-mode full.
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --data-mode) DATA_MODE="$2"; shift 2 ;;
    --fresh) FRESH_FLAG="--fresh"; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$NAME" ]] && { echo "error: --name is required" >&2; usage; }
case "$DATA_MODE" in
  full) CONFIG_PATH="${CONFIG_PATH:-configs/lora_finetune.yaml}" ;;
  calibration) CONFIG_PATH="${CONFIG_PATH:-configs/calibration.yaml}" ;;
  *) echo "error: --data-mode must be 'full' or 'calibration'" >&2; usage ;;
esac

TARBALL="$(mktemp -t tissue-classifier-deploy-XXXXXX).tar.gz"
trap 'rm -f "$TARBALL"' EXIT

log "Packing repo (excluding local-only/large dirs) -> $TARBALL"
tar czf "$TARBALL" \
  --exclude='.venv' --exclude='data/raw' --exclude='data/calibration_subset' \
  --exclude='data/local_subset' --exclude='outputs' --exclude='.git' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
  --exclude='.claude' --exclude='.worktrees' --exclude='.agents' \
  --exclude='graphify-out' --exclude='.DS_Store' \
  .

log "Copying tarball to $NAME:~/repo.tar.gz ..."
run gcloud compute scp --zone="$ZONE" "$TARBALL" "$NAME:~/repo.tar.gz"

log "Extracting on $NAME and launching bootstrap (config=$CONFIG_PATH, data-mode=$DATA_MODE, fresh='$FRESH_FLAG') ..."
run gcloud compute ssh "$NAME" --zone="$ZONE" --command="
  set -euo pipefail
  rm -rf ~/$REMOTE_DIR
  mkdir -p ~/$REMOTE_DIR
  tar xzf ~/repo.tar.gz -C ~/$REMOTE_DIR
  rm -f ~/repo.tar.gz
  bash ~/$REMOTE_DIR/scripts/gcp/_remote_bootstrap.sh '$CONFIG_PATH' '$FRESH_FLAG' '$DATA_MODE'
"

log "Done. Training launched (detached) on $NAME."
log "Monitor with: scripts/gcp/06_monitor.sh --name $NAME --zone $ZONE"
