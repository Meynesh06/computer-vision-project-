#!/usr/bin/env bash
# scripts/gcp/05_watchdog.sh
#
# Runs ON the VM (started by 04_deploy_and_run.sh as a background process,
# independent of the training process tree so a hung/crashed job can't
# block it). Every ~60s, computes accrued cost = elapsed hours since this
# watchdog started * SPOT_PRICE_USD (read from instance metadata, supplied
# at VM-creation time by 03_create_vm.sh) and, once it exceeds
# COST_CAP_USD, attempts a best-effort time-boxed checkpoint sync to GCS,
# then halts the VM.
#
# Deliberately uses an OS-level `shutdown` rather than
# `gcloud compute instances delete` on itself: the latter would require the
# VM's service account to hold compute.instances.delete on itself, broader
# IAM scope than this box should need. A stopped Spot VM stops accruing the
# (dominant) GPU-hour charge immediately; the boot disk's much smaller
# storage cost is cleaned up by 07_teardown.sh afterward.
set -euo pipefail

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
meta() {
  curl -sf -H "Metadata-Flavor: Google" "$METADATA_URL/$1" 2>/dev/null || true
}

SPOT_PRICE_USD="$(meta spot-price-usd)"
COST_CAP_USD="$(meta cost-cap-usd)"
GCS_BUCKET="$(meta gcs-bucket)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/tissue-classifier/outputs/checkpoints}"
LOG_FILE="${WATCHDOG_LOG:-$HOME/watchdog.log}"

log() {
  echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG_FILE"
}

if [[ -z "$SPOT_PRICE_USD" || -z "$COST_CAP_USD" ]]; then
  log "FATAL: spot-price-usd/cost-cap-usd metadata missing -- refusing to run unguarded"
  exit 1
fi

START_EPOCH=$(date +%s)
log "watchdog started: spot_price=\$${SPOT_PRICE_USD}/hr cost_cap=\$${COST_CAP_USD}"

while true; do
  NOW_EPOCH=$(date +%s)
  ELAPSED_HOURS=$(awk -v s="$START_EPOCH" -v n="$NOW_EPOCH" 'BEGIN { printf "%.6f", (n - s) / 3600 }')
  ACCRUED_USD=$(awk -v h="$ELAPSED_HOURS" -v p="$SPOT_PRICE_USD" 'BEGIN { printf "%.4f", h * p }')
  log "elapsed=${ELAPSED_HOURS}h accrued=\$${ACCRUED_USD} cap=\$${COST_CAP_USD}"

  if awk -v a="$ACCRUED_USD" -v c="$COST_CAP_USD" 'BEGIN { exit !(a >= c) }'; then
    log "COST CAP EXCEEDED (\$${ACCRUED_USD} >= \$${COST_CAP_USD}). Final checkpoint sync (30s budget), then halting."
    if [[ -n "$GCS_BUCKET" && -d "$CHECKPOINT_DIR" ]]; then
      if timeout 30 gsutil -m rsync -r "$CHECKPOINT_DIR" "gs://${GCS_BUCKET}/checkpoints/"; then
        log "final checkpoint sync OK"
      else
        log "final checkpoint sync FAILED or timed out (halting anyway)"
      fi
    else
      log "skipping final sync (GCS_BUCKET metadata or $CHECKPOINT_DIR not available)"
    fi
    log "halting VM now via shutdown -h now"
    sudo shutdown -h now
    exit 0
  fi

  sleep 60
done
