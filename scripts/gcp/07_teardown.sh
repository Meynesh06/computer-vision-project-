#!/usr/bin/env bash
# scripts/gcp/07_teardown.sh
#
# Deletes the VM (stops all further compute + boot-disk billing for it).
# Leaves the GCS bucket (checkpoints/data/results) untouched by default --
# that's the durable artifact set, and auto-deleting it on a misclick is
# exactly the kind of costly mistake to design against. Pass
# --delete-bucket (with its own extra type-to-confirm prompt) if you really
# want to remove it too.
#
# Usage:
#   scripts/gcp/07_teardown.sh --name NAME [--zone ZONE] [--delete-bucket]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

NAME=""
ZONE="$GCP_ZONE"
DELETE_BUCKET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --delete-bucket) DELETE_BUCKET=1; shift ;;
    -h|--help) echo "Usage: $0 --name NAME [--zone ZONE] [--delete-bucket]" >&2; exit 1 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -z "$NAME" ]] && { echo "error: --name is required" >&2; exit 1; }

if gcloud compute instances describe "$NAME" --zone="$ZONE" >/dev/null 2>&1; then
  log "Deleting VM $NAME in $ZONE (stops all further billing for it) ..."
  run gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet
else
  log "VM $NAME not found in $ZONE (already deleted, self-terminated by watchdog, or never created) -- nothing to delete."
fi

log "GCS bucket gs://${GCS_BUCKET} left intact (checkpoints/data/results) -- delete manually or pass --delete-bucket if you're sure."

if [[ "$DELETE_BUCKET" == "1" ]]; then
  log "WARNING: --delete-bucket passed. This permanently removes ALL checkpoints, data, and results in gs://${GCS_BUCKET}."
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    log "[DRY_RUN] would prompt for confirmation, then: gsutil -m rm -r gs://${GCS_BUCKET}"
  else
    read -r -p "Type the bucket name to confirm deletion ($GCS_BUCKET): " CONFIRM
    if [[ "$CONFIRM" == "$GCS_BUCKET" ]]; then
      gsutil -m rm -r "gs://${GCS_BUCKET}"
      log "Bucket deleted."
    else
      log "Confirmation did not match bucket name -- bucket NOT deleted."
    fi
  fi
fi

BILLING_ACCOUNT="$(gcloud billing projects describe "$GCP_PROJECT" --format='value(billingAccountName)' 2>/dev/null | sed 's#billingAccounts/##')"
log "Reminder: check actual spend at Console > Billing > Reports (billing account: ${BILLING_ACCOUNT:-unknown})."
