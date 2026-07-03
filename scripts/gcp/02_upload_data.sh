#!/usr/bin/env bash
# scripts/gcp/02_upload_data.sh
#
# Uploads already-downloaded local dataset zips to GCS (never re-fetches
# from Zenodo -- reuses what scripts/download_data.sh already pulled down),
# and builds + uploads a small calibration subset for Phase 3's GPU-tier
# comparison. Regional bucket, ideally in the same region the VM will run
# in, to avoid cross-region egress charges.
#
# Usage:
#   scripts/gcp/02_upload_data.sh [full|calibration|all]   (default: all)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

MODE="${1:-all}"

ensure_bucket() {
  if gsutil ls -b "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
    log "Bucket gs://${GCS_BUCKET} already exists"
  else
    log "Creating bucket gs://${GCS_BUCKET} in $GCP_REGION"
    run gsutil mb -l "$GCP_REGION" "gs://${GCS_BUCKET}"
  fi
}

upload_full_data() {
  local nct_zip="data/raw/NCT-CRC-HE-100K.zip"
  local crc_zip="data/raw/CRC-VAL-HE-7K.zip"
  if [[ ! -f "$nct_zip" || ! -f "$crc_zip" ]]; then
    log "error: expected local zips missing ($nct_zip / $crc_zip) -- run scripts/download_data.sh first"
    exit 1
  fi
  log "Uploading full dataset zips to gs://${GCS_BUCKET}/data/raw/ (the ~11GB NCT zip will take a while)"
  run gsutil -m cp "$nct_zip" "$crc_zip" "gs://${GCS_BUCKET}/data/raw/"
}

upload_calibration_subset() {
  local subset_dir="data/calibration_subset"
  if [[ -d "$subset_dir" ]]; then
    log "$subset_dir already exists locally, reusing (delete it to rebuild)"
  else
    # Sampled from the already-extracted CRC-VAL-HE-7K rather than
    # extracting the full 100K zip locally just to draw a few hundred
    # images. Train/eval subsets are drawn independently (may overlap) --
    # fine here because calibration only measures wall-clock throughput per
    # GPU config, not model accuracy/generalization.
    log "Building calibration subset from data/raw/CRC-VAL-HE-7K (train: 60/class, eval: 20/class)"
    uv run python -c "
from tissue_classifier.data import create_local_subset
create_local_subset('data/raw/CRC-VAL-HE-7K', '$subset_dir/train', per_class=60, seed=1)
create_local_subset('data/raw/CRC-VAL-HE-7K', '$subset_dir/eval', per_class=20, seed=2)
"
  fi
  log "Uploading calibration subset to gs://${GCS_BUCKET}/data/calibration_subset/"
  run gsutil -m cp -r "$subset_dir" "gs://${GCS_BUCKET}/data/"
}

ensure_bucket

case "$MODE" in
  full) upload_full_data ;;
  calibration) upload_calibration_subset ;;
  all)
    upload_full_data
    upload_calibration_subset
    ;;
  *)
    log "error: unknown mode '$MODE' (expected full|calibration|all)"
    exit 1
    ;;
esac

log "Done."
