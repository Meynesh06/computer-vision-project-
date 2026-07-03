#!/usr/bin/env bash
# scripts/download_data.sh
#
# Downloads NCT-CRC-HE-100K + CRC-VAL-HE-7K from Zenodo record 1214456 into
# data/raw/ (resumable, skips datasets already extracted). With --subset N,
# additionally builds a small stratified local subset (N images/class) into
# data/local_subset/ for Mac smoke-testing. --skip-download assumes the zips
# are already staged under data/raw/ (e.g. uploaded to a cloud VM from a GCS
# bucket) and only extracts + optionally samples them.
set -euo pipefail

DATA_DIR="data/raw"
SUBSET_N=""
SKIP_DOWNLOAD=0

# Direct Zenodo file URLs for record 1214456 (Macenko-normalized images --
# the canonical benchmark protocol for this dataset, not the *-NONORM
# variant). Verified reachable (HTTP 200) as of writing; re-check with
# `curl -I` if Zenodo ever reorganizes the record.
declare -A DATASET_URLS=(
  ["NCT-CRC-HE-100K"]="https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip?download=1"
  ["CRC-VAL-HE-7K"]="https://zenodo.org/records/1214456/files/CRC-VAL-HE-7K.zip?download=1"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      SUBSET_N="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DATA_DIR"

# Expected layout after download+extract:
#   data/raw/NCT-CRC-HE-100K/<CLASS_NAME>/*.tif
#   data/raw/CRC-VAL-HE-7K/<CLASS_NAME>/*.tif

if [[ "$SKIP_DOWNLOAD" -eq 0 ]]; then
  for name in "${!DATASET_URLS[@]}"; do
    extracted_dir="$DATA_DIR/$name"
    zip_path="$DATA_DIR/$name.zip"

    if [[ -d "$extracted_dir" ]]; then
      echo "$extracted_dir already extracted, skipping."
      continue
    fi

    echo "Downloading $name from Zenodo record 1214456 ..."
    curl -fL -C - --retry 5 --retry-delay 5 -o "$zip_path" "${DATASET_URLS[$name]}"

    echo "Extracting $zip_path -> $extracted_dir ..."
    unzip -q -o "$zip_path" -d "$DATA_DIR"
  done
else
  echo "Skipping download (--skip-download); expecting zips/extracted data already under $DATA_DIR."
  for name in "${!DATASET_URLS[@]}"; do
    extracted_dir="$DATA_DIR/$name"
    zip_path="$DATA_DIR/$name.zip"
    if [[ -d "$extracted_dir" ]]; then
      continue
    fi
    if [[ -f "$zip_path" ]]; then
      echo "Extracting $zip_path -> $extracted_dir ..."
      unzip -q -o "$zip_path" -d "$DATA_DIR"
    else
      echo "warning: neither $extracted_dir nor $zip_path found; skipping $name" >&2
    fi
  done
fi

if [[ -n "$SUBSET_N" ]]; then
  echo "Building local subset: $SUBSET_N images/class -> data/local_subset/"
  uv run python -c "
from tissue_classifier.data import create_local_subset
create_local_subset('data/raw/NCT-CRC-HE-100K', 'data/local_subset', per_class=$SUBSET_N, seed=0)
"
fi

echo "Done."
