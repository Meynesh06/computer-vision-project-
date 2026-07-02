#!/usr/bin/env bash
# scripts/download_data.sh
#
# Downloads NCT-CRC-HE-100K + CRC-VAL-HE-7K from Zenodo record 1214456 into
# data/raw/. With --subset N, additionally builds a small stratified local
# subset (N images/class) into data/local_subset/ for Mac smoke-testing,
# without requiring the full ~24GB download.
set -euo pipefail

ZENODO_RECORD="https://zenodo.org/records/1214456"
DATA_DIR="data/raw"
SUBSET_N=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      SUBSET_N="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$DATA_DIR"

echo "Downloading NCT-CRC-HE-100K and CRC-VAL-HE-7K from $ZENODO_RECORD ..."
echo "(Visit $ZENODO_RECORD to get the current direct file URLs; Zenodo file"
echo " links are not guaranteed stable enough to hardcode here.)"

# Expected layout after manual/automated download+extract:
#   data/raw/NCT-CRC-HE-100K/<CLASS_NAME>/*.tif
#   data/raw/CRC-VAL-HE-7K/<CLASS_NAME>/*.tif

if [[ -n "$SUBSET_N" ]]; then
  echo "Building local subset: $SUBSET_N images/class -> data/local_subset/"
  uv run python -c "
from tissue_classifier.data import create_local_subset
create_local_subset('data/raw/NCT-CRC-HE-100K', 'data/local_subset', per_class=$SUBSET_N, seed=0)
"
fi

echo "Done."
