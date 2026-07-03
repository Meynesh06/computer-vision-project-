#!/usr/bin/env bash
# scripts/gcp/_remote_bootstrap.sh
#
# Runs ON the VM (invoked by 04_deploy_and_run.sh over SSH, after the repo
# tarball has already been copied and extracted to ~/tissue-classifier).
# Installs uv, runs the test suite as a fail-fast sanity gate, pulls data
# from GCS, patches the config's gcs_checkpoint_bucket/num_gpus for this
# environment, then launches the cost-cap watchdog and the training run
# both detached so they survive SSH disconnect.
#
# Args: CONFIG_PATH FRESH_FLAG DATA_MODE
#   CONFIG_PATH  e.g. configs/calibration.yaml or configs/lora_finetune.yaml
#   FRESH_FLAG   "--fresh" or "" (forwarded to run_cloud_train.py)
#   DATA_MODE    "full" or "calibration" -- which GCS data prefix to pull
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

CONFIG_PATH="${1:?CONFIG_PATH required}"
FRESH_FLAG="${2:-}"
DATA_MODE="${3:?DATA_MODE required (full|calibration)}"

METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
meta() { curl -sf -H "Metadata-Flavor: Google" "$METADATA_URL/$1"; }

echo "[bootstrap] installing uv (if not already present) ..."
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[bootstrap] uv sync ..."
uv sync

echo "[bootstrap] sanity gate: uv run pytest -q (fail fast before spending GPU time on data pull) ..."
uv run pytest -q

GCS_BUCKET="$(meta gcs-bucket)"
mkdir -p data/raw

echo "[bootstrap] pulling data for DATA_MODE=$DATA_MODE from gs://${GCS_BUCKET} ..."
case "$DATA_MODE" in
  full)
    if [[ ! -d data/raw/NCT-CRC-HE-100K || ! -d data/raw/CRC-VAL-HE-7K ]]; then
      gsutil -m cp "gs://${GCS_BUCKET}/data/raw/NCT-CRC-HE-100K.zip" \
                    "gs://${GCS_BUCKET}/data/raw/CRC-VAL-HE-7K.zip" data/raw/
      bash scripts/download_data.sh --skip-download
    else
      echo "[bootstrap] full dataset already extracted, skipping pull"
    fi
    ;;
  calibration)
    if [[ ! -d data/calibration_subset ]]; then
      gsutil -m cp -r "gs://${GCS_BUCKET}/data/calibration_subset" data/
    else
      echo "[bootstrap] calibration subset already present, skipping pull"
    fi
    ;;
  *)
    echo "[bootstrap] error: unknown DATA_MODE '$DATA_MODE' (expected full|calibration)" >&2
    exit 1
    ;;
esac

echo "[bootstrap] Hugging Face token setup ..."
mkdir -p "$HOME/.cache/huggingface"
meta hf-token > "$HOME/.cache/huggingface/token"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
[[ -z "$NUM_GPUS" || "$NUM_GPUS" == "0" ]] && NUM_GPUS=1
echo "[bootstrap] patching $CONFIG_PATH: gcs_checkpoint_bucket=gs://${GCS_BUCKET}/checkpoints, num_gpus=${NUM_GPUS}"
uv run python3 - "$CONFIG_PATH" "$GCS_BUCKET" "$NUM_GPUS" <<'PYEOF'
import sys
import yaml

config_path, bucket, num_gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(config_path) as f:
    config = yaml.safe_load(f)
config["gcs_checkpoint_bucket"] = f"gs://{bucket}/checkpoints"
config["num_gpus"] = num_gpus
with open(config_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PYEOF

echo "[bootstrap] starting cost-cap watchdog (detached) ..."
nohup bash scripts/gcp/05_watchdog.sh > "$HOME/watchdog_stdout.log" 2>&1 < /dev/null &
disown

echo "[bootstrap] launching training run (detached): $CONFIG_PATH $FRESH_FLAG"
nohup uv run python scripts/run_cloud_train.py "$CONFIG_PATH" $FRESH_FLAG > "$HOME/train.log" 2>&1 < /dev/null &
disown

echo "[bootstrap] done. Training running in background; tail ~/train.log and ~/watchdog.log to follow."
