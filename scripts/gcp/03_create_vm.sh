#!/usr/bin/env bash
# scripts/gcp/03_create_vm.sh
#
# Creates a Spot GPU VM for either calibration or the full run. Refuses (no
# resources created, no charge) if the requested GPU type/count isn't
# actually offered or quota-permitted in the target zone -- confirmed live
# against the API, never assumed.
#
# GPU-count -> machine-type mapping is NOT a simple formula (e.g.
# g2-standard-32 has only 1 GPU despite 32 vCPUs) -- verified live via
# `gcloud compute machine-types describe` before wiring in. Valid L4 counts
# are exactly {1, 2, 4, 8}; "3x L4" is not a real GCE configuration.
#
# Usage:
#   scripts/gcp/03_create_vm.sh --name NAME --gpu-type l4|a100 --spot-price-usd PRICE \
#     [--gpu-count N] [--zone ZONE] [--max-run-duration DURATION] [--purpose calibration|full]
#
# --spot-price-usd is required, with NO default: GCP Spot prices change
# often and third-party pricing sites disagreed by up to ~8x for the same
# SKU when checked while building this script -- look up the current price
# at https://cloud.google.com/compute/all-pricing#gpus (or the Pricing
# Calculator) immediately before creating the VM. It's baked directly into
# the on-VM watchdog's cost-cap math (05_watchdog.sh), so a stale number
# there weakens the $ per-job safety cap.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

GPU_TYPE=""
GPU_COUNT=1
ZONE="$GCP_ZONE"
NAME=""
SPOT_PRICE_USD=""
MAX_RUN_DURATION="6h"
PURPOSE="calibration"

usage() {
  cat >&2 <<EOF
Usage: $0 --name NAME --gpu-type l4|a100 --spot-price-usd PRICE [options]

Required:
  --name NAME                 VM instance name
  --gpu-type l4|a100          GPU SKU
  --spot-price-usd PRICE      Current Spot \$/hr for this exact machine+GPU config
                              (look this up fresh -- see comment at top of script)

Options:
  --gpu-count N                 Number of GPUs: 1, 2, 4, or 8 (default: 1)
  --zone ZONE                    Zone (default: $GCP_ZONE)
  --max-run-duration DURATION    GCE-level hard backstop (default: 6h)
  --purpose calibration|full     VM label only (default: calibration)
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --gpu-count) GPU_COUNT="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --spot-price-usd) SPOT_PRICE_USD="$2"; shift 2 ;;
    --max-run-duration) MAX_RUN_DURATION="$2"; shift 2 ;;
    --purpose) PURPOSE="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done

[[ -z "$NAME" ]] && { echo "error: --name is required" >&2; usage; }
[[ -z "$GPU_TYPE" ]] && { echo "error: --gpu-type is required" >&2; usage; }
[[ -z "$SPOT_PRICE_USD" ]] && { echo "error: --spot-price-usd is required (see script header for why)" >&2; usage; }

case "$GPU_TYPE" in
  l4)
    ACCELERATOR="nvidia-l4"
    QUOTA_METRIC="NVIDIA_L4_GPUS"
    case "$GPU_COUNT" in
      1) MACHINE_TYPE="g2-standard-4" ;;
      2) MACHINE_TYPE="g2-standard-24" ;;
      4) MACHINE_TYPE="g2-standard-48" ;;
      8) MACHINE_TYPE="g2-standard-96" ;;
      *) echo "error: --gpu-count for l4 must be 1, 2, 4, or 8 (verified live GCE mapping; not a free choice)" >&2; exit 1 ;;
    esac
    ;;
  a100)
    ACCELERATOR="nvidia-tesla-a100"
    QUOTA_METRIC="NVIDIA_A100_GPUS"
    case "$GPU_COUNT" in
      1) MACHINE_TYPE="a2-highgpu-1g" ;;
      2) MACHINE_TYPE="a2-highgpu-2g" ;;
      4) MACHINE_TYPE="a2-highgpu-4g" ;;
      8) MACHINE_TYPE="a2-highgpu-8g" ;;
      *) echo "error: --gpu-count for a100 must be 1, 2, 4, or 8" >&2; exit 1 ;;
    esac
    ;;
  *) echo "error: --gpu-type must be 'l4' or 'a100'" >&2; exit 1 ;;
esac

REGION="${ZONE%-*}"

log "Verifying $ACCELERATOR x$GPU_COUNT ($MACHINE_TYPE) is actually available + quota-permitted in $ZONE ..."

if ! gcloud compute accelerator-types describe "$ACCELERATOR" --zone="$ZONE" >/dev/null 2>&1; then
  log "error: $ACCELERATOR is not offered in $ZONE. Aborting -- no resources created, no charge."
  exit 1
fi

REGION_QUOTA="$(gcloud compute regions describe "$REGION" --format=json 2>/dev/null \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if q['metric'] == '$QUOTA_METRIC':
        print(q['limit'])
        break
else:
    print(0)
")"
GLOBAL_QUOTA="$(gcloud compute project-info describe --format=json 2>/dev/null \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if q['metric'] == 'GPUS_ALL_REGIONS':
        print(q['limit'])
        break
else:
    print(0)
")"

if ! python3 -c "import sys; sys.exit(0 if float('$REGION_QUOTA') >= $GPU_COUNT else 1)"; then
  log "error: $QUOTA_METRIC quota in $REGION is $REGION_QUOTA, need >= $GPU_COUNT. Aborting -- no resources created, no charge."
  log "       Request an increase: Console > IAM & Admin > Quotas."
  exit 1
fi
if ! python3 -c "import sys; sys.exit(0 if float('$GLOBAL_QUOTA') >= $GPU_COUNT else 1)"; then
  log "error: global GPUS_ALL_REGIONS quota is $GLOBAL_QUOTA, need >= $GPU_COUNT. Aborting -- no resources created, no charge."
  log "       Request an increase: Console > IAM & Admin > Quotas > 'GPUs (all regions)'."
  exit 1
fi

log "OK - quota confirmed ($QUOTA_METRIC region=$REGION_QUOTA, GPUS_ALL_REGIONS=$GLOBAL_QUOTA)"

# HF token passed via --metadata-from-file, not embedded in a startup
# script or command line (which would land in shell history / serial logs).
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ ! -f "$HF_TOKEN_FILE" ]]; then
  log "error: HF token not found at $HF_TOKEN_FILE"
  exit 1
fi

# Deep Learning VM image: CUDA/PyTorch preinstalled, avoids burning
# calibration budget on driver setup. Verified this family exists via
# `gcloud compute images list --project=deeplearning-platform-release`
# before wiring in.
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2404-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"

log "Creating VM '$NAME' in $ZONE: $MACHINE_TYPE + ${GPU_COUNT}x $ACCELERATOR (Spot, image $IMAGE_FAMILY)"
log "Watchdog cost cap: \$${PER_JOB_COST_CAP_USD}/job at \$${SPOT_PRICE_USD}/hr (operator-supplied -- verify this is current)"

run gcloud compute instances create "$NAME" \
  --project="$GCP_PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --accelerator="type=$ACCELERATOR,count=$GPU_COUNT" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --max-run-duration="$MAX_RUN_DURATION" \
  --boot-disk-size=200GB \
  --metadata-from-file=hf-token="$HF_TOKEN_FILE" \
  --metadata="spot-price-usd=$SPOT_PRICE_USD,cost-cap-usd=$PER_JOB_COST_CAP_USD,gcs-bucket=$GCS_BUCKET" \
  --labels="purpose=$PURPOSE,project=tissue-classifier"

log "Done. Next: scripts/gcp/04_deploy_and_run.sh --name $NAME --zone $ZONE ..."
