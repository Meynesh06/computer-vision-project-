#!/usr/bin/env bash
# scripts/gcp/00_preflight.sh
#
# Single gate before any GCP spend. Verifies auth, project, billing,
# required APIs, live GPU quota/availability across candidate zones, local
# data, HF token, and that the local test suite is green. Fails loudly
# (non-zero exit) on any gap rather than letting a downstream script fail
# after money has already been committed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

FAILED=0

check() {
  local description="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    log "OK   - $description"
  else
    log "FAIL - $description"
    FAILED=1
  fi
}

# --- auth ---
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)"
if [[ -n "$ACTIVE_ACCOUNT" ]]; then
  log "OK   - authenticated as $ACTIVE_ACCOUNT"
else
  log "FAIL - no authenticated gcloud account (run: gcloud auth login)"
  FAILED=1
fi

log "Using project: $GCP_PROJECT"

# --- billing ---
BILLING_ENABLED="$(gcloud billing projects describe "$GCP_PROJECT" --format='value(billingEnabled)' 2>/dev/null || echo false)"
if [[ "$BILLING_ENABLED" == "True" ]]; then
  log "OK   - billing enabled on $GCP_PROJECT"
else
  log "FAIL - billing not enabled on $GCP_PROJECT"
  FAILED=1
fi

# --- required APIs ---
REQUIRED_APIS=(compute.googleapis.com storage.googleapis.com billingbudgets.googleapis.com)
ENABLED_APIS="$(gcloud services list --enabled --format='value(config.name)' 2>/dev/null || true)"
for api in "${REQUIRED_APIS[@]}"; do
  if grep -qx "$api" <<< "$ENABLED_APIS"; then
    log "OK   - API enabled: $api"
  else
    log "FAIL - API not enabled: $api (run: gcloud services enable $api)"
    FAILED=1
  fi
done

# --- global GPU quota (this is the cap that actually matters -- per-region
# numbers are meaningless if this is 0, which is the default for new/unused
# billing accounts) ---
GLOBAL_GPU_QUOTA="$(gcloud compute project-info describe --format=json 2>/dev/null \
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
if python3 -c "import sys; sys.exit(0 if float('$GLOBAL_GPU_QUOTA') > 0 else 1)"; then
  log "OK   - GPUS_ALL_REGIONS quota = $GLOBAL_GPU_QUOTA"
else
  log "FAIL - GPUS_ALL_REGIONS quota is 0 -- no GPU VM can be created anywhere in this project."
  log "       Request an increase: Console > IAM & Admin > Quotas > Compute Engine API > 'GPUs (all regions)'."
  FAILED=1
fi

# --- per-zone GPU type: does the SKU exist there, and is per-region quota > 0 ---
check_zone_gpu() {
  local zone="$1" gpu_type="$2" metric="$3"
  local region="${zone%-*}"
  local offered="no"
  if gcloud compute accelerator-types describe "$gpu_type" --zone="$zone" >/dev/null 2>&1; then
    offered="yes"
  fi
  local quota
  quota="$(gcloud compute regions describe "$region" --format=json 2>/dev/null \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
for q in data.get('quotas', []):
    if q['metric'] == '$metric':
        print(q['limit'])
        break
else:
    print(0)
")"
  if [[ "$offered" == "yes" ]] && python3 -c "import sys; sys.exit(0 if float('$quota') > 0 else 1)"; then
    log "OK   - $gpu_type usable in $zone (region quota $metric = $quota)"
  else
    log "WARN - $gpu_type NOT usable in $zone (offered=$offered, region quota $metric=$quota)"
  fi
}

for zone in us-central1-a us-central1-b us-east1-b us-east4-a; do
  check_zone_gpu "$zone" nvidia-l4 NVIDIA_L4_GPUS
  check_zone_gpu "$zone" nvidia-tesla-a100 NVIDIA_A100_GPUS
done

# --- local data ---
check "NCT-CRC-HE-100K.zip present" test -f data/raw/NCT-CRC-HE-100K.zip
check "CRC-VAL-HE-7K.zip present" test -f data/raw/CRC-VAL-HE-7K.zip

# --- HF token ---
check "Hugging Face token cached" test -f "$HOME/.cache/huggingface/token"

# --- local test suite ---
log "Running local test suite (uv run pytest -q) ..."
if uv run pytest -q; then
  log "OK   - local test suite green"
else
  log "FAIL - local test suite has failures"
  FAILED=1
fi

echo
if [[ "$FAILED" == "1" ]]; then
  log "PREFLIGHT FAILED -- fix the FAIL items above before spending any GCP money."
  exit 1
fi
log "PREFLIGHT PASSED."
