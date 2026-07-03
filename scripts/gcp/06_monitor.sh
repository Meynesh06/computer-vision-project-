#!/usr/bin/env bash
# scripts/gcp/06_monitor.sh
#
# Runs locally. Polls a running VM via SSH at --poll-interval, reporting
# GPU utilization and training/watchdog log progress. Detects two failure
# modes rather than looping silently:
#   - stall: output unchanged for STALL_MINUTES (possible hang)
#   - unreachable: SSH fails MAX_CONSECUTIVE_FAILURES times in a row --
#     the VM may have been preempted, self-terminated by the watchdog's
#     cost cap, or finished and been torn down.
#
# This is read-only (no mutating gcloud calls), so DRY_RUN doesn't apply.
#
# Usage:
#   scripts/gcp/06_monitor.sh --name NAME [--zone ZONE] [--poll-interval SECONDS]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

NAME=""
ZONE="$GCP_ZONE"
POLL_INTERVAL=45
STALL_MINUTES=15
MAX_CONSECUTIVE_FAILURES=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --zone) ZONE="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 --name NAME [--zone ZONE] [--poll-interval SECONDS]" >&2; exit 1 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ -z "$NAME" ]] && { echo "error: --name is required" >&2; exit 1; }

FAILURES=0
LAST_SNAPSHOT=""
LAST_CHANGE_EPOCH=$(date +%s)

log "Monitoring $NAME in $ZONE every ${POLL_INTERVAL}s (Ctrl-C stops watching; the VM keeps running)"

REMOTE_CMD='
  echo "--- nvidia-smi ---"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
  echo "--- train.log (tail) ---"
  tail -n 5 ~/train.log 2>/dev/null || echo "(no train.log yet)"
  echo "--- watchdog.log (tail) ---"
  tail -n 3 ~/watchdog.log 2>/dev/null || echo "(no watchdog.log yet)"
'

while true; do
  if OUTPUT=$(gcloud compute ssh "$NAME" --zone="$ZONE" --command="$REMOTE_CMD" 2>&1); then
    FAILURES=0
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') ====="
    echo "$OUTPUT"

    if [[ "$OUTPUT" != "$LAST_SNAPSHOT" ]]; then
      LAST_SNAPSHOT="$OUTPUT"
      LAST_CHANGE_EPOCH=$(date +%s)
    else
      STALLED_MINUTES=$(( ($(date +%s) - LAST_CHANGE_EPOCH) / 60 ))
      if (( STALLED_MINUTES >= STALL_MINUTES )); then
        log "WARNING: no change in GPU/log output for ${STALLED_MINUTES} minutes -- possible stall."
      fi
    fi

    if echo "$OUTPUT" | grep -qi "COST CAP EXCEEDED"; then
      log "Watchdog reports cost cap exceeded -- VM is halting itself. Stopping monitor."
      exit 0
    fi
  else
    FAILURES=$((FAILURES + 1))
    log "SSH attempt failed ($FAILURES/$MAX_CONSECUTIVE_FAILURES) -- VM may be preempted, self-terminated (cost cap), or torn down."
    if (( FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
      log "VM unreachable after $MAX_CONSECUTIVE_FAILURES attempts. Stopping monitor -- check status with:"
      log "  gcloud compute instances describe $NAME --zone=$ZONE"
      exit 1
    fi
  fi

  sleep "$POLL_INTERVAL"
done
