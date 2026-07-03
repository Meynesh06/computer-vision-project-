#!/usr/bin/env bash
# scripts/gcp/_lib.sh
#
# Shared helpers sourced by every scripts/gcp/*.sh script. Source with:
#   source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#
# DRY_RUN=1 previews every mutating gcloud/gsutil command instead of running
# it -- real money is at stake here, so every script supports a preview
# pass before anything executes for real.
set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_REGION="${GCP_REGION:-${GCP_ZONE%-*}}"
GCS_BUCKET="${GCS_BUCKET:-${GCP_PROJECT}-tissue-classifier}"
PER_JOB_COST_CAP_USD="${PER_JOB_COST_CAP_USD:-40}"
TOTAL_BUDGET_USD="${TOTAL_BUDGET_USD:-300}"

# This billing account invoices in INR, not USD (confirmed via
# `gcloud billing accounts describe`) -- the Billing Budgets API requires
# --budget-amount in the billing account's own currency, so budget alerts
# need a USD->INR conversion. GCP Spot/Compute pricing itself is quoted in
# USD list price everywhere else in these scripts (the watchdog cost cap,
# calibration cost projections, etc.) since that's the stable public
# reference; only the actual Budget object needs INR. Rate is a point-in-time
# snapshot (~95.4 INR/USD, checked 2026-07-03) with a small safety margin
# built in (rounded down) so a real-time rate move doesn't silently make the
# alert threshold too generous -- refresh USD_TO_INR_RATE periodically.
USD_TO_INR_RATE="${USD_TO_INR_RATE:-95}"

usd_to_inr() {
  # $1: USD amount -> whole-rupee INR amount (floor), via awk (no bc dependency)
  awk -v usd="$1" -v rate="$USD_TO_INR_RATE" 'BEGIN { printf "%d", usd * rate }'
}

if [[ -z "$GCP_PROJECT" ]]; then
  echo "error: no GCP project set. Run 'gcloud config set project <id>' or export GCP_PROJECT=<id>." >&2
  exit 1
fi

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

# run <cmd> [args...]
# Executes the command, or under DRY_RUN=1 just prints what would run.
run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "error: required command '$cmd' not found on PATH" >&2
    exit 1
  fi
}
