#!/usr/bin/env bash
# scripts/gcp/01_budget_alert.sh
#
# Creates a GCP Billing Budget scoped to this project, with email alerts at
# 50/90/100% of TOTAL_BUDGET_USD (default $300, see _lib.sh).
#
# IMPORTANT: GCP budgets are informational only -- they send email/pub-sub
# notifications when thresholds are crossed, but do NOT stop spend or
# disable billing. Real enforcement for this project comes from
# 03_create_vm.sh's --max-run-duration backstop and 05_watchdog.sh's
# per-job cost cap, not this script.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/gcp/_lib.sh

BILLING_ACCOUNT="$(gcloud billing projects describe "$GCP_PROJECT" --format='value(billingAccountName)' | sed 's#billingAccounts/##')"
if [[ -z "$BILLING_ACCOUNT" ]]; then
  log "error: could not resolve billing account for $GCP_PROJECT (is billing enabled?)"
  exit 1
fi

BILLING_CURRENCY="$(gcloud billing accounts describe "$BILLING_ACCOUNT" --format='value(currencyCode)')"
BUDGET_AMOUNT="$TOTAL_BUDGET_USD"
if [[ "$BILLING_CURRENCY" != "USD" ]]; then
  BUDGET_AMOUNT="$(usd_to_inr "$TOTAL_BUDGET_USD")"
  log "Billing account currency is $BILLING_CURRENCY, not USD -- converting"
  log "\$${TOTAL_BUDGET_USD} USD -> ${BUDGET_AMOUNT} ${BILLING_CURRENCY} (rate: $USD_TO_INR_RATE, see _lib.sh)"
fi

log "Creating a $BUDGET_AMOUNT $BILLING_CURRENCY budget (\$${TOTAL_BUDGET_USD} USD) on billing"
log "account $BILLING_ACCOUNT, scoped to $GCP_PROJECT"
log "REMINDER: this is an email alert only -- it cannot stop spend. Enforcement is the"
log "on-VM watchdog cost cap (\$${PER_JOB_COST_CAP_USD}/job) + --max-run-duration on the VM itself."

# NOTE: --budget-amount must be a bare number (e.g. "300"), not "300USD" --
# appending a currency code that doesn't match the billing account's own
# currency makes the Billing Budgets API reject the whole request with an
# unhelpful generic 400 INVALID_ARGUMENT (confirmed by isolating each flag
# against the live API before wiring this in). A bare number is always
# interpreted in the billing account's own currency.
run gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT" \
  --display-name="tissue-classifier-cloud-finetune" \
  --budget-amount="${BUDGET_AMOUNT}" \
  --filter-projects="projects/${GCP_PROJECT}" \
  --threshold-rule=percent=0.50 \
  --threshold-rule=percent=0.90 \
  --threshold-rule=percent=1.00

log "Done. Check email associated with the billing account for alert delivery, or view in"
log "Console > Billing > Budgets & alerts."
