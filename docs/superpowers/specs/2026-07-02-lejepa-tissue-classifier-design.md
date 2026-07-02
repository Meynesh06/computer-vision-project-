# leJEPA Tissue Classifier — Design

Date: 2026-07-02

## Goal

Build an image classifier for colorectal tissue histology patches using leJEPA
(a self-supervised joint-embedding predictive architecture using SIGReg loss)
as the representation-learning backbone, trained on the NCT-CRC-HE-100K
dataset (Zenodo record 1214456), predicting the full 9-class tissue taxonomy
(one of which is colorectal adenocarcinoma epithelium — the "cancer" class).

## Dataset

**NCT-CRC-HE-100K** (Zenodo 1214456):
- 100,000 H&E-stained colorectal tissue patches, 224×224px, 0.5 microns/pixel.
- 9 classes: adipose, background, debris, lymphocytes, mucus, smooth muscle,
  normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma
  epithelium.
- Macenko color-normalized version used for train/pretrain (canonical
  benchmark protocol for this dataset).
- Separate **CRC-VAL-HE-7K** validation set (7,180 patches from different
  patients) used as the held-out test set — avoids patient-level leakage
  between train and eval.
- Non-normalized version of the 100K set exists but is not used in this
  design.
- CC-BY 4.0 licensed.

## Approach

leJEPA is a self-supervised pretraining method, not a classifier on its own.
Pipeline:

1. **Self-supervised pretraining** — train a backbone with leJEPA's SIGReg
   loss on the *unlabeled* NCT-CRC-HE-100K patches (labels ignored at this
   stage). Domain-specific pretraining on histopathology images rather than
   relying on generic ImageNet-pretrained weights.
2. **Linear probe** — freeze the pretrained backbone, train a linear
   classifier head on top of its embeddings using the 9 class labels.
3. **Evaluation** — held-out CRC-VAL-HE-7K set (patient-disjoint): accuracy,
   per-class F1, confusion matrix.

**Backbone**: ViT-S/16. leJEPA supports ViT out of the box; Small is
realistic to fully pretrain in a single-GPU Colab/GCP session, unlike the
ViT-L/H configs in the leJEPA repo's published benchmarks which target
multi-GPU ImageNet-scale runs. Backbone size is a config value, not
hardcoded — can be scaled up later if more compute becomes available.

## Compute Strategy

- **Local (Mac, CPU/MPS)**: pipeline verification only. A small real subset
  (stratified sample, ~20-50 images/class ≈ 250-450 images) is downloaded
  and run through the *actual* pretrain + probe code paths for a couple of
  steps/epochs, to catch real data issues (corrupt files, label mapping,
  class imbalance) before spending cloud time. The full 24GB dataset is
  never downloaded to the Mac.
- **Cloud (GCP and/or Colab — platform-agnostic)**: full-scale SSL
  pretraining (100K images, real epoch count) and full linear probe /
  evaluation. Same pipeline code as local, driven by a different config file
  and a different `checkpoint_dir` (pointing at a GCS bucket or mounted
  Google Drive). Not committed to one specific cloud platform — the training
  code takes plain paths/configs so it runs the same way locally, in a Colab
  notebook, or on a GCP VM.

## Repo Structure

```
lejepa-tissue-classifier/
├── pyproject.toml              # uv-managed deps (torch, lejepa, etc.)
├── configs/
│   ├── local_smoke.yaml        # tiny subset, 1-2 epochs, CPU/MPS
│   ├── pretrain_ssl.yaml       # full leJEPA pretrain config (Colab/GCP)
│   └── linear_probe.yaml       # linear probe config
├── src/lejepa_tissue/
│   ├── data.py                 # dataset download, subset sampling, DataLoader/transforms
│   ├── pretrain.py             # leJEPA SIGReg pretraining loop
│   ├── probe.py                # linear probe training + eval
│   ├── metrics.py              # accuracy, per-class F1, confusion matrix
│   └── config.py               # config loading (shared across local/cloud)
├── scripts/
│   ├── download_data.sh        # fetches Zenodo record 1214456
│   ├── run_local_smoke.py      # entrypoint: local pipeline verification
│   └── run_cloud_train.py      # entrypoint: same code, cloud-scale config
├── notebooks/
│   └── colab_pretrain.ipynb    # thin wrapper calling src/ for Colab
├── tests/                      # pytest: data loading, model shapes, loss sanity
└── outputs/                    # checkpoints, metrics, confusion matrix, logs
```

`run_local_smoke.py` and `run_cloud_train.py` call the same
`src/lejepa_tissue` functions with different configs — no duplicated
pipeline logic between Mac verification and cloud training. The Colab
notebook is a thin shell that installs deps and calls into `src/`.

Environment/package management: **uv**.

## Data Flow & Error Handling

- `scripts/download_data.sh` pulls the relevant ZIPs from Zenodo, verifies
  checksums if provided, extracts into `data/raw/`. On the Mac, only a
  stratified subset is pulled (`--subset` flag) — the full download never
  happens locally.
- Class folder names map to canonical label IDs in one place (`data.py`),
  reused by both pretrain (labels ignored) and probe (labels used) stages.
- Corrupt/unreadable image files are skipped and logged, not allowed to
  crash a 100K-image epoch.
- Missing/empty class folders fail fast at startup, not mid-epoch.
- No CUDA available (Mac) auto-falls back to MPS/CPU with a smaller batch
  size via config, rather than erroring.
- Fixed seed in config; the local stratified subset is deterministic across
  runs, not re-randomized each time.

## Resilience for Cloud Runs

Long-running cloud SSL pretraining needs to survive disconnects
(Colab) and preemption (GCP preemptible VMs):

- **Periodic checkpointing**: model + optimizer + scheduler state + current
  epoch/step saved every N steps (config value) to `checkpoint_dir`, not
  only at the end. Checkpoint writes are atomic (write to temp file, then
  rename) to avoid corruption from a crash mid-write.
- **Resume-by-default**: `run_cloud_train.py` checks for an existing
  checkpoint at startup and resumes automatically unless `--fresh` is
  passed. Same entrypoint for first run and every subsequent resume.
- **Persistent storage**: `checkpoint_dir` points at a GCS bucket path or
  mounted Google Drive, not local VM/session disk, so a reclaimed VM or
  disconnected Colab runtime doesn't lose progress.
- **Logging**: metrics (loss, step, timestamp) appended to a log file in the
  same persistent location and printed to stdout, so progress is visible
  even if the process dies.
- **Explicitly out of scope**: no experiment-tracking service (e.g. W&B) and
  no automated retry/orchestration. If a run dies, it's restarted manually;
  the resume behavior makes that cheap, but nothing is self-healing.

## Testing & Verification

- `tests/` (pytest), no real dataset required: label-mapping correctness,
  config loading, model forward-pass shape checks, SIGReg loss returns a
  finite non-negative value, linear probe head output shape matches 9
  classes.
- **Local smoke test**: real small subset (~300 images) through the actual
  pretrain loop (few steps) and actual probe loop (few epochs) on CPU/MPS,
  confirming loss decreases and accuracy computes without crashing, in a
  couple of minutes. Gate before spending cloud GPU time.
- **Cloud run**: full pretrain + full linear probe, evaluated on the
  7,180-patch held-out CRC-VAL-HE-7K set, reporting accuracy / per-class F1
  / confusion matrix to `outputs/`.

## Out of Scope

- Training leJEPA's larger backbones (ViT-L/H, ConvNeXtV2-H) — deferred
  until/unless more compute is available.
- Comparing against an ImageNet-pretrained-only baseline (no domain SSL
  pretraining) — not part of this design; could be a follow-up.
- Experiment tracking service integration (W&B or similar).
- Automated cloud orchestration/retry beyond manual resume.
