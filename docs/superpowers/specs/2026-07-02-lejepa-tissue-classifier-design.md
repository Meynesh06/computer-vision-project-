# leJEPA Tissue Classifier — Design

Date: 2026-07-02

## Goal

Build an image classifier for colorectal tissue histology patches using leJEPA
(a self-supervised joint-embedding predictive architecture using SIGReg loss)
as the representation-learning backbone, trained on the NCT-CRC-HE-100K
dataset (Zenodo record 1214456), predicting the full 9-class tissue taxonomy
(one of which is colorectal adenocarcinoma epithelium — the "cancer" class).

**Note on objective**: NCT-CRC-HE-100K is a well-studied, fully labeled
benchmark with published *supervised* baselines already reaching ~94-96%+ on
CRC-VAL-HE-7K. A from-scratch SSL pretrain + linear probe is not the most
direct path to the highest possible accuracy on this specific dataset — a
supervised fine-tune would likely get there faster. The point of this project
is specifically to exercise leJEPA as an SSL method on a real, non-toy image
domain and see what it buys over off-the-shelf ImageNet features (see the
ImageNet baseline in the Approach section), not to chase the top published
accuracy number.

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

0. **ImageNet baseline (reference point)** — before domain SSL pretraining,
   train a linear probe on leJEPA's existing ImageNet-pretrained ViT-S/16
   checkpoint (frozen, no fine-tuning), on the same labels and the same
   held-out CRC-VAL-HE-7K test set used in step 3. This is cheap (no
   pretraining required) and gives a concrete number to compare the domain
   SSL-pretrained backbone against — without it there's no way to tell
   whether pretraining on tissue images actually helped.
1. **Self-supervised pretraining** — train a backbone with leJEPA's SIGReg
   loss on the *unlabeled* NCT-CRC-HE-100K patches (labels ignored at this
   stage). Domain-specific pretraining on histopathology images rather than
   relying on generic ImageNet-pretrained weights.
2. **Linear probe** — freeze the pretrained backbone, train a linear
   classifier head on top of its embeddings using the 9 class labels.
3. **Evaluation** — held-out CRC-VAL-HE-7K set (patient-disjoint): accuracy,
   per-class F1, confusion matrix. Step 0's baseline and step 2's
   domain-pretrained probe are evaluated identically and reported side by
   side.

**Backbone**: ViT-S/16. leJEPA supports ViT out of the box; Small is
realistic to fully pretrain in a single-GPU Colab/GCP session, unlike the
ViT-L/H configs in the leJEPA repo's published benchmarks which target
multi-GPU ImageNet-scale runs. Backbone size is a config value, not
hardcoded — can be scaled up later if more compute becomes available.

### Augmentation strategy (SSL pretraining)

The predictive task in any JEPA is defined by its augmentations, so this
needs to be explicit rather than left to defaults — and two things make
generic (natural-image) augmentation recipes a bad fit here:

- The training data is already Macenko color-normalized, and standard
  aggressive color jitter (hue/saturation shifts) would fight that
  normalization and reintroduce the stain variation it was meant to remove.
  Color augmentation is therefore limited to mild brightness/contrast jitter
  only, no hue/saturation jitter, as a config-controlled option (default:
  off, spot-check on before committing to the full cloud run).
- H&E tissue patches have no canonical orientation (unlike natural images,
  where "upright" is meaningful) — full dihedral augmentation (horizontal
  and vertical flips, 0/90/180/270° rotation) is valid here and should be
  included, whereas it would usually be skipped for natural-image SSL.
- Multi-crop follows leJEPA's published recipe as a starting point (2 global
  crops + 6 local crops, global at 224×224, local at 98×98, random-resized-
  crop scale ranges as config values), with the flip/rotation and limited
  color jitter above applied to every crop.
- This is a starting recipe, not a guarantee — augmentation strength is a
  config value and is expected to need a short spot-check (does the SIGReg
  loss actually decrease over a few hundred steps on the local real-data
  subset?) before committing to a full cloud run.

### SIGReg batch size

leJEPA's SIGReg loss estimates distributional statistics from each batch, so
very small batches (likely on a memory-constrained single cloud GPU) may
degrade it. The pretraining config exposes both a per-step micro-batch size
(bounded by GPU memory) and gradient accumulation to reach a larger
*effective* batch size. The effective batch size leJEPA needs for SIGReg to
work well is not assumed up front — it should be spot-checked against the
leJEPA repo's own recommendations/ablations during implementation, and the
config default set from that, rather than guessed.

## Compute Strategy

- **Local (Mac, CPU/MPS)**: pipeline verification only. A small real subset
  (stratified sample, ~20-50 images/class ≈ 250-450 images) is downloaded
  and run through the *actual* pretrain + probe code paths for a couple of
  steps/epochs, to catch real data issues (corrupt files, label mapping,
  class imbalance) before spending cloud time. The full 24GB dataset is
  never downloaded to the Mac.
- **Cloud (GCP and/or Colab — platform-agnostic)**: full-scale SSL
  pretraining (100K images) and full linear probe / evaluation. Same
  pipeline code as local, driven by a different config file and a different
  `checkpoint_dir` (pointing at a GCS bucket or mounted Google Drive). Not
  committed to one specific cloud platform — the training code takes plain
  paths/configs so it runs the same way locally, in a Colab notebook, or on
  a GCP VM.

### Compute budget

Actual training throughput depends on whatever GPU we end up with (T4, L4,
A100 all show up on Colab/GCP with very different speeds), which isn't known
until implementation. Rather than fixing an epoch count up front that might
be wildly infeasible, pretraining is **time-boxed**: target ~8-12 hours of
cumulative GPU time, run across multiple resumable sessions if needed
(Colab free/pro sessions cap around 12h/24h; the checkpoint/resume design in
"Resilience for Cloud Runs" already covers picking back up across sessions).
Whatever epoch count that time budget actually buys on the real hardware is
what gets used and reported — not decided in advance. If the SSL loss is
clearly still decreasing fast at the budget cutoff, that's a signal to
extend the budget rather than stop on a technicality.

### Device & precision handling (per environment)

Mac and cloud environments require different device/precision settings, and
the config for each should say so explicitly rather than relying on generic
auto-detection everywhere:

- `configs/local_smoke.yaml`: `device: auto`, resolving to `mps` if
  available else `cpu` (Mac has no CUDA). Small batch size, fp32 — MPS/CPU
  support for bf16 mixed precision is unreliable, so mixed precision is not
  used locally.
- `configs/pretrain_ssl.yaml` and `configs/linear_probe.yaml` (cloud):
  `device: cuda`, required. If CUDA is requested but unavailable at
  startup, the run fails fast with a clear error rather than silently
  falling back to CPU — a silent fallback would turn a multi-hour GPU job
  into a multi-day CPU one without anyone noticing until much later.
- **Precision is selected by GPU capability, not hardcoded.** leJEPA's
  published config uses bfloat16, but Colab frequently allocates T4 GPUs
  (Turing architecture), which do not support bf16. `precision: auto`
  resolves to bf16 on GPUs with compute capability ≥ 8.0 (Ampere and newer
  — A100, L4, etc.) and falls back to fp16 otherwise (e.g. T4), with an
  explicit override available in config if a specific run needs to force
  one or the other. Optimizer/schedule otherwise follow leJEPA's published
  configuration (AdamW, lr 5e-4, cosine schedule).

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
- **Normalization consistency check**: at data-load time, verify that
  CRC-VAL-HE-7K (the held-out test set) was normalized with a compatible
  Macenko reference to the training set — a mismatch here would silently
  degrade eval numbers without any error. If Zenodo doesn't document this
  explicitly, spot-check it visually/statistically (e.g. compare per-channel
  color statistics between a train and val sample) before trusting eval
  results, and note the finding in the eval output rather than assuming.
- Device selection follows the per-environment rules in "Device & precision
  handling" above: local configs resolve to MPS/CPU automatically, cloud
  configs require CUDA and fail fast if it's missing.
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
- **Cloud run**: ImageNet-baseline linear probe (step 0) first, since it's
  cheap and validates the eval path end-to-end before the expensive
  pretrain; then full SSL pretrain + full domain-pretrained linear probe
  (steps 1-2). Both probes evaluated on the 7,180-patch held-out
  CRC-VAL-HE-7K set, reporting accuracy / per-class F1 / confusion matrix
  for each to `outputs/`, side by side.

## Out of Scope

- Training leJEPA's larger backbones (ViT-L/H, ConvNeXtV2-H) — deferred
  until/unless more compute is available.
- Full supervised fine-tuning of the backbone (as opposed to a frozen linear
  probe) — could close more of the gap to published supervised baselines,
  but is a different exercise than evaluating SSL representation quality;
  could be a follow-up.
- Exhaustive SIGReg/augmentation hyperparameter tuning — the augmentation
  and batch-size settings above are spot-checked against a short local run,
  not grid-searched.
- Experiment tracking service integration (W&B or similar).
- Automated cloud orchestration/retry beyond manual resume.
