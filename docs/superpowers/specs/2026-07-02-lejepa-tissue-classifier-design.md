# Tissue Classifier — Design

Date: 2026-07-02

## Goal

Build an image classifier for colorectal tissue histology patches by
fine-tuning an existing open pathology **vision foundation model**, trained
and evaluated on the NCT-CRC-HE-100K dataset (Zenodo record 1214456),
predicting the full 9-class tissue taxonomy (one of which is colorectal
adenocarcinoma epithelium — the "cancer" class).

**Note on objective**: NCT-CRC-HE-100K is a well-studied, fully labeled
benchmark whose held-out set (CRC-VAL-HE-7K) is already near-saturated —
plain supervised / ImageNet-transfer models reach 94.3-98.3% published
accuracy, and the best published result on this exact benchmark is a plain
supervised EfficientNet-B0 (97.7% single / 98.3% ensemble), not any SSL
method. Because the dataset is 100% labeled, self-supervised pretraining's
usual label-efficiency advantage does not apply here. The most efficient,
evidence-backed path to a strong classifier on this dataset is therefore to
**leverage and fine-tune an existing pathology foundation model** — cheap
(single-GPU, sub-hour linear probe), directly relevant to the modern
biomedical-imaging skill set, and not fighting a near-saturated benchmark
with an unproven from-scratch method. This project targets a good classifier
built efficiently, not a novel-SSL-method demonstration.

**Source of this direction**: this design was pivoted away from an earlier
from-scratch leJEPA (self-supervised SIGReg) pretraining plan on the basis of
the research brief at `outputs/jepa-vs-traditional-cv-ml.md` (and its
provenance record `outputs/jepa-vs-traditional-cv-ml.provenance.md`). That
brief found the benchmark near-saturated, found no evidence that leJEPA
outperforms simpler alternatives at this data/compute scale, and found that
open pathology foundation models are fine-tunable on a single GPU in
minutes-to-hours at comparable or better reported accuracy. leJEPA is not
"wrong" — it remains theoretically sound and has one real, untested
counter-argument (in-domain SSL beating generic transfer on visually-
specialized domains) — but it is not the evidence-backed choice for this
project's stated goal. See "Out of Scope" for how from-scratch leJEPA is now
positioned as a deferred alternative.

## Dataset

**NCT-CRC-HE-100K** (Zenodo 1214456):
- 100,000 H&E-stained colorectal tissue patches, 224×224px, 0.5 microns/pixel.
- 9 classes: adipose, background, debris, lymphocytes, mucus, smooth muscle,
  normal colon mucosa, cancer-associated stroma, colorectal adenocarcinoma
  epithelium.
- Macenko color-normalized version used for training (canonical benchmark
  protocol for this dataset).
- Separate **CRC-VAL-HE-7K** validation set (7,180 patches from different
  patients) used as the held-out test set — avoids patient-level leakage
  between train and eval.
- Non-normalized version of the 100K set exists but is not used in this
  design.
- CC-BY 4.0 licensed.

## Approach

The backbone is an off-the-shelf, pretrained pathology **foundation model** —
not trained here. The classifier is built on top of it in two stages of
increasing strength and cost.

**Backbone choice**: **Virchow (or GigaPath)** — both are **Apache 2.0**
licensed with no access restrictions, and both rank among the top-performing
pathology encoders in independent benchmarks. This matters: UNI (academic-
only license) and Phikon-v2 (non-commercial) impose access/usage restrictions
that a portfolio project should avoid, whereas Virchow and GigaPath can be
downloaded and used without institutional registration or a non-commercial
gate. Virchow is the default; GigaPath is a drop-in alternative behind the
same config value if Virchow's weights or tiling assumptions prove
inconvenient. The specific model is a config value, not hardcoded.

Pipeline:

1. **Linear probe (fast baseline)** — freeze the foundation-model backbone,
   extract patch embeddings once, and train a linear classifier head on top
   of them using the 9 class labels. This is the cheap, high-signal first
   result (~30-60 min on a single GPU, ~2-4GB VRAM) and establishes the
   floor. Because the backbone is frozen, embeddings can be pre-computed and
   cached, making probe training nearly instantaneous and trivially
   repeatable.
2. **LoRA fine-tune (stronger result)** — attach low-rank adapters to the
   backbone's attention/MLP projections and fine-tune them (backbone base
   weights frozen, adapters + head trained) on the 9-class labels. This is
   the stronger result (~2-4 hours on a single GPU) and is expected to close
   part of the gap between the frozen probe and a full fine-tune, at a small
   fraction of full-fine-tune cost. LoRA rank/alpha/target-modules are config
   values.
3. **Evaluation** — held-out CRC-VAL-HE-7K set (patient-disjoint): accuracy,
   per-class F1, confusion matrix. Both the linear probe (step 1) and the
   LoRA fine-tune (step 2) are evaluated identically on the same held-out set
   and reported side by side, so the incremental value of LoRA over the
   frozen probe is explicit.

Full fine-tuning of all backbone weights is deliberately not the primary
path (see "Out of Scope"): it is the most expensive option and, per the
research, does not clearly beat the plain-supervised baseline on this
already-near-saturated benchmark, so LoRA is the better cost/benefit stopping
point for this project.

### Preprocessing & augmentation strategy

Fine-tuning a foundation model needs far less bespoke augmentation design
than SSL pretraining did (there is no self-supervised predictive task to
define via augmentations). Two things must still be handled explicitly:

- **Match the backbone's expected input transform.** Each foundation model
  ships its own required preprocessing — input resolution, and (critically)
  the mean/std normalization constants its weights were trained with. Using
  the wrong normalization silently degrades embeddings. The backbone's own
  published transform is used verbatim for both embedding extraction and
  fine-tuning; this is treated as part of the backbone config, not
  re-invented. The NCT-CRC-HE-100K patches are already 224×224, which matches
  the common foundation-model tile size, so no re-tiling is needed.
- **Light, label-preserving train-time augmentation only.** For the linear
  probe, augmentation is optional (embeddings can be cached from a single
  clean forward pass). For LoRA fine-tuning, mild augmentation helps
  generalization: H&E tissue patches have no canonical orientation, so full
  dihedral augmentation (horizontal + vertical flips, 0/90/180/270°
  rotation) is valid and included. Because the data is already Macenko
  color-normalized, aggressive hue/saturation jitter is avoided (it would
  fight the normalization); only mild brightness/contrast jitter is used, as
  a config-controlled option (default: off). No multi-crop, no SSL-style
  global/local crop recipe — those were pretraining constructs and are gone.

### Backbone loading & caching

- The backbone is loaded from its published weights (e.g. via `timm` /
  Hugging Face Hub) and run in inference mode for the linear probe. Weight
  download happens once and is cached to a persistent path (see Compute
  Strategy) so cloud sessions don't re-download.
- For the linear probe, embeddings for the full 100K train set and the 7K
  eval set are extracted once and cached to disk; probe training then reads
  cached embeddings, not images. This makes the probe stage cheap and
  repeatable and is where most of the "compute" actually goes (a single
  forward pass over the data).

## Compute Strategy

- **Local (Mac, CPU/MPS)**: pipeline verification only. A small real subset
  (stratified sample, ~20-50 images/class ≈ 250-450 images) is downloaded and
  run through the *actual* backbone-load + embed + probe (and a few LoRA
  steps) code paths, to catch real data issues (corrupt files, label mapping,
  backbone-transform mismatch, class imbalance) before spending cloud time.
  The full 24GB dataset is never downloaded to the Mac. Note the backbone
  itself may be large enough that a single forward pass is slow on CPU/MPS —
  that's fine, since the local run only touches a few hundred images.
- **Cloud (GCP and/or Colab — platform-agnostic)**: full-scale embedding
  extraction over 100K images, full linear probe, and LoRA fine-tune +
  evaluation. Same pipeline code as local, driven by a different config file
  and a different `checkpoint_dir` / `cache_dir` (pointing at a GCS bucket or
  mounted Google Drive). Not committed to one specific cloud platform — the
  code takes plain paths/configs so it runs the same way locally, in a Colab
  notebook, or on a GCP VM. A single modest GPU (T4/L4 class) is sufficient.

### Compute budget

Fine-tuning a foundation model is dramatically cheaper than the from-scratch
SSL pretraining the earlier design assumed — there is no multi-GPU-hour
pretraining run. Concretely:

- **Linear probe**: ~30-60 min end-to-end on a single GPU, dominated by the
  one-time embedding extraction over the 100K set (probe training itself is
  seconds-to-minutes on cached embeddings). ~2-4GB VRAM.
- **LoRA fine-tune**: ~2-4 hours on a single GPU.

Both fit comfortably inside a single free-Colab / short-GCP session, so the
old "time-box to 8-12 GPU-hours across multiple resumable sessions" framing
is retired. The checkpoint/resume design (below) is kept, but now primarily
as insurance against a mid-run disconnect rather than as a mechanism for
stitching a very long training run across many sessions. Actual throughput
still depends on the allocated GPU (T4/L4/A100 differ), but the totals are
small enough that exact hardware is no longer a budgeting risk.

### Device & precision handling

At this much smaller compute scale the elaborate capability-based precision
selection is no longer necessary and is simplified:

- `configs/local_smoke.yaml`: `device: auto`, resolving to `mps` if available
  else `cpu` (Mac has no CUDA). Small batch size, fp32 — MPS/CPU mixed
  precision is unreliable and unnecessary at this tiny scale.
- `configs/linear_probe.yaml` and `configs/lora_finetune.yaml` (cloud):
  `device: auto`, resolving to `cuda` if available else falling back to
  CPU/MPS. Because the workloads are short (minutes to a few hours), a
  fall-through to CPU is an inconvenience, not the multi-day disaster it would
  have been for SSL pretraining — so hard-failing when CUDA is absent is no
  longer required. A startup log line records the resolved device so an
  accidental CPU run is at least visible.
- **Precision**: use fp16/bf16 autocast on CUDA when available, fp32
  otherwise, via a single `precision: auto` that prefers bf16 on GPUs that
  support it (compute capability ≥ 8.0) and fp16 on older GPUs (e.g. T4). This
  is the same idea as before but is now a minor speed optimization on a short
  run rather than a load-bearing decision, so it stays simple with a config
  override available. Head/LoRA optimizer defaults: AdamW, lr 1e-3 for the
  linear probe head, lr 1e-4 for LoRA adapters, cosine schedule — tuned during
  the local/cloud spot-check, not treated as fixed.

## Repo Structure

```
tissue-classifier/
├── pyproject.toml              # uv-managed deps (torch, timm, peft, etc.)
├── configs/
│   ├── local_smoke.yaml        # tiny subset, few steps, CPU/MPS
│   ├── linear_probe.yaml       # frozen-backbone linear probe (Colab/GCP)
│   └── lora_finetune.yaml      # LoRA fine-tune config (Colab/GCP)
├── src/tissue_classifier/
│   ├── data.py                 # dataset download, subset sampling, DataLoader/transforms
│   ├── foundation_model.py     # backbone loading (Virchow/GigaPath), transform, embedding extraction + cache
│   ├── probe.py                # linear probe training + eval on cached embeddings
│   ├── finetune.py             # LoRA fine-tune loop (adapters + head)
│   ├── metrics.py              # accuracy, per-class F1, confusion matrix
│   └── config.py               # config loading (shared across local/cloud)
├── scripts/
│   ├── download_data.sh        # fetches Zenodo record 1214456
│   ├── run_local_smoke.py      # entrypoint: local pipeline verification
│   └── run_cloud_train.py      # entrypoint: same code, cloud-scale config
├── notebooks/
│   └── colab_finetune.ipynb    # thin wrapper calling src/ for Colab
├── tests/                      # pytest: data loading, model shapes, metric sanity
└── outputs/                    # checkpoints, embeddings cache, metrics, confusion matrix, logs
```

`run_local_smoke.py` and `run_cloud_train.py` call the same
`src/tissue_classifier` functions with different configs — no duplicated
pipeline logic between Mac verification and cloud training. The Colab notebook
is a thin shell that installs deps and calls into `src/`.

Environment/package management: **uv**.

## Data Flow & Error Handling

- `scripts/download_data.sh` pulls the relevant ZIPs from Zenodo, verifies
  checksums if provided, extracts into `data/raw/`. On the Mac, only a
  stratified subset is pulled (`--subset` flag) — the full download never
  happens locally.
- Class folder names map to canonical label IDs in one place (`data.py`),
  reused by both the linear-probe and LoRA-fine-tune stages.
- Corrupt/unreadable image files are skipped and logged, not allowed to crash
  a 100K-image embedding pass.
- Missing/empty class folders fail fast at startup, not mid-pass.
- **Backbone-transform consistency check**: at data-load time, confirm the
  preprocessing (resolution + normalization constants) applied to both the
  100K train set and the CRC-VAL-HE-7K test set matches the backbone's
  published transform. A mismatch here silently degrades embeddings without
  any error, so the resolved transform is logged and asserted identical across
  train/eval.
- **Normalization consistency check**: verify that CRC-VAL-HE-7K was Macenko-
  normalized compatibly with the training set (spot-check per-channel color
  statistics between a train and val sample if Zenodo doesn't document it),
  and note the finding in the eval output rather than assuming.
- Device selection follows the simplified rules in "Device & precision
  handling" above and logs the resolved device at startup.
- Fixed seed in config; the local stratified subset is deterministic across
  runs, not re-randomized each time.

## Resilience for Cloud Runs

Cloud runs are now short (minutes to a few hours), but they can still be
interrupted by a Colab disconnect or a GCP preemption, so the resume design
is kept — mainly to avoid re-running the one-time 100K-image embedding
extraction:

- **Embedding cache is the primary durable artifact**: extracted embeddings
  for the 100K train set and 7K eval set are written to `cache_dir` in
  persistent storage. If a run dies after extraction, the probe/LoRA stages
  resume from the cache rather than re-embedding.
- **Periodic checkpointing (LoRA)**: adapter + head + optimizer + scheduler
  state + current epoch/step saved every N steps (config value) to
  `checkpoint_dir`. Checkpoint writes are atomic (write to temp file, then
  rename). The linear probe is short enough that only its final head is saved.
- **Resume-by-default**: `run_cloud_train.py` checks for an existing embedding
  cache and checkpoint at startup and resumes automatically unless `--fresh`
  is passed. Same entrypoint for first run and every resume.
- **Persistent storage**: `checkpoint_dir` and `cache_dir` point at a GCS
  bucket path or mounted Google Drive, not local VM/session disk, so a
  reclaimed VM or disconnected Colab runtime doesn't lose progress.
- **Logging**: metrics (loss, step, timestamp) appended to a log file in the
  same persistent location and printed to stdout.
- **Explicitly out of scope**: no experiment-tracking service (e.g. W&B) and
  no automated retry/orchestration. If a run dies, it's restarted manually;
  the resume behavior makes that cheap, but nothing is self-healing.

## Testing & Verification

- `tests/` (pytest), no real dataset required: label-mapping correctness,
  config loading, backbone forward-pass / embedding shape checks, linear probe
  head output shape matches 9 classes, LoRA adapter attaches and produces
  finite gradients, metric functions return sane values on toy inputs.
- **Local smoke test**: real small subset (~300 images) through the actual
  backbone-load + embed + probe (few epochs) and a few LoRA steps on CPU/MPS,
  confirming embeddings extract, loss decreases, and accuracy computes without
  crashing, in a few minutes. Gate before spending cloud GPU time.
- **Cloud run**: linear probe first (cheap, validates the embedding + eval
  path end-to-end), then LoRA fine-tune. Both evaluated on the 7,180-patch
  held-out CRC-VAL-HE-7K set, reporting accuracy / per-class F1 / confusion
  matrix for each to `outputs/`, side by side, so the incremental value of
  LoRA over the frozen probe is explicit.

## Out of Scope

- **From-scratch leJEPA (SIGReg) SSL pretraining of a backbone** — this was
  the original design's core approach; it is deferred per the research brief
  (`outputs/jepa-vs-traditional-cv-ml.md`), which found it not evidence-backed
  as the best choice for this project's goal (near-saturated benchmark, no
  label-scarcity advantage on a fully-labeled dataset, unproven on
  histology). It remains a legitimate *future* experiment — specifically to
  test the one unrefuted counter-argument, that in-domain SSL can beat generic
  transfer on visually-specialized domains — but only as a deliberate
  SSL-method demonstration, not as the efficient path to a good classifier.
- **Full fine-tuning of all backbone weights** (as opposed to a frozen probe
  or LoRA) — more expensive and does not clearly beat the plain-supervised
  baseline on this already-near-saturated benchmark; could be a follow-up.
- **Pretraining or comparing multiple foundation models head-to-head** — one
  backbone (Virchow, GigaPath as fallback) is fine-tuned; a multi-encoder
  benchmark is out of scope.
- **Exhaustive LoRA / augmentation hyperparameter tuning** — the LoRA rank,
  learning rates, and augmentation settings are spot-checked against a short
  local run, not grid-searched.
- **Experiment tracking service integration** (W&B or similar).
- **Automated cloud orchestration/retry beyond manual resume.**
