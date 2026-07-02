# Research Plan: JEPA vs. Traditional CV/ML for Tissue Classification

Slug: `jepa-vs-traditional-cv-ml`
Date: 2026-07-02
Requested by: user, as a soundness check on the leJEPA-based design in
`docs/superpowers/specs/2026-07-02-lejepa-tissue-classifier-design.md`.

## Objective

Determine whether using JEPA-style self-supervised pretraining (specifically
leJEPA, per the existing project spec) is a *sound* choice over traditional
computer vision / machine learning approaches for the NCT-CRC-HE-100K
9-class tissue classification task — and if sound, how good it actually is
(quantitatively, against alternatives).

**Project framing**: this is a portfolio project supporting a job
application to Aganitha (Computer Vision Engineer — Biomedical Imaging,
entry-to-mid level). The JD explicitly asks for "leverage and fine-tune
modern deep learning and vision foundation models for biomedical imaging"
— not train-your-own-SSL-from-scratch specifically. The bar for "good
enough" here is a defensible, honest engineering judgment call (including
correctly concluding leJEPA is *not* the best choice, if that's what the
evidence says) — not exhaustive academic literature coverage. A verdict
that shows real critical thinking about tradeoffs is more valuable for this
purpose than a one-sided justification of a method already chosen.

"Traditional CV/ML" and the relevant alternatives are scoped to four
baselines, since each has a different soundness argument against JEPA:
- (a) classical feature-engineering + classical ML (hand-crafted texture/
  color features + SVM/Random Forest/etc.) — the pre-deep-learning approach.
- (b) standard supervised deep learning (CNN/ViT trained end-to-end with
  labels, no SSL pretraining) — the modern default.
- (c) ImageNet-pretrained transfer learning + linear probe/fine-tune (no
  domain SSL pretraining) — cheap, standard modern practice, and the most
  likely thing to beat leJEPA on a fully-labeled dataset this size.
- (d) an existing pretrained pathology foundation model (e.g. UNI, Phikon,
  CTransPath, Virchow — existence/openness/accessibility to be confirmed by
  research, not assumed), fine-tuned or linear-probed — this is the option
  that most directly matches the JD's "leverage and fine-tune foundation
  models" language.

## Key Questions

1. **Theoretical soundness**: What is the mechanism-level case for JEPA
   (and SIGReg specifically) as a representation-learning method — why is
   it claimed to avoid representation collapse, and how does that compare
   mechanistically to contrastive SSL (SimCLR), self-distillation SSL
   (DINO, BYOL), and masked-image-modeling SSL (MAE)? Is the theoretical
   basis for isotropic Gaussian regularization well-established or novel/
   unproven?
2. **General empirical standing**: On published image classification
   benchmarks (ImageNet and others), how does JEPA/leJEPA's linear-probe
   and fine-tuned performance compare to supervised training and other SSL
   methods (DINO/DINOv2, MAE, SimCLR, BYOL) at comparable model/compute
   scale? If leJEPA itself has no independent (non-author) benchmark
   numbers or no results outside natural images, say so explicitly rather
   than substituting numbers from adjacent JEPA-family methods without
   flagging the substitution.
3. **SSL benefit vs. scale/compute regime**: What does the literature say
   about *when* SSL pretraining beats supervised training — as a function
   of unlabeled data scale, label scarcity, and compute budget? Most
   documented SSL advantages appear in large-unlabeled-data or
   label-scarce regimes. NCT-CRC-HE-100K is fully labeled (100K images,
   9 clean classes) — does this project's regime (~100K images, all
   labeled, ~8-12 GPU-hours, ViT-S/16) fall inside or outside the regime
   where SSL typically shows benefit? This must be answered as a
   regime/pattern question, not as a precise number for our exact setup
   (literature cannot give that; a specific-number answer here should be
   treated as invented and rejected).
4. **Null hypothesis check (saturation)**: Do published supervised or
   ImageNet-transfer baselines already achieve very high accuracy on
   NCT-CRC-HE-100K / CRC-VAL-HE-7K (this needs a real number, not an
   assumption)? If accuracy is already near-saturated (e.g. in the
   mid-to-high 90s), there may be little headroom left for SSL pretraining
   to improve on, which would directly weaken the case for leJEPA
   regardless of its general merits.
5. **Histopathology-specific evidence**: What does the literature say about
   SSL pretraining (JEPA-family specifically if it exists, SSL broadly
   otherwise) vs. supervised training vs. classical ML for histopathology
   classification — ideally on NCT-CRC-HE-100K or closely related
   colorectal/H&E datasets?
6. **Foundation-model alternative**: Do open/accessible pretrained
   pathology foundation models exist that could realistically be
   fine-tuned or linear-probed on this dataset within the same compute
   budget? How would that option likely compare to training leJEPA from
   scratch, both technically and in terms of matching what the target job
   description asks for?
7. **Known risks/caveats**: What failure modes, data/compute requirements,
   or sensitivities (batch size, augmentation policy, dataset scale) are
   documented for JEPA/SIGReg that could undermine soundness on our
   specific constrained setup? Are there credible skeptical/critical takes
   on SSL-for-classification hype worth weighing?
8. **Verdict inputs**: Given 1-7, what would a defensible verdict look like
   — is JEPA/leJEPA a sound choice here, how does it compare to all four
   baselines (not just supervised-from-scratch), and how much (if any)
   quantitative benefit should we expect given this project's actual scale
   and compute constraints?

## Falsification Criteria

State upfront what would count as evidence *against* leJEPA being the right
choice, so the verdict isn't structured to only confirm the existing spec:
- Published supervised or transfer-learning baselines on this exact dataset
  already near ceiling accuracy (little headroom for SSL to matter).
- No independent evidence that JEPA/SIGReg specifically (vs. SSL broadly)
  outperforms simpler alternatives at this data/compute scale.
- An existing open pathology foundation model achieves comparable or better
  accuracy for a fraction of the compute/engineering effort.
- Evidence that SIGReg's batch-size or augmentation sensitivities make it
  unreliable at the batch sizes actually achievable on a single Colab/GCP
  GPU.

## Evidence Needed

- Primary leJEPA source (paper/preprint/repo docs) for SIGReg mechanism and
  published benchmark numbers — noting whether any numbers are independent
  or author-only, and whether histopathology results exist at all.
- JEPA family papers (I-JEPA, V-JEPA) for the broader JEPA lineage's
  claims and results.
- Comparable SSL method papers/benchmarks (DINO/DINOv2, MAE, SimCLR, BYOL)
  for general standing comparison, and literature on SSL-benefit-vs-scale
  more generally (label-scarce vs. fully-labeled regimes).
- Published accuracy numbers for supervised and transfer-learning baselines
  specifically on NCT-CRC-HE-100K / CRC-VAL-HE-7K (for the saturation
  check).
- Histopathology/medical-imaging SSL literature — searches targeting
  NCT-CRC-HE-100K, CRC-VAL-HE-7K, colorectal histology classification,
  H&E tissue classification with self-supervised pretraining, and
  classical (hand-crafted feature) baselines in digital pathology.
- Pathology foundation model landscape: what exists (e.g. UNI, Phikon,
  CTransPath, Virchow, or others found during search), licensing/openness,
  and typical fine-tuning/linear-probe cost and results.
- Any critical/skeptical secondary sources on SSL vs. supervised learning
  soundness claims (to avoid one-sided evidence).

## Scale Decision

Complex multi-domain research — spans ML theory, general CV benchmarks, a
scale/regime question, a domain-specific literature angle, and a
foundation-model landscape scan, and needs to land on a quantitative,
job-relevant verdict. Scale: **6 researcher subagents**, run concurrently.
Synthesis is done by the lead (not delegated). Verification and adversarial
review are separate subagent passes with stronger models.

Model tiering: subagents doing theory-dense reading or judgment-heavy
regime/saturation analysis are upgraded off haiku, since a weak model
producing confidently-wrong technical notes there would poison downstream
synthesis in a way the later review pass cannot fully recover from.
Straightforward search-and-collect tasks stay on haiku for cost.

## Task Ledger

| ID | Owner | Task | Model | Output | Status |
|----|-------|------|-------|--------|--------|
| T1 | researcher subagent | JEPA/leJEPA mechanism & theoretical soundness (Q1) | sonnet | `outputs/jepa-vs-traditional-cv-ml-research-theory.md` | in_progress |
| T2 | researcher subagent | General CV benchmark standing: JEPA/leJEPA vs. supervised vs. other SSL methods (Q2) | haiku | `outputs/jepa-vs-traditional-cv-ml-research-benchmarks.md` | in_progress |
| T3 | researcher subagent | SSL-benefit-vs-scale/compute regime + saturation/null-hypothesis check on NCT-CRC-HE-100K published accuracies (Q3, Q4) | sonnet | `outputs/jepa-vs-traditional-cv-ml-research-regime.md` | in_progress |
| T4 | researcher subagent | Histopathology-specific SSL/classical-ML evidence (Q5) | sonnet | `outputs/jepa-vs-traditional-cv-ml-research-histopath.md` | in_progress |
| T5 | researcher subagent | Pathology foundation model landscape + JD-relevance (Q6) | haiku | `outputs/jepa-vs-traditional-cv-ml-research-foundation-models.md` | in_progress |
| T6 | researcher subagent | Risks/caveats/critiques of JEPA-style SSL and SSL-for-classification claims generally (Q7) | haiku | `outputs/jepa-vs-traditional-cv-ml-research-risks.md` | in_progress |
| T7 | lead (me) | Draft synthesis from T1-T6, addressing Q8 and the falsification criteria explicitly | sonnet (lead model) | `outputs/.drafts/jepa-vs-traditional-cv-ml-draft.md` | complete |
| T8 | verifier subagent | Verify citations/URLs, add inline citations | sonnet | `outputs/.drafts/jepa-vs-traditional-cv-ml-cited.md` | complete — found 1 fabricated claim, confirmed 4 suspicious arXiv IDs as real |
| T9 | reviewer subagent | Adversarial review: unsupported claims, single-source critical claims, overstated confidence, check the verdict isn't one-sided | opus | `outputs/jepa-vs-traditional-cv-ml-verification.md` | complete — 2 FATAL, 3 MAJOR, 5 MINOR findings |
| T10 | lead (me) | Fix FATAL issues, deliver final + provenance | — | `outputs/jepa-vs-traditional-cv-ml.md` + `.provenance.md` | complete — all FATAL/MAJOR/MINOR fixed, verified on disk via grep, delivered |

## Verification Log

(To be filled as research proceeds — records what was checked and how.)

## Decision Log

- 2026-07-02: Scoped "traditional CV/ML" as two baselines initially
  (classical feature+ML, supervised-from-scratch).
- 2026-07-02: Chose deep-research over source-comparison or literature-review
  skills — the question requires theory + cross-domain benchmarks + a
  domain-specific literature angle, not just a comparison matrix or a pure
  paper survey.
- 2026-07-02: Ran an independent fable-subagent review of this plan before
  execution. Findings applied:
  - Upgraded T1 and the histopathology/regime tasks off haiku to sonnet
    (blocking finding: theory-dense and judgment-heavy tasks need a
    stronger model or they poison downstream synthesis).
  - Reframed the quantitative question (formerly Q5) to ask about
    scale/compute regimes rather than inviting a false-precision number for
    our exact setup.
  - Added an explicit null-hypothesis / saturation-check question (does
    supervised/transfer already achieve near-ceiling accuracy on this
    dataset, leaving no headroom for SSL) as a first-class question rather
    than a caveat.
  - Added a Falsification Criteria section so the plan isn't structured to
    only confirm the already-chosen leJEPA approach.
  - Added ImageNet-transfer and pathology-foundation-model fine-tuning as
    explicit baselines (was previously only 2 baselines).
- 2026-07-02: Reframed project objective around job-application context
  (Aganitha Computer Vision Engineer — Biomedical Imaging role). The JD
  asks for "leverage and fine-tune foundation models," which is not the
  same claim as "train your own SSL backbone from scratch" — added a
  dedicated researcher task (T5) on the pathology foundation model
  landscape as a fourth baseline, directly relevant to what the target
  role asks for. Bar for "good enough" recalibrated: a defensible,
  evidence-based judgment call (including a verdict against leJEPA, if
  warranted) demonstrates stronger engineering judgment for this purpose
  than exhaustive one-sided literature coverage.
