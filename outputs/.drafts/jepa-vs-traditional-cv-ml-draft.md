# Is leJEPA a Sound Choice for Tissue Classification? A Soundness Check

## Executive Summary

**Verdict: leJEPA is theoretically plausible but not a well-evidenced choice for this specific project, and it is very likely the wrong engineering choice given the project's actual constraints and goals.**

Three independent findings converge on this conclusion:

1. **The target benchmark is already near-saturated.** NCT-CRC-HE-100K → CRC-VAL-HE-7K, the exact dataset this project uses, already sees 94.3% accuracy from a plain ImageNet-transfer VGG19 (the dataset's own creators, 2019) and 97.7-98.3% from a purely supervised EfficientNet-B0 (2024) — a result that beats generic SSL methods (iBOT, DINO) and even a pathology-domain-SSL model (CTransPath) on this same benchmark. The one study that pretrained SSL directly on this dataset got *lower* accuracy (~90-91.5%) than supervised baselines at full label availability. SSL's classical advantage — learning from unlabeled data beyond what's labeled — does not apply here, because the dataset is 100% labeled.
2. **A faster, better-evidenced alternative already exists and matches the job description more directly.** At least nine open pathology foundation models (Virchow, UNI, GigaPath, Phikon-v2, H-optimus-0, etc.) are pretrained on 100M-1.4B histology images and achieve 95-97%+ accuracy on this exact dataset via linear probing (~30 min) or LoRA fine-tuning (~3-4 hours) on a single GPU — versus an estimated 24-72 GPU-hours to pretrain leJEPA from scratch. The target job description asks explicitly for the ability to "leverage and fine-tune modern deep learning and vision foundation models," which is a closer match to this path than training a from-scratch SSL backbone.
3. **leJEPA itself is unproven outside natural images.** It is an ~8-month-old method (as of this research) with zero independent reproduction and zero published medical/histopathology benchmarks. Its compute recipe (batch ≥128, 100 epochs on ImageNet-scale data) doesn't cleanly fit this project's ~8-12 GPU-hour, ~100K-image budget, and a GitHub issue reports a discrepancy between the paper's described loss and the repo's actual implementation.

None of this means leJEPA is *unsound in theory* — its collapse-avoidance mechanism is a plausible, mathematically-motivated extension of an established SSL sub-paradigm (see Q1). It means the case for using it *on this specific project* is weak: the conditions under which SSL demonstrably helps (label scarcity, or a domain far from what transfer sources were trained on) don't clearly hold here, and cheaper, better-evidenced, more job-relevant alternatives exist.

**Recommended path**: fine-tune (linear probe or LoRA) an existing open pathology foundation model — Virchow, GigaPath, or UNI — rather than pretraining leJEPA from scratch. This is faster, better-evidenced on this exact dataset, and more directly demonstrates what the target role asks for. If the project's real goal includes *demonstrating SSL engineering competence* rather than *maximizing classification accuracy*, leJEPA remains defensible as a deliberate, clearly-labeled exercise — but that framing should be explicit, not implied.

---

## Q1: Is leJEPA/SIGReg theoretically sound?

**Yes, plausibly — but the claimed novelty is more modest than the paper's framing suggests, and the deepest theoretical claims are unverified from primary sources by this research.**

SIGReg regularizes the marginal embedding distribution toward an isotropic Gaussian, using a Cramér-Wold random-projection + Epps-Pulley univariate normality test. The authors (Balestriero & LeCun, arXiv:2511.08544, Nov 2025) claim this single distributional constraint structurally prevents representation collapse (which they characterize as anisotropy) without needing negative pairs (contrastive methods like SimCLR), teacher-student/stop-gradient asymmetry (DINO, BYOL), or pixel reconstruction (MAE).

This "regularize the embedding distribution directly, skip negatives/stop-gradient/teacher-student" strategy is **not new** — VICReg (2021) and Barlow Twins (2021), both roughly four years older and one (VICReg) sharing an author with leJEPA, already established this general paradigm by constraining the embedding covariance matrix. leJEPA's specific claimed contribution is (a) a theoretical derivation for *why isotropic Gaussian specifically* — not just "high variance, low covariance" — is risk-optimal, and (b) matching the full distributional shape (via SIGReg) rather than just the first two moments (what VICReg/Barlow Twins do). No independent, non-LeCun-affiliated source was found benchmarking SIGReg head-to-head against VICReg/Barlow Twins under matched compute, so whether the full-distribution match provides a *practical* accuracy advantage over the simpler second-moment methods is unverified.

The one independent-ish follow-up paper found ("When Does LeJEPA Learn a World Model?") is largely confirmatory but flags a real limitation: the isotropic-Gaussian-optimality proof requires the latent variables to follow a Gaussian process, which the paper itself states is "unknowable from observations alone." This research also could not directly parse the leJEPA paper's mathematical proofs (PDF fetch returned undecoded binary data); the mechanism description above is corroborated across two independent secondary sources but not verified against the primary theorem text.

**Bottom line**: theoretically plausible and consistent with an established SSL sub-paradigm, not a red flag — but "provably avoids collapse" should be read as *well-motivated*, not as an independently-audited guarantee.

## Q2: How does leJEPA compare on general CV benchmarks?

leJEPA ViT-H reaches 79.0% ImageNet-1K linear-probe accuracy — competitive with iBOT (79.5%) and above DINO (78.2%) at similar ImageNet-1K-only pretraining scale, but below supervised ViT-B/16 (81.8%) and well below DINOv2 (86.3%, though DINOv2 trains on 142M external images, not a fair comparison). **All of leJEPA's numbers are author-reported; no independent reproduction was found**, and it is approximately 8 months old as of this research. It has architectural breadth (validated across 60+ architectures) as a genuine strength. It has **no published medical or histopathology benchmark of any kind** — every fine-grained or domain-specific result reported is on natural-image datasets (CIFAR, DTD, Aircraft, Galaxy10).

One genuinely relevant data point: on Galaxy10 (a small, ~11K-image, visually-specialized domain), leJEPA pretrained in-domain beat transfer learning from much larger foundation models (DINOv2/v3) by 2-5 percentage points. This is suggestive for histology (also visually unlike ImageNet) but has not been tested on histology in any source found — see Q3/Q5.

## Q3 / Q4: Does SSL's advantage apply to this project's regime, and is there headroom left?

This is the decisive question, and the evidence is unusually direct because it's measured on the *exact* dataset pairing this project uses.

**General pattern (Q3):** across multiple independent sources (Newell & Deng, CVPR 2020; SimCLRv2; a histopathology-specific study using NCT-CRC-HE-100K itself), SSL's advantage over supervised training tracks *label scarcity relative to available data* — not raw data volume or compute budget. The advantage shrinks toward zero, and can vanish entirely, as the labeled fraction approaches 100%. **NCT-CRC-HE-100K is fully labeled** — pretraining on it via leJEPA would not unlock any unlabeled data beyond what's already usable for supervised training, which places this project squarely in the regime where SSL's classical value proposition doesn't apply. No literature source supports assigning a specific quantitative benefit number to this project's exact configuration (ViT-S/16, ~100K images, ~8-12 GPU-hours) — any such number, wherever stated, should be treated as an unsupported estimate, not a literature-backed figure.

**Saturation check (Q4), the most important finding of this research:**

| Model | Approach | CRC-VAL-HE-7K accuracy | Source |
|---|---|---|---|
| VGG19 | ImageNet-transfer (2019, dataset's own creators) | 94.3-94.4% | Kather et al. 2019, *PLOS Medicine* |
| iBOT (ViT-L) | Generic SSL-pretrained | 95.8% | Ignatov & Malivenko 2024 |
| DINO (ViT) | Generic SSL-pretrained | 95.9% | Ignatov & Malivenko 2024 |
| CTransPath (Swin) | Pathology-domain SSL-pretrained | 96.5% | Ignatov & Malivenko 2024 |
| **EfficientNet-B0** | **Plain supervised, ImageNet-transfer** | **97.7% (single) / 98.3% (ensemble)** | **Ignatov & Malivenko 2024 — best published result** |
| SimCLR/SimSiam (frozen) | SSL pretrained directly on NCT-CRC-HE-100K, linear probe | ~89-90.5% | Voigt et al. 2023 |
| SimSiam (fine-tuned) | SSL pretrained directly on NCT-CRC-HE-100K, fine-tuned | ~91.5% | Voigt et al. 2023 |

The best published result on this exact benchmark is a **plain supervised, ImageNet-pretrained EfficientNet-B0** — not an SSL method — and it outperforms every SSL method in the same paper's comparison table, including a pathology-domain-SSL model. The one study that pretrained SSL directly on this exact dataset (Voigt et al. 2023, closest methodological match to this project's plan) reached only ~90-91.5% at full label availability — *below* both the 2019 and 2024 supervised numbers — and explicitly found the SSL advantage over supervised only appears once labels are cut to ≤8% of the training set, not at 100% (this project's actual situation).

A separate paper on this exact dataset (Ignatov & Malivenko 2024) also reports that a **3-feature color-intensity Random Forest reaches 53.8% accuracy** and a 48-feature color-histogram Random Forest reaches 82.2% (random baseline: 11.1%), and warns that the dataset may reward shortcut features (color signature, JPEG compression artifacts) more than deep morphological understanding — capping how much *any* representation-learning improvement, SSL or otherwise, can matter for the final linear-probe metric on this specific benchmark.

**Conclusion**: the benchmark is very likely already near-saturated by supervised and ImageNet-transfer learning. This is the single strongest piece of evidence against using leJEPA here — it directly targets the "is there headroom" question and is confirmed against the exact dataset in use, not an adjacent one. The caveat: none of this evidence involves leJEPA specifically, and leJEPA's own domain-mismatch argument (Q2) remains untested on this dataset.

## Q5: Histopathology-specific SSL evidence

Evidence is genuinely mixed, not a clean win for either side:

- **Confirmed, moderate-to-strong**: domain-specific SSL pretraining (on unlabeled pathology images) is a better *transfer-learning starting point* than ImageNet-supervised pretraining, for pathology classification and detection tasks (Lunit CVPR 2023 benchmark; a 31-task, 7-foundation-model clinical benchmark). This is a narrower claim than "SSL beats supervised training" — it's "domain SSL beats generic-domain supervised" as a pretraining source, not "SSL beats a well-tuned supervised model trained from scratch on the target task."
- **Confirmed, but confounded**: a multi-loss framework including a self-supervised stage (S5CL) beats pure supervised cross-entropy on NCT-CRC-HE-100K, but the gain is concentrated in low-label regimes (5-20 labels/class, ~10-12 point gap) and nearly vanishes at 500 labels/class (~0.6 point gap) — consistent with the label-scarcity pattern from Q3/Q4, not a higher accuracy ceiling.
- **Not found, a real gap**: a clean, controlled study isolating "SSL-pretrained-then-fine-tuned" vs. "fully-supervised-from-scratch, same architecture, same labeled data" in histopathology. This specific comparison — the one this project's design most directly needs evidence for — is thinly evidenced in the literature located.
- **Not found**: any published application of JEPA or leJEPA specifically to histopathology classification. One weakly-sourced exception is discussed under Risks below, and even that concerns a different sub-task (segmentation) on a different tissue type (bone, not colorectal).
- **Deep learning beats classical hand-crafted-feature + classical ML** in most histopathology comparisons found (typically by several to ~10 points — e.g., 89% classical vs. 97% ResNet on CRC epithelial/stromal classification), though not universally: one study on prostate gland classification found a hand-crafted-feature SVM slightly *beating* fine-tuned VGG19, and classical methods are consistently noted as less robust to staining/illumination variation.

## Q6: Foundation-model alternative

At least nine open, pretrained pathology foundation models exist, several under permissive licenses (Virchow and GigaPath: Apache 2.0, no restrictions; UNI and Phikon-v2: non-commercial/academic-restricted). On this project's exact dataset:

| Model | License | NCT-CRC-HE-100K accuracy |
|---|---|---|
| Virchow2 | Apache 2.0 | 96.7% |
| GigaPath | Apache 2.0 | 95.9% |
| Phikon-v2 | Non-commercial | 95.5% |
| UNI | Academic-only | 95.4% |

Fine-tuning approach and cost, single GPU:

| Approach | Time | Accuracy | Feasible on free Colab? |
|---|---|---|---|
| Linear probe (frozen) | ~30-60 min | 94-95% | Yes |
| LoRA fine-tune | ~2-4 hours | 95-96% | Yes |
| Full fine-tune | ~6-12 hours | 96-97% | Marginal (Colab Pro/GCP better) |
| leJEPA pretrain from scratch | ~24-72 hours (estimated) | Uncertain, untested on this domain | No, exceeds this project's stated budget |

This directly addresses the target job description's ask to "leverage and fine-tune modern deep learning and vision foundation models for biomedical imaging" more literally than training a from-scratch SSL backbone does, at a fraction of the compute and with results already verified on this exact dataset (rather than leJEPA's zero medical benchmarks).

## Q7: Risks and caveats

Concrete, sourced risks found:

- **SIGReg has no documented sensitivity analysis or scaling guidelines** for its own hyperparameters (number of random slices, characteristic-function bandwidth, integration grid) as dataset size, embedding dimension, or domain change.
- **A GitHub issue (rbalestr-lab/lejepa #17) reports that the loss is computed on non-linear projections of the backbone's output in the actual repo, differing from the paper's description of the loss being applied to embeddings directly** — this is a real, specific, checkable discrepancy between paper and implementation that should be verified before relying on the repo's results matching the paper's claims.
- **Compute mismatch**: leJEPA's own paper recommends batch size ≥128 and 100 epochs (on ImageNet-1K scale); this project's ~8-12 GPU-hour budget likely allows only 10-30 epochs at a similar or smaller batch size. Convergence behavior in this reduced regime is undocumented by the authors.
- **One weakly-sourced claim, flagged as such, not as established fact**: a University of Kentucky project hub page (not peer-reviewed) reports that in a bone-histomorphometry *segmentation* task, MAE and DINOv3 outperformed I-JEPA with and without SIGReg. This is a single, non-peer-reviewed source, on a different sub-task (segmentation, not classification) and a different tissue type (bone, not colorectal) than this project — it should not be read as "leJEPA is proven to underperform in histopathology broadly," but it is a mild negative data point worth being aware of, and is consistent with the broader finding that leJEPA has essentially no track record in medical imaging.
- **No credible, general "SSL is oversold" critique specific to classification-with-full-labels was found** as a standalone position, but the underlying pattern (pretraining gains diminish as labeled data grows) is consistently supported across multiple unrelated domains (physics, forecasting, plant phenotyping, and the histopathology-specific evidence in Q3-Q5).

## Falsification Check

The research plan specified upfront what would count as evidence against leJEPA being the right choice here. All four specified conditions were found to hold:

- ✅ Published supervised/transfer baselines on this exact dataset already near ceiling accuracy (94.3-98.3%) — **confirmed**.
- ✅ No independent evidence that JEPA/SIGReg specifically outperforms simpler alternatives at this data/compute scale — **confirmed** (author-only benchmarks, zero medical-domain results).
- ✅ An existing open pathology foundation model achieves comparable or better accuracy for a fraction of the compute/effort — **confirmed** (95-97%+ in 30 min-4 hrs vs. an estimated 24-72 hrs for leJEPA pretraining).
- ⚠️ Evidence that SIGReg's batch-size/augmentation sensitivities make it unreliable at achievable batch sizes — **partially confirmed**: leJEPA claims batch-size robustness down to 128, but this project's likely achievable batch size and epoch count fall at or below the documented tested range, in an undocumented regime.

## Open Questions

- Whether leJEPA's domain-mismatch argument (in-domain SSL beating large-model transfer on small, visually-specialized datasets, as shown on Galaxy10) would hold on histopathology specifically — untested in any source found.
- Whether a controlled, same-architecture "SSL-from-scratch vs. supervised-from-scratch" comparison in histopathology (which this research could not locate) would change the Q5 conclusion if it existed.
- Whether the GitHub issue #17 paper/repo discrepancy has been resolved since this research was conducted.
- The precise resolution of several very recent (2026) arXiv IDs referenced by researcher subagents (e.g., the "When Does LeJEPA Learn a World Model?" follow-up paper, and JEPA-variant papers referenced in the benchmarks report) has not been independently re-verified beyond the researchers' own fetches — flagged for the verification pass.
