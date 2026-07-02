# SSL (leJEPA) vs. Supervised/Transfer Learning for NCT-CRC-HE-100K — Research Findings (T3)

Scope: two independent questions. Question A is a general literature pattern (no invented number for our exact setup). Question B is a targeted search for real, cited accuracy numbers on this exact dataset pairing (NCT-CRC-HE-100K → CRC-VAL-HE-7K), used to check whether supervised/transfer baselines already saturate the benchmark.

---

## Question A — When does SSL pretraining actually help, in general?

**General pattern found across the literature: SSL's advantage is primarily a function of *label scarcity relative to available data*, not of absolute data volume or absolute compute budget in isolation. The advantage shrinks — and can disappear or reverse — as the labeled fraction approaches 100% of the available pretraining pool.**

Supporting evidence:

1. **Newell & Deng, "How Useful Is Self-Supervised Pretraining for Visual Tasks?" (CVPR 2020)** — arxiv.org/abs/2003.14323
   Using synthetic datasets with full control over label quantity and task difficulty, the authors found SSL pretraining follows one of three patterns as labeled data grows: (a) SSL always helps, (b) SSL reaches a given accuracy with fewer labels but the supervised-from-scratch baseline eventually catches up ("plateaus to the same accuracy"), or (c) SSL's benefit converges to zero *before* accuracy even plateaus. They report **pattern (c) — vanishing benefit — as the most common outcome**. They also found that linear-probe evaluation does not reliably predict fine-tuning performance, which is a methodological caution relevant to how "SSL benefit" gets measured.

2. **Chen et al., "Big Self-Supervised Models are Strong Semi-Supervised Learners" (SimCLRv2, NeurIPS 2020)** — arxiv.org/pdf/2006.10029
   Demonstrates the clearest version of the label-scarcity pattern: SimCLRv2 self-supervised pretraining + fine-tuning gives large wins specifically in **low-label regimes** — e.g., a ResNet-50 reaches 77.5% ImageNet top-1 with only 10% of labels, beating fully-supervised training with 100% of labels. The bigger the label-scarcity gap, the bigger the SSL payoff. This is a *label-fraction* effect, not a raw-data-volume effect — it says nothing about benefit when ~100% of the pretraining pool is already labeled.

3. **Balestriero et al., "A Cookbook of Self-Supervised Learning" (2023)** — arxiv.org/abs/2304.12210
   A broad practitioner survey (co-authored by LeCun). Its core framing of SSL's value proposition is precisely that it "can learn from vast unlabeled data, as opposed to supervised learning which is limited by the availability of labeled data" — i.e., the entire premise of SSL's advantage is access to *unlabeled data beyond the labeled set*. It does not give a quantitative crossover point; it is a methods/recipe guide, not a scaling-law paper.

4. **Voigt et al., "Investigation of semi- and self-supervised learning methods in the histopathological domain" (J. Pathology Informatics, 2023)** — pmc.ncbi.nlm.nih.gov/articles/PMC10070179/, DOI 10.1016/j.jpi.2023.100305
   Directly relevant because it pretrains SSL methods (SimCLR, SimSiam, PAWS, MoCo-family) **on NCT-CRC-HE-100K itself** and evaluates on CRC-HE-7K (= CRC-VAL-HE-7K), the exact dataset pair in this project. Explicit finding: "the performance consistently decreases if the available training data is reduced. A widening gap between the solely supervised trained models and finetuned encoders is discernible" — i.e., the SSL-vs-supervised gap **widens only as labels shrink to 8%, 0.8%, 0.2%** of the training pool; at 100% label availability the SSL-pretrained-then-fine-tuned encoders land around 90.5–91.5% (see Question B), while purely supervised/transfer approaches on this same pairing reach 94–98% (see Question B) — consistent with SSL bringing little or no benefit once essentially all data is already labeled.

5. **Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics" (Nov 2025)** — arxiv.org/abs/2511.08544
   The leJEPA paper itself makes a *different* argument than label scarcity: it reports that in-domain SSL pretraining on small, specialized datasets (Galaxy10, Food101, Flowers102) can **outperform transfer learning from large foundation models (DINOv2/v3, I-JEPA)** trained on generic natural images — i.e., the case for domain-specific SSL rests on *domain mismatch* between ImageNet-style pretraining and a specialized target domain, not on label scarcity. This is a plausible and directionally relevant argument for histology (which is visually very unlike ImageNet), but the specific model sizes, dataset sizes, and compute budgets used in those small-dataset experiments were not extractable from the available abstract/summary sources during this research pass — the exact regime boundaries (how small is "small enough" for this effect, what ViT scale, what epoch count) are not confirmed here.

**Explicit refusal on the quantitative question:** None of the sources above, nor any source found in this research pass, provide a validated way to predict a specific accuracy delta for a ViT-S/16 pretrained with leJEPA on exactly 100,000 images for ~8–12 GPU-hours and then linear-probed on a 9-class histology task. The literature supports a *directional* regime statement (SSL advantage tracks label scarcity, and possibly also domain-mismatch-with-transfer-sources), but any specific percentage-point improvement claimed for this exact project's compute/data budget would be an unsupported extrapolation. **This report does not produce such a number, and any number stated elsewhere for this exact configuration should be treated as an estimate, not a literature-backed figure.**

**Regime summary for this project's specific situation:** NCT-CRC-HE-100K is *fully labeled* — pretraining on it via leJEPA would not be unlocking any unlabeled data beyond what's already available for supervised training. This places the project in the part of the label-scarcity spectrum (labels ≈ 100% of available pretraining data) where the literature above (Newell & Deng; Voigt et al.) most consistently shows **SSL's classical advantage mechanism does not apply**. The only remaining argument for SSL here is the domain-mismatch argument from the leJEPA paper itself (in-domain pretraining beating ImageNet-derived transfer on specialized visual domains) — which is untested at this project's exact scale in the sources found.

---

## Question B — Actual published accuracy numbers: is there headroom left?

All numbers below are on the **exact NCT-CRC-HE-100K (train) → CRC-VAL-HE-7K (external test)** pairing used in this project, unless explicitly noted otherwise.

### Supervised / ImageNet-transfer baselines (no domain SSL)

| Model | Approach | NCT-CRC-HE-100K (internal) | CRC-VAL-HE-7K (external) | Source |
|---|---|---|---|---|
| VGG19 | ImageNet-pretrained, fine-tuned (transfer learning) | 98.7–99% | **94.3–94.4%** | Kather et al. 2019, *PLOS Medicine*, DOI 10.1371/journal.pmed.1002730 — the original paper that created both NCT-CRC-HE-100K and CRC-VAL-HE-7K. https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002730 |
| DenseNet | supervised | — | 92.9% | Ignatov & Malivenko 2024, arXiv:2409.11546, Table 2 |
| VGG19 (re-run) | supervised | — | 94.3% | same |
| Inception-v3 | supervised | — | 94.8% | same |
| ResNet-50 | supervised | — | 94.8% | same |
| VGG16 | supervised | — | 95.3% | same |
| EfficientNet-B0 (single) | ImageNet-pretrained, fine-tuned | — | **97.73%** | same — stated as new SOTA at publication |
| EfficientNet-B0 (2-model ensemble) | ImageNet-pretrained, fine-tuned | — | **98.33%** | same |

Source for the table: Ignatov, A. & Malivenko, G., "NCT-CRC-HE: Not All Histopathological Datasets Are Equally Useful," arXiv:2409.11546 (Sept 2024). https://arxiv.org/abs/2409.11546 (full text: https://arxiv.org/html/2409.11546v1). The paper explicitly confirms: "We used the conventional NCT-CRC train/validation splits in all experiments, where NCT-CRC-HE-100K data is used for training and CRC-VAL-HE-7K — for validation" — i.e., this is the exact split used in this project.

### Self-supervised / domain-SSL-pretrained results, same benchmark

| Model | Approach | CRC-VAL-HE-7K | Source |
|---|---|---|---|
| iBOT (ViT-Large) | generic SSL-pretrained | 95.8% | Ignatov & Malivenko 2024, Table 2 (citing original iBOT/DINO evaluation papers) |
| DINO (ViT) | generic SSL-pretrained | 95.9% | same |
| CTransPath (Swin) | pathology-domain SSL-pretrained (MoCo-v3-style) | 96.52% | same |
| DeepCMorph (87M params) | morphology-aware supervised/hybrid | 96.99% | same |

**Critical finding: on this exact benchmark, the best plain-supervised, ImageNet-transfer model (EfficientNet-B0, 97.73–98.33%) outperforms all the SSL-pretrained models listed (iBOT 95.8%, DINO 95.9%, CTransPath 96.52%)**, per the same comparison table in Ignatov & Malivenko 2024. SSL pretraining did not produce the best result on this dataset in the literature found.

The same paper also gives a mechanistic reason the dataset may be near-saturated for easy signals: using only 3 RGB color-intensity features + a Random Forest reaches 53.8% accuracy, and a 48-feature color histogram + Random Forest reaches 82.2% — random baseline is 11.1% (9 classes). The authors state directly: "no advanced histopathology-related features are needed to correctly classify images from the CRC-VAL-HE-7K test set, and this should be taken into account when designing and interpreting all future results obtained on this dataset," and separately warn that JPEG-compression artifacts in the released dataset are detectable by simple convolutional filters and may be exploited as shortcut features by deep models.

### SSL pretrained specifically on NCT-CRC-HE-100K, evaluated on CRC-VAL-HE-7K (closest possible match to this project's actual plan)

Voigt et al. 2023 (*Journal of Pathology Informatics*, DOI 10.1016/j.jpi.2023.100305) is the closest methodological match found: SSL pretraining (SimCLR, SimSiam, PAWS) **on NCT-CRC-HE-100K itself**, then linear-probe or fine-tune, evaluated on CRC-HE-7K (= CRC-VAL-HE-7K), exactly as planned in this project (though architectures were ResNet18/50, not ViT-S/16, and the SSL methods were not leJEPA).

- Frozen-encoder (linear probe) accuracy at 100% label availability: SimCLR ≈ 90.45%, SimSiam ≈ 89.42%, PAWS ≈ 79.25%
- Fine-tuned encoder accuracy at 100% label availability: SimSiam ≈ 91.47%, SimCLR ≈ 90.57%, PAWS ≈ 90.52%
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10070179/

These SSL-pretrained numbers (89–91.5%) are **notably lower** than the plain supervised/ImageNet-transfer numbers on the same dataset pairing (94.3% Kather 2019 VGG19-transfer; 97.7–98.3% Ignatov 2024 EfficientNet-B0). The paper's own conclusion is that the SSL-vs-supervised gap only widens in favor of SSL once labeled data is cut to 8%, 0.8%, or 0.2% of the training set — not at full (100%) label availability, which is this project's actual situation.

### Secondary/lower-confidence data point

A "D1/D2" CNN model pair reportedly achieves 95–96% on CRC-VAL-HE-7K (99.8%/99.53% on NCT-CRC-HE-100K, D2 trained on only 30% of the training set), per a ScienceDirect paper "Colon and lung cancer classification from multi-modal images using resilient and efficient neural network architectures" (PMC11089372, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11089372/). The full-text ScienceDirect page returned an HTTP 403 during this research pass, so exact architecture/pretraining details (ImageNet-pretrained vs. from-scratch) could not be independently confirmed — this number is reported here for completeness but flagged as **unverified in detail**, unlike the Kather 2019 and Ignatov 2024 numbers above, which were confirmed via direct paper fetch.

---

## Bottom line: is there headroom left for SSL, or does supervised/transfer already saturate this dataset?

**Best-supported answer: the benchmark is very likely already near-saturated by plain supervised / ImageNet-transfer learning, which weakens (but does not fully eliminate) the case for leJEPA pretraining in this specific project.**

Basis for this conclusion, weighted by source strength:

- The dataset's own creators (Kather et al. 2019) already reached 94.3–94.4% on the external CRC-VAL-HE-7K test set using plain ImageNet-transfer VGG19 — no domain SSL involved — the year the dataset was published.
- The best currently-published result on this exact benchmark (97.73% single model / 98.33% ensemble, Ignatov & Malivenko 2024) is a **plain supervised, ImageNet-pretrained EfficientNet-B0** — not an SSL method. In the same paper's own comparison table, generic SSL methods (iBOT 95.8%, DINO 95.9%) and even a pathology-domain-SSL-pretrained model (CTransPath 96.52%) all score **below** this supervised baseline on this dataset.
- The one study found that pretrains SSL directly on NCT-CRC-HE-100K and tests on CRC-VAL-HE-7K (Voigt et al. 2023) gets only ~90–91.5% at full label availability — below both the 2019 and 2024 supervised numbers — and explicitly reports that the SSL advantage only shows up when labels are artificially restricted to 8% or less of the training pool, not at 100% (this project's actual label availability).
- Independent evidence (Ignatov & Malivenko 2024) suggests the dataset may reward shortcut features (color signature, JPEG compression artifacts) more than deep morphological understanding, which caps how much any representation-learning improvement — SSL or otherwise — can matter for the final linear-probe metric.

**Caveat / what would change this conclusion:** none of the near-saturation evidence above involves leJEPA specifically, nor a ViT-S/16 backbone at this project's exact scale. The leJEPA paper's own claim — that in-domain SSL pretraining on small, visually-specialized datasets can beat large foundation-model transfer — has not been tested on NCT-CRC-HE-100K/CRC-VAL-HE-7K in any source found here, and remains a plausible but unverified argument for this domain. If the project's real goal includes generating transferable histology representations beyond this one 9-class task, or robustness to the shortcut-feature issue Ignatov & Malivenko flag, that is a different cost-benefit case than "will linear-probe accuracy on CRC-VAL-HE-7K improve" — and this report does not have evidence either way on that broader framing.

**Status: numbers reported here are sourced and traceable to specific papers/URLs (verified — confirmed via direct paper fetch for Kather 2019, Ignatov & Malivenko 2024, and Voigt et al. 2023). The "D1/D2" 95–96% figure is unverified in detail (source page blocked). No number in this report was fabricated or estimated for this project's exact leJEPA/ViT-S/16/100K-image/8–12-GPU-hour configuration — Question A's regime pattern is explicitly not converted into a point estimate, per instructions.**
