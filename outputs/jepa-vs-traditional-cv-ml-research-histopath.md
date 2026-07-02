# SSL (leJEPA-style) vs. Supervised Deep Learning vs. Classical ML for Histopathology Tissue Classification

Researcher: T4. Scope: literature check on whether self-supervised learning (SSL) is a sound choice for histopathology tissue image classification, relative to supervised deep learning and to classical (pre-deep-learning) feature engineering + classical ML. Focus dataset where possible: NCT-CRC-HE-100K / Kather colorectal cancer H&E tissue datasets.

**Verification status: unverified / literature-search only.** All claims below come from WebSearch result snippets and WebFetch summaries of paper abstracts/pages (not full close reads of PDFs — several PDF/PMC fetches failed due to DNS/binary-encoding issues in this environment, noted inline). No code was run, no numbers were independently recomputed. Treat quantitative figures as "as reported in secondary summaries of the source," not as independently checked by this researcher.

---

## RQ1: What published SSL work exists for histopathology / digital pathology classification?

SSL applied to histopathology is a real and fairly active sub-literature, with several methods evaluated specifically on NCT-CRC-HE-100K or the related Kather colorectal cancer datasets:

- **CS-CO** (MICCAI 2021) — a hybrid SSL method combining cross-stain prediction (generative) with contrastive learning, using a novel "stain vector perturbation" augmentation designed for H&E images. Evaluated on NCT-CRC-HE-100K among other H&E datasets for patch-level tissue classification and slide-level prognosis/subtyping.
  - [MICCAI 2021 page](https://miccai2021.org/openaccess/paperlinks/2021/09/01/428-Paper0322.html)
  - [GitHub repo](https://github.com/easonyang1996/CS-CO)
  - [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_5)
  - [ScienceDirect journal version](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001864)
  - Note: exact accuracy numbers on NCT-CRC-HE-100K could not be extracted from available snippets/fetches; the claim of "superiority" is the paper's own framing, not independently confirmed here.

- **S5CL** — hierarchical framework unifying fully-supervised, self-supervised, and semi-supervised contrastive learning, evaluated on NCT-CRC-HE-100K (colorectal cancer) and a Munich AML leukemia blood-smear dataset. [arXiv:2203.07307](https://arxiv.org/pdf/2203.07307) — see RQ2 for the direct-comparison numbers extracted from this paper.

- **DINO applied to pathology**: a ViT-small trained with DINO self-distillation on ~0.1M NCT-CRC-HE-100K tiles is referenced in the "Precise Location Matching Improves Dense Contrastive Learning in Digital Pathology" paper. [arXiv:2212.12105](https://arxiv.org/pdf/2212.12105)

- **Lunit's large-scale SSL pathology benchmark** (CVPR 2023) compares four SSL paradigms — MoCo v2 (contrastive), Barlow Twins (non-contrastive), SwAV (clustering), and DINO (ViT) — pretrained on ~32.6M patches from TCGA (20,994 WSIs) + an internal dataset (TULIP, 15,672 WSIs), then evaluated on downstream classification/segmentation tasks including **CRC (colorectal cancer) classification**, BACH, MHIST, PCam, and CoNSeP nuclei segmentation.
  - [arXiv:2212.04690](https://arxiv.org/abs/2212.04690)
  - [Lunit research page](https://lunit-io.github.io/research/publications/pathology_ssl/)
  - [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology)

- **Pathology foundation models** built on SSL and evaluated (in part) on CRC-type tasks: CTransPath, UNI, Virchow, Prov-GigaPath, Hibou, plus in-house DINO variants (SP21M/SP85M). [Virchow paper](https://arxiv.org/html/2309.07778), [UNI-style "general-purpose self-supervised model for computational pathology"](https://arxiv.org/pdf/2308.15474), [Towards Large-Scale Training of Pathology Foundation Models](https://arxiv.org/pdf/2404.15217), [Hibou](https://arxiv.org/pdf/2406.05074). A specialized "histopathology-aware DINO" variant (HistoDARE) reportedly reaches 99.33% accuracy on NCT-CRC-HE-100K per search snippets ([Scientific Reports](https://www.nature.com/articles/s41598-025-31438-8)) — not independently verified here.

- **A Clinical Benchmark of Public Self-Supervised Pathology Foundation Models** — evaluates 7 public/in-house SSL-pretrained models (CTransPath, UNI, Virchow, Prov-GigaPath, two in-house DINO models) against an ImageNet-supervised ResNet50 baseline, across 31 clinically relevant tasks (6,818 patients, 9,528 slides), including cancer/disease detection and biomarker prediction. [arXiv:2407.06508](https://arxiv.org/pdf/2407.06508) / [PMC12003829](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/) (PMC fetch failed due to DNS in this environment; summary drawn from arXiv HTML mirror).

- Other SSL-for-pathology work found: **CLASS-M** (adaptive stain-separation contrastive learning + pseudo-labeling) [arXiv:2312.06978](https://arxiv.org/pdf/2312.06978); a **self-supervised contrastive WSI-representation** framework using attention-based MIL + supervised contrastive learning [PMC9808093 / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2153353922007271), [PubMed](https://pubmed.ncbi.nlm.nih.gov/36605114/).

**No leJEPA-specific application to histopathology was found.** Search did not surface any published work applying JEPA or leJEPA specifically to histopathology/digital pathology; the closest analogues are DINO (self-distillation, architecturally and philosophically related to JEPA-family "predict in representation space" methods) and masked-image-modeling / contrastive approaches (CS-CO, MoCo v2, SwAV, Barlow Twins). This is a genuine gap, not a search failure I can rule out with more queries alone — it should be treated as "no evidence found," not "confirmed absent."

**Caveat on NCT-CRC-HE-100K data quality**: one paper flags that NCT-CRC-HE-100K itself has known issues — inconsistent color normalization, JPEG artifacts that differ systematically between classes, and some corrupted tissue samples — which is relevant because class-correlated artifacts can inflate apparent classification accuracy (both classical and deep) on this specific dataset. [arXiv:2409.11546 "NCT-CRC-HE: Not All Histopathological Datasets Are Equally Useful"](https://arxiv.org/abs/2409.11546)

---

## RQ2: Does SSL pretraining demonstrably beat supervised-from-scratch in histopathology? (direct comparisons only)

This is the crux question, and the evidence is **more limited and more entangled than a clean "SSL wins" headline would suggest.** Three categories of comparison were found, and they are not the same claim:

### (a) SSL-domain-pretraining vs. ImageNet-supervised-pretraining (both followed by fine-tuning/linear probe) — found, and SSL wins here
The Lunit CVPR 2023 benchmark explicitly compares pathology-domain SSL pretraining against ImageNet-supervised pretraining across linear-probe and fine-tuning evaluations, including on CRC classification, and reports: **"large-scale domain-aligned pre-training in pathology consistently out-performs ImageNet pre-training in standard SSL settings such as linear and fine-tuning evaluations, as well as in low-label regimes."** [arXiv:2212.04690](https://arxiv.org/abs/2212.04690), [Lunit page](https://lunit-io.github.io/research/publications/pathology_ssl/)

The Clinical Benchmark paper similarly reports SSL pathology foundation models (CTransPath, UNI, Virchow, Prov-GigaPath, in-house DINO) consistently beat an ImageNet-supervised ResNet50 baseline on **disease-detection tasks** (AUC > 0.9 for SSL models vs. clearly lower for the ImageNet baseline), but results are **mixed for biomarker-prediction tasks** — performance varies by tissue type and pretraining-set composition, and for immune checkpoint inhibitor (ICI) response prediction, "AUCs [are] barely above chance," with the best model averaging only ~0.6. The authors explicitly caution: **"It may be that we are reaching a limit to how much relevant histological information can be learned via SSL alone with current strategies."** [arXiv:2407.06508](https://arxiv.org/pdf/2407.06508)

Important distinction: **this is "SSL-pretrain-then-transfer" beating "ImageNet-supervised-pretrain-then-transfer," not "SSL beats supervised-from-scratch-on-the-target-task."** It is evidence that domain-relevant SSL pretraining is a better starting point than natural-image supervised pretraining — a narrower and more defensible claim than "SSL beats supervised learning in histopathology" in general.

### (b) SSL-influenced multi-loss training vs. pure supervised cross-entropy, same architecture, same dataset (NCT-CRC-HE-100K) — found, with an important caveat
The S5CL paper is the closest thing found to a controlled, same-architecture (ResNet18, ImageNet-initialized in all conditions) comparison on NCT-CRC-HE-100K, across labeled-set sizes of 5, 10, 20, 50, and 500 images/class:

| Labels/class | CrossEntropy (fully supervised) | SupConLoss | Meta Pseudo Labels | S5CL (hierarchical, incl. self-supervised stage) |
|---|---|---|---|---|
| 5  | ~76% | ~78% | ~82% | ~88% |
| 10 | ~82% | ~84% | ~87% | ~91% |
| 20 | ~88% | ~89% | ~91% | ~93% |
| 50 | ~92% | ~93% | ~94% | ~95% |
| 500 | ~96% | ~96% | ~96% | ~96.6% |

(Numbers extracted via WebFetch summary of the ar5iv HTML mirror; not independently re-verified against the raw PDF table, so treat as approximate/inferred rather than confirmed exact figures.) [arXiv:2203.07307](https://arxiv.org/pdf/2203.07307)

The paper's own abstract text states accuracy "increases by up to 9% compared to supervised cross-entropy loss" on the CRC dataset. **However, this gain is attributed to the full S5CL hierarchical method — which combines a self-supervised stage with supervised contrastive learning (SupConLoss, which still uses labels) and semi-supervised pseudo-labeling (Meta Pseudo Labels) — not to self-supervised pretraining in isolation.** No clean ablation isolating "pure SSL pretraining alone" vs. "supervised cross-entropy alone" was found in the available summary. The pattern that is clear: **the gap is largest in extremely low-label regimes (5–20 labels/class, ~10-12 point gap) and nearly vanishes at higher label counts (500/class, ~0.6 point gap)** — consistent with SSL/semi-supervised methods' typical value proposition being label-efficiency, not a higher accuracy ceiling.

### (c) Fully-supervised CNN benchmark numbers vs. SSL linear-probe numbers on the same dataset, different papers — informal, not a controlled comparison
The original Kather et al. 2019 fully-supervised CNN (ImageNet-pretrained ResNet18, fine-tuned with full labels) on NCT-CRC-HE-100K/CRC-VAL-HE-7K reportedly reaches ~99% on the training-domain set and ~94.4% on the external validation set. [Search summary; original paper: Kather et al., "Predicting survival from colorectal cancer histology slides using deep learning," PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002730) — DNS/fetch issues prevented a full read of the primary source; figures are as reported in search summaries.

Separately, DINO linear-probe accuracy on CRC classification is reported around 93.5–94.6% (ViT-S/ViT-B variants) in large-scale pathology foundation model papers ([arXiv:2404.15217](https://arxiv.org/pdf/2404.15217)), and a specialized DINO variant (HistoDARE) reportedly reaches 99.33% ([Scientific Reports](https://www.nature.com/articles/s41598-025-31438-8)).

**These numbers are in the same rough ballpark (~93–99%) and come from different papers, different splits, different eval protocols (linear probe vs. full fine-tune), and possibly different label sets — they do not constitute a controlled head-to-head comparison.** Citing them side-by-side to claim "SSL matches or slightly beats supervised" would overstate what the evidence actually shows. The honest read is: **both approaches saturate NCT-CRC-HE-100K at a high accuracy ceiling (likely partly because the dataset itself is known to have exploitable artifacts, per the RQ1 caveat), which makes it hard to detect a real SSL-vs-supervised accuracy gap on this particular benchmark even if one exists.**

### Bottom line for RQ2
- **Confirmed by direct comparison**: SSL-style pathology-domain pretraining beats *ImageNet-supervised* pretraining as a transfer-learning starting point (Lunit benchmark, clinical benchmark paper) — fairly consistent finding for detection-type tasks, weaker/mixed for fine-grained biomarker prediction.
- **Confirmed by direct comparison, but confounded**: a multi-loss framework that includes a self-supervised component beats pure supervised cross-entropy on NCT-CRC-HE-100K, especially in low-label regimes — but the self-supervised contribution is not cleanly isolated from supervised-contrastive and semi-supervised components in the one paper found doing this experiment (S5CL).
- **Not found**: a clean, controlled study isolating "SSL-pretrained-from-scratch-on-unlabeled-histopathology-data, then linear-probe/fine-tune" vs. "identical architecture trained fully supervised end-to-end from random initialization on the full labeled set" on NCT-CRC-HE-100K or a closely related dataset. This is a real gap in what I could locate, not something I'm inferring negatively from silence — it means the specific "SSL vs. supervised-from-scratch" comparison the research question asks for is thinly evidenced in histopathology specifically, even though SSL-vs-ImageNet-supervised-transfer is reasonably well evidenced.

---

## RQ3: Classical feature-engineering + classical ML — historical performance and comparison to deep learning

Classical texture/color feature + SVM/RF pipelines were the pre-deep-learning standard in digital pathology (roughly 1990s–2012), and remain a live baseline/complementary approach in some recent papers.

- **Kather et al. 2016**, "Multi-class texture analysis in colorectal cancer histology" (Scientific Reports) — the origin of the classic 5,000-image, 8-class CRC texture dataset (predecessor to NCT-CRC-HE-100K). Compared multiple hand-crafted feature families (GLCM, discrete wavelet transform, Gabor filters, Haralick features, LBP, fractal filters) with four classical classifiers (Naive Bayes, Random Forest, SVM, Multilayer Perceptron). Best classical strategy improved tumor-stroma separation from 96.9% to 98.6% accuracy (binary task; full 8-class multi-class accuracy figure not confirmed in available snippets). [Scientific Reports](https://www.nature.com/articles/srep27988) (direct fetch blocked by Nature's login-wall redirect; summary from search snippets and abstract) / [PMC mirror](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4910082/) (DNS-blocked in this environment) / [dataset on Zenodo](https://zenodo.org/records/53169) / [GitHub](https://github.com/jnkather/histology-multiclass-texture).

- **GLCM/Haralick + SVM on CRC tissue**: multiple follow-on papers report classical-feature accuracy in the low-to-mid 90s% range on CRC tissue classification tasks — e.g., 93.17% (8-class) and up to 96.02% for combined-feature SVM approaches per search summaries. [Colorectal Cancer Tissue Classification Based on Machine Learning (PACIS 2019)](https://aisel.aisnet.org/pacis2019/66/), [3D GLCM across RGB/HSV/L*A*B color spaces](https://link.springer.com/article/10.1007/s11042-022-11946-9), [Two Ensemble-CNN Approaches for CRC Tissue Type Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321410/) (this last one is a DL paper but likely cites classical baselines; not independently confirmed here).

- **Direct classical-vs-deep comparisons on CRC-adjacent tasks**: search summaries describe a study where "machine learning with automatic feature extraction" (classical pipeline) achieved 89.26% (100x) / 89.42% (200x) accuracy on epithelial/stromal region classification, versus a ResNet deep model reaching 97.07% weighted accuracy at 100x — i.e., **deep learning clearly outperformed classical features by roughly 7-8 points** in that comparison. Source: MDPI "Deep Learning Techniques for the Classification of Colorectal Cancer Tissue" [mdpi.com/2079-9292/10/14/1662](https://www.mdpi.com/2079-9292/10/14/1662) (direct fetch returned HTTP 403; figures taken from WebSearch summary snippet, not independently confirmed against the primary source).

- **Breast cancer (BreakHis) — classical vs. deep, magnification-dependent**: a comparative study fused multiple hand-crafted feature extractors and trained five classical classifiers, compared against fine-tuned VGG-19 deep learning. Deep learning approaches in this space (AlexNet, VGG16/19 ensembles) reach 85.6–95.29% depending on magnification and architecture; the paper's authors note classical handcrafted methods "lacked robustness to variations in staining, illumination, and magnification levels." [PMC8001768](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8001768/) (DNS-blocked for direct fetch; summary from search snippets).

- **A case where classical ML matched or slightly beat deep learning**: "First-Stage Prostate Cancer Identification on Histopathological Images: Hand-Driven versus Automatic Learning" (2019) directly compared exhaustive hand-crafted feature extraction (morphology, texture, fractal, contextual descriptors) + nonlinear SVM against fine-tuned VGG19 deep learning, on discrimination of artefact/benign/Gleason-3 prostate glands. Result: **the hand-driven SVM approach "reported a slight outperforming over the rest of experiments," with final multi-class accuracy of 0.876 ± 0.026.** [PMC7514840](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7514840/) / [MDPI](https://www.mdpi.com/1099-4300/21/4/356) (DNS-blocked for direct PMC fetch; MDPI not independently fetched either; summary from search snippets — this is a notable exception worth flagging honestly since it cuts against a simple "DL always wins" narrative).

- **General historical/narrative framing**: a recent systematic review states hand-crafted-feature machine learning was the dominant paradigm in the 1990s–2000s, with deep learning introduced to histopathology in the 2010s and now representing the state of the art for slide analysis, while hand-crafted features retain value for interpretability and as a complementary signal (hybrid handcrafted+deep ensembles are an active research direction). [Biological feature-based machine learning in histopathological images: a systematic review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12858367/) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2153353925001257) — DNS-blocked for direct fetch, summary from search snippet only.

### Bottom line for RQ3
Classical feature-engineering + classical ML reliably reaches **high-80s to mid-90s% accuracy** on colorectal/breast/prostate histopathology tissue classification tasks — respectable but generally **a few to ~10 percentage points below well-tuned deep learning** in most comparisons found (e.g., 89% classical vs. 97% ResNet on CRC epithelial/stromal classification; BreakHis classical baselines below DL ensembles). There is at least **one credible counter-example** (prostate gland classification, hand-crafted SVM slightly beating fine-tuned VGG19) showing the gap is not universal, especially on smaller or more constrained datasets/tasks. Classical methods also appear less robust to staining/illumination/magnification variation, per multiple sources, which matters operationally even when raw accuracy is close.

---

## RQ4: Is the evidence for SSL's benefit in histopathology strong/consistent, or thin/mixed?

**Thin/mixed, not strong**, when the question is narrowed to the specific claim "SSL beats supervised-from-scratch, same architecture, same labeled dataset, in histopathology." Summary of what's actually established vs. not:

**Reasonably well-established (multiple independent sources, consistent direction):**
- Domain-relevant SSL pretraining (on unlabeled pathology images) is a better transfer-learning starting point than ImageNet-supervised pretraining, for pathology classification and detection tasks, including CRC classification specifically. (Lunit benchmark, Clinical Benchmark paper)
- Deep learning (supervised or SSL-pretrained) generally outperforms classical hand-crafted-feature + classical-ML pipelines in histopathology tissue classification, though the margin varies (a few points to ~10 points) and is not universal (prostate gland counter-example).

**Weakly established / single-source / confounded:**
- SSL (as part of a hierarchical multi-loss framework) beats pure supervised cross-entropy on NCT-CRC-HE-100K, with the benefit concentrated in low-label regimes and shrinking to near-zero at higher label counts (S5CL). This is suggestive of real value but is not a clean isolation of the SSL component, and comes from one paper.
- SSL foundation models' benefit is inconsistent across task types even within pathology: strong/consistent for disease detection (AUC > 0.9), weak-to-near-chance for some biomarker-prediction tasks (Clinical Benchmark paper), with the authors themselves flagging a possible ceiling on SSL-alone approaches.

**Not found (a real gap, not a negative finding):**
- A controlled, same-architecture, same-dataset comparison of "SSL-pretrained-then-fine-tuned" vs. "fully supervised-from-scratch" (both starting from random initialization, not ImageNet weights) on NCT-CRC-HE-100K or a closely related histopathology dataset. Every SSL comparison found either (a) compares SSL-domain-pretraining against ImageNet-supervised-pretraining (a different question), or (b) confounds SSL with supervised-contrastive/semi-supervised components (S5CL), or (c) compares numbers across separate papers/protocols rather than within one controlled experiment.
- Any published application of JEPA or leJEPA specifically to histopathology.

### Strength-of-evidence ratings

| Claim | Rating | Basis |
|---|---|---|
| SSL pretraining (domain-specific) beats ImageNet-supervised pretraining as a transfer-learning start-point in histopathology | **Moderate-to-Strong** | Multiple independent studies (Lunit CVPR'23 benchmark, clinical benchmark of 7 foundation models across 31 tasks), consistent direction for detection tasks; weaker for biomarker prediction |
| SSL beats fully-supervised-from-scratch (same architecture, same labeled data) in histopathology specifically | **Weak / Mixed** | No clean controlled comparison found; best available evidence (S5CL) confounds SSL with supervised-contrastive and semi-supervised components, and shows the gap vanishing as labeled data increases — consistent with a label-efficiency benefit rather than a proven accuracy-ceiling benefit |
| Deep learning (supervised or SSL) beats classical hand-crafted-feature + classical ML in histopathology tissue classification | **Moderate** | Consistent direction across CRC and breast comparisons (typically DL wins by several to ~10 points), but not universal — at least one credible counter-example (prostate gland classification) and noted robustness/interpretability trade-offs favoring classical methods in some contexts |
| leJEPA (or JEPA-family methods generally) have been validated for histopathology | **No evidence found** | Not a rating so much as an absence — treat as an open question, not as "JEPA is unsuitable" |

**Overall honest takeaway**: There is a reasonably solid case that pathology-domain SSL pretraining is a better foundation than generic ImageNet-supervised pretraining, and a reasonably solid (if not exhaustive) case that deep learning generally beats classical feature engineering in this domain. But the narrower, more decision-relevant question — does SSL pretraining beat a well-tuned fully-supervised model trained from scratch on the same architecture and same labeled data, in histopathology specifically — is not cleanly answered by the literature located in this search. The one paper that comes closest (S5CL) shows a benefit that shrinks toward zero as labeled data grows and is not attributable to SSL alone. Given that this is meant to inform a real engineering decision, this gap should be treated as a genuine open risk rather than papered over: if the target use case has abundant labels, the literature does not strongly support that SSL pretraining will beat careful supervised training; if labels are scarce, the (weaker, confounded) evidence leans toward SSL/semi-supervised approaches being worth trying.

---

## Sources consulted (full list)

- [CS-CO MICCAI 2021](https://miccai2021.org/openaccess/paperlinks/2021/09/01/428-Paper0322.html), [GitHub](https://github.com/easonyang1996/CS-CO), [Springer](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_5), [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1361841522001864)
- [S5CL arXiv:2203.07307](https://arxiv.org/pdf/2203.07307)
- [Precise Location Matching Dense Contrastive Learning arXiv:2212.12105](https://arxiv.org/pdf/2212.12105)
- [Lunit SSL Benchmark arXiv:2212.04690](https://arxiv.org/abs/2212.04690), [Lunit page](https://lunit-io.github.io/research/publications/pathology_ssl/), [GitHub](https://github.com/lunit-io/benchmark-ssl-pathology)
- [Clinical Benchmark of SSL Pathology Foundation Models arXiv:2407.06508](https://arxiv.org/pdf/2407.06508) / [PMC12003829](https://pmc.ncbi.nlm.nih.gov/articles/PMC12003829/)
- [Virchow arXiv:2309.07778](https://arxiv.org/html/2309.07778)
- [General-purpose SSL model for computational pathology arXiv:2308.15474](https://arxiv.org/pdf/2308.15474)
- [Towards Large-Scale Training of Pathology Foundation Models arXiv:2404.15217](https://arxiv.org/pdf/2404.15217)
- [Hibou arXiv:2406.05074](https://arxiv.org/pdf/2406.05074)
- [HistoDARE Scientific Reports](https://www.nature.com/articles/s41598-025-31438-8)
- [CLASS-M arXiv:2312.06978](https://arxiv.org/pdf/2312.06978)
- [Self-supervised contrastive WSI representation, PMC9808093](https://pmc.ncbi.nlm.nih.gov/articles/PMC9808093/) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2153353922007271) / [PubMed](https://pubmed.ncbi.nlm.nih.gov/36605114/)
- [NCT-CRC-HE dataset quality critique arXiv:2409.11546](https://arxiv.org/abs/2409.11546)
- [Kather et al. 2016, Scientific Reports](https://www.nature.com/articles/srep27988) / [PMC4910082](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4910082/) / [Zenodo dataset](https://zenodo.org/records/53169) / [GitHub](https://github.com/jnkather/histology-multiclass-texture)
- [Kather et al. 2019, PLOS Medicine (survival prediction / CNN benchmark)](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002730)
- [Colorectal Cancer Tissue Classification Based on Machine Learning, PACIS 2019](https://aisel.aisnet.org/pacis2019/66/)
- [3D GLCM across color spaces](https://link.springer.com/article/10.1007/s11042-022-11946-9)
- [Two Ensemble-CNN Approaches for CRC Tissue Classification, PMC8321410](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321410/)
- [Deep Learning Techniques for Classification of CRC Tissue, MDPI](https://www.mdpi.com/2079-9292/10/14/1662)
- [Conventional ML vs. Deep Learning for BreakHis, PMC8001768](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8001768/)
- [Conventional ML and DL Multi-Classification of Breast Cancer, PMC7256154](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7256154/)
- [First-Stage Prostate Cancer Identification: Hand-Driven vs. Automatic Learning, PMC7514840](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7514840/) / [MDPI](https://www.mdpi.com/1099-4300/21/4/356)
- [Biological feature-based ML in histopathology: systematic review, PMC12858367](https://pmc.ncbi.nlm.nih.gov/articles/PMC12858367/) / [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2153353925001257)
- [Multi-class Tissue Classification with Handcrafted and Deep Features, Springer](https://link.springer.com/chapter/10.1007/978-3-030-84522-3_42)

## Known access failures during this research (for transparency)
- `pmc.ncbi.nlm.nih.gov` and `www.ncbi.nlm.nih.gov` were unreachable via WebFetch in this environment (DNS resolution errors) — several PMC-hosted papers could only be characterized via WebSearch snippets, not full-text fetch.
- `nature.com` and `springer.com` article pages redirected to login walls; content characterized via search snippets/abstracts only.
- `mdpi.com` returned HTTP 403 on direct fetch; characterized via search snippets only.
- The S5CL arXiv PDF fetch initially returned corrupted/binary content; a working extraction was obtained via the ar5iv.labs.arxiv.org HTML mirror instead.
