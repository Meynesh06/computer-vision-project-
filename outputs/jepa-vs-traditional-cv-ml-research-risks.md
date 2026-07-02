# leJEPA for Image Classification Pretraining: Research on Risks, Caveats, and Failure Modes

**Project Context:** Evaluating leJEPA (self-supervised learning with SIGReg loss) as a pretraining method for a ViT-S/16 image classifier backbone with constrained compute (~8-12 GPU-hours, ~100,000 training images, single GPU).

---

## 1. Known Failure Modes and Sensitivities in JEPA-Style SSL and SIGReg

### Batch Size Robustness (Positive Finding)
leJEPA demonstrates **insensitivity to batch sizes from 128 to 1024 with less than 1% performance drop**, which is unusual and favorable compared to other SSL methods. The paper reports batch sizes ≥128 as starting points and shows stability across this range. [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)

### SIGReg Loss: Incomplete Sensitivity Analysis (Risk)
While SIGReg guarantees bounded gradients and curvature, **the loss function entails choices (number of slices M, characteristic function bandwidth σ, integration grid t, random-seed strategy) but has no sensitivity analysis, principled defaults, or guidelines for scaling these hyperparameters with dataset size K, embedding dimension N, architecture, and domain**. Additionally, **there is no error bounds analysis for the integral discretization, and no theory quantifying how averaging affects test-level statistical power, false-positive control, or convergence guarantees when used as a loss**. [https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss](https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss)

### Open Questions on Optimal Geometry
While the paper proves that the isotropic Gaussian minimizes worst-case risk, **it remains an open question whether specific downstream tasks might benefit from structured, anisotropic embedding geometries** rather than the enforced isotropy. [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)

### Robustness to Distribution Mismatch Not Studied
SIGReg's robustness to **heavy tails (e.g., Student-t distributions), skewed/elliptical targets, or outliers is not studied**, leaving uncertainty about behavior on datasets with atypical feature distributions. [https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss](https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss)

### GitHub Issue: Loss Implementation Discrepancy
Issue #17 in the leJEPA repository reports that **the SIGReg loss is computed on non-linear projections of a ViT backbone's output rather than on the embeddings themselves, whereas the paper describes the loss being applied to embeddings with isotropic Gaussian distribution constraints**. This implementation-to-paper mismatch raises questions about whether published results match claimed theoretical properties. [https://github.com/rbalestr-lab/lejepa/issues/17](https://github.com/rbalestr-lab/lejepa/issues/17)

### Training Documentation Gaps
Users report **missing hyperparameter documentation for ImageNet-1k pretraining Table 2 specifications**, and there are open questions about ResNet50 training configurations on ImageNet100. [https://github.com/galilai-group/lejepa/issues](https://github.com/galilai-group/lejepa/issues)

---

## 2. Critical Takes: Is Self-Supervised Pretraining Oversold When Labels Are Available?

### Finding: Minimal Gain With Full Label Set
**When complete labels are available, SSL pretraining provides negligible improvement over training from scratch.** In HEP-JEPA (a JEPA variant for collider physics), the pretraining-then-finetuned JEPA model "performs almost identically to the model trained from scratch when the complete set of labels is available for training." [https://arxiv.org/pdf/2502.03933](https://arxiv.org/pdf/2502.03933)

### Finding: Domain-Specific Variability in SSL Gains
In collider physics, forecasting tasks, and structured data domains, **the no pre-training baseline remains competitive and achieves best or near-best performance across datasets**, suggesting that SSL's value proposition is task-dependent rather than universal. For forecasting in particular, "forecasting is largely unaffected by pre-training." [https://arxiv.org/pdf/2605.19462](https://arxiv.org/pdf/2605.19462)

### Finding: Supervised Pretraining Can Outperform SSL
In plant phenotyping tasks, **supervised pretraining generally outperforms self-supervised pretraining**, contradicting claims that SSL is universally superior. [https://arxiv.org/html/2407.12210v1](https://arxiv.org/html/2407.12210v1)

### Finding: Benchmarking Protocols May Overstate SSL Gains
A critical review of SSL benchmarking practices found that **most performance differences between discriminative and generative SSL methods can be explained by changes to model backbones rather than the SSL objective itself**, and **it is infeasible to evaluate SSL methods on all possible downstream tasks; standard evaluation protocols may not fully capture real-world performance**, raising concerns about whether advertised SSL gains are specific to particular benchmarks rather than representing universal improvements. [https://arxiv.org/html/2407.12210v1](https://arxiv.org/html/2407.12210v1)

### Finding: Pretraining Gains Diminish With Larger Labeled Datasets
Research shows that **both supervised and self-supervised pre-training methods fail to scale as the labeled dataset size increases**, indicating that pretraining benefits diminish in high-data regimes. With 100K labeled images available, this suggests SSL pretraining may provide minimal ROI. [https://arxiv.org/html/2407.12210v1](https://arxiv.org/html/2407.12210v1)

### Finding: Tabular Data SSL Can Hurt Performance
On tabular data, **pretrained TabNet and Saint models fail to improve or even hurt the performance of supervised learning**, while self-training always boosts performance, suggesting that SSL gains are not universal across domains. [https://arxiv.org/pdf/2302.14013](https://arxiv.org/pdf/2302.14013)

---

## 3. Compute and Data Requirements: Stated Specifications vs. 100K Images, Single GPU

### Stated leJEPA Requirements (from Paper and Repo)

**Training Duration on Standard Benchmarks:**
- ImageNet-1K: 100 epochs (primary results) [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)
- ImageNet-100: 400 epochs for stability testing [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)

**Batch Size Recommendations:**
- Starting point: batch size ≥128 [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)
- Tested range: 128–1024 without significant performance degradation [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)

**GPU Hardware:**
- The paper demonstrates training on 1.8B parameter ViT-g models with "stable training loss," implying use of significant GPU resources in experiments, but specific GPU hour budgets are **not documented**. [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)
- The repository emphasizes "distributed training-friendly codebase (~50 lines of core code)" and "linear time and memory complexity," but **concrete GPU memory or training time specifications are absent from official documentation**. [https://github.com/galilai-group/lejepa](https://github.com/galilai-group/lejepa)

**Data Augmentation Strategy:**
- Multi-crop approach: 2 global views + 6 local views per image (8 views total), with color and geometric transformations [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)

### Mismatch with 100K Images, Single GPU Budget

**Risk 1: Batch Size at the Edge of Stability**
- With 100K images and a batch size of ≥128, you can fit ~781 minibatches per epoch.
- leJEPA claims batch size robustness from 128–1024, but **128 is the lower bound of tested range**; behavior below 128 is undocumented.
- On a single GPU with limited VRAM, batch sizes of 128+ for ViT-S/16 may not be feasible depending on image resolution and GPU type (e.g., < 8GB VRAM).

**Risk 2: Insufficient Epochs for Convergence**
- The paper recommends **100 epochs for ImageNet-1K** (1.3M images). [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)
- With 100K images and an ~8–12 GPU-hour budget, you can likely run only **10–30 epochs** with batch size 128 and moderate-resolution images.
- leJEPA's training stability claims are based on full experiments; **convergence behavior at 10–30 epochs on a small dataset is undocumented**.

**Risk 3: No Documented Minimum Dataset Size**
- The paper evaluates leJEPA on "10+ datasets, 60+ architectures with varying scales and domains," but does **not specify minimum dataset size thresholds** for stable training. [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3)
- Galaxy10 (~11K images) is shown to work well, but comparison is against transfer learning from larger models (DINOv2), not against training from scratch on the same limited dataset.

---

## 4. Domain Shift: Applying SSL Pre-trained on Natural Images to Medical/Histopathology Imaging

### General SSL Benefit for Domain Shift
**SSL pretraining on domain-specific unlabeled data can alleviate domain shift between ImageNet and medical/histopathology datasets**, as medical imaging has fundamentally different frequency patterns, anatomical features, and noise characteristics than natural images. [https://research.google/blog/self-supervised-learning-advances-medical-image-classification/](https://research.google/blog/self-supervised-learning-advances-medical-image-classification/)

However, **domain shift due to differences in imaging equipment, acquisition protocols, and diagnostic objectives within medical imaging itself (e.g., between hospitals) typically compromises model reliability**. [https://arxiv.org/pdf/2510.27213](https://arxiv.org/pdf/2510.27213)

### Critical Finding: leJEPA Underperforms in Histopathology
**In histopathology datasets, MAE and DINOv3 still outperform I-JEPA both with and without SIGReg**, suggesting leJEPA has room for improvement in this specific medical imaging domain. [https://hub.ai.uky.edu/self-supervised-dual-domain-segmentation-for-static-and-dynamic-bone-histomorphometry-using-lejepa/](https://hub.ai.uky.edu/self-supervised-dual-domain-segmentation-for-static-and-dynamic-bone-histomorphometry-using-lejepa/)

This is particularly concerning because:
1. **leJEPA is a new method (late 2025)** with limited published evaluations on medical imaging.
2. **Existing histopathology SSL work (MAE, DINOv3) already outperforms it**, implying domain-specific tuning or alternative methods may be more practical.

### MedJEPA Status
While there is active research on MedJEPA for medical imaging ([https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/](https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/)), this is still **developmental and not yet published**, meaning leJEPA's applicability to medical imaging remains largely unproven.

### Quantified Domain Shift in Histopathology
Domain shift in digital pathology is significant: **WSIs captured at different medical centers exhibit shifts due to differences in slide preparation, staining protocol, scanner properties, etc., and data from the same medical center can experience shifts over time from changes in the capturing pipeline**. [https://arxiv.org/html/2601.12493v1](https://arxiv.org/html/2601.12493v1)

---

## Summary of Most Concrete, Credible Risks

| Risk | Severity | Source |
|------|----------|--------|
| **SIGReg lacks sensitivity analysis and scaling guidelines** | High | [https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss](https://deepwiki.com/rbalestr-lab/lejepa/3.1-sigreg-loss) |
| **Loss implementation in repo differs from paper specification** | Medium | [https://github.com/rbalestr-lab/lejepa/issues/17](https://github.com/rbalestr-lab/lejepa/issues/17) |
| **With full labels available, JEPA provides negligible gain over training from scratch** | High | [https://arxiv.org/pdf/2502.03933](https://arxiv.org/pdf/2502.03933) |
| **Pretraining gains diminish with larger labeled datasets (100K is relatively large)** | High | [https://arxiv.org/html/2407.12210v1](https://arxiv.org/html/2407.12210v1) |
| **ViT-S trains poorly from scratch on small datasets (<100K); requires ImageNet-scale pretraining** | High | [https://arxiv.org/pdf/2010.11929](https://arxiv.org/pdf/2010.11929) |
| **Single-GPU, 8–12 GPU-hour budget insufficient for recommended 100 epochs; convergence at 10–30 epochs undocumented** | Medium | [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3) |
| **Batch size ≥128 is recommended minimum; lower batch behavior undocumented** | Medium | [https://arxiv.org/html/2511.08544v3](https://arxiv.org/html/2511.08544v3) |
| **leJEPA underperforms MAE and DINOv3 on histopathology datasets** | High (if domain-shifted) | [https://hub.ai.uky.edu/self-supervised-dual-domain-segmentation-for-static-and-dynamic-bone-histomorphometry-using-lejepa/](https://hub.ai.uky.edu/self-supervised-dual-domain-segmentation-for-static-and-dynamic-bone-histomorphometry-using-lejepa/) |
| **leJEPA's suitability for medical imaging is unproven; MedJEPA still developmental** | Medium | [https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/](https://ucsc-ospo.github.io/project/osre26/nelbl/medjepa/) |

---

## Recommendations for This Project

Given the research findings:

1. **If labels are already available**, consider **supervised learning from scratch or with light augmentation** as a strong baseline before investing GPU budget in SSL pretraining. The HEP-JEPA finding suggests pretraining will provide minimal ROI. [https://arxiv.org/pdf/2502.03933](https://arxiv.org/pdf/2502.03933)

2. **If targeting medical/histopathology imaging**, leJEPA's documented underperformance vs. MAE/DINOv3 suggests those methods may be safer choices despite less recent publication dates.

3. **If proceeding with leJEPA**, validate the loss implementation against the paper spec (Issue #17) and run ablations to confirm SIGReg behaves as claimed on your dataset size.

4. **Single-GPU constraint is tight**: 8–12 GPU-hours for ViT-S/16 on 100K images is marginal. Batch size ≥128 may be unachievable; behavior below documented range is unknown.

---

## Note on Speculative Risks (Not Included Above)

This research did **not find**:
- Published critiques arguing SSL is fundamentally oversold (though domain-specific evidence suggests variable ROI)
- Documented failure modes of leJEPA on very small batches (<128)
- Explicit recommendations against using ViT-S on 100K-scale datasets (only against <100K)
- Specific leJEPA documentation of failure on domain-shifted datasets (only indirect evidence from histopathology evaluation)

These are areas of genuine uncertainty, not documented risks. Claims beyond the sourced findings above should be treated as speculation.
