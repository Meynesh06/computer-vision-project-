# Verification Pass: jepa-vs-traditional-cv-ml-cited.md

Reviewer (adversarial verification) pass over `outputs/.drafts/jepa-vs-traditional-cv-ml-cited.md`.
Scope: correctness/rigor of the argument and its evidence, balance, and overstated confidence — not prose.
Date: 2026-07-02.

Method note: the brief already carries a citation-verification pass (its own "Verification Notes").
This pass does NOT re-litigate every citation; it stress-tests the *argument* built on top of them,
the internal consistency between body and Verification Notes, and the balance of the case. I ran a
small number of independent spot-checks (below) to calibrate; the primary findings are logic/consistency-based.

Independent checks I performed this pass:
- Re-fetched the leJEPA abstract (arXiv:2511.08544): confirms the 79% ViT-H/14 ImageNet-1K linear-probe
  figure; the abstract does NOT contain the Galaxy10 in-domain-vs-transfer comparison, batch size, or epoch
  recipe — those live in the paper body and remain effectively single-source and not re-verifiable from the
  landing page.
- WebSearch on "Virchow2 / UNI / GigaPath / NCT-CRC-HE-100K accuracy": returned widely varying,
  protocol-dependent numbers (e.g., "95.8%", "99.01%") and benchmark studies that score these models on
  aggregate metrics (Virchow2 0.82, UNI2 0.79, Prov-GigaPath 0.787 avg) rather than the specific single
  NCT-CRC accuracy figures in the brief's Q6 table. This corroborates the brief's own admission that the
  UNI/GigaPath figures are not independently confirmable — and shows the *Virchow2 96.7%* and *Phikon-v2
  95.5%* figures are equally un-pinned, not just the two the brief flags.

---

## FATAL findings (must fix before delivery)

### F1. Internal contradiction: Q6 claims foundation-model results are "already verified on this exact dataset"; the Verification Notes say they are NOT
- Q6 (line ~93) states the foundation-model path comes "with results **already verified on this exact
  dataset** (rather than leJEPA's zero medical benchmarks)." The Executive Summary point 2 similarly presents
  "achieve 95-97%+ accuracy on this exact dataset" as established fact.
- The brief's own Verification Notes (Items in the Q6 table and "What could NOT be verified") say the opposite:
  UNI 95.4% and GigaPath 95.9% are **"not independently confirmed"** (paywalled); Virchow2's 96.7% is only
  "not contradicted" (not confirmed); Phikon-v2 95.5% was not checked either way.
- Net: **none of the four foundation-model accuracy figures were independently confirmed**, yet the body twice
  asserts them as "verified." My WebSearch this pass could not pin any of the four to a clean primary source.
- Why FATAL: "verified" is a load-bearing provenance word in this workspace's conventions, and this is a
  decision document. Asserting "already verified" over unverified numbers is exactly the kind of claim that
  embarrasses the brief if a reader opens the sources. Fix: downgrade "already verified" to "reported (not
  independently re-verified; UNI/GigaPath paywalled)" and soften Exec Summary point 2 accordingly.

### F2. A likely-fabricated citation (hub.ai.uky.edu) is still printed in the body Risks section
- The brief's own verification found the hub.ai.uky.edu page does **not** contain the "MAE/DINOv3 outperform
  I-JEPA with/without SIGReg" comparison attributed to it, and labels it "likely fabricated / unsupported."
  Yet the claim remains in Q7 Risks (line ~102) and Open Questions (line ~120), merely re-labelled
  "unconfirmed / likely unsupported."
- This is an *anti-leJEPA* claim. Leaving a likely-fabricated citation in the delivered artifact — in the
  section that makes the case against the method — is a provenance defect regardless of hedging. The brief's
  own Verification Notes (Item 1, and "What was found to be likely fabricated") recommend removal.
- Why FATAL: the task explicitly asks whether this should be removed rather than flagged; the honest answer is
  **yes, remove it**. A final decision artifact should not carry a claim its own verification believes was
  hallucinated. Keeping it — even hedged — imports a fabricated data point into the argument. Fix: delete the
  sub-claim from Q7 and Open Questions; if the segmentation-vs-classification / bone-vs-colorectal caveat is
  worth keeping, restate it as "no JEPA/leJEPA histopathology result was found at all," with no citation to a
  claim that does not exist on the page.

---

## MAJOR findings (should fix)

### M1. Executive Summary verdict overreaches relative to the body ("very likely the wrong engineering choice")
- The verdict asserts leJEPA "is **very likely the wrong engineering choice**." The body supports a weaker
  claim: leJEPA is *probably not the highest-accuracy or best-evidenced choice* for this dataset. Those are not
  the same. "Wrong" implies it would fail or is a mistake; the evidence shows only that (a) supervised/transfer
  already scores high and (b) cheaper alternatives exist — i.e., leJEPA is likely *suboptimal for accuracy*,
  not *wrong*.
- The body itself repeatedly undercuts "wrong": Q1 concludes leJEPA is theoretically sound and "not a red
  flag"; Q2 concedes the Galaxy10 in-domain result is "genuinely relevant"; Q4's own conclusion caveats that
  "none of this evidence involves leJEPA specifically" and its domain-mismatch argument "remains untested on
  this dataset"; Q5 admits the single most decision-relevant study (controlled SSL-vs-supervised on histology)
  was "not found."
- Critically, the null hypothesis the project was asked to test was about **accuracy saturation**, and the
  brief conflates "won't beat a supervised baseline on accuracy" with "wrong engineering choice." For a
  **portfolio project supporting a job application**, demonstrating SSL engineering competence can itself be
  the objective — which the brief only concedes in the final sentence of the Exec Summary, after the strong
  verdict has already landed. Fix: condition the verdict on the objective ("if the goal is maximizing
  accuracy/efficiency on this benchmark, leJEPA is likely suboptimal; if the goal is demonstrating SSL
  competence, it is defensible") and drop or soften "very likely the wrong engineering choice."

### M2. The brief is structurally one-sided; leJEPA's strongest counter-argument (Galaxy10 domain-mismatch) is under-developed AND not actually rebutted by the central argument
- Space asymmetry: the case *against* leJEPA occupies the entire Exec Summary, Q3/Q4 (the largest section),
  Q6, Q7, and the Falsification Check. leJEPA's best pro-argument — in-domain SSL beating large-model transfer
  on a small, visually-specialized dataset (Galaxy10) — gets ~3 sentences in Q2 and a line in Open Questions.
- Deeper problem (a logical gap, not just space): the brief's load-bearing rebuttal is "the dataset is 100%
  labeled, so SSL's label-efficiency advantage doesn't apply" (Q3/Q4). But the Galaxy10 result is **not a
  label-efficiency claim** — it is a *representation-quality-under-domain-shift* claim. The brief conflates the
  two throughout Q3/Q4 ("SSL's advantage = label scarcity"), so its central argument never actually engages
  leJEPA's actual mechanism of alleged benefit. The two are different axes; refuting one does not refute the
  other.
- The saturation argument *does* partially address it (little headroom for any method if supervised hits
  97.7%), and there is even stronger un-used evidence sitting in the brief: the pathology foundation models
  (themselves in-domain SSL) score 95–96.7%, **below** supervised EfficientNet-B0's 97.7% — which is direct
  evidence *against* the Galaxy10 pattern holding on this dataset. The brief never connects this, so it both
  undersells its own best rebuttal and leaves the domain-mismatch counter-argument standing more than
  necessary.
- Falsification Check is one-directional: it enumerates only anti-leJEPA conditions (all "confirmed") and
  never weighs the one pro-leJEPA data point (Galaxy10) as a partial counter. A genuinely adversarial
  falsification section should record what evidence *would* support leJEPA and whether any was found — it was
  (Galaxy10, theoretical soundness). Fix: engage the domain-mismatch argument on its own terms, use the
  foundation-model-vs-EfficientNet gap as the actual rebuttal, and make the Falsification Check two-sided.

### M3. A core pillar (finding #2: "faster, better-evidenced alternative exists") rests on accuracy figures that are all unverified
- Exec Summary finding #2 and Q6 present foundation models at "95–97%+ on this exact dataset" as the
  better-evidenced alternative. But per F1, all four figures (Virchow2 96.7%, UNI 95.4%, GigaPath 95.9%,
  Phikon-v2 95.5%) are unconfirmed against primary sources; two are paywalled, one is "not contradicted," one
  was not checked. My WebSearch could not independently confirm any of the four as the specific NCT-CRC number.
- The qualitative claim ("open pathology foundation models exist and perform strongly on colorectal
  classification") is well-supported (benchmark studies corroborate Virchow2/UNI/GigaPath as top performers).
  The *specific numbers* are not. Since finding #2 is one of three pillars and partly rests on "better-
  evidenced," the brief should either (a) source at least one figure to an open benchmark paper (e.g., the
  medrxiv/researchgate comprehensive benchmark surfaced in search), or (b) restate the pillar qualitatively
  and stop implying the numbers are established.

---

## MINOR findings (optional polish)

### m1. Cross-table comparison is not apples-to-apples
- The Q4 table is headed "CRC-VAL-HE-7K accuracy"; the Q6 table is headed "NCT-CRC-HE-100K accuracy." The
  brief compares numbers across the two (e.g., EfficientNet 97.7% vs. foundation models 95–96.7%) without
  flagging that eval set / protocol (linear probe vs. fine-tune, split, patient overlap) may differ. The
  conclusion likely survives, but the direct numeric comparison should carry a "protocols differ" caveat.

### m2. Exec Summary point 2 slightly overstates the fine-tuning tier
- Exec Summary: "95-97%+ ... via linear probing (~30 min) or LoRA fine-tuning (~3-4 hours)." The Q6 approach
  table attributes 97% to **full fine-tune (6–12 hrs)**, with linear probe at 94–95% and LoRA at 95–96%. So
  "97% via linear probe/LoRA" overstates by one tier. Align the headline range with the table.

### m3. Ignatov & Malivenko 2024 carries heavy single-source load
- The specific ranking "supervised EfficientNet-B0 > CTransPath > DINO > iBOT" and the RF-shortcut warning
  both come from one non-peer-reviewed arXiv preprint (2409.11546). The saturation conclusion is triangulated
  (Kather 2019, Voigt 2023), so it holds, but the *comparative ranking* claim is single-source and should be
  labelled as such (preprint, not peer-reviewed).

### m4. Virchow / Virchow2 row conflation
- The Q6 row is labelled "Virchow2" (arXiv:2408.00738) but the verification note and the 632M-vs-1.28B param
  discussion attach to the *original* Virchow (2309.07778). Param count is not load-bearing for any
  conclusion, so this is cosmetic, but the row mixes two models' provenance. The Virchow param-count
  discrepancy itself is adequately handled (flagged, not propagated into a claim) and needs no further action.

### m5. GigaPath citation-year inconsistency
- Already flagged in Verification Notes ("Nature 2024" label vs. `s41591-025-...` ID → 2025). Cosmetic; fix
  the year for tidiness.

---

## Items the brief already handles adequately (no action needed)
- Virchow 632M-vs-1.28B param discrepancy: flagged, not propagated into any conclusion. Adequate as-is
  (see m4 for cosmetic tidy).
- GitHub issue #17 / repo-rename (rbalestr-lab → galilai-group): verified via GitHub API, terminology nuance
  correctly noted. Solid.
- The four 2026 arXiv IDs: independently confirmed via arXiv API. Solid.
- Kather 2019, Ignatov 2024 numeric claims, Voigt 2023, Lunit benchmark, VICReg identity: directly re-fetched
  and confirmed. Solid.
- Honest "not found" admissions (no controlled SSL-vs-supervised histology study; no JEPA histopathology
  result) are a genuine strength and should be preserved.

---

## Overall verdict
**Not ready to deliver as-is.** The evidentiary spadework is strong and unusually honest, but two FATAL issues
must be fixed first: (F1) the body claims foundation-model results are "already verified" when the brief's own
notes say they are not, and (F2) a citation the brief believes is fabricated is still printed in the Risks
section. Beyond those, the brief tilts one-sided against leJEPA (M1/M2): the headline verdict ("very likely
the wrong engineering choice") is stronger than the hedged body earns, and leJEPA's best counter-argument
(the Galaxy10 domain-mismatch result) is both under-developed and never actually rebutted by the central
label-saturation argument — which addresses a different axis. Fixing F1/F2 and rebalancing M1/M2 (including
using the foundation-model-vs-EfficientNet gap as the real rebuttal) would make this a defensible, genuinely
adversarial artifact.
