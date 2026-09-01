# NAVSIM/OpenScene independent corpus: G / F① / C-hazard readouts for LTF (Latent TransFuser)

> Work order: [`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md).
> Feasibility and corpus QC: [`navsim_openscene_mining_report_en.md`](navsim_openscene_mining_report_en.md).
> **The single cross-corpus "did it replicate" adjudication is in [`cross_corpus_generality_report_en.md`](cross_corpus_generality_report_en.md)** — this report records numbers only, so the three candidate reports cannot drift apart.

---

## Methods

**Candidate**: LTF (Latent TransFuser); readable positions: 8 encoder self-attention modules (`TransfuserBackbone`, `latent=True`). It belongs to the **TransFuser family**.

**Stimuli**: NAVSIM/OpenScene test split, 147 logs / 1880 scenes, with A = 397, D2a = 382 (after
caliper matching) and **D2cV = 134** (falsification floor). **The event-family names are identical to
G1's** (A / D2a / D2cV) because the criteria are identical too — the only thing that changed this
round is the data source.

**Protocol**: `detect_events` / `pick_frames` / `frame_record` / `n1_match` are all reused
**unchanged**, with thresholds identical field for field. The three readout scripts did not even need
their event-family arguments changed. **Only two things changed, both registered**: the data-access
layer (`ns1_navsim_geometry.py`) and the crop principal-point row, 450 → 560 (§NS/A46 — leaving it
would have silently contaminated every readout).

**Power**: this corpus has more positives (397) and more geometry-matched negatives (382) than G1
(291 / 283), and a smaller falsification floor (134 vs 212). The measured CI width of the
primary-minus-floor readout is comparable to G1's, so "indeterminate" here **cannot** be uniformly
attributed to insufficient power — see the cross-corpus report for the per-cell adjudication.

---

## Results

**Table 1. G axis (vision_mean, primary pooling), alongside the same readout on G1.**

| Quantity | NAVSIM independent corpus | G1 (nuScenes) |
| --- | --- | --- |
| Primary CV-AUC(A vs D2a) | 0.630 [0.592, 0.669], p = 3.1e-10 | — |
| Falsification floor CV-AUC(vs D2cV) | 0.620 | — |
| **Primary − floor** | **+0.011** [−0.038, +0.065] | **+0.070 [+0.017, +0.126]** |
| 10 fold-assignment seeds | +0.022 ± 0.022 | see the G1 report |
| region_mean sensitivity | +0.032 [−0.013, +0.080] | see the G1 report |
| Random-direction / permutation floor | 0.513 / 0.527 | — |
| Direction geometry purity ρ(proj, log area) | −0.186 (p = 2.1e-07) | — |
| **Verdict** | **indeterminate** | **PASS** (the only G-axis positive in this line of work) |

**Table 2. F① action-level counterfactual (architecture-neutral).**

| Quantity | NAVSIM | G1 |
| --- | --- | --- |
| $b$(A) [m/s] | +0.024 [+0.010, +0.036], **significant** | — |
| **b-AUC(A vs D2a)** | **0.546 [0.496, 0.593]** | **0.583 [0.519, 0.639]** (PASS) |
| b-AUC(A vs D2cV, falsification floor) | 0.510 | — |
| **Verdict** | **indeterminate** (CI spans 0.5) | see right column |

**Table 3. C-hazard (layer-wise activation patching on the clean↔ghost pairing).**

| Quantity | NAVSIM | G1 |
| --- | --- | --- |
| patch-ALL sufficient-cut-set check | +1.000 | see right column |
| Spearman(layer, recovery) | **+0.786** ⇒ **interior peak** | interior peak ρ=+0.929 / **L6** / no commitment layer / $C_m$ 0.789 PASS |
| $C_m$ = top-2 layer share | 0.752 [0.664, 0.844] (diffuse baseline 0.250) | see right column |
| Responsible-layer mode / normalized entropy | **L7** / 0.706 | |
| Commitment layer | **none** | |
| **Verdict** | **PASS: C significantly above the diffuse baseline** | |

---

## Discussion

See [`cross_corpus_generality_report_en.md`](cross_corpus_generality_report_en.md).
