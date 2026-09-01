# NAVSIM/OpenScene independent corpus: G / F① / C-hazard readouts for DiffusionDriveV2

> Work order: [`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md).
> Feasibility and corpus QC: [`navsim_openscene_mining_report_en.md`](navsim_openscene_mining_report_en.md).
> **The single cross-corpus "did it replicate" adjudication is in [`cross_corpus_generality_report_en.md`](cross_corpus_generality_report_en.md)** — this report records numbers only, so the three candidate reports cannot drift apart.

---

## Methods

**Candidate**: DiffusionDriveV2; readable positions: 8 encoder self-attention modules (fed real lidar; NAVSIM point-cloud reader, §NS/A47). It belongs to the **TransFuser family**.

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
| Primary CV-AUC(A vs D2a) | 0.573 [0.533, 0.613], p = 0.0004 | — |
| Falsification floor CV-AUC(vs D2cV) | 0.525 | — |
| **Primary − floor** | **+0.048** [−0.020, +0.111] | +0.025 [−0.044, +0.097] |
| 10 fold-assignment seeds | −0.003 ± 0.029 | see the G1 report |
| region_mean sensitivity | −0.001 [−0.054, +0.052] | see the G1 report |
| Random-direction / permutation floor | 0.541 / 0.537 | — |
| Direction geometry purity ρ(proj, log area) | −0.070 (p = 0.061) | — |
| **Verdict** | **indeterminate** | indeterminate |

**Table 2. F① action-level counterfactual (architecture-neutral).**

| Quantity | NAVSIM | G1 |
| --- | --- | --- |
| $b$(A) [m/s] | −0.078 [−0.206, +0.058], n.s. | — |
| **b-AUC(A vs D2a)** | **0.471 [0.421, 0.524]** | **0.556 [0.501, 0.613]** (PASS) |
| b-AUC(A vs D2cV, falsification floor) | 0.450 | — |
| **Verdict** | **indeterminate** (CI spans 0.5) | see right column |

**Table 3. C-hazard (layer-wise activation patching on the clean↔ghost pairing).**

| Quantity | NAVSIM | G1 |
| --- | --- | --- |
| patch-ALL sufficient-cut-set check | **+0.642 ✗ 自检不通过** | see right column |
| Spearman(layer, recovery) | **+0.667** ⇒ **interior peak** | interior peak ρ=+0.810 / L4 / patch-ALL **+0.552 ✗** / indeterminate |
| $C_m$ = top-2 layer share | 0.874 [0.764, 0.973]（名义值，不作判据） (diffuse baseline 0.250) | see right column |
| Responsible-layer mode / normalized entropy | **L4** / 0.595 | |
| Commitment layer | **none** | |
| **Verdict** | **indeterminate** (patch-ALL sufficient-cut-set check fails — the same failure as on G1) | |

---

## Discussion

See [`cross_corpus_generality_report_en.md`](cross_corpus_generality_report_en.md).
