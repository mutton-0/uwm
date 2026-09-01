# Second scenario type (lead-vehicle braking): G / F① / C-hazard readouts for LTF (Latent TransFuser)

> Work order: [`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md), task two.
> Corpus and matching QC: [`lead_vehicle_brake_mining_report_en.md`](lead_vehicle_brake_mining_report_en.md).
> The single cross-scenario "did it generalize" adjudication: [`second_scenario_generality_report_en.md`](second_scenario_generality_report_en.md).

---

## Methods

**Candidate**: LTF (Latent TransFuser); readable positions: 8 encoder self-attention modules (`TransfuserBackbone`, `latent=True`). It belongs to the **TransFuser family**.

**Stimuli**: second-scenario corpus, LB = 103 / LBn = 48 / LBv = 31 across 154 scenes. LB / LBn / LBv
are **isomorphic to, but differently named from,** G1's A / D2a / D2cV.

**Protocol**: every formula, statistical convention and adjudication gate is **unchanged**; only the
event-family names are parameterized (`--pos LB --neg LBn --floor LBv`). This is what the claim "the
four-axis **definitions** transfer" means operationally — had a formula needed to change for the
scenario to run, the claim would already be refuted. G axis: direction fitted in-fold, peak layer
selected in-fold, scene-level 4-fold CV, primary readout = CV-AUC(LB vs LBn) minus its own LBv
falsification floor. F①: the architecture-neutral action-level counterfactual,
$b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$. C-hazard: layer-wise activation patching on
the clean (steady following) ↔ ghost (hard braking) pairing, with the patch-ALL sufficient-cut-set
self-check and the recovery-profile shape diagnostic.

**A power limitation to state up front**: this scenario's sample is one third to one sixth of G1's,
so CIs are about 1.5–2.5× wider. "CI spans zero" here is therefore **more a statement about power
than about failure to replicate**, and the body distinguishes the two case by case.

---

## Results

**Table 1. G axis (vision_mean, primary pooling), alongside the same readout on the G1 ghost-probe scenario.**

| Quantity | Second scenario (lead braking) | G1 (ghost probe) |
| --- | --- | --- |
| Primary CV-AUC(LB vs LBn) | 0.492 [0.392, 0.591], p = 0.868 | — |
| Falsification floor CV-AUC(vs LBv) | 0.459 | — |
| **Primary − floor** | **+0.032** [−0.098, +0.164] | **+0.070 [+0.017, +0.126]** |
| 10 fold-assignment seeds | +0.000 ± 0.067 | see the G1 report |
| Random-direction / permutation floor | 0.530 / 0.514 | — |
| Peak layer | L4–L5 | — |
| **Verdict** | **indeterminate** | **PASS** (the first G-axis positive in this line of work) |

**Table 2. F① action-level counterfactual (architecture-neutral).**

| Quantity | Second scenario | G1 |
| --- | --- | --- |
| $b$(LB) [m/s] | +0.004 [−0.013, +0.022] | — |
| **b-AUC(LB vs LBn)** | **0.560 [0.471, 0.651]** | **0.583 [0.519, 0.639]** |
| b-AUC(LB vs LBv, falsification floor) | 0.594 | — |
| **Verdict** | **indeterminate** (CI spans 0.5) | **PASS** |

**Table 3. C-hazard (pairing source = clean steady following ↔ ghost hard braking).**

| Quantity | Second scenario | G1 |
| --- | --- | --- |
| patch-ALL sufficient-cut-set check | +1.000 (must lie in [0.7, 1.3]) ⇒ **passes** | passes |
| Spearman(layer, recovery) | **+0.786** ⇒ **interior peak** | interior peak ρ=+0.929 / **L6** / $C_m$ 0.789 PASS |
| $C_m$ = top-2 layer share | 0.799 [0.738, 0.862] (diffuse baseline 0.250) | see right column |
| Responsible-layer mode / normalized entropy | **L6** / 0.709 | |
| Commitment layer (deepest layer with mean recovery ≥ 0.9) | **none** (no single layer reaches 0.9 recovery) | |
| **Verdict** | **PASS: C significantly above the diffuse baseline** | |

---

## Discussion

This report records numbers only. The cross-scenario "did it generalize" adjudication is given in one
place, [`second_scenario_generality_report_en.md`](second_scenario_generality_report_en.md), so that
the three candidate reports cannot drift apart from one another.
