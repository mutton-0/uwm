# Second scenario type (lead-vehicle braking): G / F① / C-hazard readouts for SimLingo

> Work order: [`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md), task two.
> Corpus and matching QC: [`lead_vehicle_brake_mining_report_en.md`](lead_vehicle_brake_mining_report_en.md).
> The single cross-scenario "did it generalize" adjudication: [`second_scenario_generality_report_en.md`](second_scenario_generality_report_en.md).

---

## Methods

**Candidate**: SimLingo; readable positions: 24 decoder layers (InternVL2-1B / Qwen2-0.5B, hidden 896). It belongs to the **pure transformer stack**.

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
| Primary CV-AUC(LB vs LBn) | 0.600 [0.505, 0.694], p = 0.0495 | — |
| Falsification floor CV-AUC(vs LBv) | 0.645 | — |
| **Primary − floor** | **−0.045** [−0.177, +0.081] | +0.035 [−0.026, +0.097] |
| 10 fold-assignment seeds | −0.033 ± 0.036 | see the G1 report |
| Random-direction / permutation floor | 0.485 / 0.489 | — |
| Peak layer | L0 | — |
| **Verdict** | **indeterminate** | indeterminate |

**Table 2. F① action-level counterfactual (architecture-neutral).**

| Quantity | Second scenario | G1 |
| --- | --- | --- |
| $b$(LB) [m/s] | −0.059 [−0.330, +0.211] | — |
| **b-AUC(LB vs LBn)** | **0.491 [0.383, 0.589]** | 0.534 [0.469, 0.592] |
| b-AUC(LB vs LBv, falsification floor) | 0.499 | — |
| **Verdict** | **indeterminate** (CI spans 0.5) | indeterminate |

**Table 3. C-hazard (pairing source = clean steady following ↔ ghost hard braking).**

| Quantity | Second scenario | G1 |
| --- | --- | --- |
| patch-ALL sufficient-cut-set check | +0.965 (must lie in [0.7, 1.3]) ⇒ **passes** | passes |
| Spearman(layer, recovery) | **−0.921** ⇒ **cascade** | cascade ρ=−0.997 / commitment L3 / $C_m$ n/a |
| $C_m$ = top-2 layer share | 0.182 [0.159, 0.207] (diffuse baseline 0.083) | see right column |
| Responsible-layer mode / normalized entropy | L0 / 0.484 | |
| Commitment layer (deepest layer with mean recovery ≥ 0.9) | **L8 / 24 (depth 0.38)** | |
| **Verdict** | **indeterminate** (cascade profile, so the $C_m$ formula's premise is structurally violated) | |

---

## Discussion

This report records numbers only. The cross-scenario "did it generalize" adjudication is given in one
place, [`second_scenario_generality_report_en.md`](second_scenario_generality_report_en.md), so that
the three candidate reports cannot drift apart from one another.
