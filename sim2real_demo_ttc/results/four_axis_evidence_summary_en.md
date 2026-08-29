# Four-Axis Evidence Synthesis: Where in the Causal Chain the Leaderboard–Deployment Disconnect Occurs

> This document is the final synthesis required by
> `docs/four_axis_proof_experiment_workorder.md` §6 and is written for direct use in the paper's
> Experiments chapter. Full methods and item-level readouts for the four sub-experiments are in
> [`g_positive_calibration_report_en.md`](g_positive_calibration_report_en.md),
> [`f_axis_ttc_gradient_report_en.md`](f_axis_ttc_gradient_report_en.md),
> [`i_axis_domain_pairing_report_en.md`](i_axis_domain_pairing_report_en.md) and
> [`c_axis_concentration_report_en.md`](c_axis_concentration_report_en.md). The axis-naming map
> (legacy R/S → G/F) is in [`axis_naming_alignment.md`](axis_naming_alignment.md); all
> pre-registrations and amendments are in [`amendments.md`](amendments.md).

---

## 1. The claim and this round's burden of proof

The central claim of this work is that **public leaderboard rankings (e.g. nuScenes/NAVSIM
open-loop scores) may become disconnected from real deployment rankings, and that the four axes
G (Grounding), F (Faithfulness), I (Invariance) and C (Concentration) localize which link of the
causal chain that disconnect occurs in** — perception not grounded (G), grounded but not driving
the action (F), driving it but unstable across domains (I), or failing diffusely and therefore
expensive to repair (C).

A complete proof of that claim requires a three-way comparison (selection-protocol §6, acceptance
item 0): the public leaderboard ranking, this framework's ranking, and the ranking of real
post-training outcomes. **This round does not supply the third line.** Its burden of proof is
therefore restricted to a narrower question that the present assets can answer:

> **Are the four axes separately measurable, mutually non-redundant, and able to yield information
> that no single scalar score can structurally provide?**

All evidence below serves that restricted question; inferences beyond it are explicitly marked as
outstanding in §5.

---

## 2. Master table

**Table 1. Four-axis readouts for two end-to-end driving policies, all measured on stimuli and protocols held identical across models.**

| Axis | Readout (definition) | SimLingo | DiffusionDrive | Verdict |
| --- | --- | --- | --- | --- |
| **G** Grounding | CV-AUC(A vs geometry-balanced negatives D2a), `vision_mean` | 0.515 [0.467, 0.562], $p$ = 0.547 | 0.559 [0.512, 0.606], $p$ = 0.015 | — |
| **G** | primary − own D2cV falsification floor (single run) | +0.035 [−0.026, +0.097] | +0.009 [−0.047, +0.065] | **indeterminate** |
| **G** | as above, across 10 CV fold-assignment seeds | +0.002 ± 0.024 | +0.012 ± 0.032 | **indeterminate** |
| **G** | primary − label-permutation null (`vision_mean` / `region_mean`) | +0.021 / +0.056 | +0.065 / +0.024 | **orderings reverse across poolings ⇒ no robust directional difference** |
| **F** Faithfulness | dose slope $d\Delta v/d\alpha$ of injecting $v_{brake}$ (@L22, query tokens; **pre-registered +α-only protocol**) | **−0.0468 [−0.0517, −0.0421]** | not measured | SimLingo **responds** |
| **F** | the same slope predicted analytically via the head Jacobian (compared on the matched ±α full-ladder slope, −0.0405 [−0.0451, −0.0366]) | −0.0432 ± 0.0084 (ratio **1.07**, 7% discrepancy) | not applicable (diffusion action head) | **H3 partially holds** |
| **F** | $\cos(v_{brake}^{obs},\, v_{brake})$ (observational vs injection) | +0.0137 [−0.0487, +0.0644], $p$ = 0.645 | not measured | **indeterminate** (colinearity ≤ 0.064) |
| **F** | $\rho$(proj$_{v_{brake}}$, real human $a_{brake}$) | +0.083, $p$ = 0.011 | not measured | significant, correct sign, weak |
| **I** Invariance | $I_m = 1 - D_{L^*}$ (representation side) | **0.606**, $D_{L^*}$ = 0.394 [0.374, 0.419] | 0.203, $D_{L^*}$ = 0.797 [0.756, 0.840] | order SimLingo > DD |
| **I** | behavioural domain sensitivity $\mathbb{E}\lvert\Delta v_{cmd}\rvert/\overline{v_{cmd}}$ | 0.346 [0.249, 0.454] | **0.115 [0.080, 0.153]** | order **DD > SimLingo (reversed)** |
| **I** | interference angle $\lvert\theta_{L^*}\rvert = \lvert\cos(v_{domain}, v_{hazard})\rvert$ | 0.026 (reference $1/\sqrt{896}$ = 0.033) | 0.001 (reference $1/\sqrt{256}$ = 0.063) | both near-orthogonal |
| **C** Concentration | $C_m$ = top-2 layer share of recovery | not measured (no equivalent patching output) | **0.858 [0.771, 0.939]** (diffuse baseline 0.250) | **PASS** |
| **C** | causal peak layer vs divergence peak layer | not measured | causal L6; JS / 1−CKA peak L7; separability peak L5 | the three do not coincide |

**Table 2. Cross-distribution generalization of frozen directions (Stage E red line 1; every cell uses the frozen direction and frozen layer, with both the positive and the negative class absent from that direction's fitting set).**

| Frozen direction | Extraction domain / contrast | Held-out re-test | AUC | 95% CI | Same-layer random-direction floor | Excess |
| --- | --- | --- | --- | --- | --- | --- |
| $v_{hazard}^{clean}$ (nuScenes, region_mean, $L^*$ = 9) | nuScenes A vs D2a | nuScenes B vs D | 0.543 | [0.473, 0.614] | 0.518 ± 0.088 | +0.026 |
| $v_{hazard}^{clean}$ | — | nuScenes C vs D | 0.501 | [0.452, 0.549] | 0.516 ± 0.043 | −0.016 |
| $v_{hazard}^{clean}$ | — | **CARLA Hcar vs Ncar (cross-domain)** | 0.544 | [0.524, 0.565] | 0.483 | +0.061 |
| $v_{hazard}^{carla}$ (CARLA, region_mean) | CARLA Hcar vs Ncar (in-domain CV-AUC **0.808**) | **nuScenes A vs D2a (cross-domain)** | **0.490** | [0.443, 0.537] | 0.491 | **−0.001** |
| $v_{danger}^{lang}$ (language pairs, $L^*$ = 10) | offline language stimulus set | nuScenes B vs D | 0.579 | [0.509, 0.649] | 0.492 ± 0.068 | +0.087 |
| $v_{brake}$ (behavioural labels, $L^*$ = 22) | the model's own braking residual | nuScenes B vs D | 0.501 | [0.431, 0.571] | 0.495 ± 0.044 | +0.006 |
| $v_{hazard}^{clean,vision}$ (nuScenes, vision_mean, $L^*$ = 21) | nuScenes A vs D2a | **tier_m independent mining round, A vs D (no geometric matching)** | 0.673 | [0.637, 0.708] | 0.505 | +0.168 † |
| as above | — | n1_d2 A vs **geometry-matched** D | 0.517 | — | — | ≈ 0 † |

† These two rows must be read as a pair: the tier_m mining round predates the `bbox_xyxy` field and
therefore has **no geometry-balanced negatives**, so only unmatched class-D negatives are available;
the same frozen direction attains only 0.517 against n1_d2's geometry-matched D. The gap of
**+0.156** is the magnitude of the geometric confound, consistent with this project's earlier N1
finding. The "independent mining round" cell is therefore adjudicated **indeterminate**, not passed.

---

## 3. Three empirical results supporting the central claim

### 3.1 A single scalar cannot carry the representational and the behavioural side at once (I axis)

On **one and the same** rendering-domain pairing (CARLA engine rendering ↔ world-model
photorealistic re-rendering; identical scene, geometry, actors and ego ground truth, with rendering
style the only variable), the two models' representational and behavioural readouts produce
**exactly opposite orderings**: on the representational side $D_{L^*}$ favours SimLingo
(0.394 vs 0.797), while on the behavioural side the domain sensitivity favours DiffusionDrive
(0.115 vs 0.346), and both pairs of scene-level bootstrap CIs are non-overlapping. Any procedure
that compresses "domain robustness" into one number must therefore pick one of these orderings and
discard the other. **This is a direct, reproducible instance of "a single score is insufficient to
characterize deployment robustness", and it depends on no assumption about any leaderboard.**

### 3.2 The information is in the representation but does not drive the action (F axis)

SimLingo's driving-query representation **linearly carries** time-to-collision information: ridge
regression with TTC as target attains a held-out $r$ = +0.320 under scene-level GroupKFold
(scene-level permutation $p$ < 1.0 × 10⁻³). Yet the cosine between that direction and the model's
own brake-driving axis $v_{brake}$ is −0.005, dead centre of the random null; and the observational
direction fitted to real human deceleration $a_{brake}$ has cosine +0.014 with $v_{brake}$
(95% CI [−0.049, +0.064], i.e. colinearity bounded by $\lvert\cos\rvert \le 0.064$, at most 0.4% of
shared variance). Meanwhile $v_{brake}$ **is itself causally effective**: injecting it produces a
dose slope of −0.0468 [−0.0517, −0.0421] m/s per α, independently reproduced by an **analytic
prediction** through the action-head Jacobian — on the matched ±α full-ladder slope, analytic
−0.0432 ± 0.0084 versus measured −0.0405 [−0.0451, −0.0366] (ratio 1.07, a 7% discrepancy;
event-level Pearson $r$ = +0.526); for a second deep direction, $v_{hazard}^{clean}$@L23, the ratio
is 0.98 with event-level $r$ = +0.993. **In other words, the axis that moves the action and the axis that encodes
the hazard are geometrically unrelated** — precisely the representation-level signature of causal
confusion (de Haan et al. 2019), and precisely the link an open-loop score cannot see: an open-loop
score checks whether the action is close to a reference trajectory, never what drives the action.

(Restriction: a single-frame model is structurally blind to object motion and TTC contains a
velocity term, so a substantial part of $r$ = +0.320 should be attributed to the correlation
between TTC and **distance**. The argument is unaffected: whether the direction carries TTC or
distance, it is orthogonal to the brake axis either way.)

### 3.3 Repair cost is orthogonal to performance metrics and readable only by intervention (C axis)

DiffusionDrive's trajectory degradation under the rendering-domain shift is causally highly
concentrated: $C_m$ = 0.858 [0.771, 0.939], far above the diffuse baseline of 0.250, with the top-1
layer alone accounting for 0.618 and the modal responsible layer being the deep fusion stage of the
encoder, **L6** (5 of 12 degraded scenes). Yet three quantities that all *look* like localization
give **three different peak layers**: attention divergence JS peaks at L7, feature divergence
1 − CKA peaks at L7, t-SNE separability peaks at L5, while causal recovery peaks at L6 — where JS is
close to its minimum across layers. **"How much performance was lost" (which a leaderboard can
report) and "where the loss enters and what fixing it costs" (which it cannot) are orthogonal
questions**, and only intervention (activation patching with paired input swapping) answers the
latter; divergence, CKA, or correlation with a downstream score cannot substitute for it.

---

## 4. A negative result that must be stated alongside: the G axis currently lacks the resolution to adjudicate grounding

T-G is the only methodological positive-calibration experiment in this set, asking whether the
G-axis yardstick can read "present" in a model that very probably has grounding. The answer is
**indeterminate**: DiffusionDrive's "primary minus own D2cV falsification floor" = +0.009 with a
scene-level bootstrap 95% CI of [−0.047, +0.065], crossing zero; across 10 fold-assignment seeds it
is +0.012 ± 0.032, **an sd larger than the effect**. Across the four protocol variants
(two models × two poolings) every pairwise gap in this difference lies within one sd and every value
is itself indistinguishable from zero, so no pair exhibits the "qualitative difference" the decision
rule requires. (These figures are the **stimulus-set-equalized** version: the first DiffusionDrive
cache had only 141 D2cV negatives against SimLingo's 212; caching the 116 missing events equalized
both sides to D2a = 283, D2cV = 212, D2bV = 78; see `amendments.md` §FA/A19.)

A stronger restriction also applies: even the ordering of perception-side signal strength is not
robust — the excess over the permutation null is DiffusionDrive +0.065 vs SimLingo +0.021 under
`vision_mean`, but DiffusionDrive +0.024 vs SimLingo +0.056 under `region_mean`, **an ordering that
reverses across poolings**. Accordingly, the use of G-axis readouts in this paper is restricted:
**they may report single-model signal strength within a specified pooling only, may not be used to
rank models, and may not on their own support an assertion about whether a model is genuinely
grounded in the hazard concept rather than in the mere presence of any VRU** — the latter
requires a distinguishable D2cV floor, not achieved this round. The bottleneck is localized to the
sample size of the D2cV negatives (141 matched negatives), not to pooling choice or model choice.

**The same limitation holds after changing the trigger criterion.** Replacing the positive class on
the SimLingo side from A (VRU emergence) with B (close cut-in) or C (generic TTC drop), with every
other protocol element unchanged: all three criteria read above their own permutation nulls
(+0.179 / +0.056 / +0.026), yet **none is distinguishable from its own D2cV falsification floor**
(differences −0.022 [−0.082, +0.043] / +0.003 [−0.062, +0.060] / +0.023 [−0.045, +0.091], all
crossing zero). This is therefore not a peculiarity of one scenario type but the general resolution
ceiling of the current G-axis protocol (see
[`direction_validity_report.md`](direction_validity_report.md) §5.3).

The cross-distribution re-tests (Table 2) tighten this restriction further: **the only direction
with strong in-domain readout power, $v_{hazard}^{carla}$ (CARLA in-domain CV-AUC 0.808), attains
AUC = 0.490 when transferred to nuScenes — exactly at its same-layer random-direction floor
(excess −0.001)**. That is, strong in-domain discriminability does not entail that the direction
encodes a transferable hazard concept. This negative result points in the direction of the central
claim — **a high score measured in one domain does not carry over to another** — while also showing
that what the framework's G axis currently delivers is "readable / not readable in this domain",
not yet "grounded / not grounded".

**The same direction is negated a third time on the manipulation side.** Injecting
$v_{hazard}^{carla}$ into SimLingo on nuScenes frames yields (i) a full ±α ladder slope of
**+0.0142 [+0.0077, +0.0216]** — injecting this "hazard direction" makes the model *accelerate*
rather than decelerate, the opposite of the causal expectation — and (ii) a specificity ratio
|Δlateral|/|Δv_plan| of **1.06** (compare $v_{brake}$ at 0.11 and $v_{hazard}^{clean}$@L23 at 0.17),
i.e. the lateral response is as large as the longitudinal one. It is therefore not a
"sign-reversed hazard axis" but **a non-specific perturbation direction**. Readout AUC 0.808,
cross-domain AUC 0.490, and a sign-reversed non-specific steering response: **three independent
lines of evidence converge on the conclusion that high in-domain discriminability can contain no
transferable hazard semantics at all.**

---

## 5. What remains unproven (listed explicitly to avoid over-claiming)

1. **The three-way comparison is incomplete.** This round contains no ranking of real
   post-training outcomes (the third line of selection-protocol §6 acceptance item 0), and did not
   re-measure either model's public leaderboard score. The proposition "leaderboard rank ≠
   deployment rank" is therefore shown here to be **measurable and decomposable**, not shown to
   **have occurred**.
2. **F and C were each measured on one model only** (F on SimLingo, C on DiffusionDrive), so
   Table 1 does not constitute a complete cross-model matrix over the four axes.
3. **The domain pairing is not a 3DGS reconstruction** but a world-model photorealistic
   re-rendering; the scope of the I-axis conclusion is written as "CARLA rendering ↔ world-model
   photorealistic rendering", with the definitive version deferred to real↔3DGS in the Madison
   phase.
4. **DiffusionDrive's native training domain is not independently verified**, so every
   interpretation depending on "which side is in-domain" is conditional.
5. **The C verdict is restricted to "localization holds"** and excludes "patching that layer
   suffices"; the latter requires re-testing with a consequence-related score (pseudo-closed-loop
   PDM-style margin) and was not done this round.

---

## Appendix: instrument-side / specimen-side two-line summary (internal reporting discipline; not for the paper body)

> **Instrument side.** Each of the four axes had its readout protocol independently calibrated once
> this round — G by cross-model positive calibration (conclusion: insufficient resolution, bottleneck
> at the D2cV sample size); F by the held-out predictive power of the observational direction itself
> (CV $r$ = +0.074, permutation $p$ = 0.040, instrument valid); I by within-model normalized ratios
> reported on both sides (representational and behavioural orderings are opposite, showing that
> neither side may be used alone); C by patch-ALL = +1.00 (12/12), establishing the readable layers
> as a sufficient cut set. Six amendments were registered during execution (DV/A12, A14, A15, A16,
> A17 and FA/A13), of which A14, A16 and A17 converted already-obtained positive results back into
> negative or indeterminate ones.
> **Specimen side.** SimLingo presents the combination "concept not readable, pathway drivable, but
> the drivable pathway is unrelated to hazard semantics"; DiffusionDrive presents "perception-side
> signal present but not clearing the falsification floor, large cross-domain representational drift
> with small behavioural drift, and highly localizable failure". Neither profile can be expressed by
> any single score.
