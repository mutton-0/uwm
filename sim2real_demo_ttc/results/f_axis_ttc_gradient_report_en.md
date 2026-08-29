# T-F: Perception–Action Faithfulness (F axis) — Colinearity Test between an Observationally Fitted Brake Direction and the Injection-Calibrated $v_{brake}$

> Axis letters follow the G/F/I/C/D naming finalized on 2026-08-29 (mapping in
> [`axis_naming_alignment.md`](axis_naming_alignment.md): **the S axis of earlier files is the F
> axis of this report**). Pre-registration: [`amendments.md`](amendments.md) §FA.0; registered
> design deviation: §FA.1 deviation 2. Numerical artifacts: `f_axis_ttc_gradient.json` (primary),
> `f_axis_ttc_gradient_sens_{ttc,clean,both}.json` (sensitivity), `f_axis_abrake.json` (behavioural
> ground truth).

---

## Motivation

The F axis (Faithfulness) asks whether a policy's continuous control output is causally driven by
**the same features** that G established as grounded, or is instead produced along another pathway
that happens to correlate with correct behaviour inside the training distribution while being
causally independent of it (theoretical lineage: Causal Confusion in Imitation Learning,
de Haan et al. 2019). It follows G in the chain: seeing correctly does not entail that what was
seen drives the action.

This project already possesses an **injection-based** F-axis direction, $v_{brake}$: extracted at
the driving-query positions using the model's **own** braking residual as the label, and having
passed the full W1 steering battery. It carries a structural weakness, however — its label comes
from the model's own output, so it risks self-confirmation: it may be no more than "the principal
component of the model's output speed", unrelated to whether braking was in fact warranted.

This experiment tests it with a **fully independent observational route**: a latent direction
$v_{brake}^{obs}$ is regressed from **the real deceleration a human driver applied in the same
events** (recomputed from nuScenes `ego_pose`), and we ask whether it is colinear with
$v_{brake}$. The only thing the two routes share is the model's hidden state; one label source is
inside the model, the other outside it. Colinearity would give the F axis mutual corroboration by
two methods; orthogonality would show that the model's internal "brake axis" is unrelated to real
braking severity — **which is direct representation-level evidence for the disconnect between
leaderboard scores (fitting the model's own output distribution) and real deployment (which
requires real deceleration in response to real hazards).**

---

## Method

**Behavioural ground truth (strong ground truth).** The event ledger stores only two frames of
`ego_speed_mps` per condition, insufficient to characterize how hard the human braked after the
hazard emerged. We therefore return to the raw nuScenes `ego_pose` sequence (all CAM_FRONT
sample_data including sweeps, ≈ 12 Hz), take two central differences to obtain longitudinal
acceleration with one 3-point moving average after each, and record the **minimum** over the window
$[t_{emergence},\, t_{emergence}+3\,\mathrm{s}]$ as $a_{brake}$. All **7003** events are covered;
distribution: p5 = −6.83, p25 = −1.42, p50 = −0.84, p75 = −0.19, p95 = +0.18 m/s².
**This is the real behaviour of a human driver, not a proxy derived from the model's own output**,
satisfying red line 2 of the direction-discovery plan's Stage E.

**Features.** The **same position and same pooling as $v_{brake}$**: mean over the driving-query
segment (`query_mean`), all 24 layers. The **ghost** condition is used (time-aligned with the
$a_{brake}$ measurement window); clean and both are sensitivity arms. Usable events n = 2285
(677 scenes); the remainder are excluded because their cache predates the `query_mean` schema.

**Target and de-confounding.** The regression target is $-a_{brake}$ (negated so that larger means
harder braking, matching the sign convention of $v_{brake}$'s label 1 = "brakes more"), with
$v_{at\,emergence}$ **regressed out first**: without this the fit would recover a "speed axis"
(faster vehicles brake harder is trivially true, and the prompt states `Current speed` explicitly,
so speed information is already in the activations). The measured speed main effect is
$\beta$ = +0.252 per (m/s), explaining 8.3% of variance, and is removed.

**Regressor.** Per-layer ridge regression; $\alpha$ selected by **scene-level GroupKFold (4 folds)
within the training folds**, never looking at the primary readout. The primary layer is L = 22
($v_{brake}$'s own calibration layer, `t1q_axes.json: brake_layer = 22`).

**Pre-registered primary readout (single).**
$\cos(v_{brake}^{obs},\, v_{brake})\big|_{L=22}$, reported with $z$ and two-sided $p$ against a null
distribution of cosines from 200 random unit directions, and a 95% CI from **scene-level bootstrap
(500 resamples, refitting the direction on each resample)**. Everything else (other layers, other
conditions, TTC as target) is a sensitivity analysis.

**Instrument-side validity check (registered in advance).** If $v_{brake}^{obs}$ itself has no
held-out predictive power for $a_{brake}$, then "cos not significant" would be an instrument-side
conclusion (the direction was never fitted) rather than a specimen-side one (the direction does not
exist). We therefore also report the held-out CV correlation $r$ at L22 against a **scene-level
permutation null** (100 permutations).

**Three-state decision rule (registered in advance).** Significantly above the null → PASS;
not significant but same sign → indeterminate; opposite sign or inside the null → FAIL.

---

## Results

**Table 1. Colinearity between the observationally fitted brake direction $v_{brake}^{obs}$ and the injection-calibrated $v_{brake}$ at layer 22 of SimLingo (n = 2285 events, 677 scenes).**

| Arm | Regression target | Condition | $\cos(v_{brake}^{obs}, v_{brake})$ | 95% CI (scene bootstrap) | $z$ vs random null | $p$ (two-sided) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary** | $-a_{brake}$ (real human) | ghost | **+0.0137** | [−0.0487, +0.0644] | +0.35 | 0.645 | **indeterminate** |
| Sensitivity | $-a_{brake}$ | clean | −0.0611 | [−0.1044, +0.0111] | −1.96 | 0.050 | — |
| Sensitivity | $-a_{brake}$ | both | −0.0316 | [−0.0818, +0.0329] | −1.05 | 0.285 | — |
| Sensitivity | $-\mathrm{TTC}$ | ghost | −0.0047 | [−0.0577, +0.0423] | −0.22 | 0.865 | — |

Random-direction null: mean = +0.0024, sd = 0.0324 (theoretical $1/\sqrt{896}$ = 0.0334; empirical
and theoretical agree).

**Table 2. Instrument-side validity check: does the observational direction predict real human braking at all?**

| Arm | Regression target | held-out CV $r$ @L22 | scene-permutation null (mean ± sd) | $p$ | Conclusion |
| --- | --- | --- | --- | --- | --- |
| **Primary** | $-a_{brake}$ | **+0.0744** | −0.0025 ± 0.0438 | 0.040 | instrument valid (the direction was fitted) |
| Sensitivity | $-a_{brake}$, clean | +0.0653 | — | 0.070 | marginal |
| Sensitivity | $-a_{brake}$, both | +0.0723 | — | 0.050 | marginal |
| Sensitivity | $-\mathrm{TTC}$ | **+0.3204** | +0.0021 ± 0.0449 | < 1.0 × 10⁻³ | instrument strongly valid |

**Table 3. Supporting readout (not the pre-registered primary): does $v_{brake}$'s own projection predict real human braking?**

| Readout | Spearman $\rho$ | 95% CI (scene bootstrap) | $p$ |
| --- | --- | --- | --- |
| $\rho\big(\mathrm{proj}_{v_{brake}}(h_{22}),\; -a_{brake}\ \text{residual}\big)$ | +0.0237 | [−0.0372, +0.0886] | 0.258 |

> **Instrument side.** The observational direction has significant (if weak) held-out predictive
> power for **real human deceleration** (CV $r$ = +0.074, scene-level permutation $p$ = 0.040), and
> far stronger power when the target is TTC ($r$ = +0.320, $p$ < 1.0 × 10⁻³). The instrument-side
> explanation ("the direction was never fitted") is therefore excluded, and **"cos not significant"
> is a specimen-side conclusion**. The null's sd matches the theoretical $1/\sqrt{d}$, confirming
> the cosine test's null is correctly specified. The upper bound of the primary CI is +0.064, so
> the colinearity of the two directions is **bounded by $\lvert\cos\rvert \le 0.064$, i.e. they
> share at most 0.4% of variance**.
> **Specimen side.** SimLingo contains a direction from which real braking severity can be read
> linearly, and it contains a direction that drives the model to brake — **but these are not the
> same direction**. Its driving-query representation even encodes TTC linearly and rather well
> ($r$ = +0.320), and that TTC direction is likewise orthogonal to the brake axis (cos = −0.005).

---

## Discussion

**1. Update to the four-axis picture: the F axis fails to obtain two-method corroboration, and
this is a specimen-side fact.** The pre-registered primary readout gives $\cos$ = +0.014, inside
the random null; by the registered rule the verdict is **indeterminate** (same sign, not
significant). Unlike a generic "underpowered" outcome, however, this experiment's instrument-side
check **passes**: the observational direction was genuinely fitted and does predict real human
deceleration out of sample. The precise statement is therefore: **at this sample size we can
exclude moderate or greater colinearity between the two directions ($\lvert\cos\rvert \le 0.064$
at 95% confidence) but cannot exclude very weak colinearity.** That is substantially more
informative than "we found nothing".

**2. The sharpest finding: the model encodes TTC linearly, and that encoding is orthogonal to its
brake axis.** With TTC as target the held-out CV $r$ = +0.320 (permutation $p$ < 1.0 × 10⁻³), i.e.
the driving-query representation **does** linearly carry information about how close, or how soon,
the collision is; yet the cosine between that direction and $v_{brake}$ is −0.005, dead centre of
the null. This is the representation-level signature of causal confusion: **the information is
inside, but it is not what drives the action.** Translated to the core contribution: a model may
well score respectably on an open-loop leaderboard (because its actions correlate with correct
behaviour within the training distribution) while its actions are **not driven by the hazard
information it has already encoded** — a leaderboard score cannot distinguish these two situations;
the F axis can. (**A necessary restriction**: a single-frame model is structurally blind to object
motion, and TTC contains a velocity term, so a substantial part of the $r$ = +0.320 should be
attributed to the correlation between TTC and **distance** rather than to the model actually
computing TTC. This restriction does not affect the argument: whether that direction carries TTC
or distance, it is orthogonal to the brake axis either way.)

**3. Relation to the existing T1-L negative result: a third independent reproduction of the same
break.** Prior work reported that $v_{danger}^{lang}$ (the language concept direction) is close to
orthogonal to $v_{brake}$, and that its projection correlates with the behavioural quantity $b$
with a **sign opposite to the causal direction** ($\rho$ = −0.161, $p$ = 0.006). The present
experiment replaces the "concept end" from **language** with **real-world behavioural ground
truth**, and the break persists. The only element shared between the two tests is $v_{brake}$, so
this is not an artefact of one particular way of extracting a concept direction; rather,
**within this model's driving-query representation, the "brake-driving axis" genuinely lacks a
geometric relationship to hazard / deceleration semantics**.

**4. Consistency of the sensitivity analyses.** All four arms (ghost/clean/both × $a_{brake}$/TTC)
give cosines inside the null; the largest absolute value is 0.061, and the signs are inconsistent
(+0.014 / −0.061 / −0.032 / −0.005). Signs flipping around zero across protocol variants is itself
a direct expression of "no colinearity", not of "signal under one protocol".

**5. What this readout may not claim.** (i) The experiment **does not refute the causal status of
$v_{brake}$**: it passed the W1 battery, and injecting it does move behaviour (this round's Stage-D
analytic arm predicts $d\Delta v/d\alpha$ = −0.0432 ± 0.0084 against a matched-protocol measurement
(±α full ladder) of −0.0405 [−0.0451, −0.0366], a ratio of 1.07).
What is refuted is the stronger claim that it is colinear with real braking severity.
(ii) $a_{brake}$ comes from a second difference of ego_pose and carries quantization noise;
two 3-point moving averages suppress but do not remove it, so it dilutes correlations and
CV $r$ = 0.074 should be read as a lower bound. (iii) The event set was selected by ghost-probe
mining criteria and is not a random sample of naturalistic driving.

**6. Three-state verdict.** **Indeterminate** (registered rule: same sign, not significant), with
the effect-size bound $\lvert\cos\rvert \le 0.064$ (upper end of the 95% CI) reported alongside.

---

## Record of self-correction

1. **The regression target "real $a_{brake}$" had no ready field in the event ledger**; it was
   recomputed from nuScenes `ego_pose`, with the algorithm and window registered in
   `amendments.md` §FA.1 deviation 2 before any analysis was run.
2. **The first version of the script lacked the instrument-side validity check** and reported only
   the cosine and its null. On that version alone, "indeterminate" could not have distinguished
   "the direction does not exist" from "the direction was never fitted". The held-out CV $r$ with a
   scene-level permutation null was added during execution, together with the correlation between
   $v_{brake}$'s projection and $a_{brake}$ as a **supporting readout** (explicitly labelled as not
   primary and not a substitute for the pre-registered readout). This strengthened the design
   without altering any number already produced.
3. **The TTC-target arm shows far stronger instrument-side predictive power than the primary arm
   ($r$ = 0.320 vs 0.074, yet it was kept as a sensitivity analysis and not promoted to primary.**
   Reason: work-order §3 step 2 states explicitly that the pre-registered primary readout uses
   $a_{brake}$. Promoting it would be textbook post-hoc selection, even though doing so would make
   the argument in Discussion §2 stronger.

---

## One-sentence update to the causal-chain picture

> The F-axis readout rewrites the "grounding → action" link from *unverified* to **verified as
> broken**: the model contains a direction from which real braking severity can be read linearly
> (held-out CV $r$ = +0.074, $p$ = 0.040) and a direction that genuinely drives braking (dose slope
> −0.047), yet their colinearity is bounded at $\lvert\cos\rvert \le 0.064$ — a link an open-loop
> leaderboard score cannot see by construction, since it tests whether the action resembles a
> reference trajectory, never what drives the action.
