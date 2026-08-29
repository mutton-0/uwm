# Headline Three-Line Evidence: Public Leaderboard Ranking vs the G/F/I/C Matrix vs a Real Post-Train Outcome

> Numerical source of truth: [`h1_three_line_evidence.json`](h1_three_line_evidence.json); table
> fragment: [`h1_three_line_evidence.md`](h1_three_line_evidence.md); generator:
> `scripts/h1_three_line_table.py`. Full methods for the sub-experiments are in
> [`axis_matrix_completion_report_en.md`](axis_matrix_completion_report_en.md),
> [`pilot_posttrain_report_en.md`](pilot_posttrain_report_en.md) and
> [`four_axis_evidence_summary_en.md`](four_axis_evidence_summary_en.md).
> **No arbitrarily weighted composite score is computed**; the reason is given in Discussion §4.

---

## Motivation

The central claim of this work is that **public leaderboard rankings may become disconnected from
real deployment rankings, and that the G/F/I/C axes localize which link of the causal chain the
disconnect occurs in.** A complete demonstration requires three lines side by side: ① public
leaderboard scores, ② this framework's four-axis matrix, and ③ a real post-training outcome. This
report places all three in one table and annotates **comparability** cell by cell — because in the
course of this work comparability itself turned out to be the strongest single argument.

---

## Method

**Line 1 (public leaderboard)** uses the published scores supplied by the project owner, quoted
verbatim with no conversion: SimLingo's Driving Score on CARLA Leaderboard 2.0 = 6.87;
DiffusionDrive's PDMS on NAVSIM navtest = 88.1. This report deliberately does **not** attempt to
normalize or map them onto a common scale — their incomparability is the fact being reported.

**Line 2 (four-axis matrix)** contains readouts measured for both models on **one and the same
nuScenes ghost-probe stimulus set** (the G1 corpus plus the N1 D2a/D2b/D2c/D2cV negative system)
under **one and the same statistical protocol** (scene-level three-way splits or K-fold, scene-level
bootstrap, three-state adjudication, per-layer null distributions). Every cell carries an additional
**cross-model comparability** column, with a comparable substitute supplied wherever a cell is not
comparable.

**Line 3 (pilot post-train)**, detailed in `pilot_posttrain_report_en.md`: SimLingo, fixed budget
(only the 229k-parameter `speed_wps_head` is trained), three arms (baseline / task-only /
task + coupling), with both the behavioural score and the coupling quantity itself read out on
held-out scenes.

---

## Results

**Table 1. Line 1 — public leaderboard scores. The two numbers are not on the same yardstick.**

| Model | Benchmark | Metric | Score | Range | Protocol | Native domain |
| --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA Leaderboard 2.0 | Driving Score | **6.87** | 0–100 | closed-loop simulation (CARLA) | CARLA (sim) |
| DiffusionDrive | NAVSIM navtest | PDMS | **88.1** | 0–100 | pseudo-closed-loop (non-reactive log replay) | NAVSIM / OpenScene (real)† |

† The training domain of the checkpoint used here (`diffusiondrive_sim_navhard.ckpt`) comes from its
name and the operator's account and is **not independently verified**.

**Table 2. Line 2 — the G/F/I/C matrix, measured on one and the same stimulus set with one and the same protocol.**

| Axis | Readout | SimLingo | DiffusionDrive | Cross-model comparable | Verdict |
| --- | --- | --- | --- | --- | --- |
| **G** | primary − own D2cV falsification floor | +0.035 [−0.026, +0.097] | +0.009 [−0.047, +0.065] | yes | both indeterminate |
| **F**① | action-level counterfactual b-AUC(A vs D2a) | 0.534 [0.469, 0.592] | 0.553 [0.486, 0.623] | yes | both indeterminate |
| **F**① | action change induced by hazard frames, b(A) [m/s] | **+0.307 [+0.112, +0.500]** | +0.010 [−0.030, +0.045] | within-model | reacts non-specifically vs does not react |
| **F**② | steering ±α full-ladder slope [m/s per σ] | −0.0405 ($v_{brake}$) | −0.00000 ($v_{hazard}^{dd}$); in-house upper bound $v_{brake}^{dd}$ only +0.00002 | **no** | instrument has no resolving power on DiffusionDrive |
| **I** | $I_m = 1 - D_{L^*}$ (representation side) | **0.606** | 0.203 | yes | SimLingo > DD |
| **I** | behavioural domain sensitivity | 0.346 | **0.115** | yes | **DD > SimLingo (reversed)** |
| **C** | $C_m$ = top-2 layer share of recovery | 0.176 (baseline 0.083) | 0.858 (baseline 0.250) | **no** | see shape statistics |
| **C** | recovery-profile Spearman(layer, recovery) | **−0.997** (cascade) | **+0.881** (interior peak) | yes | failures **enter at different places** |
| **C** | responsible-layer mode / normalized entropy | **L0** / 0.279 | **L6** / 0.685 | yes | vision input interface vs deep fusion stage |

**Table 3. Line 3 — pilot post-train on SimLingo under a fixed budget (229k trainable parameters), evaluated on held-out scenes.**

| Arm | Training objective | b-AUC(A vs D2a) | 95% CI | $\Delta_{1\sigma}$ [m/s] | $\cos(g,\hat v_{hazard})$ |
| --- | --- | --- | --- | --- | --- |
| A0 | not trained (baseline) | 0.544 | [0.461, 0.621] | +0.091 | +0.046 |
| A1 | task + distillation ($\lambda$ = 0, attribution control) | 0.592 | [0.496, 0.680] | −0.076 | −0.016 |
| A2 | task + distillation + **coupling** | 0.593 | [0.499, 0.679] | **−0.301** | **−0.068** |

Contrasts: b-AUC **A2 − A0 = +0.051 [−0.055, +0.148]**; **A2 − A1 = +0.001 [−0.004, +0.007]**;
A1 − A0 = +0.050. Coupling: $\Delta\cos$(A2 − A0) = **−0.113**;
$\Delta\Delta_{1\sigma}$(A2 − A0) = **−0.392** m/s.
**Verdict: partially holds** — the coupling is verifiably changed; the behavioural score shows no
significant increment over the task-only control.

> **Instrument side.** Every cell of line 2 was produced on the same stimulus set under the same
> statistical protocol, with comparability annotated cell by cell; the two non-comparable cells
> (F② and C's $C_m$) are supplied with comparable substitutes (the action-level counterfactual and
> the profile-shape statistics). Line 3 includes an attribution control arm, so a behavioural
> improvement can be attributed or have attribution denied.
> **Specimen side.** The three lines yield three non-overlapping pictures: the public scores cannot
> be ranked at all; the four-axis matrix is indeterminate for both models on G and F but yields
> **two different repair prescriptions** on C; the pilot shows that link F can genuinely be repaired
> without that repair buying any behavioural gain.

---

## Discussion

**1. The most important content of line 1 is not "who scores higher" but "ranking is undefined on
public scores".** There is no meaningful ordering between 6.87 and 88.1: different benchmarks (CARLA
Leaderboard 2.0 vs NAVSIM navtest), different domains (simulation vs real log replay), different
protocols (closed loop vs non-reactive pseudo-closed loop) and different metric definitions
(Driving Score = route completion multiplied by an infraction penalty product; PDMS = a weighted
combination of sub-scores). Placing them side by side yields exactly one piece of information:
**the public score system cannot even define a ranking.** This is not a limitation of the present
work but **the first argument for its central claim** — if two candidates cannot be ranked at all,
then the "leaderboard ranking" in the phrase "leaderboard ranking is disconnected from deployment
ranking" does not exist, and model selection must rest on something else. The four axes of line 2
are measured for both models on **one stimulus set under one protocol** and therefore align cell by
cell — **this is the first and most basic increment this framework provides over public scores.**

**2. Line 2's conclusion: the axes do align, but in this round only the C axis produces a
substantive cross-model distinction.** The G axis is indeterminate for both models (both difference
CIs straddle zero, and the two poolings order the models oppositely); the comparable F-axis readout
(action-level counterfactual) is likewise indeterminate for both. The one axis that distinguishes
them is **C**: the two models' domain failures **enter at entirely different places**. SimLingo's
recovery profile decreases monotonically from L0 (Spearman −0.997), i.e. the failure has already
entered at the **vision-encoder-to-LLM interface**; replacing layer 0's vision tokens with the
sim-side ones returns behaviour completely to the sim side (patch-ALL recovery = +1.004).
DiffusionDrive's profile has an **interior peak** (Spearman +0.881; recovery at L0/L1 is essentially
zero; responsibility concentrates at L6). **The same diagnostic yields two entirely different repair
prescriptions: fix the visual front end in one case, attach LoRA to the deep fusion stage in the
other.** No public score can supply this: neither 6.87 nor 88.1 contains any information about where
the failure is or what fixing it would cost.

**3. Line 2 also yields a methodological conclusion: the axes' operationalizations are not all
portable.** Operationalized as a steering effect size, the F axis is measurable and even analytically
predictable on a continuous regression head (analytic/empirical ratio 0.98–1.07), but **entirely
unmeasurable within ±32σ** on an anchored diffusion head — where even the by-construction-effective
$v_{brake}^{dd}$ cannot move the output. Operationalized as a top-2 layer share, the C axis holds on
an interior-peaked profile but has its **premise violated** on a cascade profile. This table
therefore marks those two rows as non-comparable and **refuses** to fold them into a single number
with any weighting.

**4. Why there is no total score.** The four axes differ in units, adjudication state and
comparability: G and I are within-model normalized readouts that align across models; F's steering
protocol is dimensionally different between the two models; C's top-2-share formula has its premise
violated on SimLingo. Weighting these four cells into one number amounts to **treating
"non-comparable" and "indeterminate" as either zero or as the median**, and either treatment makes
the provenance of the final ranking untraceable — as soon as someone asks "why is A ahead of B?",
the answer lands on the weights rather than on the evidence. This work would rather deliver a matrix
with blanks and "not comparable" labels than a scalar that looks clean but cannot be audited.

**5. Line 3 tightens the claim to the strength it should have.** The pilot demonstrates that the
F-axis diagnosis points at an internal quantity that is **interventionable, measurable and verifiably
changed** (the coupling moved by a factor of 3.3), which is among the strongest available forms of
evidence for the claim that the four axes localize a link of the causal chain. The same experiment
also demonstrates that **changing that link does not automatically improve deployment behaviour**
(A2 − A1 = +0.001 [−0.004, +0.007], with three parallel behavioural readouts agreeing). The accurate
claim is therefore: **the four axes localize which link the disconnect occurs in**, and **not** that
four-axis scores predict post-training gains. Establishing the latter requires line 3 to be genuinely
closed (a larger budget, or a closed-loop consequence metric), which this round did not do.

**6. The composite picture across the three lines.** Public scores: no ranking possible. Four-axis
matrix: G and F indeterminate, C yields two different repair prescriptions. Pilot post-train: the
link the diagnosis points at can be repaired, but at a fixed small budget the repair produces no
behavioural gain. Taken together the three lines support a more cautious and more defensible version
of the original claim: **public scores are insufficient to produce a selection ranking, and the four
axes can deliver an auditable per-link diagnosis under a common protocol; but the step from
"diagnosis" to "post-training gain" has not been established in this round.**

---

## Record of self-correction

1. **The work-order file was missing** (`docs/headline_three_line_pilot_workorder.md` is not on disk;
   see §HL/A23). This report was executed against the specification given in the project owner's
   message, which is transcribed item by item in §HL/A23 for later comparison. Should the original
   work order prescribe a different table structure or different criteria, the report must be
   rearranged accordingly.
2. **The public leaderboard scores are quoted verbatim and were not independently reproduced in this
   round.** 6.87 and 88.1 were supplied by the project owner; neither CARLA Leaderboard 2.0 nor
   NAVSIM navtest was re-run here. The only use this report makes of the two numbers is to argue that
   they are **incomparable**, and that argument depends only on the definitional differences between
   the two benchmarks, not on the accuracy of the values.
3. **Line 3 covers SimLingo only.** No pilot post-train was run for DiffusionDrive, because this
   round established that its steering instrument has no resolving power (§HL/A25) while the coupling
   loss is defined through the measurable injection response $\Delta_{1\sigma}$; on that model the
   quantity is identically zero, so the coupling term carries no gradient signal. This is dictated by
   the architecture, not by the budget.
