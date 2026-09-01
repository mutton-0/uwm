# F-axis severity gradient: does response magnitude scale with hazard severity?

> Work order: [`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md), task one.
> **No new inference**: every $b$ comes from the existing F① caches (the same loaders as
> `f_axis_action_counterfactual.py`); every dose coordinate comes from the geometric/kinematic
> metadata in `mining/events_all.jsonl`.
> Numeric artifact: `f_axis_severity_gradient.json`; script: `scripts/f_axis_severity_gradient.py`.
> Autonomous decisions: [`amendments.md`](amendments.md) §SG/A41.

---

## Methods

### Pre-registered hypothesis (registered before measurement)

> A candidate that "reacts but non-specifically" ($b(A)$ significantly non-zero but of the same order
> as $b$(D2a)) is expected to show response magnitude $b$ that **does not scale significantly with
> approach speed** within the A-class events; LTF (which reacts specifically) is expected to show a
> stronger scaling relationship.

### Dose coordinates, and a design tension to state up front (§SG/A41)

The work order specifies **approach speed $v_{close}$ as the primary dose coordinate**. But all three
main-comparison candidates this round (SimLingo / LTF / DiffusionDriveV2) are **single-frame models**,
and "relative velocity is structurally unobservable to a single-frame model" is **the very premise**
on which the D2cV falsification floor rests (§CE/A34). "$b$ does not scale with $v_{close}$" is
therefore **expected by construction** for a single-frame model, and cannot on its own support the
conclusion "this candidate lacks a grading mechanism".

We do not change the work order's primary readout; we add a **required control**:

| Dose coordinate | Role | Observable to a single-frame model | Expected sign |
| --- | --- | --- | --- |
| $v_{close}$ = $d_{long}$ / TTC | **primary (as specified)** | **no** | + |
| $d_{long}$ (longitudinal distance) | **required control (added here)** | yes | − |
| TTC | sensitivity analysis (confounds distance and speed) | no | − |

Only reading the first two together separates two entirely different situations:

* no scaling on $v_{close}$ **and** scaling on $d_{long}$ ⇒ **a grading mechanism exists; the model
  is simply blind to the velocity dimension**;
* no scaling on either ⇒ **the grading mechanism itself is absent** (what the hypothesis intends).

Using $d_{long}$ as the dose proxy for single-frame models is pre-existing discipline, not invented
here: the earlier S1 curve already recorded "SimLingo is strictly single-frame and measurably blind
to speed ⇒ TTC cannot serve as the dose axis; use $d_{long}$".

$v_{close}$ is recovered exactly from the mining pipeline's own quantity: the pipeline defines
$\mathrm{TTC} = d_{long} / v_c$, so $v_{close} = d_{long} / \mathrm{TTC}$ *is* $v_c$, not a new
approximation.

### Statistical conventions

Scene-level bootstrap (multiple events in one scene are not independent; 5000 resamples); dose
coordinates winsorized at the pre-registered 99th percentile on both sides ($v_{close}$ has two
upper-tail track-estimation artifacts above 50 m/s; **only the dose coordinate is clipped, no sample
is deleted**); three-state adjudication: slope CI excluding 0 with the correct sign → grading shown;
CI including 0 → **indeterminate** (which may not be reported as "no grading"); wrong sign → reported
as such. Sensitivity analyses: Spearman $\rho$, isotonic $R^2$, and the Jonckheere–Terpstra trend
test (reusing the earlier S1 methodology).

### Confound control (added here, see Results §1)

$v_{close}$ correlates with several **single-frame-visible** quantities: $d_{long}$ ($r$ = +0.25),
$\log$ imaged area (−0.18), $|lat|$ (−0.25), eccentricity (−0.15) and ego speed (+0.25). Any
univariate relationship on $v_{close}$ must therefore first be cleared of "these visible quantities
leaking through correlation". We report two multivariate specifications:

* **basic**: $b \sim v_{close} + d_{long} + v_{ego}$;
* **full**: plus $\log$ imaged area, $|lat|$, eccentricity, and a two-wheeler class indicator.

---

## Results

**Table 1. Univariate dose–response. Primary readout = the scene-level bootstrap CI of the slope. $n$ = events / scenes.**

| Policy | mean $b(A)$ [m/s] | Dose | Slope | 95% CI (scene boot) | Spearman $\rho$ | isotonic $R^2$ | JT $z$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo (288/149) | +0.307 | $v_{close}$ | −0.0802 | [−0.1432, −0.0148] | −0.186 (p=.002) | 0.002 | −3.17 | **opposite to expectation** |
| | | $d_{long}$ | −0.0140 | [−0.0292, +0.0023] | −0.117 (p=.048) | 0.037 | −2.17 | indeterminate |
| | | TTC | +0.0995 | [−0.0335, +0.2353] | +0.045 (p=.445) | 0.004 | +0.78 | indeterminate |
| **LTF** (291/151) | +0.020 | $v_{close}$ | −0.0061 | [−0.0100, −0.0028] | −0.224 (p<.001) | 0.000 | −3.56 | **opposite to expectation** |
| | | **$d_{long}$** | **−0.00169** | **[−0.00259, −0.00072]** | **−0.279 (p<.001)** | **0.091** | **−4.62** | **grading shown** |
| | | TTC | −0.0013 | [−0.0116, +0.0100] | −0.084 (p=.151) | 0.024 | −1.50 | indeterminate |
| DiffusionDriveV2 (291/151) | +0.201 | $v_{close}$ | +0.0358 | [−0.0143, +0.0894] | +0.031 (p=.603) | 0.039 | +0.51 | indeterminate |
| | | $d_{long}$ | +0.0050 | [−0.0068, +0.0164] | +0.018 (p=.762) | 0.003 | +0.04 | indeterminate |
| | | TTC | −0.0518 | [−0.1520, +0.0460] | −0.039 (p=.503) | 0.035 | −0.63 | indeterminate |
| DiffusionDrive (flat baseline) | +0.010 | $v_{close}$ | −0.0091 | [−0.0229, +0.0040] | −0.082 (p=.166) | 0.000 | −1.01 | indeterminate |
| | | $d_{long}$ | −0.0016 | [−0.0062, +0.0025] | −0.104 (p=.077) | 0.032 | −1.53 | indeterminate |

### 1. The "reverse grading" on $v_{close}$ disappears under full control — for every candidate

Both SimLingo and LTF give a **negative slope with a CI excluding 0** on $v_{close}$ — read literally,
"**the faster the approach, the lighter the braking**", the same signature as the earlier S1 curve's
"the more urgent, the less it brakes". But that reading **cannot be accepted at face value**, because
$v_{close}$ correlates with several single-frame-visible quantities.

**Table 2. $v_{close}$ partial coefficients from multivariate regression (scene-level bootstrap CIs). `*` = CI excludes 0.**

| Policy | Basic spec, $v_{close}$ | 95% CI | Full spec, $v_{close}$ | 95% CI | Conclusion under full control |
| --- | --- | --- | --- | --- | --- |
| SimLingo | **−0.0837*** | [−0.1661, −0.0027] | −0.0781 | [−0.1752, +0.0132] | **CI includes 0** |
| LTF | −0.0038 | [−0.0097, +0.0015] | −0.0047 | [−0.0119, +0.0032] | CI includes 0 |
| DiffusionDriveV2 | +0.0377 | [−0.0297, +0.1127] | +0.0466 | [−0.0411, +0.1341] | CI includes 0 |
| DiffusionDrive | −0.0104 | [−0.0222, +0.0007] | −0.0110 | [−0.0256, +0.0032] | CI includes 0 |

* **LTF**: controlling only distance and ego speed already drives $v_{close}$ to zero — the univariate
  "reverse grading" is entirely $d_{long}$ leaking through correlation.
* **SimLingo**: $v_{close}$ **survives** the basic specification (−0.0837, CI excluding 0), but the CI
  includes 0 once imaged area / $|lat|$ / eccentricity / class are added. **That is: the "faster means
  less braking" relationship is ultimately explicable by single-frame-visible imaging geometry**, with
  no need — and no warrant — to assume the model perceives speed.
* **Not one** of the four candidates' $v_{close}$ partial coefficients survives full control.

**This agrees with the structural expectation**: single-frame models are structurally blind to
relative velocity and should not have an independent response on the velocity dimension. Had any
survived, *that* would have been the anomaly requiring investigation.

### 2. On the **observable** dose, only LTF shows grading

$d_{long}$ is the one dose coordinate observable to a single-frame model, and the result is clean:

| Policy | F① specificity (established) | Grading on $d_{long}$ |
| --- | --- | --- |
| **LTF** | **specific** (b-AUC 0.583 [0.519, 0.639]; the only single-frame candidate whose CI lies entirely above 0.5) | **grading shown** (slope CI excludes 0, correct sign, isotonic $R^2$ = 0.091, JT $z$ = −4.62) |
| SimLingo | non-specific (b-AUC 0.534 [0.469, 0.592]) | indeterminate (CI barely includes 0, $R^2$ = 0.037) |
| DiffusionDriveV2 | non-specific (b-AUC 0.556 [0.501, 0.613], large b(A)) | indeterminate ($\rho$ = +0.018, $R^2$ = 0.003, JT $z$ = +0.04 — entirely flat) |
| DiffusionDrive | b(A) itself not significant | indeterminate |

### 3. Answer to the work order's core question: **specificity and graded response co-occur**

The work order asks whether "specificity" and "graded response to severity" are two faces of one
mechanism deficit.

**On this candidate pool, they co-occur.** The one candidate that reacts specifically on F① (LTF) is
also the only one that shows grading on the observable dose; the two candidates that "react but
non-specifically" (SimLingo, DiffusionDriveV2) are both indeterminate on the observable dose, and
DiffusionDriveV2 is **entirely flat** ($\rho$ = +0.018, JT $z$ = +0.04).

**But the strength must be bounded**: this is co-occurrence at $n$ = 3 (plus a flat baseline), not an
established regularity. Co-occurrence is also not identity of mechanism — this design cannot separate
"one mechanism produces both specificity and grading" from "two mechanisms happen to align on these
three candidates".

One noteworthy **counter-direction**: DiffusionDriveV2's $b(A)$ = +0.201 is the second largest in the
table (behind SimLingo's +0.307) — it **reacts strongly** yet is **entirely flat** in dose.
**Response magnitude and grading ability are two different things**, and this cell is the most direct
evidence for that.

---

## Discussion

**Methodologically, the most valuable product of this task is not the slopes but the adjudication
that "$v_{close}$ cannot stand alone as the primary readout".** The work order specifies $v_{close}$
because it corresponds directly to how the question "something darts out at speed" is posed, and that
motivation is right; but for single-frame candidates it is simultaneously the "structurally
invisible quantity" on which the D2cV falsification floor depends. Staying consistent between those
two facts requires promoting the $d_{long}$ control from optional to required. **The same discipline
imposing the same constraint in two different tasks is what distinguishes a discipline from an ad hoc
decision.**

**Relation to the earlier S1 conclusion**: the earlier SimLingo result was "direction correct but
isotonic $R^2$ close to zero". This round reproduces the near-zero $R^2$ in the new framework (0.037
on $d_{long}$, 0.002 on $v_{close}$) and supplies half of the reason: SimLingo's grading on the
**observable** dose is itself indeterminate. The earlier "the more urgent, the less it brakes"
observation is **partly retained and partly explained away** — it still holds and is significant
univariately, but no longer holds independently once single-frame-visible geometry is controlled.
This is an **interpretive revision**, not a numerical one (the univariate numbers are unchanged digit
for digit).

**Limitations**:
1. All three main-comparison candidates are single-frame models, so this task actually only tested
   "grading vs an observable dose"; "grading vs the velocity dimension" is **structurally
   unmeasurable** on this pool. The multi-frame candidates (Alpamayo-R1, AutoVLA) could have done it,
   but their $b(A)$ is indistinguishable from zero (§4.2.7), leaving no dose analysis to speak of.
2. Every isotonic $R^2$ is ≤ 0.091: even for LTF, which "shows grading", the dose explains under 10%
   of the variance. **"Grading shown" is a statement about the sign and significance of a slope, not
   about predictive power.**
3. $b$ is one number per event (averaged over the clean/ghost frame pairs), with no within-event
   repeats, so residual variance cannot be decomposed into "model stochasticity" and "true
   between-event differences".

---

## Self-correction record

1. **The tension between the work order's primary readout and existing discipline was not handled
   silently** (§SG/A41): the work order specifies $v_{close}$ as primary, but this round's candidates
   are all single-frame and structurally blind to speed. We **executed the work order's primary
   readout** while promoting the $d_{long}$ control to a required item read alongside it, rather than
   replacing the primary — replacing it would have made this report unalignable with the work order.
2. **The univariate result briefly supported the more eye-catching conclusion "SimLingo brakes less
   the more urgent it gets"**; the CI includes 0 once the full specification is added. **We report the
   fully-controlled conclusion** and leave the univariate numbers in Table 1 unchanged for audit.
3. **$v_{close}$ has two upper-tail artifacts above 50 m/s** (nuScenes 2 Hz annotation differencing
   noise). We pre-registered winsorization of the **dose coordinate** at the 99th percentile rather
   than deleting samples; the number of events affected is serialized with the results
   (`n_winsorized`).
