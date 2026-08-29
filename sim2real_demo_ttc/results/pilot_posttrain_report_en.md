# Pilot Post-Train: A Coupling Auxiliary Loss Designed from the F-Axis Diagnosis, and a Test of Whether the Coupling Actually Changed

> Pre-registration and every judgement call are recorded in [`amendments.md`](amendments.md) §HL/A26.
> Numerical artifacts: `p2_pilot_posttrain.json`, `p2_head_A1_task_only.pt`,
> `p2_head_A2_task_plus_coupling.pt`; feature cache `variants/n1_d2/pilot_query_states.npz`.

---

## Motivation

This line of work has already established an F-axis diagnosis: **SimLingo contains both a direction
from which real braking severity can be read linearly and a direction that genuinely drives braking,
yet the two are close to orthogonal** (colinearity bounded by $|\cos| \le 0.064$, upper end of the
95% CI). In one sentence: **the information is inside, but it is not what drives the action.**

The value of a diagnosis lies not in its precision but in whether it points at something that can
actually be changed. This experiment designs a **fixed-budget pilot post-train** from that diagnosis
and measures two things before and after: ① the behavioural score and ② **the coupling quantity
itself**.

② is mandatory because measuring only ① walks into a specific trap: any post-training improves the
behavioural score somewhat, and declaring "the diagnosis is validated" on that basis mistakes
"fine-tuning works" for "the diagnosis is right". This experiment pre-registers an attribution
control to close that gap, and **in hindsight that control turned out to be decisive.**

---

## Method

### Budget and trainable parameters

**Only `speed_wps_head` is trained** (`Linear(896→256) → SiLU → Linear(256→2, bias=False)`, 229k
parameters); everything else is frozen (vision encoder, the 24-layer LLM trunk, the final RMSNorm,
the route head).

This cut is not chosen to save compute but because **it is a direct test of the core hypothesis of
selection-protocol §1** — "for a candidate that already encodes hazard internally, post-training
only has to connect the readout to the action ⇒ cheap". With everything before the head frozen, each
sample's input feature $h_{23}[\text{query}]$ is a constant, so the head can be **trained offline
after a single forward pass over the corpus**; the entire pilot costs 2296 forward passes of GPU.

### Data and split

Class A (VRU emergence) and class D2a (geometry-balanced static-object negatives) of the nuScenes
n1_d2 corpus; 2 conditions × 2 frames per event = **2296 samples / 574 events / 234 scenes**.
**Scene-level halving**: train 117 scenes (1040 samples) / test 117 scenes (1256 samples). The task
target $v_{human}$ is the human driver's actual mean speed over $[t+0.5,\ t+1.0]$ s after the frame,
obtained by twice differencing the nuScenes `ego_pose` sequence (mean 5.32 m/s).

### Loss

$$\mathcal{L} = \underbrace{\mathrm{MSE}(v_{cmd},\, v_{human})}_{\text{task}}
+ \beta\underbrace{\mathrm{MSE}(wp,\, wp_0)}_{\text{distillation regularizer}}
+ \lambda\underbrace{\mathrm{ReLU}\big(\Delta_{1\sigma} + m\big)}_{\textbf{coupling}}$$

with $\Delta_{1\sigma} = v_{cmd}(f + 1\sigma\,\hat v_{hazard}) - v_{cmd}(f)$, margin $m$ = 0.05 m/s,
$\beta$ = 1.0 (identical across arms), $\lambda$ = 1.0 (arm A2 only), Adam lr = 1e-4, 200 full-batch
epochs.

**Key design choice**: $\Delta_{1\sigma}$ **is exactly the α = +1 dose response measured by the
steering experiment**, not a newly invented proxy. The quantity being repaired and the quantity
being verified are the same, without which "the coupling changed" could not be aligned with the
existing steering readouts. The distillation regularizer forbids overturning the planner wholesale
($wp_0$ = the original model's waypoints).

### Coupling direction

$\hat v_{hazard}$@L23 is fitted by ridge regression from the query-segment mean of
$\mathrm{RMSNorm}(h_{23})$ onto $-a_{brake}$ (the human's real longitudinal deceleration, residualized
on $v_{at\,emergence}$), **using the training split only and the ghost condition only** (the hazard
appears only in ghost frames, matching T-F's pre-registered condition). The ridge $\alpha$ is chosen
by scene-level GroupKFold within the training split (α = 100, train CV $r$ = +0.008); the held-out
split is never consulted.

### Arms and pre-registered decision rule

| Arm | Training objective |
| --- | --- |
| **A0** baseline | not trained (original head) |
| **A1** task-only | task + distillation ($\lambda$ = 0) — **attribution control** |
| **A2** task + coupling | task + distillation + coupling |

**Two pre-registered primary readouts, both required**: ① the **behavioural score** b-AUC(A vs D2a)
on held-out scenes, $b = v_{cmd}(\text{clean}) - v_{cmd}(\text{ghost})$; ② **the coupling quantity
itself**, $\Delta_{1\sigma}$ and $\cos(g,\hat v_{hazard})$ (with $g = \partial v_{cmd}/\partial f$
computed exactly by autograd). **Decision rule**: the diagnosis counts as validated only if ② has
actually changed **and** ①'s improvement in A2 is significantly larger than in A1. Two further
behavioural readouts (MAE and $\rho(b, a_{brake})$) are reported alongside so that the conclusion
does not depend on a single metric.

---

## Results

**Table 1. Pilot post-train under a fixed budget (speed_wps_head only, 229k trainable parameters): behavioural scores and the coupling quantity itself, all on held-out scenes.**

| Arm | Training objective | b-AUC(A vs D2a) | 95% CI | MAE($v_{cmd}$, human) [m/s] | $\rho(b,\, a_{brake})$ | $p$ | $\Delta_{1\sigma}$ [m/s] | $\cos(g,\hat v_{hazard})$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | not trained (baseline) | 0.544 | [0.461, 0.621] | 2.006 | +0.168 | 0.040 | **+0.091** | **+0.046** |
| A1 | task + distillation ($\lambda$ = 0) | 0.592 | [0.496, 0.680] | 0.671 | +0.160 | 0.049 | −0.076 | −0.016 |
| A2 | task + distillation + **coupling** | 0.593 | [0.499, 0.679] | 0.673 | +0.162 | 0.047 | **−0.301** | **−0.068** |

Held-out class-A events n = 145, class-D2a n = 141. $\rho$ is Spearman correlation against the
human's real $a_{brake}$.

**Table 2. Pre-registered contrasts. The A2 − A1 row is the attribution control that decides whether any behavioural gain can be credited to the coupling loss.**

| Contrast | Δ b-AUC | 95% CI (scene bootstrap) | $\Delta\cos$ | $\Delta\Delta_{1\sigma}$ [m/s] |
| --- | --- | --- | --- | --- |
| A2 − A0 (the apparent "total gain") | **+0.051** | [−0.055, +0.148] | **−0.113** | **−0.392** |
| **A2 − A1 (attribution control)** | **+0.001** | **[−0.004, +0.007]** | −0.052 | −0.225 |
| A1 − A0 (task fine-tuning alone) | **+0.050** | — | −0.061 | −0.167 |

**How to read this.** The coupling was **decisively changed**: $\Delta_{1\sigma}$ flipped from
**+0.091** (injecting the hazard direction made the model *accelerate* — the opposite of the causal
expectation) to **−0.301** (injecting it now decelerates, the correct sign), a factor of 3.3;
$\cos(g,\hat v_{hazard})$ flipped from +0.046 to −0.068. Yet the behavioural improvement of **+0.051
is almost entirely A1's** (+0.050): A2's increment over A1 is only **+0.001, 95% CI
[−0.004, +0.007]**, indistinguishable from zero. The other two behavioural readouts agree: MAE 0.671
vs 0.673 and $\rho(b, a_{brake})$ +0.160 vs +0.162.

> **Instrument side.** The coupling term did work: all three coupling readouts
> ($\Delta_{1\sigma}$, $\cos$, and their A2 − A1 differences) move consistently and substantially,
> and $\Delta_{1\sigma}$ is the same quantity as the steering experiment's α = +1 dose response, so
> "the coupling changed" rests on no newly invented proxy. The task term also worked:
> MAE($v_{cmd}$, human speed) fell from 2.006 to 0.671, a factor of three.
> **However the coupling direction itself passes only a weak instrument-side check**:
> $\hat v_{hazard}$@L23 correlates with the residualized real $a_{brake}$ on held-out data
> (ghost condition, n = 628) at $\rho$ = **+0.053 (p = 0.183)** — correct sign, not significant.
> **Specimen side.** At this budget, coupling the action to that hazard readout changes the coupling
> quantity by a factor of 3.3 and buys no measurable behavioural gain. Task fine-tuning alone (A1)
> already contributes 54% of the coupling change ($\Delta\cos$ −0.061 of −0.113): fitting the action
> head to the human's real deceleration automatically produces part of the pathway connection.

---

## Discussion

**1. Three-state verdict: partially holds.** The coupling did change (criterion ① satisfied:
$|\Delta\cos|$ = 0.113 > 0.05 and $|\Delta\Delta_{1\sigma}|$ = 0.392 > 0.02), but the behavioural
increment over the task-only control is not significant (criterion ② not satisfied: the A2 − A1
95% CI includes zero). Precisely stated: **the pathway was repaired, and it bought no measurable
behavioural gain.**

**2. The attribution control is the most important structure in this experiment, and in hindsight
it was decisive.** Running only A0 and A2 would have shown a b-AUC improvement of **+0.051** and
supported a claim that "the F-axis diagnosis is validated". The A1 arm shows that **+0.050 of that
+0.051 comes from plain task fine-tuning** and has nothing to do with the coupling term. **This is
exactly the misattribution that the requirement "do not assume the diagnosis is validated merely
because the behavioural score improved" refers to** — and here it would have occurred: without the
control, a difference of 0.001 and a difference of 0.051 look identical.

**3. Coupling strength and behavioural score are dissociable.** A1 already moved the coupling by
54%, and A2 pushed it to 100% ($\Delta\cos$ −0.061 → −0.113; $\Delta_{1\sigma}$ −0.167 → −0.392),
while the behavioural score did not budge (+0.001 [−0.004, +0.007]). **At this budget and under this
behavioural metric, F-axis coupling is not the binding constraint.** Treating "representation–action
coupling" as a knob that can be optimized in isolation with behaviour expected to follow does not
hold here.

**4. What this means for the central claim — it is simultaneously supportive and limiting.**
Supportive: the F-axis diagnosis points at an internal quantity that is **interventionable,
measurable, and demonstrably changed** — not a correlational description that can only be observed.
That is among the strongest available forms of evidence for the claim that the four axes localize a
link in the causal chain. Limiting: **changing that link does not automatically improve deployment
behaviour**, so the four-axis matrix can tell you *where* the disconnect is but **cannot yet
guarantee that repairing that link improves deployment performance**. Establishing the latter
requires a larger post-training budget or a genuine closed-loop consequence metric. This report
therefore does **not** claim predictive validity of the four-axis diagnosis for post-training gains.

**5. What this experiment may not claim.** (i) The coupling direction $\hat v_{hazard}$@L23 does not
reach significance on held-out data ($\rho$ = +0.053, $p$ = 0.183), so the accurate statement is
"coupling the action to a direction that reads hazard only in a weak-signal sense", **not** "connecting
the action to the hazard readout does not help" — the latter would require the coupling target to
pass a significance test first. (ii) Training 229k head parameters for 200 epochs is a deliberately
small budget; the conclusion is "at this budget", and a larger budget (e.g. LoRA on the deep layers)
is not excluded from giving a different answer. (iii) The b-AUC baseline of 0.544 is itself close to
0.5, limiting dynamic range; but three parallel readouts agree, so the conclusion does not rest on
b-AUC alone.

---

## One-sentence update to the causal-chain picture

> The pilot post-train upgrades the F axis from "an observed orthogonality" to "an internal quantity
> that can be intervened upon and verified to have changed", while adding a hard limitation to the
> picture: **repairing link F does not, at a fixed small budget, automatically propagate into
> improved deployment behaviour** — the four axes localize where the disconnect is, not yet which
> repair is most worthwhile.

---

## Record of self-correction

1. **The fitting protocol for the coupling direction was corrected mid-run.** The first version fitted
   $\hat v_{hazard}$@L23 on pooled (clean + ghost) samples with a fixed ridge $\alpha$ = 1e3; its
   held-out $\rho$ was **−0.028, i.e. the wrong sign** — the direction being coupled to did not read
   hazard out of sample at all. Switching to **the ghost condition only** (the hazard appears only in
   ghost frames, matching T-F's pre-registered condition) with $\alpha$ chosen by scene-level
   GroupKFold inside the training split turned $\rho$ positive at +0.053 (still not significant). The
   three-arm conclusion is **identical in direction** across both versions (coupling changed,
   behaviour unchanged), so this correction does not alter the verdict, only its geometric basis.
   Both versions' numbers are on disk.
2. **The behavioural score was changed from a single metric to three parallel readouts.** The first
   version reported b-AUC alone. Because the b-AUC baseline of 0.544 has limited dynamic range,
   MAE($v_{cmd}$, human speed) and $\rho(b, a_{brake})$ were added. All three agree, so the conclusion
   does not depend on metric choice.
3. **The work-order file was missing** (see §HL/A23). Every design decision in this experiment
   (training budget, form of the coupling term, three-arm structure, definition of the behavioural
   score, decision thresholds) was made autonomously and is itemized in §HL/A26. Should the original
   work order specify a different budget or metric, this report's conclusions would need to be re-run
   accordingly.
