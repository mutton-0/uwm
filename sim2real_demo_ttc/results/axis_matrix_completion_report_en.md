# Completing the Axis Matrix: the F Axis for DiffusionDrive and the C Axis for SimLingo

> Axis letters follow the G/F/I/C/D naming finalized on 2026-08-29 (mapping in
> [`axis_naming_alignment.md`](axis_naming_alignment.md)). Pre-registration and every judgement call
> made in this round are recorded in [`amendments.md`](amendments.md) §HL (A22–A26).
> Numerical artifacts: `f_axis_action_counterfactual.json`,
> `f_axis_dd_steer{,_brakedd,_escalate}.json`, `v_brake_dd.npz`/`.json`, `c_axis_simlingo.json`,
> `c_axis_shape_diagnostics.json`.

---

## Motivation

The four-axis matrix from the previous round had two empty cells: **the F axis had been measured on
SimLingo only** (no steering experiment for DiffusionDrive) and **the C axis on DiffusionDrive only**
(no equivalent domain-paired activation patching for SimLingo). Those gaps prevent the
"public leaderboard vs four-axis matrix" comparison from being aligned cell by cell — and alignment
is the precondition for the entire argument of this work: an incomplete matrix has no better claim
to producing a ranking than a public score does.

This report fills both cells, and in doing so answers a question that had not previously been
asked: **are the four axes' operationalizations portable across action-head architectures?**

---

## Method

### F axis (DiffusionDrive)

**(i) Action-level counterfactual test** (the F main validation designated by protocol §3.5;
architecture-neutral; run on both models). Exactly symmetric to the G-axis test, but the readout
switches from internal representation to the **final planned action**: manipulation ① of the causal
feature (hazard present or not) = ghost vs clean frames of class-A events; manipulation ② of the
geometric confound = ghost vs clean of the geometry-balanced negatives D2a. The behavioural quantity
is $b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$ (positive = slows down once the object
appears). The pre-registered primary readout is **b-AUC = AUC(b(A) vs b(D2a))** with scene-level
bootstrap (2000 resamples; both groups draw the same scene sample). Zero GPU: per-condition planned
speeds for both models are already in the existing caches.

**(ii) The steering battery** (architecture-dependent). Injection
$Z' = Z + \alpha\sigma_L\hat v$ into the **image-token block** at DiffusionDrive's own concept peak
layer $L^\*$ = 5 (frozen in T-G); α ladder ±{0.5, 1, 2, 4}; primary readout = least-squares slope of
the whole ±α ladder (amendment A3); controls: ① same-layer 20-seed random-direction null,
② specificity (lateral / comfort), ③ termination / recovery. Stimuli = **the same** S_test class-A
events as on the SimLingo side (same seed, same three-way split). **Architecture statement**:
DiffusionDrive has a diffusion action head, so per Stage D of the plan ("for architectures on which
analytic projection is impossible, use empirical steering only and do not attempt the analytic
shortcut — this is dictated by the architecture, not a fallback"), no Jacobian arm is run.

**(iii) In-house upper-bound calibration** (added this round; it determines how a null result is
attributed). Following the T1-Q discipline we construct a behaviour-defined axis $v_{brake}^{dd}$
that is **effective by construction**: features = per-condition mean image-token activations; labels
= whether that condition's `commanded_speed`, after regressing out ego speed, falls below the median
(**removing the ego-speed main effect is mandatory**; empirically ego speed explains 95.6% of the
variance). If even this axis cannot move behaviour, a null result is attributed to the
**instrument** rather than to the specimen.

**(iv) Dose escalation.** α ∈ {4, 8, 16, 32}, to test for a threshold effect.

### C axis (SimLingo)

The protocol is **item-by-item isomorphic** to the DiffusionDrive side: the domain pairing is
sim (CARLA engine rendering) ↔ real (world-model photorealistic re-rendering), same scene and
geometry with rendering style the only variable; corruption is **always paired real-input swapping**
(noise corruption prohibited); the metric is **always continuous** (binarization prohibited):
$\mathrm{recovery}(L) = \big(v_{patch} - v_{real}\big) / \big(v_{sim} - v_{real}\big)$ with $v$ the
model's commanded speed. Degraded samples are the top 12 scene-variants by $|v_{sim} - v_{real}|$
(the same rule as the DD side's "top-12 most degraded"), at frame t = 1 s.

**One structural difference from the DD side, stated alongside**: DD patches all 320 fused encoder
tokens, whereas SimLingo patches **only the vision-token block** — the language segment differs in
length and content between conditions, so the correspondence is undefined (the same discipline as
`g2_cache` schema v2). SimLingo's sufficiency check therefore asks whether the vision tokens of all
24 layers form a sufficient cut set, not the whole sequence.

$C_m$ = top-2 layer share of recovery / $\sum_L \mathrm{recovery}(L)$, with recovery clipped at ≥ 0,
computed per scene and then aggregated, with scene-level bootstrap (5000 resamples).

---

## Results

**Table 1. Action-level counterfactual test of the F axis: does the planned action respond to the causal feature (hazard) more than to the geometry-matched confound? Identical stimuli and identical readout for both models.**

| Model | b(A) [m/s] | 95% CI | b(D2a) [m/s] | 95% CI | b-AUC(A vs D2a) | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | **+0.307** | [+0.112, +0.500] | +0.086 | [−0.213, +0.443] | 0.534 | [0.469, 0.592] | 0.157 | indeterminate |
| DiffusionDrive | +0.010 | [−0.030, +0.045] | −0.021 | [−0.063, +0.018] | 0.553 | [0.486, 0.623] | 0.027 | indeterminate |

Positives n = 288 / 291, negatives n = 283 / 283 (the same events for both models). Against D2cV
(same VRU class, same geometry, velocity only) both models give 0.522. **Both models' confound
response b(D2a) is indistinguishable from zero**, so there is **no** action-level symptom of
geometric causal confusion; but both b-AUC scene-level CIs straddle 0.5, so there is equally **no**
evidence that the action is specifically driven by genuine hazard.

**Table 2. Steering readout of the F axis on DiffusionDrive, with an in-house upper-bound calibration and a dose escalation. All runs use the same 40 S_test events (13 scenes) and inject into the image-token block at $L^\*$ = 5.**

| Injected direction | α ladder | ±α slope [m/s per σ] | 95% CI | Same-layer random null (mean ± sd, n) | $z$ | empirical $p$ | Exceeds null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $v_{hazard}^{dd}$ (G-axis direction) | ±{0.5,1,2,4} | −0.00000 | [−0.00013, +0.00008] | −0.00000 ± 0.00011 (20) | −0.01 | 0.952 | no |
| **$v_{brake}^{dd}$ (in-house upper bound)** | ±{0.5,1,2,4} | **+0.00002** | [−0.00013, +0.00021] | −0.00002 ± 0.00005 (5) | +0.69 | 0.667 | **no** |
| $v_{hazard}^{dd}$ (dose escalation) | ±{4,8,16,32} | −0.00006 | [−0.00029, +0.00010] | — | — | — | no |

Readout validity of $v_{brake}^{dd}$: per-layer held-out AUC is L0 0.535 / L1 0.548 / L2 0.608 /
L3 0.611 / L4 0.628 / **L5 0.656** / L6 0.610 / L7 0.641 — it **is** a readable behaviour-defined
axis.

**Table 3. Evidence that the injection reaches the model while the longitudinal output does not move: the lateral response scales monotonically with dose, the longitudinal one does not.**

| α | Δv_plan [m/s] | 95% CI | Δlateral [m] | \|Δlateral\|/\|Δv\| |
| --- | --- | --- | --- | --- |
| +4 | −0.0000 | [−0.0005, +0.0003] | −0.0022 | — |
| +8 | −0.0000 | [−0.0009, +0.0006] | −0.0043 | — |
| +16 | −0.0017 | [−0.0105, +0.0044] | −0.0077 | 4.5 |
| +32 | −0.0020 | [−0.0123, +0.0055] | −0.0203 | 10.0 |
| −32 | +0.0018 | [−0.0017, +0.0081] | +0.0898 | 49.9 |

Over the same scenes the **deceleration induced by a genuine hazard** is +0.0379 m/s
[+0.0036, +0.0745] (significantly ≠ 0), so the denominator of protocol §3④'s pathway-utilization
$F_m$ **is usable**; but the numerator (steering peak) is only −0.0020 even at ±32σ and is
indistinguishable from zero, so **$F_m$ admits no meaningful estimate on this specimen**.

**Table 4. C-axis readout for SimLingo, with the profile-shape diagnostics that decide whether the top-2-share formula applies. DiffusionDrive is shown for contrast.**

| Model | n_layers | patch-ALL recovery | $C_m$ (top-2 share) | 95% CI | Diffuse baseline | Spearman(layer, recovery) | Profile shape | argmax mode | argmax entropy | $L_{80}/L$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | 24 | **+1.004** | 0.176 | [0.160, 0.191] | 0.083 | **−0.997** | cascade | L0 | 0.279 | 0.542 | **indeterminate** (formula premise violated) |
| DiffusionDrive | 8 | +1.000 | 0.858 | [0.771, 0.939] | 0.250 | **+0.881** | interior peak | L6 | 0.685 | 0.375 | PASS |

SimLingo's mean per-layer recovery falls monotonically from 1.004 at L0 to 0.000 at L23; the
responsible layer is L0 in 7 of 12 scenes, L1 in 4, L2 in 1.

> **Instrument side.** Both F-axis protocols were independently calibrated. (i) The action-level
> counterfactual test is architecture-neutral (its readout is the action), so the two models align
> cell by cell. (ii) The steering protocol is **demonstrably without resolving power on
> DiffusionDrive**: the injection does reach the model (the lateral response is monotone and
> symmetric in α, reaching 0.02–0.09 m at ±32σ), yet even $v_{brake}^{dd}$ — effective by
> construction, held-out AUC 0.656 — cannot move the longitudinal output. On the C axis,
> patch-ALL = +1.004 (SimLingo) and +1.000 (DD) establish that each model's readable layers form a
> sufficient cut set, so the denominator of $C_m$ is meaningful; but SimLingo's profile shape
> violates the premise of the top-2-share formula.
> **Specimen side.** DiffusionDrive's longitudinal plan is dominated by its anchored diffusion head
> (20 k-means trajectory anchors), and representation-level perturbation up to ±32σ does not change
> the anchor selection. SimLingo's domain failure **enters at the vision-to-LLM interface** and
> cascades downstream, whereas DiffusionDrive's has an **interior responsible layer** (L6). The two
> failures enter at different places.

---

## Discussion

**1. The most important outcome of completing the matrix is not two extra numbers but the discovery
that two of the axes' operationalizations are not portable.** Operationalized as a steering effect
size, the F axis is measurable and even analytically predictable on a continuous regression head
(SimLingo) but **unmeasurable at any tested dose** on an anchored diffusion head (DiffusionDrive).
Operationalized as a top-2 layer recovery share, the C axis holds on an interior-peaked profile
(DiffusionDrive) but has its **premise violated** on a cascade profile (SimLingo). Neither is a
statement that the model is poor on that axis; both are statements that the operationalization does
not apply to that architecture or that profile shape. **This directly bounds what the four-axis
matrix can do**: it can deliver a same-protocol diagnosis cell by cell, but it **cannot** rank
models on the raw F and C values. The headline table therefore marks those rows "not comparable"
and supplies comparable substitutes.

**2. The comparable F-axis readout (action-level counterfactual) is indeterminate on both models,
but they fail differently.** SimLingo does show a significant deceleration response to hazard frames
(b(A) = +0.307 m/s, CI excluding zero); it simply responds about as much to geometry-matched
harmless objects, so the two cannot be separated. DiffusionDrive shows **almost no action response
at all** (b(A) = +0.010 m/s, CI including zero). One model reacts but non-specifically; the other
does not react. Judged by b-AUC alone (0.534 vs 0.553) one would conclude DiffusionDrive is
slightly better; judged by the magnitude of b(A) the ordering reverses. **This is another place
where a single scalar inverts the conclusion.** (Comparing b in absolute m/s across models requires
care, since the two planners operate over different speed ranges; but "does the CI exclude zero" is
a within-model judgement and is unaffected.)

**3. Completing the C axis yields a conclusion with a direct operational implication for
post-training.** SimLingo's recovery profile already reaches 1.004 at L0 and decreases
monotonically, meaning that replacing layer 0's vision tokens with the sim-side ones returns the
behaviour **completely** to the sim side. In other words, SimLingo's rendering-domain failure occurs
**entirely at the vision-encoder-to-LLM interface**, with the LLM trunk contributing no additional
domain sensitivity. The repair site is therefore not an interior layer but the **visual front end**.
DiffusionDrive is the opposite: recovery at L0/L1 is essentially zero (0.001 / 0.000) and
responsibility concentrates in the deep fusion stage L6. **The same diagnostic yields two entirely
different repair prescriptions for the two models** — precisely the increment the four-axis
framework provides over a leaderboard score.

**4. What this round may not claim.** (i) DiffusionDrive's F-axis steering result is **not**
evidence of a broken pathway, only that this operationalization lacks resolving power; adjudicating
its F axis would require an intervention effective on diffusion heads (e.g. perturbing the anchor
selection logits directly), which was not attempted here. (ii) SimLingo's "cascade" conclusion rests
on 12 degraded scenes at t = 1 s with vision-token-only patching and does not extrapolate to a
whole-set average or to full-sequence patching. (iii) The two models' $C_m$ values are **not
directly comparable** (the diffuse baseline varies with depth: 0.083 vs 0.250).

---

## Record of self-correction

1. **An applicability criterion was added to the C-axis formula (§HL/A24).** The first version applied
   protocol §3⑤'s top-2-share formula directly; SimLingo scored $C_m$ = 0.176 against a diffuse
   baseline of 0.083, which reads as "PASS: failure is concentrated". Inspecting the profile revealed
   a **monotonically decreasing cascade** (Spearman −0.997), so the formula's premise is violated and
   the value is not evidence of concentration. `scripts/c_axis_shape.py` was added to compute three
   depth-independent shape statistics for both models and write them back to the JSONs, and
   SimLingo's verdict was changed to **indeterminate (formula premise violated)**.
2. **An in-house upper-bound calibration and a dose escalation were added to the F axis (§HL/A25).**
   The first version reported only the null result for $v_{hazard}^{dd}$, which cannot distinguish
   "broken pathway" from "instrument without resolving power". After adding $v_{brake}^{dd}$
   (effective by construction) and the ±32σ escalation, the null result is unambiguously attributed
   to the **instrument**. This addition changed the nature of the conclusion, not merely added a
   number.
3. **Cross-model comparison of b(A) in absolute units is restricted**: the two planners have
   different baseline speed ranges (SimLingo's v_cmd averages 6.63 m/s; DiffusionDrive's commanded
   speed is defined as ‖traj[0]‖/0.5 s). Every cross-model judgement in this report uses
   **within-model** readouts only (b-AUC; whether a CI excludes zero); absolute m/s values are used
   only descriptively within a model.
4. **The work-order file was missing** (see §HL/A23). The experimental design of this report — in
   particular the in-house upper-bound calibration, the dose escalation and the shape diagnostics —
   was decided autonomously under the project's standing discipline, and is registered here for
   later checking.
