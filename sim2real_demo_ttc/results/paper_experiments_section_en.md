# Experiments

> This file is a complete draft of the paper's Experiments chapter. It is not a concatenation of the
> internal reports but a reorganization along the paper's narrative logic: first establish that
> public scores cannot produce a ranking, then present the four-axis matrix under a common protocol,
> then use a pilot intervention to move the diagnosis from observation to actionability, and finally
> draw the boundaries honestly. Every number is traceable to an artifact under `results/` (source
> files are given at the end of each section).

---

## 4.1 Setup

### 4.1.1 Policies under test

All experiments are run on two end-to-end driving policies that differ in both **architecture
family** and **native training domain**. This is deliberate: the paper argues about the
**portability** of a diagnostic protocol and its boundaries, and portability failures would remain
invisible if both specimens belonged to the same family.

| Policy | Trunk | Action head | Native domain | Readable positions |
| --- | --- | --- | --- | --- |
| SimLingo | InternVL2-1B (Qwen2-0.5B, 24 decoder layers, hidden 896) | continuous regression head: `Linear(896→256) → SiLU → Linear(256→2)`, predictions `cumsum`-ed | CARLA (sim) | 24 decoder layers |
| DiffusionDrive | TransFuser two-branch encoder (ResNet-34 × 2 + GPT-style fusion) | anchored diffusion head (20 k-means trajectory anchors) | NAVSIM / OpenScene (real)† | 8 encoder self-attention modules |

† The checkpoint's training domain comes from its name and the operator's account and was not
independently verified by us; every interpretation depending on "which side is in-domain" is a
conditional conclusion.

### 4.1.2 A single shared stimulus set

All representation-side and action-side readouts are produced on **one and the same stimulus set**,
shared by both models:

* **Ghost-probe event corpus**: 7003 events mined from nuScenes trainval, with three independent
  positive trigger criteria — A (VRU emergence, n = 291), B (close cut-in, n = 103) and C (generic
  TTC drop, n = 554);
* **Geometry-balanced negative system**: D2a (harmless static objects caliper-matched to the
  positives on log imaged area and eccentricity, n = 283), D2b (out-of-corridor context contrast),
  and D2c / **D2cV** (same VRU class, same geometry, **differing only in a relative velocity that a
  single-frame model is physically unable to observe**, n = 212). D2cV is this paper's
  **falsification floor**: any claimed "hazard discriminability" that cannot be separated from it is
  not evidence of a hazard concept;
* **Domain pairing**: two renderings of the same scene (CARLA engine rendering ↔ world-model
  photorealistic re-rendering) with geometry, actors and ego ground truth all locked and rendering
  style the only variable, constituting $do(\text{appearance})$. 72 scene-variants × 3 timestamps =
  216 pairs.

**Cross-model input alignment**: to feed the same nuScenes corpus into DiffusionDrive we implemented
an input adapter that makes the ego-state anchoring (both conditions share the clean-frame speed),
the region-token definition and the behavioural quantity item-by-item isomorphic across the two
sides.

### 4.1.3 Operationalizing the four axes

* **G (Grounding)**: a discriminative direction is fitted from
  $\delta = Z(\text{ghost}) - Z(\text{clean})$ on the fold's $S_{dir}$, the peak layer is chosen by
  AUC on the fold's $S_{sel}$, and results are reported by scene-level 4-fold CV. The primary
  readout is **CV-AUC(A vs D2a) minus that model's own D2cV falsification floor** (both negatives
  projected with the same direction at the same peak layer).
* **F (Faithfulness)**: (i) an **action-level counterfactual test** (architecture-neutral) exactly
  symmetric to G but with the readout switched to the planned action, primary readout
  b-AUC(A vs D2a) with $b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$; (ii) **representation
  injection** (architecture-dependent), $Z' = Z + \alpha\sigma_L\hat v$, primary readout the
  least-squares slope of the whole ±α ladder.
* **I (Invariance)**:
  $D_L = \mathbb{E}\lVert Z_L(x_{real}) - Z_L(x_{sim})\rVert / \mathbb{E}\lVert Z_L(x)\rVert$ and
  $I_m = 1 - D_{L^*}$, reported alongside the **behavioural** domain sensitivity
  $\mathbb{E}|\Delta v_{cmd}| / \overline{v_{cmd}}$.
* **C (Concentration)**: domain-paired activation patching in which corruption is always **paired
  real-input swapping** (noise corruption prohibited) and the metric is always **continuous**
  (binarization prohibited); $C_m$ = top-2 layer share of recovery, accompanied by a shape diagnostic
  of the recovery profile (§4.4.4).

### 4.1.4 Statistical and reporting discipline

Scene-level resampling throughout (multiple frames of one scene are not independent); one
pre-registered primary readout per experiment with everything else marked as sensitivity analysis;
three-state adjudication (PASS / FAIL / **indeterminate**), with insufficient power always recorded
as indeterminate rather than forced into a binary; random-direction controls carry **their own
per-layer null distribution**; all cross-model comparisons use **within-model normalized** quantities
only. An amendment ledger is maintained throughout; the experiments reported here registered **26**
amendments, five of which converted an already-obtained positive result back into a negative or
indeterminate one (§4.4).

> Sources: `n1_report.md`, `axis_naming_alignment.md`, `amendments.md`,
> `results/diffusiondrive_g1_adapter/`.

---

## 4.2 Main Results

### 4.2.1 Public scores cannot define a ranking

| Policy | Benchmark | Metric | Score | Range | Protocol |
| --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA Leaderboard 2.0 | Driving Score | 6.87 | 0–100 | closed-loop simulation |
| DiffusionDrive | NAVSIM navtest | PDMS | 88.1 | 0–100 | non-reactive pseudo-closed-loop (log replay) |

There is **no meaningful ordering** between these two numbers: different benchmarks, different
domains, different protocols and different metric definitions (Driving Score is route completion
multiplied by an infraction-penalty product; PDMS is a weighted combination of sub-scores). Placing
them side by side yields exactly one piece of information: **within the public score system,
"ranking" is undefined.**

This is not a limitation but the paper's point of departure. If two candidates cannot be ranked at
all, the first half of the phrase "public leaderboard ranking is disconnected from deployment
ranking" is empty, and model selection must rest on other evidence. The four-axis matrix of the next
section is measured for both models on **one stimulus set under one statistical protocol** and
therefore aligns cell by cell — the first and most basic increment this framework offers over public
scores.

### 4.2.2 The four-axis matrix under a common protocol

**Table 1. G/F/I/C readouts for two policies, measured on one and the same stimulus set under one and the same statistical protocol. The "comparable" column is part of the result, not a caveat.**

| Axis | Readout | SimLingo | DiffusionDrive | Comparable | Verdict |
| --- | --- | --- | --- | --- | --- |
| G | primary − own D2cV falsification floor | +0.035 [−0.026, +0.097] | +0.009 [−0.047, +0.065] | yes | both indeterminate |
| F① | action-level counterfactual b-AUC(A vs D2a) | 0.534 [0.469, 0.592] | 0.553 [0.486, 0.623] | yes | both indeterminate |
| F① | action change induced by hazard frames, b(A) [m/s] | +0.307 [+0.112, +0.500] | +0.010 [−0.030, +0.045] | within-model | reacts non-specifically vs does not react |
| F② | injection ±α full-ladder slope [m/s per σ] | −0.0405 | −0.00000 | **no** | instrument without resolving power on DiffusionDrive |
| I | $I_m = 1 - D_{L^*}$ (representation side) | 0.606 | 0.203 | yes | SimLingo > DD |
| I | behavioural domain sensitivity | 0.346 | 0.115 | yes | **DD > SimLingo (reversed)** |
| C | recovery-profile Spearman(layer, recovery) | −0.997 (cascade) | +0.881 (interior peak) | yes | failures enter at different places |
| C | responsible-layer mode / normalized entropy | L0 / 0.279 | L6 / 0.685 | yes | vision interface vs deep fusion |

The matrix yields three findings, developed in §4.2.3–§4.2.5.

### 4.2.3 The C axis: one diagnostic, two different repair prescriptions

Both models undergo layer-wise activation patching on **the same domain pairing**. SimLingo's
recovery profile decreases **monotonically** from 1.004 at L0 to 0.000 at L23 (Spearman $-0.997$,
$p$ = 1.3 × 10⁻²⁶): replacing layer 0's vision tokens with the sim-side ones returns behaviour
**completely** to the sim side, and the later the patched layer the less is recovered. This is a
cascade signature — **the rendering-domain failure has already entered at the vision-encoder-to-LLM
interface**, and the LLM trunk contributes no additional domain sensitivity. DiffusionDrive is the
opposite: recovery at L0/L1 is essentially zero (0.001 / 0.000) and there is an **interior
responsible layer** L6 (mean recovery 0.380; the argmax of 5 of 12 degraded scenes), with Spearman
$+0.881$.

Patch-ALL recovery is +1.004 and +1.000 respectively, establishing that each model's readable layers
form a **sufficient cut set**, so neither profile is an artefact of a particular layer.

**This is the only cell in this round that yields a substantive cross-model distinction, and what it
yields is exactly what public scores cannot:** 6.87 and 88.1 both report how much performance was
lost, never where the loss enters or what fixing it costs. Under this diagnostic, SimLingo's repair
site is the **visual front end** and DiffusionDrive's is the **deep fusion stage** — two entirely
different prescriptions.

### 4.2.4 The F axis: the information is in the representation but does not drive the action

In SimLingo, the driving-query representation **linearly carries** time-to-collision information:
ridge regression with TTC as target attains held-out $r$ = **+0.320** under scene-level GroupKFold
(permutation $p$ < 1.0 × 10⁻³). Yet the cosine between that direction and the model's own
brake-driving axis $v_{brake}$ is only $-0.005$, dead centre of the random null; and an
observational direction fitted to the **human driver's real longitudinal deceleration** $a_{brake}$
(recomputed from nuScenes `ego_pose`) has cosine $+0.014$ with $v_{brake}$, 95% CI
[$-0.049$, $+0.064$], bounding their colinearity at $|\cos| \le 0.064$ (at most 0.4% shared
variance). Meanwhile $v_{brake}$ **is itself causally effective**: injecting it yields a dose slope
of $-0.0405$ [$-0.0451$, $-0.0366$] m/s per σ.

**That is: the axis that moves the action and the axis that encodes the hazard are geometrically
unrelated.** This is the representation-level signature of causal confusion, and it is the link an
open-loop score structurally cannot see: an open-loop score checks whether the action is close to a
reference trajectory, never **what drives** the action.

(Restriction: a single-frame model is structurally blind to object motion and TTC contains a
velocity term, so a substantial part of $r$ = +0.320 should be attributed to the correlation between
TTC and **distance**. The argument is unaffected: whether the direction carries TTC or distance, it
is orthogonal to the brake axis either way.)

### 4.2.5 The I axis: representational and behavioural stability order the models oppositely

On the same rendering-domain pairing, the representational $D_{L^*}$ favours SimLingo
(0.394 vs 0.797; $I_m$ 0.606 vs 0.203) while the behavioural domain sensitivity favours
DiffusionDrive (0.115 vs 0.346), and **both pairs of scene-level bootstrap CIs are
non-overlapping**. Any procedure that compresses "domain robustness" into one number must pick one
of these orderings and discard the other. For both models $v_{domain}$ is close to orthogonal to
their own $v_{hazard}$ (all $|\theta|$ below the $1/\sqrt{d}$ reference for their dimension), so the
observed domain sensitivity does **not** operate by contaminating the hazard direction.

### 4.2.6 Intervening on the diagnosed link

We designed a fixed-budget post-training run from the diagnosis of §4.2.4: **only `speed_wps_head`
is trained** (229k parameters; everything else frozen), with loss
$\mathcal{L} = \mathrm{MSE}(v_{cmd}, v_{human}) + \beta\,\mathrm{MSE}(wp, wp_0) + \lambda\,\mathrm{ReLU}(\Delta_{1\sigma} + m)$,
where $\Delta_{1\sigma} = v_{cmd}(f + 1\sigma\hat v_{hazard}) - v_{cmd}(f)$ **is exactly the α = +1
dose response measured by the injection experiment** (the quantity being repaired and the quantity
being verified are the same). Training only the head is itself a test: if the hypothesis "hazard is
already encoded internally and only the readout-to-action connection is missing" holds, the repair
should be cheap.

**Table 2. Pilot post-train on SimLingo (229k trainable parameters), held-out scenes. A1 is the attribution control.**

| Arm | Objective | b-AUC(A vs D2a) | 95% CI | $\Delta_{1\sigma}$ [m/s] | $\cos(g,\hat v_{hazard})$ |
| --- | --- | --- | --- | --- | --- |
| A0 | not trained | 0.544 | [0.461, 0.621] | +0.091 | +0.046 |
| A1 | task + distillation ($\lambda = 0$) | 0.592 | [0.496, 0.680] | −0.076 | −0.016 |
| A2 | task + distillation + coupling | 0.593 | [0.499, 0.679] | **−0.301** | **−0.068** |

**The coupling is decisively changed**: $\Delta_{1\sigma}$ flips from $+0.091$ (injecting the hazard
direction made the model *accelerate*, the opposite of the causal expectation) to $-0.301$, a factor
of 3.3; $\cos(g,\hat v_{hazard})$ flips from $+0.046$ to $-0.068$. **Yet the behavioural score shows
no increment**: A2 − A0 = $+0.051$ [$-0.055$, $+0.148$] looks like a success, but the attribution
control gives A1 − A0 = $+0.050$ and A2 − A1 = $+0.001$ [$-0.004$, $+0.007$]. Two further parallel
behavioural readouts agree (MAE 0.671 vs 0.673; $\rho(b, a_{brake})$ +0.160 vs +0.162).

The conclusion has two halves that must be stated together: **the diagnosed link can genuinely be
repaired** (among the strongest available evidence that the four axes localize a link of the causal
chain), **and at this budget repairing it does not automatically improve deployment behaviour.**

> Sources: `h1_three_line_evidence.json`, `c_axis_simlingo.json`, `c_axis_concentration.json`,
> `f_axis_ttc_gradient.json`, `f_axis_action_counterfactual.json`, `i_axis_domain.json`,
> `p2_pilot_posttrain.json`.

---

## 4.3 Why no composite score is reported

The four axes differ in units, adjudication state and comparability: G and I are within-model
normalized readouts that align across models; F's injection protocol is **dimensionally different**
between the two models; C's top-2-share formula has its **premise violated** on SimLingo. Weighting
these four cells into a scalar amounts to treating "non-comparable" and "indeterminate" as either
zero or as the median, and either treatment makes the provenance of the final ranking untraceable:
as soon as someone asks "why is A ahead of B?", the answer lands on the weights rather than on the
evidence. We therefore deliver a matrix **with blanks and explicit "not comparable" labels** rather
than a scalar that looks clean but cannot be audited.

---

## 4.4 Ablation-like Analyses: why these numbers can be believed

Every item in this section is a **negative check**: its purpose is not to make numbers look better
but to exclude the case in which numbers look good while meaning nothing. This work registered 26
amendments during execution, five of which converted an already-obtained positive result back into a
negative or indeterminate one; the five most consequential are given below.

### 4.4.1 The falsification floor: the primary readout must separate from "any VRU is present"

If the G-axis primary readout were compared only against random or permutation nulls, "one more
person in the frame" would be mistaken for "hazard was read out". We therefore require separation
from **D2cV** (same VRU class, same geometry, differing only in a relative velocity unobservable to
a single-frame model). The two models' differences are $+0.035$ [$-0.026$, $+0.097$] and $+0.009$
[$-0.047$, $+0.065$], and across 10 CV fold-assignment seeds $+0.002 \pm 0.024$ and
$+0.012 \pm 0.032$ — **an sd of the same order as, or larger than, the effect**. The G axis is
therefore indeterminate at this sample size, with the bottleneck localized to the D2cV sample size
(212), not to pooling or model choice.

**A correction that must be recorded**: the first version refitted a separate direction for each
negative class, inflating DiffusionDrive's difference to $+0.093$ [$0.004$, $0.174$] with a CI
excluding zero. Switching to **a shared primary-readout direction** reduced it to $+0.047$, and
equalizing the stimulus sets on both sides (DiffusionDrive originally had only 141 D2cV negatives
against SimLingo's 212) reduced it further to $+0.009$. **A positive result was thereby rolled back
to indeterminate, twice.**

### 4.4.2 In-house upper-bound calibration: attributing a null result to instrument or specimen

For each model we additionally construct a behaviour-defined axis that is **effective by
construction** (labelled by the model's **own** braking residual). If even that axis cannot move
behaviour, the null result belongs to the instrument. SimLingo's $v_{brake}$ has an injection slope
of $-0.0405$, so the effect is real. DiffusionDrive's $v_{brake}^{dd}$ (peak per-layer held-out AUC
0.656) has a slope of only $+0.00002$ [$-0.00013$, $+0.00021$], and escalating the dose to
$\pm 32\sigma$ still yields a longitudinal change of only $-0.0020$ m/s. **The injection does reach
the model** — the lateral offset varies monotonically and symmetrically with α, reaching 0.02–0.09 m
at ±32σ. DiffusionDrive's F② null is therefore an **instrument-side** conclusion: its longitudinal
plan is dominated by the anchored diffusion head, and representation perturbations within the tested
dose do not change the anchor selection. **Corollary**: operationalized as an injection effect size,
the F axis is **not comparable across action-head families**; cross-model alignment must use the
architecture-neutral action-level counterfactual test.

### 4.4.3 Independent corroboration by the analytic Jacobian

SimLingo's action head satisfies the conditions for analytic projection, and
$v_{cmd} = 2\lVert \mathrm{head}(f_1) + \mathrm{head}(f_2)\rVert$ depends only on positions 1 and 2
of the speed_wps segment, so $g = \partial v_{cmd}/\partial h_{23}$ is obtained **exactly** by
autograd. For deep injections the analytic prediction matches the empirical measurement closely:
$v_{brake}$@L22 analytic $-0.0432 \pm 0.0084$ vs measured $-0.0405$ (ratio **1.07**);
$v_{hazard}$@L23 ratio **0.98** with event-level Pearson $r$ = **+0.993** [+0.975, +0.999]. Shallow
and mid-layer injections are systematically underestimated by factors of 6–144. **The validity
boundary is thereby quantified as the nonlinear depth between injection site and action head**:
candidates at $L \ge 22$ can be pre-screened analytically (cost ≈ one forward plus one backward
pass), while $L < 22$ requires the empirical sweep.

### 4.4.4 Shape diagnostics of the recovery profile: a formula's premise must also be tested

The formula $C_m$ = top-2 layer share **presupposes an interior peak in the profile**. SimLingo's
profile is a monotonically decreasing cascade (Spearman $-0.997$), in which the top-2 share measures
nothing but a multiple of $1/L$. Read literally, SimLingo's $C_m$ = 0.176 above a diffuse baseline
of 0.083 would be interpreted as "PASS: failure is concentrated" — **a misreading**. We therefore add
an applicability criterion ($\rho < -0.7$ ⇒ cascade ⇒ formula inapplicable) and use three
depth-independent shape statistics for cross-model comparison: Spearman(layer, recovery), the
fraction of layers needed to reach 80% of recovery mass, and the normalized entropy of the
responsible-layer argmax. $C_m$'s diffuse baseline varies with depth (0.250 at 8 layers vs 0.083 at
24), so **its value is not comparable across models**.

### 4.4.5 The attribution control: an improved behavioural score is not a validated diagnosis

Had the pilot run only the baseline and coupling arms, it would have shown a b-AUC improvement of
$+0.051$ and declared the diagnosis validated. The task-only control (A1) shows that $+0.050$ of
that $+0.051$ has nothing to do with the coupling term. **Without this control, a difference of
0.001 and a difference of 0.051 are indistinguishable in the results.** An incidental finding: task
fine-tuning alone already contributes 54% of the coupling change ($\Delta\cos$ $-0.061$ of
$-0.113$), i.e. fitting the action head to the human's real deceleration automatically produces part
of the pathway connection; the coupling term doubles it without buying any additional behavioural
gain — **coupling strength and behavioural score are dissociable**.

### 4.4.6 Multiple-comparison nulls must be enumerated, not extrapolated

In an "argmax over $N$ candidates" setting (we projected all 24 × 4864 = 116,736 FFN value vectors
into vocabulary space), thresholding by the observed variance plus a Gaussian tail is unsafe. The
empirical 99.9th percentile of that null is 0.100–0.154 whereas the Gaussian-tail threshold is only
0.071–0.084; on the $v_{domain}$ and $v_{danger}^{lang}$ axes the null's $\alpha$ quantile reaches
**0.641** and **0.688**, nearly an order of magnitude above the Gaussian extrapolation. Switching to
**full-population enumeration** (116,546 vectors, so that $\alpha$ = 0.05/950 = 5.26 × 10⁻⁵ can be
read off as an empirical quantile) reduced the number of "doubly corroborated" candidates from
**11 to 0**.

> Sources: `amendments.md` (all 26 amendments), `analytic_vs_empirical.md`,
> `c_axis_shape_diagnostics.json`, `cosine_matrix.json`, `generalizable_tips.md`.

---

## 4.5 Discussion of Limitations

**1. The third line of the three-way comparison is not genuinely closed.** Our pilot post-train is a
single-model, single-metric validation under a fixed small budget (229k parameters), not a full
measurement of the target variable "held-out-domain safety-score increment under a fixed
post-training budget". The paper therefore establishes that the four axes are measurable,
non-redundant, and that the link they localize can be intervened upon — **not** that four-axis scores
predict post-training gains. The latter requires a larger budget or a closed-loop consequence metric.

**2. The G axis can currently report only "readable / not readable", not "grounded / not
grounded".** Both models' primary-minus-floor readouts are indeterminate, bottlenecked by the D2cV
sample size (212). The limitation persists when the trigger criterion is changed (three independent
criteria A / B / C give differences of $+0.003$ / $-0.022$ / $+0.023$, all with CIs crossing zero),
so it is not a peculiarity of one scenario type.

**3. Parts of the F and C operationalizations are not portable.** F's injection protocol is
unmeasurable on an anchored diffusion head; C's top-2-share formula has its premise violated on a
cascade profile. Neither is a statement that the model is poor on that axis; both are statements
that the operationalization does not apply to that architecture or profile shape. We therefore
decline to fold them into a single score.

**4. The domain pairing is not a 3DGS reconstruction.** We use a same-geometry dual-rendering pair
(CARLA engine rendering ↔ world-model photorealistic re-rendering), whose causal structure matches
$do(\text{appearance})$, but the scope of the conclusion should be stated as that pairing rather than
as sim ↔ real in general.

**5. The random-direction control for injection experiments has no resolving power at this sample
size.** Across 7 per-layer null distributions, none of 11 directions exceeds its own same-layer null —
**including** $v_{brake}$, which is effective by construction and independently confirmed by the
analytic method. The null sd varies by a factor of 50 across layers (0.0012 to 0.0642), of the same
order as or larger than the effects under test. All steering causal verdicts in this paper are
therefore recorded as indeterminate with the attribution explicitly on the instrument side, and
**must not be read as "these directions have no causal effect"**.

**6. The pilot's coupling target passes only a weak instrument-side check.** $\hat v_{hazard}$@L23
correlates with the residualized real $a_{brake}$ on held-out data at $\rho$ = $+0.053$
($p$ = 0.183) — correct sign, not significant. The accurate statement is "coupling the action to a
direction that reads hazard only in a weak-signal sense", not "connecting the action to the hazard
readout does not help".

**7. DiffusionDrive's native training domain is not independently verified**, so every interpretation
depending on "which side is in-domain" is conditional.

---

## Appendix: internal reporting discipline (not part of the paper body)

> **Instrument side.** Each of the four axes was independently calibrated once in this work: G by
> cross-model positive calibration (conclusion: insufficient resolution, bottlenecked by the D2cV
> sample size); F by each model's own in-house upper-bound direction (effective on SimLingo, without
> resolving power on DiffusionDrive); I by within-model normalized ratios reported on both the
> representational and the behavioural side (the two orderings are opposite, showing neither may be
> used alone); C by patch-ALL ≈ 1.0 establishing the readable layers as a sufficient cut set,
> followed by a profile-shape diagnostic testing the formula's premise.
> **Specimen side.** SimLingo presents the combination "concept not readable, pathway drivable, the
> drivable pathway unrelated to hazard semantics, domain failure cascading from the visual
> interface"; DiffusionDrive presents "perception-side signal present but not clearing the
> falsification floor, longitudinal output unmovable by representation perturbation, large
> cross-domain representational drift with small behavioural drift, failure concentrated in the deep
> fusion stage". Neither profile can be expressed by any single score.
