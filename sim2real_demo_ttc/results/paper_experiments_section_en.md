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

All experiments are run on **six** end-to-end driving policies that together form a controlled
matrix over **encoder family**, **action-head family** and **native training domain**. This is
deliberate: the paper argues about the **portability** of a diagnostic protocol and its boundaries.
Portability failures remain invisible if every specimen belongs to the same family; but if one only
ever compares across families, one cannot separate "the difference comes from the encoder" from
"the difference comes from the action head". The candidate pool therefore supplies both a
**cross-family** contrast (SimLingo vs. the TransFuser family vs. two VLAs) and a
**same-encoder / different-head** contrast (LTF vs. DiffusionDrive vs. DiffusionDriveV2, which share
one `TransfuserBackbone` under a continuous regression head, an anchored diffusion head and an
improved diffusion head respectively). The latter directly refutes a corollary of the previous
version in §4.4.2.

| Policy | Trunk (encoder) | Action head | Input temporality | Native domain | Readable positions |
| --- | --- | --- | --- | --- | --- |
| SimLingo | InternVL2-1B (Qwen2-0.5B, 24 decoder layers, hidden 896) | continuous regression head: `Linear(896→256) → SiLU → Linear(256→2)`, predictions `cumsum`-ed | single-frame | CARLA (sim) | 24 decoder layers |
| DiffusionDrive | TransFuser two-branch encoder (ResNet-34 × 2 + GPT-style fusion) | anchored diffusion head (20 k-means trajectory anchors) | single-frame | NAVSIM / OpenScene (real)† | 8 encoder self-attention modules |
| LTF (Latent TransFuser) | **same as above** (`TransfuserConfig(latent=True)`; no lidar input, the BEV branch runs on a learned latent) | continuous regression head (native TransFuser trajectory regression) | single-frame | NAVSIM (real)† | 8 encoder self-attention modules |
| TransFuser | **same as above** | **same as above** | single-frame | NAVSIM (real)† | — under this repository it is the *same* candidate as LTF, see §CE/A28 |
| DiffusionDriveV2 | **same as above** (fed a real lidar BEV histogram) | improved anchored diffusion head + PDM-score trajectory selection | single-frame | NAVSIM (real)† | 8 encoder self-attention modules |
| Alpamayo-R1 (10B) | Cosmos-Reason1 (Qwen2.5-VL-7B family, 36 decoder layers, hidden 4096) | autoregressive action-token decoding | **multi-frame** (history frame sequence) | NVIDIA PhysicalAI-AV (real)† | 36 decoder layers |
| AutoVLA | Qwen2.5-VL-3B (36 decoder layers, hidden 2048), LoRA merged | autoregressive action-token decoding | **multi-frame** (3 cameras × 4 timesteps) | NAVSIM (real)† | 36 decoder layers |

† These checkpoints' training domains come from their names and the operators' accounts and were not
independently verified by us; every interpretation depending on "which side is in-domain" is a
conditional conclusion.

**TransFuser is not listed as a seventh candidate.** In this repository (SimScale),
`transfuser_agent.yaml` and `LTF/ltf_sim_navtest.ckpt` are one configuration and one set of weights;
three independent checks (config diff, state_dict key-set comparison, and per-event output
comparison on the shared stimulus set) confirm the two are indistinguishable under this paper's
stimuli and readouts. Following our standing discipline we do not manufacture a spurious independent
data point, and instead record the finding itself as a result (§CE/A28).

**Input temporality is a new column this round because it decides whether the D2cV falsification
floor holds at all.** D2cV is constructed so that "the only difference is a relative velocity a
single-frame model is *physically unable* to observe". For multi-frame candidates (Alpamayo-R1,
AutoVLA) relative velocity *is* observable, and D2cV degenerates into an ordinary hard negative
rather than a falsification floor. The G column of Table 1 must therefore be read in two groups,
whose main readouts are defined differently (§CE/A34).

### 4.1.2 A single shared stimulus set

All representation-side and action-side readouts are produced on **one and the same stimulus set**,
shared by all six candidates:

* **Ghost-probe event corpus**: 7003 events mined from nuScenes trainval, with three independent
  positive trigger criteria — A (VRU emergence, n = 291), B (close cut-in, n = 103) and C (generic
  TTC drop, n = 554);
* **Geometry-balanced negative system**: D2a (harmless static objects caliper-matched to the
  positives on log imaged area and eccentricity, n = 283), D2b (out-of-corridor context contrast),
  and D2c / **D2cV** (same VRU class, same geometry, **differing only in a relative velocity that a
  single-frame model is physically unable to observe**, n = 212). D2cV is this paper's
  **falsification floor**: any claimed "hazard discriminability" that cannot be separated from it is
  not evidence of a hazard concept. **D2cV constitutes a falsification floor only for single-frame
  candidates**; see §4.1.1 and §4.4.1;
* **Domain pairing**: two renderings of the same scene (CARLA engine rendering ↔ world-model
  photorealistic re-rendering) with geometry, actors and ego ground truth all locked and rendering
  style the only variable, constituting $do(\text{appearance})$. 72 scene-variants × 3 timestamps =
  216 pairs.

**Cross-model input alignment**: to feed the same nuScenes corpus into every candidate we
implemented one input adapter per candidate, each making the ego-state anchoring (both conditions
share the clean-frame speed), the region-token definition and the behavioural quantity
item-by-item isomorphic across candidates. The adapters were the **dominant cost** of expanding the
pool this round, and this is deliberate:
**direction discovery and measurement are both performed on the nuScenes G1 corpus with the N1
negative system, never in each candidate's own training or simulation domain.** Had we instead
discovered directions natively and transferred them to nuScenes for validation, what we measured
would be each candidate's sim2real gap rather than the paper's actual claim — that under one shared
set of real stimuli, public leaderboard ranking and four-axis diagnosis lead to different selection
decisions. The one axis that genuinely requires a real↔sim domain pairing is **I**, because
invariance is defined over two domains; this is a requirement of that axis's definition, not a
general strategy.

How the five adapters relate: LTF and DiffusionDriveV2 directly import and reuse the **entire
front end** of the DiffusionDrive adapter (image, state, token and pooling conventions field for
field), replacing only the agent construction — this is the precondition for the same-encoder
controlled comparison to be valid at all, since any front-end difference would contaminate it.
Alpamayo-R1 and AutoVLA are VLA architectures with their own front ends, but their readout position
(prefill pooling over the language tower's decoder layers) and behavioural quantity ($v_{plan}$)
remain isomorphic to the rest.

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
* **C (Concentration)**: paired activation patching in which corruption is always **paired
  real-input swapping** (noise corruption prohibited) and the metric is always **continuous**
  (binarization prohibited); $C_m$ = top-2 layer share of recovery, accompanied by a shape diagnostic
  of the recovery profile (§4.4.4). The same method is applied to **two pairing sources**, reported
  in separate columns and not comparable to each other: **C-domain** (sim↔real rendering pairs,
  asking at which layer the rendering-domain failure enters) and **C-hazard** (G1 clean↔ghost pairs,
  asking at which layer the hazard-induced behavioural change enters, §CE/A27). Every C readout must
  first pass the **patch-ALL sufficient-cut-set self-check**: swapping all patchable tokens at once
  must yield a recovery near 1; if it falls outside $[0.7, 1.3]$ the patched tokens are not a
  sufficient cut set under that pairing, the per-layer shares are uninterpretable, and the verdict is
  indeterminate.

### 4.1.4 Statistical and reporting discipline

Scene-level resampling throughout (multiple frames of one scene are not independent); one
pre-registered primary readout per experiment with everything else marked as sensitivity analysis;
three-state adjudication (PASS / FAIL / **indeterminate**), with insufficient power always recorded
as indeterminate rather than forced into a binary; random-direction controls carry **their own
per-layer null distribution**; all cross-model comparisons use **within-model normalized** quantities
only. An amendment ledger is maintained throughout; the experiments reported here registered **39**
amendments, seven of which converted an already-obtained positive result back into a negative or
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

Expanding the candidate pool **does not improve this situation; it only changes its shape.** Of the
six candidates: SimLingo has only a CARLA Driving Score; LTF, DiffusionDrive, DiffusionDriveV2 and
AutoVLA nominally share NAVSIM PDMS but come from different splits, different versions of the metric
cache and different self-reported author pipelines (AutoVLA's PDMS exists only as the checkpoint
filename `AutoVLA_PDMS_89.ckpt`, which we did not re-run); and Alpamayo-R1 appears on neither
benchmark. In other words, after growing the pool from 2 to 6, **"comparable public scores" still
cover only a subset of the pool, and comparability even within that subset is nominal only.** We
therefore cite no public score we did not re-run ourselves as evidence in the main text.

This is not a limitation but the paper's point of departure. If candidates cannot be ranked at
all, the first half of the phrase "public leaderboard ranking is disconnected from deployment
ranking" is empty, and model selection must rest on other evidence. The four-axis matrix of the next
section is measured for all six models on **one stimulus set under one statistical protocol** and
therefore aligns cell by cell — the first and most basic increment this framework offers over public
scores.

### 4.2.2 The four-axis matrix under a common protocol

**Table 1. G/F/I/C readouts for six policies, measured on one and the same stimulus set under one and the same statistical protocol. Every cell is either a number with a scene-level CI, "n.m." (not measured, budget) or "n/a" (the operationalization does not apply — the reason is given in the notes). "n/a" is part of the result, not a caveat.**

**(a) Single-frame candidates — the D2cV falsification floor holds**

| Policy | G: primary − own D2cV floor | sd over 10 fold-seeds | F①: b-AUC(A vs D2a) | F①: b(A) [m/s] | F②: injection slope [m/s per σ] | I: $I_m$ (representation) | I: behavioural domain sensitivity | C-domain: profile shape / responsible layer | C-hazard: $C_m$ / responsible layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | +0.035 [−0.026, +0.097] | 0.024 | 0.534 [0.469, 0.592] | **+0.307** [+0.112, +0.500] | **−0.0405** (measurable) | **0.606** (L9) | 0.346 [0.249, 0.454] | cascade ρ=−0.997 / L0, entropy 0.279 | n.m. |
| DiffusionDrive | +0.009 [−0.047, +0.065] | 0.032 | 0.553 [0.486, 0.623] | +0.010 [−0.030, +0.045] | −0.00000 (not measurable) | 0.203 (L5) | 0.115 [0.080, 0.153] | interior peak ρ=+0.881 / L6, entropy 0.685 | **0.801** [0.710, 0.884] / L6, entropy 0.297 |
| **LTF** | **+0.070 [+0.017, +0.126]** | 0.028 | **0.583 [0.519, 0.639]** | +0.020 [+0.006, +0.033] | +0.00040 [−0.00065, +0.00143] (not measurable) | 0.583 (L6) | 0.160 [0.099, 0.235] | n.m. | **0.789** [0.725, 0.857] / L6, entropy 0.576 |
| DiffusionDriveV2 | +0.025 [−0.044, +0.097] | 0.026 | **0.556 [0.501, 0.613]** | **+0.201** [+0.052, +0.343] | not measurable | **n/a**¹ | n/a¹ | n.m. | **indeterminate**² |

**(b) Multi-frame candidates — D2cV is not a falsification floor (§CE/A34); the G column uses the permutation null instead**

| Policy | G: CV-AUC(A vs D2a) | permutation floor | random-direction floor | direction geometry purity ρ(proj, log area) | F①: b-AUC(A vs D2a) | F①: b(A) [m/s] | F② | I | C-hazard: profile shape / commitment layer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Alpamayo-R1 | 0.562 [0.502, 0.623] | 0.504 ± 0.012 | 0.508 ± 0.029 | +0.019 (p=0.73) | 0.446 [0.374, 0.519] | −0.080 [−0.178, +0.017] | n.m. | **n/a**³ | cascade ρ=−0.873 / **L16** (depth 0.47); $C_m$ **n/a**⁵ |
| **AutoVLA** | **0.608 [0.562, 0.654]** | 0.504 ± 0.020 | 0.522 ± 0.020 | +0.045 (p=0.28)⁴ | 0.508 [0.451, 0.563] | +0.037 [−0.035, +0.119] | n.m. | **n/a**³ | cascade ρ=−0.859 / **L20** (depth 0.58); $C_m$ **n/a**⁵ |

> ¹ DiffusionDriveV2 must be fed a real lidar BEV histogram, and the domain-paired corpus is purely
>   rendered with no point clouds; an all-zero histogram would fold "missing lidar" into the domain
>   divergence. The gap is on the stimulus side, not the model side (§CE/A36).
> ² The patch-ALL sufficient-cut-set self-check fails (median +0.552, far from 1) ⇒ the per-layer
>   shares are uninterpretable under this pairing and the verdict is indeterminate (§4.4.4).
> ³ The domain-paired corpus has one frame per timestep; both multi-frame candidates need 4 (§CE/A36).
> ⁴ AutoVLA's pooled ρ is not significant, but **within positives** ρ=+0.132 (p=0.025) and
>   ρ(proj, ecc) pooled = +0.098 (p=0.019): the direction is not fully orthogonal to imaging
>   geometry. D2a is caliper-matched on log area and ecc, so the **between-group** confound is
>   controlled; the residual within-group correlation is weaker than in the first version of this
>   readout, which used only the A+D2a cache and was fold-assignment noise (§CE/A38).
> ⁵ Both VLA candidates' recovery profiles are **step/cascade shaped** ($\rho < -0.7$), so under the
>   pre-existing applicability criterion of §4.4.4 the top-2-share formula's premise is violated and
>   $C_m$ adjudicates as indeterminate. This is **structural**: in a pure transformer stack the
>   residual stream is the only pathway, so patching any early layer leaves every deeper layer
>   clean-derived and recovery saturates at 1.0 early. The TransFuser family escapes this only
>   because each fusion block re-injects un-patched CNN features (§CE/A39). **$C_m$ is therefore not
>   comparable between the TransFuser family and VLA stacks**; the substitute readout shared by both
>   groups is the **commitment layer** (deepest layer with mean recovery ≥ 0.9, i.e. "how deep before
>   the decision is fixed") — all three TransFuser-family members have **no** commitment layer (no
>   single layer reaches 0.9), which is another way of stating "interior peak".
> C-domain and C-hazard are the **same patching method applied to two pairing sources** (sim↔real
> rendering pairs vs G1 clean↔ghost), not two metrics; the two columns are not comparable to
> each other.

The matrix yields five findings, developed in §4.2.3–§4.2.7.

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

**What this cell yields is exactly what public scores cannot:** 6.87 and 88.1 both report how much
performance was lost, never where the loss enters or what fixing it costs. Under this diagnostic,
SimLingo's repair site is the **visual front end** and DiffusionDrive's is the **deep fusion
stage** — two entirely different prescriptions.

**Changing the pairing source lets the same method answer a different question (C-hazard,
§CE/A27).** Replacing the "sim↔real rendering" pairing with "G1 clean↔ghost" changes the question
from "where does the rendering-domain failure enter" to "where does the hazard-induced behavioural
change enter". Under that pairing, DiffusionDrive and LTF — **two candidates sharing one encoder
architecture** — both localize the responsible layer at **L6** ($C_m$ 0.801 [0.710, 0.884] and 0.789
[0.725, 0.857] respectively, against a diffuse baseline of 0.250, with patch-ALL = +1.000 for both).
Two candidates give a **concordant** diagnosis here, and they are precisely the pair that shares an
encoder — which reads as a convergent-validity check on the readout. The next-but-one paragraph
qualifies how much that concordance actually establishes.

**DiffusionDriveV2 is indeterminate under the same pairing, and that cell deserves its own note.**
Its patch-ALL recovery median is only **+0.552**, far from 1 — swapping all 320 fused tokens at once
returns behaviour only halfway. Under the discipline of §4.1.3 this means the patched tokens are not
a sufficient cut set under this pairing and the per-layer shares are uninterpretable. **Its nominal
$C_m$ is 0.845 (median 0.990), the highest among the three TransFuser-family members — which share
one 8-layer diffuse baseline of 0.250, so that comparison is legitimate — and we adjudicate it
indeterminate rather than best.** This cell is a direct expression of the paper's reporting
discipline: when the self-check fails, the best-looking number is exactly the one that must not be
reported.

**Measuring the two VLA candidates turned this same cell into a conclusion about the
operationalization itself (§CE/A39).** Alpamayo-R1's and AutoVLA's recovery profiles are the
**exact opposite** of the TransFuser family's: not an interior peak but a **step** — recovery
saturates near 1.0 across L0–L20 for AutoVLA and L2–L16 for Alpamayo before decaying to 0
(Spearman $-0.859$ and $-0.873$ respectively). Both pass the patch-ALL self-check
(+1.000 / +1.001), so this is not instrument failure.

The reason is structural: **patching layer $L$'s output leaves every layer deeper than $L$ to be
recomputed from the clean side.** In a pure autoregressive transformer stack the residual stream is
the only pathway, so patching any early layer suffices to return behaviour fully to clean and the
profile must saturate. The TransFuser family escapes this **only because each of its fusion blocks
re-injects un-patched image features from the CNN branches**. Changing the patch scope does not help
either: patching only the image-token segment still leaves AutoVLA's profile saturated across
L0–L21 (Spearman $-0.876$).

This conclusion has consequences in both directions, and both must be written down.
**Forward**: $C_m$ is **not comparable between the TransFuser family and VLA stacks**, and both VLAs'
$C_m$ adjudicate as indeterminate under the pre-existing criterion of §4.4.4 — the same treatment
SimLingo's C-domain cascade received, not a new standard invented for them.
**Backward**: DiffusionDrive's and LTF's interior peak at L6 is a **property of TransFuser's
stage-wise re-injection design**, not a general fact about where hazard signals enter a network.
We read DiffusionDrive's and LTF's agreement above as a convergent-validity check; it now needs one
more clause: **they concord partly because they share one fusion design** — and the two VLAs concord
with each other too (both step-shaped, commitment layers L16 and L20), for the same reason: a shared
stack structure. **Concordance appearing along architecture-family lines is itself evidence that
what is being measured is the architecture, not only the model.**

One shape quantity remains well defined under a step profile and comparable between the two VLAs:
the **commitment layer** (the deepest layer with mean recovery ≥ 0.9, i.e. "how deep before the
decision is fixed"). AutoVLA's is L20 / 36 (depth 0.58), Alpamayo's is L16 / 36 (depth 0.47). All
three TransFuser-family members have **no** commitment layer (no single layer reaches 0.9), which is
another way of stating "interior peak".

**One further instrument check on the Alpamayo side**: its rollout is stochastic sampling
($top_p$ = 0.98, $T$ = 0.6), whereas AutoVLA decodes greedily at $top_k$ = 1. Across 5 seeds on the
same input, $v_{plan}$ has sd **0.430**, i.e. **0.330×** the median clean−ghost gap (1.303). Our
protocol locks both conditions and every patch run to one seed, so this noise is common-mode and
largely cancels (the measured patch-ALL of +1.001 is the evidence), making the floor a
**conservative upper bound**. It is nonetheless large enough that the wobble between 0.88 and 1.01
along Alpamayo's profile **cannot be interpreted**, and the commitment layer L16 is localized only
to the granularity of "mid-stack".

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

**With the expanded pool, this conclusion acquires a version that does not depend on anything
peculiar to SimLingo.** The architecture-neutral action-level counterfactual (F①) yields three
sharply different shapes across the six candidates:

| Shape | Candidates | Signature |
| --- | --- | --- |
| **Strong but non-specific reaction** | SimLingo, DiffusionDriveV2 | $b(A)$ significantly non-zero (+0.307 / +0.201) but $b$(D2a) of the same order or CI-covered |
| **Weak but specific reaction** | LTF | $b(A)$ is only +0.020, yet it is the only single-frame candidate whose b-AUC CI lies entirely above 0.5 (0.583) |
| **No reaction at all** | AutoVLA, Alpamayo-R1 | $b(A)$ **and** $b$(D2a) both indistinguishable from 0 |

The three shapes call for three different repairs: raise selectivity, raise gain, or connect the
pathway at all. And AutoVLA's raw G readout (0.608) is indistinguishable from the table's highest,
LTF's 0.623, so
**"the information is in the representation but does not drive the action" is cleaner on AutoVLA
than on SimLingo** (§4.2.7) — SimLingo at least reacts, merely non-specifically; AutoVLA has the
most clearly readable representation and a completely unmoved action.

(Restriction: a single-frame model is structurally blind to object motion and TTC contains a
velocity term, so a substantial part of $r$ = +0.320 should be attributed to the correlation between
TTC and **distance**. The argument is unaffected: whether the direction carries TTC or distance, it
is orthogonal to the brake axis either way.)

### 4.2.5 The I axis: representational and behavioural stability order the models oppositely

On the same rendering-domain pairing, the representational $D_{L^*}$ favours SimLingo
(0.394 vs 0.797; $I_m$ 0.606 vs 0.203) while the behavioural domain sensitivity favours
DiffusionDrive (0.115 vs 0.346), and **both pairs of scene-level bootstrap CIs are
non-overlapping**. Any procedure that compresses "domain robustness" into one number must pick one
of these orderings and discard the other. For all three measured candidates $v_{domain}$ is close to
orthogonal to their own $v_{hazard}$ (all $|\theta|$ below the $1/\sqrt{d}$ reference for their
dimension), so the observed domain sensitivity does **not** operate by contaminating the hazard
direction.

**Adding LTF strengthens this into a sharper claim.** LTF and DiffusionDrive share one
`TransfuserBackbone` **architecture** (different weights, different action heads), yet their $I_m$
differ by 0.38 (0.583 vs 0.203, non-overlapping CIs) and their behavioural domain sensitivities by
nearly 40% (0.160 vs 0.115). **The I-axis readout is therefore a property of the weights, not of the
architecture**: after two separate training runs, the same encoder code already distributes its
domain sensitivity differently along depth.

This claim only stands with a self-check attached. The peak layers differ (DD at L5, LTF at L6), so
the ordering could be an artefact of layer selection. Layer-matched comparison (per-layer $D_L$,
vision_mean):

| L | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | 0.033 | 0.001 | 0.667 | 0.056 | 0.633 | **0.797** | 0.730 | 0.956 |
| LTF | 0.156 | 0.004 | 0.777 | 0.362 | 0.679 | 0.672 | **0.417** | 0.588 |

Only L5–L7 favour LTF; L0–L4 all reverse. **"LTF is more domain-invariant than DiffusionDrive" is
therefore a conclusion conditional on "at each model's own concept peak layer"**, and must not be
read as "the whole encoder is more invariant" (§CE/A37). We put this qualification in the body
rather than a footnote, because it is precisely what distinguishes this framework from "reporting a
robustness score".

### 4.2.6 The first G-axis positive: the falsification floor is not uncrossable

The G-axis primary readouts of the first two candidates were both indeterminate (§4.4.1), which
leaves a genuine ambiguity: is it that "the model has no hazard concept", or that "our D2cV floor is
set so high that no model could cross it"? **LTF supplies evidence against the second reading**:
primary − own D2cV floor = **+0.070 [+0.017, +0.126]** (vision_mean), with a region_mean sensitivity
analysis of +0.085 [+0.031, +0.140] and $+0.056 \pm 0.028$ across 10 CV fold-assignment seeds — the
first time the sd is **smaller** than the effect. It is the first G-axis readout in this line of work
whose CI excludes 0.

Its value is not "LTF is better" but that it **calibrates the whole table with a positive**: under
the same D2cV floor, the same stimulus set and the same statistical protocol, some candidate does
cross the floor, so the other candidates' indeterminacy **cannot** be attributed to the floor being
uncrossable in principle. Before this, the conclusion of §4.4.1 could only read "indeterminate at
this sample size"; it can now read "indeterminate at a sample size at which another candidate
crossed".

Notably, LTF is also the only one of the four single-frame candidates that simultaneously satisfies
"b(A) significantly non-zero" and "b-AUC(A vs D2a) CI excluding 0.5" (0.583 [0.519, 0.639]) — it is
the candidate whose G and F point the **same** way. It contrasts directly with the next section.

### 4.2.7 The G/F dissociation is cleanest on AutoVLA

AutoVLA's raw G-axis readout (CV-AUC of A vs D2a) is 0.608 [0.562, 0.654], 2.9 sd above the
permutation floor mean, $p = 7.7 \times 10^{-6}$, with both pooling conventions agreeing on L20 and
giving near-identical readouts. On the raw-AUC scale — the one scale comparable across the two
groups — it is second only to LTF's 0.623 [0.577, 0.668], and the two CIs overlap heavily, so
**AutoVLA's and LTF's representational readout strengths are indistinguishable**;
and the F① closest to 0.5 in the pool (0.508 [0.451, 0.563]), with $b(A)$ **and** $b$(D2a) both
indistinguishable from 0 — its planned speed shows no detectable response at all to an object
appearing, not merely a non-specific one.

**This is the cleanest instance of the paper's central claim**: hazard-relevant information is in
the representation, more linearly readable than in any other candidate, and it does not reach the
action. The shape differs from SimLingo's: SimLingo has $b(A) = +0.307$, significantly non-zero,
with an equally large $b$(D2a) (**a strong but non-specific reaction**); AutoVLA does not react at
all. The two failure modes call for opposite repairs — raise the selectivity of the action in one
case, connect the pathway at all in the other — and any single composite score collapses them into
the same number.

The conclusion carries one qualification: AutoVLA's discriminative direction is not fully orthogonal
to imaging geometry (within positives $\rho(\log\text{area}) = +0.132$, $p = 0.025$; pooled
$\rho(ecc) = +0.098$, $p = 0.019$), so "high G" should be read as "**a large amount of linearly
readable, hazard-related information that is partly entangled with imaging geometry**", not as
"AutoVLA understands hazard better". For contrast, the same diagnostic on Alpamayo gives
$\rho = +0.019$ (n.s.), but its G is also much lower (0.562). **Semantic purity and readout strength
run in opposite directions across this candidate pool**, which is an observation worth testing
further rather than an established regularity — the more so because this very diagnostic moved
between the first (A+D2a-only) and final versions of the readout (§CE/A38).

### 4.2.8 Intervening on the diagnosed link

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
across encoder families; C's top-2-share formula has its **premise violated** on SimLingo. Weighting
these four cells into a scalar amounts to treating "non-comparable" and "indeterminate" as either
zero or as the median, and either treatment makes the provenance of the final ranking untraceable:
as soon as someone asks "why is A ahead of B?", the answer lands on the weights rather than on the
evidence. We therefore deliver a matrix **with blanks and explicit "not comparable" labels** rather
than a scalar that looks clean but cannot be audited.

**Growing the pool from 2 to 6 turns this argument from "it should be so in principle" into
something countable.** Of the 54 cells in Table 1, **12 carry no usable number**: 4 are n/a (the
operationalization does not apply), 5 are n.m. (not measured within budget), and 3 read "instrument
without resolving power" (F② injection across all three TransFuser-family members). A further
**10 cells carry a number but adjudicate as indeterminate**: three single-frame candidates' G, four
candidates' F①, DiffusionDriveV2's C-hazard (whose nominal value is the highest of the three
TransFuser-family members), and both VLAs' $C_m$ (step-shaped profile, so the formula's premise is
structurally violated). Synthesizing a scalar would require an imputation decision for each of those
22 cells. And their reasons fall into **five distinct kinds**: stimulus-side gaps
(DiffusionDriveV2's I axis lacks lidar; the multi-frame candidates lack temporal frames), violated
operationalization premises (the D2cV floor for multi-frame candidates; $C_m$ on a VLA stack),
instruments without resolving power (F② injection on diffusion heads and on the TransFuser encoder),
insufficient statistical power (most G and F① cells), and a failed self-check (DiffusionDriveV2's
C-hazard). **Filling all five kinds of absence with one imputed value collapses five different
statements of "we do not know" into one statement of "we know".**

More concretely: **no candidate in this table dominates on every measurable axis.** LTF is the best
single-frame candidate on G and F① but its I-axis ordering holds only at the peak layer; AutoVLA ties LTF for
the highest raw G yet has the F① closest to 0.5 (0.508); DiffusionDriveV2 has the second-largest $b(A)$ after
SimLingo but its C-hazard is indeterminate on a failed self-check; SimLingo is best on the
representational side of I and worst on the behavioural side. **Any choice of weights would decide
who ranks first, and no set of weights can be justified from the data itself.**

---

## 4.4 Ablation-like Analyses: why these numbers can be believed

Every item in this section is a **negative check**: its purpose is not to make numbers look better
but to exclude the case in which numbers look good while meaning nothing. This work registered 39
amendments during execution, seven of which converted an already-obtained positive result back into a
negative or indeterminate one; the five most consequential are given below.

### 4.4.1 The falsification floor: the primary readout must separate from "any VRU is present"

If the G-axis primary readout were compared only against random or permutation nulls, "one more
person in the frame" would be mistaken for "hazard was read out". We therefore require separation
from **D2cV** (same VRU class, same geometry, differing only in a relative velocity unobservable to
a single-frame model). The four single-frame candidates' differences are
SimLingo $+0.035$ [$-0.026$, $+0.097$], DiffusionDrive $+0.009$ [$-0.047$, $+0.065$],
DiffusionDriveV2 $+0.025$ [$-0.044$, $+0.097$] and **LTF $+0.070$ [$+0.017$, $+0.126$]**; across 10
CV fold-assignment seeds, $+0.002 \pm 0.024$, $+0.012 \pm 0.032$, $+0.004 \pm 0.026$ and
$+0.056 \pm 0.028$ respectively. For **the first three, the sd is of the same order as, or larger
than, the effect**, so the G axis is indeterminate at this sample size, with the bottleneck
localized to the D2cV sample size (212), not to pooling or model choice.

**LTF is the sole exception, and it changes how this whole paragraph reads** (see §4.2.6): its
effect exceeds twice the fold-assignment sd for the first time, and its CI excludes 0. Before this,
"three candidates are all indeterminate" and "the floor is set so high that nobody could cross it"
were indistinguishable explanations; LTF's positive rules out the latter. **Under the same floor,
the same stimulus set and the same statistical protocol, a candidate did cross, so the remaining
candidates' indeterminacy is a conclusion about those candidates, not about the floor.**

**The two multi-frame candidates (Alpamayo-R1, AutoVLA) do not enter this table.** D2cV is
constructed so that "the only difference is a relative velocity a **single-frame** model is
*physically unable* to observe". For a multi-frame model relative velocity *is* observable, D2cV
degenerates into a legitimate hard negative and **ceases to be a falsification floor**; "A vs D2cV
is significant" is not a leakage alarm for them. Their G axis is therefore reported against the
label-permutation null and the random-direction floor, and grouped separately in Table 1
(§CE/A34). This is not a relaxed standard for them but **the correct application of the same
discipline under a different input temporality** — forcing a floor whose premise does not hold
yields a "pass" or "fail" that means nothing either way.

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
dose do not change the anchor selection. **Corollary (corrected 2026-08-31, see §CE/A31)**: operationalized as an injection effect size,
the F axis is **not comparable across encoder families**; cross-model alignment must use the
architecture-neutral action-level counterfactual test.
**The original corollary said "across action-head families" and has been refuted by this round's
same-encoder controlled comparison**: LTF shares the identical `TransfuserBackbone(latent=True)`
encoder with DiffusionDrive and has a continuous regression head rather than a diffusion head, so by
the original corollary it should be measurable — **it measurably is not**
($v_{hazard}^{ltf}$ slope +0.00040 [−0.00065, +0.00143]; the by-construction-effective
$v_{brake}^{ltf}$ only −0.00004; neither exceeds its same-layer null). DiffusionDriveV2, a third
member of the same encoder family, gives the same result. Measurability is therefore determined by
the **encoder / injection site**: SimLingo (ViT + LLM residual stream) is measurable and reproducible
by an analytic Jacobian, while all three TransFuser-family members (under two different action heads)
are not.

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
responsible-layer argmax. $C_m$'s diffuse baseline varies with depth (0.250 at 8 layers vs 0.083 and
0.056 at 24 and 36), so **its value is not comparable across models**.

**With the expanded pool, this criterion is upgraded from "one model happens to be a cascade" to
"one class of architecture must be a cascade".** The five candidates' C-hazard profiles separate
cleanly **by architecture family**: all three TransFuser-family members are **increasing** profiles
(Spearman $+0.881$ / $+0.929$ / $+0.810$, interior peak at L6), and both VLAs are **step** profiles
($-0.859$ / $-0.873$, saturating at 1.0 from around L0). The reason is in §4.2.3: in a pure
transformer stack, patching layer $L$ leaves every deeper layer clean-derived, so early saturation is
forced; the TransFuser family escapes only because each fusion block re-injects un-patched CNN
features. The applicability criterion is therefore not a just-in-case robustness appendix: **it cuts
the candidate pool exactly along the architecture-family boundary.** Read literally, $C_m$ would
place the two VLAs' 0.079 / 0.091 (against a 36-layer baseline of 0.056) in one ordering with the
three TransFuser members' 0.789–0.845 (against an 8-layer baseline of 0.250), producing a ranking
determined entirely by depth and architecture and unrelated to whether failure is concentrated.

Under a step profile we report the **commitment layer** instead (the deepest layer with mean recovery
≥ 0.9): the one informative quantity in that regime, and comparable between the two VLAs (AutoVLA
L20/36, Alpamayo L16/36). None of the three TransFuser-family members has a commitment layer (no
single layer reaches 0.9 recovery) — **the same quantity degenerating to opposite ends on the two
families is precisely what shows it is a shape statistic and not a score.**

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

> Sources: `amendments.md` (all 39 amendments), `analytic_vs_empirical.md`,
> `c_axis_shape_diagnostics.json`, `cosine_matrix.json`, `generalizable_tips.md`.

---

## 4.5 Discussion of Limitations

**1. The third line of the three-way comparison is not genuinely closed.** Our pilot post-train is a
single-model, single-metric validation under a fixed small budget (229k parameters), not a full
measurement of the target variable "held-out-domain safety-score increment under a fixed
post-training budget". The paper therefore establishes that the four axes are measurable,
non-redundant, and that the link they localize can be intervened upon — **not** that four-axis scores
predict post-training gains. The latter requires a larger budget or a closed-loop consequence metric.

**2. For three of the four single-frame candidates the G axis can currently report only "readable /
not readable", not "grounded / not grounded".** SimLingo's, DiffusionDrive's and DiffusionDriveV2's
primary-minus-floor readouts are indeterminate, bottlenecked by the D2cV sample size (212). The
limitation persists when the trigger criterion is changed (three independent criteria A / B / C give
differences of $+0.003$ / $-0.022$ / $+0.023$, all with CIs crossing zero), so it is not a
peculiarity of one scenario type. **LTF is the one candidate that crosses the floor**, which narrows
this from "a limitation of the method" to "a limitation of those three candidates at this sample
size".

**3. The falsification floor does not hold for multi-frame candidates, and no ready substitute
exists.** D2cV rests on relative velocity being structurally unobservable to a single-frame model,
which is false for Alpamayo-R1 and AutoVLA. For these two we fall back to the permutation null and
the random-direction floor, which can only exclude "pure chance" and **cannot exclude "what was read
out is the presence of any VRU"** — precisely what D2cV exists to guard against. **The G columns of
Table 1(b) and Table 1(a) are therefore readouts of different constructs and must not be compared
across the two groups.** Filling this gap requires constructing a further negative class that is
"same class, same geometry, **same relative velocity**, differing only in the label", which on
nuScenes means re-mining the corpus and was out of scope this round.

**4. Parts of the F and C operationalizations are not portable, and both non-portability boundaries
fall along architecture families.** F's injection protocol is unmeasurable across the entire
TransFuser encoder family (three members under two different action heads, all unmeasurable,
§4.4.2); C's top-2-share formula has its premise **structurally** violated across the entire VLA
stack (both members give step profiles, §4.4.4, §CE/A39), and on DiffusionDriveV2 the patch-ALL
sufficient-cut-set check additionally fails, forcing an indeterminate verdict. **Neither boundary is
randomly placed: F② cuts along the encoder family and $C_m$ cuts along "is it a pure transformer
stack".** None of these is a statement that the model is poor on that axis; all are statements that
the operationalization does not apply to that encoder, profile shape or pairing. We therefore
decline to fold them into a single score.

**One backward consequence that must be acknowledged**: DiffusionDrive and LTF localize C-hazard to
the same layer L6, which we presented as a convergent-validity check. Measuring the two VLAs shows
that **this interior peak is a property of TransFuser's stage-wise re-injection design**, and the
two concord partly because they share one fusion design. That convergent validity therefore holds
only *within* the encoder family and is not evidence for the general validity of the C-hazard
readout.

**5. The I axis covers only half the pool, and every gap is on the stimulus side.**
DiffusionDriveV2 needs real lidar and both multi-frame candidates need 4 timesteps, while the
domain-paired corpus is single-frame and purely rendered. These three cells are marked n/a because
the *stimuli* are missing, not because the models are unmeasurable; padding with zeros or duplicated
frames would fold variables unrelated to rendering style into the domain divergence (§CE/A36).
**Moreover, the I-axis ordering we did measure holds only at each model's own peak layer**: LTF's
and DiffusionDrive's per-layer $D_L$ reverse sign on L0–L4 (§CE/A37).

**6. The domain pairing is not a 3DGS reconstruction.** We use a same-geometry dual-rendering pair
(CARLA engine rendering ↔ world-model photorealistic re-rendering), whose causal structure matches
$do(\text{appearance})$, but the scope of the conclusion should be stated as that pairing rather than
as sim ↔ real in general.

**7. The random-direction control for injection experiments has no resolving power at this sample
size.** Across 7 per-layer null distributions, none of 11 directions exceeds its own same-layer null —
**including** $v_{brake}$, which is effective by construction and independently confirmed by the
analytic method. The null sd varies by a factor of 50 across layers (0.0012 to 0.0642), of the same
order as or larger than the effects under test. All steering causal verdicts in this paper are
therefore recorded as indeterminate with the attribution explicitly on the instrument side, and
**must not be read as "these directions have no causal effect"**.

**8. The pilot's coupling target passes only a weak instrument-side check.** $\hat v_{hazard}$@L23
correlates with the residualized real $a_{brake}$ on held-out data at $\rho$ = $+0.053$
($p$ = 0.183) — correct sign, not significant. The accurate statement is "coupling the action to a
direction that reads hazard only in a weak-signal sense", not "connecting the action to the hazard
readout does not help".

**9. The native training domains of all five NAVSIM-family candidates are not independently verified**, so every interpretation
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
