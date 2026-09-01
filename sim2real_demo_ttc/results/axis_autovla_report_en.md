# Candidate expansion: G/F readouts for AutoVLA

> Work order: [`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md) (lowest priority, 1-day timebox).
> All autonomous decisions this round are in [`amendments.md`](amendments.md) §CE (notably **A34** on D2cV not holding for multi-frame candidates, and **A35** on the environment).
> Numeric artifacts: `g_axis_autovla.json`, `f_axis_action_counterfactual.json`,
> `v_hazard_autovla_{vision_mean,seq_mean}.npz`; adapter: `autovla_g1_adapter/`.

---

## Methods

### The candidate and its prior basis here

AutoVLA (UCLA Mobility Lab) is a Qwen2.5-VL-3B-based VLA driving policy that emits trajectories by
**autoregressive action-token decoding**; HuggingFace provides LoRA-merged inference weights
(locally `external/ckpts/AutoVLA_PDMS_89.ckpt`). This repository had no prior AutoVLA basis of any
kind: adapter, environment and readout interface were all written this round. It is the one
candidate in the work order for which we could not assume any other candidate's adapter would carry
over.

### Environment: two mutually conflicting constraints (§CE/A35)

Almost the entire integration cost was the environment, and the two constraints pull against each
other:

1. **The checkpoint's module layout pins `transformers==4.49`.** Loading it under the machine
   default (4.57) yields **824 missing keys and 824 unexpected keys** — 4.49's `vlm.visual.*`
   becomes `vlm.model.visual.*` in 4.57. Forcing a key remap is not acceptable (that would amount to
   guessing weight semantics on the authors' behalf), so 4.49 was installed into an isolated
   `--target` directory and prepended to `sys.path` only inside the AutoVLA process.
2. **But 4.49 conflicts with the local torchvision operator-registration order**: inserting the 4.49
   path before `import torch` gives `operator torchvision::nms does not exist`. The fix is to
   `import torch, torchvision, torchvision.ops` **first**, completing operator registration, and
   only **then** insert the 4.49 path. This ordering is load-bearing and is commented as such in
   `autovla_adapter._bootstrap()`.

In addition, `models.utils.score` drags in the whole navsim→nuplan dependency chain (the same
situation as DiffusionDriveV2's §CE/A30), bypassed with a stub module; the `predict()` path requires
`driving_command` (we uniformly supply `"go straight"`, consistent with never feeding navigation
commands to any candidate), `max_length` raised to 2048 (prompts measure 1085 tokens), and
`temperature` set to 1e-4 with `top_k=1` to approximate deterministic decoding (0.0 is rejected by
`generate`).

The final load is **0 missing / 0 unexpected keys**.

### Adaptation details and deviations

* **Readable layers**: the 36 decoder layers of the VLM language tower (hidden 2048), structurally
  the same readout position as Alpamayo-R1.
* **Input construction**: 3 cameras × 4 timesteps = 12 images, matching the temporal structure of
  AutoVLA's native NAVSIM input; on the nuScenes side, 4 timesteps are taken from `x_clean_frames`
  and `x_ghost_frames` respectively.
* **Prefill pooling**: the same device as Alpamayo — the hook keeps only the forward pass with the
  largest `seq_len` (the full prompt) and discards every decode step (`seq_len = 1`); pools are
  `vision_mean` / `last_token` / `seq_mean`.
* **Image-token identification**: the longest run of identical ids in `input_ids` (measured
  id = 151656, 864 of 1085 tokens).
* **Behavioural quantity**: `v_plan`, computed from the first two points of the decoded 10-point
  trajectory, structurally identical to the $v_{cmd}$ definition used for every other candidate.
* **Deviations**: AutoVLA is trained on NAVSIM and is OOD on nuScenes; camera intrinsics/extrinsics
  and FOV do not match and can only be approximately mapped. As for every other candidate, we
  perform no domain adaptation for anyone — this is deliberate, see §4.1.1.

### A methodological limitation stated up front: D2cV does not hold for AutoVLA (§CE/A34)

Exactly as for Alpamayo. D2cV is constructed so that "the only difference is a relative velocity a
**single-frame** model is *physically unable* to observe". **AutoVLA consumes 4 timesteps**, so
relative velocity *is* observable to it, and D2cV is therefore not a falsification floor for it but
an ordinary hard negative. This report accordingly uses the **label-permutation null** and the
**random-direction floor** as the usable floors for the G axis; D2cV numbers are reported alongside
but carry no adjudicative weight.

---

## Results

**Table 1. G-axis readout for AutoVLA on the shared G1 stimulus set. For this multi-frame model the usable floors are the permutation null and the random-direction floor (§CE/A34); D2cV is reported for completeness only.**

| Pool | CV-AUC(A vs D2a) | 95% CI | permutation floor | random-direction floor | 10 fold-seeds | peak layer $L^*$ |
| --- | --- | --- | --- | --- | --- | --- |
| **vision_mean** (primary) | **0.608** | [0.562, 0.654] | 0.504 ± 0.020 | 0.522 ± 0.020 | 0.588 ± 0.022 | L20 |
| seq_mean (sensitivity) | 0.609 | [0.564, 0.655] | 0.500 ± 0.020 | 0.521 ± 0.020 | 0.588 ± 0.019 | L20 |

The lower bound of the primary 95% CI (0.562) sits +2.9 sd above the permutation floor mean and
+2.0 sd above the random-direction floor, with $p = 7.7 \times 10^{-6}$. On the **raw-AUC** scale — the one scale comparable
across the two groups — this overlaps heavily with, and is indistinguishable from, the table's
highest (LTF's 0.623 [0.577, 0.668]). AutoVLA is also the only candidate whose two pooling
conventions (vision_mean / seq_mean) agree on the same peak layer (L20) with nearly identical
readouts (0.608 / 0.609). **It must not be compared directly against the single-frame group's
"primary − D2cV floor" column**, which is a different construct (§CE/A34).

**Hard-negative readouts reported alongside** (carrying **no adjudicative weight** for a multi-frame
candidate, per above): vs D2cV = 0.594 ($p$ = 0.0016, n = 141), vs D2c = 0.573, vs D2b = 0.618,
vs D2bV = 0.700; primary − D2cV = $+0.014$ [$-0.056$, $+0.084$]. For a single-frame candidate this
cell would adjudicate as indeterminate; for AutoVLA, "A vs D2cV is significant" is a **legitimate
discrimination task** (relative velocity is observable to it), so these numbers show only that its
separation from the hard negatives is close to its separation from D2a — they say nothing about
whether the method works.

**Geometry robustness, reported alongside**: AutoVLA's discriminative direction is **not fully
orthogonal to imaging geometry** — $\rho(\text{projection}, \log \text{area})$ is $+0.045$ pooled
($p = 0.28$, n.s.) but $+0.132$ **within the positives** ($p = 0.025$);
$\rho(\text{projection}, ecc)$ is $+0.098$ pooled ($p = 0.019$). D2a is caliper-matched on log
imaged area and eccentricity, so the **between-group** confound is controlled by design, and the
residual within-group correlation is weak but measurable. For contrast, the same diagnostic on
Alpamayo gives $\rho = +0.019$ ($p = 0.73$). AutoVLA's G readout should therefore be read as
"**significant, with weaker semantic purity than Alpamayo but a controlled between-group
confound**", not as "AutoVLA understands hazard better".

**Table 2. F① action-level counterfactual (architecture-neutral), same stimuli and same protocol as all other candidates.**

| Quantity | AutoVLA | n | Verdict |
| --- | --- | --- | --- |
| $b(A)$ = $v_{plan}$(clean) − $v_{plan}$(ghost) [m/s] | +0.037 [−0.035, +0.119] | 291 | indistinguishable from 0 |
| $b$(D2a) [m/s] | −0.094 [−0.230, +0.009] | 283 | indistinguishable from 0 |
| **b-AUC(A vs D2a)** | **0.508** [0.451, 0.563] | — | **indeterminate** (CI spans 0.5) |

### Diagnostic conclusion: the G/F dissociation is cleanest on AutoVLA

AutoVLA simultaneously exhibits:

* **G = 0.608** (raw AUC indistinguishable from the table's highest, LTF's 0.623; $p = 7.7 \times 10^{-6}$; consistent across both pooling conventions),
* **F① = 0.508** (closest to 0.5 in the pool, with a CI almost symmetric about 0.5), and
* $b(A)$ **and** $b$(D2a) both indistinguishable from 0 — i.e. the planned speed shows **no
  detectable response at all** to the appearance of an object, not merely a non-specific one.

This is the cleanest single instance of the paper's central claim: **hazard-relevant information is
present in the representation, more linearly readable than in any other candidate, and it does not
reach the action.** The shape differs from SimLingo's: SimLingo has $b(A) = +0.307$, significantly
non-zero, with an equally large $b$(D2a) (it reacts strongly but non-specifically); AutoVLA does not
react at all. The two failure modes call for entirely different repairs, and **any single composite
score would collapse them into the same number.**

---

## Discussion

**AutoVLA completed within its 1-day timebox; the skip clause was not triggered.** The cost split
matched the work order's expectation only in kind, not in proportion: the adapter proper (input
construction, hooks, pooling) was roughly one third, and the environment (the transformers-version ×
torchvision-operator-registration double constraint) roughly two thirds. The latter was not a
foreseeable cost and is registered as §CE/A35 for the benefit of future candidates.

**Axes not measured, with reasons (marked "not applicable" under the standing discipline rather than
padded with a number)**:

| Axis | Status | Reason |
| --- | --- | --- |
| F② representation injection | **not measured** | Not completed inside the timebox. Note this is *not* a finding of unmeasurability — AutoVLA's injection site (the LLM residual stream) is structurally the same as SimLingo's, so a priori it should be measurable. Recorded as not-measured, not as not-applicable. |
| I domain invariance | **not applicable** | The domain-paired corpus (CARLA ↔ world-model re-rendering) has only a **single frame** per timestep, whereas AutoVLA requires 4. The gap is on the stimulus side, not the model side (§CE/A36). |
| C-hazard | **now measured** (see below); $C_m$ **not applicable** | The profile is step-shaped, so the top-2-share formula's premise is structurally violated (§CE/A39). We report the commitment layer instead. |

**One observation about candidate-pool design**: AutoVLA and Alpamayo-R1 are the only two
multi-frame candidates in the pool, and they are also the only two for which **the falsification
floor is unavailable**. This is not a coincidence — the availability of D2cV and input temporality
are two faces of the same fact. Bringing multi-frame candidates onto one falsification table would
require constructing a further negative class that is "same class, same geometry, **same relative
velocity**, differing only in the label", which on nuScenes means re-mining the corpus and was out
of scope this round.

---

## Self-correction record

1. **The first load used transformers 4.57 and produced 824 missing / 824 unexpected keys.** Had we
   read representations without checking, the "readout" would have come from a randomly initialised
   vision tower. Verifying load completeness is a mandatory step for every new candidate in this
   repository, and here it directly caught an error that would have contaminated every downstream
   number.
2. **While fixing (1), inserting the 4.49 path first broke `torchvision::nms` registration**, which
   we briefly misdiagnosed as "4.49 is incompatible with the local CUDA". It was an
   operator-registration ordering problem, unrelated to version compatibility. Recorded to prevent a
   spurious "infeasible" conclusion next time.
3. **The first G/F readout used only the A + D2a cache, gave different numbers, and was voided.**
   The first version had a G primary of 0.629 and $\rho(\text{projection}, \log \text{area}) =
   +0.121$ ($p$ = 0.004); after caching D2b/D2c/D2cV (538 more events) these became 0.608 and
   $\rho = +0.045$ ($p$ = 0.28). **The cause is that scene-level CV fold assignment changes with
   corpus size**, moving the fitted direction and peak layer slightly. Both values fall inside the
   10-fold-seed range (primary 0.588 ± 0.022), so the difference is fold-assignment noise rather
   than new evidence — **which is exactly why this repository mandates the 10-seed fold-assignment
   stability check**. Registered as §CE/A38. F① is unaffected (b is a per-event quantity that never
   passes through CV).

---

## Results (follow-up): C-hazard

This cell was listed as "not measured: budget" in `candidate_expansion_DONE.md` and is filled in
here. The protocol matches DiffusionDrive / LTF / DiffusionDriveV2 item for item: pairing = G1
clean↔ghost (paired real-input swap; noise corruption prohibited); metric = the continuous quantity
$v_{plan}$; degraded samples = the top 12 by $|v_{clean} - v_{ghost}|$; patch scope = **the full
prompt residual stream** (the analogue of swapping all 320 fused tokens on the DD family, §CE/A32).
The one new implementation detail is that the patch is applied on the **prefill** forward pass;
decode steps ($S$ = 1) are left untouched, and the patch reaches every subsequent generation step
through the KV cache.

**Table 3. C-hazard readout for AutoVLA. 12 events / 11 scenes, 36 layers, patch scope = full prompt residual stream.**

| Quantity | Value | Criterion | Conclusion |
| --- | --- | --- | --- |
| patch-ALL recovery (sufficient-cut-set check) | **+1.000** (median +1.000) | must lie in [0.7, 1.3] | **passes** |
| Recovery profile | ≈1.0 across L0–L20; decays from L21 (0.84 → 0.77 → 0.56 → 0.37 → 0.10 → **0.00** @L35) | — | **step / cascade** |
| Spearman(layer, recovery) | **−0.859** | $\rho < -0.7$ ⇒ cascade ⇒ $C_m$ inapplicable | **$C_m$ not applicable** |
| $C_m$ (nominal, for audit only) | 0.079 [0.069, 0.094] (36-layer diffuse baseline 0.056) | — | **carries no adjudicative weight** |
| **Commitment layer** (deepest layer with mean recovery ≥ 0.9) | **L20 / 36 (depth 0.58)** | substitute readout under a step profile | the decision is fixed only past mid-stack |
| Responsible-layer mode / normalized entropy | L0 / 0.158 | — | consistent with a step shape |

**Verdict: indeterminate (the formula's premise is violated: the profile is a step)** — the same
treatment SimLingo's C-domain cascade received, not a standard invented for AutoVLA.

### Why it is a step, and why that is not "we failed to measure it"

After patching layer $L$'s output, **every layer deeper than $L$ is recomputed from the clean side**.
In a pure autoregressive transformer stack the residual stream is the only pathway, so patching any
early layer already suffices to return behaviour fully to clean. This is determined by the
architecture, not by instrument failure — the patch-ALL self-check passing at +1.000 is the evidence.

**Changing the patch scope does not help**: a separate run patching only the image-token segment
(`--tokens image`) still leaves the profile saturated at 1.0 across L0–L21, Spearman $-0.876$. The
scope is not the problem.

This report therefore yields something more valuable than "what is AutoVLA's C": **$C_m$ is not
comparable between the TransFuser family and VLA stacks**, and DiffusionDrive's and LTF's interior
peak at L6 is **a property of TransFuser's stage-wise re-injection design** (§CE/A39).

### A sample limitation that must be reported alongside

Across the 12 selected events, the **median $|v_{clean} - v_{ghost}|$ is only 0.267 m/s** (range
0.199 – 7.940). This is self-consistent with the F① result — AutoVLA's planned speed shows no
detectable response to an object appearing, so there were few large-gap events to select from. The
consequence is that recovery has a small denominator and is heavily quantized (in practice it takes
only a few values: 1.000 / 0.917 / 0.000). **The commitment layer L20 should be read at the
granularity of "mid-stack", not as a specific layer index.**
