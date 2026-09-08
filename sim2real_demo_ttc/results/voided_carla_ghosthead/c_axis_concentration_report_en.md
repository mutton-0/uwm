# T-C: Failure Concentration (C axis) — Layer-wise Causal Localization in DiffusionDrive

> Axis letters follow the G/F/I/C/D naming finalized on 2026-08-29; the mapping from the
> legacy R/S/I/C/D notation used in earlier files is given in
> [`axis_naming_alignment.md`](axis_naming_alignment.md).
> Pre-registration: [`amendments.md`](amendments.md) §FA.0. Numerical artifacts:
> `c_axis_concentration.json`, `c_axis_tsne.json`, `figures/c_axis_tsne.png`.

---

## Motivation

The final link of the four-axis causal chain concerns **repair cost**. When a policy degrades
under domain shift, is that degradation attributable to a spatially concentrated, identifiable
mechanism inside the network — one amenable to cheap targeted intervention, e.g. LoRA on two or
three layers — or is it a diffuse representational drift that requires broad retraining? This is
what the C axis (Concentration) measures; its theoretical lineage is Localization-for-Editing
(Hase et al. 2023; Meng et al. 2022).

Position in the causal chain: G asks *did the model see it correctly*, F asks *did what it saw
drive the action*, I asks *does that chain survive a domain switch*, and **C asks: when it does
not, where is the break and how expensive is it to fix**. High C means the gap between public
leaderboard scores and real deployment behaviour is **localizable and cheap to repair**;
low C means the same gap implies retraining-scale cost. These two situations are
indistinguishable from a leaderboard score, which is precisely the increment the four-axis
framework provides over open-loop score ranking.

Per work-order §5 this task is scoped as **reuse-first and low-cost**: the layer-wise activation
patching for DiffusionDrive was already completed in prior work
(`outputs/ghosthead_infer/patching/recovery.csv`). This experiment performs only the
standardized readout, inference and adjudication; **no causal experiment was re-run**.

---

## Method

**Specimen and stimuli.** DiffusionDrive (checkpoint `diffusiondrive_sim_navhard.ckpt`) evaluated
on the ghosthead domain-paired scene set. The pairing renders one and the same scene twice:
`transfered` = CARLA engine rendering (reference side) and `origin` = world-model photorealistic
re-rendering (degraded side). Scene geometry, framing, actors and ego ground truth are identical;
rendering style is the only variable. Patching uses the **n = 12** scene-variants with the largest
real-side degradation, at frame t = 1 s.

**Causal readout protocol (frozen in advance, following Zhang & Nanda's best practice).**
Corruption is always **swapping of paired real inputs** (noise-based corruption is prohibited):
a real-input forward pass is run, the output of the encoder SelfAttention at layer L is replaced
wholesale by the corresponding sim-side activation of the same scene, and the fraction of the
trajectory returned to the sim prediction is recorded as $\mathrm{recovery}(L)$. The metric is
always **continuous trajectory recovery** (binarization is prohibited). The readable layers are
the 8 `encoder_selfatt` modules of the TransFuser encoder (4 scales × 2 blocks). Methodological
self-consistency was established in the prior work: patching all 8 layers yields
recovery = +1.00 in 12/12 scenes.

**Pre-registered primary readout.**

$$C_m \;=\; \frac{\text{sum of recovery over the top-2 layers}}{\sum_L \mathrm{recovery}(L)}$$

Three protocol rules, all registered before the analysis: (i) $\mathrm{recovery}(L) < 0$ means
that substituting the reference activation made matters *worse* and is not evidence of
localization, so the primary readout **clips recovery at 0** before forming the ratio, with the
unclipped version reported alongside as a sensitivity analysis; (ii) $C$ is computed
**per scene and then aggregated**, not by aggregating recovery first; (iii) bootstrap resampling
uses the **scene** as the resampling unit (n = 12, 5000 resamples).

**Decision rule.** The diffuse baseline is $2/L = 2/8 = 0.250$, the top-2 share expected if
recovery were spread uniformly over the 8 layers. Three-state adjudication: scene-level bootstrap
95% CI entirely above the baseline ⇒ PASS; entirely below ⇒ FAIL; straddling it ⇒ indeterminate.

**Auxiliary (exploratory).** t-SNE of the 8-layer activations for the same scenes, coloured by
sim (reference / normal) vs real (degraded), together with the full-space silhouette coefficient.
**This is explicitly labelled correlational evidence and cannot substitute for the causal
conclusion obtained by patching.**

---

## Results

**Table 1. Concentration of causal responsibility across the 8 encoder self-attention layers of DiffusionDrive under paired sim↔real input swapping (n = 12 scene-variants, t = 1 s).**

| Readout | Definition | Estimate | 95% CI | Reference | Verdict |
| --- | --- | --- | --- | --- | --- |
| $C_m$ (primary) | top-2 share of recovery, recovery clipped at ≥ 0 | 0.858 | [0.771, 0.939] | diffuse baseline 0.250 | **PASS** |
| $C_m$ (sensitivity) | as above, unclipped | 0.347 | [−0.364, 0.962] | diffuse baseline 0.250 | indeterminate (denominator crosses zero) |
| top-1 share | single-layer share of recovery | 0.618 | [0.503, 0.744] | diffuse baseline 0.125 | — |
| top-3 share | — | 0.934 | [0.887, 0.977] | diffuse baseline 0.375 | — |
| top-4 share | — | 0.988 | [0.974, 0.997] | diffuse baseline 0.500 | — |

**Table 2. Layer-wise localization of the causal site, and its dissociation from divergence-based symptoms.**

| Layer | mean recovery (clipped ≥ 0) | scenes with argmax here | JS(sim‖real) | 1 − CKA |
| --- | --- | --- | --- | --- |
| L0 | 0.001 | 0 | 0.138 | 0.145 |
| L1 | 0.000 | 0 | 0.000 | 0.267 |
| L2 | 0.125 | 1 | 0.038 | 0.012 |
| L3 | 0.008 | 0 | 0.043 | 0.000 |
| L4 | 0.040 | 1 | 0.031 | 0.101 |
| L5 | 0.200 | 2 | 0.044 | 0.112 |
| **L6** | **0.380** | **5** | 0.009 | 0.284 |
| L7 | 0.345 | 3 | **0.288** | **0.775** |

The causal peak is at **L6** (mean recovery 0.380; argmax in 5 of 12 scenes; L4–L6 jointly
account for 8/12 = 0.667), whereas both divergence measures peak at **L7** (JS = 0.288,
1 − CKA = 0.775) and L6 has nearly the lowest JS of any layer (0.009). The normalized entropy of
the responsible-layer distribution is **0.685** (0 = fully concentrated in one layer,
1 = uniform over 8 layers).

**Table 3. Auxiliary (correlational) evidence: separability of degraded vs reference activations.**

| Layer | silhouette (sim vs real, full space) | n |
| --- | --- | --- |
| L0 | 0.090 | 432 |
| L5 | 0.112 | 432 |
| L6 | 0.073 | 432 |
| L7 | 0.033 | 432 |

The t-SNE panel is `figures/c_axis_tsne.png` (open circles mark the 12 scenes that entered the
causal patching set). All four silhouettes are < 0.12, i.e. **sim and real do not form two clean
clusters in the pooled activation space**, and the most separable layer (L5) is neither the causal
peak (L6) nor the divergence peak (L7).

> **Instrument side.** The corruption protocol (paired input swapping) and the metric (continuous
> trajectory recovery) were frozen before the analysis; patch-ALL = +1.00 (12/12) establishes that
> the 8 self-attention modules form a sufficient cut set, so the denominator of $C_m$ is
> meaningful. The distance between the primary readout and the diffuse baseline (0.858 vs 0.250)
> greatly exceeds the scene-level bootstrap width (±0.09), so the verdict does not hinge on any
> single scene. **The unclipped sensitivity arm has a CI crossing zero, which shows the clipping
> rule is not optional but a precondition for the ratio to be defined.**
> **Specimen side.** For DiffusionDrive, the trajectory degradation induced by the sim→real
> rendering shift is causally concentrated in the deep fusion stage of the encoder (chiefly L6,
> with L4–L7 covering 8/12 scenes). This places the model in the "localizable, cheap to repair by
> targeted intervention" bracket, and targeted intervention should be placed at **L6/L5**, not at
> the L7/L0 sites suggested by divergence measures.

---

## Discussion

**1. Update to the four-axis picture.** The C axis yields the cleanest positive result among the
four sub-experiments of this round: $C_m$ = 0.858 [0.771, 0.939], far above the diffuse baseline
of 0.250, with the top-1 layer alone accounting for 0.618. Consequently, once G/F/I have located
*which link* of the perception–action chain is broken, C additionally tells us that in
DiffusionDrive **that break is spatially concentrated**, placing the repair budget in the
"LoRA on two or three layers" bracket rather than the "full-parameter retraining" bracket.
This is exactly the information a public leaderboard cannot supply: a NAVSIM open-loop score
reports how much performance was lost, not which layer the loss enters through, nor what fixing
it would cost.

**2. Symptom ≠ cause, independently reproduced here.** The divergence peaks (JS and 1 − CKA, both
at L7) are dissociated from the causal peak (recovery, at L6), and L6 has close to the lowest
attention divergence of any layer. Following the intuition "intervene where the representation
changes most" would place LoRA at L7/L0, i.e. at the site with the most conspicuous *symptom*
rather than the strongest causal contribution. The t-SNE silhouette profile supplies yet a third
ordering (peak at L5). The three orderings do not coincide, which constitutes evidence
independent of this project's other experiments that **separability, divergence and causal
responsibility are three distinct quantities that cannot substitute for one another**.

**3. What this readout may not claim.** (i) A high C establishes that the failure is
*localizable*, not that "patching L6 suffices" — the latter is a stronger claim requiring the
training-free recovery test of selection-protocol §3.5 (re-scoring the patched scene with a
pseudo-closed-loop PDM-style score / collision margin rather than trajectory L2). That test was
not performed here, so the C verdict is restricted to "localization holds" and excludes
"optimal repair site". (ii) Prior work already established that causal-layer strength **does not
predict** PDM degradation magnitude (correlation ≈ 0): a stable site does not imply a predictable
consequence, so $C_m$ must not be used to infer how severe a degradation will be.
(iii) n = 12 and the sample is the top-12 most degraded scenes, hence selective; the readout
characterizes concentration **conditional on degradation**, not a whole-set average.

**4. Three-state verdict.** **PASS** (primary CI entirely above the diffuse baseline). The verdict
applies only to the clipped protocol; the unclipped protocol is indeterminate, and both are
reported side by side.

---

## Record of self-correction

1. **The clip-at-zero rule for negative recovery was registered before the analysis, and proved
   to be necessary rather than cosmetic**: unclipped, $C_m$ = 0.347 with 95% CI [−0.364, 0.962],
   because the denominator $\sum_L \mathrm{recovery}$ can cross zero and the ratio loses meaning.
   Both versions are stored; neither was selected on the basis of its value.
2. **The t-SNE colouring deviates from the literal wording of the work order.** Work-order §5
   specifies colouring by "failure vs normal". This project has no per-sample failure/normal label
   (the PDM-style total is discretized by binary gates and 46% of scenes tie across the two
   inputs), so the colouring instead uses **exactly the reference/degraded definition used by the
   patching experiment** (sim = reference, real = degraded), with open circles additionally
   marking the 12 scenes that entered the causal patching set. This is a substitution of protocol,
   not a fabrication of labels, and is registered here.
3. **No comparison arm was run for SimLingo** (work-order §5 step 3 marks this as optional).
   Reason: there is no equivalent domain-paired activation-patching output on the SimLingo side,
   and producing one would require rebuilding the entire patching pipeline, exceeding the
   "reuse-first, low-cost" scope of this task. The C-axis readout is therefore a
   **single-model readout** and no cross-model ranking is drawn from it.

---

## One-sentence update to the causal-chain picture

> The C-axis readout moves the failure located by G→F→I **from "it exists" to "it has a price"**:
> in DiffusionDrive the degradation induced by the rendering-domain shift is causally concentrated in
> a single deep encoder layer ($C_m$ = 0.858 [0.771, 0.939] against a diffuse baseline of 0.250),
> placing this disconnect in the "repairable with LoRA on two or three layers" bracket — information
> no public leaderboard score provides.
