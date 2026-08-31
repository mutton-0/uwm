# Candidate Expansion: G/F/C-axis Readouts for LTF (Latent TransFuser)

> Work order: [`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md).
> All judgement calls of this round are recorded in [`amendments.md`](amendments.md) §CE (A27–A33).
> Numerical artifacts: `g_axis_ltf.json`, `f_axis_action_counterfactual.json`,
> `f_axis_dd_steer_ltf{,_brake}.json`, `c_axis_hazard_ltf.json`, `v_brake_ltf.npz`;
> adapter: `ltf_g1_adapter/`.

---

## Methods

### Candidate and weights

LTF = **Latent TransFuser**: the TransFuser architecture with the real lidar branch replaced by a
learnable BEV latent (this repository's `transfuser_agent.yaml` hard-codes `latent: True`). The
weights are the official SimScale `ltf_sim_navtest.ckpt` (224 MB, obtainable via
`ckpt/download_ckpts.sh`).

**Relation to DiffusionDrive (the most valuable structural fact of this round)**: the two **share the
identical `TransfuserBackbone(latent=True)` encoder** and differ only in the action head — LTF uses a
**continuous regression head** (`TrajectoryHead`: a single trajectory query through an MLP), whereas
DiffusionDrive uses an **anchored diffusion head** (20 k-means trajectory anchors). LTF therefore
constitutes a **same-encoder, different-head controlled comparison** (Discussion §3).

### Stimuli and adapter

**Exactly the same** set as the existing candidates: the nuScenes ghost-probe G1 corpus plus the N1
negative system (A 291 / D2a 283 / D2b 311 / D2bV 78 / D2c 343 / **D2cV 212**), group sizes matching
DiffusionDrive cell for cell. The adapter `ltf_g1_adapter/ltf_adapter.py` **directly inherits** the
entire front end of `diffusiondrive_g1_adapter` (4:1 sky-removing crop → resize(2048, 512);
clean-frame-anchored ego state; bbox → 8×32 image-token grid; five poolings; behavioural quantity
`commanded_speed = ‖traj[0]‖/0.5 s`) and replaces only the agent construction, so the cross-model
protocol is **field-for-field identical** rather than "a similar one written again".

### Operationalization of the three axes (item-by-item identical to the existing candidates)

* **G**: a per-layer linear discriminant direction is fitted from
  $\delta = Z(\text{ghost}) - Z(\text{clean})$ on the fold's $S_{dir}$; the peak layer is chosen by
  AUC on the fold's $S_{sel}$; results are reported by scene-level 4-fold CV. The primary readout is
  **CV-AUC(A vs D2a) minus that model's own D2cV falsification floor** (both negatives projected with
  the same direction at the same peak layer), reported alongside a random-direction floor, a
  label-permutation null, and **stability across 10 CV fold-assignment seeds**.
* **F① (architecture-neutral)**: the action-level counterfactual test. Manipulation ① of hazard
  presence = ghost vs clean of class A; manipulation ② of the geometric confound = ghost vs clean of
  D2a; $b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$; primary readout b-AUC(A vs D2a) with
  scene-level bootstrap.
* **F② (architecture-dependent)**: injection $Z' = Z + \alpha\sigma_L\hat v$ into the image-token
  block at the concept peak layer, ±α ∈ {0.5, 1, 2, 4}, primary readout the whole-ladder slope;
  controls are a same-layer 20-seed random null, specificity, termination/recovery, plus an
  **in-house upper-bound calibration** $v_{brake}^{ltf}$ (labelled by LTF's own `commanded_speed`
  residualized on ego speed, per-layer discriminant, peak held-out AUC 0.684 at L5).
* **C-hazard**: a **clean↔ghost paired real-input swap** within the same G1 event, replacing all 320
  fused tokens layer by layer, with
  $\mathrm{recovery}(L) = (v_{patch} - v_{ghost})/(v_{clean} - v_{ghost})$, on the 12 class-A events
  with the largest $|v_{clean} - v_{ghost}|$, accompanied by a recovery-profile shape diagnostic.

---

## Results

**Table 1. G-axis readout for LTF on the shared G1 stimulus set, against its own D2cV falsification floor. Group sizes are identical to those used for DiffusionDrive.**

| Pooling | Negative class | n (pos/neg) | CV-AUC | 95% CI | $p$ | primary − D2cV floor | 95% CI (scene bootstrap) | difference over 10 seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean` (primary) | D2a | 291 / 283 | 0.623 | [0.577, 0.668] | 3.76 × 10⁻⁷ | **+0.070** | **[+0.017, +0.126]** | **+0.056 ± 0.028** |
| `vision_mean` | D2cV (falsification floor) | 291 / 212 | 0.553 | [0.502, 0.603] | 0.043 | — | — | — |
| `region_mean` (sensitivity) | D2a | 291 / 283 | 0.613 | [0.567, 0.658] | 3.00 × 10⁻⁶ | **+0.085** | **[+0.031, +0.140]** | **+0.082 ± 0.013** |
| `region_mean` | D2cV | 291 / 212 | 0.528 | [0.477, 0.579] | 0.281 | — | — | — |

Random-direction floor 0.516 ± 0.028; label-permutation null 0.529 ± 0.023 (`vision_mean`); peak
layer $L^\*$ = 6. Across 10 fold-assignment seeds the difference ranges over **[+0.017, +0.105]**
(`vision_mean`) and **[+0.061, +0.102]** (`region_mean`) — **positive in 10 of 10 seeds**.
Geometric robustness: $\rho$(projection, log area) = +0.049 ($p$ = 0.240) and
$\rho$(projection, eccentricity) = −0.007 ($p$ = 0.871), **neither significant** — this readout is
not reading "large and central".

**Table 2. F-axis readouts: architecture-neutral action-level counterfactual (F①) and representation injection (F②).**

| Readout | Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| F① | $b$(A), action change induced by hazard frames [m/s] | **+0.0203** | [+0.0064, +0.0330] | — | significantly ≠ 0 |
| F① | $b$(D2a), action change induced by the geometric confound [m/s] | −0.0156 | [−0.0397, +0.0039] | — | indistinguishable from 0 |
| F① | **b-AUC(A vs D2a)** | **0.583** | **[0.519, 0.639]** | 5.85 × 10⁻⁴ | **PASS** |
| F① | b-AUC(A vs D2cV) | 0.555 | [0.487, 0.614] | 0.035 | marginal |
| F② | ±α full-ladder slope for $v_{hazard}^{ltf}$@L6 [m/s per σ] | +0.00040 | [−0.00065, +0.00143] | — | does not exceed the same-layer null (+0.00019 ± 0.00088, 20 seeds, $z$ = +0.24) |
| F② | **in-house upper bound $v_{brake}^{ltf}$@L5** | **−0.00004** | [−0.00011, +0.00001] | — | **does not exceed the null** ($z$ = −0.76) |
| F② | specificity \|Δlateral\|/\|Δv\| at α = +4 | 43.9 | — | — | the effect is almost entirely lateral |

Over the same scenes, genuine-hazard-induced deceleration is +0.0452 m/s [+0.0077, +0.0754]
(significantly ≠ 0), so the denominator of the pathway-utilization $F_m$ is usable; but the numerator
(injection peak +0.0002) is indistinguishable from zero, so **$F_m$ admits no meaningful estimate**.

**Table 3. C-hazard readout: where the hazard-induced behavioural change enters the network (G1 clean↔ghost paired input swap, 14 most-degraded class-A events (events whose run-time denominator falls below 0.05 are dropped)).**

| Quantity | Value | 95% CI | Reference |
| --- | --- | --- | --- |
| patch-ALL recovery (sufficient-cut-set check) | **+1.000** | — | ≈ 1 means the 8 fused-token blocks form a sufficient cut set |
| $C_m$ = top-2 layer share of recovery | **0.789** | [0.725, 0.857] | diffuse baseline 0.250 |
| recovery-profile Spearman(layer, recovery) | +0.929 | — | > −0.3 ⇒ interior peak, top-2 formula applicable |
| responsible-layer mode / normalized entropy | **L6** / 0.576 | — | — |

Verdict: **PASS** (the $C_m$ CI lies entirely above the diffuse baseline and the profile is
interior-peaked, so the formula's premise holds).

> **Instrument side.** LTF and DiffusionDrive have identical group sizes cell for cell
> (A 291 / D2a 283 / D2cV 212 …) and the adapter inherits the same front end, so their G/F①/C
> readouts are strictly same-protocol. The positive G-axis result is consistent **across both
> poolings and all 10 fold-assignment seeds**, and neither geometric-robustness correlation is
> significant, which excludes the "reading large-and-central" explanation. On the C axis,
> patch-ALL = +1.000 passes the sufficient-cut-set check. **The F② null has been attributed to the
> instrument by the in-house upper-bound calibration**: even $v_{brake}^{ltf}$ — effective by
> construction, held-out AUC 0.684 — cannot move the longitudinal output.
> **Specimen side.** LTF is the **only candidate so far in this line of work whose G axis clears its
> own D2cV falsification floor while simultaneously passing the F① action-level counterfactual**. Its
> hazard signal is readable in the deep encoder (L6) and does drive the action (b(A) significantly
> positive, b(D2a) not), yet **representation-level injection cannot move its longitudinal output**.

---

## Discussion

**1. LTF is this work's first positive G-axis result, and it changes how every earlier
"indeterminate" should be read.** In the previous three rounds SimLingo, DiffusionDrive and
DiffusionDriveV2 all had primary-minus-floor CIs straddling zero, and the instrument-side explanation
("the whole G-axis protocol cannot read *present* in any model") could not be excluded — this is
precisely the question the T-G positive-calibration experiment set out to settle and failed to. Under
**exactly the same stimuli, negative system and statistical protocol**, LTF gives +0.070
[+0.017, +0.126] (primary pooling) and +0.085 [+0.031, +0.140] (sensitivity pooling), positive in
10/10 seeds. **This is the missing positive calibration**: the G-axis protocol *can* read "present",
so the other candidates' "indeterminate" verdicts can now be attributed to specimen properties with
considerably more confidence rather than to an instrument defect.

**2. LTF is simultaneously the first F① positive, and the two axes agree on the same candidate.**
b(A) = +0.0203 [+0.0064, +0.0330] is significantly positive while b(D2a) is not, and
b-AUC = 0.583 [0.519, 0.639] (bootstrap P(AUC ≤ 0.5) = 0.005). LTF therefore not only "sees" the
hazard but the thing it sees does drive the action — the first instance in this work of two
consecutive links (G→F) of the causal chain being connected at once.

**3. The same-encoder controlled comparison overturns one of last round's corollaries (§CE/A31).**
Last round attributed DiffusionDrive's F② unmeasurability to its anchored diffusion head and wrote
into the paper body that "the F axis is not comparable across **action-head** families". LTF shares
the same encoder and has a continuous regression head, so by that corollary it should be measurable —
**it measurably is not** ($v_{hazard}^{ltf}$ slope +0.00040; the in-house upper bound
$v_{brake}^{ltf}$ only −0.00004; neither exceeds its null). The correct attribution is therefore:
**the measurability of injection is determined by the encoder / injection site, not by the action
head.** SimLingo (ViT + LLM residual stream, injected at the driving-query positions) is measurable
and is independently reproduced by an analytic Jacobian; TransFuser-family encoders are not, under
either action head. A plausible but **unverified in this round** mechanistic hypothesis is that
σ-scale token-level perturbations are diluted by `_bev_downscale` + BEV upsampling + the transformer
decoder's cross-attention.

**4. C-hazard places the hazard signal's entry at L6, agreeing with DiffusionDrive.** The two models
sharing an encoder both localize responsibility at **L6** under the same pairing ($C_m$ = 0.801 and 0.789
respectively), and DiffusionDrive's **C-domain** (sim↔real pairing) responsible layer is also L6. In other
words, **the same layer is both where the hazard signal enters and where the rendering-domain failure
enters** — for this encoder family L6 is an information convergence point, which is directly usable
when choosing a site for targeted repair.

**5. What this readout may not claim.** (i) The F② null is **not** evidence that LTF's pathway is
broken — F① already shows its action does vary with hazard; the null shows only that **this injection
operationalization has no resolving power on this encoder**. (ii) The positive G-axis result was
obtained on the nuScenes ghost-probe corpus while LTF's native training domain is NAVSIM; it supports
no inference about its NAVSIM performance. (iii) C-hazard rests on the most-degraded class-A events (k = 14, after the run-time denominator guard) and therefore characterizes concentration **conditional on degradation**, not a whole-set
average.

---

## Record of self-correction

1. **Correction of the F② attribution (§CE/A31).** Last round attributed DiffusionDrive's injection
   unmeasurability to the action head; this round's same-encoder controlled comparison refutes that
   attribution. The corollary sentence in paper §4.4.2 has been rewritten to "not comparable across
   **encoder** families".
2. **Correction of the C-hazard patch scope (§CE/A32).** The first version patched only the first 256
   image tokens and failed the patch-ALL sufficient-cut-set check (LTF −0.030). Patching all 320
   fused tokens gives patch-ALL = +1.000. The first version's number ($C_m$ = 0.989) is **void**.
3. **TransFuser is not listed separately (§CE/A28).** This repository's `transfuser_agent.yaml` *is*
   the LTF configuration (`latent: True`), the official checkpoint list contains only
   `ltf_sim_navtest.ckpt` for the TransFuser family, and the present stimulus set has no lidar, so a
   lidar-equipped TransFuser is not executable under these conditions. LTF therefore covers the
   TransFuser family, stated explicitly rather than skipped silently.
