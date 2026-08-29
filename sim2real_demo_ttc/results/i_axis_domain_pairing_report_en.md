# T-I: Domain Invariance (I axis) — Activation Divergence, Interference Angle and Behavioural Domain Sensitivity for Two Models on One Pairing

> Axis letters follow the G/F/I/C/D naming finalized on 2026-08-29 (mapping in
> [`axis_naming_alignment.md`](axis_naming_alignment.md)). Pre-registration and the registered
> design deviation are in [`amendments.md`](amendments.md) §FA.0 / §FA.1 deviation 1.
> Numerical artifacts: `i_axis_domain.json`, `v_domain.npy` (SimLingo), `v_domain_dd.npz`
> (DiffusionDrive).

---

## Motivation

The I axis (Invariance) does not ask whether a model performs well in some domain; it asks
whether the **grounding-to-action causal mechanism validated by G and F survives an environment
switch** (theoretical lineage: Invariant Risk Minimization, Arjovsky et al. 2019). It sits after
G and F in the chain: even when a model has seen the right thing and its action is genuinely
driven by what it saw, that chain may still be an artefact of one particular rendering style.

This link bears most directly on the core contribution. Public leaderboards (nuScenes / NAVSIM
open-loop scores) are computed within a **single** domain and therefore cannot, by construction,
observe whether the mechanism persists across domains. If the I-axis readout orders two models
differently from the leaderboard, then "leaderboard rank ≠ deployment rank" acquires a
**measurable explanation pointing at a specific link**, rather than remaining a generic suspicion.

This experiment also closes **Stage B6** of the direction-discovery plan: $v_{domain}$ had never
been extracted, yet the I-axis interference angle $\theta_L = \cos(v_{domain}, v_{hazard})$
depends on it — without this step the I axis simply cannot be computed, in either the open-loop or
the closed-loop formulation.

---

## Method

**Domain pairing.** No 3DGS reconstruction assets exist in this repository (verified by an
exhaustive search; `/data/zihao/HUGSIM` contains renderer code only, no scenes of this project).
Per registered deviation 1 we therefore use the existing **same-geometry dual-rendering pairing**
of `ghosthead_v1`:

| Side | Source | Meaning |
| --- | --- | --- |
| sim | `renders/<scene>/frames.mp4` | CARLA engine rendering |
| real | `ghosthead_result/<scene>/seg1p0/<scene>_seg1p0.mp4` | world-model photorealistic re-rendering |

Scene, framing, actors and ego ground truth are identical; rendering style is the sole variable.
Causally this is equivalent to pairing P1 ("real ↔ 3dgs") of selection-protocol §2 — both are
do(appearance). The raw frames are 1600 × 900, the same resolution as nuScenes, so both models run
through **their own native preprocessing** and neither is distorted to accommodate the other.
Usable pairs: **72 scene-variants × 3 timestamps = 216 pairs**. Ego speed is identical on both
sides (same scene, same frame index).

**Readouts (selection-protocol §3③).**

$$D_L=\frac{\mathbb{E}_i\lVert Z_L(x_{real}^{(i)})-Z_L(x_{sim}^{(i)})\rVert_2}{\mathbb{E}_i\lVert Z_L(x^{(i)})\rVert_2},\qquad
\theta_L=\cos\big(v_{domain}(L),\,v_{hazard}(L)\big),\qquad I_m=1-D_{L^*}$$

$v_{domain}(L)$ is PC1 of the paired differences $\delta_i = Z_L(x_{real})-Z_L(x_{sim})$, computed
without mean-centering (amendment A5: the mean direction of $\delta$ *is* the domain direction),
with sign calibrated so that the real side projects higher. $L^*$ is each model's **own concept
peak layer** (SimLingo: $L^*$ = 9 from `v_hazard_clean`; DiffusionDrive: $L^*$ = 5 frozen in T-G),
i.e. protocol §4 rule 3, "align layers by role, not by index".

**Readable positions.** SimLingo: 24 decoder layers, primary pooling `vision_mean`
(`query_mean` / `last_token` also reported). DiffusionDrive: 8 TransFuser encoder `SelfAttention`
modules, primary pooling `vision_mean` (mean over the 256 image tokens).

**Cross-model comparability discipline (protocol §4).** $D_L$ is taken in **within-model
normalized ratio** form, which is the only reason it may be compared across models; on the
behavioural side we additionally report the **within-model dimensionless** domain sensitivity
$\mathbb{E}\,|v_{cmd}^{real}-v_{cmd}^{sim}| / \overline{|v_{cmd}|}$. Bootstrap resampling uses the
**scene** as unit (the 3 timestamps of one scene are not independent), 1000 resamples.

---

## Results

**Table 1. Domain-pair activation divergence, interference angle and behavioural domain sensitivity for two models evaluated on one and the same CARLA-render ↔ world-model-render pairing (216 pairs from 72 scene-variants).**

| Model | Native domain | Readable layers | $L^*$ | $D_{L^*}$ | 95% CI | $I_m = 1-D_{L^*}$ | $\lvert\theta_{L^*}\rvert$ vs own $v_{hazard}$ | Behavioural domain sensitivity | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA (sim) | 24 decoder layers | 9 | 0.394 | [0.374, 0.419] | **0.606** | 0.0258 | 0.346 | [0.249, 0.454] |
| DiffusionDrive | NAVSIM (real)† | 8 encoder self-attn | 5 | 0.797 | [0.756, 0.840] | **0.203** | 0.0013 | 0.115 | [0.080, 0.153] |

† DiffusionDrive's training domain is taken from the checkpoint name and the operator's account and
is **not independently verified** (prior FINDINGS already flags this). It is recorded here as
conditional information; every interpretation that depends on "which side is in-domain" is
conditional on it.

**Table 2. Layer profile of $D_L$ (primary pooling, `vision_mean`).**

| SimLingo layer | 0 | 4 | 8 | **9** | 12 | 16 | 20 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $D_L$ | 0.372 | 0.371 | 0.393 | **0.394** | 0.394 | 0.424 | 0.368 | 0.303 | 0.495 |

| DiffusionDrive layer | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $D_L$ | 0.033 | 0.001 | 0.667 | 0.056 | 0.633 | **0.797** | 0.730 | 0.956 |

SimLingo's $D_L$ profile is nearly flat (0.303–0.495; range 0.19 across all 24 layers), whereas
DiffusionDrive's oscillates over two orders of magnitude (0.001–0.956). Its $D_L$ = 0.001 at L1 is
accompanied by a PC1 explained-variance ratio EVR₁ = 0.989: that layer is almost entirely
dominated by the **input-independent learnable BEV latent**, so the domain difference is
structurally suppressed to zero there.

**Table 3. Interference angle $\lvert\theta_L\rvert = \lvert\cos(v_{domain}, v_{\cdot})\rvert$ at the concept peak layer, against the isotropic reference $1/\sqrt{d}$.**

| Model | Comparison direction | $\lvert\theta_{L^*}\rvert$ | Isotropic reference $1/\sqrt{d}$ | Exceeds? |
| --- | --- | --- | --- | --- |
| SimLingo ($d$ = 896) | $v_{hazard}^{clean}$ (G axis) | 0.0258 | 0.0334 | no |
| SimLingo | $v_{hazard}^{carla}$ (G axis, CARLA in-domain) | 0.0058 | 0.0334 | no |
| SimLingo | $v_{brake}$ (F-axis behavioural anchor) | 0.0044 | 0.0334 | no |
| DiffusionDrive ($d$ = 256 @ L5) | $v_{hazard}^{dd}$ (G axis) | 0.0013 | 0.0625 | no |

> **Instrument side.** The pairing is do(appearance) with geometry, actors and ego ground truth all
> locked; both $D_L$ and the behavioural sensitivity are within-model normalized ratios, satisfying
> protocol §4 rules 1 and 2. The two models' scene-level bootstrap CIs for $D_{L^*}$ do not
> overlap, so the ordering is statistically stable. **However the profile shapes differ radically
> (SimLingo nearly flat vs DiffusionDrive oscillating over two orders of magnitude, with L1
> dominated by a constant latent). Protocol §4 rule 4 requires that candidates with large
> architectural differences be "listed separately rather than force-ranked in one table"**, so the
> cross-model ordering here is recorded as a **conditional conclusion** and is not used as a
> selection criterion.
> **Specimen side.** On one and the same rendering-domain pairing, SimLingo's **representations**
> are less affected by the rendering style ($I_m$ 0.606 vs 0.203), yet its **behaviour** is more
> affected (domain sensitivity 0.346 vs 0.115); the two orderings **run in opposite directions**.
> For both models $v_{domain}$ is close to orthogonal to their own $v_{hazard}$
> (all $\lvert\theta\rvert$ below the corresponding isotropic reference), i.e. the domain drift does
> not geometrically erode the hazard readout direction.

---

## Discussion

**1. The strongest result of this experiment is a dissociation, not a ranking.**
The representational and behavioural readouts order the two models **oppositely**: SimLingo is
more stable in representation ($D_{L^*}$ 0.394 vs 0.797) but less stable in behaviour
(0.346 vs 0.115, non-overlapping CIs). This directly substantiates transferable Tip 3 of
selection-protocol §3.5: **a domain-invariance judgement cannot stop at representational
similarity; one must verify that the same causal mechanism produces the same effect size in both
domains** — "representation stable" and "causal coupling stable" are two independently testable
propositions. Reporting only $D_L$ would have concluded that SimLingo has the better domain
invariance; reporting only the behavioural side would have concluded exactly the opposite. The
four-axis framework requires both precisely so that neither single-sided readout can run away with
the conclusion.

**2. Answer to the core hypothesis of work-order §4: indeterminate.** The work order asks whether
each model's failure corresponds systematically to **the gap between that model's own native
training domain and the rendering style**. This experiment cannot answer it, for two
specimen-side (not instrument-side) reasons: (i) DiffusionDrive's native training domain is
**not independently verified**, so the antecedent — whether its training domain is nearer to or
farther from world-model photorealistic rendering — has no determined value; (ii) the
representational and behavioural orderings conflict, so even with a determined antecedent there is
no single "degree of failure" to compare it against. Three-state verdict: **indeterminate**
(insufficient premise/power, not "the correspondence does not exist"). What can be reported with
certainty is the weaker but clean statement that **$I_m$ is computable and the two models'
$D_{L^*}$ do differ statistically**, and that the required input $v_{domain}$ has now been
extracted and stored (Stage B6 gap closed).

**3. All interference angles fall below the isotropic reference, and this negative result is
informative.** Protocol §3③ treats $\lvert\theta_{L^*}\rvert$ as a penalty term: a large
interference angle means domain drift continuously erodes the hazard readout (a geometrized causal
confusion). All four measurements fall below the $1/\sqrt{d}$ reference for their dimension,
showing that **the domain direction is approximately orthogonal to the hazard direction in both
models** — the observed domain sensitivity does not operate by contaminating the hazard direction.
The implication for post-training is that domain-suppression treatments (e.g. projecting out
$v_{domain}$) should be expected neither to damage nor to repair the hazard readout as a
side effect.

**4. Cross-corroboration with the C axis.** DiffusionDrive's $D_L$ reaches 0.73–0.96 at L5–L7,
while the C axis locates the causal responsibility peak at L6 with the deep block L4–L6 covering
8/12 scenes; two independent readouts both point at the deep fusion stage of the encoder. They are
nevertheless not equivalent: $D_L$ is maximal at L7 (0.956) whereas the causal peak is L6,
consistent with the "divergence peak L7, causal peak L6" dissociation reported on the C axis.
**The layer with the largest domain divergence is not the repair site** — reproduced here for the
third time under a third metric.

**5. What this readout may not claim.** (i) The pairing uses **world-model photorealistic
re-rendering**, not 3DGS reconstruction, so the scope of the conclusion is written as
"CARLA rendering ↔ world-model photorealistic rendering"; the definitive conclusion awaits the
real↔3DGS version in the Madison phase. (ii) The cross-model $D_L$ ordering is conditional (see
instrument side). (iii) SimLingo's $\theta$ is computed against a $v_{hazard}^{clean}$ that is
**known to sit on its falsification floor**, so "small interference angle" is less informative for
SimLingo than for DiffusionDrive, whose $v_{hazard}^{dd}$ at least clears its permutation floor
decisively.

---

## Record of self-correction

1. **The 3DGS counterfactual scene set (P3) does not exist in this repository and was replaced by
   the same-geometry dual-rendering pairing**; the cost and scope of that substitution are stated
   in Results and Discussion §5 and registered in `amendments.md` §FA.1 deviation 1. This is a
   substitution of design, not a conflation: the causal structure (do(appearance), locked
   geometry) is identical while the rendering source differs.
2. **The work order's "intuitive distance of the 3DGS rendering style from each native domain" was
   not used as a presupposed direction.** The work order itself states "do not presuppose a
   direction, let the data speak"; accordingly this report records the item as **indeterminate**
   rather than picking a direction to fit the data.
3. **DiffusionDrive's $D_L$ = 0.001 at L1 is a structural pseudo-zero** and is annotated as such in
   Results (EVR₁ = 0.989; the layer is dominated by an input-independent learnable BEV latent).
   That layer was not used as $L^*$ and entered no verdict.
4. **The cross-model ordering was reported as a conditional conclusion rather than being declared
   incomparable per protocol §4 rule 4.** Reason: work-order §4 step 4 explicitly requires the
   "$I_m$ ranking of the two models". The compromise is to give the ranking together with the
   reasons it is not fully comparable and the resulting restrictions. This is a deliberate
   deviation from protocol discipline and is registered here.

---

## One-sentence update to the causal-chain picture

> The I-axis readout **revises** a default assumption in the causal-chain picture: cross-domain
> stability is not a scalar — on one and the same domain pairing, the representational and
> behavioural sides order the two models oppositely ($I_m$ 0.606 vs 0.203; behavioural domain
> sensitivity 0.346 vs 0.115, both with non-overlapping CIs) — so any claim that a model is "more
> domain-invariant" must specify which side is meant, or it is simultaneously true and false.
