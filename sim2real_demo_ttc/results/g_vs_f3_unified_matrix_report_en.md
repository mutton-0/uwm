# G-VS / F-3 unified six-candidate matrix (v1 primary readouts)

> Work order: [`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md), **core deliverable**.
> Feasibility and task definition: [`g_vs_feasibility_report_en.md`](g_vs_feasibility_report_en.md).
> Per-candidate numbers: `g_vs_axis_*_en.md`, `f3_occlusion_axis_*_en.md`.
> Autonomous decisions: [`amendments.md`](amendments.md) §GF/A49–A52.

---

## 0. Why the old G/F were replaced (motivation, not a feature request)

Repeated rounds across three corpora (G1 / lead braking / NAVSIM) produced the same pattern:
**G/F readouts built around the semantic construct "hazard" inherently depend on the mining
pipeline's human criteria for "what counts as dangerous", and are consequently unstable across both
scenarios and data sources.**

| Old readout | G1 | New scenario (lead braking) | New data source (NAVSIM) |
| --- | --- | --- | --- |
| G (primary − D2cV floor) | LTF **+0.070 PASS**, rest indeterminate | all indeterminate | **all indeterminate, LTF included** (comparable power) |
| F① (b-AUC) | LTF, DDv2 **PASS** | all indeterminate | only SimLingo replicates digit for digit; both PASSes fell |

G-VS and F-3 are designed to replace both with tests that have **objective GT / need no semantic
criteria**:

* **G-VS**: does the representation correspond to real scene structure (SAM segmentation pseudo-GT)?
  A **verifiable perceptual-correctness** question that never involves the subjective judgement
  "dangerous";
* **F-3**: remove a genuinely present key entity — does the action disappear with it? An **input-level
  causal intervention** that needs no matched negatives.

Literature anchors: Embodied Interpretability (arXiv 2605.00321) and its blind action / spurious
correlation / post-hoc rationalization trichotomy (this round covers only the necessity test
corresponding to blind action); Causal Imitative Model (Samsami et al. 2021, arXiv:2112.03908), its
causal-inversion framework and the inertia / collision failure signatures; Hewitt & Liang 2019 for
probe selectivity discipline.

---

## 1. The unified matrix (six candidates, no grouping)

**Table 1. G-VS / F-3 readouts for all six candidates. Neither new axis distinguishes single- from multi-frame, or language from no language — this is the key simplification over the old G/F.**

| Policy | G-VS: selectivity (primary) | G-VS: mIoU trained / rand / pos | G-VS verdict | F-3: necessity ratio $R$ | F-3: $R_{ctrl}$ | F-3 verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **SimLingo** | **+0.0273 [+0.0128, +0.0424]** | 0.396 / 0.369 / 0.356 | **PASS** | **+0.461 [+0.166, +0.789]** | +0.155 [−0.048, +0.399] | indeterminate (CI spans 0.5) |
| **DiffusionDrive** | **+0.0381 [+0.0211, +0.0551]** | 0.382 / 0.344 / 0.333 | **PASS** | no baseline response | +0.026 [−0.130, +0.151] | indeterminate¹ |
| LTF | +0.0191 [−0.0006, +0.0385] | 0.367 / 0.348 / 0.333 | indeterminate | **+0.113 [+0.044, +0.191]** | +0.029 [−0.031, +0.081] | **FAIL**² |
| DiffusionDriveV2 | +0.0159 [−0.0029, +0.0353] | 0.347 / 0.332 / 0.333 | indeterminate | no baseline response | +0.074 [−0.254, +0.413] | indeterminate¹ |
| Alpamayo-R1 | **n.m.**³ | — | — | no baseline response | −0.086 [−0.296, +0.017] | indeterminate¹ |
| AutoVLA | **n.m.**³ | — | — | no baseline response | +0.055 [−0.098, +0.213] | indeterminate¹ |

> ¹ **Indeterminate: the baseline response itself is indistinguishable from 0** ⇒ there is no response
>   available to test for necessity. **This is not a failed necessity test; it is an untestable one**
>   (§GF/A52) — the two imply entirely different repairs, see §3.
> ² **FAIL**: $R$'s CI upper bound of 0.191 < 0.5. Occluding the key entity removes only **11%** of the
>   response, while an equal-sized grey patch elsewhere removes 3% (CI spans 0). That is: the action
>   does respond to hazard frames, but **that response is not driven by this entity's visibility**.
> ³ **n.m. (not measured, not "not applicable")**: the two VLAs' video-token layouts. AutoVLA's layout
>   was in fact solved this round (3 cameras × `[2,18,32]`, 2×2 merged ⇒ the front camera's last
>   temporal group gives 9×16 = 144 tokens uniformly covering the front image), but the random-init
>   control arm's implementation is unresolved; Alpamayo's 6-camera grouping was not verified. **We do
>   not fill the six cells by "averaging tokens"** — such an mIoU would be uninterpretable, worse than
>   a blank (§GF/A50).

**F-3 covers 6/6; G-VS covers 4/6.**

---

## 2. G-VS: two candidates pass, but every absolute mIoU is low

**The two that pass**: DiffusionDrive (selectivity +0.038) and SimLingo (+0.027); both have
scene-level paired CIs excluding 0, i.e. **their representations do carry linearly readable
object-ness information**. LTF (+0.019) and DiffusionDriveV2 (+0.016) sit on the 0 line and adjudicate
indeterminate.

**But a limitation must be reported alongside**: all four candidates' trained mIoU falls in
**0.35–0.40**, while the `position_only` floor (token coordinates alone) already reaches **0.333**.
That is, **most of the absolute mIoU comes from the spatial prior; the representation's net
contribution is only 0.02–0.04.**

That net contribution is nonetheless credible because it is a **paired difference**: same images, same
layer, same grid, same probe, with only "were the weights trained" varying. But the smallness of the
net contribution must be stated plainly, not glossed over by reporting a significant selectivity.

**The low mIoU has an independent attribution anchor** (feasibility report §1.1): all three
TransFuser-family candidates carry a **BEV semantic segmentation head in their original training
objective**, so their features must contain segmentation information. Given that, a probe recovering
only +0.02–0.04 net points to the **readout** side rather than the **representation** side:

* **the token grid is coarse**: 8×32 for the TransFuser family, so one token covers roughly 50×50 px
  of the original image, and a pedestrian at medium-to-long range often occupies less than one token;
* **the linear probe's capacity floor**: this is deliberate (Hewitt & Liang's discipline), and the
  price is a conservative readout;
* **BEV vs image space**: they were trained for **BEV** segmentation, and image-space object-ness need
  not be preserved in a linearly readable form.

**The correct reading of G-VS is therefore "the representations contain linearly readable object-ness
information, in small quantity", not "these models cannot see the scene".** That distinction is
critical for the next version's design.

---

## 3. F-3: the cleanest result of this round, and the most negative

**None of the six candidates passes the necessity test.** But "not passing" splits into two kinds with
entirely different diagnostic meaning:

### 3.1 LTF: **FAIL** — there is a response, but this entity does not drive it

| Quantity | Value |
| --- | --- |
| $b_{ghost}$ (action change caused by hazard frames) | **−0.0245 [−0.0384, −0.0102], significant** |
| $b_{occ}$ (after occluding the entity) | −0.0170 [−0.0303, −0.0032], **still significant** |
| **Necessity ratio $R$** | **+0.113 [+0.044, +0.191]** ⇒ only 11% removed |
| Control arm $R_{ctrl}$ (patch elsewhere) | +0.029 [−0.031, +0.081], indistinguishable from 0 |

**The ctrl arm is what makes this verdict stand**: painting an equal-sized grey patch elsewhere leaves
the action essentially unchanged (3%); painting it over the hazard entity also leaves it essentially
unchanged (11%). **So "occlusion" as an operation is not ineffective; what is ineffective is occluding
*this entity*.** This is precisely the **blind action** signature of Embodied Interpretability: the
action appears to respond to hazard scenes, but the hazard entity is not what drives it.

What does drive it is not answered this round — that is Future Work's F-2 (spurious correlation /
attention analysis). Causal Imitative Model suggests two candidates: **inertia** (following the ego's
own speed history) and **collision** (following global scene statistics).

### 3.2 The other four: **indeterminate** — there is no response to test at all

DiffusionDrive, DiffusionDriveV2, Alpamayo-R1 and AutoVLA all have $b_{ghost}$ indistinguishable from
0. A necessity test presupposes a response; when the premise fails, **FAIL must not be recorded**.

**Why this distinction is substantive**:
* **FAIL (LTF)**: there is a response but it is wired to the wrong evidence ⇒ the repair is to *find
  the driver and rewire the action onto the correct evidence*;
* **indeterminate (the other four)**: there is no response at all ⇒ the repair is to *first produce a
  response to hazard frames*.

Recording both as "did not pass" would collapse two entirely different prescriptions into one number —
the same reason this paper consistently refuses a composite score.

### 3.3 Every control arm behaves correctly, which is what validates the design

All six candidates' $R_{ctrl}$ fall within **[−0.086, +0.155]** with CIs spanning 0, i.e. "a grey patch
elsewhere" does not return the action to baseline. **This successfully separates the occ arm's two
effects** and is the precondition for the F-3 verdicts being credible (§GF/A51).

---

## 4. The relation between the two new axes: passing G-VS does not imply passing F-3

| Policy | G-VS | F-3 |
| --- | --- | --- |
| SimLingo | **PASS** | indeterminate ($R$ = 0.461, closest to passing) |
| DiffusionDrive | **PASS** | indeterminate (no baseline response) |
| LTF | indeterminate | **FAIL** |
| DiffusionDriveV2 | indeterminate | indeterminate (no baseline response) |

**DiffusionDrive is the clearest cell**: its representation **does** carry linearly readable
object-ness information (the highest selectivity in the table, +0.038), yet its action shows **no
response at all** to hazard frames. **"Seeing" and "acting on it" are two independent things** — the
same conclusion the old G/F system produced repeatedly ("the information is in the representation but
does not drive the action"), **except that this time it comes from an objective-GT perceptual test
plus an input-level causal intervention, with no reliance on any "hazard" criterion.**

That is this round's core increment over the old system: **the same conclusion, on a body of evidence
that does not depend on a semantic construct.**

---

## 5. Limitations

1. **G-VS covers only 4/6** (§GF/A50). AutoVLA's token layout is solved but the implementation is
   stuck; Alpamayo's is unsolved.
2. **G-VS's absolute mIoU is low and its net contribution small** (0.02–0.04), with attribution
   leaning to the readout side (coarse grid + linear probe). The BEV segmentation head provides an
   independent anchor, but **no grid-resolution ablation was run**.
3. **G-VS is binary object-ness, not semantic segmentation.** "Can it tell a pedestrian from a traffic
   cone" was not measured; that is left to G-VL.
4. **F-3's occlusion is grey-fill, not inpainting.** The grey patch is itself out of distribution; the
   ctrl arm controls for "adding a grey patch" but not for "the patch changing global image
   statistics".
5. **The two VLAs' F-3 event counts are capped** (Alpamayo 99, AutoVLA 119, others 282) for cost, and
   Alpamayo is additionally subject to the coverage prefilter (186/288) and sampling noise.
6. **Only the G1 corpus was used.** Per work order §3, replication on lead braking / NAVSIM is a bonus
   and was not attempted this round.
