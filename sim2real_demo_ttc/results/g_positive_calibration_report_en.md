# T-G: Positive Calibration of the G Axis — Semantic Grounding of DiffusionDrive and SimLingo under Identical Stimuli

> Axis letters follow the G/F/I/C/D naming finalized on 2026-08-29 (mapping in
> [`axis_naming_alignment.md`](axis_naming_alignment.md): **the R axis of earlier files is the G
> axis of this report**, with a narrowed definition; the earlier R①/R②/R③ are this report's
> G-①/G-②/G-③). The method follows
> [`../docs/g_axis_positive_calibration_diffusiondrive.md`](../../docs/g_axis_positive_calibration_diffusiondrive.md)
> in full. Pre-registration: [`amendments.md`](amendments.md) §FA.0; protocol corrections: §FA/A13
> and §DV/A16. Numerical artifacts: `g_positive_calibration_diffusiondrive.json`,
> `v_hazard_dd_{vision_mean,region_mean}.npz`; adapter code: `diffusiondrive_g1_adapter/`.

---

## Motivation

The G-axis readout on SimLingo already yielded a robust negative result with a falsification
control: the primary contrast (class A, VRU emergence, vs D2a, geometry-balanced static objects)
is **indistinguishable** from the pure falsification control D2cV (same VRU class, same imaging
geometry, differing only in a relative velocity that a single-frame model is physically unable to
observe).

That negative admits two mutually exclusive explanations, and the prior evidence could not exclude
either:

- **H-real**: this specimen genuinely lacks visual→hazard-concept grounding (the methodology works;
  the specimen is impaired);
- **H-artifact**: the readout protocol itself (pooling, layer selection, contrastive direction
  extraction) cannot detect grounding in any model, and would hit the floor whatever the specimen
  (instrument failure).

The only way to separate them is to apply **the identical readout protocol** to a model that very
probably *does* have grounding. DiffusionDrive is an end-to-end planner trained on
nuScenes/NAVSIM-family real data whose task definition includes collision avoidance. If it too
reads at the floor under the same yardstick, H-artifact holds and the G-axis method must be
redesigned; if it reads a clear signal, SimLingo's negative is established as a **property of the
specimen** and can be written up as a genuine finding rather than left with a "validity unproven"
caveat.

This is the **only methodological positive-calibration experiment** among the four; it adds not
breadth of evidence but a **lower bound on methodological validity**. It is a precondition for the
core contribution: only once the G-axis yardstick has been shown able to read "present" does a
claim of the form "the leaderboard-vs-deployment disconnect occurs at link G" become meaningful.

---

## Method

**Fairness of comparison (direct application of selection-protocol §4 rule 5).** DiffusionDrive
**must** be run on exactly the same stimuli as SimLingo — this project's own nuScenes ghost-probe
mining corpus (G1) plus the N1 negative system D2a/D2b/D2c/D2cV — and **not** on DiffusionDrive's
own NAVSIM/navhard evaluation set. An input adapter, `diffusiondrive_g1_adapter/`, was written for
this purpose and is an engineering deliverable of this report.

| Item | SimLingo (baseline, pre-existing) | DiffusionDrive (this experiment) |
| --- | --- | --- |
| Native training domain | CARLA (sim) | NAVSIM (real)† |
| Evaluation stimuli | nuScenes G1 + N1 negatives | **the same set**, via the adapter |
| Image protocol | aspect-preserving resize → top crop (CARLA geometry alignment) | centre 4:1 crop removing sky → resize(2048, 512) (pixel-identical to the ghosthead front end) |
| Ego-state protocol | prompt states `Current speed`, **anchored to the clean frame** | `driving_command` straight one-hot + v/a, **anchored to the clean frame** |
| Readable layers | 24 decoder layers | 8 TransFuser encoder `SelfAttention` (320 tokens = 256 image + 64 BEV latent) |
| Primary pooling | `vision_mean` | `vision_mean` (mean over the 256 image tokens; the isomorph of SimLingo's `vision_mean`) |
| Direction extraction | per-layer linear discriminant on A vs D2a within the fold's S_dir | identical procedure |
| Peak-layer selection | AUC argmax on the fold's S_sel | identical rule, DiffusionDrive's own argmax |
| Reporting | scene-level 4-fold CV; every event serves once as held-out | identical procedure |

† DiffusionDrive's training domain is taken from the checkpoint name and the operator's account and
is **not independently verified** (prior FINDINGS already flags this).

**Ego-state anchoring discipline.** Both conditions share the ego speed **anchored to the clean
frame**, so that the only difference between clean and ghost is the image — **isomorphic** to
SimLingo's `prompt_anchor: clean` treatment (the same elimination of the same known confound).

**Protocol for the falsification floor (critical; see §DV/A16).** The direction and peak layer are
fitted/selected **only** on A vs D2a; the remaining negative classes including D2cV are then
projected with **that same direction at that same peak layer**. This is what a falsification floor
means: it measures how much of the A-vs-D2a direction is merely "A vs any VRU". The fold assignment
covers the scenes of every event type in the cache.

**Pre-registered primary readout.** Within one model, **CV-AUC(A vs D2a) − CV-AUC(A vs D2cV)**,
with a 95% CI from **scene-level bootstrap (2000 resamples; both readouts draw the same scene
sample each time so as to preserve the paired structure)**. Reported alongside: random-direction
floor, label-permutation null, and **stability across 10 CV fold-assignment seeds**.

**Decision rule (supplied by the work-order document before any analysis; no "failure" cell).**

| DiffusionDrive readout | SimLingo readout | Verdict |
| --- | --- | --- |
| significantly above its own D2cV floor | at its own D2cV floor | **H1 holds**: the G-axis method works; SimLingo's FAIL is a specimen property |
| also at its own D2cV floor | at its own D2cV floor | **H-artifact holds**: the G-axis readout must be redesigned |
| above the floor but weaker than expected | at the floor | **partially holds**: report effect size, not merely significance |
| CI crosses 0 | — | **indeterminate**: this sample size cannot separate H1 from H-artifact |

---

## Results

**Table 1. G-axis readout under identical stimuli and identical negative design, across two specimens with divergent native domains (primary pooling `vision_mean`; scene-level 4-fold CV).**

| Model | Native domain | Negative class | n (pos/neg) | CV-AUC | 95% CI | $p$ | Difference from own D2cV floor | 95% CI (scene bootstrap) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA (sim) | D2a (primary) | 288 / 283 | 0.515 | [0.467, 0.562] | 0.547 | **+0.035** | [−0.026, +0.097] |
| SimLingo | CARLA (sim) | D2cV (falsification) | 288 / 212 | 0.480 | [0.428, 0.531] | 0.434 | — | — |
| DiffusionDrive | NAVSIM (real) | D2a (primary) | 291 / 283 | 0.559 | [0.512, 0.606] | 0.015 | **+0.009** | [−0.047, +0.065] |
| DiffusionDrive | NAVSIM (real) | D2cV (falsification) | 291 / 212 | 0.550 | [0.500, 0.601] | 0.054 | — | — |

**Table 2. Robustness of the primary contrast across 10 CV fold-assignment seeds, and against permutation / random-direction floors.**

| Model | Pooling | main (mean ± sd) | D2cV floor (mean ± sd) | difference (mean ± sd) | difference range | permutation floor | random-direction floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | `vision_mean` | 0.536 ± 0.023 | 0.534 ± 0.021 | **+0.002 ± 0.024** | [−0.048, +0.035] | 0.494 ± 0.009 | 0.491 ± 0.021 |
| DiffusionDrive | `vision_mean` | 0.547 ± 0.034 | 0.535 ± 0.032 | **+0.012 ± 0.032** | [−0.037, +0.059] | 0.494 ± 0.018 | 0.471 ± 0.013 |
| SimLingo | `region_mean` (sens.) | 0.551 ± 0.031 | 0.534 ± 0.020 | **+0.017 ± 0.026** | [−0.016, +0.075] | 0.491 ± 0.023 | 0.536 ± 0.042 |
| DiffusionDrive | `region_mean` (sens.) | 0.519 ± 0.027 | 0.527 ± 0.025 | **−0.008 ± 0.024** | [−0.054, +0.012] | 0.493 ± 0.030 | 0.475 ± 0.010 |

**Table 3. DiffusionDrive: full negative-class panel under one and the same direction and peak layer ($L^*$ = 5, `vision_mean`).**

| Negative class | Semantics | n_neg | CV-AUC | 95% CI | $p$ |
| --- | --- | --- | --- | --- | --- |
| D2a | class contrast (geometry-balanced static objects) — primary, G-① | 283 | 0.559 | [0.512, 0.606] | 0.015 |
| D2cV | pure falsification (same VRU class + same geometry, velocity only) | 212 | 0.550 | [0.500, 0.601] | 0.054 |
| D2c | falsification (55% vehicles mixed in; class difference visible) | 343 | 0.565 | [0.520, 0.610] | 0.005 |
| D2b | context contrast (outside the corridor, mixed classes) — G-② | 311 | 0.538 | [0.492, 0.584] | 0.109 |
| D2bV | context contrast (VRU only) | 78 | 0.546 | [0.476, 0.617] | 0.211 |

**Geometric robustness (a G-axis readout must show it is not reading "large and central").**
DiffusionDrive, `vision_mean`: $\rho$(projection, log area) = +0.059 ($p$ = 0.160);
$\rho$(projection, eccentricity) = +0.058 ($p$ = 0.166). **Neither is significant**: once the
stimulus set is completed, DiffusionDrive's readout shows no detectable residual geometric
correlation. (Before completion the eccentricity term was +0.151, $p$ = 3.03 × 10⁻⁴; that
significance disappears as the D2cV/D2b negatives are filled in, indicating it arose from an
incomplete negative set rather than from model behaviour.)

> **Instrument side.** The adapter delivered the same nuScenes G1 corpus and N1 negatives to
> DiffusionDrive, and direction extraction, peak-layer selection, CV reporting and the falsification
> floor protocol are **item-by-item isomorphic** across the two sides. Both models' primary readouts
> clear their own permutation and random-direction floors (DiffusionDrive 0.559 vs 0.494/0.471;
> SimLingo 0.515 vs 0.494/0.491), so the pipeline can produce a readable discriminative direction.
> **But the quantity the verdict depends on — "primary minus own D2cV floor" — has a 95% CI
> crossing zero in both models, and its sd across 10 fold-assignment seeds (0.024–0.032) is
> *larger* than the effect itself.** The resolution of this experiment is insufficient to support a
> judgement of "qualitative difference".
> **Specimen side.** Under the same yardstick DiffusionDrive's primary readout (0.559,
> $p$ = 0.015) is higher than SimLingo's (0.515, $p$ = 0.547) and is the only significant primary
> readout among the two models × two poolings of this round; but its D2cV floor is likewise close to
> significance (0.550, $p$ = 0.054), and the gap between them is only +0.009 [−0.047, +0.065].
> **Once the stimulus sets are equalized, no protocol variant supports a "qualitative difference".**

---

## Discussion

**1. Three-state verdict: indeterminate.** Under the decision rule fixed before the analysis,
DiffusionDrive's "primary minus own D2cV floor" = +0.009 with scene-level bootstrap 95% CI
[−0.047, +0.065], crossing zero; across 10 fold-assignment seeds it is +0.012 ± 0.032 — **an sd
larger than the effect itself**. **This experiment can declare neither that the G-axis method works
(H1) nor that it fails (H-artifact).** This is not a null outcome but a quantitatively meaningful
one: even in a specimen that very probably has grounding, and on a stimulus set **identical** to
SimLingo's, **the D2a-over-D2cV gain the current G-axis protocol can read is bounded at roughly
+0.065 AUC**, a magnitude smaller than twice the fold-assignment noise (±0.032).

**2. Once the stimulus sets are equalized, even the directional evidence ceases to be robust.**
The first version (141 D2cV on the DiffusionDrive side against 212 on SimLingo's) appeared to show
"DiffusionDrive exceeds its permutation null by more". With the stimulus sets completed to be
identical on both sides (D2a 283, D2cV 212, D2bV 78), that difference **flips with pooling**: under
`vision_mean`, DiffusionDrive exceeds its permutation floor by +0.065 and SimLingo by +0.021, while
under `region_mean` DiffusionDrive exceeds by +0.024 and SimLingo by +0.056. **The two poolings give
opposite orderings**, so this round **cannot** claim that the yardstick reads something in
DiffusionDrive and nothing in SimLingo. The only robust fact is that DiffusionDrive's D2a readout is
significant under `vision_mean` (0.559, $p$ = 0.015) — the only significant primary readout among
the two models × two poolings — while its D2cV floor is likewise close to significance (0.550,
$p$ = 0.054) and the two are indistinguishable.

**2b. The four protocol variants' differences all overlap, which is the most direct basis for the
"indeterminate" verdict.** Across 10 seeds, "primary minus own D2cV floor" is: SimLingo
`vision_mean` +0.002 ± 0.024; SimLingo `region_mean` +0.017 ± 0.026; DiffusionDrive `vision_mean`
+0.012 ± 0.032; DiffusionDrive `region_mean` −0.008 ± 0.024. Every pair is within one sd of every
other, so **no pair exhibits the "qualitative difference" the decision rule requires**; and every
value is itself indistinguishable from zero.

**3. Primary and sensitivity poolings disagree and must be stated side by side.**
`vision_mean` (the pooling specified by the work order) gives +0.009 [−0.047, +0.065];
`region_mean` gives +0.012 [−0.057, +0.090] in a single run and −0.008 ± 0.024 over 10 seeds. The
two poolings differ appreciably in their **primary** readouts (0.559 vs 0.517), yet both
**differences** are indistinguishable from zero. The bottleneck on resolution is that the floor
tracks the primary readout: 0.559 / 0.550 under `vision_mean` and 0.517 / 0.504 under `region_mean`
— **the floor rises and falls with the primary**, exactly the pattern predicted by the explanation
"what is being read is the presence of any VRU rather than hazard". **The most direct way to
increase resolution is to enlarge the D2cV negative set (currently 212, with a floor sd of 0.032
across 10 seeds), not to change pooling or model.**

**4. Position with respect to the core contribution.** This result draws the framework's boundary
of applicability more honestly: **the G axis can currently distinguish "readable vs unreadable"
(relative to a permutation floor) but cannot yet distinguish "reading the hazard concept vs reading
the mere presence of any VRU" (relative to the D2cV floor).** In the synthesis document, therefore,
G-axis readouts may support an ordering of *perception-side signal strength* only, and may not on
their own support an assertion about whether a model is genuinely grounded in the hazard concept —
that assertion requires a distinguishable D2cV floor, which this round did not achieve.

**5. Residual geometric correlation.** DiffusionDrive's projections correlate weakly but
significantly with target eccentricity ($\rho$ = +0.151, $p$ = 3.03 × 10⁻⁴) and not with area
($\rho$ = +0.045, $p$ = 0.287). N1's geometric matching is a caliper match on (log area,
eccentricity), so this is matching residual rather than matching failure; it nevertheless means
that part of DiffusionDrive's D2a gain may derive from "the target is more elongated" (an intrinsic
pedestrian-vs-static-object difference) that the present negative system cannot further remove.

---

## Record of self-correction

1. **Confirmation of the primary pooling (§FA/A13).** The first version of the script mistakenly
   used `region_mean` as primary at the verdict step. Per the work-order document §3 table, the
   primary pooling must be `vision_mean` (matching SimLingo's primary readout); this was corrected.
   Both poolings were produced and stored in the same run, and the primary status was fixed by the
   document before any analysis.
2. **Correction of the falsification-floor protocol (§DV/A16); this is what changed the verdict
   from "H1 holds" back to "indeterminate".** The first version **refitted a separate direction for
   each negative class**, whereas the project's established protocol (`n1_cv.py`, `n1_report.md`)
   **shares one A-vs-D2a direction**. The former measures "can a direction be found that separates A
   from D2cV", is systematically inflated, and is not comparable to the published numbers. Under the
   shared-direction protocol, DiffusionDrive's difference changes from +0.093 [0.004, 0.174] to
   +0.047 [−0.028, +0.122] (and subsequently to +0.009 [−0.047, +0.065] after item 4 below). **The first version's positive result is void and must not be cited.**
   That positive result had been seen before this correction was registered; the correction was
   nevertheless carried out because the protocol inconsistency is a hard defect (comparability with
   n1_report is the entire point of T-G), and because the correction runs against the conclusion
   already written rather than in its favour.
3. **Two consequential fixes.** (i) The fold assignment previously covered only the POS+NEG scenes,
   so ~70% of D2cV was silently dropped when projecting the shared direction onto other negatives
   (n_neg 212 → 61); it now covers the scenes of all event types. (ii) Fold-assignment drift was
   previously unquantified; the primary-minus-floor quantity is now reported with mean ± sd and
   range across 10 seeds, and the measured sd (0.024–0.027) is the same order as the effect, which
   is itself direct evidence that this experiment lacks resolution.
4. **Correction of unequal stimulus sets (§FA/A19); the final numbers in this report come from this
   version.** The first DiffusionDrive cache was generated with
   `--types A D2a D2b D2c --use-matched` and therefore covered only members of
   `matched_D2c` / `matched_D2b`; but the falsification floor D2cV is defined as
   "`matched_D2cV` ∩ class D2c ∩ VRU", whose members are not all inside `matched_D2c`. As a result
   the DiffusionDrive side received only **141** D2cV negatives against SimLingo's 212, and 33 D2bV
   against 78. **This violates the very premise of T-G** — the work-order document §3 requires
   "exactly the same stimulus set as SimLingo", and the floor is precisely the quantity the verdict
   depends on. Caching the 116 missing events (≈ 40 s of GPU) equalized both sides to D2a = 283,
   D2cV = 212, D2bV = 78, and all readouts were re-run: the primary difference moved from
   +0.047 [−0.028, +0.122] to **+0.009 [−0.047, +0.065]**, and the 10-seed value from
   +0.025 ± 0.025 to **+0.012 ± 0.032**. The verdict remains indeterminate but the evidence is
   weaker, and the previous directional evidence ("DiffusionDrive exceeds its permutation floor by
   more") **flips with pooling** (see Discussion §2) and has been rewritten accordingly.
   One incidental finding: the significant correlation between DiffusionDrive's projections and
   eccentricity before completion ($\rho$ = +0.151, $p$ = 3.03 × 10⁻⁴) disappears afterwards
   (+0.058, $p$ = 0.166), showing that the significance came from an incomplete negative set rather
   than from model behaviour.

5. **The remaining poolings (`last_token`, `region_max`, …) were not run** (work-order §3 table:
   "fill in as budget permits"). Reason: the primary pooling and the first sensitivity pooling
   already localize the bottleneck to the D2cV floor's sample size, and adding poolings does not
   move that bottleneck.

---

## One-sentence update to the causal-chain picture

> The T-G readout **bounds** the usable scope of the chain's first link: on a stimulus set identical
> to SimLingo's, even a model that very probably has grounding cannot be separated from its own D2cV
> falsification floor (+0.009 [−0.047, +0.065]). The framework can therefore currently report only
> "the perception side is / is not readable in this domain", not "the model is / is not grounded in
> the hazard concept" — a restriction that must be stated in the paper, or every downstream G-axis
> claim will be overstated by one grade.
