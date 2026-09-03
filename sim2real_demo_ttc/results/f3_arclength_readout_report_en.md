# F-3 with an Arc-Length Readout: Action as the Trajectory's Own Extent (Independent of the Hazard's Position)

> Work order: 2026-09-03 (second one). Follows
> [`f3_distance_readout_report_en.md`](f3_distance_readout_report_en.md).
> Autonomous decisions: [`amendments.md`](amendments.md) §FE/A65–A67.
> **Pure re-analysis** of the already-stored `f3_occlusion_*_dist.json`; **no new forward inference
> was run.** Artifacts: `f3_arclength_readout.json`; script `scripts/f3_arclength_analysis.py`.
> **No existing result was modified or deleted**; new results carry the `_arclen` suffix.

---

## 0. Direct answer to the work order's question

> **After switching to a readout that does not depend on the entity's position, did any verdict
> change — particularly for the four candidates previously showing "no baseline response"?**

**Not one verdict changed.** Under both definitions ($L^{arc}$ and $L^{end}$) all six candidates
adjudicate **exactly as they do under the speed readout**:

| Candidate | Arc-length verdict | Speed-version (G1) verdict | Changed? |
| --- | --- | --- | --- |
| DiffusionDrive | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| LTF | **FAIL** | **FAIL** | no |
| DiffusionDriveV2 | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| SimLingo | indeterminate (the two $R$ estimators disagree, §3.4) | indeterminate ($R$'s CI spans 0.5) | no (different reason) |
| Alpamayo-R1 | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| AutoVLA | indeterminate (no baseline response) | indeterminate (no baseline response) | no |

**All four candidates named in the work order (DD, DDv2, Alpamayo-R1, AutoVLA) still have
$b^{(L)}_{ghost}$ spanning zero.** This is the **third** independent readout in this line to return
a null (speed → distance → arc length). **Being a second attempt did not make me go looking for a
"this time we finally detected it" result**: the one cell that looked like a changed verdict
(SimLingo, indeterminate → FAIL) turned out to be an **estimator artefact** and was caught and
returned to indeterminate in §3.4.

**The round was nevertheless not wasted; it produced two substantive things:**
1. **The work order's technical premise is confirmed**: arc length does remove last round's
   geometric confound (measured in §2.2) — the problem was not "distance as a quantity" but
   "distance depending on an external reference point". The fix worked; the verdicts simply did not
   change once it did.
2. **An incidental finding that bears on a published number** (§3.4): F-3's necessity ratio $R$ has
   always been estimated as a **mean of per-event ratios**; under a **ratio of aggregate means**,
   **LTF's FAIL on the discovery corpus in the paper's Table 3 becomes indeterminate**. Its
   replication on NAVSIM is robust under both estimators (still FAIL).

---

## 1. Motivation and procedure

**Last round's problem**: $d_{plan}$ (endpoint distance to the hazard) was severely confounded by
geometry in cross-frame comparison — **81%–108%** of $b^{(d)}_{ghost}$ came from the entity's own
approach (median longitudinal range 30.5 → 24.6 m), plus a median **ego displacement of 7.2–8.2 m**
between the two frames, neither of which can be removed from a cross-frame comparison. Root cause:
**distance depends on the hazard's absolute position and is therefore not start-invariant.**

**This round's quantity** (computed from the waypoint sequence alone, introducing no external
reference point):

$$L^{arc}_{plan} = \lVert w_0\rVert + \sum_i \lVert w_{i+1}-w_i\rVert,
\qquad L^{end}_{plan} = \lVert w_{-1}\rVert$$

with $w$ the ego-frame waypoints (the ego origin being the vehicle's position at that frame). **The
work order requires both definitions to be computed and compared** — see §2.3. A third variant
$L^{arc\text{-}nw}$ (excluding the origin segment) was computed as a robustness check; it agrees
with $L^{arc}$ throughout and is not tabulated separately.

**Start-invariance is constructional**: both expressions are built only from **relative**
displacements between waypoints and from the waypoints to the ego origin; neither contains the
entity's position nor the ego's world pose, so last round's confound cannot arise structurally.
§2.2 verifies this empirically.

**Sign conventions (different again from both previous rounds; do not mix them)**:

| Quantity | Definition | Expected direction | Rationale |
| --- | --- | --- | --- |
| $b^{(L)}_{ghost}$ | $L(ghost)-L(clean)$ | **negative** | seeing the hazard ⇒ the plan contracts |
| $\delta^{(L)}_{occ}$ | $L(occ)-L(ghost)$ | **positive** | erasing the hazard ⇒ the plan extends again |

$R^{(L)} = 1 - b^{(L)}_{occ}/b^{(L)}_{ghost} = -\delta^{(L)}_{occ}/b^{(L)}_{ghost}$, the same
functional form as the speed version, so the PASS threshold of 0.5 carries over unchanged. The
denominator guard reuses the distance version's $|b^{(L)}_{ghost}|\ge 0.5$ m (same units).

**No new inference was run**: the full four-arm trajectories for all six candidates were stored last
round via `--save-traj`.

---

## 2. Results

### 2.1 Main table

**Table 1. Arc-length readouts for six candidates (G1 corpus, $L^{arc}$ definition, metres).**

| Candidate | n | traj points | mean $L(ghost)$ | $b^{(L)}_{ghost}$ | $\delta^{(L)}_{occ}$ | $\delta^{(L)}_{ctrl}$ | $\lvert\delta_{occ}\rvert-\lvert\delta_{ctrl}\rvert$ | Specific? | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | 282 | 8 | 11.43 | −0.697 [−1.453, +0.013] spans 0 | +0.285 [+0.063, +0.527] | −0.036 [−0.273, +0.230] | +0.119 [−0.174, +0.407] | no | indeterminate (no baseline response) |
| LTF | 282 | 8 | 14.37 | **−0.892 [−1.528, −0.263]** sig. | +0.024 [−0.097, +0.142] | −0.084 [−0.252, +0.065] | −0.042 [−0.167, +0.071] | no | **FAIL** |
| DiffusionDriveV2 | 282 | 8 | 13.82 | −0.630 [−1.296, +0.107] spans 0 | −0.095 [−0.338, +0.130] | −0.002 [−0.246, +0.214] | +0.211 [+0.010, +0.431] | no | indeterminate (no baseline response) |
| SimLingo | 282 | 10 | 16.73 | **−1.314 [−2.300, −0.363]** sig. | **+0.534 [+0.160, +0.931]** | +0.019 [−0.227, +0.265] | **+0.454 [+0.096, +0.859]** | **yes** | indeterminate (estimators disagree, §3.4) |
| Alpamayo-R1 | 119 | 64 | 42.12 | +0.770 [−2.029, +3.385] spans 0 | −0.535 [−1.452, +0.315] | +0.059 [−1.103, +1.091] | −0.828 [−2.016, +0.454] | no | indeterminate (no baseline response) |
| AutoVLA | 119 | 10 | 28.04 | −0.237 [−1.255, +0.639] spans 0 | −0.216 [−0.632, +0.231] | −0.325 [−0.915, +0.300] | −0.098 [−0.467, +0.297] | no | indeterminate (no baseline response) |

**$\delta^{(L)}_{occ}$ is significant and specific for exactly one candidate, SimLingo** (+0.534
[+0.160, +0.931]; paired quantity +0.454 [+0.096, +0.859]; $\delta_{ctrl}$ non-significant) — **and
in the expected direction** (erasing the hazard extends the plan). This agrees with the speed
version (on G1, SimLingo is likewise the only candidate whose within-frame erasure effect passes the
specificity test), so **three independent action readouts (speed / distance / arc length) point the
same way on the same data.**

DDv2's paired quantity is significantly positive (+0.211 [+0.010, +0.431]) but its $\delta_{occ}$ is
itself non-significant and points the wrong way, so under the standing three-part criterion it does
not count as specific — treated exactly as in the two previous rounds.

### 2.2 The work order's technical premise is confirmed: arc length does remove the geometric confound

**Table 2. Empirical verification of start-invariance** — Spearman correlation of each per-event
readout with the between-frame ego displacement stored last round.

| Candidate | $\rho\,(b^{(L)}_{ghost},\ \text{ego displacement})$ | $\rho\,(b^{(d)}_{ghost},\ \text{ego displ.})$ (last round, distance) |
| --- | --- | --- |
| DiffusionDrive | **+0.034** | −0.495 |
| LTF | +0.241 | −0.612 |
| DiffusionDriveV2 | **+0.008** | −0.472 |
| SimLingo | +0.112 | −0.494 |
| Alpamayo-R1 | −0.124 | +0.105 |
| AutoVLA | +0.128 | −0.180 |

The distance version correlates strongly with ego displacement in four candidates (**−0.47 to
−0.61**) — that was the geometric confound. The arc-length version stays within **|ρ| ≤ 0.24**, and
three of four are inside 0.13. ⇒ **The work order's technical judgement was correct: switching to a
quantity with no external reference point does remove the confound.** This round's null therefore
**cannot** be attributed to a geometry-contaminated readout — this readout is clean.

**One qualification**: LTF's +0.241 is the largest of the six. Arc length contains no ego pose by
construction, so this cannot be geometry leaking in directly; it is more likely genuine covariation
("in scenes where the ego drives faster the model also plans further"), since $L$ correlates
strongly with speed (§2.4). Reported as measured rather than as zero.

### 2.3 The two definitions of $L$ give identical conclusions

**Table 3. $L^{arc}$ vs $L^{end}$ (the work order asks that neither be reported alone).**

| Candidate | $\delta_{occ}$ (arc) | $\delta_{occ}$ (end) | Specific (arc/end) | Verdict (arc/end) |
| --- | --- | --- | --- | --- |
| DiffusionDrive | +0.285 [+0.063, +0.527] | +0.284 [+0.062, +0.529] | no / no | indet. / indet. |
| LTF | +0.024 [−0.097, +0.142] | +0.011 [−0.108, +0.129] | no / no | FAIL / FAIL |
| DiffusionDriveV2 | −0.095 [−0.338, +0.130] | −0.088 [−0.331, +0.135] | no / no | indet. / indet. |
| SimLingo | +0.534 [+0.160, +0.931] | +0.523 [+0.148, +0.921] | yes / yes | indet. / indet. |
| Alpamayo-R1 | −0.535 [−1.452, +0.315] | −0.672 [−1.590, +0.196] | no / no | indet. / indet. |
| AutoVLA | −0.216 [−0.632, +0.231] | −0.209 [−0.615, +0.212] | no / no | indet. / indet. |

**The two definitions agree on significance, on specificity and on the verdict in all 6 × 2 = 12
cells.** Point-estimate differences are small (largest 0.14, for Alpamayo — whose 64-point horizon
is the longest, so path length and straight-line distance naturally diverge most: $L^{arc}$ 42.12 vs
$L^{end}$ 39.75). ⇒ On these data, "path length" and "endpoint straight-line distance" are **not two
readouts that would give different conclusions**; both principally measure "how far the model
intends to travel". The third variant $L^{arc\text{-}nw}$ agrees as well.

### 2.4 This readout is not information independent of the speed version

**Table 4. Per-event Spearman correlation of $\delta^{(L)}_{occ}$ with the speed version's within-frame erasure effect $d_{occ}=v(ghost)-v(occ)$.**

| Candidate | Spearman $\rho$ |
| --- | --- |
| DiffusionDrive | −0.375 |
| LTF | −0.271 |
| DiffusionDriveV2 | **−0.619** |
| SimLingo | **−0.725** |
| Alpamayo-R1 | −0.458 |
| AutoVLA | +0.141 |

**The negative sign is a convention artefact** (speed uses $d_{occ}=v(ghost)-v(occ)$ while arc
length uses $\delta_{occ}=L(occ)-L(ghost)$, so "erasing the hazard ⇒ drives faster/further" gets
opposite signs) ⇒ it is $|\rho|$ that measures shared variation, and it is **0.38–0.73** across four
candidates.

**This explains why no verdict changed**: $L$ is essentially "mean speed over the planning horizon ×
horizon length", while $v_{plan}$ is "instantaneous speed at the first step". **They are the same
physical quantity read over different time windows, not two independent pieces of information.** The
scenario the work order envisaged — the model changing its path rather than its speed — requires a
quantity **insensitive to speed and sensitive only to lateral displacement** (e.g. the trajectory's
lateral offset, or its minimum lateral clearance to the entity). Both $L^{arc}$ and $L^{end}$ are
longitudinal extents and are **necessarily dominated by speed**. This is an intrinsic limitation of
this round's design and is stated as such.

### 2.5 (a) paired vs (b) group-mean comparison: identical point estimates, CIs differing 4.8–13.6×

**The work order asks for both and for a statement of whether they agree.** Mathematically, on the
same event set with no missing arm,

$$\overline{L(occ)-L(ghost)} \;=\; \overline{L(occ)}-\overline{L(ghost)}$$

(linearity of the mean) ⇒ **the point estimates must be identical to the digit**. Any difference can
only appear in the CI, and depends on how the bootstrap is done.

**Table 5. (a) paired vs (b) group means ($L^{arc}$).**

| Candidate | (a) paired point est. | (b) difference of group means | $\lvert$gap$\rvert$ | (a) CI half-width | (b1) shared resample | (b2) independent resample | (b2)/(a) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | +0.285018 | +0.285018 | 1.8 × 10⁻¹⁵ | 0.2321 | 0.2321 | 1.6971 | **7.3×** |
| LTF | +0.023567 | +0.023567 | 1.8 × 10⁻¹⁵ | 0.1199 | 0.1199 | 1.5946 | **13.3×** |
| DiffusionDriveV2 | −0.095013 | −0.095013 | 1.2 × 10⁻¹⁵ | 0.2342 | 0.2342 | 1.5877 | **6.8×** |
| SimLingo | +0.534034 | +0.534034 | 6.1 × 10⁻¹⁵ | 0.3852 | 0.3852 | 1.8538 | **4.8×** |
| Alpamayo-R1 | −0.535178 | −0.535178 | 9.1 × 10⁻¹⁵ | 0.8836 | 0.8836 | 6.0099 | **6.8×** |
| AutoVLA | −0.216303 | −0.216303 | 4.7 × 10⁻¹⁵ | 0.4312 | 0.4312 | 4.4311 | **10.3×** |

**Three conclusions:**
1. **The point estimates are identical to the digit** (gaps of order 10⁻¹⁵ = floating-point
   rounding) ⇒ the mathematical expectation is confirmed empirically; no discrepancy to explain.
2. **(b1) — both groups resampled together — is exactly equivalent to (a)**, with CI half-widths
   identical to the digit, because the shared resample preserves the pairing.
3. **(b2) — each group resampled independently — inflates the CI by 4.8–13.6×.** Under that
   protocol **SimLingo's $\delta_{occ}$ (+0.534; half-width 0.385 → 1.854) would go from significant
   to non-significant**, and none of the six would be significant.

⇒ **All of the paired design's statistical power comes from "same events, same resample", not from
the readout itself.** Treating the four arms as two independent "occluded / un-occluded" groups
throws away an order of magnitude of power. The group means themselves show why: occ group
11.718 [10.520, 12.958] vs ghost group 11.433 [10.237, 12.660] for DD — the two CIs almost entirely
overlap, because **between-event variation (a 10–13 m spread in plan length) is two orders of
magnitude larger than the between-arm difference (0.29 m)**, and only pairing cancels the former.

### 2.6 The two $R^{(L)}$ estimators and SimLingo's apparent verdict change

See §3.4 — this is an independent finding of this round and is set out separately.

> **Instrument side.** **No new inference was run**; every readout is re-derived from the full
> trajectories stored last round. Start-invariance was **verified empirically** (correlation with
> ego displacement falls from the distance version's −0.47…−0.61 to |ρ| ≤ 0.24); the two $L$
> definitions agree on all 12 cells; (a)/(b) point estimates match to 10⁻¹⁵. The six candidates'
> horizons differ substantially (8 / 10 / 64 points), so absolute $L$ values are **not comparable
> across candidates** and every adjudication is a **within-candidate** comparison between arms.
> **Specimen side.** With a clean, start-invariant readout, **none** of the four candidates named in
> the work order is rescued; $\delta^{(L)}_{occ}$ is significant and specific for SimLingo alone,
> consistent with both the speed and distance versions.

---

## 3. Discussion

**1. Three independent readouts and the same null — the evidentiary value of that is rising.**
Speed (first-order, first step), distance (zeroth-order, reference-point dependent) and arc length
(zeroth-order, start-invariant) have quite different failure modes — speed is contaminated by the
clean arm, distance by geometry, arc length by being speed-dominated — yet all three point the same
way on the same data: **these candidates' actions and paths are genuinely not driven by the entity's
visibility.** A single null invites the suspicion that the readout was poorly chosen; **three
constructionally different readouts agreeing makes that explanation progressively harder to hold.**

**2. But this round's own limitation must be stated: it cannot detect purely lateral avoidance.**
Both $L^{arc}$ and $L^{end}$ are **longitudinal** extents, strongly correlated with speed
(|ρ| 0.38–0.73, §2.4). A model that responds to a hazard by **moving aside without changing speed**
would be equally invisible to this round's readout. The work order's motivation ("the reaction shows
up as a change of path rather than of speed") is therefore **only partly covered**: covered for
"change of path *length*", not for "change of path *shape* / lateral offset". A quantity genuinely
orthogonal to speed must be **lateral** (e.g. the trajectory's lateral offset relative to baseline,
or its minimum lateral clearance to the entity) — that is what should be tried next, and **this
round did not opportunistically switch to it**; it is registered as unfinished.

**3. The work order's premise is confirmed, but "fixed it" and "can now detect it" are two different
things.** Last round diagnosed the distance readout as geometry-confounded because it is not
start-invariant; this round substituted a start-invariant quantity and measured the confound away
(§2.2). **That confirms last round's diagnosis was right.** But the verdicts still did not change,
so last round's null was **not** caused by that confound. Together the two rounds rule out "the
readout was defective" as the explanation for the null.

**4. A reusable statistical discipline (corollary of §2.5).** Comparing the four arms as "two
independent groups" throws away 4.8–13.6× of power in a paired design **while leaving the point
estimate untouched** ⇒ nothing looks wrong in the point estimate, and only the CI quietly widens.
Every four-arm readout in this project must use paired / shared resampling; this round confirms that
quantitatively.

---

## 4. A finding that bears on a published number: the $R$ estimator (§FE/A67)

F-3's necessity ratio $R$ has, since it was designed, been estimated as a **mean of per-event
ratios** over a scene-level bootstrap. The per-event denominator $b_{ghost}$ can be small (the
denominator guard only partly mitigates this), the ratios are heavy-tailed, and
$\mathbb{E}[X/Y]\neq \mathbb{E}[X]/\mathbb{E}[Y]$. This round additionally computed the **ratio of
aggregate means**, $R = -\overline{\delta_{occ}}/\overline{b_{ghost}}$, under the same scene-level
bootstrap.

**Table 6. The two estimators on the speed version's already-published cells.**

| Cell | mean of per-event ratios $R$ | Verdict | ratio of aggregate means $R$ | Verdict | Agree? | Does $R$ enter the verdict? |
| --- | --- | --- | --- | --- | --- | --- |
| **G1 × LTF** | **+0.113 [+0.044, +0.191]** | **FAIL** | **+0.283 [+0.158, +0.680]** | **indeterminate** | **no** | **yes** (gate open) |
| NAVSIM × LTF | +0.008 [−0.068, +0.078] | FAIL | +0.010 [−0.121, +0.172] | FAIL | yes | yes |
| G1 × SimLingo | +0.461 [+0.166, +0.789] | indet. | +0.387 [+0.156, +0.875] | indet. | yes | yes |
| NAVSIM × SimLingo | +0.114 [−0.399, +0.627] | indet. | +0.857 [+0.350, +3.525] | indet. | yes | yes |
| G1 × DiffusionDrive | +0.052 [−0.086, +0.184] | FAIL | +0.647 [−2.580, +3.828] | indet. | no | **no** (gate closed) |
| G1 × DDv2 (+lidar) | +0.133 [−0.099, +0.372] | FAIL | −0.089 [−1.164, +0.486] | FAIL | yes | no |

**Scope of the impact (stated exactly, not inflated):**
* **Exactly one published verdict is affected: LTF's FAIL on the discovery corpus in the paper's
  Table 3.** Under the ratio of means the CI upper bound is +0.680 > 0.5, so that cell becomes
  indeterminate.
* **The paper's replication claim is unaffected**: LTF's FAIL **on NAVSIM is robust under both
  estimators** (+0.008 / +0.010, CI upper bounds 0.078 / 0.172, both far below 0.5). The statement
  that "LTF's blind-action signature reproduces on NAVSIM" still holds.
* The disagreements for DD and DDv2 are **inconsequential**: their gate ($b_{ghost}$ significant) is
  closed anyway, so $R$ never enters their verdict.

**I did not unilaterally edit the paper, nor unilaterally switch estimators.** Reason: "which
estimator is the one F-3 wants" is a methodological question deserving a deliberate decision — the
mean of per-event ratios estimates "the average necessity ratio across events", the ratio of means
estimates "the fraction of the average response that was removed", and F-3's PASS threshold
("occlusion must remove more than half of the response") reads closer to the latter. That choice
would affect **every** $R$ readout in this line and **should not be settled in passing during a round
about arc length**. This round therefore reports it as a finding, registers it as the
highest-priority open decision, and — within this round's own adjudications — **returns a verdict of
indeterminate whenever the two estimators disagree** (as they do for SimLingo).

---

## 5. Record of self-correction

1. **SimLingo briefly looked like a changed verdict (indeterminate → FAIL); I checked and returned
   it to indeterminate.** The per-event ratio mean gives $R^{(L)}$ = −0.063 [−0.248, +0.088] ⇒ FAIL,
   while the ratio of means gives +0.407 [+0.107, +1.327] ⇒ indeterminate. They disagree, so neither
   is adopted. **Without that check this report would have announced "the arc-length readout
   adjudicated SimLingo as FAIL" as a new finding.** It does not hold: it is an estimator artefact.
2. **DDv2's paired quantity is significantly positive (+0.211) but does not count as specific**
   (its $\delta_{occ}$ is non-significant and points the wrong way) — treated as in the previous two
   rounds, without relaxation.
3. **This round's limitation (blindness to purely lateral avoidance) is one I raised myself, not one
   I was asked about** (§3.2). The work order's motivation was "path rather than speed", yet $L$
   correlates with speed at |ρ| up to 0.73 ⇒ this round covers only half of that motivation.
   **The claim "arc length is zeroth-order and therefore more sensitive" was not carried into the
   conclusion unexamined.**
4. **I did not opportunistically try a lateral readout.** Having found $L$ to be speed-dominated,
   switching to lateral offset is the obvious next step; but changing the action definition within
   the same round is protocol shopping, so it is registered as unfinished and left for
   pre-registration.
5. **I did not unilaterally modify the paper or change the $R$ estimator** (reasoning at the end of
   §4); the finding and its precise scope are documented instead.

---

## 6. Unfinished items

| Item | Status | Reason / recommendation |
| --- | --- | --- |
| **Decide which $R$ estimator to use** | **open (highest priority)** | affects every $R$ readout in this line; the FAIL in the paper's Table 3 cell G1 × LTF depends on it. Recommend a dedicated round that lays out both definitions, how each matches the literal PASS threshold, and a recomputation of all cells, before deciding |
| **A lateral readout** (trajectory lateral offset / minimum lateral clearance to the entity) | not done | this round shows $L$ is speed-dominated (\|ρ\| up to 0.73); a quantity genuinely orthogonal to speed must be lateral. **Should be pre-registered before running** |
| Rebuilding F-3's gate around a **within-frame** contrast | not done | the two previous rounds each added support; the denominator still has to be defined |
| Extending the arc-length readout to lead-brake / NAVSIM | not done | verdicts on G1 are identical to the speed version, so extension carries little information; and the $R$ estimator is unresolved |
| Folding into the paper | **not done, but one item now needs a paper-level decision** | the arc-length readout is itself a null and triggers no table change; the $R$-estimator issue in §4 touches one Table 3 cell and should be handled once the estimator is settled |
