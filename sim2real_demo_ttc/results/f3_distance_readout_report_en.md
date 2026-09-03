# F-3 with a Distance Readout: Replacing Commanded Speed with Trajectory-Endpoint Distance to the Hazard

> Work order: this round (2026-09-03). Follows [`f3_all_done_FINAL.md`](f3_all_done_FINAL.md).
> Autonomous decisions: [`amendments.md`](amendments.md) §FD/A63–A64.
> Artifacts: `f3_occlusion_{dd,ltf,ddv2,simlingo,alpa,autovla}_dist.json` (containing **full planned
> trajectories**), `f3_distance_readout.json`; script `scripts/f3_distance_analysis.py`.
> **No existing result was modified or deleted**; all new results carry the `_dist` suffix.

---

## 0. Direct answer to the work order's question

> **Under the distance readout, is $b^{(d)}_{ghost}$ still indistinguishable from zero for the four
> candidates that currently show "no baseline response" (DiffusionDrive, DiffusionDriveV2,
> Alpamayo-R1, AutoVLA)?**

**All six candidates' $b^{(d)}_{ghost}$ are "significantly non-zero" — but 81%–108% of that
significance is contributed by the hazard entity's own approach, not by any model response.** Once
that term is subtracted explicitly, **no candidate shows a significant, specific path response on
the properly controlled readout.**

The conclusion of this round is therefore: **switching to a more sensitive zeroth-order spatial
readout still detects no response**; and additionally — **the primary readout specified by the work
order is unusable under the present clean/ghost frame-selection design** (§3.2), which is the most
consequential finding here.

**No protocol was adjusted and no data was cherry-picked to make the new idea look useful**: all six
candidates, all four arms, all 288/119 events were run, with none dropped. Applying the speed
version's adjudication rule mechanically would have produced **six FAILs** — which would look like
"the new readout is more sensitive and finally detects something". This report explicitly **declines
to adopt** those six FAILs; the reasoning is in §3.2.

---

## 1. Motivation and procedure

**Motivation (from the work order)**: $v_{plan}$ is computed from the trajectory's **first step**
only — an instantaneous speed command over the shortest time window — and all planning information
beyond that first step is discarded. If a model's reaction to a hazard shows up as **a change of
path** (moving aside, adjusting where the trajectory terminates) rather than as an immediate change
of the next speed command, the speed readout cannot see it. Distance is a **zeroth-order spatial
quantity** whereas a speed difference is **second-order**, so the former should have a better
signal-to-noise ratio.

**Procedure**:
1. All six candidates (dd / ltf / ddv2 / simlingo / alpa / autovla) rerun F-3's four arms on the G1
   corpus, **reusing the existing input construction and occlusion logic without changing a line**,
   adding only a `--save-traj` switch that stores the **full trajectory** (the existing
   `f3_occlusion_*.json` files store scalars only, so pure re-analysis cannot recover it).
   DDv2 runs with lidar occlusion (m = 0.25), matching the existing fixed protocol.
2. The hazard entity's position **reuses the same projection geometry**
   (`f3_window_boxes.WindowBoxes.box3d_for`), recomputed in ego coordinates at the timestamp of
   **the frame F-3 actually queries** — **not** the position at `d_long_at_emergence`.
3. Readouts are defined in §2. Three-state adjudication, scene-level bootstrap (5000), and the
   mandatory ctrl control arm are **all retained without relaxation**.

**The switch is off by default and existing artifacts are bit-identical**: rerunning DD's first six
events without `--save-traj` reproduces the committed `f3_occlusion_dd.json` **digit for digit
(0 of 24 numbers differ)** and writes no trajectory fields.

---

## 2. Notation and sign conventions (**different from the speed version; do not mix them**)

For each arm $a\in\{clean,ghost,occ,ctrl\}$, take that arm's planned trajectory **endpoint** (the
final waypoint of the horizon, not the first step) $w^a_{-1}$, and the hazard entity's ego-frame
position $p_{ent}$ **at the frame that arm queries**:

$$d^{a}_{plan} = \lVert w^{a}_{-1}[:2] - p_{ent}[:2]\rVert_2 \quad(\text{metres})$$

| Quantity | Definition | **Expected direction** | Corresponding speed-version direction |
| --- | --- | --- | --- |
| $b^{(d)}_{ghost}$ | $d^{ghost}_{plan}-d^{clean}_{plan}$ | **positive** (seeing the hazard ⇒ endpoint farther from it) | speed-version $b_{ghost}$ expected **negative** (slowing) |
| $b^{(d)}_{occ}$ | $d^{occ}_{plan}-d^{clean}_{plan}$ | → 0 if occlusion is effective | same |
| $\delta_{occ}$ | $d^{occ}_{plan}-d^{ghost}_{plan}$ | **negative** (erasing the hazard ⇒ endpoint moves back toward it) | — |

**Derivation of the necessity ratio** (identical functional form to the speed version, so the PASS
threshold of 0.5 carries over unchanged):

$$R^{(d)} \;=\; 1-\frac{b^{(d)}_{occ}}{b^{(d)}_{ghost}}
\;=\;\frac{b^{(d)}_{ghost}-b^{(d)}_{occ}}{b^{(d)}_{ghost}}
\;=\;\frac{d^{ghost}_{plan}-d^{occ}_{plan}}{b^{(d)}_{ghost}}
\;=\;\frac{-\,\delta_{occ}}{b^{(d)}_{ghost}}$$

$R^{(d)}$ is a ratio of two **same-signed** quantities, so the sign flip noted in the table above
**does not affect it**: if occlusion fully removes the response then $d^{occ}\approx d^{clean}$,
so $b^{(d)}_{occ}\approx 0$ and $R^{(d)}\approx 1$. **Note that its structure is exactly isomorphic
to the speed version: the numerator $-\delta_{occ}$ is a within-frame quantity (unconfounded) while
the denominator is a cross-frame quantity (confounded)** — precisely the structure identified for
the speed version in §FC/A61.

**The denominator guard uses a different threshold**: distance is a zeroth-order quantity measured
in metres, so the speed version's $|b_{ghost}|\ge 0.02$ m/s is not transferable; this round uses
$|b^{(d)}_{ghost}|\ge 0.5$ m. (The threshold does not affect this round's conclusion — see §3.2, the
gate itself is invalid.)

---

## 3. Results

### 3.1 Three layers of readout, in increasing order of control

**Table 1. Distance readouts for six candidates (G1 corpus). $d$ is in metres.**

| Candidate | n (with clean arm) | traj points | mean $d^{ghost}_{plan}$ | $b^{(d)}_{ghost}$ (**work order's primary**) | of which **pure geometry** | **confound-adjusted** |
| --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | 282 (230) | 8 | 18.87 | −8.120 [−8.979, −7.323] | −8.673 [−9.401, −8.003] | +0.554 [+0.029, +1.106] |
| LTF | 282 (230) | 8 | 15.39 | −8.594 [−9.422, −7.786] | −9.209 [−9.908, −8.516] | +0.615 [+0.047, +1.178] |
| DiffusionDriveV2 | 282 (230) | 8 | 15.31 | −8.423 [−9.458, −7.454] | −8.362 [−9.181, −7.573] | −0.062 [−0.686, +0.515] |
| SimLingo | 282 (230) | 10 | 13.07 | −8.569 [−9.743, −7.337] | −9.253 [−10.063, −8.425] | +0.684 [−0.219, +1.677] |
| Alpamayo-R1 | 119 (89) | 64 | 25.66 | **+5.482 [+2.619, +8.600]** | +4.423 [+2.284, +6.639] | +1.060 [−2.200, +4.307] |
| AutoVLA | 119 (94) | 10 | 8.54 | −4.874 [−6.954, −2.994] | −4.525 [−6.551, −2.751] | −0.349 [−1.182, +0.500] |

**Definitions of the three layers** (increasing control):
1. **$b^{(d)}_{ghost}$** (work order's primary): ghost frame vs clean frame, **each using the entity
   position from its own frame**.
2. **Pure-geometry term** $= d^{null}-d^{clean}_{plan}$, where
   $d^{null}=\lVert w^{clean}_{-1}-p_{ent}(t_{ghost})\rVert$ — i.e. the **clean arm's trajectory**
   measured against the **ghost frame's entity position**: how large a $b^{(d)}$ would arise from the
   entity's approach alone if the plan did not change at all.
3. **Confound-adjusted** $= d^{ghost}_{plan}-d^{null}$ — both trajectories measured against **the
   same point**.

### 3.2 The primary readout is unusable: 81%–108% of the gate is opened by geometry (the key finding)

**Table 2. Gate-validity diagnostics.**

| Candidate | pure geometry as a share of $b^{(d)}_{ghost}$ | adjusted term significant? | median **ego displacement** between the two frames | adjusted effect / that residual | gate valid? |
| --- | --- | --- | --- | --- | --- |
| DiffusionDrive | **107%** | yes | 7.5 m | 0.073 | **no** |
| LTF | **107%** | yes | 7.5 m | 0.082 | **no** |
| DiffusionDriveV2 | **99%** | no | 7.5 m | 0.008 | **no** |
| SimLingo | **108%** | no | 7.5 m | 0.091 | **no** |
| Alpamayo-R1 | **81%** | no | 8.2 m | 0.130 | **no** |
| AutoVLA | **93%** | no | 7.2 m | 0.048 | **no** |

The entity's median longitudinal range moves from **30.5 m to 24.6 m** (approaching by ~6 m). If the
plan were **completely unchanged** in ego coordinates, that approach alone would shrink $d_{plan}$
and drive $b^{(d)}_{ghost}$ negative. Empirically the **pure-geometry term accounts for essentially
the whole primary readout** (81%–108%).

**⇒ Applying the speed version's three-state rule mechanically yields six spurious FAILs.** The
speed version's gate is "$b_{ghost}$ is significant"; in the distance version that gate **can be
opened by the entity's own approach alone**, independently of whether the model reacts. Once opened,
$R^{(d)}\approx 0$ (Table 3), so all six cells adjudicate FAIL — which would look like "the new
readout is more sensitive and has finally caught these four candidates", when **not one of them
holds**. This report **does not adopt** those six FAILs; they are retained in
`f3_distance_readout.json` under `verdict_mechanical` for inspection, while `verdict` has been
changed to "indeterminate: the primary readout's gate is opened by geometric confounding".

**The adjusted layer cannot serve as a conclusion either**: DD's +0.554 and LTF's +0.615 are
significantly positive and in the expected direction, but **the ego itself moved a median of 7.5 m
between the two frames**, and that term cannot be removed from any cross-frame comparison; the
adjusted effect is only **0.008–0.130×** that residual. In other words, **an effect more than an
order of magnitude smaller than an uncontrolled residual must not be read as a model response**.

### 3.3 The one properly controlled readout: $\delta_{occ}$ (same frame, same reference point, same ego origin)

$\delta_{occ}=d^{occ}_{plan}-d^{ghost}_{plan}$: occ and ghost are **the same frame**, referenced to
**the same entity position**, planned from **the same ego origin**, differing only by the grey patch.
**None of the three confounds above is present.**

**Table 3. The controlled readout and its mandatory ctrl control arm.** (Expected direction:
$\delta_{occ}$ **negative**.)

| Candidate | $\delta_{occ}$ | $\delta_{ctrl}$ | $\lvert\delta_{occ}\rvert-\lvert\delta_{ctrl}\rvert$ | Significant? | Specific? | $R^{(d)}$ | $R^{(d)}_{ctrl}$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DiffusionDrive | −0.122 [−0.330, +0.047] | −0.040 [−0.310, +0.197] | +0.024 [−0.237, +0.276] | no | no | +0.066 [−0.024, +0.217] | +0.029 [−0.023, +0.110] |
| LTF | −0.006 [−0.101, +0.087] | +0.016 [−0.101, +0.141] | −0.024 [−0.129, +0.070] | no | no | −0.007 [−0.030, +0.018] | +0.014 [−0.012, +0.042] |
| DiffusionDriveV2 | +0.022 [−0.173, +0.232] | −0.065 [−0.227, +0.100] | **+0.206 [+0.043, +0.390]** | no | no | +0.002 [−0.069, +0.083] | −0.045 [−0.177, +0.052] |
| SimLingo | **−0.306 [−0.630, −0.024]** | −0.072 [−0.311, +0.160] | +0.296 [−0.005, +0.630] | **yes** | **no** (paired quantity's CI grazes 0) | −0.009 [−0.102, +0.074] | +0.031 [−0.029, +0.097] |
| Alpamayo-R1 | −0.052 [−0.703, +0.625] | +0.592 [−0.405, +1.462] | −0.920 [−1.862, +0.021] | no | no | −0.062 [−0.223, +0.112] | +0.004 [−0.167, +0.169] |
| AutoVLA | −0.183 [−0.467, +0.126] | −0.228 [−0.745, +0.292] | +0.060 [−0.213, +0.352] | no | no | +0.052 [−0.074, +0.187] | +0.031 [−0.052, +0.122] |

**Of six candidates, $\delta_{occ}$ is significant for exactly one (SimLingo); zero pass the full
specificity test.** SimLingo's paired quantity is +0.296 [−0.005, +0.630], whose **lower bound
grazes zero** (−0.005) — under the existing three-part criterion ($\delta_{occ}$ significant AND
$\delta_{ctrl}$ not significant AND the paired quantity significantly positive) it **does not count
as specific**. **It was not relaxed just because it came close**: the speed version treats LTF on G1
the same way (`specific=False`).

DDv2's paired quantity is significantly positive (+0.206), but its $\delta_{occ}$ is itself
non-significant and points the wrong way, so it likewise does not count as specific.

> **Instrument side.** `--save-traj` is off by default and existing readouts are bit-identical with
> it off (0 of 24 numbers differ); trajectory-vs-`commanded_speed` consistency was verified
> ($\lVert w_0\rVert/\Delta t$ matches the stored scalar digit for digit). Entity positions reuse the
> mining-time projection, taken at the timestamp of **the frame F-3 actually queries** rather than at
> emergence. **The primary readout's gate is diagnosed as invalid** (geometry share 81%–108%), and
> the one unconfounded readout $\delta_{occ}$ is measured with four symmetric arms sharing frame and
> reference point. The six candidates' planning horizons differ substantially (8 / 10 / 64 points),
> so absolute $d_{plan}$ values are **not comparable across candidates**; every adjudication here is
> a **within-candidate** comparison between arms.
> **Specimen side.** Under a zeroth-order spatial readout, **not one** of the six candidates shows a
> significant, specific path response; **none** of the four "no baseline response" candidates named
> in the work order is rescued. SimLingo is the only candidate with a significant $\delta_{occ}$
> (−0.306 m), and it does not pass the specificity test.

---

## 4. Discussion

**1. The work order's hypothesis (distance is more sensitive than speed) is not supported here, but
not because distance is useless.** The $\delta_{occ}$ layer is exactly the cleaner, zeroth-order,
four-arm-symmetric quantity the work order envisaged. What it measures is: five of six candidates
span zero, and one (SimLingo) is significant but not specific. **This agrees with the speed
version's conclusion** (on G1, SimLingo is likewise the only candidate whose $d_{occ}$ passes the
specificity test). Two independent action readouts agreeing on the same data is **independent
support** for the conclusion that these candidates' actions are not driven by the entity's
visibility.

**2. This round's most valuable output is a methodological conclusion, not a new number.**
**Any cross-frame action readout whose action quantity is not start-invariant will be dominated by
geometry.** Speed is start-invariant ($v_{plan}$ does not depend on where the ego is in the world),
so even though the speed version's $b_{ghost}$ is weakened by the clean-arm contamination of
§FC/A61, it is at least measuring the model's output. Distance is **not** start-invariant: the ego
moving 7.5 m plus the entity approaching 6 m together manufacture a −8 m "significant response",
while the effect actually under test is of order 0.5 m. ⇒ **When switching the action to a spatial
quantity, the contrast must be switched to a within-frame contrast at the same time**, or the new
readout's signal-to-noise gets worse rather than better.

**3. What this implies for the outstanding "rebuild F-3's gate around $d_{occ}$" item.**
`f3_all_done_FINAL.md` lists that as the highest-priority unfinished item. This round adds support:
**under the distance readout, the cross-frame denominator is again the contaminated term and the
within-frame numerator is again the clean one** — the structure is identical across two different
action readouts. That indicates the problem lies in **the cross-frame comparison design**, not in
the choice of action quantity.

**4. One inter-candidate difference that should not be overlooked.** Alpamayo-R1's
$b^{(d)}_{ghost}$ is **positive** (+5.48), opposite to the other five. The cause is its much longer
planning horizon (64 points vs 8–10), which places the endpoint well beyond the entity and flips the
sign of the geometric term (+4.42). **This is not "Alpamayo responds while the others do not"** but
the same geometric confound expressed differently at a different horizon — further evidence that this
primary readout is not comparable across candidates.

---

## 5. Record of self-correction

1. **The work order's primary readout $b^{(d)}_{ghost}$ was judged unusable, and I did not change the
   protocol to "rescue" it.** Instead it was run and reported as specified, with the pure-geometry
   and adjusted terms **computed explicitly**, diagnostics explaining why it is unusable, and the
   mechanical verdict preserved in a `verdict_mechanical` field for inspection. **No more favourable
   action definition was substituted** (e.g. minimum distance from the trajectory to the entity) to
   make the result look better — that would be post-hoc protocol selection.
2. **Applying the speed version's rule mechanically produces six FAILs, which I explicitly decline to
   adopt.** Adopting them would turn this report into the attractive-looking claim that "the new
   readout adjudicated all four 'no baseline response' candidates as FAIL". It is false: the gate is
   opened by geometry. **Reporting "indeterminate" is preferable.**
3. **SimLingo's specificity test misses by a hair (paired CI lower bound −0.005) and was not
   relaxed.** This matches existing discipline (the speed version judges LTF on G1 `specific=False`).
4. **The denominator guard was changed from 0.02 to 0.5**: the speed version's threshold is in m/s,
   the distance version's in m, so it is not transferable. The change **does not affect this round's
   conclusion** (the gate is already invalid), but is recorded to avoid any impression that the old
   threshold was carried over.
5. **`--save-traj` defaults to off and was verified bit-for-bit** (0 of 24 numbers differ, no
   trajectory-field leakage), so existing speed-version artifacts and the committed paper numbers are
   unaffected by this round's code changes.

---

## 6. Unfinished items

| Item | Status | Reason |
| --- | --- | --- |
| Rebuilding F-3's gate around a **within-frame** contrast (numerator and denominator both within one frame) | not done | this round adds independent support for it (§4.3), but it remains a methodological change requiring the denominator to be defined first |
| Extending the distance readout to lead-brake / NAVSIM | not done | the controlled readout already gives a consistent null on G1 and the primary protocol is judged invalid; the cross-frame problem should be solved first |
| Other spatial readouts such as "minimum distance from the trajectory to the entity" | not done | switching the action definition post hoc is protocol shopping; if pursued, it should be **pre-registered** first |
| Folding the distance readout into the paper | not done | this round yields a null result plus a methodological limitation and changes no existing verdict ⇒ it triggers no table change; whether it belongs in Validity Checks is for the next round to decide |
