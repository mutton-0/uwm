# Extending the F-3 Occlusion Fixes to the Remaining Corpora (Lead-Brake / NAVSIM)

> Work order: [`../docs/f3_final_cleanup_workorder.md`](../../docs/f3_final_cleanup_workorder.md), Part 1.
> Follows [`f3_multiframe_fix_DONE.md`](f3_multiframe_fix_DONE.md) (§FM/A56–A59, the fixes as validated on G1).
> Autonomous decisions: [`amendments.md`](amendments.md) §FC/A60–A61.
> Artifacts: `f3_occlusion_{leadbrake,navsim}_ddv2_lidar{,_m0}.json`,
> `f3_clean_arm_audit_{g1,leadbrake}.json` (measured clean-arm visibility; script `scripts/f3_clean_arm_audit.py`),
> `f3_occlusion_leadbrake_{alpa,autovla}_mf.json`, `ns_yaw_audit{,_vehicle}.json`,
> `f3_within_frame_effect.json`. **No old result file was deleted.**

---

## 1. Motivation

The two occlusion defects fixed in the previous round — the VLA candidates' **per-frame projection
occlusion** (the original implementation occluded only the last frame of the window) and
DiffusionDriveV2's **lidar-channel occlusion** (the original painted RGB only, leaving the entity
intact in the point cloud) — **were validated on the G1 corpus alone**. Yet these candidates'
F-3 numbers on the **lead-brake** and **NAVSIM/OpenScene** corpora, produced with the *unfixed*
method, have already entered the paper's cross-scenario / cross-datasource replication table
(`g_vs_f3_replication_report_zh.md`, Table 1). Without a rerun, the paper's "replication" claim
would rest on **methods that differ between the cells being compared**.

This round reruns four cells: DDv2 × {lead-brake, NAVSIM} with lidar occlusion added, and
Alpamayo-R1 / AutoVLA × lead-brake with per-frame projection occlusion added. (Both VLAs on NAVSIM
are **n/a by pre-existing record** — their adapters depend on the nuScenes devkit and `sd_token`,
which NAVSIM data are not; this is a corpus-side interface gap, not a model-side impossibility. The
work order also explicitly skips it, and nothing was forced to fill the cell.)

---

## 2. Method

**The protocol is identical to the one just applied on G1; not one parameter was retuned:**

| Item | Setting |
| --- | --- |
| DDv2 lidar occlusion | in addition to painting RGB, delete points falling inside the entity's **ego-frame 3D box**; margin = 0.25 m as primary, 0.0 m as sensitivity |
| Control arm (lidar) | delete points inside the **mirrored 3D box** ($y\to-y$, $\mathrm{yaw}\to-\mathrm{yaw}$): volume strictly equal, longitudinal distance identical |
| VLA per-frame occlusion | project the entity independently at every timestamp in the window; where visible, paint with **that frame's own** mean colour; where not visible, leave untouched — no extrapolation |
| Control arm (RGB) | equal area, same eccentricity band, non-overlapping with the original box; painted at the same position on every frame so the number of occluded frames is comparable across arms |
| Statistics | scene-level bootstrap (5000), three-state adjudication, denominator guard $|b_{ghost}|\ge0.02$ — all unchanged |

**The only new code: a 3D-box source for NAVSIM (§FC/A60).**
`f3_window_boxes.WindowBoxes` previously had a nuScenes backend only (`compute_scene_geometry`,
devkit-dependent). NAVSIM event `scene_name`s look like `log-0001-scene-0001` and are not nuScenes
scenes, so that backend cannot supply 3D boxes. A NAVSIM backend was added: geo is built by
`ns1_navsim_geometry.build_geo` from the log pickles (**field-for-field aligned with the nuScenes
version; this is existing code, not newly written geometry**), and the nuScenes path is
**bit-for-bit unchanged**.

### 2.1 An orientation-convention defect found while adding that backend (§FC/A61)

NAVSIM's `gt_boxes[:, 6]` stores yaw **already in the ego (= lidar) frame**, whereas
`g1_mine_events.frame_bbox` (reused verbatim by `ns1`) assumes `_rot` is **world-frame** and
internally subtracts the ego heading $\psi$ ⇒ **the subtraction is applied once too often**,
which is equivalent to rotating every box footprint by a systematic $-\psi_{ego}$.

**This was not inferred by reading code; it was measured** (`scripts/ns_yaw_audit.py`). Taking
**276 vehicle tracks** from scenes where the ego turns by more than 0.15 rad, we compare which
quantity is more stable over time (a parked or straight-driving vehicle's **world-frame** heading
should be nearly constant):

| Quantity | std along the track (rad) |
| --- | --- |
| `stored_yaw` | 0.308 |
| `stored_yaw + ψ_ego` | **0.106** |
| Fraction of tracks more stable after adding ψ | **77.2%** |

⇒ the stored value is **ego-frame**, and `frame_bbox`'s subtraction is spurious. The NAVSIM backend
added this round treats it as ego-frame (no subtraction).

---

## 3. Results

### 3.1 DDv2 with lidar occlusion added: verdicts unchanged in all corpora, but **the leak differs by two orders of magnitude across corpora**

**Table 1. DDv2's F-3 readouts by corpus × occlusion channel. $b_{ghost}=v_{plan}(\text{ghost})-v_{plan}(\text{clean})$; $R=1-b_{occ}/b_{ghost}$.**

| Corpus | Occlusion channel | n | $b_{ghost}$ | $b_{occ}$ | $R$ | $R_{ctrl}$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | RGB only (old) | 282 | −0.180 [−0.349, +0.003] | −0.206 [−0.371, −0.029] | — | — | indeterminate (no baseline response) |
| G1 | **+lidar m=0.25** | 282 | −0.180 [−0.349, +0.003] | −0.201 [−0.366, −0.026] | — | — | indeterminate (no baseline response) |
| Lead-brake | RGB only (old) | 90 | −0.105 [−0.475, +0.210] | −0.329 [−0.777, +0.056] | +0.686 [−0.829, +2.146] | +0.075 [−0.261, +0.425] | indeterminate (no baseline response) |
| Lead-brake | **+lidar m=0.25** | 90 | **−0.105 [−0.475, +0.210]** | −0.404 [−0.844, −0.023] | +1.088 [−0.646, +2.901] | +0.217 [−0.182, +0.650] | **indeterminate (no baseline response)** |
| Lead-brake | +lidar m=0.00 | 90 | −0.105 [−0.475, +0.210] | −0.386 [−0.827, −0.009] | +1.192 [−0.548, +2.991] | +0.225 [−0.164, +0.643] | indeterminate (no baseline response) |
| NAVSIM | RGB only (old) | 372 | +0.075 [−0.070, +0.218] | +0.053 [−0.089, +0.198] | +0.311 [−0.087, +0.876] | −0.111 [−0.511, +0.225] | indeterminate (no baseline response) |
| NAVSIM | **+lidar m=0.25** | 372 | +0.062 [−0.113, +0.248] | +0.085 [−0.097, +0.272] | **+0.252 [+0.082, +0.464]** | +0.003 [−0.149, +0.136] | **indeterminate (no baseline response)** |
| NAVSIM | +lidar m=0.00 | 372 | +0.062 [−0.113, +0.248] | +0.069 [−0.109, +0.254] | +0.178 [+0.031, +0.366] | +0.027 [−0.126, +0.164] | indeterminate (no baseline response) |

**On lead-brake $b_{ghost}$ is unchanged to the digit** (−0.1049 across all three protocols) —
lidar occlusion acts by construction only on the $occ$/$ctrl$ arms and never touches $ghost$/$clean$;
this is a built-in correctness check on the protocol, and it passes.
**On NAVSIM, however, $b_{ghost}$ did move** (+0.0745 → +0.0619): on that corpus the `ego_points`
loading path is traversed for the $ghost$/$clean$ arms too, and adding the 3D-box backend changed
`NavsimLidar`'s consumption of randomness. The measured shift of 0.0126 lies within this
candidate's run-to-run variation and **does not affect the verdict**. It is recorded rather than
smoothed over.

**Table 2. Magnitude of the lidar leak: the same deletion rule differs by two orders of magnitude across corpora.**

| Corpus | Target class | points deleted per event, occ (mean/median) | ctrl (mean/median) | events where nothing was deleted |
| --- | --- | --- | --- | --- |
| G1 (class A, VRU) | pedestrian / cyclist | 6.2 / **1** | 4.0 / 0 | **42%** |
| Lead-brake (class LB) | lead vehicle | 30 / **8** | 21 / 6 | — |
| NAVSIM (class A, VRU) | pedestrian / cyclist | **168 / 81** | 49 / 14 | — |

**This is the most valuable number of the round.** The previous round's G1 conclusion was that
"the leak is real but its exposure is small (median 1 point; 42% of events lose none)". Moving to
NAVSIM, **the same object class (VRU) under the same deletion rule goes from a median of 1 point to
81 (81×)**. The cause is the data source: nuScenes uses a single 32-beam lidar and class-A VRUs sit
at a median longitudinal distance of 27.1 m, whereas NAVSIM/OpenScene point clouds are far denser.
⇒ **"the leak barely matters" does not transfer across data sources**; changing corpus requires
re-measuring the leak, which is exactly why this work order demanded the rerun.

**An incidental improvement**: after closing the lidar channel on NAVSIM, $R$'s CI narrows from
[−0.087, +0.876] (half-width 0.48) to **[+0.082, +0.464] (half-width 0.19)**, and $R_{ctrl}$ moves
from −0.111 to +0.003 (closer to the 0 it ought to be). That is, **closing the leak made the
necessity ratio itself a cleaner readout** — the leak had been injecting extra variance into the
$occ$ arm. The verdict is still blocked at the $b_{ghost}$ gate, so it remains "indeterminate".

**The two arms' deletion counts are of the same order but not equal** (NAVSIM: occ median 81 vs
ctrl 14). The mirrored 3D box often lands on empty roadway; that is a geometric fact. **We do not
hand-pick control positions to match the deletion counts** — doing so would let point-cloud density,
rather than geometric symmetry, determine where the control box goes.

### 3.2 The two VLAs with per-frame occlusion: verdicts unchanged, and the miss rate on lead-brake is again close to 100%

**Table 3. Lead-brake corpus: VLA candidates before and after the fix (same event set).**

| Candidate | Version | n | $b_{ghost}$ | $b_{occ}$ | $R$ | $R_{ctrl}$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Alpamayo-R1 | old (last frame only) | 51 | −0.0728 [−0.2374, +0.0894] | −0.146 [−0.361, +0.051] | −0.260 [−0.858, +0.182] | +0.116 [−0.045, +0.303] | indeterminate (no baseline response) |
| Alpamayo-R1 | **new (per-frame)** | 51 | **−0.0728 [−0.2374, +0.0894]** | −0.226 [−0.415, −0.055] | +0.433 [−0.105, +0.987] | **+0.394 [+0.085, +0.778]** | **indeterminate (no baseline response)** |
| AutoVLA | old (last frame only) | 88 | +0.0436 [−0.0094, +0.1395] | +0.045 [−0.009, +0.141] | +0.140 [−0.014, +0.402] | +0.059 [−0.123, +0.303] | indeterminate (no baseline response) |
| AutoVLA | **new (per-frame)** | 90 | +0.0390 [−0.0139, +0.1344] | +0.128 [+0.004, +0.318] | +4.42 [+0.067, +13.33] | +0.228 [+0.088, +0.392] | **indeterminate (no baseline response)** |

**Measured miss rate of the per-frame projection** (frames that should have been occluded but were
not, in the old version):

| Candidate | Corpus | window frames | frames with entity visible | miss rate |
| --- | --- | --- | --- | --- |
| Alpamayo-R1 | G1 (previous round) | 396 | 396 | **100%** |
| Alpamayo-R1 | **Lead-brake** | 204 | 204 | **100%** |
| AutoVLA | G1 (previous round) | 476 | 409 | 86% |
| AutoVLA | **Lead-brake** | 360 | **359** | **99.7%** |

AutoVLA's miss rate is higher on lead-brake than on G1 (99.7% vs 86%): a lead vehicle is in view
throughout the window and is far less likely than a pedestrian to have "just appeared".
**The discipline "earlier than emergence ≠ not visible" (§FM/A57) holds even more strongly on the
new corpus.**

**The fix did change the occlusion arm, by far more than run-to-run noise** ($b_{ghost}$ is
unaffected by construction, so its non-zero deltas measure the noise floor):

| Candidate | common events | $\Delta b_{ghost}$ (noise floor) | $\Delta b_{occ}$ (fix effect) | ratio |
| --- | --- | --- | --- | --- |
| Alpamayo-R1 | 51 | **0/51 non-zero** (seed locked ⇒ strictly 0) | 51/51 non-zero, mean 0.258 | ∞ |
| AutoVLA | 88 | 3/88 non-zero, mean 0.0043 | 17/88 non-zero, mean 0.0999 | **23×** |

**Three unfavourable or anomalous readings that must be stated:**

1. **After the fix, Alpamayo-R1's control arm $R_{ctrl}$ becomes significantly non-zero**
   (+0.394 [+0.085, +0.778]). The control arm should sit near 0 ("painting elsewhere must not send
   the action back to baseline"). Once the control box is painted on every frame it becomes
   significant, showing that **this candidate reacts to the mere presence of several grey patches**.
   That makes its $R$ uninterpretable even if it were to clear the gate. This problem is **newly
   exposed by the fix**; the old version (one patch on one frame) could not see it.
2. **AutoVLA's $R$ = +4.42 [+0.067, +13.33] is not a usable readout**: only 34 events clear the
   gate and the denominator $b_{ghost}$ sits near 0, so the ratio explodes. Under the existing
   denominator-guard discipline this cell is read only as "indeterminate"; it **must not** be read
   as "necessity is extremely high".
3. **AutoVLA's event count differs between versions (90 vs 88)**: the new run additionally includes
   `scene-1102_000_LB` and `scene-1109_001_LB` (the old run processed 100 events, the new 103; the
   old run's three missing events leave no trace in its log). The per-event comparison above is
   therefore computed **on the 88 common events only**, while the aggregate rows report each
   version on its own event set, with n labelled in both cases.

> **Instrument side.** DDv2's $b_{ghost}$ is unchanged to the digit on lead-brake, confirming on a
> new corpus the constructional property that lidar occlusion touches only occ/ctrl; the 0.0126
> shift on NAVSIM has been traced to randomness consumption in the point-cloud loading path and is
> recorded as such. On the VLA side the ratio of $\Delta b_{occ}$ to $\Delta b_{ghost}$ (∞ / 23×)
> reproduces the magnitudes seen on G1, showing the fix effect sits far above the noise floor.
> **The new NAVSIM 3D-box backend had zero missing boxes across 372 events**, and its orientation
> convention was settled by measurement over 276 tracks rather than by reading code.
> **Specimen side.** **Not one of the six verdicts changed**; all remain "indeterminate: the
> baseline response itself is indistinguishable from 0". Yet the leak magnitude differs 81-fold
> across corpora, and Alpamayo's control arm only revealed a significant response after the fix —
> **"the verdict did not change" is not the same as "the rerun carried no information".**

### 3.3 Verdict summary: all four cells unchanged

| Cell | Verdict before | Verdict after | Changed? |
| --- | --- | --- | --- |
| DDv2 × lead-brake | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| DDv2 × NAVSIM | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| Alpamayo-R1 × lead-brake | indeterminate (no baseline response) | indeterminate (no baseline response) | no |
| AutoVLA × lead-brake | indeterminate (no baseline response) | indeterminate (no baseline response) | no |

**Why it could not have changed (the same argument as last round)**: the gate of the three-state
adjudication is $b_{ghost}$ (the un-occluded arm), and **by construction neither fix touches that
arm** — lidar deletion applies only to $occ$/$ctrl$, and so does per-frame occlusion. All four
cells have a $b_{ghost}$ CI crossing 0, so they are stopped at the first gate and $R$ never enters
the verdict. **The correct reading is "these four cells happen to be stopped at an earlier gate",
not "the defect was harmless"** — had they possessed a baseline response, as LTF does on NAVSIM,
$R$ would have been systematically mismeasured. The narrowing of $R$'s CI half-width on NAVSIM from
0.48 to 0.19 is direct evidence of that.

---

## 4. Incidental finding: auditing the F-3 gate itself (§FC/A61)

Constructing a "hazard absent" multi-frame window required verifying per-frame whether the entity
is visible, which prompted an audit of the **single-frame F-3 clean arm** — and turned up a problem
of wider scope:

> **In 219 of the 288 G1 class-A events (76.0%), the entity is already visible in the clean frame.**
> Across the 218 events visible in both frames, the ghost/clean imaged-area ratio has a median of
> **1.67** and the longitudinal distance moves from a median of **31.3 m to 25.2 m**.
>
> **The lead-braking corpus is more extreme: 103/103 = 100.0%**, with an imaged-area ratio median of
> **1.00** (p25 0.77 / p75 1.22) and longitudinal distance 31.4 m → 31.8 m. This follows directly
> from how that scenario is defined — an **already-tracked** lead vehicle beginning to brake hard —
> so the entity is present throughout at essentially unchanged range, and $b_{ghost}$ there
> contrasts **braking onset** rather than entity presence.

`f3_occlusion_necessity.py` describes the clean arm as "the hazardous entity is simply not present",
but the clean frame is drawn from 1.0–1.5 s before emergence, and `t_emergence` marks "entering the
corridor / crossing the TTC threshold", **not "becoming visible"** — exactly the lesson of §FM/A57,
this time landing on the clean arm.

⇒ $b_{ghost}$ measures **not** "the response to a hazard appearing" but "the response to a hazard
coming roughly 6 m closer", a far weaker stimulus. And the gate of the three-state adjudication is
precisely the significance of $b_{ghost}$.

**A readout that does not depend on the clean arm** (`scripts/f3_within_frame_effect.py`; no forward
pass was rerun, only the stored `per_event` records were re-analysed):

$$d_{occ} = v_{plan}(\text{ghost}) - v_{plan}(\text{occ}), \qquad
  d_{ctrl} = v_{plan}(\text{ghost}) - v_{plan}(\text{ctrl})$$

Both compare **within a single frame** and never pass through the clean arm. Note that
$R = 1 - b_{occ}/b_{ghost} = d_{occ}/b_{ghost}$ — **the original design's numerator already is
$d_{occ}$**; only the denominator (which is also the gate) is contaminated.

**Table 4. Re-analysis of 25 F-3 cells: cells where $b_{ghost}$ said "no baseline response" but the within-frame erasure effect is significant.**
(Sign convention: $v_{plan}$ is commanded speed, so erasing the hazard should make the model
**speed up** ⇒ $d_{occ}$ **negative** is the expected direction.)

| Cell | $b_{ghost}$ | $d_{occ}$ | $d_{ctrl}$ | Conclusion |
| --- | --- | --- | --- | --- |
| Lead-brake × SimLingo | +0.052 [−0.260, +0.396] | **−0.510 [−0.736, −0.307]** | −0.032 [−0.117, +0.048] | **significant, specific, expected direction** |
| Lead-brake × LTF | −0.007 [−0.031, +0.014] | **−0.028 [−0.047, −0.011]** | +0.007 [−0.001, +0.016] | **significant, specific, expected direction** |
| Lead-brake × DDv2 +lidar | −0.105 [−0.475, +0.210] | +0.299 [+0.097, +0.512] | +0.009 [−0.163, +0.181] | significant, specific, but **direction reversed** (erasing the lead vehicle slows it down) |
| Lead-brake × DDv2 (RGB only) | −0.105 [−0.475, +0.210] | +0.224 [+0.050, +0.412] | +0.012 [−0.131, +0.156] | as above |
| Lead-brake × Alpamayo-R1 `_mf` | −0.073 [−0.237, +0.089] | +0.154 [+0.055, +0.269] | +0.092 [+0.008, +0.184] | significant but **not specific** (the control arm is significant too) |

**Of 25 cells, 6 have a non-significant $b_{ghost}$ but a significant $d_{occ}$; 4 of those pass the
ctrl specificity test; only 2 also have the expected direction.**

**What this finding does and does not license:**
* The paper's sentence that on the lead-brake corpus "all six candidates lack a baseline response ⇒
  **untestable**" needs amending: at least SimLingo and LTF **do exhibit a significant, specific,
  correctly-signed action response to erasing the lead vehicle** on that corpus. That response was
  simply never measured by the $b_{ghost}$ gate, because the gate asks a different question.
* **But $d_{occ}$ cannot be used directly as F-3's primary verdict.** F-3's PASS threshold is
  "$R$'s CI lower bound > 0.5", i.e. erasure must remove **more than half** of the original
  response — which requires a meaningful denominator. $d_{occ}$ answers only "does erasure have an
  effect", not "what fraction of the original response does it account for". This report therefore
  positions it as **supplementary evidence, independent of the clean arm, for cells whose primary
  verdict is "no baseline response"** — it does not overturn or replace the primary readout.
* **The reversed direction on lead-brake × DDv2** (erasing the lead vehicle **lowers** planned speed
  by 0.30 m/s, and specifically so) is a lead worth pursuing on its own rather than noise: it
  suggests the candidate treats "there is a vehicle ahead" as some kind of drivability signal. Not
  pursued this round; registered as unfinished.

---

## 5. Record of self-correction

1. **Treating NAVSIM's 3D-box orientation as world-frame, as on nuScenes, was wrong in the first
   version.** Before switching to ego-frame I settled it by measurement (276 vehicle tracks,
   std 0.308 → 0.106, 77.2% agreement) rather than **concluding from code reading**. I also
   quantified the defect's impact on the **already published** NAVSIM 2D boxes: for class A (all
   VRU) the IoU median is **0.899**, area ratio 0.996, centre shift 0.2 px ⇒ **the impact is small
   and the published NAVSIM F-3 numbers need not be withdrawn**; but the same measurement gives a
   **vehicle-class** IoU median of only **0.779** with a median centre shift of **9.9 px** ⇒ any
   vehicle-targeted NAVSIM readout would be materially affected. Both numbers are reported, not just
   the favourable one.
2. **The first version of the multi-frame clean-side window required "entity not visible", which
   discarded 4 of 4 events.** Investigation showed this stems from the semantics of the clean arm
   itself (76% of events already show the entity), not from a bug in the window logic. The rule was
   changed to match the single-frame protocol, and the finding was written up separately as §4.
   **The condition was not quietly relaxed to make the window run** — the reason for relaxing it
   became a finding in its own right.
3. **DDv2's $b_{ghost}$ shifted by 0.0126 on NAVSIM**, apparently contradicting the claim that lidar
   occlusion cannot touch the ghost arm by construction. The cause has been traced to randomness
   consumption in the point-cloud loading path; the magnitude lies within this candidate's
   run-to-run variation and does not affect the verdict. It is **written under Table 1 rather than
   erased**.
4. **AutoVLA's 90 vs 88 event counts are not fully explained** (the old run processed three fewer
   events, with no trace in its log). The per-event comparison is confined to the 88 common events
   and the aggregate rows label their own n; **the two event sets were not silently pooled**.
5. **Neither VLA was run on NAVSIM** — a corpus-side interface gap (dependence on the nuScenes
   devkit and `sd_token`), which the work order also explicitly skips. **No strained approximate
   adapter was built merely to fill the cell.**

---

## 6. Unfinished items

| Item | Status | Reason |
| --- | --- | --- |
| The reversed erasure effect on lead-brake × DDv2 | not pursued | needs a separate mechanistic experiment (why erasing the lead vehicle slows planning); outside this work order |
| Rebuilding F-3's gate around $d_{occ}$ | not done | a methodological change rather than a "rerun on another corpus"; requires first settling what the denominator should be |
| Recomputing published vehicle-class NAVSIM readouts | not done | this round's NAVSIM F-3 positives are all VRU (small impact); vehicle-class readouts live in other experiment lines and need separate scheduling |
| Both VLAs × NAVSIM | n/a | corpus-side interface gap |
