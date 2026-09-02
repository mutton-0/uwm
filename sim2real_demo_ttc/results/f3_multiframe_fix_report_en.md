# F-3 multi-frame occlusion bug: fix, re-run, and old-vs-new comparison

> Work order: [`../docs/f3_multiframe_occlusion_fix_workorder.md`](../../docs/f3_multiframe_occlusion_fix_workorder.md).
> Autonomous decisions: [`amendments.md`](amendments.md) §FM/A56–A58.

---

## 0. Direct conclusion: the bug is real and broader than the work order estimated, but it changed no verdict on this sample

| Question | Answer |
| --- | --- |
| Is the bug real? | **Yes, and both candidates are affected** |
| How large is the exposure? | Alpamayo **396/396 frames** (100%) should have been occluded and were not; AutoVLA **409/476** (86%) |
| Did any verdict change after the fix? | **No. Both candidates remain "indeterminate: the baseline response itself is indistinguishable from 0"** |
| So was the fix pointless? | **No.** The fix touches $b_{occ}$, whereas these two cells are gated on $b_{ghost}$ (the un-occluded arm) — the fix **cannot by construction** change them. Had these candidates possessed a baseline response, $R$ would have been systematically mismeasured. This is a **correctness defect that this particular sample happens not to expose**, not a harmless one. |

**The work order's estimate for AutoVLA needs correcting.** It inferred from `t_emergence` that
AutoVLA's earlier frames were probably before emergence and therefore could not show the entity.
**Per-frame projection refutes this**: on average **3.44 of AutoVLA's 4 window frames** project the
entity, and 85 of 119 events have all 4 visible. The reason is that **`t_emergence` marks "enters the
corridor / crosses the TTC threshold", not "becomes visible"**: a pedestrian is typically already
clearly visible on the sidewalk before entering the corridor. **"Earlier than emergence" ≠ "not
visible"** — a distinction worth recording as discipline in its own right (§FM/A57).

---

## 1. What the fix does

**Old implementation**: build the model's true multi-timestep input window, then paint out the
entity's box only in the **last frame** ($t_0$), leaving earlier frames untouched.

**New implementation** (`scripts/f3_window_boxes.py` + `f3_occlusion_vla.py`): for **every timestamp
in the window**, independently determine whether the entity is visible and occlude it if so.

* **The projection logic is reused from mining, unchanged**: `g1_mine_events.compute_scene_geometry`
  + `frame_bbox`. Both VLAs consume **genuine nuScenes CAM_FRONT images** (Alpamayo via nearest
  neighbour, AutoVLA by walking the sample `prev` chain), and the mining grid is exactly that scene's
  full set of CAM_FRONT frames, so "window timestamp → grid index" is an **exact match, not an
  interpolation**; any deviation beyond 60 ms is counted as a failure (measured `frames_off_grid` = 0).
* **Each frame is filled with its own mean colour**, not with $t_0$'s.
* **A frame in which the entity is genuinely not visible is left untouched** (projected out of frame,
  negative depth, or unannotated) — no extrapolation, no approximation; counted as `frames_no_box`.
* **The control arm is likewise occluded frame by frame** with the same control box, keeping "number
  of occluded frames" comparable between the entity and control arms.
* **A new guard**: if no frame in an event's window projects the entity at all, that event's occ arm is
  a no-op, so it is recorded as `occ_no_frame_hit` and **excluded from the statistics** (measured: 0
  for both candidates).
* Alpamayo's two front cameras (the 120° wide and 30° tele positions, rows 1 and 3 after sorting
  `camera_indices`) are **both occluded, frame by frame**.

**Measured per-frame projection statistics**:

| Candidate | Window frames | Visible and occluded | Genuinely invisible | Timestamp mismatch | Frames occluded per event |
| --- | --- | --- | --- | --- | --- |
| Alpamayo-R1 | 396 | **396 (100%)** | 0 | 0 | mean **4.00** (all 99 events at 4) |
| AutoVLA | 476 | **409 (86%)** | 67 | 0 | mean **3.44** (4 frames ×85 / 3 ×14 / 2 ×7 / 1 ×13) |

**That is: under the old implementation, Alpamayo left 3 frames per event and AutoVLA an average of
2.44 frames still showing the hazard entity.**

---

## 2. Old vs new

**Table 1. G1 corpus, four-arm readouts and verdicts (scene-level bootstrap). Event counts match the old runs, so the cells compare directly.**

| Candidate | Version | $b_{ghost}$ | $b_{occ}$ | $b_{ctrl}$ | $R$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **Alpamayo-R1** (n=99) | old (last frame only) | +0.0328 [−0.1228, +0.1907] | +0.0270 [−0.1207, +0.1832] | +0.0317 [−0.1290, +0.1956] | +0.028 [−0.127, +0.180] | indeterminate (no baseline response) |
| **Alpamayo-R1** (n=99) | **new (per-frame)** | +0.0328 [−0.1228, +0.1907] | +0.0345 [−0.1090, +0.1805] | +0.0506 [−0.0780, +0.1837] | **−0.033 [−0.290, +0.188]** | **indeterminate (no baseline response)** |
| **AutoVLA** (n=119) | old (last frame only) | −0.0551 [−0.2194, +0.0277] | −0.0383 [−0.2065, +0.0608] | −0.0590 [−0.2213, +0.0223] | −0.590 [−1.607, +0.033] | indeterminate (no baseline response) |
| **AutoVLA** (n=119) | **new (per-frame)** | −0.0558 [−0.2206, +0.0275] | −0.0337 [−0.2018, +0.0665] | −0.0575 [−0.2171, +0.0246] | **−0.433 [−1.464, +0.193]** | **indeterminate (no baseline response)** |

**Verdicts: identical before and after the fix for both candidates — "indeterminate: the baseline
response itself is indistinguishable from 0".**

### 2.1 Why the verdicts could not have changed (stating this matters, or "no change" will be misread as "the bug was harmless")

The three-state gate is $b_{ghost}$: **a necessity test presupposes a response**. And $b_{ghost}$ is
the **un-occluded arm** (clean vs ghost), which **the fix does not touch at all**. Both candidates'
$b_{ghost}$ CIs span 0, so adjudication stops at that gate and $R$ never enters it.

**"The verdict did not change" therefore cannot be read as "the bug was harmless"**, only as **"these
two candidates happen to be stopped at an earlier gate, so the segment the bug affects was never
used"**. Had they possessed a baseline response like LTF, $R$ would have been systematically
mismeasured.

### 2.2 The fix did change the occlusion arm, by far more than run-to-run noise

$b_{ghost}$ should be unaffected by the fix by construction, so **its measured non-zero movement is a
ready-made run-to-run noise floor**:

| Candidate | $\Delta b_{ghost}$ (noise floor) | $\Delta b_{occ}$ (fix effect) | Ratio |
| --- | --- | --- | --- |
| Alpamayo-R1 | non-zero **0/99**, mean \|Δ\| **0.0000** | non-zero **99/99**, mean \|Δ\| **0.1144** (max 1.497) | **∞** (noise floor is 0) |
| AutoVLA | non-zero 3/119, mean \|Δ\| 0.0012 | non-zero 24/119, mean \|Δ\| **0.0196** (max 0.271) | **16.3×** |

* **Alpamayo's noise floor is exactly 0** (the protocol locks both conditions and every patch run to
  one seed, §CE/A39), so all 99/99 non-zero $\Delta b_{occ}$ values are genuine fix effects.
* **AutoVLA has a small noise floor** (3/119 events' $b_{ghost}$ moved, max 0.103): its decoding is
  `temperature=1e-4 + top_k=1`, **close to but not** bit-deterministic, and floating-point differences
  between runs occasionally flip a token. The fix effect is still **16×** larger, so it is not drowned
  by noise. This noise floor had never been quantified before; it falls out of this round and is
  recorded in §FM/A58.

**$R$'s point estimate did shift materially** (Alpamayo +0.028 → −0.033; AutoVLA −0.590 → −0.433), but
both CIs remain wide and span 0.5, so even with the gate lifted they would still be indeterminate.

---

## 3. Audit: do the single-frame candidates have a comparable leak? (work order §1.5)

| Candidate | Auxiliary input | A leak? | Conclusion |
| --- | --- | --- | --- |
| SimLingo / DD / LTF | `spd(ev)` = mean ego speed from `x_clean_frames` | **No** | A scalar, taken from the **clean** side, carrying no visual information, and shared by all three condition arms ⇒ it cannot carry entity information |
| SimLingo / DD / LTF | image | **No** | A single-frame model consumes one image; occluding it removes the entity entirely, and there is no "history frame" channel |
| **DiffusionDriveV2** | **real lidar point cloud** (`lidar.ego_points(sd_token)`) | **Yes, and not yet fixed** | see below |

**DiffusionDriveV2 has a leak of the same nature through a different channel**: F-3's occlusion paints
only the **RGB image**, while DDv2 simultaneously consumes the ghost frame's **real lidar point cloud**
(§CE/A29 and §CE/A33 require feeding real lidar). **The pedestrian remains fully present in the point
cloud**, so for DDv2 "occlusion" has never removed the entity from the input — only from the camera.

**Current impact**: DDv2's F-3 verdict is "indeterminate: no baseline response" (consistently across
all three corpora), which — as with Alpamayo/AutoVLA — is gated at $b_{ghost}$, so **present
conclusions are unaffected**. **But this is a correctness defect of the same nature as this work
order's and must be registered** (§FM/A58): the correct treatment is to also delete (or replace with
ground points) the points falling inside that 3D box. This work order's scope covers multi-frame
occlusion only, so it was not done here and is listed as an open item.

---

## 4. Audit: the assumption that `x_{cond}_frames[0]` is the decision frame (work order §0, final paragraph)

| Check | Result |
| --- | --- |
| Does that frame always exist? | All 288 A-class events have `x_ghost_frames[0]` with a bbox ⇒ **existence holds** |
| Is it always usable? | **No.** Alpamayo's `covers()` prefilter cuts 288 → **186** (the 6.4 s future trajectory cannot be constructed), and `--limit 100` then truncates to 99. **"This frame is unusable" is a real, recurring situation, not a hypothetical**; the existing code handles it by skipping and counting, which is the correct discipline. |
| Is taking $t_0$ as the window end reasonable? | Both adapters' windows already end at the target frame (Alpamayo's `img_ts` ends at `t0_us`; AutoVLA walks back along `prev`), so this **matches the models' real input** and is not an assumption we imposed. |

---

## 5. Limitations

1. **Re-run on the G1 corpus only**, not extended to lead braking / NAVSIM (work order §2.3 explicitly
   does not require it). Note that Alpamayo / AutoVLA are already n/a on NAVSIM (their adapters depend
   on nuScenes-format data).
2. **The `--limit` caps were kept** (Alpamayo 100 → 99 events, AutoVLA 120 → 119), so this is not a
   full-corpus run. It matches the old runs cell for cell, but the sample is not large enough to move
   $R$ from "indeterminate" to a definite verdict.
3. **DDv2's lidar-channel leak is unfixed** (§3) and is listed as an open item.
4. **No "average over repeated multi-frame inferences" denoising** was done (work order §2.3 leaves
   this to the next work order). AutoVLA's run-to-run noise floor is now quantified (mean \|Δ\| 0.0012)
   and can serve as that work order's input.
5. **Alpamayo's sampling noise** (`top_p=0.98, T=0.6`) remains: this round drives it to 0 by locking
   the seed, but that guarantees reproducibility, not robustness (already registered in §CE/A39).


---

# Addendum: closing DiffusionDriveV2's lidar channel (§FM/A59)

> Follows the open item registered in §3 of this report. No old result file was deleted; the new runs
> are saved as `f3_occlusion_ddv2_lidar.json` (primary) and `f3_occlusion_ddv2_lidar_m0.json`
> (sensitivity).

## A.0 Direct conclusion

| Question | Answer |
| --- | --- |
| Is the lidar channel closed? | **Yes**, with symmetric point removal on both the occ and ctrl arms |
| Did the verdict change? | **No.** Still "indeterminate: the baseline response itself is indistinguishable from 0" |
| Did $b_{ghost}$ / $R$ change? | $b_{ghost}$ is **unchanged to the digit** (−0.1803; unaffected by construction); $R$ still does not enter the verdict |
| **A correction to my own previous claim** | Last round's §FM/A58 stated "**the pedestrian remains fully present in the point cloud**". **That was an overstatement.** Measured on G1's A-class events, the entity's 3D box contains a **median of 1 point**, and **42% of events contain none at all** |

## A.1 What the fix does (symmetric across arms)

* **occ arm**: delete points falling inside the entity's **ego-frame 3D box**. That 3D box is
  **the same source** as the RGB side's 2D box — same `p_ego` centre, same `_size`, same
  nearest-keyframe orientation — so "deleting points" and "painting the box" act on the same physical
  entity rather than on two independent approximations.
* **ctrl arm**: delete points inside the **mirrored 3D box** ($y \to -y$, $yaw \to -yaw$ about the
  ego longitudinal axis). Strictly equal volume at the same longitudinal range — the same geometric
  construction as the RGB control box's horizontal mirror. **Not occ-only**, otherwise "extent of
  removal" would not be comparable between the arms.
* **Margin = 0.25 m** on each axis. **This was not tuned to delete more points**: nuScenes
  annotation/calibration error is roughly 0.1–0.3 m and points were measured falling 5 cm outside the
  box; the 2D side's `bbox_to_tokens` already dilates by 1 token. **Margin = 0.0 is reported alongside
  as a sensitivity.**

## A.2 Three configurations compared (G1, n = 282 / 147 scenes)

| Configuration | $b_{ghost}$ | $b_{occ}$ | $b_{ctrl}$ | Verdict |
| --- | --- | --- | --- | --- |
| RGB only (old) | −0.1803 [−0.3487, +0.0032] | −0.2060 [−0.3707, −0.0290] | −0.1767 [−0.3461, +0.0074] | indeterminate (no baseline response) |
| **RGB + lidar, m = 0.25 (primary)** | **−0.1803 [−0.3487, +0.0032]** | −0.2009 [−0.3658, −0.0256] | −0.1745 [−0.3435, +0.0097] | **indeterminate (no baseline response)** |
| RGB + lidar, m = 0.00 (sensitivity) | −0.1803 [−0.3487, +0.0032] | −0.2014 [−0.3659, −0.0256] | −0.1779 [−0.3469, +0.0061] | indeterminate (no baseline response) |

**All three configurations give the identical verdict.** As with the multi-frame fix, the gate is
$b_{ghost}$ (the un-occluded arm), and closing the lidar channel affects only $b_{occ}$ / $b_{ctrl}$,
so it **cannot change the verdict by construction**.

## A.3 Fix effect vs noise floor

$b_{ghost}$ is unaffected by construction, so its measured movement is the run-to-run noise floor:

| Configuration | $\Delta b_{ghost}$ (noise floor) | $\Delta b_{occ}$ (lidar-closure effect) | Events with $b_{occ}$ entirely unchanged |
| --- | --- | --- | --- |
| m = 0.25 | **0/282 non-zero, mean 0.0000** | 104/282 non-zero, mean **+0.0051 [−0.0036, +0.0145]** | **178/282** (no points in the box to begin with) |
| m = 0.00 | 0/282 non-zero, mean 0.0000 | 68/282 non-zero, mean +0.0046 [−0.0028, +0.0131] | 214/282 |

* **DDv2 is the only bit-deterministic candidate of the three F-3 subjects** (noise floor exactly 0; no
  sampling in decoding);
* **but $\Delta b_{occ}$'s CI spans 0**, so the behavioural change from closing the lidar channel is
  **indistinguishable from zero**.

## A.4 Why the effect is so small: the entity is barely in the point cloud to begin with

| Quantity | Value |
| --- | --- |
| Entity longitudinal range, percentiles (10/25/50/75/90) | 17.3 / 20.8 / **27.1** / 39.0 / 49.4 m |
| Points removed, occ arm (m = 0.25) | mean **6.2**, **median 1** |
| Points removed, ctrl arm (m = 0.25) | mean 4.0, median 0 |
| Points removed, occ arm (m = 0.00) | mean 2.1, median 0 |

nuScenes uses a 32-beam lidar, and the A-class VRUs sit at a median range of **27 m**, where a
pedestrian returns only a handful of points and frequently none. **The real exposure of this "lidar
leak" on this corpus is therefore far smaller than I asserted last round.**

**The fix is still necessary**, for the same reasons as the multi-frame one: (i) it is a correctness
issue and should not rest on the accident that this corpus happens to be point-sparse; (ii) on
close-range events or a higher-beam sensor (e.g. NAVSIM/OpenScene's MergedPointCloud) the exposure
would immediately grow; (iii) with both arms now symmetric, the readout's meaning is clean.

## A.5 An imperfect match that must be stated

**"Extent of removal" is only of the same order between the arms, not equal**: occ removes a mean of
6.2 / median 1, ctrl a mean of 4.0 / median 0 (m = 0.25). The mirrored volume is strictly equal in
size and range, but it frequently lands on **empty road surface** where there were no points anyway.

**How this is handled**: report both arms' removal distributions honestly, and **do not hand-pick a
control placement whose point count matches** — doing so would let point-cloud density determine the
control box's position and introduce a new selection bias. Under the current convention both
$\Delta b_{occ}$ and $\Delta b_{ctrl}$ are indistinguishable from 0, so this imperfect match does not
affect the conclusion.
