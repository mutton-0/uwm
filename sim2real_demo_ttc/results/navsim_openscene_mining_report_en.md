# NAVSIM / OpenScene independent corpus: feasibility assessment and mining report

> Work order: [`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md).
> Scripts: `scripts/ns1_navsim_geometry.py` (data-access layer + mining), `scripts/n1_match.py` (reused, unchanged).
> Config: `configs/navsim_corpus.yaml`. Core deliverable: [`cross_corpus_generality_report_en.md`](cross_corpus_generality_report_en.md).
> Autonomous decisions: [`amendments.md`](amendments.md) §NS/A45–A48.

---

## Methods

### 1. Feasibility assessment (work order §3's 1-day timebox; actual time < 2 hours)

| Check | Verdict | Evidence |
| --- | --- | --- |
| Raw logs on this machine? | **yes** | `/data/dataset/navsim/dataset/navsim_logs/test/`, 147 `.pkl` files |
| Camera images on this machine? | **yes** | `sensor_blobs/test/<log>/CAM_F0/*.jpg`, 1920×1080, 219 GB total |
| Point clouds on this machine? | **yes** | `<log>/MergedPointCloud/*.pcd` (required by DDv2) |
| Do annotations support the mining criteria? | **yes** | per frame `anns`: `gt_boxes[N,7]`, `gt_names`, **`gt_velocity_3d`**, `track_tokens` |
| Is there a reusable reading interface? | **yes, but not used this round** | `navsim/common/dataclasses.py` (`Scene`/`Frame`/`Annotations`). We read the `.pkl` directly: mining needs only the raw fields, and going through the dataclasses adds a conversion layer |
| Independently collected from nuScenes? | **yes (with one necessary qualification, below)** | nuPlan fleet, four maps: Las Vegas / Boston / Pittsburgh / Singapore |

**Feasibility verdict: pass.** The stop-loss clause was not triggered.

**A qualification on "independence" that must be stated**: nuScenes was collected in **Boston and
Singapore**, and this corpus contains `us-ma-boston` (5337 events) and `sg-one-north` (964 events),
so the two **overlap geographically**. What is independent: a different fleet, a different sensor
configuration (1920×1080 vs 1600×900, different intrinsics and distortion), different collection
times, and a different annotation pipeline. The accurate phrasing is therefore **"independently
collected", not "geographically disjoint"**. We report a sensitivity analysis restricted to **Las
Vegas + Pittsburgh** (geographically disjoint from nuScenes, 8750 events) alongside the main readout,
which uses the full corpus — splitting by map halves the sample and would itself introduce a power
problem.

### 2. Corpus construction: only the data-access layer changed

**No criterion was changed.** `scripts/ns1_navsim_geometry.py` does one thing — turn NAVSIM logs into
the `geo` structure the G1 mining pipeline understands — and then:

```python
evs = G1.detect_events(geo, scene, cfg)      # ← criteria and thresholds unchanged
clean, ghost = G1.pick_frames(geo, t, cfg)   # ← frame windows unchanged
rec[...] = G1.frame_record(geo, j, o)        # ← event-record construction unchanged
```

`configs/navsim_corpus.yaml` derives from `configs/n1_d2.yaml` with **every threshold in the mining
block held identical**; only `work_dir` and the image root differ. **Why thresholds must not be
tuned**: retuning criteria for a new data source would turn "same scenario structure, new data
source" into "new scenario *and* new data source", and the result would no longer be attributable.

Negative matching reuses `scripts/n1_match.py` **without a single change**.

### 3. Data-source differences, checked item by item (the mandatory checklist for cross-source transfer)

| Item | nuScenes | NAVSIM/OpenScene | Handling |
| --- | --- | --- | --- |
| Frame rate | 2 Hz (keyframes) | 2 Hz | identical, nothing to do |
| Object position | global frame → needs world→ego | **already in lidar frame**, and `lidar2ego` is identity (measured: translation [0,0,0], rotation [1,0,0,0]) | used directly |
| Object velocity | **not annotated; obtained by 2 Hz differencing** | **`gt_velocity_3d` given directly** (measured: static objects \|v\| ≈ 0.001 m/s ⇒ absolute, global frame) | used directly, rotated into ego frame |
| Ego velocity | position differencing | `ego_dynamic_state[:2]`, **already in ego frame** (cross-checked against differencing: agrees to < 0.15 m/s) | used directly |
| Camera | CAM_FRONT 1600×900, principal point (800, 450), no distortion coefficients | CAM_F0 1920×1080, principal point **(960, 560)**, 5 distortion coefficients | **see §NS/A46**; projection remains pinhole (registered as a deviation) |
| Parked annotation | `vehicle.parked` / `vehicle.stopped` attributes | **no attribute field** | switched to a kinematic definition (§NS/A45) |
| Weather / lighting | parseable from the scene description | **not annotated** | `is_night` / `is_rain` set False with the gap declared |
| Categories | fine-grained (`human.pedestrian.adult`, …) | coarse, 7 classes | mapping below; native names retained in `object_class_native` |

**Category mapping** (downstream code selects on nuScenes prefixes, so a mapping is required; native
names are not lost):

| NAVSIM native | Mapped to | G1 class |
| --- | --- | --- |
| `pedestrian` / `bicycle` | `human.pedestrian.adult` / `vehicle.bicycle` | **vru** |
| `vehicle` | `vehicle.car` | vehicle |
| `traffic_cone` / `barrier` / `czone_sign` | `movable_object.*` | **static** |
| `generic_object` | `static_object.generic` | **static** |

### 4. The cost reversal: confirmed, but it does not come from the model side

The work order expected that "DiffusionDrive/LTF/DDv2 are natively trained on NAVSIM, so feeding them
this data should be smoother than feeding them nuScenes was". **The cost did reverse — but the
reversal happened on the data side, not the model side.**

* **Nothing was saved on the model side.** The three adapters consume **raw RGB plus ego speed**, which
  is data-source agnostic; wiring them to nuScenes and to NAVSIM costs the same (namely, nothing).
  "NAVSIM-native" never shows up on the inference path — because we never used NAVSIM's dataloader in
  the first place, we reused our own adapter front end.
* **Three things were saved on the data side**, all in mining: (i) object velocity is given directly
  (on nuScenes we differentiate 2 Hz annotations, which is exactly the source of the 32 artifacts with
  \|a\| > 10 m/s² in §LB/A44); (ii) positions are already in the ego frame, removing the world→ego
  transform; (iii) the category set natively contains `traffic_cone` / `barrier` / `czone_sign` /
  `generic_object` — precisely the "harmless by category" static material D2a needs, and far richer
  than nuScenes' `movable_object.*`. This shows up directly in the matching quality (see Results).
* **But one cost was added**: the crop principal-point row of §NS/A46. It hides in a constant in the
  **model front end**, and without an item-by-item check of camera intrinsics it silently contaminates
  every readout. **This offsets part of what the data side saved.**

Net conclusion: **the cost of cross-source transfer lies mainly in "checking data-source differences
item by item", not in "whether the model was natively trained on this source".**

---

## Results

### Corpus scale

| Quantity | Value |
| --- | --- |
| logs / scenes | 147 / **1880** (146 further scenes skipped for having < 20 frames) |
| Total events | **15051** (1850 scenes contain events) |
| **A (VRU emergence, positive)** | **397** (163 scenes) |
| B (close cut-in) / C (TTC drop) | 178 / 1111 |
| D2a (geometry-matched statics) | 2725 |
| D2c (same class in corridor, TTC always high) | 2255 |
| Maps | Las Vegas 6744 / Boston 5337 / Pittsburgh 2006 / Singapore 964 (by event count) |
| Native classes of A | pedestrian 371 / bicycle 26 |
| Native classes of D2a | generic_object 1584 / traffic_cone 783 / barrier 338 / czone_sign 20 |

**A = 397 exceeds G1's 291.** This matters directly for this round's core goal: adjudicating LTF's
G-axis positive requires power, and this corpus's power is **not lower** than G1's.

### Negative-matching quality control

**Table 1. N1 caliper matching (log imaged area ≤ 0.15 dex, eccentricity ≤ 0.06) — same convention, same script as G1.**

| Negative | Pool | Pairs | Match rate | SMD area (pre → post) | SMD ecc (pre → post) |
| --- | --- | --- | --- | --- | --- |
| D2a | 2678 | **382** | 99% | +1.29 → **+0.02** | +0.27 → **−0.01** |
| D2b | 3500 | 381 | 99% | −0.02 → +0.01 | −1.43 → +0.00 |
| D2c | 2222 | 383 | 99% | −0.20 → −0.01 | +0.29 → +0.00 |
| **D2cV (falsification floor)** | 915 | **296** | **77%** | −0.03 → **−0.01** | +0.35 → **+0.04** |

All |SMD| ≤ 0.04, **better matched than either the G1 or the lead-braking corpus** (whose LBv reached
−0.328, limited by a pool of only 37). The reason is cost-reversal item (iii): NAVSIM's richer static
categories give D2a a candidate pool of 2678, leaving the caliper ample choice.

**D2cV contributes n = 134 to the analysis** (the matched list of 296 spans both sides and must be
intersected with the cached events). This is the one quantity smaller than G1's, but because A
(397 vs 291) and D2a (382 vs 283) are both larger, the measured CI width of the primary-minus-floor
readout is **0.103**, slightly *narrower* than G1's 0.109 (see the cross-corpus report §2).

---

## Discussion

**The purpose of this report is to nail down comparability.** The usual failure mode of a
cross-source comparison is not statistical but definitional: the two sides end up measuring different
things. We decompose comparability into three claims, each with checkable evidence:

1. **Criteria are comparable**: `detect_events` / `pick_frames` / `frame_record` are all imported from
   `g1_mine_events` with thresholds untouched; negative matching reuses `n1_match.py` unchanged.
2. **Geometry is comparable**: camera intrinsics and extrinsics were checked item by item and the
   principal-point discrepancy corrected (§NS/A46); the distortion difference is registered as a
   deviation (unbiased between groups; absolute values not comparable across corpora).
3. **Statistics are comparable**: scene-level bootstrap, pre-registered primary readout, three-state
   adjudication and the 10-seed fold-assignment stability check all carry over, and this round's CI
   width matches G1's (0.103 vs 0.109), which is what makes "non-replication" separable from
   "insufficient power".

**One incomparability recorded honestly**: the **absolute values** of imaged area and eccentricity are
not comparable across corpora (different resolution, principal point, and pinhole projection despite
NAVSIM's distortion). This affects none of this round's readouts — all of them are within-corpus
between-group contrasts, and the geometric matching was performed within each corpus.

---

## Self-correction record

1. **§NS/A46 is the most dangerous item this round, and it raises no error.** `CROP_CENTER_ROW = 450`
   sits in the model front end; its semantics are "the camera's principal-point row", but its value
   was hard-coded to nuScenes'. Keeping it would have fed all three models a band shifted 110 px
   upward, silently depressing every readout. **The lesson generalizes**: cross-source transfer
   requires an item-by-item check of camera intrinsics, because "constants that look like generic
   preprocessing" frequently encode the original source's geometry.
2. **§NS/A45**: NAVSIM has no attribute annotations, so `is_parked` was redefined kinematically. This
   is a definitional deviation rather than an equivalent substitution; its scope is declared (it feeds
   only D2aP, never the primary readout).
3. **"Independently collected" is not "geographically disjoint"**: this corpus includes Boston and
   Singapore, overlapping nuScenes. This is stated explicitly in Methods §1, with a Las Vegas +
   Pittsburgh sensitivity analysis provided. **We do not write "independent data source" as a stronger
   claim than the facts support.**
