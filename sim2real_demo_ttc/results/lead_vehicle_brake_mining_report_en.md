# Second scenario type — mining report: lead-vehicle braking (LB)

> Work order: [`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md), task two.
> Scripts: `scripts/lb1_mine_lead_brake.py` (candidate scan), `scripts/lb2_build_events.py` (thresholds, matching, serialization).
> Config: `configs/lead_brake.yaml`. Numeric artifacts: `variants/lead_brake/mining/lead_brake_{survey,mining_stats}.json`.
> Autonomous decisions: [`amendments.md`](amendments.md) §LB/A42–A43.

---

## Methods

### Why this scenario

G1's three positive trigger criteria (A: VRU emergence, B: close cut-in, C: generic TTC drop) all
share **one causal structure**: "a new entity suddenly enters or approaches the path". Here the
hazard originates in a **state change of an already-tracked entity** — a lead vehicle that has been
visible all along suddenly brakes hard. This tests whether the **definitions** of G/F/I/C transfer,
not whether one set of thresholds transfers.

### Reuse, not rebuild

All geometry (corridor test, ego-frame projection, imaged area / eccentricity, visibility, 2D boxes)
is **imported directly** from `g1_mine_events.compute_scene_geometry` and `frame_record`, unchanged;
clean/ghost frame selection reuses `pick_frames` with windows identical to G1 field for field
(clean = [−1.5, −0.5] s, ghost = [0.0, 1.0] s, 2 frames per condition). The `mining:` block of
`configs/lead_brake.yaml` is field-identical to `configs/n1_d2.yaml`; only a `lead_brake:` block is
added. **No new tooling was introduced for this scenario.**

### Event families (in one-to-one correspondence with G1's A / D2a / D2cV)

| Family | Definition | G1 counterpart |
| --- | --- | --- |
| **LB** | An in-corridor, already-tracked lead vehicle undergoes a large deceleration within the observation window. clean = the steady-following window before brake onset; ghost = the window after it | A (positive) |
| **LBn** | Also an in-corridor lead vehicle, caliper-matched on **distance / lead speed / ego speed**, with no deceleration in the window (steady following) | D2a (geometry-matched negative) |
| **LBv** | Also an in-corridor lead vehicle, matched on the same three variables, **decelerating but only mildly** | **D2cV (falsification floor)** |

Lead-vehicle criteria: class in `vehicle.{car,truck,bus,trailer,construction,emergency}`, not parked,
inside the corridor, camera-visible, $d_{long}$ > 2 m, own speed > 1 m/s — and all of these must hold
**throughout both the clean and ghost windows**, otherwise it is not the same causal structure
(a target that leaves the corridor mid-window degenerates into "entity departs", a different
question). Deceleration is the central difference of the target's **own speed** (not relative speed),
so it is decoupled from ego acceleration. Only the strongest-deceleration onset frame is kept per
target, so one braking manoeuvre is not split into several events.

### Why LBv is this scenario's falsification floor

D2cV's logic: hold "same VRU class + same imaged geometry" fixed so the only difference is
**relative velocity** — a quantity a single-frame model is **structurally unable** to observe. Any
claimed "hazard discriminability" that cannot be separated from it is not evidence of a hazard
concept.

LBv is isomorphic rather than a copy: it holds "also a lead vehicle + same distance + same lead speed
+ same ego speed + **also decelerating**" fixed, so the only difference is the **magnitude of the
deceleration** — again a quantity that requires inter-frame differencing. A single-frame model can
see following distance, vehicle type and brake lights; it cannot see deceleration.

### Brake lights: an adaptation deviation that must be reported with the results (§LB/A43)

The work order asks us to design this "according to whether brake-light annotation is available".
**nuScenes has no brake-light annotation**, so matching on brake-light state is impossible. We use a
stronger substitute construction instead: **LBv's negatives are themselves decelerating** (only
mildly), so brake lights are **likely on in both arms** and the difference is compressed onto
deceleration magnitude alone.

**The cost must be stated**: we **cannot verify that the brake lights are actually on.** This is
recorded as a limitation, not as a verified control. If a model were merely "reading brake lights",
this design **tends to** expose that (LB and LBv have similar brake-light states, so a
brake-light-driven model should score near 0.5 on that pair) — but that inference rests on the
unverified assumption above.

---

## Results

### Threshold-callback record (set from the distribution plus physical plausibility, not by fiat)

A full scan over all 850 nuScenes trainval scenes yields **302 candidate lead-vehicle windows across
228 scenes**. Percentiles of the strongest in-window deceleration $a_{\min}$ (m/s², negative =
decelerating):

| p1 | p2 | p5 | p10 | p25 | p50 | p75 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| −27.98 | −16.83 | −13.67 | −10.20 | −5.72 | −2.36 | −0.10 | −0.01 |

**Callback #1 (positive upper bound).** Taking a tail percentile directly (e.g. p10 = −10.2) would
admit samples with $a$ < −10 m/s². **The physical braking limit of a road vehicle is roughly
−8 to −9 m/s²** (emergency braking on dry asphalt), so the **32 samples with $a$ < −10 are judged
artifacts of nuScenes' 2 Hz annotation differencing and dropped**, not treated as "more severe
positives". This mirrors G1's own distribution-driven threshold callbacks, with one addition: this
scenario also has a **physical** upper bound.

**Callback #2 (positive lower bound).** $a \le -3.0$ m/s², between p25 (−5.72) and p50 (−2.36).
−3 m/s² is a clearly perceptible deceleration (≈ 0.3 g) and yields 103 positives, matching the work
order's "first produce a Tier-S batch (tens of events)".

**Callback #3 (LBv floor band).** $-1.5 \le a \le -0.3$ m/s². The lower bound of −0.3 guarantees the
vehicle **is actually decelerating** (so brake lights have a reason to be on); the gap between −1.5
and the positives' −3.0 is a 1.5-wide buffer preventing the two families from bleeding into each
other at the boundary. The 45 samples in the [−3.0, −1.5) grey band **enter neither family**.

**Callback #4 (LBn steady following).** $a \ge -0.2$ m/s². The 5 samples in the (−0.3, −0.2) grey
band are excluded.

| Family | After thresholds | After caliper matching | Match rate |
| --- | --- | --- | --- |
| LB | 103 | 103 (positives are not matched) | — |
| LBn | 80 | **48** | 48/103 |
| LBv | 37 | **31** | 31/103 |

**182 events across 154 scenes** are written: LB = 103, LBn = 48, LBv = 31. The schema is
**field-identical** to G1's `events_all.jsonl`, so all three downstream adapters (SimLingo,
DiffusionDrive, LTF) and every analysis script are reused without modification.

### Negative-matching quality control

**Table 1. Standardized mean differences (SMD). Convention: |SMD| < 0.1 is good, < 0.25 acceptable. $a_{\min}$'s SMD **should** be large — it is the one manipulated variable.**

| Variable | LB vs LBn | LB vs LBv |
| --- | --- | --- |
| $d_{long}$ (longitudinal distance) | **−0.152** | −0.328 |
| Lead speed | **+0.026** | −0.291 |
| Ego speed | +0.239 | **−0.205** |
| Imaged area | +0.216 | +0.253 |
| **$a_{\min}$ (manipulated)** | **−4.000** | **−3.319** |

**LBn is well matched** (two of the three matching variables at |SMD| < 0.25; ego speed at 0.239 sits
at the edge of acceptable).

**LBv's matching is the binding limitation of this first batch**: $d_{long}$ (−0.328) and lead speed
(−0.291) exceed 0.25. The cause is that **the negative pool itself is too small** — the whole
trainval set contains only 37 candidates in the mild-deceleration band, 31 of which fall inside the
caliper, so the assignment is essentially forced rather than algorithm-limited. A better matching
algorithm does not fix it (we already moved from greedy nearest-neighbour to optimal assignment, see
§LB/A42, and LBv did not move by a single event). **This means LBv has limited power as a
falsification floor, and any adjudication resting on it must be discounted accordingly.**

---

## Discussion

**This batch is positioned as "pipeline closure at Tier-S scale", not as statistical power.** The
work order states explicitly: "first produce a Tier-S batch, verify the pipeline closes, do not chase
full statistical power in one pass". LB = 103 / LBn = 48 / LBv = 31 against G1's A = 291 / D2a = 283
/ D2cV = 212 is roughly one third to one sixth the sample, so CIs in this scenario are about 1.5–2.5×
wider than G1's. **This must be held in mind when reading the four-axis results**, or "insufficient
power" will be misread as "does not replicate".

**Feasible routes to a larger sample (out of scope this round)**: (i) relaxing the positive threshold
to $a \le -2.0$ (near p50) would raise LB to ≈ 150 but dilutes the semantics of "hard braking";
(ii) LBv is the real bottleneck and can only grow with more corpus (beyond nuScenes) or a wider mild
band, the latter eroding the buffer; (iii) 1:k matching would improve negative utilization but
requires changing the statistical convention (pair weights), which we do not touch this round.

---

## Self-correction record

1. **The first version matched negatives on distance and lead speed only, omitting ego speed**
   (§LB/A42). The measured ego-speed SMD for LB vs LBn was **+0.600** — hard braking by a lead
   vehicle correlates with the ego vehicle travelling fast, and ego speed **strongly drives** every
   model's commanded speed, so without matching it, any G/F difference would be contaminated by
   "this group was simply driving faster". Adding the third matching variable brought it to +0.239.
2. **The matching algorithm was changed from greedy nearest-neighbour to optimal assignment
   (Hungarian)** (§LB/A42). Greedy results depend on traversal order (requiring a random seed) and,
   with a small negative pool, tend to consume good negatives early. Optimal assignment minimizes
   total pair distance, is order-independent and reproducible. LBn rose from 46 to 48 with improved
   SMDs; **LBv did not move at all** — its pool is already exhausted by the caliper, which is itself
   the evidence that LBv's limitation lies in the corpus and not in the algorithm.
3. **A physical plausibility bound is not something percentiles can supply** (callback #1). Had we
   only called back by percentile, the 32 annotation artifacts with $a$ < −10 m/s² would have entered
   LB as "the most severe positives", and their clean/ghost contrasts would have been treated as the
   strongest signal. **We excluded them by the physical braking limit and record the count.**
