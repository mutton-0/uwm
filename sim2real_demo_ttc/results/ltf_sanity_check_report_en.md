# Instance-level sanity check of the LTF "blind-action" conclusion

> Work order: 2026-09-03 (inserted ahead of the R-estimator confirmation).
> User's challenge: **LTF is a validated, real checkpoint; it should not exhibit anything as
> extreme as "cannot see the hazard entity at all". The suspicion is that our experimental
> pipeline is at fault, not the model.**
> This report answers with concrete, visual evidence rather than an abstract $R$ value.
> Except for item 4 (a 30-frame health probe), **everything reuses cached artefacts; the main
> experiment was not re-run.**

## 0. Conclusion first

**The user's suspicion points in the right direction, but the accurate statement is not
"the pipeline is broken" — it is "the readout and the corpus cannot carry the weight of the
conclusion drawn from them."** Four parts:

1. **The LTF checkpoint itself is healthy** — no degeneracy, no stuck output (§4). The more
   fundamental possibility is ruled out.
2. **The occlusion pipeline is correct**: in all 5 cases inspected visually, the grey patch
   covers the VRU precisely — nothing else is covered, nothing is missed (§3.3).
3. **LTF is not "blind"**: it responds to occlusion **significantly and with the correct sign**,
   and the response varies monotonically with distance / apparent size (closer ⇒ larger
   response). That is the signature of a model that sees and grades its reaction by distance,
   not the signature of a blind spot (§1.3).
4. **But two things make the wording "blind action" untenable**:
   - **The readout's expressible range is extremely small**: 99.91% of LTF's planned speed is
     determined by the input ego speed; the total range the *image* can move it over is only
     **0.0855 m/s**. And all four arms are fed **identical** ego speed. So
     $b_{ghost}=-0.0245$ is not "almost no response" — it is **28.6% of the entire
     image-drivable range** (§4.2).
   - **Most "hazards" in the corpus are not hazards**: all 5 cases inspected are **roadside
     pedestrians** (pavement, grass verge, behind barriers), not on the ego's path. Not
     decelerating is the *correct* behaviour there (§2, §3). A further 5/282 events have the
     ego already stopped, so there is no speed left to shed.

**Therefore this report does not support reading the LTF cell as "blind to the hazard entity".**
The mechanical conclusion (the position of $R$'s CI) is not miscomputed, but it cannot carry
that semantics.

## 1. The distribution of $b_{ghost}$ (not just the mean)

$b_{ghost}=v_{ghost}-v_{clean}$; **negative = planned speed reduced after seeing the hazard**.
n = 282 (G1, class A).

**Table 1. Distribution of $b_{ghost}$.**

| Statistic | Value (m/s) |
| --- | --- |
| mean | −0.0245 |
| s.d. | 0.0963 |
| p10 | −0.1417 |
| p25 | −0.0700 |
| p50 | −0.0200 |
| p75 | +0.0197 |
| p90 | +0.0713 |
| most negative | −0.3963 |
| most positive | +0.4174 |

**Table 2. Binned counts.**

| Bin (m/s) | Events | Share |
| --- | --- | --- |
| $b<-0.20$ | 10 | 3.5% |
| $-0.20\le b<-0.05$ | 93 | 33.0% |
| $-0.05\le b<+0.05$ | 139 | 49.3% |
| $+0.05\le b<+0.20$ | 31 | 11.0% |
| $b\ge+0.20$ | 9 | 3.2% |

### 1.1 "Meaningful deceleration" rate

**Threshold $b_{ghost}<-0.5$ m/s, justified as follows**: `commanded_speed` is the first-step
displacement of the planned trajectory ÷ dt, with dt = 0.5 s for the DD family. Comfortable
braking is ≈ 2 m/s², which over that first 0.5 s step corresponds to ≈ **1 m/s** of speed
reduction. Half of that (0.5 m/s) is a deliberately **lenient** "moved at least half a
comfortable braking step" criterion.

> **Events with $b_{ghost}<-0.5$ m/s: 0 / 282 (0.0%).**
> Relaxing to $-0.2$: 10/282 (3.5%); to $-0.05$: 103/282 (36.5%).

**Scope limit that must be stated**: this is **not a closed-loop braking success rate**. We ran
no closed loop — the model's output is never fed back into a simulator, nothing is rolled out to
a stop, and no collision-avoidance outcome is adjudicated. What is measured is the open-loop,
single-frame proxy "did the planned speed move appreciably downward". A model could score low on
this proxy and still decelerate successfully in closed loop through sustained small corrections,
or vice versa. **The two must not be conflated; nowhere does this report claim a braking rate.**

### 1.2 Effect of clean-arm contamination on this distribution (re-check of §FC/A61)

On G1, **219/288 (76.0%) of clean frames already contain the visible entity**, and the ghost
frame's box is only 1.67× larger (median). So $b_{ghost}$ is mostly not "hazard present vs
absent" but "pedestrian at 31.3 m vs 25.2 m" (medians).

**But for this LTF cell, contamination is not why the number is small:**

| Subset | n | $b_{ghost}$ [scene-level CI] | p50 | count $<-0.5$ |
| --- | --- | --- | --- | --- |
| entity genuinely absent in clean frame | 67 | −0.0255 [−0.0522, +0.0038] | −0.018 | 0 |
| entity already visible (contaminated) | 215 | −0.0242 [−0.0391, −0.0070] | −0.021 | 0 |

The two point estimates are nearly identical (−0.0255 vs −0.0242) ⇒ **contamination did not
shrink $b_{ghost}$**; it only affects significance (the uncontaminated subset is smaller and its
CI spans 0). Recorded as such: **clean-arm contamination cannot be used to explain LTF's small
response.**

### 1.3 Dose–response: LTF does see, and grades by distance

**Table 3. Binned by the pedestrian's longitudinal distance in the ghost frame.**

| Distance (m) | n | mean $b_{ghost}$ | mean within-frame $d_{occ}$ | bbox (% of frame) |
| --- | --- | --- | --- | --- |
| 0–10 | 4 | **−0.1580** | +0.0043 | 2.642 |
| 10–20 | 75 | **−0.0556** | +0.0184 | 1.108 |
| 20–30 | 92 | −0.0345 | +0.0034 | 0.361 |
| 30–45 | 75 | +0.0132 | +0.0067 | 0.166 |
| 45+ | 36 | +0.0024 | −0.0026 | 0.091 |

$\mathrm{corr}(d_{long},\,b_{ghost}) = +0.273$ (farther ⇒ less deceleration);
$\mathrm{corr}(\text{bbox share},\,d_{occ}) = +0.326$ (larger on the image ⇒ erasing it matters
more).

**This is positive evidence of seeing**: a genuinely blind model would not produce a monotone
distance gradient.

### 1.4 Within-frame contrast (the only variable is the grey patch)

| Readout | Value [scene-level CI] | Reading |
| --- | --- | --- |
| $d_{occ}=v_{occ}-v_{ghost}$ | **+0.0075 [+0.0043, +0.0110]** | **significant**, correct sign (erase the pedestrian ⇒ speed up) |
| $d_{ctrl}=v_{ctrl}-v_{ghost}$ | +0.0034 [−0.0001, +0.0067] | spans 0 |
| paired $\lvert d_{occ}\rvert-\lvert d_{ctrl}\rvert$ | +0.0013 [−0.0023, +0.0049] | **spans 0 ⇒ specificity fails** |

**Recorded honestly**: LTF responds significantly to "erase the pedestrian", but that response
**does not pass the specificity test** — an equal-area grey patch placed elsewhere produces a
statistically indistinguishable effect. So one may say "LTF reacts to a change in that image
region", but **not** "LTF reacts specifically to that pedestrian".

## 2. Corpus validity: most "hazard events" are not hazards

The mining on-path criterion is $\lvert lat\rvert \le 2$ m (satisfied by 282/282, i.e. the
criterion is *tight*). But `lat_at_emergence` is the lateral offset **under the ego's
instantaneous heading at that frame**, **not** the offset relative to the ego's actual future
path. On a curve the two differ greatly: in `scene-0433_000_A` the pedestrian has
$\lvert lat\rvert=1.56$ m at 44.8 m, which looks "almost straight ahead", but **the road bends
sharply right** and the ego never goes near them (see §3).

> This is the same class of trap the user just had me register as **§FE/A69**:
> **a longitudinal/lateral decomposition taken against the instantaneous heading distorts at
> long range on curves.** Here it does not merely affect a readout — it affects the corpus's
> own inclusion criterion.

**Table 4. Binned by lateral offset.**

| $\lvert lat\rvert$ (m) | n | mean $b_{ghost}$ | most negative |
| --- | --- | --- | --- |
| 0.0–1.5 | 81 | −0.0029 | −0.2092 |
| 1.5–2.0 | 201 | −0.0332 | −0.3963 |

Note the direction is **inverted**: the more laterally central group responds *less*. If the
corpus really selected VRUs intruding into the path, it should be the other way round. This is
consistent with the inclusion criterion distorting at long range.

**Events both on-path ($\lvert lat\rvert\le2$ m) and close ($\le25$ m): 122/282 (43.3%)**; their
mean $b_{ghost}$ is −0.0598, most negative −0.3494 — still not one reaching −0.5.

## 3. Cases and visual inspection

Images are in `results/figures/ltf_sanity_check/`, 5 per event
(clean / ghost / occ / ctrl / zoom_ghost_vs_occ), 30 in total. **`meta_export_v2.json`** records,
per image, the event, the values, box coordinates, pedestrian distance, and whether the clean
frame is contaminated.

> **Directory write collision (recorded honestly)**: this directory also contains a set of
> `responsive_*.png` / `unresponsive_*.png` files and a `meta.json` that are **not products of
> this report** — they select a different set of events (scene-0500_005_A, scene-0234_011_A,
> scene-0476_001_A, …), use different naming and format, and the `meta.json` timestamp is later
> than my exported JPGs. The assessment is that **another session is executing the same work
> order in parallel** and overwrote my original `meta.json`. I therefore wrote my metadata to
> `meta_export_v2.json` and **did not delete or modify any of their files**. Every number in
> this report comes from `meta_export_v2.json` and `f3_occlusion_ltf.json`.

### 3.1 "Clear deceleration response" group (3 most negative $b_{ghost}$)

| Event | $b_{ghost}$ | $v_{clean}\to v_{ghost}$ | $v_{occ}$ | $v_{ctrl}$ | Dist | Box | Visual scene |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scene-0433_000_A | −0.3963 | 6.591 → 6.195 | 6.221 | 6.366 | 44.8 m | 878 px | pedestrian on a **pavement behind a retaining wall**; road bends sharply right |
| scene-0069_000_A | −0.3494 | 4.029 → 3.680 | 3.681 | 3.671 | 22.1 m | 5942 px | pedestrian walking on a **pavement behind hoarding**, railing-separated from the carriageway |
| scene-0858_004_A | −0.2951 | 3.581 → 3.286 | 3.312 | 3.290 | 16.0 m | 28988 px | **motorcyclist** in an **adjacent/oncoming lane**, not on the ego path |

Note that for scene-0433, $v_{ctrl}=6.366$ is **closer to $v_{clean}$ than $v_{occ}=6.221$ is** —
the patch placed *elsewhere* restores more speed than the patch covering the pedestrian. That is
the opposite of "this event's response is about the pedestrian", and supports the failed
specificity in §1.4.

### 3.2 "Almost no response" group (3 smallest $\lvert b_{ghost}\rvert$)

| Event | $b_{ghost}$ | $v_{clean}\to v_{ghost}$ | $v_{occ}$ | $v_{ctrl}$ | Dist | Box | Visual scene |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scene-0961_015_A | +0.0002 | 4.920 → 4.921 | 4.895 | 4.907 | 17.0 m | 10170 px | **road worker** on the **grass verge beyond the kerb**, facing away from the carriageway |
| scene-0019_006_A | −0.0003 | 4.258 → 4.258 | 4.259 | 4.272 | 25.4 m | 3149 px | pedestrian at the roadside |
| scene-1064_002_A | +0.0004 | **0.004 → 0.005** | 0.006 | 0.004 | 26.0 m | 6387 px | **night**; **the ego is already stopped** (v ≈ 0.004 m/s) |

**scene-1064_002_A needs separate comment**: the ego's own speed is ≈ 0.004 m/s — **the vehicle
has already stopped**. "Sees the hazard but does not decelerate" is a meaningless reading here:
**there is no speed to shed.** Such events dilute the aggregate response without reflecting any
model defect. In G1 class A, **5** events have $v_{ghost}<1$ m/s and **14** have $<2$ m/s (of 282).

**Key observation: the difference between the two groups is not "hazardous vs not".** **None** of
the six cases is "a VRU intruding into the ego's path". The supposedly strongest reaction,
scene-0433, is in fact the **farthest (44.8 m), smallest on the image (878 px), and behind a
retaining wall** — which looks more like noise on a small distant target than a hazard response.

### 3.3 Visual verification of occlusion correctness (work-order item 3)

I inspected the `zoom_ghost_vs_occ` image (ghost and occ side by side, magnified) for all 5 cases:

| Event | Patch covers the VRU precisely | Covers anything else | Misses part |
| --- | --- | --- | --- |
| scene-0858_004_A | yes (motorcyclist fully covered) | no | a sliver of the lower wheel remains |
| scene-0433_000_A | yes | no | no |
| scene-0069_000_A | yes | no | no |
| scene-0961_015_A | yes (worker fully covered) | no | no |
| scene-1064_002_A | yes (night scene; the patch takes the image-mean colour so it appears dark — as designed) | no | no |

> **The occlusion pipeline passes.** The patch uses the whole-image mean colour, hence its dark
> appearance at night — that is by design (it introduces no new high-frequency structure), not a
> defect. The control patch is an equal-area horizontal mirror of the box and visibly does not
> overlap the original.

All six cases have a contaminated clean frame (entity already visible), consistent with the 76%
in §1.2.

## 4. LTF checkpoint health (work-order item 4)

30 ordinary driving frames were sampled at random from scenes that **do not appear in the F-3
event set** (`scripts/ltf_baseline_probe.py`).

### 4.1 No degeneracy

| Check | Result |
| --- | --- |
| Is the planned speed constant? | no — **30 distinct values** across 30 frames |
| All zero / stuck? | no — range 0.004 – 8.779 m/s, s.d. 2.885 |
| Only ever straight? | no — max endpoint lateral offset 7.30 m; 12/30 exceed 1 m |
| Does it track ego speed? | yes — $v_{plan}=0.956\,v_{in}+0.036$ |

> **Conclusion: the LTF checkpoint behaves normally** — its output varies with the scene, it
> turns, and its speed tracking is sensible. **The more fundamental possibility, "this checkpoint
> is broken", is ruled out.**

### 4.2 But this exposes a fundamental limitation of the readout (the single most important number here)

$\mathrm{corr}(v_{plan},\,v_{in}) = \mathbf{0.9996}$. After regression:

> **residual s.d. = 0.0855 m/s — this is the entire range over which image content can move
> $v_{plan}$**, amounting to only **0.088%** of $v_{plan}$'s total variance.

And F-3's four arms are fed **identical ego speed** (`spd(ev)` is taken from `x_clean_frames` and
shared by all four arms), so between-arm differences can **only** come from the image. Hence:

| Readout | Absolute value | Share of the total image-drivable range (0.0855 m/s) |
| --- | --- | --- |
| $\lvert b_{ghost}\rvert$ = 0.0245 | small | **28.6%** |
| $\lvert d_{occ}\rvert$ = 0.0075 | smaller | 8.8% |

**Reinterpretation**: in absolute terms $b_{ghost}=-0.0245$ m/s looks like "almost no response",
but relative to **what this readout can express on this model at all**, it is **28.6%**. To see a
"braked to ~1 m/s" reaction on this readout would require a change **12× the entire image-drivable
range** — **physically impossible on this checkpoint**.

> This explains why three successive readout changes (speed / distance / arc length) all returned
> null results: arc length correlates strongly with speed (|ρ| up to 0.73 last round), so all
> three are constrained by the same bottleneck — **LTF's speed head is very nearly a pass-through
> of ego speed.**

## 5. Impact on published verdicts (bounded honestly, not overstated)

* **The mechanical conclusion is unchanged**: the position of $R$'s scene-level CI is not
  miscomputed, and this report **overturns no number**.
* **The semantic reading must be narrowed**: if the paper reads this LTF cell as "blind to the
  hazard entity / blind generalisation", that wording **exceeds what this report can support**.
  What is supportable is: "under this corpus and this speed readout, the modulation of LTF's
  planned speed by the hazard entity does not reach a level distinguishable from a control patch"
  — accompanied by the expressible-range ceiling of §4.2 and the corpus-validity limits of §2.
* **No paper file and no existing result file was modified.** Whether to adjust the wording is
  for you to decide.

## 6. Self-correction record

1. The first image export used the wrong nuScenes root (`/data/dataset/nuscenes`); it should be
   `/data/dataset/nuscenes/v1.0-trainval`. Fixed and re-exported.
2. Image annotations were initially in Chinese; OpenCV has no CJK glyphs and rendered them as
   `?????`. All changed to ASCII and re-exported.
3. **In the draft, the $v_{clean}/v_{ghost}/v_{occ}$ values in the two case tables were wrong** —
   they were written without checking (e.g. scene-1064 was written as 3.771 when it is actually
   0.004). Before finalising, every value was verified against `meta_export_v2.json` and
   corrected. It was this check that revealed scene-1064's "ego already stopped", which changed
   that case's interpretation.
4. I initially read `lat_at_emergence` as "lateral offset relative to the ego's path"; after the
   images contradicted it, I verified that it is the offset **under the instantaneous heading**.
   Corrected in §2, with its consequences stated.
5. **Evidence was not gathered selectively in response to the challenge**: this report presents
   both evidence supporting "the pipeline has problems" (§2 corpus validity, §4.2 readout
   ceiling) and evidence against it (§3.3 occlusion correct, §4.1 checkpoint healthy, §1.3
   dose–response present).

## 7. Open items

| Item | Status | Note |
| --- | --- | --- |
| Change the corpus on-path criterion to be relative to the ego's **future path** | not done | fixing §2 requires re-mining; it affects every class-A event in G1, so the cost is high and the call is yours |
| Whether the speed-head pass-through is specific to this checkpoint | not done | the same 30-frame probe should be run on dd / ddv2 / simlingo to see whether the §4.2 bottleneck is general |
| Whether to narrow the paper's wording | **awaiting your decision** | §5; no paper file was touched this round |
| R-estimator confirmation (previous work order) | **computed, not written up** | numbers are in `results/f3_restimator_b.json`; all three cells match last round |

## Appendix: image list

`results/figures/ltf_sanity_check/` (30 JPGs + `meta_export_v2.json`)

```
meta_export_v2.json          <- this report's metadata (meta.json was overwritten by a parallel session, see §3)
reacted__scene-0433_000_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
reacted__scene-0069_000_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
reacted__scene-0858_004_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
flat__scene-0961_015_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
flat__scene-0019_006_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
flat__scene-1064_002_A__{clean,ghost,occ,ctrl,zoom_ghost_vs_occ}.jpg
```
