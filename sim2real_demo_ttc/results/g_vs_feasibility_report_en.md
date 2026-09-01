# G-VS feasibility assessment: an existing segmentation head vs. SAM pseudo-GT

> Work order: [`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md), §1's half-day timebox. Actual time < 2 hours.
> Autonomous decisions: [`amendments.md`](amendments.md) §GF/A49–A52.

---

## 1. Conclusions on the two items to check

| Item | Conclusion |
| --- | --- |
| Does the TransFuser family already have a segmentation auxiliary head? | **Yes, but it cannot serve as this round's G-VS** |
| Can SAM run here to generate pseudo-GT? | **Yes** (required a download, completed; 0.9 s/image) |

### 1.1 The TransFuser family does carry a segmentation head, but it is in **BEV space**

`navsim/agents/transfuser/transfuser_model.py:41` defines `_bev_semantic_head`, and
`transfuser_config.py` sets `use_bev_semantic = True` with `bev_semantic_weight = 10.0` — i.e. all
three TransFuser-family candidates carry semantic segmentation **in their original training
objective**, over 7 classes:

| label | class | source |
| --- | --- | --- |
| 1 | road (LANE + INTERSECTION) | map polygons |
| 2 | walkways | map polygons |
| 3 | centerline | map linestrings |
| 4 | static_objects (cones / barriers / czone signs / generic) | annotation boxes |
| 5 | vehicles | annotation boxes |
| 6 | pedestrians | annotation boxes |

**Why it cannot be read directly as G-VS**, for three reasons, none of which is dispensable:

1. **Different space.** It outputs a **bird's-eye-view** semantic map (`bev_pixel_size = 0.25 m`),
   whereas the SAM pseudo-GT is in **image space**. These are not the same prediction task and their
   mIoUs are not interchangeable.
2. **It covers only 3 of 6 candidates.** SimLingo, Alpamayo-R1 and AutoVLA have no such head. The
   work order requires "all six measured uniformly, regardless of single/multi-frame or presence of
   language"; measuring three with a ready-made head and three with a probe yields mIoUs that are not
   comparable across candidates, defeating the point of uniformity.
3. **It is a training objective, not a probe.** Reading it asks "how well does the model do on a task
   it was explicitly trained for", whereas G-VS asks "is scene structure **linearly readable** from
   the representation". The former needs no selectivity control; the latter requires one. They are
   not the same question.

**One substantive use of it, recorded**: it proves the three TransFuser-family candidates' features
**must** contain segmentation information (otherwise the training objective could not be learned). So
if their G-VS probe readouts come out near the random-init floor, the problem can be localized to the
**probe / grid resolution** side rather than "the information is not there". This gives an independent
anchor for interpreting the low mIoU values (see the unified-matrix report's discussion).

### 1.2 SAM is feasible

| Item | Result |
| --- | --- |
| Dependency | `segment_anything` was not installed; pip install succeeded |
| Weights | `sam_vit_b_01ec64.pth` (375 MB), downloaded from `dl.fbaipublicfiles.com` |
| Inference cost | **0.9 s / image** (ViT-B, `points_per_side=16`, cuda:0) |
| Full-corpus cost | 291 A-class frames in G1 ⇒ **< 5 minutes** |

**Pseudo-GT quality check (statistics over all 291 frames)**: median masks per image **45**; object
pixel fraction mean **0.244**, percentiles [0.116, 0.235, 0.381]. The distribution is sane — neither
collapsed to all-background nor smeared into all-object.

---

## 2. The resulting G-VS task definition (pre-registered)

SAM's automatic masks are **class-agnostic instance masks**. Turning them into a well-posed dense
prediction task requires an objective labelling rule. This round uses **binary "object-ness"
segmentation**:

    class 1 (object)     : covered by a SAM mask whose area is < 5% of the frame
    class 0 (background) : everything else (no mask, or road/sky/building-scale stuff regions)

* It uses **SAM only**, introducing no human semantic criterion — exactly the motivation for escaping
  the "hazard" construct;
* the area threshold separates stuff from things, a standard use of SAM's automatic masks, and the
  threshold is serialized with the results;
* binary mIoU is well defined, comparable across candidates, and gives the random-init control a clear
  meaning.

**What it is not** (stating this matters, or the mIoU will be over-read): this is not semantic
segmentation (no classes) and not instance segmentation (individuals are not distinguished). G-VS asks
whether "**where the objects are**" is linearly readable from the representation, not whether the
model can tell a pedestrian from a traffic cone. The latter is left to Future Work's **G-VL**.

## 3. Three control arms (all required)

| Arm | Meaning |
| --- | --- |
| trained | token features from the trained model |
| **random_init** | the **same architecture, randomly initialized**, on the same images, same layer, same grid (Hewitt & Liang 2019's control task) |
| **position_only** | token (row, col) coordinates alone — isolates the spatial prior that objects tend to sit in the lower half of the frame |

**Primary readout = selectivity = mIoU(trained) − mIoU(random_init)**, as a per-scene paired
difference with bootstrap. Reporting absolute mIoU alone is insufficient: the token grid carries a
spatial prior, and the `position_only` arm alone reaches 0.33 in practice; without subtracting it and
the random-init floor, the claim "the representation contains this information" does not stand.

---

## 4. Feasibility verdict

**Pass.** The SAM route is feasible at very low cost; the existing segmentation head is unsuitable for
this round's purpose, with the reasons recorded item by item. This round proceeds with "SAM pseudo-GT
+ linear probe + two floor controls".
