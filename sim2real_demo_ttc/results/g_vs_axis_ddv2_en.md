# G-VS (segmentation consistency): readout for DiffusionDriveV2

> Work order: [`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md) §1.
> Task definition and feasibility: [`g_vs_feasibility_report_en.md`](g_vs_feasibility_report_en.md)。
> Unified six-candidate adjudication: [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md)。

## Methods

**Task**: binary "object-ness" segmentation. Pseudo-GT is generated automatically by SAM (ViT-B) on
the G1 A-class ghost frames; masks smaller than 5% of the frame are labelled object, everything else
background. **No human semantic criterion is involved.**

**Features**: the TransFuser two-branch encoder's 8×32 image-token grid (covering the 4:1 crop band), taking layer 6's **token-level (unpooled)** representation. The existing
caches hold pooled vision_mean / region_mean, whose spatial structure is gone and which cannot support
a segmentation probe, so we re-ran the forward pass.

**Probe**: a single linear layer (logistic regression, lbfgs, `class_weight="balanced"`).
**Deliberately no hidden layer** — the larger the probe's capacity, the more it can solve the task by
itself and the less trustworthy selectivity becomes (Hewitt & Liang 2019).

**Primary readout**: **selectivity = mIoU(trained) − mIoU(random_init)**, under scene-level 4-fold CV
with a scene-level paired bootstrap.

## Results

**Table 1. Held-out mIoU for the three arms (binary, scene-level bootstrap CI).**

| Arm | mIoU | 95% CI | Meaning |
| --- | --- | --- | --- |
| **trained** | **0.3475** | [0.3343, 0.3606] | the trained model |
| random_init | 0.3316 | [0.3146, 0.3479] | same architecture, randomly initialized (control-task floor) |
| position_only | 0.3331 | [0.3212, 0.3452] | token coordinates only (spatial-prior floor) |

| Quantity | Value |
| --- | --- |
| tokens / feature dim / scenes | 74496 / 512 / 151 |
| object-token fraction | 0.292 |
| **selectivity (primary readout)** | **+0.0159 [-0.0029, 0.0353]** |
| **Verdict** | **indeterminate: selectivity's scene-level CI spans 0** |

## Discussion

See [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md)。
