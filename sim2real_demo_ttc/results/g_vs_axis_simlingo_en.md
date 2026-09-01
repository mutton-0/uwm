# G-VS (segmentation consistency): readout for SimLingo

> Work order: [`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md) §1.
> Task definition and feasibility: [`g_vs_feasibility_report_en.md`](g_vs_feasibility_report_en.md)。
> Unified six-candidate adjudication: [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md)。

## Methods

**Task**: binary "object-ness" segmentation. Pseudo-GT is generated automatically by SAM (ViT-B) on
the G1 A-class ghost frames; masks smaller than 5% of the frame are labelled object, everything else
background. **No human semantic criterion is involved.**

**Features**: InternVL2-1B vision tokens (dynamic_preprocess tile grid, 16×16 per tile), taking layer 12's **token-level (unpooled)** representation. The existing
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
| **trained** | **0.3963** | [0.3866, 0.4056] | the trained model |
| random_init | 0.3691 | [0.3544, 0.3831] | same architecture, randomly initialized (control-task floor) |
| position_only | 0.3564 | [0.3436, 0.3697] | token coordinates only (spatial-prior floor) |

| Quantity | Value |
| --- | --- |
| tokens / feature dim / scenes | 148992 / 896 / 151 |
| object-token fraction | 0.232 |
| **selectivity (primary readout)** | **+0.0273 [0.0128, 0.0424]** |
| **Verdict** | **PASS: the representation does carry object-ness information (selectivity's scene-level CI excludes 0)** |

## Discussion

See [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md)。
