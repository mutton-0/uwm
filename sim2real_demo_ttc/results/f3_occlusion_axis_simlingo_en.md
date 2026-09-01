# F-3 (occlusion necessity test): readout for SimLingo

> Work order: [`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md), §2.
> Unified six-candidate adjudication: [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md).

## Methods

**Design**: four arms, all run on every A-class event.

| Arm | Input |
| --- | --- |
| clean | the frame in which the hazard entity is simply not present |
| ghost | the hazard entity is visible (original) |
| **occ** | ghost, with the hazard entity's projected box filled with the image mean colour |
| **ctrl** | ghost, with an equal-area, equal-eccentricity, non-overlapping box filled **elsewhere** |

**The ctrl arm is required, not an appendix** (§GF/A51): the occ arm does two things at once —
"removes this entity" and "adds a grey patch". With occ alone, a model that reacts to any grey patch
would be misread as passing the necessity test.

**Primary readout**: $b_{ghost} = v_{plan}(\text{ghost}) - v_{plan}(\text{clean})$,
$b_{occ} = v_{plan}(\text{occ}) - v_{plan}(\text{clean})$, and the
**necessity ratio $R = 1 - b_{occ} / b_{ghost}$** (1 = fully necessary, 0 = not necessary at all).
Adjudication: $R$'s scene-level CI lower bound > 0.5 → PASS; upper bound < 0.5 → FAIL; spanning 0.5 →
indeterminate. **Denominator guard**: $R$ is computed only on events with $|b_{ghost}| \ge$ 0.02.

> Note: $b_{ghost}$ here has the opposite sign convention to F①'s $b$ (defined there as clean − ghost).
> $R$ is a ratio and is unaffected by the convention.

## Results

**Table 1. Four-arm readouts (scene-level bootstrap). 282 events.**

| Quantity | Value | 95% CI | Significantly ≠ 0? |
| --- | --- | --- | --- |
| $b_{ghost}$ (original response) | -0.4441 | [-0.6979, -0.1897] | **yes** |
| $b_{occ}$ (residual after occlusion) | -0.2703 | [-0.5226, -0.0273] | **yes** |
| $b_{ctrl}$ (patch elsewhere) | -0.4334 | [-0.6755, -0.1873] | **yes** |

| Quantity | Value |
| --- | --- |
| **Necessity ratio $R$ (primary)** | **+0.461 [0.166, 0.789]** |
| Control arm $R_{ctrl}$ (should be ≈ 0) | +0.155 [-0.048, 0.399] |
| Events past the gate | 273 |
| **Verdict** | **indeterminate: the necessity ratio's scene-level CI spans 0.5** |

## Discussion

See [`g_vs_f3_unified_matrix_report_en.md`](g_vs_f3_unified_matrix_report_en.md).
