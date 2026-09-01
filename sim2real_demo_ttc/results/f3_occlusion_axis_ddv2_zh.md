# F-3（遮挡必要性检验）：DiffusionDriveV2 的读数

> 工单：[`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md) §2。
> 六候选统一判定：[`g_vs_f3_unified_matrix_report_zh.md`](g_vs_f3_unified_matrix_report_zh.md)。

## Methods

**设计**：四个臂，每个 A 类事件都跑满。

| 臂 | 输入 |
| --- | --- |
| clean | 危险实体本来就不在场的帧 |
| ghost | 危险实体可见（原始） |
| **occ** | ghost + 把危险实体的投影框涂为图像均值色 |
| **ctrl** | ghost + 在**别处**涂一个同面积、同离心率带、不重叠的框 |

**ctrl 臂是必需项不是附录**（§GF/A51）：occ 臂同时做了「移除该实体」与「加一块灰斑」两件事，
只有 occ 时，一个对任意灰斑都会反应的模型会被误判为通过必要性检验。

**主读数**：$b_{ghost} = v_{plan}(\text{ghost}) - v_{plan}(\text{clean})$，
$b_{occ} = v_{plan}(\text{occ}) - v_{plan}(\text{clean})$，
**必要性比 $R = 1 - b_{occ} / b_{ghost}$**（1 = 完全必要，0 = 完全不必要）。
判定：$R$ 的 scene 级 CI 下界 > 0.5 → PASS；上界 < 0.5 → FAIL；跨 0.5 → 不可估。
**分母守卫**：只在 $|b_{ghost}| \ge$ 0.02 的事件上算 $R$。

> 注：此处 $b_{ghost}$ 与 F① 的 $b$ 差一个符号（F① 定义为 clean − ghost）。
> $R$ 是比值，不受符号约定影响。

## Results

**Table 1. 四臂读数（scene 级 bootstrap）。事件数 282。**

| 量 | 值 | 95% CI | 是否显著 ≠ 0 |
| --- | --- | --- | --- |
| $b_{ghost}$（原始响应） | -0.1803 | [-0.3487, 0.0032] | 否 |
| $b_{occ}$（遮挡后残余） | -0.2060 | [-0.3707, -0.029] | **是** |
| $b_{ctrl}$（涂别处） | -0.1767 | [-0.3461, 0.0074] | 否 |

| 量 | 值 |
| --- | --- |
| **必要性比 $R$（主读数）** | **+0.135 [-0.094, 0.367]** |
| 对照臂 $R_{ctrl}$（应接近 0） | +0.074 [-0.254, 0.413] |
| 过门槛事件数 | 264 |
| **判定** | **不可估：**基线响应本身与 0 不可区分**（b_ghost 的 CI 跨 0）⇒ 没有可供必要性检验的响应。这不是必要性检验失败，是无从检验。** |

## Discussion

见 [`g_vs_f3_unified_matrix_report_zh.md`](g_vs_f3_unified_matrix_report_zh.md)。
