# G-VS（分割一致性）：LTF 的读数

> 工单：[`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md) §1。
> 任务定义与可行性：[`g_vs_feasibility_report_zh.md`](g_vs_feasibility_report_zh.md)。
> 六候选统一判定：[`g_vs_f3_unified_matrix_report_zh.md`](g_vs_f3_unified_matrix_report_zh.md)。

## Methods

**任务**：二类「物体性」分割。伪 GT 由 SAM（ViT-B）在 G1 的 A 类 ghost 帧上自动生成，
面积 < 画幅 5% 的掩码判为 object，其余为 background。**不含任何人工语义判据。**

**特征**：TransFuser 双分支编码器的 8×32 图像 token 网格（覆盖 4:1 裁剪带），取第 6 层的 **token 级（未池化）**表征。
既有缓存里的 vision_mean / region_mean 是池化量，空间结构已被抹掉，不能用于分割探针，故重新前向。

**探针**：单层线性（逻辑回归，lbfgs，`class_weight="balanced"`）。
**刻意不加隐层**——探针容量越大越容易自行完成任务，selectivity 越不可信（Hewitt & Liang 2019）。

**主读数**：**selectivity = mIoU(trained) − mIoU(random_init)**，
scene 级 4 折 CV + scene 级配对 bootstrap。

## Results

**Table 1. 三条臂的 held-out mIoU（二类，scene 级 bootstrap CI）。**

| 臂 | mIoU | 95% CI | 含义 |
| --- | --- | --- | --- |
| **trained** | **0.3672** | [0.353, 0.3814] | 训练好的模型 |
| random_init | 0.3481 | [0.3326, 0.364] | 同架构随机初始化（control task 地板） |
| position_only | 0.3331 | [0.3212, 0.3452] | 只用 token 坐标（空间先验地板） |

| 量 | 值 |
| --- | --- |
| token 数 / 特征维 / scene 数 | 74496 / 512 / 151 |
| object token 占比 | 0.292 |
| **selectivity（主读数）** | **+0.0191 [-0.0006, 0.0385]** |
| **判定** | **不可估：selectivity 的 scene 级 CI 跨 0** |

## Discussion

见 [`g_vs_f3_unified_matrix_report_zh.md`](g_vs_f3_unified_matrix_report_zh.md)。
