# 第二场景类型（前车急刹）：SimLingo 的 G / F① / C-hazard 读数

> 工单：[`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md) 任务二。
> 语料与匹配质检见 [`lead_vehicle_brake_mining_report_zh.md`](lead_vehicle_brake_mining_report_zh.md)。
> 跨场景"泛化了吗"的统一判定见 [`second_scenario_generality_report_zh.md`](second_scenario_generality_report_zh.md)。

---

## Methods

**候选**：SimLingo，可读位置 24 个 decoder layer（InternVL2-1B / Qwen2-0.5B，hidden 896）。属 **纯 transformer 栈**。

**刺激集**：第二场景语料 LB = 103 / LBn = 48 / LBv = 31（154 scene）。
LB / LBn / LBv 与 G1 的 A / D2a / D2cV **同构但不同名**。

**口径**：全部公式、统计纪律与判定门**一字未改**，只把事件族名参数化
（`--pos LB --neg LBn --floor LBv`）。这正是"四轴**定义**可迁移"这一命题的执行形式——
若需要改公式才能跑，命题就已经被否证了。
G 轴：折内提方向、折内选峰层、scene 级 4 折 CV，主读数 = CV-AUC(LB vs LBn) − 自身 LBv 证伪地板；
F①：架构中立的行动层反事实，$b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$；
C-hazard：clean（平稳跟车）↔ ghost（急刹）配对的逐层 activation patching，
含 patch-ALL 充分割集自检与恢复剖面形状诊断。

**必须先讲清楚的功效限制**：本场景样本量为 G1 的 1/3 ~ 1/6，CI 宽度约 1.5 ~ 2.5 倍。
因此"CI 跨 0"在本场景上**更多地是功效陈述而不是复现失败陈述**，正文逐条区分。

---

## Results

**Table 1. G 轴（vision_mean 主口径），并列 G1 鬼探头场景上的同一读数。**

| 量 | 第二场景（前车急刹） | G1（鬼探头） |
| --- | --- | --- |
| 主读数 CV-AUC(LB vs LBn) | 0.600 [0.505, 0.694]，p = 0.0495 | —— |
| 证伪地板 CV-AUC(vs LBv) | 0.645 | —— |
| **主读数 − 证伪地板** | **−0.045** [−0.177, +0.081] | +0.035 [−0.026, +0.097] |
| 10 seed 折分配 | −0.033 ± 0.036 | 见 G1 报告 |
| 随机方向地板 / 置换地板 | 0.485 / 0.489 | —— |
| 峰层 | L0 | —— |
| **判定** | **不可估** | 不可估 |

**Table 2. F① 行动层反事实（架构中立）。**

| 量 | 第二场景 | G1 |
| --- | --- | --- |
| $b$(LB) [m/s] | −0.059 [−0.330, +0.211] | —— |
| **b-AUC(LB vs LBn)** | **0.491 [0.383, 0.589]** | 0.534 [0.469, 0.592] |
| b-AUC(LB vs LBv，证伪地板) | 0.499 | —— |
| **判定** | **不可估**（CI 跨 0.5） | 不可估 |

**Table 3. C-hazard（配对源 = clean 平稳跟车 ↔ ghost 急刹）。**

| 量 | 第二场景 | G1 |
| --- | --- | --- |
| patch-ALL 充分割集自检 | +0.965（须落在 [0.7, 1.3]）⇒ **通过** | 通过 |
| Spearman(层号, recovery) | **−0.921** ⇒ **级联** | 级联 ρ=−0.997 / 承诺层 L3 / $C_m$ n/a |
| $C_m$ = top-2 层占比 | 0.182 [0.159, 0.207]（弥散基线 0.083） | 见右列 |
| 责任层众数 / 归一化熵 | L0 / 0.484 | |
| 承诺层（mean recovery ≥ 0.9 的最深层） | **L8 / 24（深度 0.38）** | |
| **判定** | **不可估**（剖面呈级联，$C_m$ 公式前提结构性不成立） | |

---

## Discussion

本报告只落数字。跨场景"泛化了吗"的判定统一在
[`second_scenario_generality_report_zh.md`](second_scenario_generality_report_zh.md) 给出，
避免三份候选报告各说各话。
