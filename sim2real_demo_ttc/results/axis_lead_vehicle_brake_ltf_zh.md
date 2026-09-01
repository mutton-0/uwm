# 第二场景类型（前车急刹）：LTF (Latent TransFuser) 的 G / F① / C-hazard 读数

> 工单：[`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md) 任务二。
> 语料与匹配质检见 [`lead_vehicle_brake_mining_report_zh.md`](lead_vehicle_brake_mining_report_zh.md)。
> 跨场景"泛化了吗"的统一判定见 [`second_scenario_generality_report_zh.md`](second_scenario_generality_report_zh.md)。

---

## Methods

**候选**：LTF (Latent TransFuser)，可读位置 8 个 encoder self-attention（`TransfuserBackbone`，`latent=True`）。属 **TransFuser 系**。

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
| 主读数 CV-AUC(LB vs LBn) | 0.492 [0.392, 0.591]，p = 0.868 | —— |
| 证伪地板 CV-AUC(vs LBv) | 0.459 | —— |
| **主读数 − 证伪地板** | **+0.032** [−0.098, +0.164] | **+0.070 [+0.017, +0.126]** |
| 10 seed 折分配 | +0.000 ± 0.067 | 见 G1 报告 |
| 随机方向地板 / 置换地板 | 0.530 / 0.514 | —— |
| 峰层 | L4–L5 | —— |
| **判定** | **不可估** | **PASS**（本工作线第一个 G 轴阳性） |

**Table 2. F① 行动层反事实（架构中立）。**

| 量 | 第二场景 | G1 |
| --- | --- | --- |
| $b$(LB) [m/s] | +0.004 [−0.013, +0.022] | —— |
| **b-AUC(LB vs LBn)** | **0.560 [0.471, 0.651]** | **0.583 [0.519, 0.639]** |
| b-AUC(LB vs LBv，证伪地板) | 0.594 | —— |
| **判定** | **不可估**（CI 跨 0.5） | **PASS** |

**Table 3. C-hazard（配对源 = clean 平稳跟车 ↔ ghost 急刹）。**

| 量 | 第二场景 | G1 |
| --- | --- | --- |
| patch-ALL 充分割集自检 | +1.000（须落在 [0.7, 1.3]）⇒ **通过** | 通过 |
| Spearman(层号, recovery) | **+0.786** ⇒ **内部峰** | 内部峰 ρ=+0.929 / **L6** / $C_m$ 0.789 PASS |
| $C_m$ = top-2 层占比 | 0.799 [0.738, 0.862]（弥散基线 0.250） | 见右列 |
| 责任层众数 / 归一化熵 | **L6** / 0.709 | |
| 承诺层（mean recovery ≥ 0.9 的最深层） | **不存在**（任何单层 recovery < 0.9） | |
| **判定** | **PASS：C 显著高于弥散基线** | |

---

## Discussion

本报告只落数字。跨场景"泛化了吗"的判定统一在
[`second_scenario_generality_report_zh.md`](second_scenario_generality_report_zh.md) 给出，
避免三份候选报告各说各话。
