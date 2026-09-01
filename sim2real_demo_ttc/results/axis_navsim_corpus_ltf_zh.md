# NAVSIM/OpenScene 独立语料：LTF (Latent TransFuser) 的 G / F① / C-hazard 读数

> 工单：[`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md)。
> 可行性与语料质检：[`navsim_openscene_mining_report_zh.md`](navsim_openscene_mining_report_zh.md)。
> **跨语料"复现了吗"的统一判定见 [`cross_corpus_generality_report_zh.md`](cross_corpus_generality_report_zh.md)**——
> 本报告只落数字，避免三份候选报告各说各话。

---

## Methods

**候选**：LTF (Latent TransFuser)，可读位置 8 个 encoder self-attention（`TransfuserBackbone`，`latent=True`）。属 **TransFuser 系**。

**刺激集**：NAVSIM/OpenScene test split，147 log / 1880 scene，
A = 397、D2a = 382（caliper 匹配后）、**D2cV = 134**（证伪地板）。
**事件族名与 G1 完全相同**（A / D2a / D2cV），因为判据也完全相同——
本轮唯一变的是数据源。

**口径**：`detect_events` / `pick_frames` / `frame_record` / `n1_match` 全部复用，**一行未改**；
阈值逐字段不动。G/F①/C-hazard 三个读出脚本连事件族名参数都不用换。
**变的只有两处，都已登记**：数据源接入层（`ns1_navsim_geometry.py`）
与裁剪主点行 450 → 560（§NS/A46，若不改会静默污染全部读数）。

**功效**：本语料的正例（397）与几何匹配负例（382）都多于 G1（291 / 283），
证伪地板（134）少于 G1（212）。主读数 − 地板的 CI 宽度实测与 G1 相当，
故本轮的"不可估"**不能**一律按功效不足解释——逐格判定见跨语料报告。

---

## Results

**Table 1. G 轴（vision_mean 主口径），并列 G1 上的同一读数。**

| 量 | NAVSIM 独立语料 | G1（nuScenes） |
| --- | --- | --- |
| 主读数 CV-AUC(A vs D2a) | 0.630 [0.592, 0.669]，p = 3.1e-10 | —— |
| 证伪地板 CV-AUC(vs D2cV) | 0.620 | —— |
| **主读数 − 证伪地板** | **+0.011** [−0.038, +0.065] | **+0.070 [+0.017, +0.126]** |
| 10 seed 折分配 | +0.022 ± 0.022 | 见 G1 报告 |
| region_mean 敏感性 | +0.032 [−0.013, +0.080] | 见 G1 报告 |
| 随机方向 / 置换地板 | 0.513 / 0.527 | —— |
| 方向几何纯度 ρ(投影, log area) | −0.186 (p = 2.1e-07) | —— |
| **判定** | **不可估** | **PASS**（本工作线唯一的 G 轴阳性） |

**Table 2. F① 行动层反事实（架构中立）。**

| 量 | NAVSIM | G1 |
| --- | --- | --- |
| $b$(A) [m/s] | +0.024 [+0.010, +0.036]，**显著** | —— |
| **b-AUC(A vs D2a)** | **0.546 [0.496, 0.593]** | **0.583 [0.519, 0.639]**（PASS） |
| b-AUC(A vs D2cV，证伪地板) | 0.510 | —— |
| **判定** | **不可估**（CI 跨 0.5） | 见右列 |

**Table 3. C-hazard（clean↔ghost 配对的逐层 activation patching）。**

| 量 | NAVSIM | G1 |
| --- | --- | --- |
| patch-ALL 充分割集自检 | +1.000 | 见右列 |
| Spearman(层号, recovery) | **+0.786** ⇒ **内部峰** | 内部峰 ρ=+0.929 / **L6** / 承诺层不存在 / $C_m$ 0.789 PASS |
| $C_m$ = top-2 层占比 | 0.752 [0.664, 0.844]（弥散基线 0.250） | 见右列 |
| 责任层众数 / 归一化熵 | **L7** / 0.706 | |
| 承诺层 | **不存在** | |
| **判定** | **PASS：C 显著高于弥散基线** | |

---

## Discussion

见 [`cross_corpus_generality_report_zh.md`](cross_corpus_generality_report_zh.md)。
