# 完成标志：G-VS / F-3 的跨场景 · 跨数据源复现

> 承接 [`g_vs_f3_DONE.md`](g_vs_f3_DONE.md)。完成日期：2026-09-02。
> 修正案：[`amendments.md`](amendments.md) §GF/A53–A55（累计 55 条）。
> **复用已有挖掘产出，未重新挖数据**；只把 gvs1~gvs3、f3_occlusion_* 换语料重跑。

---

## 1. 覆盖情况

| 语料 | G-VS | F-3 |
| --- | --- | --- |
| G1（已有） | 4 候选 | **6 候选** |
| **前车急刹** | **4 候选**（SimLingo / DD / LTF / DDv2） | **6 候选**（含 Alpamayo-R1 / AutoVLA） |
| **NAVSIM/OpenScene** | **4 候选** | **4 候选**（两个 VLA 见下） |

**未覆盖的格与理由（如实记录，不硬凑）**：
* **两个 VLA 的 G-VS（三语料一致）**：video token 布局未接完，§GF/A50。
  AutoVLA 的布局本轮之前已解出（3 相机 × `[2,18,32]`），卡在随机初始化对照臂实现。
* **Alpamayo-R1 / AutoVLA 在 NAVSIM 上的 F-3**：两者的适配器分别依赖 nuScenes devkit
  与 nuScenes `sd_token`，NAVSIM 数据不是这个格式 ⇒ **语料侧接口缺失，非模型侧不可测**。

**两条必需对照臂在两个新语料上全部照跑**：G-VS 的 `random_init` + `position_only`（各 8 格），
F-3 的灰斑 `ctrl` 臂（16 格）。一格未省。

## 2. 核心问题的回答

| 结论 | 复现 | 判定 |
| --- | --- | --- |
| **LTF 的 F-3 FAIL** | NAVSIM **复现**（$R$ +0.008 [−0.068, +0.078]，$b_{ghost}$ −0.0249 vs G1 −0.0245）；前车急刹**无从检验**（基线响应消失） | ✅ **可检验处复现** |
| **DiffusionDrive 高 G-VS + 零 F-3 响应** | 「零 F-3 响应」**3/3**；「高 G-VS 选择性」**1/3** | ⚠️ **一半复现** |

## 3. 新轴 vs 老轴的稳健性（工单要求的比较）

| 轴 | 跨场景 | 跨数据源 | 性质 |
| --- | --- | --- | --- |
| 老 G | 0/3 | 0/4 | **点估计塌陷**（+0.070 → +0.011） |
| 老 F① | 0/3 | 1/4 | 两个 PASS 全掉 |
| 老 C-hazard | 6/6 | 8/8 | 最稳 |
| **新 G-VS** | 判定 1/4；**点估计 4/4 在 [+0.016, +0.041] 窄带** | 判定 2/4；点估计同上 | **读数稳、判定不稳** |
| **新 F-3** | 六候选全部无基线响应 ⇒ 无从检验 | **LTF FAIL 复现**；「无基线响应」3/3 | 可测时复现；负面结论最稳 |

**结论：新轴比老 G/F 稳（点估计不再塌陷），但未达到 C-hazard 的水平。**
G-VS 的三态判定仍在语料间翻，原因是效应量（0.016~0.041）与 CI 半宽（0.015~0.035）同量级。

## 4. 一条本轮才暴露的不利发现（未因是自家新轴而放松标准）

**NAVSIM 上 `position_only` 地板（只用 token 坐标）超过了 trained mIoU**：
DDv2 0.371 vs 0.355；SimLingo 0.411 vs 0.408。

* selectivity 是对 `random_init` 的配对差、坐标先验被抵消 ⇒ **PASS 判定不受影响，无需回退**；
* **但 "G-VS PASS" 不能读成"表征比知道坐标更有用"**。G-VS 目前只支持
  "表征含有超出随机初始化的物体性信息"这一较弱主张。

**若当初只报 selectivity、不设 position_only 臂，本轮就不会暴露它。**
两条必需对照臂里，这次是 position_only 抓到了问题（§GF/A53）。

## 5. 产出物路径

| 路径 | 说明 |
| --- | --- |
| `results/g_vs_leadbrake_{simlingo,dd,ltf,ddv2}.json` | 前车急刹的 G-VS（含三臂） |
| `results/g_vs_navsim_{simlingo,dd,ltf,ddv2}.json` | NAVSIM 的 G-VS（含三臂） |
| `results/f3_occlusion_leadbrake_{simlingo,dd,ltf,ddv2,alpa,autovla}.json` | 前车急刹的 F-3（六候选，含 ctrl 臂） |
| `results/f3_occlusion_navsim_{simlingo,dd,ltf,ddv2}.json` | NAVSIM 的 F-3（含 ctrl 臂） |
| `results/g_vs_f3_replication_report_{zh,en}.md` | **核心交付**：三语料逐条对比 + 新老轴稳健性比较 |
| `variants/{lead_brake,navsim_corpus}/sam_gt/*.png` | 两个语料的 SAM 伪 GT（103 / 397 帧） |
| `variants/{lead_brake,navsim_corpus}/gvs_tokens_*.npz` | token 特征（含随机初始化臂） |

**脚本改动（均为参数化，既有 G1 结果逐位不变）**：
`gvs3_extract_tokens.py` 加 `--pos / --corpus / --crop-center-row`；
`f3_occlusion_necessity.py` 加 `--corpus`；`f3_occlusion_vla.py` 加 `--pos`。

## 6. 并入论文

`results/paper_experiments_section_{zh,en}.md` 新增 **Table 1(v1-rep)**，
紧跟 Table 1(v1) 之后（**同一批读数换语料重测，不是新表**），并在注里给出：
两条核心结论的复现情况、新轴 vs 老轴的稳健性对比、以及 §4 的 position_only 不利发现。
修正案计数 52 → **55**。

## 7. 未完成项

| 项 | 状态 | 理由 |
| --- | --- | --- |
| 两个 VLA 的 G-VS（三语料） | 未测 | video token 布局未接完（§GF/A50） |
| Alpamayo / AutoVLA 在 NAVSIM 的 F-3 | **n/a** | 语料侧接口缺失（依赖 nuScenes devkit / sd_token） |
| G-VS 的判定稳定性改进 | 未做 | 需更细 token 网格 / 更多事件 / 多层特征。**本轮是"换语料重跑"，不含方法改动** |
| B / C 类正例 | 未测 | 三语料都只用了 A（NAVSIM/G1）与 LB（前车急刹） |
