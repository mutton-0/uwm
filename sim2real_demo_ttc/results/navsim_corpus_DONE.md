# 完成标志：NAVSIM/OpenScene 独立复现语料（跨数据源泛化检验）

> 工单：[`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md)。
> 完成日期：2026-09-01。修正案：[`amendments.md`](amendments.md) §NS/A45–A48（累计 48 条）。
> **可行性判定：通过，未触发止损条款**（1 天时间盒，实际用时 < 2 小时）。

---

## 1. 可行性评估结论（工单 §3）

| 检查项 | 结论 |
| --- | --- |
| OpenScene 原始日志在本机 | ✅ `/data/dataset/navsim/dataset/navsim_logs/test/` 147 个 `.pkl` |
| 相机图像 / 点云在本机 | ✅ `sensor_blobs/test/` 219 GB，含 CAM_F0 与 MergedPointCloud |
| 标注支持挖掘判据 | ✅ `gt_boxes` / `gt_names` / **`gt_velocity_3d`** / `track_tokens` |
| 有可复用的数据读取接口 | ✅ `navsim/common/dataclasses.py`（本轮直接读 `.pkl`，理由见挖掘报告） |
| 与 nuScenes 独立采集 | ✅ **但地理上有重叠**（含波士顿、新加坡），已如实限定并做地理不相交敏感性分析 |

**未硬凑缩水语料**：最终语料 A = 397 > G1 的 291，功效**不低于** G1。

## 2. 三条待验证问题的判定（工单 §0 优先级）

| # | 问题 | 判定 |
| --- | --- | --- |
| **1** | **LTF 的 G 轴阳性（+0.070）能否在独立数据源复现？** | ❌ **不复现，且功效充分** |
| 2 | C 轴架构级规律能否再跨一次数据源？ | ✅ **复现（8/8 格）** |
| 3 | F① 三种失效形态分布是否一致？ | ⚠️ **部分一致（1/4 复现）** |

### 问题 1 的完整判定（核心交付）

| | G1 | 前车急刹（同源换场景） | **NAVSIM（换数据源）** |
| --- | --- | --- | --- |
| 主读数 − 地板 | **+0.070 [+0.017, +0.126]** ✅ | +0.032 [−0.098, +0.164] | **+0.011 [−0.038, +0.065]** |
| CI 宽度 | 0.109 | 0.262（2.4×，功效不足） | **0.103（0.94×，功效相当）** |
| 正例 / 地板 n | 291 / 212 | 103 / 31 | **397 / 134** |
| 判定 | PASS | 不可估（**功效受限**） | **不可估，但功效充分 ⇒ 有信息量的不复现** |

**瓶颈在哪里（呼应工单要求）**：**本轮没有瓶颈。** 与前车急刹那次
"LBv 只有 31 个、是语料硬上限"不同，本轮的 CI 宽度**比 G1 还略窄**，
因此"不可估"不能推给功效。地理不相交子集（拉斯维加斯 + 匹兹堡）上更负：**−0.044 [−0.111, +0.038]**。

**塌陷的是差值不是主读数**：A vs D2a 的 CV-AUC 从 G1 的 0.623 升到 **0.630**（$p$ = 3.1e-10），
升上去的是**证伪地板**（0.553 → **0.620**）。
即 LTF 把"A vs 同类别同几何、只差相对速度的 VRU"也分得同样好——
而这是单帧模型**结构性不可能**做到的判别。
**若只看主读数，本轮会得出"阳性复现得更强了"，而那是错的。**

**不宣称**：不能据此断言 G1 的 +0.070 是噪声。准确表述是
**该阳性不具备跨数据源稳健性，不能作为"LTF 具备危险概念"的证据**。

## 3. 成本反转：确认，但来源不是模型侧（工单 §2 明确要求记录）

工单预期"三个 NAVSIM 原生模型接入这批数据应比当初接入 nuScenes 更顺"。**成本确实反转了，但：**

* **模型侧没省**：三个适配器吃的是原始 RGB + 自车速度，与数据源无关，接哪边工作量都是零。
  "NAVSIM 原生"在推理路径上**没有体现**——因为我们本来就没走 NAVSIM 的 dataloader。
* **数据侧省了三处**（全在挖掘环节）：① `gt_velocity_3d` 直接给（nuScenes 靠 2 Hz 差分，
  那正是 §LB/A44 里 32 个 |a| > 10 m/s² 伪影的来源）；② 位置已在 ego 系；
  ③ 类别天然含 `traffic_cone`/`barrier`/`czone_sign`/`generic_object`，
  正是 D2a 所需素材 ⇒ **D2a 匹配后 SMD 仅 +0.02，优于 G1 与前车急刹两个语料**。
* **多了一处成本**：§NS/A46 的裁剪主点行，藏在**模型前端**的常量里，不核对相机内参就静默污染全部读数。

**净结论：跨数据源迁移的成本主要在"数据源差异逐项核对"，不在"模型是不是这个数据源原生训练的"。**

## 4. 产出物路径

| 路径 | 说明 |
| --- | --- |
| `scripts/ns1_navsim_geometry.py` | **数据源接入层**：把 NAVSIM 日志变成 G1 的 `geo` 结构；判据/帧窗口/事件记录全部 import 自 `g1_mine_events`，一行未改 |
| `configs/navsim_corpus.yaml` / `_simlingo.yaml` | 由 `n1_d2.yaml` 派生，mining 段阈值逐字段不动 |
| `variants/navsim_corpus/mining/{events_all.jsonl, navsim_mining_stats.json, matched_*.txt, geo_disjoint_events.txt}` | 语料（15051 事件 / 1880 scene）+ 匹配清单 + 地理不相交子集清单 |
| `variants/navsim_corpus/results/n1_matching.json` | 负例匹配质检（全部 \|SMD\| ≤ 0.04） |
| `results/navsim_openscene_mining_report_{zh,en}.md` | **可行性评估 + 数据源差异逐项核对 + 阈值/匹配质检 + 成本反转分析** |
| `results/axis_navsim_corpus_{dd,ltf,ddv2}_{zh,en}.md` | 三候选的 G / F① / C-hazard 读数（并列 G1 同一读数） |
| `results/cross_corpus_generality_report_{zh,en}.md` | **核心交付**：与 G1 逐条对比，回答"LTF 阳性是否跨数据源复现" |
| `results/g_axis_navsim_{ltf,dd,ddv2}.json`、`g_axis_navsim_ltf_geodisjoint.json` | G 轴读数 + 地理敏感性对照 |
| `results/b1_v_hazard_navsim_simlingo_ns_{vision,region}.json` | SimLingo 的 G 轴（第四个候选，工单列为"视可行性再定"，实际完成） |
| `results/f_axis_navsim_action_counterfactual.json` | 四候选的 F① |
| `results/c_axis_hazard_navsim_{simlingo,ltf,dd,ddv2}.json` | 四候选的 C-hazard |

### 脚本参数化（复用而非复制，均已在 G1 语料上回归检验）

| 脚本 | 新增 |
| --- | --- |
| `results/diffusiondrive_g1_adapter/dd_adapter.py` | `set_crop_center_row()`（默认值不变 ⇒ 既有 nuScenes 读数逐位不变） |
| `results/*/run_g1_cache*.py` | `--crop-center-row`；DDv2 另加 `--corpus {nuscenes,navsim}` |
| `results/ddv2_g1_adapter/ddv2_adapter.py` | `NavsimLidar`（读 `.pcd`，天然 ego 系） |
| `scripts/c_axis_hazard_patch.py` | `--corpus`、`--crop-center-row` |
| `scripts/b1_hazard_clean.py` | `--arm navsim` |
| `scripts/g_axis_dd_readout.py` | `--restrict-events`（地理敏感性分析用） |

## 5. 并入论文正文（不另开表）

`results/paper_experiments_section_{zh,en}.md`：

* **Table 1(c) 新增**：跨数据源复现检验，与 (a) 逐格并列（**同一批读数在第二个数据源上的重测**，不是新表）；
* **§4.2.6 标题与结论收窄**：「证伪地板不是无法跨越的」→「**在 G1 上**，证伪地板不是无法跨越的」，
  并加一段 ⚠️ 框注明跨语料结果；**原读数不删**，只收窄适用范围；
* **§4.4.1** 同步：LTF 阳性排除"地板过高"这一解释的效力限定在 G1；
* **§4.5 限制 2** 同步：**在跨语料意义上，本文尚无任何候选给出稳健的 G 轴阳性**；
* 修正案计数 40 → **48**。

## 6. 未完成项（如实记录）

| 项 | 状态 | 理由 |
| --- | --- | --- |
| Alpamayo-R1 / AutoVLA | 未测 | 工单 §2 明确排除 |
| NAVSIM trainval split | 未用 | 本机只有 test split（147 log）。扩样仍有空间，但本轮功效已足够判决核心问题 |
| G1 的 +0.070 是真效应还是噪声 | **不可判定** | 两语料在相机几何、标注管线、城市构成上都不同，"效应真实但语料敏感"与"效应本就是噪声"在本设计下不可区分 |
| 地理完全不相交的独立复现 | 部分完成 | 已做拉斯维加斯 + 匹兹堡子集（结论不变且更强），但样本量减半 |
| C 轴在本数据源上的族平衡 | 1 vs 3 | 纯 transformer 只有 SimLingo，不如 §CE/A40 的 3 vs 3 均衡 |

---

## 7. 本轮到此为止，等待外部复核

两条泛化检验（同源换场景 / 同场景换数据源）至此都已完成，构成完整的泛化论证。
**不再自主开新方向。**
