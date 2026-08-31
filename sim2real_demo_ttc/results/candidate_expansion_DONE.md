# 候选池扩展：完成标志

> 工单：[`../../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)
> 承接：`headline_pilot_DONE.md`。本轮全部自行决策登记在 [`amendments.md`](amendments.md) §CE/A27–A38（12 条）。
> 完成日期：2026-08-31。

---

## 1. 每个候选的最终状态

| 候选 | 状态 | G | F① | F② | I | C | 报告 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Alpamayo-R1** (10B) | **部分完成**（G/F 已测，F②/C 预算内未测） | 0.562 [0.502, 0.623]，显著高于置换地板 0.504±0.012 | 0.446 [0.374, 0.519] 不可估 | n.m. | **n/a**（语料单帧） | n.m. | `axis_alpamayo_report_{zh,en}.md` |
| **LTF** (Latent TransFuser) | **完整测完 G/F/I/C** | **+0.070 [+0.017, +0.126]** ← 本工作线**第一个 G 轴阳性** | **0.583 [0.519, 0.639] PASS** | 不可测（斜率 +0.00040，上界 −0.00004） | $I_m$ 0.583 (L6)，行为端 0.160 | **C-hazard PASS** $C_m$ 0.789 [0.725, 0.857]，L6 | `axis_ltf_report_{zh,en}.md` |
| **TransFuser** | **确认与 LTF 为同一候选，不单列**（三项独立检查） | —— | —— | —— | —— | —— | `axis_transfuser_report_{zh,en}.md` |
| **DiffusionDriveV2** | **完整测完 G/F/C**（I 轴 n/a） | +0.025 [−0.044, +0.097] 不可估 | **0.556 [0.501, 0.613] PASS**，b(A) +0.201 | 不可测 | **n/a**（域配对语料无 lidar） | **不可估**（patch-ALL 自检不通过，中位数 0.552） | `axis_diffusiondrivev2_report_{zh,en}.md` |
| **AutoVLA** | **部分完成**（G/F 已测，F②/C 预算内未测），**1 天时间盒内完成，未触发跳过条款** | 0.608 [0.562, 0.654]，$p$ = 7.7e-6，与 LTF 的 0.623 不可区分 | 0.508 [0.451, 0.563] 不可估 | n.m. | **n/a**（语料单帧） | n.m. | `axis_autovla_report_{zh,en}.md` |
| UniAD / VAD / SparseDrive | **未尝试**（工单明确排除：mmcv 1.x 与 sm_120 不兼容） | —— | —— | —— | —— | —— | —— |

**没有一个候选被静默跳过；没有一个"不适用"格是用填充值凑出来的。**

## 2. 本轮的实质发现（不是"又测了几个模型"）

1. **G 轴的第一个阳性**（LTF，+0.070 [+0.017, +0.126]，10 seed sd 0.028）。
   这把此前"三个候选都不可估"从**方法的限制**收窄为**那些候选的限制**——
   同一条 D2cV 地板、同一份刺激集下有候选跨过去了，故地板不是不可跨越的。→ 论文 §4.2.6、§4.4.1。
2. **上一轮推论被同编码器受控对照否定**（§CE/A31）。原推论："F② 注入法不可跨**动作头**族比较"。
   LTF 与 DiffusionDrive 共用同一 `TransfuserBackbone`、动作头不同（连续回归 vs 扩散），
   按原推论 LTF 应当可测，**实测同样不可测**；DiffusionDriveV2 作为第三个成员结果相同。
   ⇒ 可测性由**编码器/注入位点**决定，不是动作头。→ 论文 §4.4.2 已改写。
3. **D2cV 证伪地板对多帧候选不成立**（§CE/A34）。这不是给多帧候选放宽标准，
   而是同一条纪律在不同输入时序性下的正确应用。Table 1 因此按输入时序性分成 (a)(b) 两组，
   **两组的 G 列不是同一个构念，不可跨组比较**。→ 论文 §4.1.1、§4.4.1、§4.5 限制 3。
4. **I 轴读数是权重属性、不是架构属性**（§CE/A37）。LTF 与 DiffusionDrive 同一编码器架构，
   $I_m$ 相差 0.38（CI 不重叠）。但同层对照显示该排序**只在各自峰层上成立**（L0–L4 全部反号），
   这个限定写进了正文而不是脚注。→ 论文 §4.2.5。
5. **G 与 F 的分离在 AutoVLA 上最干净**：原始 G 读数与全表最高者不可区分，
   F① 却是全表最接近 0.5 的（0.508），且 $b(A)$ 与 $b$(D2a) **都**与 0 不可区分——
   不是"反应不特异"，是**根本不反应**。→ 论文 §4.2.7。
6. **DiffusionDriveV2 的 C-hazard 名义值全表最高（$C_m$ 0.845）而我们判它不可估**，
   因为 patch-ALL 充分割集自检不通过（中位数 0.552）。
   自检不通过时，看起来最好的数字恰恰是最不能报的数字。→ 论文 §4.2.3。

## 3. 产出物路径

### 3.1 输入适配器（本轮主要成本项，符合工单 §0 的预期）

| 路径 | 说明 |
| --- | --- |
| `results/ltf_g1_adapter/{ltf_adapter.py, run_g1_cache_ltf.py}` | 直接 `import dd_adapter` 复用**全部前端**，只换 agent 构造 |
| `results/ddv2_g1_adapter/{ddv2_adapter.py, run_g1_cache_ddv2.py}` | 同上 + 现场构造 nuScenes lidar BEV 直方图 + 绕过 PDM metric cache |
| `results/alpamayo_g1_adapter/alpa_g1_cache.py` | VLA 前端另写；prefill 阶段钩子内当场池化 |
| `results/autovla_g1_adapter/{autovla_adapter.py, run_g1_cache_autovla.py}` | 同上；含 transformers 4.49 隔离安装的 bootstrap |
| `results/diffusiondrive_g1_adapter/dd_adapter.py`（改） | 新增 `set_patch` / `_apply_patch`（tokens = image \| all） |

### 3.2 表征缓存（可由上述脚本重建，已在 `.gitignore`）

`variants/n1_d2/{ltf_cache, ddv2_cache, alpa_cache, autovla_cache}/*.npz`
（LTF / DDv2 各 1228 事件；Alpamayo 797、AutoVLA 1112 —— 两个多帧候选因
覆盖预筛（须存在完整历史帧与 6.4 s 未来轨迹）可用率低于单帧候选）
`variants/i_domain/acts_ltf.npz`（432 帧 / 72 场景）

### 3.3 数值产出物

| 文件 | 内容 |
| --- | --- |
| `results/g_axis_{ltf,ddv2,alpamayo,autovla}.json` | 四个新候选的 G 轴全部读数（含地板、几何稳健性、10 seed 折分配稳定性） |
| `results/v_hazard_{ltf,ddv2,autovla,alpa}_*.npz` | 冻结的判别方向（逐层） |
| `results/f_axis_action_counterfactual.json` | F① 六个候选一张表 |
| `results/f_axis_dd_steer_{ltf,ltf_brake,ddv2,ddv2_brake}.json` | F② 注入 + 站内上界标定 |
| `results/c_axis_hazard_{dd,ltf,ddv2}.json` | C-hazard（含 patch-ALL 充分割集自检字段） |
| `results/i_axis_domain.json` | I 轴，新增 `ltf` 条目与 `same_encoder_layer_matched_check` 字段 |

### 3.4 报告（每候选中英双语，期刊 Methods/Results 结构）

`results/axis_{ltf,transfuser,diffusiondrivev2,alpamayo,autovla}_report_{zh,en}.md`（10 份）

### 3.5 并入论文（**同一张 Table 1，不是另开的表**）

`results/paper_experiments_section_{zh,en}.md`，本轮改动：

* **§4.1.1** 待测策略表 2 → 7 行（含 TransFuser≡LTF 的说明），新增"输入时序性"列；
* **§4.1.2** 刺激集：新增五个适配器的复用关系与"为什么在 G1 语料上发现方向"的方法学声明；
* **§4.1.3** C 轴操作化：明确 C-domain / C-hazard 两种配对源 + patch-ALL 充分割集自检为前置门；
* **§4.1.4 / §4.4 引言 / 真源** 修正案计数 26 → **38**，回退计数 5 → **7**；
* **§4.2.1** 新增"候选池扩到 6 个后公开分数的可比性只是名义上的"；
* **§4.2.2 Table 1** 2 模型 → **6 候选**，按输入时序性分 (a)(b) 两组，54 格，n/a 与 n.m. 分别标注并给理由；
* **§4.2.3** 新增 C-hazard 段（DD/LTF 收敛效度）+ DDv2 自检不通过段；
* **§4.2.4** 新增 F① 三种失效形态表（反应不特异 / 反应弱而特异 / 根本不反应）；
* **§4.2.5** 新增 LTF 的 I 轴 + **同层对照自检**（排序只在峰层成立）；
* **§4.2.6（新）** G 轴的第一个阳性：证伪地板不是无法跨越的；
* **§4.2.7（新）** G 与 F 的分离在 AutoVLA 上最干净；
* **§4.2.8** 原 §4.2.6，顺延；
* **§4.3** 不给综合分的论证从"原则上"变成可数的（14 格无可用数字 + 11 格不可估，缺失理由分五类）；
* **§4.4.1** 加入 LTF 阳性与多帧分组；**§4.4.2** 按 §CE/A31 改写推论（动作头 → 编码器）；
* **§4.5** 限制 7 条 → **9 条**（新增多帧候选无替代地板、I 轴缺口全在语料侧）。

### 3.6 修正案

`results/amendments.md` §CE/A27–A38（12 条），其中两条把已得阳性改回不可估：
**A33**（DDv2 的 C-hazard 首版 $C_m$ 0.932 作废重跑 → 不可估）、
**A34**（Alpamayo 的"A vs D2cV 显著"从判据降级为并列报告）。

---

## 4. 未完成项与理由（如实记录，不掩盖）

| 项 | 状态 | 理由 |
| --- | --- | --- |
| Alpamayo-R1 / AutoVLA 的 F② | 未测 | 预算。**不是"不可测"的结论**——两者注入位点（LLM 残差流）与 SimLingo 同构，先验上应当可测。 |
| Alpamayo-R1 / AutoVLA 的 C-hazard | 未测 | 预算。配对与缓存均已具备，是本工作线最直接的下一步。 |
| LTF / SimLingo 的 C-domain / C-hazard 交叉格 | 未测 | 预算。SimLingo 只测了 C-domain，LTF 只测了 C-hazard。 |
| 多帧候选的证伪地板 | **无法在本语料上补** | 需要另造"同类别、同几何、**同相对速度**、只差标签"的负例类，意味着重新挖掘 nuScenes。 |
| DiffusionDriveV2 / 多帧候选的 I 轴 | **n/a** | 语料侧缺失（无点云 / 无时序帧），非模型侧不可测。用零填充会污染读数。 |
