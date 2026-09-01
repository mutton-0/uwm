# 完成标志：F① 严重度梯度 + 第二场景类型（前车急刹）

> 工单：[`../docs/severity_gradient_and_second_scenario_workorder.md`](../../docs/severity_gradient_and_second_scenario_workorder.md)。
> 完成日期：2026-09-01。修正案：[`amendments.md`](amendments.md) §SG/A41、§LB/A42–A44（累计 44 条）。
> 两个任务并行推进，互不阻塞，均已完成。

---

## 任务一：F① 严重度梯度分析

**状态：完成**（零新推理，全部复用 F① 已有缓存与挖掘元数据）。

### 对工单核心问题的回答

> "'特异性'和'按严重程度分级响应'是不是同一个机制缺陷的两个侧面？"

**在本候选池上共现**：唯一在 F① 上"反应特异"的候选（LTF）也是唯一在**可观测剂量**上
"显示分级"的候选；两个"反应但不特异"的候选在可观测剂量上都判不可估，
其中 DiffusionDriveV2 是**完全平线**。

| 候选 | F① 特异性（既有） | $d_{long}$（可观测剂量）分级 | $v_{close}$（工单主读数）全控后 |
| --- | --- | --- | --- |
| **LTF** | **特异**（b-AUC 0.583 PASS） | **显示分级**（斜率 −0.00169 [−0.00259, −0.00072]，iso $R^2$ 0.091，JT $z$ −4.62） | CI 含 0 |
| SimLingo | 不特异（0.534） | 不可估（−0.0140 [−0.0292, +0.0023]） | CI 含 0 |
| DiffusionDriveV2 | 不特异（0.556，b(A) +0.201 大） | 不可估（$\rho$ +0.018，JT $z$ +0.04，**完全平线**） | CI 含 0 |
| DiffusionDrive（平线基线） | b(A) 不显著 | 不可估 | CI 含 0 |

**强度限定**：$n$ = 3 上的共现，不是已建立的规律；"共现"也不等于"同一机制"。
一条反例方向：DiffusionDriveV2 的 $b(A)$ = +0.201 全表第二大（反应很强），剂量上却完全平线——
**反应幅度与分级能力是两件事**。

### 本任务最有价值的产物是一条判定而不是几条斜率

工单指定 $v_{close}$ 作主读数（动机正确：直接对应"高速窜出"的问法），
但三个主对比候选**全是单帧模型**，$v_{close}$ 恰恰是 D2cV 证伪地板赖以成立的
"结构性不可见量"。我们**执行了工单主读数**，同时把 $d_{long}$ 对照升为必需项（§SG/A41）。

**并且抓到了一个会被误报的结论**：SimLingo 与 LTF 在 $v_{close}$ 上都给出 CI 不含 0 的负斜率
（字面读 = "越紧迫越不刹"，与早期 S1 同一签名），
但加入单帧可见的成像几何（$\log$ 面积 / $|lat|$ / 离心率 / 类别）后，
**四个候选无一存活**。这条早期观察因此被**部分解释掉**：
单变量上仍显著，但不再独立于可见几何。属**解释性修订**，单变量数字逐位未变。

### 产出物

| 路径 | 说明 |
| --- | --- |
| `scripts/f_axis_severity_gradient.py` | 分析脚本（零前向；含单变量 + 基本/全控两个多元规格） |
| `results/f_axis_severity_gradient.json` | 全部读数（三个剂量坐标 × 四个候选 × 单变量/多元） |
| `results/f_axis_severity_gradient_report_{zh,en}.md` | 中英双语报告，期刊 Methods/Results 结构，三线表 |

---

## 任务二：第二场景类型（前车急刹）

**状态：完成**（第一批三个候选：SimLingo / DiffusionDrive / LTF，Tier-S 规模）。

### 语料

全量 nuScenes trainval 850 scene → 候选前车窗口 302 个 → 落盘 **182 事件 / 154 scene**：
**LB = 103**（大幅减速，$-10 \le a \le -3$ m/s²）、
**LBn = 48**（平稳跟车，三变量 caliper 匹配）、
**LBv = 31**（**证伪地板**：同样在减速但幅度小，$-1.5 \le a \le -0.3$）。
schema 与 G1 的 `events_all.jsonl` 逐字段相同，三个适配器与全部分析脚本零改动复用。

### 对工单核心检验的回答

> "鬼探头场景上发现的模式在这个结构不同的场景上是否复现？"

| 模式 | 结果 | 判定 |
| --- | --- | --- |
| **C 轴的架构级规律** | 三候选 × 两场景 = 6 格全部一致；符号无重叠；LTF 与 DD 两场景责任层同为 **L6**；SimLingo 两场景同为级联 | ✅ **复现** |
| **LTF 的 G 轴阳性** | +0.070 → +0.032 [−0.098, +0.164]；10 seed 均值落到 +0.000；CI 宽 2.4 倍 | ⚠️ **不可估**（功效受限，点估计亦降；既不能确认也不能排除） |
| **LTF 的 F① 特异性** | 0.583 → 0.560 [0.471, 0.651]，差 0.023 ≪ CI 半宽；对证伪地板方向一致 | ⚠️ **不可估**（点估计复现，显著性丢于功效） |
| **SimLingo 的 G 轴** | 主读数 0.600（p = 0.0495，显著）但**证伪地板 0.645 更高** | ❌ **被证伪地板反超** |

**四轴的定义全部迁移成功，一行公式未改**——只把事件族名参数化
（`--pos LB --neg LBn --floor LBv`），并在 G1 语料上做了回归检验（LTF 的 G/F 读数逐位复现）。

### 两条值得单独记的结果

1. **新场景的证伪地板第一次使用就抓到一个会被误报的阳性**：
   若只报 SimLingo 的主读数 0.600 与 p = 0.0495，会得到"SimLingo 读出了前车急刹"，而它是错的——
   同一条方向把急刹车与**轻度减速**车分得更开（0.645）。
2. **表征端（G）比机制端（C）更依赖场景**：C 轴 6/6 复现，G 轴 0/3 复现。
   C 测的是"信息从哪一层进入网络"（网络的路由结构），
   G 测的是"某条方向能否分开两类刺激"（**直接依赖这两类刺激是什么**）。
   **四个轴对"换场景"的敏感度不同，这一点此前没有被测过。**

### 产出物

| 路径 | 说明 |
| --- | --- |
| `scripts/lb1_mine_lead_brake.py` | 候选扫描（复用 `g1_mine_events.compute_scene_geometry`，未改一行） |
| `scripts/lb2_build_events.py` | 阈值 + 最优指派 caliper 匹配 + 落盘（schema 同 G1） |
| `configs/lead_brake.yaml` | `mining:` 段与 `n1_d2.yaml` 逐字段相同，只新增 `lead_brake:` 段 |
| `configs/lead_brake_simlingo.yaml` | 由 `n1_d2.yaml` 派生，**只改 work_dir 一行** |
| `variants/lead_brake/mining/{events_all.jsonl, lead_brake_survey.json, lead_brake_mining_stats.json}` | 语料 + 分布调查 + 阈值/匹配质检 |
| `results/lead_vehicle_brake_mining_report_{zh,en}.md` | 挖掘报告：判据、**阈值回调记录 #1–#4**、负例匹配 SMD 质检 |
| `results/axis_lead_vehicle_brake_{simlingo,dd,ltf}_{zh,en}.md` | 三候选的 G / F① / C-hazard 读数（并列 G1 同一读数） |
| `results/second_scenario_generality_report_{zh,en}.md` | **核心交付**：与鬼探头场景逐条对比，回答"泛化了吗" |
| `results/b1_v_leadbrake_simlingo_lb_vision.json` | SimLingo 的 G 轴（新增 `--arm leadbrake`） |
| `results/g_axis_leadbrake_{ltf,dd}.json` | LTF / DD 的 G 轴 |
| `results/f_axis_leadbrake_action_counterfactual.json` | 三候选的 F① |
| `results/c_axis_hazard_leadbrake_{simlingo,ltf,dd}.json` | 三候选的 C-hazard（含 `commitment_layer`） |

### 脚本参数化（复用而非复制，均已在 G1 语料上回归检验）

| 脚本 | 新增参数 |
| --- | --- |
| `scripts/g_axis_dd_readout.py` | `--pos / --neg / --floor / --report-negs / --stimuli` |
| `scripts/f_axis_action_counterfactual.py` | `--work / --pos / --neg / --floor / --models` |
| `scripts/c_axis_hazard_patch.py` | `--work / --pos / --sl-config` |
| `scripts/b1_hazard_clean.py` | `--arm leadbrake` |

---

## 自我更正记录（本轮 4 条，全文见 amendments.md）

1. **§SG/A41**：工单主读数 $v_{close}$ 与"单帧盲于速度"纪律的张力。
   **不改主读数**，把 $d_{long}$ 对照升为必需项；并新增全控多元规格，
   把单变量上那条抢眼的"越紧迫越不刹"降级为"可由可见几何解释"。
2. **§LB/A42**：负例匹配漏了自车车速（SMD +0.600 → +0.239），
   且贪心匹配改为最优指派。**两处都在跑推理之前抓出来**，未污染任何读数。
   教训：换场景时不能照抄上一场景的匹配变量清单。
3. **§LB/A43**：nuScenes 无刹车灯标注，LBv 改用"两臂都在减速"的替代构造；
   **"刹车灯都亮"是未经核实的假设**，记为限制，不得据此把"LBv 反超"读成"模型只看刹车灯"。
4. **§LB/A44**：阈值回调需要一条分位数给不出的**物理上界**。
   按分位数取尾部会把 32 个标注伪影（$a$ < −10 m/s²）当成最严重的正例，已剔除并记录。

---

## 未完成项（如实记录，不掩盖）

| 项 | 状态 | 理由 |
| --- | --- | --- |
| 第二场景的 LBv 样本量 | **31（语料上限）** | 全量 trainval 只有 37 个轻度减速候选，31 个入 caliper。**这是本场景证伪地板功效的硬上限**，换匹配算法无用（已验证）。G 轴与 F① 在该场景的全部"不可估"都应先按功效理解。 |
| 第二场景的 DiffusionDriveV2 / Alpamayo-R1 / AutoVLA | 未测 | 工单第一批范围只要三个候选。因此本场景上的 C 轴架构级规律只有 1 个纯 transformer vs 2 个 TransFuser 样本，弱于鬼探头场景的 3 vs 3。 |
| 第二场景的 I 轴 | **未测（按工单）** | 工单："I 轴不需要新数据，域配对是独立资产，复用现有的"。严格说本轮只检验了 G/F/C 三轴的场景迁移性，不是四轴。 |
| LTF 的 G 轴阳性是否跨场景成立 | **未判决** | 需要扩样。可行路径：LB 阈值放宽到 $a \le -2.0$ 可到 ~150 正例，但真正瓶颈是 LBv，只能靠 nuScenes 之外的语料。 |
| 多帧候选的 $v_{close}$ 剂量分析 | 结构性不可做 | Alpamayo-R1 / AutoVLA 本可测速度维分级，但它们的 $b(A)$ 与 0 不可区分，剂量分析无从谈起。 |

---

## 本轮到此为止，等待外部复核

两个任务均已完成并落盘。**不再自主开新方向。**
