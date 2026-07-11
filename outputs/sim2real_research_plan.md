# 低成本高效 Sim-to-Real 对齐：基于能力探针的数据选择与因果定位的参数高效微调

**Efficient Sim-to-Real Alignment for End-to-End Driving via Capability-Probed Data Selection and Causally-Localized Parameter-Efficient Adaptation**

> 研究对象：SimScale / DiffusionDrive（NAVSIM，PDM 评测）
> 文档性质：论文大纲 + 实施计划 + 理论背景 + 阶段性实证
> 结论状态标注：**[已证实]** 有本项目实验支撑 / **[假设]** 有机制依据但未验证 / **[待验证]** 需后续实验

---

## 摘要 (Abstract)

面向 sim 上预训练、需在真实场景快速适配的端到端驾驶规划器，我们提出一套**低成本**的 sim2real 对齐方法。核心主张有二：

1. **注意力散度 ≠ 因果责任**。我们用**激活修补（activation patching）**证明：origin(real)→transfer(sim) 造成的轨迹偏移，其因果责任集中在**深层融合层（L4–L6，尤以 L6）**，而**注意力散度最大的层（L0/L7）因果贡献几乎为零**。因此以注意力散度定位微调参数是错误的；必须用因果手段。**[已证实，2 token，待扩样本]**

2. **用小规模诊断换取大规模免生产**。第一步在少量 sim-real 配对上，用**能力探针**定位弱能力，产出 `scenario_tag → 弱能力` 的（带精度的）映射与因果层集；第二步在**已有的大规模真实数据集**上，仅凭其自带的 scenario tag 配置训练数据比例，**不做 sim2real 配对、不逐场景 probe**，从而大幅降低数据生产与标注成本。

在此基础上以**因果定位的 LoRA** 做参数高效微调，目标以 GT 为主、teacher 特征对齐补缺、KL 锚防结构崩溃。

---

## 1. 引言 (Introduction)

### 1.1 动机
端到端规划器在仿真域训练后部署到真实域会出现性能退化（sim2real gap）。传统缓解手段——大规模真实数据重训、生成配对合成数据、全参数微调——**成本高**。本文追求"**用最少的数据生产与可训练参数，换取真实域 PDM 的最大提升**"。

### 1.2 问题定义
给定 sim 上训练好的待升级 agent `f_θ`，真实域数据分布 `D_real`，在
- 不生产大规模配对数据、
- 不对大规模数据逐场景标注、
- 仅解冻极少参数（LoRA）

的约束下，最大化 `f_θ` 在真实域的闭环 PDM 分数。

### 1.3 三个驱动性观察
- **O1（症状≠病因）**：可视化/散度分析能指出"哪层的表征变了"，但**不能**指出"哪种驾驶能力退化了"，更不能指出"该改哪个参数"。**[已证实]**
- **O2（能力可复用现成头）**：目标模型自带 BEV 语义头（road/centerline）与 agent 检测头，**探针的一半是免训练、自带 GT 的**。**[已证实]**
- **O3（tag 是廉价代理）**：真实数据集自带 scenario tag；若能在小集上学出 `tag→弱能力` 的可靠映射，就能在大集上用零成本的 tag 做数据选择。**[假设]**

### 1.4 贡献 (Contributions)
1. **因果定位方法**：以激活修补将"注意力散度"证伪为微调定位依据，给出可复现的因果层定位流程。**[已证实]**
2. **能力探针诊断**：复用模型内建头 + 严格探针协议（选择性 control task、逐层扫描、PDM 相关性、因果消融）把"层"翻译为"能力"。**[方法]**
3. **两步低成本流水线**：小集诊断 → `tag↔能力` 映射；大集仅用 tag 配比、免配对免 probe。**[方法]**
4. **因果定位的参数高效微调**：LoRA 位点由因果给出（非散度），目标 GT 为主 + teacher 补缺 + KL 锚。**[方法]**

---

## 2. 相关工作 (Related Work)

| 方向 | 代表工作 | 与本文关系 |
|---|---|---|
| 参数高效微调 PEFT | LoRA；AdaLoRA（重要度分配秩）；SoRA（稀疏低秩）；BitFit；**Surgical Fine-Tuning**（按相对梯度范数选层，专为分布漂移）；RoSA（稀疏+低秩） | LoRA 位点选择的方法基座 |
| 稀疏微调/重要度 | FISH Mask（Fisher 选子集）；Diff/Movement Pruning；Taylor 重要度（Molchanov）；SNIP/GraSP | 参数重要度打分工具 |
| 知识蒸馏/特权学习 | KD（Hinton）；FitNets；**Attention Transfer**；**Learning by Cheating**（特权 BEV 专家→视觉学生，两阶段）；Mean Teacher | teacher 特征对齐目标 |
| **表征对齐 / 防"致盲"** | **Don't Blind Your VLA (2025)** — Visual Representation Alignment(VRA)：把 VLA 的视觉表征锚到**冻结的语义丰富视觉 teacher（SigLIP/CLIP）**，防止窄数据微调导致视觉表征塌缩、损害 OOD 泛化；配 **VL-Think** 诊断套件 | **直接支撑本文 teacher 对齐**：解决"teacher 是谁"（用通用冻结视觉基座，而非任务专用强 planner）与"防结构崩溃"的动机 |
| **表征收敛 / 对齐度量** | **Platonic Representation Hypothesis (Isola 2024)**：以 **kernel** 刻画表征（对应输入的相似核相同即"对齐"）；**CKA (Kornblith 2019)** | 领域对齐的**度量与训练目标**；本仓 `build_bokeh_layer_diff_analysis.py` 已实现 `linear_cka`，可直接复用 |
| 分布鲁棒/重加权 | **Group DRO**（最差组重加权，为分布漂移设计）；Adversarial Reweighting；Focal Loss | 数据配比的理论依据 |
| 数据混合比 | DoReMi（学习数据混合比） | 可学习配比的方法基座 |
| 可解释性 | Probing Classifier（Hewitt–Liang 的 control task 选择性）；**Activation Patching / 因果中介** | 探针与因果定位的方法论 |
| 驾驶域适配 | Tent（测试时仅更新归一化仿射参数） | 极简子集适配的下界基线 |

### 2.1 与最接近工作的定位（vs. Don't Blind Your VLA）
本文与 *Don't Blind Your VLA* 共享"**用冻结视觉 teacher 的表征对齐来保住 OOD 泛化**"这一核心思想，但在两处关键设计上不同，构成本文差异化贡献：

1. **微调位点：全 linear vs. 因果子集**。该工作对 VLA **所有 linear 层**施加 LoRA；本文用**激活修补**实证"注意力散度大的层非因果层"，主张把 LoRA **因果定位到子集（当前证据 → L4–L6）**，以更少参数达到对齐。
2. **数据来源：随机化生成 vs. 已有真实数据按能力选**。该工作用 MPLib 运动规划器**生成 1400 条演示**并在 16 桌×16 物体×位姿扰动上做**域随机化**；本文**不生产数据**，而是在**已有真实数据集**上用 `tag→弱能力` 映射配置比例（成本创新点）。两者可互补：其随机化是"造多样性"，本文是"选薄弱处"。

---

## 3. 预备知识 (Preliminaries)

### 3.1 DiffusionDrive 结构（维度取自真实前向）
- **Backbone**：ResNet34 图像编码器（layer1–4，features_only）+ LiDAR 可学习 latent + FPN。
- **GPT 融合**：4 尺度 × 2 Block = **8 个 encoder self-attention**（`encoder_selfatt_0..7`），每层 **4 heads**；token 数 **320 = 图像 256 (8×32) + lidar/BEV 64 (8×8)**。
- **BEV 解码**：`tf_decoder` 3 层 8 头，query 31 = 1 ego + 30 agents。
- **扩散轨迹头**：DDIM **2 步** × `diff_decoder` 2 层，含 `cross_bev`（轨迹自身 8 采样点）/`cross_agent`(30)/`cross_ego`(1)。
- **输出**：`trajectory`(8 poses × 3，20 模态经 cls argmax 选出) / `agent_states`(30×5)、`agent_labels`(30) / `bev_semantic_map`。
- **内建语义类**：`1=road`(LANE+INTERSECTION)、`2=walkways`、`3=centerline`(LANE+LANE_CONNECTOR)、`4+=agents`。
- **唯一推理随机源**：扩散头 `torch.randn` 噪声 → 因果实验须固定 seed。

### 3.2 PDM 评测
NAVSIM 闭环子指标（碰撞/可行驶区/方向/红灯/进度/TTC/车道保持/舒适）加权得 `score`。**PDM 不可微**——只能作选择/验证指标与配比双层目标，**不能作训练 loss**。

### 3.3 Sim-real 配对设定
`origin` = 真实相机输入；`transfer` = 同场景经风格迁移到 sim（`*_after_transfer.jpg`，开源生成）。两者仅相机图不同，故差异只能经 image backbone → GPT 融合传播。

### 3.4 表征对齐的度量与目标（借鉴 *Don't Blind Your VLA* / PRH）
本文的 teacher 特征对齐（4.5）在方法论上采纳两点外部结论：

- **teacher 选型 = 冻结的语义丰富视觉基座**（SigLIP / CLIP；此即 *Don't Blind Your VLA* 的做法）。它解决了先前悬置的"teacher 是谁"——用**通用视觉基座**而非任务专用强 planner，避免"对齐到自己/对齐到另一个弱模型"的天花板问题。**驾驶特例提醒**：SigLIP/CLIP 偏外观语义，对**几何/深度**覆盖弱；驾驶需兼顾几何，故 teacher 宜在语义基座外**再并入几何自监督基座（如 DINOv2）或深度模型**，与 P2/P5 的 depth 能力对应。
- **对齐度量 = kernel / CKA 对齐**（Platonic Representation Hypothesis）。PRH 用"**相似核**"刻画表征：两表征对应输入的核相同即"对齐"。这给出一个**既是诊断指标、又是可微训练目标**的统一量。**本仓 `build_bokeh_layer_diff_analysis.py` 已实现 `linear_cka`**，可直接用于（a）度量 origin↔transfer / student↔teacher 的表征对齐程度，（b）作为 4.5 中 VRA 项的损失。
- **动机框架 = "别把 VLA 弄瞎"**：窄数据微调会使视觉表征塌缩（"致盲"），损害 OOD。VRA 是**轻量锚定**，与本文 KL-锚（4.5）作用一致、可互补或二选一。

---

## 4. 方法 (Method)

### 4.1 Stage-0 诊断：散度分析及其不充分性

**散度度量**：逐层图像 key 分布的 KL / JS / Wasserstein；PMI（逐点互信息）；注意力重心漂移；熵（聚焦/分散）；图像 key vs lidar key 的质量占比（`img_frac/lid_frac`，几何 vs 外观依赖）。

**head 级细化 [已证实, n=18]**：
- L0(0.185)、L7(0.268) 散度最大；L6(0.012)、L1(0.002) 最小。
- **head 平均掩盖了约 42% 的峰值差异** → 必须保留 head 维。
- L1 由 head0 单头主导（18/18），但幅度极小；L7 弥散（4 head 均高）；L3 较集中（head3）。

**关键：散度是症状，不是病因（见 4.3 因果证据）**。据此 4.2 引入探针把"层"翻译为"能力"，4.3 引入因果修补定位"该改的层"。

### 4.2 能力探针 (Capability Probes)

将"哪种认知能力退化"显式化为可测量的读出头：

| 探针 | 能力 | GT 来源 | 现成度 |
|---|---|---|---|
| P1 车道中位线 | 通行几何 (control) | seg 头 class3 / 地图中位线 | **免训** |
| P2 路况/可行驶 | 路面语义 (segen) | seg 头 class1 / 地图 | **免训** |
| P3 agent 动静 | 动静划分 (segen) | 需相邻帧速度 | 待建 GT+头 |
| P4 动态 agent 预测 | 距离/紧迫 (depth) | 需未来帧 | 待建 GT+头 |
| （建议补）P5 最近距离 | 深度/紧迫 (depth) | 标注距离 | 待建，验证 `img_frac↔depth` |

**探针协议（保证指标可信）**：
1. origin 上探针精度必须高（证明干净输入下能力存在）；
2. **选择性**：配 control task（随机标签）报 `acc − control`，防探针"自我解码"；
3. **逐层扫描**：在 L0–L7 + BEV feat + decoder 上各训线性探针，定位能力"在哪解码、在哪断裂"——**实测替代"深层=语义"的类比**；
4. **PDM 相关性**：逐 token 探针退化须与 PDM delta 相关（验收关卡）；
5. **因果消融**：对齐 teacher 修复后，探针与 PDM 同步恢复。

> 自我纠正记录：早期"L0→control、L7→segen"的层→能力映射被降级为**[假设]**；架构上参与者语义划分实际在 decoder 的 `cross_agent`，segen 候选应从 encoder L7 改到 decoder。

### 4.3 因果定位：为 LoRA 选位（本文核心证据）

**方法**：激活修补。origin 前向缓存各 self-attn 层输出；transfer 前向时把第 L 层输出替换为 origin 的，测轨迹回到 origin 的比例
`recovery(L) = 1 − ‖traj_patch_L − traj_origin‖ / ‖traj_transfer − traj_origin‖`。

**结果 [已证实, n=2, 待扩]**：

| 层 | recovery (aa96f52b) | recovery (ca9e7281) | 18-token JS |
|---|---|---|---|
| L0 | −0.00 | −0.00 | 0.185（最高）|
| L1 | −0.00 | +0.00 | 0.002 |
| L2 | 0.01 | 0.03 | 0.040 |
| L3 | −0.00 | −0.01 | 0.091 |
| L4 | 0.00 | **0.65** | 0.025 |
| L5 | 0.03 | **0.73** | 0.036 |
| **L6** | **0.82** | **0.80** | 0.012（次低）|
| L7 | 0.16 | 0.05 | 0.268（最高）|
| ALL | **1.000** | **1.000** | — |

**三条结论**：
1. `patch ALL = 1.000` 精确 → 8 个 self-attn 输出是 origin/transfer 差异的**充分割集**，方法学自洽。
2. **因果层是 L4–L6（尤 L6）**；L0/L1/L3 因果贡献≈0 → **证伪"L1 导致路面误判"的读图猜测**。
3. **散度最大层 ≠ 因果层**：L0/L7 散度最高但因果弱；L6 散度次低却因果最强 → **按散度放 LoRA 会错过 L6**。这是"症状≠病因"的直接实证，也是本文方法论卖点。

### 4.4 两步低成本流水线

**Step 1（少量、配对、贵但量小）—— 诊断，产出两样东西**
- 数据侧：`scenario_tag → 弱能力` 的映射，每条带 **精度** `precision(T→X)=P(场景在 X 上退化 | tag=T)`；
- 参数侧：因果层集（激活修补）。
- 流程：生成小规模 sim-real 对 → 跑 agent → 按 PDM 退化 > x% 筛"黄金样本"(teacher 高分且 student-transfer 低分) → 过探针取 top-2 弱能力 label → 统计 `tag↔能力`。

**Step 2（海量、免配对、免 probe）—— 训练**
- 直接从 NAVSIM 等**已有真实数据集**按自带 tag 取数；
- **配比作为可学习量**（而非 loss 权重），解一个**预算约束下最大化有用信号**的分配：

  maximize  Σ_X  gap_X · coverage_X,  coverage_X = Σ_T p_T · precision(T→X)
  s.t.  Σ p_T = 1,  KL(p ‖ p_base) ≤ ρ

  用 precision 加权 → 惩罚"覆盖广但命中低"的 tag；held-out PDM 校准（DoReMi/双层）。
- **廉价审计**：对被 up-sample 的桶各抽小样本过探针，核对 `precision` 在大集上是否迁移（保住 O3 假设，成本仅数百次 probe）。

> 关键分工：**数据配比决定"把容量花在哪个能力"；不变性靠 objective（teacher+KL）赋予**。纯 real 微调之所以能改善 sim2real，机制假设是"能力表征练稳后更域鲁棒"——**[待验证]，是本文核心实证点**。

### 4.5 参数高效微调 (Adaptation)

**LoRA 位点**：由 4.3 因果给出（当前证据 → L4–L6），**非** L0/L7；**分能力**定位（P1–P4 各自逐层扫描）。

**反向传播完整性 [已证实概念]**：冻结中间层只置 `requires_grad=False`，**反向传播照穿冻结层**，仅不更新其权重；非连续选层(如只 L4/L6)合法。两端/多点适配器须**联合训练**以处理耦合。

**目标函数**（PDM 只做选择/验证，不入 loss）：
```
L = Σ_capability  w_c · L_GT_c            # 主：有 GT 的能力(轨迹模仿 + seg/agent CE)
  + λ_vra · D_kernel(f_student, f_teacher) # VRA 表征对齐(借鉴 Don't Blind Your VLA):
                                           #   teacher=冻结语义视觉基座(SigLIP/CLIP)+几何基座(DINOv2/深度),
                                           #   D_kernel=CKA/kernel 对齐(PRH), 防"致盲"塌缩
  + λ_kl  · KL(f_θ' ‖ f_θ_pretrained)      # 锚：对冻结预训练模型, 防结构崩溃(与 VRA 同向, 可二选一)
```
- **VRA 取代先前"‖f_student−f_teacher‖ 且 teacher 须在 real 上真强"的模糊表述**：teacher 改用**通用冻结视觉基座**（不需要一个比待测 agent 更强的 planner），对齐用 **CKA/kernel**（PRH）而非裸 L2——更稳、且天然是表征级度量。
- **拒绝可学习 loss 权重**：易造成"标准移动/退化"（对抗式会塌缩到最差单点、追标签噪声）→ 把可学习量移到**数据配比侧**，每样本 loss 标准固定不变。
- **VRA/KL 与因果 LoRA 的分工**：VRA/KL 是"**别把表征弄崩**"的全局锚（防致盲）；因果 LoRA 是"**在正确的层增能力**"的局部适配（4.3 定位）。前者防塌缩、后者促对齐，二者正交互补——这也是本文相较"全 linear 层 LoRA + VRA"的差异（见 §2.1）。

---

## 5. 阶段性实验 (Preliminary — 本轮真实结果)

- **5.1 head 级散度（n=18）**：见 4.1；L0/L7 散度最高、L6 最低、head 平均掩盖 42% 峰值、L1 单头主导稳定。缓存脚本：`scripts/build_bokeh_layer_diff_analysis.py`（补齐 selfatt）；可视化：`build_bokeh_encoder_selfattn_rgb_analysis.py`（含 head 面板）。
- **5.2 激活修补（n=2）**：见 4.3，L6 因果主导、散度≠因果。脚本：`scratchpad/activation_patching.py`。
- **5.3 数据口径纠错 [重要]**：现有评分表 `exp_zero_cams/final_pdm_origin` 实为**相机清零(致盲)**基线，18 token 全 0；`transfer` 表 18 中仅 2 非零。**故"origin(real)>sim"筛选在现数据不可做**，须按 `evaluation_guide.md` **重跑 navmini 全量真实 origin PDM**（生成 mini 真实 origin 评分）后再扩激活修补。

---

## 6. 计划实验 (Planned)

1. **扩激活修补至 ≥30 个真实退化 token**（重跑 PDM 后按 `real>sim 且轨迹差异` 选），验证 L6/深层主导的稳定性；细分图像 token vs lidar token 修补。
2. **探针落地**：P1/P2 用内建 seg 头即刻度量 origin↔transfer IoU 退化；逐层扫描定位能力断裂层；P3/P4/P5 建 GT 与头。
3. **PDM 相关性验收**：探针退化 ↔ PDM delta 的 token 级相关；`img_frac ↔ depth` 检验。
4. **tag↔能力映射**：估 precision，做大集抽样审计验证迁移。
5. **配比优化 + LoRA 微调**：对比 baseline（均匀采样 / 全配对生产）在 held-out PDM 上的增益。
6. **消融**：LoRA 位点（因果 L4–6 vs 散度 L0/7 vs 全层）、目标项（GT / +teacher / +KL）、配比策略。

---

## 7. 成本分析 (Cost)

| | 省掉 | 花掉 |
|---|---|---|
| 数据生产 | 大集**不配对、不生成 sim** | 小集配对生成（navmini 级）|
| 标注 | 大集**不逐场景 probe** | 小集 probe + 大集**小样本审计** |
| 训练 | 冻结绝大多数参数（仅因果层 LoRA）| — |

**效率指标 = 单位训练/数据成本的 PDM 增益 vs baseline（均匀采样、全量配对）**。

---

## 8. 局限与效度威胁 (Limitations & Threats to Validity)

- **样本量**：因果结论目前仅 2 token；head 统计 18 token（且为特定风格迁移）。结论须扩样本坐实。
- **参照非真值**：`recovery` 以 origin 为参照，量的是**域偏移的因果责任**，非"谁更 PDM-正确"。
- **O3 假设未证**：`tag→能力` 映射的可迁移性、以及"real-only 上样改善 sim2real"的机制，都是待验证的核心假设。
- **teacher 身份未定**：若无真正强于待测 agent 的 teacher，特征对齐退化为"对齐自己"，天花板受限。
- **整层修补的粒度**：未分离"该层自身权重致错"与"该层输出承载上游误差"；需输入/输出分别修补进一步分解。

---

## 9. 结论 (Conclusion)

我们把 sim2real 对齐拆成"**便宜的诊断 + 便宜的数据选择 + 便宜的参数适配**"。方法论上的核心贡献是用激活修补**实证**了"注意力散度不能用于定位微调参数"，并给出"探针定位能力 + 因果定位参数 + tag 配比选数据"的低成本闭环。所有强主张均标注了证据状态，后续按第 6 节计划逐条验证。

---

## 附录 A：复现脚本

| 用途 | 脚本 |
|---|---|
| 抓取 encoder self-attn + 特征缓存 | `scripts/build_bokeh_layer_diff_analysis.py` / `scratchpad/capture_selfattn_more_tokens.py` |
| head 级散度可视化 | `scripts/build_bokeh_encoder_selfattn_rgb_analysis.py` |
| 激活修补因果定位 | `scratchpad/activation_patching.py` |
| 重跑 PDM / 生成评分 | 见 `outputs/evaluation_guide.md` §5–6 |

## 附录 B：术语
PMI 逐点互信息 · JS/KL/Wasserstein 分布散度 · PEFT 参数高效微调 · LoRA 低秩适配 · DRO 分布鲁棒优化 · Activation Patching 激活修补 · Probing Classifier 探针分类器 · PDM NAVSIM 闭环评分。
