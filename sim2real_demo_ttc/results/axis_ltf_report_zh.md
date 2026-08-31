# 候选扩展：LTF（Latent TransFuser）的 G/F/C 轴读数

> 工单：[`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)。
> 本轮全部自行决策见 [`amendments.md`](amendments.md) §CE（A27–A33）。
> 数值产出物：`g_axis_ltf.json`、`f_axis_action_counterfactual.json`、
> `f_axis_dd_steer_ltf{,_brake}.json`、`c_axis_hazard_ltf.json`、`v_brake_ltf.npz`；
> 适配器：`ltf_g1_adapter/`。

---

## Methods

### 候选与权重

LTF = **Latent TransFuser**：TransFuser 架构，但用可学习 BEV latent 顶替真实 lidar 分支
（本仓库 `transfuser_agent.yaml` 写死 `latent: True`）。权重为 SimScale 官方
`ltf_sim_navtest.ckpt`（224 MB，`ckpt/download_ckpts.sh` 一键获取）。

**与 DiffusionDrive 的关系（本轮最有价值的结构性事实）**：两者**共用同一个
`TransfuserBackbone(latent=True)` 编码器**，只在动作头上不同——
LTF 是单 trajectory query 过 MLP 的**连续回归头**（`TrajectoryHead`），
DiffusionDrive 是 **anchored 扩散头**（20 个 k-means 轨迹锚）。
因此 LTF 构成一次**同编码器、异动作头**的受控对照（见 Discussion §3）。

### 刺激集与适配器

与既有候选**完全同一份**：nuScenes 鬼探头 G1 语料 + N1 负例体系
（A 291 / D2a 283 / D2b 311 / D2bV 78 / D2c 343 / **D2cV 212**），
逐格与 DiffusionDrive 的组规模一致。
适配器 `ltf_g1_adapter/ltf_adapter.py` **直接继承** `diffusiondrive_g1_adapter` 的全部前端
（图像 4:1 裁剪去天空 → resize(2048, 512)、ego 状态 clean 帧锚定、bbox → 8×32 图像 token 网格、
五种池化、行为量 `commanded_speed = ‖traj[0]‖/0.5 s`），只替换 agent 构造，
保证跨模型口径**逐字段同一套**，不是"另写一套差不多的"。

### 三个轴的操作化（与既有候选逐条一致）

* **G**：折内 $S_{dir}$ 上由 $\delta = Z(\text{ghost}) - Z(\text{clean})$ 提逐层线性判别方向，
  折内 $S_{sel}$ 上按 AUC 选峰层，scene 级 4 折 CV 报数；
  主读数 = **CV-AUC(A vs D2a) 减去自身 D2cV 证伪地板**（同一条方向、同一峰层投影两个负类）；
  并列随机方向地板、标签置换零分布、**10 个 CV 折分配 seed 的稳定性**。
* **F①（架构中立）**：行动层反事实测试。操纵①危险有无 = A 的 ghost vs clean；
  操纵②几何混淆 = D2a 的 ghost vs clean；$b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$；
  主读数 b-AUC(A vs D2a)，scene 级 bootstrap。
* **F②（架构相关）**：在概念峰层的图像 token 段注入 $Z' = Z + \alpha\sigma_L\hat v$，
  ±α ∈ {0.5, 1, 2, 4}，主读数为整条阶梯斜率；对照 = 同层 20 seed 随机零分布、特异性、
  termination/recovery，外加**站内上界标定** $v_{brake}^{ltf}$（用 LTF 自身 `commanded_speed`
  对 ego 速度回归后的残差二分为标签，扣掉车速主效应后逐层判别，held-out AUC 峰值 0.684@L5）。
* **C-hazard**：G1 同事件的 **clean↔ghost 配对真实输入互换**，逐层替换全部 320 个融合 token，
  $\mathrm{recovery}(L) = (v_{patch} - v_{ghost})/(v_{clean} - v_{ghost})$；
  取 $|v_{clean} - v_{ghost}|$ 最大的 14 个 A 类事件（运行时分母 < 0.05 者剔除）；附恢复剖面形状诊断。

---

## Results

**Table 1. G-axis readout for LTF on the shared G1 stimulus set, against its own D2cV falsification floor. Group sizes are identical to those used for DiffusionDrive.**

| Pooling | Negative class | n (pos/neg) | CV-AUC | 95% CI | $p$ | 主读数 − D2cV 地板 | 95% CI (scene bootstrap) | 10 seed 上的差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean`（主口径） | D2a | 291 / 283 | 0.623 | [0.577, 0.668] | 3.76 × 10⁻⁷ | **+0.070** | **[+0.017, +0.126]** | **+0.056 ± 0.028** |
| `vision_mean` | D2cV（证伪地板） | 291 / 212 | 0.553 | [0.502, 0.603] | 0.043 | — | — | — |
| `region_mean`（敏感性） | D2a | 291 / 283 | 0.613 | [0.567, 0.658] | 3.00 × 10⁻⁶ | **+0.085** | **[+0.031, +0.140]** | **+0.082 ± 0.013** |
| `region_mean` | D2cV | 291 / 212 | 0.528 | [0.477, 0.579] | 0.281 | — | — | — |

随机方向地板 0.516 ± 0.028、标签置换零分布 0.529 ± 0.023（`vision_mean`）；峰层 $L^\*$ = 6。
10 个折分配 seed 上的差值范围为 **[+0.017, +0.105]**（`vision_mean`）与 **[+0.061, +0.102]**（`region_mean`），
**10/10 个 seed 全部为正**。
几何稳健性：$\rho$(投影, log 面积) = +0.049（$p$ = 0.240）、$\rho$(投影, 离心率) = −0.007（$p$ = 0.871），
**两项均不显著** —— 该读数不是在读"大而居中"。

**Table 2. F-axis readouts: architecture-neutral action-level counterfactual (F①) and representation injection (F②).**

| Readout | Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| F① | $b$(A)，危险帧引起的动作变化 [m/s] | **+0.0203** | [+0.0064, +0.0330] | — | 显著 ≠ 0 |
| F① | $b$(D2a)，几何混淆引起的动作变化 [m/s] | −0.0156 | [−0.0397, +0.0039] | — | 与 0 不可区分 |
| F① | **b-AUC(A vs D2a)** | **0.583** | **[0.519, 0.639]** | 5.85 × 10⁻⁴ | **PASS** |
| F① | b-AUC(A vs D2cV) | 0.555 | [0.487, 0.614] | 0.035 | 边缘 |
| F② | 注入 $v_{hazard}^{ltf}$@L6 的 ±α 全阶梯斜率 [m/s per σ] | +0.00040 | [−0.00065, +0.00143] | — | 未超同层零分布（+0.00019 ± 0.00088，20 seed，$z$ = +0.24） |
| F② | **注入站内上界 $v_{brake}^{ltf}$@L5** | **−0.00004** | [−0.00011, +0.00001] | — | **未超零分布**（$z$ = −0.76） |
| F② | 特异性 \|Δ横向\|/\|Δv\| @α = +4 | 43.9 | — | — | 效应几乎全在横向 |

同批场景中真实危险诱发的减速为 +0.0452 m/s [+0.0077, +0.0754]（显著 ≠ 0），
即通路利用率 $F_m$ 的分母可用；但分子（注入峰值 +0.0002）与 0 不可区分 ⇒ **$F_m$ 无法给出有意义估计**。

**Table 3. C-hazard readout: where the hazard-induced behavioural change enters the network (G1 clean↔ghost paired input swap, 12 most-degraded A events).**

| Quantity | Value | 95% CI | Reference |
| --- | --- | --- | --- |
| patch-ALL recovery（充分割集自检） | **+1.000** | — | 接近 1 即 8 层融合 token 是充分割集 |
| $C_m$ = top-2 层 recovery 占比 | **0.789** | [0.725, 0.857] | 弥散基线 0.250 |
| 恢复剖面 Spearman(层号, recovery) | +0.929 | — | > −0.3 ⇒ 内部峰，top-2 公式适用 |
| 责任层 argmax 众数 / 归一化熵 | **L6** / 0.576 | — | — |

判定：**PASS**（$C_m$ 的 CI 完全高于弥散基线，且剖面为内部峰、公式前提成立）。

> **仪器侧**：LTF 与 DiffusionDrive 的组规模逐格相同（A 291 / D2a 283 / D2cV 212 …），
> 适配器直接继承同一套前端，因此二者的 G/F①/C 读数是严格同口径的。
> G 轴的正结果在**两种池化、10 个折分配 seed 上一致为正**，且几何稳健性两项均不显著，
> 排除了"读的是大而居中"这一解释。C 轴的 patch-ALL = +1.000 通过充分割集自检。
> **F② 的零结果已由站内上界标定分流到仪器侧**：连按构造必然有效的 $v_{brake}^{ltf}$
> （held-out AUC 0.684）都推不动纵向输出。
> **标本侧**：LTF 是本工作线迄今**唯一一个 G 轴越过自身 D2cV 证伪地板、
> 且 F① 行动层反事实同时通过**的候选。它的危险信号在编码器深层（L6）可读、
> 也确实驱动了动作（b(A) 显著为正、b(D2a) 不显著），但**表征层注入推不动它的纵向输出**。

---

## Discussion

**1. LTF 是本工作线的第一个 G 轴阳性，这直接改变了此前所有"不可估"的解读。**
前三轮里 SimLingo、DiffusionDrive、DiffusionDriveV2 的 G 轴主读数减证伪地板全部跨 0，
当时无法排除"整套 G 轴口径读不出'有'"这一仪器侧解释（这正是 T-G 正向校准实验想解决而没解决的问题）。
LTF 在**完全相同的刺激集、负例体系与统计口径**下给出 +0.070 [+0.017, +0.126]（主口径）与
+0.085 [+0.031, +0.140]（敏感性口径），10/10 个 seed 全正。
**这就是此前缺失的正向校准**：G 轴口径**能**读出"有"，因此其余候选的"不可估"可以更有把握地
归为标本属性而非仪器缺陷。

**2. LTF 同时是 F① 的第一个阳性，两个轴在同一候选上一致。**
b(A) = +0.0203 [+0.0064, +0.0330] 显著为正而 b(D2a) 不显著，
b-AUC = 0.583 [0.519, 0.639]（bootstrap P(AUC ≤ 0.5) = 0.005）。
即 LTF 不但"看见了"，而且"看见的东西确实驱动了动作"——
这是四轴因果链上 G→F 两环**同时接通**的第一个实例。

**3. 同编码器受控对照推翻了上一轮的一个推论（见 §CE/A31）。**
上一轮把 DiffusionDrive 的 F② 不可测归因于其 anchored 扩散头，
并据此在论文正文写下"F 轴不可跨**动作头**族比较"。
LTF 与 DiffusionDrive 共用同一编码器、动作头是连续回归头，按该推论应当可测——
**实测同样不可测**（$v_{hazard}^{ltf}$ 斜率 +0.00040，站内上界 $v_{brake}^{ltf}$ 仅 −0.00004，均未超零分布）。
因此正确的归因是：**注入法的可测性由编码器/注入位点决定，不由动作头决定**。
SimLingo（ViT + LLM 残差流，注入 driving query 位置）可测且可被解析 Jacobian 独立复现；
TransFuser 系编码器（两种动作头都一样）不可测。
一个合理但**本轮未验证**的机制假说：融合 token 经 `_bev_downscale` + BEV 上采样 +
transformer decoder 的 cross-attention 之后，σ 尺度的 token 级扰动被稀释。

**4. C-hazard 把危险信号的进入位置定在 L6，与 DiffusionDrive 一致。**
两个共用编码器的模型在同一配对下都把责任层定在 **L6**（$C_m$ 分别为 0.801 与 0.789），
而 DiffusionDrive 的 **C-domain**（sim↔real 域配对）责任层也是 L6。
即：**同一层既是危险信号进入网络的位置，也是渲染域失效进入的位置**——
对该编码器族而言，L6 是一个"信息汇聚点"，这对定向修复的位点选择是直接可用的结论。

**5. 本读数不能声称的东西。** ① F② 的零结果**不是**"LTF 的通路断裂"——
F① 已证明它的动作确实随危险变化；零结果只说明**该注入操作化在该编码器上无分辨力**。
② G 轴阳性是在 nuScenes 鬼探头语料上得到的，LTF 的原生训练域是 NAVSIM；
本读数不构成对其 NAVSIM 表现的任何推断。
③ C-hazard 基于退化最狠的 A 类事件（k = 14，运行时分母守卫后），是**退化条件下**的集中度，不是全集平均。

---

## 自我更正记录

1. **F② 归因的更正（§CE/A31）**：上一轮把 DiffusionDrive 的注入不可测归因到动作头，
   本轮的同编码器受控对照否定了该归因。论文正文 §4.4.2 的推论句已相应改写为"不可跨**编码器**族比较"。
2. **C-hazard 的 patch 范围修正（§CE/A32）**：首版只 patch 前 256 个图像 token，
   patch-ALL 充分割集自检不通过（LTF −0.030）。改为 patch 全部 320 个融合 token 后
   patch-ALL = +1.000。首版数字（C_m = 0.989）**作废**。
3. **TransFuser 未单列（§CE/A28）**：本仓库的 `transfuser_agent.yaml` 即 LTF 配置（`latent: True`），
   官方 ckpt 清单里 TransFuser 系只有 `ltf_sim_navtest.ckpt`，且本刺激集无 lidar，
   带 lidar 的 TransFuser 在本条件下不可执行。故以 LTF 覆盖 TransFuser 系，如实写明而非静默跳过。
