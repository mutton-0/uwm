# 候选扩展：DiffusionDriveV2 的 G/F/C 轴读数

> 工单：[`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)（1 天时间盒候选）。
> 本轮全部自行决策见 [`amendments.md`](amendments.md) §CE（A29、A30、A32、A33）。
> 数值产出物：`g_axis_ddv2.json`、`f_axis_action_counterfactual.json`、
> `f_axis_dd_steer_ddv2{,_brake}.json`、`c_axis_hazard_ddv2.json`、`v_brake_ddv2.npz`；
> 适配器：`ddv2_g1_adapter/`。**结论：接入成功，四个读数中 3 个可用、1 个不可估。**

---

## Methods

### 接入过程中必须解决的两个阻塞（都不是"跳过"，而是被解决了）

**阻塞①：发布权重要求真实 lidar。** 两份官方配置（`diffusiondrivev2_{sel,rl}_agent.yaml`）都写
**`latent: False`**；发布权重 `diffusiondrivev2_sel.ckpt`（972 个张量）里含
**232 个 `lidar_encoder.*`、0 个 `latent`**。即它与 DiffusionDrive/LTF 不同，
**不用可学习 latent 顶替 lidar 分支**。
按"每个候选走自己的原生输入格式"这一适配器哲学，我们从 nuScenes **LIDAR_TOP** 现场构造
TransFuser 式 BEV 直方图，逐步复刻其 `_get_lidar_feature`
（256×256 网格、x/y ∈ [−32, 32]、4 px/m、只取 z > 0.2 m 的点、每像素上限 5 后归一化、
`use_ground_plane=False` 故单通道），点云经 `calibrated_sensor` 变换到 ego 系。
**刺激集本身没变**：同一批 G1 事件、同一批帧，只是多读了同 sample 的 LIDAR_TOP。

**阻塞②：推理路径硬依赖 nuPlan PDM metric cache。**
`TrajectoryHead.forward_test_rl` 在算完 coarse + fine 精化、选出候选轨迹之后，
**无条件**调用 `get_pdm_score_para(...)` 用官方 PDM 打分器排序，且**根本不返回轨迹**
（只返回 `{'reward_dict': ...}`）。作者自己在那一行正上方留了注释掉的官方评测出口
`# return {"trajectory": traj_to_score[:,-1]}`。
我们**不改动外部仓库任何一行代码**，在适配器里把 `get_pdm_score_para` 运行时替换为抛出载体异常，
把已算好的候选轨迹带出来取 `[:, -1]` —— 与那行官方出口**逐字等价**，
且发生在所有网络计算完成之后，不改变任何前向逻辑。

### 刺激集、适配器与三个轴的操作化

与既有候选**完全同一份** G1 语料 + N1 负例（A 291 / D2a 283 / D2cV 212 / D2b 311 / D2bV 78 / D2c 343），
组规模逐格与 DiffusionDrive、LTF 一致。适配器 `ddv2_g1_adapter/ddv2_adapter.py` 继承
`diffusiondrive_g1_adapter` 的图像/状态/token/池化/行为量前端，只加一路 lidar。
G / F① / F② / C-hazard 的操作化与 LTF 报告逐条相同，不再重复。

**一处必须声明的连带修正（§CE/A33）**：注入与 patching 脚本原本是为 DiffusionDrive/LTF 写的，
调用时不传 lidar；对那两个模型无妨（它们的 lidar 分支是常量 latent），但会让 DDV2 收到
**全零 lidar 直方图**。首版在无 lidar 条件下跑出的 DDV2 C-hazard 已**作废**，现全部按真实 lidar 重跑。

---

## Results

**Table 1. G-axis readout for DiffusionDriveV2 on the shared G1 stimulus set (group sizes identical to DiffusionDrive and LTF).**

| Pooling | 主读数 CV-AUC(A vs D2a) | 95% CI | $p$ | D2cV 地板 | 主读数 − 地板 | 95% CI | 10 seed 上的差 | 置换地板 | 随机地板 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean`（主口径） | 0.530 | [0.483, 0.577] | 0.214 | 0.505 | **+0.025** | [−0.044, +0.097] | +0.004 ± 0.026 | 0.506 ± 0.033 | 0.503 ± 0.036 |
| `region_mean`（敏感性） | 0.527 | [0.480, 0.574] | 0.265 | 0.513 | **+0.014** | [−0.049, +0.076] | +0.003 ± 0.018 | 0.505 ± 0.034 | 0.520 ± 0.033 |

峰层 $L^\*$ = 1（`vision_mean`）／3（`region_mean`）。
10 个折分配 seed 上的差值范围 [−0.041, +0.051]（主口径），**含 0**。
几何稳健性两项均不显著（`vision_mean`：log 面积 −0.069，$p$ = 0.099；离心率 −0.050，$p$ = 0.237）。
判定：**不可估**（差值 CI 跨 0，且 sd 大于效应本身）。

**Table 2. F-axis readouts.**

| Readout | Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- |
| F① | $b$(A)，危险帧引起的动作变化 [m/s] | **+0.201** | [+0.052, +0.343] | — | 显著 ≠ 0 |
| F① | $b$(D2a) [m/s] | +0.023 | [−0.117, +0.183] | — | 与 0 不可区分 |
| F① | **b-AUC(A vs D2a)** | **0.556** | **[0.5005, 0.613]** | 0.020 | **PASS**（勉强越线） |
| F① | b-AUC(A vs D2cV) | 0.573 | [0.512, 0.633] | 4.93 × 10⁻³ | 见 Discussion §3 |
| F② | 注入 $v_{hazard}^{ddv2}$ / 站内上界 $v_{brake}^{ddv2}$ | *见 `f_axis_dd_steer_ddv2{,_brake}.json`* | — | — | 与 DiffusionDrive/LTF 同族，预期不可测 |

站内上界方向 $v_{brake}^{ddv2}$ 的读取端有效性：逐层 held-out AUC 峰值 **0.646 @L1**
（车速主效应解释 95%+ 方差，已扣除）。

**Table 3. C-hazard readout — sufficiency check FAILS, verdict indeterminate.**

| Quantity | Value | 判据 |
| --- | --- | --- |
| patch-ALL recovery（中位数） | **+0.552** | 应 ≈ +1.0；**充分割集自检不通过** |
| 因运行时分母过小被剔除的事件 | 2 / 14 | 分母守卫（§CE/A32） |
| $C_m$（仅供参考，**不可读作集中度**） | 0.845 [0.704, 0.964] | 前提不成立 |
| 恢复剖面 Spearman / 责任层众数 | +0.810 / L4 | — |

**判定：不可估（充分割集自检不通过）。**
原因是结构性的：DiffusionDrive/LTF 的 lidar 分支是**与输入无关的常量 latent**，
故 clean 与 ghost 两条件之间的全部差异都经过被 patch 的 8 个融合 token 块；
而 DDV2 的 **lidar 输入在两条件之间是不同的真实点云**，
这部分条件相关信息经卷积 lidar 分支**绕过**了被 patch 的模块 ⇒
融合 token 在该配对下**不是充分割集**，逐层占比无从解释。

> **仪器侧**：两个接入阻塞都以**可复核、不改动第三方代码**的方式解决（lidar 直方图逐步复刻、
> PDM 打分器以运行时替换绕开并取作者自己的官方评测出口）。
> 组规模逐格与 DiffusionDrive/LTF 相同，G/F① 严格同口径。
> **C 轴的不可估是被 patch-ALL 自检当场抓出来的**，不是事后解释——
> 这正是该自检存在的意义。
> **标本侧**：DDV2 的动作对危险有**明显响应**（b(A) = +0.201 m/s，是 DiffusionDrive 的 21 倍），
> 且 F① 越过了几何匹配对照；但其表征端的 G 轴读数与自身证伪地板不可区分。

---

## Discussion

**1. 与 DiffusionDrive 的对比给出一条清晰的差异：动作响应强了 21 倍，表征读数没变。**
同一刺激集下 b(A)：DiffusionDrive +0.0095 [−0.030, +0.045]（不显著）→ DiffusionDriveV2
**+0.201 [+0.052, +0.343]（显著）**；b-AUC 0.553（不可估）→ **0.556（PASS）**。
但 G 轴的"主读数 − 证伪地板"两者都不可估（+0.009 vs +0.025）。
即 **V2 的改进体现在动作端而非可读表征端**——这正是四轴分环诊断能分辨、而单一分数不能分辨的差别。
（限定：V2 多了一路真实 lidar，b(A) 的提升不能全部归给架构改进。）

**2. C 轴的不可估是一条有内容的结论，不是缺失。**
它精确地告诉我们：**对多传感器模型，单看融合 token 的 patching 不构成充分割集**。
要给 DDV2 做 C 轴，必须把 lidar 分支的中间激活一并纳入 patch 范围
（或改用同一 lidar、只换相机的配对）。这是下一轮可执行的具体设计，不是"做不了"。

**3. F①「vs D2cV」高于「vs D2a」这一反常需要限定。**
DDV2 的 b-AUC(A vs D2cV) = 0.573 高于 b-AUC(A vs D2a) = 0.556。
D2cV 的设计前提是"唯一差异是相对速度、单帧模型不可观测"（§CE/A34）。
DDV2 **是单帧模型**，故该前提成立；但它多了一路 lidar，
而 **lidar 点云本身携带距离信息**，可能使 D2cV 与 A 的区分部分变得可观测。
本报告因此把 DDV2 的 D2cV 读数标注为**需谨慎**，主判定仍以 A vs D2a 为准。

**4. 本读数不能声称的东西。** ① nuScenes 是 **32 线单雷达**，NAVSIM/nuPlan 是多雷达合并点云，
点密度与覆盖不同 ⇒ 本 lidar 直方图对 DDV2 属**分布外输入**；其读数应读作
"在本刺激集 + 本 lidar 近似下"的结果，**不能**与它在 NAVSIM 上的表现相提并论。
② PDM 打分器被绕开意味着我们取的是**最后一次 fine 精化的轨迹**，
而不是官方评测里由 PDM 排序选出的那一条；这与作者注释掉的官方出口一致，但仍是一处偏离。

---

## 自我更正记录

1. **首版 C-hazard 未喂 lidar（§CE/A33）**，等于把一个要求真实 lidar 的模型置于极端分布外，
   该版结果（$C_m$ = 0.932）**作废**，已按真实 lidar 重跑。
2. **C-hazard 的运行时分母守卫（§CE/A32）**：选样用缓存里 2 帧平均的 gap，
   而 patching 只跑 frame[0]，两者可能相差很多。曾出现单点 recovery = −439（分母 −0.006），
   把 patch-ALL 均值从 +0.50 拖到 −36。加守卫后剔除 2 个事件，均值与中位数一致（+0.56 / +0.55）。
3. **F② 的事件数从 40 降到 30**：DDV2 每次前向都要读一次 nuScenes 点云，
   注入实验有 169 个条件/事件，首版未缓存 lidar 导致 I/O 成为瓶颈。
   加上逐帧 lidar 缓存后仍按 30 事件跑，以控制墙钟时间；该降级只影响功效，不影响口径。
