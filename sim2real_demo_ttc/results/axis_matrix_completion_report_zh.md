# 轴矩阵补齐：DiffusionDrive 的 F 轴与 SimLingo 的 C 轴

> 轴名依据 2026-08-29 定稿的 G/F/I/C/D（映射见 [`axis_naming_alignment.md`](axis_naming_alignment.md)）。
> 预注册与本轮全部自行决策见 [`amendments.md`](amendments.md) §HL（A22–A26）。
> 数值产出物：`f_axis_action_counterfactual.json`、`f_axis_dd_steer{,_brakedd,_escalate}.json`、
> `v_brake_dd.npz`/`.json`、`c_axis_simlingo.json`、`c_axis_shape_diagnostics.json`。

---

## Motivation／动机

上一轮的四轴矩阵有两个空格：**F 轴只在 SimLingo 上测过**（DiffusionDrive 未做 steering），
**C 轴只在 DiffusionDrive 上测过**（SimLingo 无等价的域配对 activation patching）。
两个空格使"公开榜单排名 vs 四轴矩阵"的对比无法逐格对齐——而对齐正是本工作线全部论证的前提：
若四轴矩阵本身残缺，它就不比公开分数更有资格给出排名。

本报告补齐这两格，并在补齐过程中回答一个此前未被问过的问题：
**四个轴的操作化方式，是否在不同动作头架构之间可移植？**

---

## Method／方法

### F 轴（DiffusionDrive）

**① 行动层反事实测试（协议 §3.5 指定的 F 主验证，架构中立，两模型同时跑）。**
与 G 轴的测试完全对称，但读出对象从内部表征换成**最终规划动作**：
操纵 ①因果特征（危险目标有无）= A 类事件的 ghost 帧 vs clean 帧；
操纵 ②几何混淆特征 = D2a 几何平衡负例的 ghost vs clean。
行为量 $b = v_{plan}(\text{clean}) - v_{plan}(\text{ghost})$（正 = 目标出现后减速）。
预注册主读数 **b-AUC = AUC(b(A) vs b(D2a))**，scene 级 bootstrap（2000 次，两组共用同一批重采样 scene）。
零 GPU：两个模型的逐条件规划速度都已在既有缓存里。

**② steering 四件套（架构相关）。** 在 DiffusionDrive 自己的概念峰层
$L^\*$ = 5（T-G 冻结）的**图像 token 段**注入 $Z' = Z + \alpha\sigma_L\hat v$，
$\alpha$ 阶梯 ±{0.5, 1, 2, 4}；主读数 = 整条 ±α 阶梯的最小二乘斜率（修正案 A3）；
对照 ①同层 20 seed 随机方向零分布；②特异性（横向/舒适度）；③termination / recovery。
刺激集 = 与 SimLingo 侧**同一批** S_test A 类事件（同 seed、同三分）。
**架构声明**：DiffusionDrive 是扩散动作头，按计划 Stage D「对无法做解析投影的架构只用经验 steering，
不尝试解析捷径——这是架构决定的，不是退而求其次」，故不跑 Jacobian 解析臂。

**③ 站内上界标定（本轮新增，决定零结果归因）。** 沿用 T1-Q 纪律另造一条**按构造必然有效**的
行为定义轴 $v_{brake}^{dd}$：特征 = 逐条件图像 token 均值激活，
标签 = 该条件下 `commanded_speed` 对 ego 速度回归后的残差是否低于中位数
（**必须扣掉车速主效应**，实测车速解释 95.6% 方差）。
若连它都推不动行为，零结果归**仪器侧**而非标本侧。

**④ 剂量升级测试。** $\alpha$ ∈ {4, 8, 16, 32}，检验是否存在阈值效应。

### C 轴（SimLingo）

口径与 DiffusionDrive 侧**逐条同构**：域配对 = sim（CARLA 引擎渲染）↔ real（世界模型真实感重绘），
同场景同几何，唯一变量是渲染风格；corruption **一律为配对真实输入互换**（禁用噪声破坏）；
指标**一律为连续量**（禁用二值化）：
$\mathrm{recovery}(L) = \big(v_{patch} - v_{real}\big) / \big(v_{sim} - v_{real}\big)$，
$v$ = 模型下发的目标速度。退化样本按 $|v_{sim} - v_{real}|$ 取前 12 个 scene-variant
（与 DD 侧"取 real 退化最狠的 top-12"同规则），帧取 t = 1 s。

**与 DD 侧的一处结构性差异（须并列声明）**：DD patch 的是编码器全部 320 个融合 token；
SimLingo 只 patch **vision token 段**——语言段在两条件之间长度与内容都会变，对应关系无定义
（与 `g2_cache` schema v2 的同一条纪律）。故 SimLingo 的充分割集检验问的是
"24 层的 vision token 是否是差异的充分割集"，而非全序列。

$C_m$ = top-2 层 recovery 占比 / $\sum_L \mathrm{recovery}(L)$，recovery 先 clip 至 ≥ 0，
逐场景先算再汇总，scene 级 bootstrap（5000 次）。

---

## Results／结果

**Table 1. Action-level counterfactual test of the F axis: does the planned action respond to the causal feature (hazard) more than to the geometry-matched confound? Identical stimuli and identical readout for both models.**

| Model | b(A) [m/s] | 95% CI | b(D2a) [m/s] | 95% CI | b-AUC(A vs D2a) | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | **+0.307** | [+0.112, +0.500] | +0.086 | [−0.213, +0.443] | 0.534 | [0.469, 0.592] | 0.157 | 不可估 |
| DiffusionDrive | +0.010 | [−0.030, +0.045] | −0.021 | [−0.063, +0.018] | 0.553 | [0.486, 0.623] | 0.027 | 不可估 |

正例 n = 288／291，负例 n = 283／283（两模型同一批事件）。
以 D2cV（同类别 VRU、同几何、只差速度）为负类时，两模型均为 0.522。
**两个模型的混淆响应 b(D2a) 都与 0 不可区分** ⇒ 动作层面**没有**几何混淆的症状；
但 b-AUC 的 scene 级 CI 都跨 0.5 ⇒ 也**没有**"动作被真危险特异驱动"的证据。

**Table 2. Steering readout of the F axis on DiffusionDrive, with an in-house upper-bound calibration and a dose escalation. All runs use the same 40 S_test events (13 scenes) and inject into the image-token block at $L^\*$ = 5.**

| Injected direction | α ladder | ±α slope [m/s per σ] | 95% CI | Same-layer random null (mean ± sd, n) | $z$ | empirical $p$ | Exceeds null |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $v_{hazard}^{dd}$ (G 轴方向) | ±{0.5,1,2,4} | −0.00000 | [−0.00013, +0.00008] | −0.00000 ± 0.00011 (20) | −0.01 | 0.952 | 否 |
| **$v_{brake}^{dd}$（站内上界）** | ±{0.5,1,2,4} | **+0.00002** | [−0.00013, +0.00021] | −0.00002 ± 0.00005 (5) | +0.69 | 0.667 | **否** |
| $v_{hazard}^{dd}$（剂量升级） | ±{4,8,16,32} | −0.00006 | [−0.00029, +0.00010] | — | — | — | 否 |

$v_{brake}^{dd}$ 的读取端有效性：逐层 held-out AUC 为
L0 0.535 / L1 0.548 / L2 0.608 / L3 0.611 / L4 0.628 / **L5 0.656** / L6 0.610 / L7 0.641，
即它**确实是一条可读的行为定义轴**。

**Table 3. Evidence that the injection reaches the model but the longitudinal output does not move: lateral response scales monotonically with dose while longitudinal does not.**

| α | Δv_plan [m/s] | 95% CI | Δlateral [m] | \|Δlateral\|/\|Δv\| |
| --- | --- | --- | --- | --- |
| +4 | −0.0000 | [−0.0005, +0.0003] | −0.0022 | — |
| +8 | −0.0000 | [−0.0009, +0.0006] | −0.0043 | — |
| +16 | −0.0017 | [−0.0105, +0.0044] | −0.0077 | 4.5 |
| +32 | −0.0020 | [−0.0123, +0.0055] | −0.0203 | 10.0 |
| −32 | +0.0018 | [−0.0017, +0.0081] | +0.0898 | 49.9 |

同批场景中**真实危险诱发的减速**为 +0.0379 m/s [+0.0036, +0.0745]（显著 ≠ 0），
即协议 §3④ 的通路利用率 $F_m$ 的**分母是可用的**；但分子（steering 峰值）在 ±32σ 下仍只有 −0.0020，
且与 0 不可区分 ⇒ **$F_m$ 在本标本上无法给出有意义的估计**。

**Table 4. C-axis readout for SimLingo, and the profile-shape diagnostics that decide whether the top-2-share formula applies. DiffusionDrive shown for contrast.**

| Model | n_layers | patch-ALL recovery | $C_m$ (top-2 share) | 95% CI | Diffuse baseline | Spearman(layer, recovery) | Profile shape | argmax mode | argmax entropy | $L_{80}/L$ | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | 24 | **+1.004** | 0.176 | [0.160, 0.191] | 0.083 | **−0.997** | 级联 | L0 | 0.279 | 0.542 | **不可估**（公式前提不成立） |
| DiffusionDrive | 8 | +1.000 | 0.858 | [0.771, 0.939] | 0.250 | **+0.881** | 内部峰 | L6 | 0.685 | 0.375 | PASS |

SimLingo 的逐层 recovery 均值自 L0 的 1.004 单调降到 L23 的 0.000；
责任层 argmax 在 12 个场景中 7 个落在 L0、4 个落在 L1、1 个落在 L2。

> **仪器侧**：F 轴的两条口径都做过独立标定。①行动层反事实测试是架构中立的（读出对象是动作），
> 两模型可逐格对齐；②steering 口径在 DiffusionDrive 上**被证明无分辨力**——
> 注入确实到达了模型（横向响应随 α 单调且对称，±32σ 下达 0.02–0.09 m），
> 但连**按构造必然有效**的 $v_{brake}^{dd}$（held-out AUC 0.656）都推不动纵向输出。
> C 轴的 patch-ALL = +1.004（SimLingo）与 +1.000（DD）证明两侧的可读层各自都是充分割集，
> $C_m$ 的分母有意义；但 SimLingo 的剖面形状使 top-2 占比公式**前提不成立**。
> **标本侧**：DiffusionDrive 的纵向规划由 anchored 扩散头（20 个 k-means 轨迹锚）主导，
> 表征层扰动在 ±32σ 内不改变锚的选择；SimLingo 的域失效**在视觉输入接口即已进入**并向下游级联，
> 而 DiffusionDrive 的域失效有一个**内部责任层**（L6）。两者的失效"进入位置"不同。

---

## Discussion／讨论

**1. 补齐后的最重要发现不是补上了两个数，而是发现两个轴的操作化方式不可移植。**
F 轴若以 steering 效应量操作化，在连续回归头（SimLingo）上可测、可解析预测，
在 anchored 扩散头（DiffusionDrive）上**在任何测试剂量下都不可测**；
C 轴若以 top-2 层 recovery 占比操作化，在内部峰剖面（DiffusionDrive）上成立，
在级联剖面（SimLingo）上**前提不成立**。
两者都不是"该模型这一轴不好"，而是"该操作化方式在该架构/该剖面形态上不适用"。
**这直接限制了四轴矩阵能做什么**：它可以逐格给出**同口径下的诊断**，
但**不能**把 F 与 C 的原始数值跨模型排名。三线表因此把这两行标为"不可比"，并给出替代的可比量。

**2. F 轴的可比读数（行动层反事实）在两个模型上都是不可估，但两者的失败方式不同。**
SimLingo 对危险帧有显著的减速响应（b(A) = +0.307 m/s，CI 不含 0），
只是它对几何匹配的无害物也有相近的响应，故区分不出来；
DiffusionDrive 则**对危险帧几乎没有任何动作响应**（b(A) = +0.010 m/s，CI 含 0）。
即：一个是"反应了但不特异"，一个是"根本不反应"。
若只看 b-AUC（0.534 vs 0.553），会得到 DiffusionDrive 略优的印象；
看 b(A) 的绝对量则相反。**这是又一处单一标量会把方向搞反的地方。**
（跨模型比较 b 的绝对 m/s 需谨慎：两个规划器的速度区间不同；
但"CI 是否含 0"是模型内判断，不受此影响。）

**3. C 轴补齐后给出一条对 post-train 有直接操作含义的结论。**
SimLingo 的恢复剖面在 L0 即达 1.004 并单调递减，意味着：
把第 0 层的 vision token 换成 sim 侧的，行为**完全**回到 sim 侧。
换句话说，SimLingo 的渲染域失效**全部发生在视觉编码到 LLM 的接口处**，
LLM 主干本身没有额外贡献域敏感性。修复位点因此不是某个中间层，而是**视觉前端**。
DiffusionDrive 则相反：L0/L1 的 recovery 近 0（0.001/0.000），责任集中在深层融合段 L6。
**同一套诊断在两个模型上给出了两个完全不同的修复处方**——这正是四轴框架相对榜单分数的增量。

**4. 本轮不能声称的东西。** ① DiffusionDrive 的 F 轴 steering 结果**不是**"通路断裂"的证据，
只是"该操作化方式无分辨力"；要判定其 F 轴需要另一种对扩散头有效的干预
（例如直接扰动锚选择的 logits，本轮未做）。
② SimLingo 的 C 轴"级联"结论基于 12 个退化场景、t = 1 s、仅 vision token 段，
不能外推到全场景平均或全序列 patching。
③ 两个模型的 C_m 数值**不可直接比较**（弥散基线随层数变化：0.083 vs 0.250）。

---

## 自我更正记录

1. **新增 C 轴公式适用性判据（§HL/A24）**。首版直接套用协议 §3⑤ 的 top-2 占比公式，
   SimLingo 得 $C_m$ = 0.176 > 弥散基线 0.083，会被读成"PASS：失效集中"。
   核查剖面后发现是**单调递减的级联**（Spearman −0.997），公式前提不成立，
   该数值不构成集中度证据。已新增 `scripts/c_axis_shape.py` 对两个模型统一计算三个
   与层数无关的形状量并写回 JSON，SimLingo 的判定改为**不可估（公式前提不成立）**。
2. **新增 F 轴的站内上界标定与剂量升级（§HL/A25）**。首版只跑了 $v_{hazard}^{dd}$ 的零结果，
   无法区分"通路断裂"与"仪器无分辨力"。补跑 $v_{brake}^{dd}$（按构造必然有效）与 ±32σ 升级后，
   零结果被明确归因到**仪器侧**。这一补充改变了结论的性质，不只是增加了一个数。
3. **b(A) 的跨模型绝对量比较受限**：两个规划器的基线速度区间不同
   （SimLingo v_cmd 均值 6.63 m/s，DiffusionDrive 的 commanded_speed 定义为 ‖traj[0]‖/0.5s）。
   本报告的跨模型判断一律只用**模型内**读数（b-AUC、CI 是否含 0），绝对 m/s 仅作模型内描述。
4. **原工单文件缺失**（见 §HL/A23），本报告的实验设计（尤其是站内上界标定、剂量升级、
   形状诊断这三项）均为按项目一贯纪律自行决策，特此登记以便回来核对。
