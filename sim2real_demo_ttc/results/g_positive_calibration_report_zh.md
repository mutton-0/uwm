# T-G：G 轴正向校准——同一刺激集下 DiffusionDrive 与 SimLingo 的语义奠基性对照

> 轴名依据 2026-08-29 定稿的 G/F/I/C/D 命名（映射见 [`axis_naming_alignment.md`](axis_naming_alignment.md)：
> **历史文件中的 R 轴 = 本报告的 G 轴**，且定义已收窄；历史文件中的 R①/R②/R③ = 本报告的 G-①/G-②/G-③）。
> 方法全文依据 [`../docs/g_axis_positive_calibration_diffusiondrive.md`](../../docs/g_axis_positive_calibration_diffusiondrive.md)。
> 预注册见 [`amendments.md`](amendments.md) §FA.0，口径修正见 §FA/A13、§DV/A16。
> 数值产出物：`g_positive_calibration_diffusiondrive.json`、`v_hazard_dd_{vision_mean,region_mean}.npz`；
> 适配器代码：`diffusiondrive_g1_adapter/`。

---

## Motivation／动机

SimLingo 上的 G 轴读数已经给出一个带证伪控制的稳健负结果：主读数（A 类 VRU 突现 vs D2a 几何平衡静态物）
与纯证伪控制 D2cV（同类别 VRU、同几何，唯一差异是单帧模型物理上不可能看到的相对速度）**无法区分**。

这个负结果有两种互斥解释，此前的证据不足以排除其中一种：

- **H-real**：SimLingo 这个标本本身缺乏视觉→危险概念的 grounding（方法论有效，标本真的有病）；
- **H-artifact**：整套读出口径（池化、层选择、对比式方向提取）本身读不出"有 grounding"，换哪个模型都落地板（仪器失效）。

区分二者的唯一方式，是把**同一套读出口径**用在一个**大概率真的有 grounding** 的模型上。
DiffusionDrive 是 nuScenes/NAVSIM 系真实数据训练、任务定义即含碰撞规避的端到端规划器，
若它在同一把尺子下也读不出信号，H-artifact 成立、G 轴方法需要推翻重来；
若能读出清晰信号，SimLingo 的负结果被坐实为**标本属性**，可以写进论文当作真实发现，
而不是留一个"未获效度"的尾巴。

本实验是四轴证明集里**唯一一个方法学正向校准性质的实验**：它补的不是证据广度，而是
**方法学有效性下界**。它对核心贡献是前置条件——只有当 G 轴的尺子被证明能读出"有"，
"公开榜单排名与真实部署排名脱节发生在 G 这一环"这类判断才有意义。

---

## Method／方法

**比较公平性（选型协议 §4 规则 5 的直接应用）。** DiffusionDrive **必须**跑在与 SimLingo
完全相同的刺激集上——项目自有的 nuScenes 鬼探头挖掘语料（G1）+ N1 的 D2a/D2b/D2c/D2cV 负例体系，
**不使用 DiffusionDrive 自己的 NAVSIM/navhard 评测集**。为此新写了输入适配器
`diffusiondrive_g1_adapter/`（本报告的工程产出物）。

| 项 | SimLingo（基线，已有） | DiffusionDrive（本实验） |
| --- | --- | --- |
| 原生训练域 | CARLA (sim) | NAVSIM (real)† |
| 评测刺激集 | nuScenes G1 + N1 负例 | **同一份**，经适配器转换 |
| 图像口径 | resize 保持长宽比 → 裁上部（CARLA 几何对齐） | 居中裁 4:1 去天空 → resize(2048, 512)（与 ghosthead 前端逐像素同套） |
| ego 状态口径 | prompt 写 `Current speed`，**clean 帧锚定** | `driving_command` 直行 one-hot + v/a，**clean 帧锚定** |
| 可读层 | 24 个 decoder layer | 8 个 TransFuser encoder `SelfAttention`（320 token = 256 图像 + 64 BEV latent） |
| 主口径池化 | `vision_mean` | `vision_mean`（256 个图像 token 均值，SimLingo `vision_mean` 的同构物） |
| 方向提取 | 折内 S_dir 上 A vs D2a 逐层线性判别 | 同一流程 |
| 峰层选择 | 折内 S_sel 上 AUC argmax | 同一规则，DiffusionDrive 自己的 argmax |
| 报数 | scene 级 4 折 CV，全部事件都当过 held-out | 同一流程 |

† DiffusionDrive 的训练域来自 ckpt 命名与使用者说明，**未经独立核实**（既有 FINDINGS 已立警示）。

**ego 状态锚定纪律。** 两个条件共用 **clean 帧锚定**的 ego 速度，使 clean/ghost 之间唯一的差异是图像——
与 SimLingo 侧 `prompt_anchor: clean` 的处理**同构**（同一个已知混淆的同一种消除方式）。

**证伪地板的口径（关键，见 §DV/A16）。** 方向与峰层**只**在 A vs D2a 上拟合/选择，
D2cV 等其余负类用**同一条方向、同一个峰层**投影后报数。这才是"证伪地板"的定义——
量的是"这条 A-vs-D2a 方向里有多少只是'A vs 任意 VRU'"。折映射覆盖缓存内全部事件类型的场景。

**预注册主读数。** 同一模型内部 **CV-AUC(A vs D2a) − CV-AUC(A vs D2cV)**，
以 **scene 级 bootstrap（2000 次，两条读数同步取同一批 scene 以保留配对结构）** 给 95% CI。
并列报告：随机方向地板、标签置换零分布、**10 个 CV 折分配 seed 上的稳定性**。

**决判规则（跑数前由工单文档给定，不留"失败"格）。**

| DiffusionDrive 读数 | SimLingo 读数 | 判定 |
| --- | --- | --- |
| 显著高于自身 D2cV 地板 | 落在自身 D2cV 地板 | **H1 成立**：G 轴方法有效，SimLingo 的 FAIL 是标本属性 |
| 也落在自身 D2cV 地板 | 落在自身 D2cV 地板 | **H-artifact 成立**：G 轴读出口径需重新设计 |
| 高于地板但弱于预期 | 落在地板 | **部分成立**：报效应量而非仅显著性 |
| CI 跨 0 | — | **不可估**：本样本量不足以区分 H1 与 H-artifact |

---

## Results／结果

**Table 1. G-axis readout under identical stimuli and identical negative design, across two specimens with divergent native domains (primary pooling `vision_mean`; scene-level 4-fold CV).**

| Model | Native domain | Negative class | n (pos/neg) | CV-AUC | 95% CI | $p$ | 与自身 D2cV 地板之差 | 95% CI (scene bootstrap) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA (sim) | D2a（主读数） | 288 / 283 | 0.515 | [0.467, 0.562] | 0.547 | **+0.035** | [−0.026, +0.097] |
| SimLingo | CARLA (sim) | D2cV（证伪控制） | 288 / 212 | 0.480 | [0.428, 0.531] | 0.434 | — | — |
| DiffusionDrive | NAVSIM (real) | D2a（主读数） | 291 / 283 | 0.559 | [0.512, 0.606] | 0.015 | **+0.009** | [−0.047, +0.065] |
| DiffusionDrive | NAVSIM (real) | D2cV（证伪控制） | 291 / 212 | 0.550 | [0.500, 0.601] | 0.054 | — | — |

**Table 2. Robustness of the primary contrast across 10 CV fold-assignment seeds, and against permutation / random-direction floors.**

| Model | Pooling | main (mean ± sd) | D2cV floor (mean ± sd) | difference (mean ± sd) | difference range | permutation floor | random-direction floor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | `vision_mean` | 0.536 ± 0.023 | 0.534 ± 0.021 | **+0.002 ± 0.024** | [−0.048, +0.035] | 0.494 ± 0.009 | 0.491 ± 0.021 |
| DiffusionDrive | `vision_mean` | 0.547 ± 0.034 | 0.535 ± 0.032 | **+0.012 ± 0.032** | [−0.037, +0.059] | 0.494 ± 0.018 | 0.471 ± 0.013 |
| SimLingo | `region_mean` (敏感性) | 0.551 ± 0.031 | 0.534 ± 0.020 | **+0.017 ± 0.026** | [−0.016, +0.075] | 0.491 ± 0.023 | 0.536 ± 0.042 |
| DiffusionDrive | `region_mean` (敏感性) | 0.519 ± 0.027 | 0.527 ± 0.025 | **−0.008 ± 0.024** | [−0.054, +0.012] | 0.493 ± 0.030 | 0.475 ± 0.010 |

**Table 3. DiffusionDrive: full negative-class panel under one and the same direction and peak layer ($L^*$ = 5, `vision_mean`).**

| Negative class | Semantics | n_neg | CV-AUC | 95% CI | $p$ |
| --- | --- | --- | --- | --- | --- |
| D2a | 类别对照（几何平衡静态物）——主读数 G-① | 283 | 0.559 | [0.512, 0.606] | 0.015 |
| D2cV | 纯证伪控制（同类别 VRU + 同几何，只差速度） | 212 | 0.550 | [0.500, 0.601] | 0.054 |
| D2c | 证伪控制（混 55% 车辆，类别差可见） | 343 | 0.565 | [0.520, 0.610] | 0.005 |
| D2b | 上下文对照（走廊外，混类别）——G-② | 311 | 0.538 | [0.492, 0.584] | 0.109 |
| D2bV | 上下文对照（VRU only） | 78 | 0.546 | [0.476, 0.617] | 0.211 |

**几何稳健性（G 轴必须自证不是在读"大而居中"）。** DiffusionDrive `vision_mean`：
$\rho$(投影, log 面积) = +0.059（$p$ = 0.160）；$\rho$(投影, 离心率) = +0.058（$p$ = 0.166）。
**两项均不显著**，即在补齐刺激集后，DiffusionDrive 的读数没有可检出的几何残余相关
（补齐前离心率项曾为 +0.151，$p$ = 3.03 × 10⁻⁴；该显著性随 D2cV/D2b 负例补齐而消失，
说明它来自负例集不完整而非模型行为）。

> **仪器侧**：适配器把同一份 nuScenes G1 语料 + N1 负例喂进了 DiffusionDrive，
> 两侧的方向提取、峰层选择、CV 报数与证伪地板口径**逐条同构**；两个模型的主读数都显著高于
> 各自的置换零分布与随机方向地板（DiffusionDrive 0.559 vs 0.494/0.471；SimLingo 0.515 vs 0.494/0.491），
> 说明管线本身能造出可读的判别方向。**但决判所依赖的"主读数 − 自身 D2cV 地板"这一差值，
> 在两个模型上的 95% CI 都跨 0，且 10 个折分配 seed 上的 sd（0.024~0.032）**大于**效应量本身** ——
> 本实验的分辨率不足以支撑"质的差异"这一判断。
> **标本侧**：DiffusionDrive 在同一把尺子下的主读数（0.559，$p$ = 0.015）高于 SimLingo（0.515，$p$ = 0.547），
> 且是本轮唯一 D2a 显著的模型；但它的 D2cV 地板同样接近显著（0.550，$p$ = 0.054），
> 两者之差仅 +0.009 [−0.047, +0.065]。**在补齐刺激集之后，没有任何一个口径能支持"质的差异"。**

---

## Discussion／讨论

**1. 三态判定：不可估。** 按跑数前给定的决判规则，DiffusionDrive 的"主读数 − 自身 D2cV 地板"
= +0.009，scene 级 bootstrap 95% CI [−0.047, +0.065] 跨 0；10 个折分配 seed 上为 +0.012 ± 0.032
（**sd 大于效应本身**）。**本实验既不能宣称 G 轴方法有效（H1），也不能宣称它失效（H-artifact）。**
这不是"没做出来"，而是一个有明确数量含义的结论：即使在一个大概率真有 grounding 的标本上、
在与 SimLingo **完全相同**的刺激集下，**当前 G 轴口径能读出的"D2a 相对 D2cV 的增益"上界约为
+0.065 AUC**，而这个量级已经小于折分配噪声（±0.032）的两倍。

**2. 补齐刺激集之后，连"方向性证据"也不再稳健。** 首版（DiffusionDrive 侧 D2cV 仅 141 条，
SimLingo 侧 212 条）曾观察到"DiffusionDrive 相对置换零分布高出更多"这一方向性差异。
把刺激集补齐到两侧完全一致（D2a 283、D2cV 212、D2bV 78）后，该差异**随池化口径翻转**：
`vision_mean` 上 DiffusionDrive 高出置换地板 +0.065、SimLingo +0.021；
而 `region_mean` 上 DiffusionDrive +0.024、SimLingo +0.056。
**两个口径给出相反的排序**，因此本轮**不能**宣称"尺子在 DiffusionDrive 上读出了东西、
在 SimLingo 上没有"。唯一稳健的事实是：DiffusionDrive 的 D2a 读数在 `vision_mean` 上显著
（0.559，$p$ = 0.015），是本轮两个模型四个口径中唯一显著的主读数；但它的 D2cV 地板
同样接近显著（0.550，$p$ = 0.054），两者不可区分。

**2b. 四条口径的差值互相重叠，这是"不可估"最直接的依据。** 10 seed 上的
"主读数 − 自身 D2cV 地板"：SimLingo `vision_mean` +0.002 ± 0.024、SimLingo `region_mean` +0.017 ± 0.026、
DiffusionDrive `vision_mean` +0.012 ± 0.032、DiffusionDrive `region_mean` −0.008 ± 0.024。
四个数值两两之间的距离都在 1 个 sd 以内，**没有任何一对呈现工单决判规则所要求的"质的差异"**；
四个数值本身也都与 0 不可区分。

**3. 主口径与敏感性口径不一致，必须并列陈述。** `vision_mean`（工单指定的主口径）给出
+0.009 [−0.047, +0.065]，`region_mean` 给出 +0.012 [−0.057, +0.090]（单次）／−0.008 ± 0.024（10 seed）。
两个口径的**主读数**差别不小（0.559 vs 0.517），但两者的**差值**都与 0 不可区分。
本实验分辨率的瓶颈是"主读数与地板同步升降"：`vision_mean` 上主读数 0.559 / 地板 0.550，
`region_mean` 上 0.517 / 0.504 —— **地板几乎跟着主读数一起走**，这正是"读出的是任意 VRU 的存在
而非危险"这一解释所预言的模式。**提高分辨率的最直接办法是扩大 D2cV 负例集
（当前 212 条，10 seed 上地板自身的 sd 已达 0.032），而不是换池化或换模型。**

**4. 对核心贡献的位置。** 本实验的结论把四轴框架的适用边界画得更诚实：
**G 轴目前有能力区分"读得出 vs 读不出"（相对置换地板），但还没有能力区分
"读出的是危险概念 vs 读出的是任意 VRU 的存在"（相对 D2cV 地板）。**
因此在合成文档里，G 轴的读数只能支撑"感知端信号强弱"的排序，
不能单独支撑"该模型是否真正奠基于危险概念"的断言——后者需要 D2cV 地板可区分，本轮未达到。

**5. 残余几何相关。** DiffusionDrive 的投影与目标框离心率有弱但显著的正相关
（$\rho$ = +0.151，$p$ = 3.03 × 10⁻⁴），与面积无关（$\rho$ = +0.045，$p$ = 0.287）。
N1 的几何匹配是在（log 面积, 离心率）上做 caliper 匹配的，故这属于匹配残差而非匹配失败；
但它意味着 DiffusionDrive 的 D2a 增益中有一部分可能来自"目标形状更细长"（行人 vs 静物的固有差异），
这一项无法用当前负例体系进一步扣除。

---

## 自我更正记录

1. **主口径的确认（§FA/A13）**：首版脚本在 H1 判定处误取 `region_mean` 作主口径。
   按工单文档 §3 表格明文，主口径应为 `vision_mean`（与 SimLingo 主读数口径一致），已更正。
   两个口径的数字在同一次运行中同时产出、同时落盘，主口径地位由文档在跑数前指定。
2. **证伪地板口径的修正（§DV/A16），这条把结论从"H1 成立"改回了"不可估"**：
   首版对每个负类**各自重新拟合了一条方向**，而项目既有口径（`n1_cv.py`、`n1_report.md`）是
   **共享同一条 A-vs-D2a 方向**。前者量的是"能不能找到一条分开 A 与 D2cV 的方向"，
   系统性偏高，且与已发表数字不可比。改用共享方向口径后，
   DiffusionDrive 的差值由 +0.093 [0.004, 0.174] 变为 +0.047 [−0.028, +0.122]
   （该数字随后又因下述第 5 条的刺激集补齐进一步变为 +0.009 [−0.047, +0.065]）。
   **首版的阳性结果已作废，不得引用。** 该阳性结果在本条修正登记之前已被看到；
   仍然执行修正，是因为口径不一致是硬伤（与 n1_report 可比性是 T-G 全部意义所在），
   且修正方向不利于已写下的结论，不是择优。
3. **连带修正两处**：① 折映射此前只覆盖 POS+NEG 的场景，导致同一方向投影其余负类时
   ~70% 的 D2cV 被静默丢掉（n_neg 212 → 61），已改为覆盖全部事件类型的场景；
   ② 此前未量化折分配漂移，现对主读数−地板加报 10 个 seed 的均值 ± sd 与范围，
   实测 sd（0.024~0.027）与效应量同量级，这一事实本身构成本实验分辨率不足的直接证据。
4. **刺激集不等的修正（§FA/A19），本报告的最终数字来自这一版**：首轮 DiffusionDrive 缓存按
   `--types A D2a D2b D2c --use-matched` 生成，只覆盖 `matched_D2c`/`matched_D2b` 的成员；
   而证伪地板 D2cV 的定义是 "`matched_D2cV` ∩ D2c 类 ∩ VRU"，成员并不全在 `matched_D2c` 里。
   结果 DiffusionDrive 侧只拿到 **141** 条 D2cV（SimLingo 侧 212 条），D2bV 33 条（对 78 条）。
   **这直接违反 T-G 的立身之本**——工单文档 §3 明文"必须用与 SimLingo 完全相同的刺激集"，
   而地板恰恰是判定所依赖的量。补缓存 116 个缺失事件（约 40 s GPU）使两侧
   D2a = 283、D2cV = 212、D2bV = 78 完全一致后重跑：主口径差值由 +0.047 [−0.028, +0.122]
   变为 **+0.009 [−0.047, +0.065]**，10 seed 由 +0.025 ± 0.025 变为 **+0.012 ± 0.032**。
   判定仍为不可估，但证据更弱；且原先"DiffusionDrive 相对置换地板高出更多"的方向性证据
   **随池化口径翻转**（见 Discussion §2），已相应改写。
   另一处连带发现：补齐前 DiffusionDrive 投影与离心率的显著相关（$\rho$ = +0.151，$p$ = 3.03 × 10⁻⁴）
   在补齐后消失（+0.058，$p$ = 0.166），说明那项显著性来自负例集不完整而非模型行为。

5. **未跑 `last_token` 与 `region_max` 等其余池化口径**（工单 §3 表格写"按预算酌情补齐"）。
   原因：主口径与首个敏感性口径已显示瓶颈在 D2cV 地板的样本量，加跑池化不改变该瓶颈。

---

## 对因果链画像的更新（一句话）

> T-G 的读数**限定**了因果链首环的可用范围：在与 SimLingo 完全相同的刺激集上，
> 连一个大概率真有 grounding 的模型也无法与自身的 D2cV 证伪地板区分（+0.009 [−0.047, +0.065]），
> 因此本框架目前只能报"感知端在该域读得出/读不出"，还不能报"是否真正奠基于危险概念"——
> 这条限定必须写进论文，否则 G 轴的所有下游断言都会被高估一档。
