# T-I：域不变性（I 轴）——同一域配对下两个模型的激活散度、干涉角与行为端敏感度

> 轴名依据 2026-08-29 定稿的 G/F/I/C/D 命名（映射见 [`axis_naming_alignment.md`](axis_naming_alignment.md)）。
> 预注册与设计偏离登记见 [`amendments.md`](amendments.md) §FA.0 / §FA.1 偏离 1。
> 数值产出物：`i_axis_domain.json`、`v_domain.npy`（SimLingo）、`v_domain_dd.npz`（DiffusionDrive）。

---

## Motivation／动机

I 轴（Invariance，域不变性）检验的不是"某个域下模型好不好"，而是**经 G、F 验证成立的
"奠基—动作"因果机制，在环境切换下是否保持不变**（理论谱系：Invariant Risk Minimization,
Arjovsky et al. 2019）。它在因果链上排在 G、F 之后：即使模型看对了、也确实由看对的东西驱动了
动作，这条链条仍可能是某一个渲染风格下的特定产物。

这一环对核心贡献的意义最直接：公开榜单（nuScenes/NAVSIM 开环分数）是在**单一域**上打的分，
按定义无法观测"换个域这条机制还在不在"。若 I 轴读数在两个模型上给出与榜单不同的排序，
那么"榜单排名与真实部署排名脱节"就有了一个**可测量的、指向具体环节的**解释，
而不只是一句一般性的怀疑。

本实验同时补上方向发现计划的 **Stage B6 缺口**：$v_{domain}$ 此前从未被提取，
而 I 轴的干涉角公式 $\theta_L = \cos(v_{domain}, v_{hazard})$ 依赖它——不做这一步，I 轴（开环与
闭环版本都）根本算不出来。

---

## Method／方法

**域配对。** 本仓库内不存在 3DGS 重建资产（全盘检索确认；`/data/zihao/HUGSIM` 仅有渲染器代码，
无本项目场景）。故按预注册的偏离 1 使用 `ghosthead_v1` 已有的**同几何双渲染配对**：

| 侧 | 来源 | 含义 |
| --- | --- | --- |
| sim | `renders/<scene>/frames.mp4` | CARLA 引擎渲染 |
| real | `ghosthead_result/<scene>/seg1p0/<scene>_seg1p0.mp4` | 世界模型生成的真实感重绘 |

同一场景、同一构图、同一 actor 与 ego-GT，唯一变量是渲染风格 ⇒ 因果结构上与选型协议 §2 的
P1「域配对 real↔3dgs」等价（都是 do(appearance)）。原始帧为 1600×900，与 nuScenes 同分辨率，
因此两个模型都能走**各自的原生预处理**，不需要为迁就对方而改口径。
可用配对：**72 个 scene-variant × 3 个时刻 = 216 对**。两侧 ego 速度完全相同（同场景同帧号）。

**读数（选型协议 §3③）。**

$$D_L=\frac{\mathbb{E}_i\lVert Z_L(x_{real}^{(i)})-Z_L(x_{sim}^{(i)})\rVert_2}{\mathbb{E}_i\lVert Z_L(x^{(i)})\rVert_2},\qquad
\theta_L=\cos\big(v_{domain}(L),\,v_{hazard}(L)\big),\qquad I_m=1-D_{L^*}$$

$v_{domain}(L)$ = 配对差 $\delta_i = Z_L(x_{real})-Z_L(x_{sim})$ 的 PC1（不去均值，沿用修正案 A5 的理由：
$\delta$ 的均值方向本身就是域方向），符号按"real 侧投影更大"校准。
$L^*$ 取各模型**自己的概念峰层**（SimLingo：`v_hazard_clean` 的 L\* = 9；DiffusionDrive：T-G 冻结的
L\* = 5），即协议 §4 规则 3「层按角色对齐、不按层号」。

**可读位置。** SimLingo：24 个 decoder layer，主口径池化 `vision_mean`（另报 `query_mean`/`last_token`）。
DiffusionDrive：8 个 TransFuser encoder `SelfAttention`，主口径 `vision_mean`（256 个图像 token 均值）。

**跨模型可比性纪律（协议 §4）。** $D_L$ 取**模型内归一化**的比值形式，这是它可以跨模型比较的唯一理由；
行为端另报**模型内无量纲**的域敏感度 $\mathbb{E}\,|v_{cmd}^{real}-v_{cmd}^{sim}| / \overline{|v_{cmd}|}$。
bootstrap 以 **scene 为重采样单位**（同一场景的 3 个时刻不是独立样本），1000 次重抽。

---

## Results／结果

**Table 1. Domain-pair activation divergence, interference angle and behavioural domain sensitivity for two models evaluated on one and the same CARLA-render ↔ world-model-render pairing (216 pairs from 72 scene-variants).**

| Model | Native domain | Readable layers | $L^*$ | $D_{L^*}$ | 95% CI | $I_m = 1-D_{L^*}$ | $\lvert\theta_{L^*}\rvert$ vs own $v_{hazard}$ | Behavioural domain sensitivity | 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA (sim) | 24 decoder layers | 9 | 0.394 | [0.374, 0.419] | **0.606** | 0.0258 | 0.346 | [0.249, 0.454] |
| DiffusionDrive | NAVSIM (real)† | 8 encoder self-attn | 5 | 0.797 | [0.756, 0.840] | **0.203** | 0.0013 | 0.115 | [0.080, 0.153] |

† DiffusionDrive 的训练域来自 ckpt 命名与使用者说明，**未经独立核实**（既有 FINDINGS 已就此立警示）。
本表把它记为条件信息，所有依赖"谁是 in-domain"的解读都以此为条件。

**Table 2. Layer profile of $D_L$ (primary pooling, `vision_mean`).**

| SimLingo layer | 0 | 4 | 8 | **9** | 12 | 16 | 20 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $D_L$ | 0.372 | 0.371 | 0.393 | **0.394** | 0.394 | 0.424 | 0.368 | 0.303 | 0.495 |

| DiffusionDrive layer | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $D_L$ | 0.033 | 0.001 | 0.667 | 0.056 | 0.633 | **0.797** | 0.730 | 0.956 |

SimLingo 的 $D_L$ 剖面近乎平坦（0.303–0.495，全 24 层极差 0.19）；DiffusionDrive 的剖面在
两个数量级间剧烈振荡（0.001–0.956），且 L1 的 $D_L$ = 0.001 伴随 PC1 解释方差比 EVR₁ = 0.989 ——
该层几乎完全被**与输入无关的可学习 BEV latent** 主导，域差被结构性地压成零。

**Table 3. Interference angle $\lvert\theta_L\rvert = \lvert\cos(v_{domain}, v_{\cdot})\rvert$ at the concept peak layer, against the isotropic reference $1/\sqrt{d}$.**

| Model | 比较方向 | $\lvert\theta_{L^*}\rvert$ | 各向同性参照 $1/\sqrt{d}$ | 是否超出 |
| --- | --- | --- | --- | --- |
| SimLingo ($d$ = 896) | $v_{hazard}^{clean}$ (G 轴) | 0.0258 | 0.0334 | 否 |
| SimLingo | $v_{hazard}^{carla}$ (G 轴, CARLA in-domain) | 0.0058 | 0.0334 | 否 |
| SimLingo | $v_{brake}$ (F 轴行为锚) | 0.0044 | 0.0334 | 否 |
| DiffusionDrive ($d$ = 256 @ L5) | $v_{hazard}^{dd}$ (G 轴) | 0.0013 | 0.0625 | 否 |

> **仪器侧**：域配对是 do(appearance)（几何/actor/ego-GT 全锁死），$D_L$ 与行为敏感度均取
> 模型内归一化的比值形式，满足协议 §4 规则 1、2；两模型的 $D_{L^*}$ 的 scene 级 bootstrap CI 不重叠，
> 排序在统计上稳健。**但 $D_L$ 剖面形态差异极大（SimLingo 近平坦 vs DiffusionDrive 两数量级振荡，
> 且 L1 被常量 latent 主导），按协议 §4 规则 4，架构差异过大的候选应"单列，不强行同表排序"**——
> 故本表的跨模型排序记为**条件结论**，不作为选型判据。
> **标本侧**：在同一条渲染域配对上，SimLingo 的**表征**受渲染风格影响更小（$I_m$ 0.606 vs 0.203），
> 但它的**行为**受影响更大（域敏感度 0.346 vs 0.115），两个排序**方向相反**；
> 两个模型的 $v_{domain}$ 与各自 $v_{hazard}$ 均**近正交**（$\lvert\theta\rvert$ 全部不超过各自的各向同性参照），
> 即域漂移没有在几何上侵蚀危险读出方向。

---

## Discussion／讨论

**1. 本实验最强的一条结论是一个 dissociation，而不是一个排序。**
表征端与行为端把两个模型排成了**相反的顺序**：SimLingo 表征更稳（$D_{L^*}$ 0.394 vs 0.797）
但行为更不稳（0.346 vs 0.115，两组 CI 不重叠）。这直接坐实了选型协议 §3.5 的可迁移 Tip 3：
**域不变性判断不能止步于表征相似度，必须验证同一因果机制在两域下的效应量是否一致**——
"表征稳定"与"因果耦合稳定"是两个独立需要验证的命题。
如果只报 $D_L$，本实验会给出"SimLingo 域不变性更好"的结论；
若只报行为端，会给出恰好相反的结论。四轴框架要求两者并列，正是为了不让任一单侧读数独走。

**2. 对工单 §4 核心假设的回答：不可估。** 工单问的是"每个模型的失效是否系统性地对应
**该模型自己的原生训练域与渲染风格之间的差异大小**"。本实验无法回答，原因有二，均为标本侧限制
而非仪器缺陷：① DiffusionDrive 的原生训练域**未经独立核实**（ckpt 命名之外无证据），
"它的训练域离世界模型真实感渲染更近还是更远"这个前件本身没有确定值；
② 表征端与行为端的排序相反，即使前件确定，也无法用单一的"失效程度"去与之比对。
故三态判定：**不可估**（功效/前提不足，不是"对应关系不存在"）。
可以确定性地报告的是更弱但干净的一条：**$I_m$ 可算、两模型的 $D_{L^*}$ 在统计上确实不同**，
且 I 轴所需的输入 $v_{domain}$ 已被提取并落盘（Stage B6 缺口已补）。

**3. 干涉角全部落在各向同性参照以下，这条信息是有价值的负结果。** 协议 §3③ 把 $\lvert\theta_{L^*}\rvert$
定义为扣分项：干涉角大 = 域漂移持续侵蚀危险读出（几何化的因果混淆）。四组测量全部
不超过各自维度的 $1/\sqrt{d}$ 参照，说明**两个模型的域方向与危险方向近似正交**，
即观察到的域敏感性不是通过"污染危险方向"这条路径起作用的。
这对 post-train 的含义是：域抑制处理（如把 $v_{domain}$ 投影掉）预期**不会**顺带损伤危险读出，
但也**不会**顺带修好它。

**4. 与 C 轴的交叉印证。** DiffusionDrive 的 $D_L$ 在 L5–L7 达到 0.73–0.96，
而 C 轴测得的因果责任峰在 L6、深层 L4–L6 覆盖 8/12 场景——两条独立读数都指向编码器深层融合段。
但注意二者并不等价：$D_L$ 在 L7 最大（0.956）而因果峰在 L6，与 C 轴报告中"散度峰 L7、因果峰 L6"
的分离一致。**域散度大的层不是修复位点**，这一点在本实验中第三次以不同度量复现。

**5. 本读数不能声称的东西。** ① 本实验用的是**世界模型真实感重绘**，不是 3DGS 重建，
故结论的适用范围写作"CARLA 渲染 ↔ 世界模型真实感渲染"，最终结论以麦迪逊阶段的 real↔3DGS 版本为准；
② 跨模型 $D_L$ 排序为条件结论（架构差异过大，见仪器侧）；
③ SimLingo 侧的 $\theta$ 是与一条**已知落在证伪地板上**的 $v_{hazard}^{clean}$ 算的，
"干涉角小"因此对 SimLingo 的信息量弱于对 DiffusionDrive（后者的 $v_{hazard}^{dd}$ 至少显著超出置换地板）。

---

## 自我更正记录

1. **3DGS 反事实场景集（P3）在本仓库内不存在，已用同几何双渲染配对替代**，代价与适用范围
   已在 Results 与 Discussion §5 明写，并在 `amendments.md` §FA.1 偏离 1 登记。这是设计替换，
   不是把不同的东西当成同一个东西：因果结构（do(appearance)、几何锁死）一致，渲染来源不同。
2. **工单 §4 的"3DGS 渲染风格相对各自训练域的直觉距离"未作为预设方向使用**。
   工单原文即写明"不预设方向，让数据说话"；本报告据此把该项处理为**不可估**，
   而不是挑一个方向去附会数据。
3. **DiffusionDrive 的 L1 层 $D_L$ = 0.001 属结构性伪零**，已在 Results 中标注（EVR₁ = 0.989，
   该层被与输入无关的可学习 BEV latent 主导）。该层未被用作 $L^*$，也未参与任何判定。
4. **跨模型排序未按协议 §4 规则 4 的"单列"处理成不可比，而是给出了条件结论**。
   理由：工单 §4 步骤 4 明确要求给出"两个模型的 $I_m$ 排序"。折中办法是给出排序并同时给出
   不可比的理由与限定；本条为对协议纪律的一次有意偏离，特此登记。

---

## 对因果链画像的更新（一句话）

> I 轴的读数**修正**了因果链画像的一个默认假设：跨域稳定性不是一个标量——
> 同一条域配对上，表征端与行为端把两个模型排成了相反的顺序（$I_m$ 0.606 vs 0.203，
> 行为端域敏感度 0.346 vs 0.115，两组 CI 均不重叠），因此"某模型更 domain-invariant"
> 这类主张必须指明是表征端还是行为端，否则它同时为真也同时为假。
