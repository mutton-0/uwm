# T-C：失效集中度（C 轴）——DiffusionDrive 逐层因果修补的集中度读数

> 轴名依据 2026-08-29 定稿的 G/F/I/C/D 命名；与历史文件中旧记号的映射见
> [`axis_naming_alignment.md`](axis_naming_alignment.md)。
> 预注册见 [`amendments.md`](amendments.md) §FA.0。数值产出物：`c_axis_concentration.json`、
> `c_axis_tsne.json`、`figures/c_axis_tsne.png`。

---

## Motivation／动机

四轴因果链的最后一环回答的是**修复成本**：当一个策略在域迁移下退化时，这个退化是可以归因到
网络中一个空间上集中、可辨识的机制（适合低成本定向干预，例如只在两三层上挂 LoRA），
还是弥散分布的表征偏移（必须更大范围重训练）？这就是 C 轴（Concentration，失效集中度）
所刻画的量，理论谱系对应 Localization-for-Editing（Hase et al. 2023；Meng et al. 2022）。

本实验在因果链上的位置：G 回答"看对了吗"、F 回答"看对的东西驱动动作了吗"、
I 回答"这条链换个域还成立吗"，而 **C 回答"不成立的时候，坏在哪、修起来贵不贵"**。
若 C 高，公开榜单分数与真实部署效果的脱节是**可定位、可低成本修复**的；
若 C 低，同样的脱节意味着重训练级别的成本——这两种情况在榜单分数上完全看不出区别，
这正是四轴框架相对开环分数排名的增量所在。

本实验按工单 §5 定位为**复用为主的低成本任务**：DiffusionDrive 的逐层 activation patching
已在既有工作中跑完（`outputs/ghosthead_infer/patching/recovery.csv`），本实验只做口径化的
读数整理、统计推断与判定，**不重新跑因果实验**。

---

## Method／方法

**标本与刺激集。** DiffusionDrive（ckpt `diffusiondrive_sim_navhard.ckpt`）在 ghosthead 域配对
场景集上的推理结果。域配对为同一场景的两路渲染：`transfered` = CARLA 引擎渲染（作参考侧），
`origin` = 世界模型生成的真实感重绘（作退化侧）；同场景、同构图、同 actor、同 ego-GT，
唯一变量是渲染风格。逐层修补取 real 退化最狠的 **n = 12** 个 scene-variant，帧为 t = 1 s。

**因果读数口径（写死，沿用 Zhang & Nanda 最佳实践）。** corruption 一律为**配对真实输入互换**
（禁用噪声破坏）：跑 real 前向，把第 L 层 encoder SelfAttention 的输出整体换成同一场景 sim 侧的
对应激活，测轨迹回到 sim 的比例 $\mathrm{recovery}(L)$；指标一律为**连续轨迹 recovery**（禁用二值化）。
可读层为 TransFuser 编码器的 8 个 `encoder_selfatt`（4 尺度 × 2 block）。
方法自洽性已由既有工作确认：patch 全部 8 层时 recovery = +1.00（12/12）。

**预注册主读数。**

$$C_m \;=\; \frac{\text{top-2 层 recovery 之和}}{\sum_L \mathrm{recovery}(L)}$$

口径纪律三条，均在跑数前登记：
① $\mathrm{recovery}(L) < 0$ 表示"换上参考侧激活反而更差"，不是定位证据，**主读数先 clip 到 0**
再算占比；不 clip 的版本作为敏感性分析并列报告。② **逐场景先算 $C$ 再汇总**，不是先汇总
recovery 再算一个 $C$。③ bootstrap 以 **scene 为重采样单位**（n = 12，5000 次重抽）。

**判据。** 弥散基线 = $2/L = 2/8 = 0.250$，即 8 层均匀分布时 top-2 应占的份额。
三态判定：主读数的 scene 级 bootstrap 95% CI 完全高于基线 ⇒ PASS；完全低于 ⇒ FAIL；
跨过基线 ⇒ 不可估。

**辅助（探索性）。** 对同一批场景的 8 层激活做 t-SNE，按 sim（参考／正常）vs real（退化）着色，
并报全空间 silhouette 系数。**此项明确标注为相关性证据，不能替代 patching 的因果结论。**

---

## Results／结果

**Table 1. Concentration of causal responsibility across the 8 encoder self-attention layers of DiffusionDrive under paired sim↔real input swapping (n = 12 scene-variants, t = 1 s).**

| Readout | Definition | Estimate | 95% CI | Reference | Verdict |
| --- | --- | --- | --- | --- | --- |
| $C_m$（主读数） | top-2 层 recovery 占比，recovery clip 至 ≥ 0 | 0.858 | [0.771, 0.939] | 弥散基线 0.250 | **PASS** |
| $C_m$（敏感性） | 同上，不 clip | 0.347 | [−0.364, 0.962] | 弥散基线 0.250 | 不可估（分母过零，比值不稳） |
| top-1 层占比 | 单层 recovery 占比 | 0.618 | [0.503, 0.744] | 弥散基线 0.125 | — |
| top-3 层占比 | — | 0.934 | [0.887, 0.977] | 弥散基线 0.375 | — |
| top-4 层占比 | — | 0.988 | [0.974, 0.997] | 弥散基线 0.500 | — |

**Table 2. Layer-wise localization of the causal site, and its dissociation from divergence-based symptoms.**

| Layer | mean recovery (clip ≥ 0) | argmax 场景数 | JS(sim‖real) | 1 − CKA |
| --- | --- | --- | --- | --- |
| L0 | 0.001 | 0 | 0.138 | 0.145 |
| L1 | 0.000 | 0 | 0.000 | 0.267 |
| L2 | 0.125 | 1 | 0.038 | 0.012 |
| L3 | 0.008 | 0 | 0.043 | 0.000 |
| L4 | 0.040 | 1 | 0.031 | 0.101 |
| L5 | 0.200 | 2 | 0.044 | 0.112 |
| **L6** | **0.380** | **5** | 0.009 | 0.284 |
| L7 | 0.345 | 3 | **0.288** | **0.775** |

因果峰在 **L6**（recovery 均值 0.380，12 个场景中 5 个的 argmax 落在此层，L4–L6 合计占 8/12 = 0.667）；
而两种散度指标的峰都在 **L7**（JS = 0.288、1 − CKA = 0.775），L6 的 JS 近全层最低（0.009）。
责任层分布的归一化熵为 **0.685**（0 = 完全集中于一层，1 = 8 层完全均匀）。

**Table 3. Auxiliary (correlational) evidence: separability of degraded vs reference activations.**

| Layer | silhouette (sim vs real, 全空间) | n |
| --- | --- | --- |
| L0 | 0.090 | 432 |
| L5 | 0.112 | 432 |
| L6 | 0.073 | 432 |
| L7 | 0.033 | 432 |

t-SNE 图见 `figures/c_axis_tsne.png`（空心圈标出进入因果修补的 12 个场景）。
四层的 silhouette 均 < 0.12，即**在池化激活空间里 sim 与 real 并不形成清晰的两簇**；
且可分性最高的层（L5）既不是因果峰（L6）也不是散度峰（L7）。

> **仪器侧**：corruption 口径（配对输入互换）与指标口径（连续轨迹 recovery）在跑数前写死，
> patch-ALL = +1.00（12/12）证明 8 个 self-attn 是差异的充分割集，$C_m$ 的分母有意义；
> 主读数与弥散基线的距离（0.858 vs 0.250）远大于 scene 级 bootstrap 的宽度（±0.09），
> 判定不依赖单个场景。**未 clip 的敏感性臂 CI 跨零，说明 clip 纪律不是可选项而是比值定义成立的前提。**
> **标本侧**：DiffusionDrive 在 sim→real 渲染迁移下的轨迹退化，其因果责任高度集中于编码器深层
> 融合段（L6 为主、L4–L7 覆盖 8/12 场景），属"可定位、可低成本定向修复"档；
> 定向干预应放在 **L6/L5**，而不是按散度指标指向的 L7/L0。

---

## Discussion／讨论

**1. 对四轴因果链画像的更新。** C 轴给出本轮四个子实验中**最干净的一个阳性结果**：
$C_m$ = 0.858 [0.771, 0.939]，远高于弥散基线 0.250，且 top-1 层单独就占 0.618。
这意味着：一旦 G/F/I 定位出"脱节发生在感知—动作链条的某一环"，C 轴进一步告诉我们，
**该脱节在 DiffusionDrive 上是空间集中的**，修复预算落在"两三层 LoRA"这一档，
而不是"全参重训练"那一档。这正是公开榜单分数完全无法提供的信息：
NAVSIM 开环分数只告诉你退化了多少，不告诉你退化从哪一层进入、也不告诉你修它要花多少钱。

**2. 症状 ≠ 病因，本实验独立复现。** 散度峰（JS/1−CKA，均在 L7）与因果峰（recovery，L6）分离，
而 L6 的注意力散度近全层最低。若按"哪层变化最大就改哪层"的直觉去放 LoRA，会放到 L7/L0，
即**症状最明显但因果贡献不是最强**的位置。t-SNE 的 silhouette 剖面又给出第三种排序（峰在 L5），
三种排序互不重合——这构成一条独立于本项目其他实验的证据：
**可分性、散度、因果责任是三个不同的量，不能互相代替。**

**3. 本读数不能声称的东西。** ① C 高只说明失效**可定位**，不说明"修 L6 就够了"——
后者是更强的主张，须按选型协议 §3.5 的训练无关恢复测试（patch 后用伪闭环 PDM 式评分／碰撞余量
重新评估后果指标）验证，本实验未做，故 C 的判定在报告中限定为"定位成立"，不含"修复位点最优"。
② 既有工作已证明因果层强度**不预测** PDM 退化幅度（相关 ≈ 0）——位点稳定 ≠ 后果强度可预测，
因此不得用 $C_m$ 去反推退化会有多严重。③ n = 12 且样本是"real 退化最狠的 top-12"，
有选择性；本读数刻画的是**退化场景条件下**的集中度，不是全场景平均。

**4. 三态判定结论。** **PASS**（主读数 CI 完全高于弥散基线）。
判定仅适用于 clip ≥ 0 的口径；不 clip 口径为"不可估"，两者已并列报告。

---

## 自我更正记录

1. **负 recovery 的处理在跑数前登记为 clip ≥ 0，事后证明该纪律是必需的**：不 clip 时
   $C_m$ = 0.347，95% CI [−0.364, 0.962] —— 分母 $\sum_L \mathrm{recovery}$ 可以过零，
   比值失去意义。两版数字均已落盘，未择优呈现。
2. **t-SNE 的着色定义偏离工单字面表述**。工单 §5 写"按失效 vs 正常着色"。
   本项目没有逐样本的"失效/正常"标签（PDM 式 total 被二值门离散化、46% 场景两路打平），
   故改用**与 patching 完全同一套的参考/退化定义**（sim = 参考、real = 退化）着色，
   并额外用空心圈标出真正进入因果修补的 12 个场景。此为口径替换，不是标签臆造，特此登记。
3. **未对 SimLingo 跑对照版 C**（工单 §5 步骤 3 标注为"不强制"）。原因：SimLingo 侧没有
   等价的域配对 activation patching 产出，新跑需要重建整条 patching 管线，成本超出本工单
   "复用为主、低成本"的定位。因此本报告的 C 轴读数为**单模型读数**，不做跨模型排序。

---

## 对因果链画像的更新（一句话）

> C 轴的读数把 G→F→I 定位出的失效**从"存在"推进到"可定价"**：在 DiffusionDrive 上，
> 渲染域迁移引起的退化因果集中于编码器深层单层（$C_m$ = 0.858 [0.771, 0.939]，弥散基线 0.250），
> 因此这条脱节属于"两三层 LoRA 可修"的档位，而这一信息任何公开榜单分数都不提供。
