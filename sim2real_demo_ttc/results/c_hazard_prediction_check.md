# 预测检验结果标志：纯 transformer 栈的 C-hazard 剖面

> 检验对象：`candidate_expansion_DONE_supplement.md` §5 登记的可证伪预测（源自 §CE/A39）。
> 修正案：[`amendments.md`](amendments.md) §CE/A40。完成日期：2026-09-01。
> **本轮到此为止，等待外部复核**（见 amendments.md 末尾的收尾说明）。

---

## 1. 被检验的预测（登记在先，测量在后）

> **剖面形状由"残差流是不是唯一通路"决定 ⇒ 任何纯 transformer 栈的 C-hazard 剖面都应当是级联。**

登记该预测时，手上只有两个纯 transformer 候选（Alpamayo-R1、AutoVLA），
**SimLingo 的 C-hazard 尚未测**——它是 InternVL2-1B（Qwen2-0.5B decoder）的纯 transformer 栈，
因此构成一次真正的留出检验，而不是对已有数据的重新描述。

## 2. 判定：**预测被证实**

SimLingo C-hazard（12 事件 / 11 场景 / 24 层；配对 = G1 clean↔ghost；
patch 范围沿用 SimLingo 自己 C-domain 的约定，即 vision token 段）：

| 量 | 值 | 判据 | 结论 |
| --- | --- | --- | --- |
| patch-ALL（充分割集自检） | 均值 +0.997，中位数 **+1.009** | 须落在 [0.7, 1.3] | **通过**（仪器有效） |
| Spearman(层号, recovery) | **−0.997** | $\rho < -0.7$ ⇒ 级联 | **级联 —— 与预测一致** |
| 剖面 | 0.997 → 0.980 → 0.898 → … → 0.000 | —— | 单调递减 |
| $C_m$（名义值，不作判据） | 0.187 [0.135, 0.279]（24 层弥散基线 0.083） | 公式前提不成立 | **not applicable** |
| 承诺层 | **L3 / 24（深度 0.17）** | 级联下的替代读数 | 决策在很浅处即被定死 |
| **C-hazard 判定** | **不可估（公式前提不成立：剖面单调递减）** | | |

**六个候选按架构族完全分离，无一例外：**

| 架构族 | 候选 | 层数 | Spearman(层号, recovery) | 剖面 | 承诺层 |
| --- | --- | --- | --- | --- | --- |
| TransFuser 系 | DiffusionDrive | 8 | **+0.929** | 内部峰 @L6 | **不存在** |
| TransFuser 系 | LTF | 8 | **+0.929** | 内部峰 @L6 | **不存在** |
| TransFuser 系 | DiffusionDriveV2 | 8 | **+0.810** | 内部峰 @L4 | **不存在** |
| 纯 transformer | **SimLingo（留出检验）** | 24 | **−0.997** | 级联 | L3（深度 0.17） |
| 纯 transformer | Alpamayo-R1 | 36 | **−0.873** | 级联 | L16（深度 0.47） |
| 纯 transformer | AutoVLA | 36 | **−0.859** | 级联 | L20（深度 0.58） |

**3/3 递增 vs 3/3 级联，符号之间没有任何重叠。**
这条边界因此从"一条可证伪的预测"升级为**已在六个候选上验证、且通过一次留出检验的架构级规律**，
正文 §4.4.4 的措辞已相应升级。

## 3. 预测中被证伪的那半句（不淡化）

§CE/A39 原文说纯 transformer 栈"**早段必然饱和于 1.0**"。**这半句说过头了。**

SimLingo 的剖面是**渐进递减**（0.997 → 0.980 → 0.898 → 0.901 → 0.875 → …），
不是 AutoVLA 那种**平台式阶跃**（L0–L20 恒为 1.0）。三者的承诺层深度是 **0.17 / 0.47 / 0.58**，
差异超过三倍。

准确表述应为：

> **纯 transformer 栈的 C-hazard 剖面必然单调递减（级联）；但"饱和平台有多宽"不是架构决定的。**

这一更正**加强而非削弱**了承诺层的价值：它是一个真正有区分度的读数，
而不是一个恒等于"很深"的常数。**定性预测成立，定量外推不成立**，两者都已写进正文。

## 4. 预测之外的一项收获：两条配对源给出同一条剖面

SimLingo 的 C-hazard 剖面与它自己的 **C-domain** 剖面，配对源毫不相干
（G1 clean↔ghost 危险配对 vs CARLA↔世界模型渲染域配对），结果：

| 比较 | 值 |
| --- | --- |
| Pearson $r$ | **0.968**（$p$ = 9.4 × 10⁻¹⁵） |
| Spearman $r$ | **0.996**（$p$ = 3.5 × 10⁻²⁴） |
| 逐层 \|差\| 均值 / 最大 | 0.084 / 0.255 |
| 承诺层 | **两者同为 L3** |

**同一个模型、两种毫不相干的配对来源、同一条剖面。**
这直接支持 §CE/A39 的核心论断：**剖面形状是模型的路由属性，不是配对来源的属性。**
它也是本文里少见的一次"同一读数在两条独立证据链上收敛"，
且与 DD/LTF 的那次收敛不同——那次是两个共用编码器的模型，这次是同一个模型的两条配对源。

## 5. 顺带更正的一处数字串号

§CE/A39 的对照表把 DiffusionDrive 的 **C-hazard** Spearman 写成 +0.881，
那实际上是它的 **C-domain** Spearman；C-hazard 是 **+0.929**（见 `c_axis_hazard_dd.json`）。
已在 `amendments.md` 与 `paper_experiments_section_{zh,en}.md` §4.4.4 更正。
结论不受影响（两者都是正号、都是内部峰），但两条不同配对源的数字不能串用。

## 6. 产出物路径

| 路径 | 说明 |
| --- | --- |
| `results/c_axis_hazard_simlingo.json` | SimLingo 的 C-hazard 全部读数，含 `commitment_layer` |
| `results/c_hazard_prediction_check.md` | 本文件 |
| `scripts/c_axis_hazard_patch.py`（改） | `--model` 增加 `simlingo`（复用它已有的 `set_patch`，不新写 patching 实现） |
| `results/paper_experiments_section_{zh,en}.md`（改） | Table 1 的 SimLingo C-hazard 格；§4.2.3 / §4.3 / §4.4.4 / §4.5 措辞升级 |
| `results/amendments.md`（改） | §CE/A40 + 收尾说明 |

## 7. 本轮到此为止

按外部要求，**不再自主开新的补充方向**。
已知未完成项（Alpamayo / AutoVLA 的 F②、LTF 的 C-domain、五个候选 C-hazard 均为 12 事件的样本量限制等）
如实列在 `candidate_expansion_DONE_supplement.md` §5 与 `candidate_expansion_DONE.md` §4，
**等待外部复核后再决定下一步。**
