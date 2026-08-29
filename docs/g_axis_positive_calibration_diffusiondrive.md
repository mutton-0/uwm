# G 轴正向校准实验:DiffusionDrive 上的视觉→危险概念判别力验证

> 范围声明:本实验**只回答一个问题**——"G 轴的读出方法本身,在一个大概率真的有 grounding 的模型上,能不能正确读出'有'"。不重复定义方法论,复用 [direction_vector_discovery_validation_plan.md](direction_vector_discovery_validation_plan.md) 的 Stage B 提取流程与 [vla_base_model_selection_protocol.md](vla_base_model_selection_protocol.md) 的 G 轴公式;不重新设计负例,直接复用 SimLingo 上已验证过的 N1 负例体系(`n1_report.md`)。

---

## 1. 背景与动机(对应期刊论文 Motivation 段)

SimLingo 上的 G 轴读数(`n1_report.md`)已经给出一个带证伪控制的稳健负结果:主读数 D2a(AUC=0.568)与纯证伪控制 D2cV(AUC=0.574,类别相同、唯一差异是单帧模型物理上不可能看到的相对速度)**无法区分**——即读出的"信号"落在地板上。

这个负结果有两种互斥的解释,而当前证据不足以排除其中一种:

- **H-real**:SimLingo 这个模型本身确实缺乏视觉→危险概念的 grounding(方法论有效,标本真的有病);
- **H-artifact**:整套读出口径(池化方式、层选择、对比学习式方向提取)本身就读不出"有 grounding"这件事,不管换哪个模型都会落地板(方法论本身失效)。

区分这两种解释的唯一方式,是把**同一套读出口径**用在一个**大概率真的有 grounding** 的模型上——DiffusionDrive 是 nuScenes/NAVSIM 原生训练,任务定义就包含碰撞规避,若它在同一套 G 轴口径下仍然读不出信号,H-artifact 成立,整个 G 轴方法需要推翻重来;若能读出清晰信号,H-real 成立,SimLingo 的负结果被坐实为标本属性,而不是仪器缺陷。

---

## 2. 假设(对应 Methods 段 Hypothesis)

**H1(方法有效性)**:在完全相同的读出口径(相同事件语料、相同负例设计 D2a/D2cV、相同 CV-AUC 峰层选择规则)下,DiffusionDrive 的 CV-AUC(A vs D2a) 显著高于置换零分布与 D2cV 证伪地板;SimLingo 的对应读数落在地板上——两者在同一把尺子下呈现**质的差异**,而非都失败或都成功。

**决判规则(不留"失败"格)**:

| DiffusionDrive 读数 | SimLingo 读数(已知,来自 n1_report.md) | 判定 |
| --- | --- | --- |
| 显著高于 D2cV 地板 | 落在 D2cV 地板 | **H1 成立**:G 轴方法有效,SimLingo 的 FAIL 是标本属性 |
| 也落在 D2cV 地板 | 落在 D2cV 地板 | **H-artifact 成立**:G 轴读出口径需要重新设计(池化/层选择/对比范式),这是比继续跑标本更急迫的修复 |
| 显著高于 D2cV 地板但弱于预期 | 落在 D2cV 地板 | **部分成立**:方法能读出方向但灵敏度有限,需报告效应量而非仅报告显著性 |

---

## 3. 方法(对应 Methods 段 Setup)

**比较公平性纪律(选型协议 §4 规则 5 的直接应用)**:为使这次校准成立,DiffusionDrive **必须**在与 SimLingo 完全相同的刺激集上跑——即项目自有的 nuScenes 鬼探头挖掘语料(G1,现 1628 事件 cache)+ N1 的 D2a/D2b/D2c/D2cV 负例体系,**不使用 DiffusionDrive 自己的 NAVSIM/navhard 评测集**(那是另一件事,给 headline 实验的公开分数栏用,不是本实验的输入)。

| 项 | SimLingo(已完成,基线) | DiffusionDrive(本实验,待执行) |
| --- | --- | --- |
| 训练/原生域 | CARLA | NAVSIM(据 BridgeSim 表) |
| 评测刺激集 | nuScenes G1 语料 + N1 负例 | **同一份** nuScenes G1 语料 + N1 负例(需构建输入适配器) |
| 正例 | A(VRU 突现),n=283(匹配后) | 同一批事件,经适配器转换 |
| 负例(主读数) | D2a(几何平衡静态物) | 同一批,经适配器转换 |
| 负例(证伪控制) | D2cV(同类别、仅速度不同) | 同一批,经适配器转换 |
| 方向提取 | S_dir 有监督判别,scene 级 4 折 CV | 同一流程,换模型的 hidden states |
| 峰层选择 | CV-AUC argmax(按角色不按层号) | 同一规则,DiffusionDrive 自己的 argmax |
| 池化口径 | region/vision_mean/last_token/query 四种 | 至少跑 vision_mean(与 SimLingo 主读数口径一致,便于直接对比),其余口径按预算酌情补齐 |

**工程依赖**:需要为 DiffusionDrive 写一个输入适配器,把 nuScenes G1 语料的图像/元数据转换成它训练时的多相机+BEV 输入格式;`ghosthead_infer/` 现有脚本(patching/CKA)已经在跑 DiffusionDrive,说明模型加载与 forward 接口已经打通,适配器成本主要是"喂 G1 语料"而不是"从零接入模型"。

---

## 4. 结果表模板(对应 Results 段 Table 1,执行后填数)

**Table 1. G-axis readout under identical stimuli and negative design, across specimens with divergent native domains.**

| Model | Native domain | Negative class | n_neg | CV-AUC | 95% CI | p (vs permutation null) | 与 D2cV 地板可区分? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimLingo | CARLA(sim) | D2a(主读数) | 283 | 0.568 | [0.521, 0.615] | 0.0048 | **否**(D2cV=0.574,不可区分) |
| SimLingo | CARLA(sim) | D2cV(证伪控制) | 212 | 0.574 | [0.523, 0.624] | 0.0049 | — |
| DiffusionDrive | NAVSIM(real) | D2a(主读数) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| DiffusionDrive | NAVSIM(real) | D2cV(证伪控制) | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

**读法**:关注的不是单个 AUC 绝对值,而是**同一模型内部** D2a 与 D2cV 两行的差距——SimLingo 这两行几乎重合(方法在它身上读不出区分);若 DiffusionDrive 的 D2a 明显高于自己的 D2cV,即构成 H1 成立的直接证据。

**仪器侧/标本侧两行摘要(沿用 final_report.md 报告纪律,执行后填):**

> **仪器侧**:[同一套 D2a/D2cV 负例设计与峰层选择规则在 DiffusionDrive 上的表现,决定 G 轴读出口径本身是否成立]
> **标本侧**:[DiffusionDrive 是否表现出视觉→危险概念的判别力,以及与 SimLingo 的对照说明了什么]

---

## 5. 与四轴总体实验的位置(对应 Discussion 段一句话定位)

本实验是 G/F/I/C 四轴证明集里**唯一一个"方法学正向校准"性质的实验**——F(TTC 梯度拟合)、I(3DGS 域配对)、C(复用 DiffusionDrive 既有 patching)都是在扩充证据广度,而本实验是在补一个此前缺失的**方法学有效性下界**:没有它,SimLingo 的所有负结果都可以被质疑"是不是仪器不行",有了它(若 H1 成立),SimLingo 的负结果才能被写进论文当作**标本层面的真实发现**,而不是留一个"未获效度"的尾巴。

---

## 6. 产出物清单

```
results/
  g_positive_calibration_diffusiondrive.json   ← DiffusionDrive 版 Table 1 全部数字
  g_positive_calibration_report.md              ← 仪器侧/标本侧两行摘要 + H1 判定
  diffusiondrive_g1_adapter/                    ← 输入适配器代码
```
