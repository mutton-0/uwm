# 工单:G-VS(分割一致性)+ F-3(遮挡必要性检验)——简化版 G/F,六候选统一

> **自主执行授权**,不需要逐步请示。这是对 G/F 的一次重大简化:放弃"量化危险意识"这个不可验证的构念,换成有客观 GT 的感知正确性检验(G)和不依赖语义标注的因果必要性检验(F)。现有的复杂 G(D2a/D2cV/速度回归)与复杂 F①/F②(危险特异性/注入)**全部保留、不删除**,作为"为什么语义轴这么难量化"的二级证据体系。本工单只新增 G-VS 与 F-3,是本轮论文的 v1 主定义。

## 0. 核心动机(写进报告,不要只当成功能需求)

多轮实验(G1/前车急刹/NAVSIM 三个语料)反复证明:围绕"危险"这个语义构念构造的 G/F 读数,天然依赖挖掘管线对"什么算危险"的人工判据,导致跨场景/跨数据源都不稳定。G-VS 和 F-3 的设计目标是**换成有客观 ground truth、不需要人工语义判据的检验**:
- **G-VS**:模型内部表征能不能正确对应真实场景结构(分割 GT),这是可验证的感知正确性,不涉及"危险"这个主观判断;
- **F-3**:拿掉真实存在的关键实体,动作会不会跟着消失(必要性检验),这是直接的输入层因果操作,不需要构造匹配负例。

文献锚点:Embodied Interpretability(arXiv 2605.00321,§ blind action / spurious correlation / post-hoc rationalization 三分法,本工单只做其中的 blind action 对应的必要性检验);Causal Imitative Model(Samsami et al. 2021, arXiv:2112.03908,causal inversion 框架,inertia/collision 两种失效签名);Hewitt & Liang 2019(探针 selectivity/control task 纪律,G-VS 训探针时必须报)。

## 1. G-VS:分割一致性(六候选统一,不分单帧/多帧、有无语言)

**先做可行性检查(半天时间盒)**:
1. TransFuser 系(DiffusionDrive/LTF/DiffusionDriveV2)的架构里,原始训练目标是否已经包含 BEV 语义分割这个辅助头?如果有,直接读现成输出,零训练成本。
2. 本机能否跑 SAM(Segment Anything)对同一批图像生成分割伪 GT?检查 checkpoint 是否可下载、推理成本。

**若无现成分割头,按以下方案训探针**:
- 用 SAM 对 G1 语料的图像生成伪 GT 分割(检查伪 GT 质量,抽检几十张人工过一遍);
- 从每个候选的内部特征(已有缓存的 vision_mean/region_mean 等池化表征,或需要的话取未池化的 token 级特征)训一个**轻量线性/浅层探针**,预测分割伪 GT;
- **必须报 selectivity 对照(Hewitt & Liang 2019)**:探针本身的容量不能替代模型完成任务——用随机初始化模型的同架构特征训同一个探针作对照,若对照探针也能达到相近 mIoU,说明是探针自己在"脑补",不是模型表征里真有这个信息;
- VLA 候选(SimLingo/Alpamayo-R1/AutoVLA)取其视觉编码器部分的特征做同样的探针,不涉及语言输出。

**主读数**:held-out mIoU(探针预测 vs SAM 伪 GT),scene 级 bootstrap 出 CI,对照随机初始化探针的 mIoU 地板。

## 2. F-3:遮挡必要性检验(六候选统一)

- 复用 G1 挖掘阶段已经算好的事件几何(bounding box),对每个 A 类(危险)事件的 ghost 帧,**把危险实体所在区域打码遮住**,生成一个"遮挡版"输入;
- 分别喂给模型:原始 ghost 帧(有危险实体可见)、遮挡版 ghost 帧(危险实体被遮住)、clean 帧(危险实体本来就不在场)三种输入;
- **主读数**:$v_{plan}(\text{遮挡版}) - v_{plan}(\text{clean})$ 是否接近零、且显著小于 $v_{plan}(\text{原始 ghost}) - v_{plan}(\text{clean})$——即遮住关键实体后,动作应该"退回"到没有危险时的基线,这是必要性检验的核心逻辑;
- 三态判定:若遮挡后动作确实退回基线,说明该模型的动作真的是被这个实体的可见性因果驱动的(通过必要性检验);若遮挡后动作没有显著变化(还是像看到了危险一样反应),说明该动作的驱动因素**不是**这个实体本身,是别的东西(呼应"盲目泛化"这个失效签名,以及 Causal Imitative Model 讲的 inertia/collision 两种失效)。

## 3. 范围(这一轮先只做 G1,不追求三语料齐全)

主战场是 **G1(discovery 语料)**,六个候选全部覆盖。前车急刹/NAVSIM 两个语料上的复现检验作为**有余力再做的加分项**,不是本轮硬性要求——按"先占坑"的原则,先把 G1 上六候选统一出数,论文用这个做主结果,后续论文修订/下一篇再补跨场景复现。

## 4. 统计纪律(照旧)

scene 级 bootstrap;预注册单一主读数(G-VS 用 mIoU,F-3 用遮挡后动作差值);三态判定(PASS/FAIL/不可估);每个数字仪器侧/标本侧两行摘要。

## 5. 产出物

```
results/
  g_vs_feasibility_report_{zh,en}.md          ← 分割头/SAM 可行性评估
  g_vs_axis_{候选}_{zh,en}.md                  ← 六候选的 G-VS 读数(含探针 selectivity 对照)
  f3_occlusion_axis_{候选}_{zh,en}.md          ← 六候选的 F-3 读数
  g_vs_f3_unified_matrix_report_{zh,en}.md    ← 核心交付:六候选统一的 G-VS/F-3 矩阵,不再分组(a)/(b)
  g_vs_f3_DONE.md                              ← 完成标志
```

完成后更新 `results/paper_experiments_section_{zh,en}.md`:
- Table 1 新增 G-VS / F-3 两列(六候选统一,不分组),作为 v1 主读数;
- 现有的复杂 G/F①/F② 结果保留,改写成"为什么语义轴难量化"的二级分析章节(不删除任何已有数字);
- Future Work 新增 G-VL、F-1(CoC 事后合理化)、F-2(虚假相关/注意力分析)三项,标注文献锚点(Embodied Interpretability、Causal Imitative Model)。

自主执行到完成,不需要逐步请示;遇到岔路按一贯纪律记入自我更正记录。
