# 工单:G/F/I/C 四轴证明实验(打包执行版)

> **自主执行授权**:本工单授权一次性自主执行到底,不需要逐步请示。遇到本工单未覆盖的岔路,按项目一贯纪律自行决策并在报告的"自我更正记录"中如实记录,不要停下来等指令。
> **范围声明**:本工单只打包编排"证明 G/F/I/C 四轴"这四个子实验,不重新定义方法论。各子实验的方法细节分别见:
> - G:[g_axis_positive_calibration_diffusiondrive.md](g_axis_positive_calibration_diffusiondrive.md)(已完整写好,直接按其执行)
> - F/I/C 的方法锚点:[vla_base_model_selection_protocol.md](vla_base_model_selection_protocol.md)、[direction_vector_discovery_validation_plan.md](direction_vector_discovery_validation_plan.md)

---

## 0. 时刻不要忘记的核心贡献(每份产出物都要回扣这一点)

我们不是在证明"SimLingo 好不好"或"DiffusionDrive 好不好"。我们在证明:

> **公开榜单排名(如 nuScenes/NAVSIM 开环分数)与真实部署效果排名可能脱节,而 G/F/I/C 四轴刻画的是这个脱节具体发生在因果链的哪一环**(感知未奠基 G?奠基了但未驱动动作 F?驱动了但跨域不稳 I?失效弥散难修 C?)。

四个子实验各自证明的是这条因果链上的一环**存在、可测、可致因果区分**,不是孤立地"测出一个分数"。每份报告结尾必须有一句话说明:本实验的读数如何支撑或修正这条因果链的画像。

---

## 1. 任务总览

| 任务 | 轴 | 一句话目标 | 主要标本 | 状态 |
|---|---|---|---|---|
| **T-G** | G(Grounding) | 检验 G 轴读出口径本身是否有效(正向校准) | DiffusionDrive(对照 SimLingo 已有负结果) | 待执行,方法已定稿 |
| **T-F** | F(Faithfulness) | 用连续 TTC 梯度拟合一条观测法刹车方向,与已验证的注入法 $v_{brake}$ 做余弦一致性检验 | SimLingo | 待执行 |
| **T-I** | I(Invariance) | 3DGS 鬼探头场景集上,失效是否精确对应"该模型自己的训练域缺口" | SimLingo + DiffusionDrive | 待执行 |
| **T-C** | C(Concentration) | 失效因果定位是否集中在少数层;t-SNE 作为辅助可视化 | DiffusionDrive(复用既有 patching)+ SimLingo(如有余力) | 复用为主,低成本 |

---

## 2. T-G:G 轴正向校准

直接按 [g_axis_positive_calibration_diffusiondrive.md](g_axis_positive_calibration_diffusiondrive.md) 全文执行,不在本工单重复。核心提醒:**必须用与 SimLingo 完全相同的刺激集(G1 nuScenes 语料 + N1 的 D2a/D2cV 负例)**,不得用 DiffusionDrive 自己的 NAVSIM 评测集替代——否则失去可比性,整个校准的意义就没了。

---

## 3. T-F:TTC 梯度拟合的观测法刹车方向

**假设**:若 F 轴(概念→行为通路)真实存在,那么从"不同 TTC 下真实 a_brake 严重程度"这个连续观测量中,应该能在隐空间里回归出一条方向,且这条方向应与已经过 W1 因果校准通过的注入方向 $v_{brake}$(§ demo final_report.md)高度共线(余弦相似度显著高于随机方向零分布)。

**方法**:
1. 沿用 G1 事件语料中所有可算 TTC 与 a_brake 的事件(不限于 A/D2a 分类,尽量扩大样本以获得连续梯度而非离散台阶);
2. 每个事件取模型内部同一层(与 $v_{brake}$ 校准时用的同一层,L22)的隐激活,回归目标为该事件的真实 a_brake 幅度(可选:也报告以 TTC 本身为回归目标的版本,作为敏感性分析,但预注册主读数用 a_brake);
3. 回归系数向量(或岭回归/PLS 方向,按数据量选择,记录选择理由)即为观测法方向 $v_{brake}^{obs}$;
4. **核心读数**:$\cos(v_{brake}^{obs}, v_{brake})$,与随机方向零分布(≥20 个随机方向的余弦分布)比较,报告 p 值与 95% CI;
5. scene 级 bootstrap,预注册这一条为 F 轴唯一主读数,其余(如不同层、不同回归方法)一律标记为敏感性分析。

**三态判定**:显著高于零分布 → PASS(F 轴用两种独立方法互相印证);不显著但方向一致 → 不可估(功效不足,不是方向不存在);方向相反或零分布内 → FAIL(如实报告,不得回避)。

---

## 4. T-I:3DGS 域配对失效对应测试

**假设**:若 I 轴(跨域稳定性)真实刻画的是"训练域缺口",那么每个模型在 3DGS 鬼探头场景集上的失效模式,应该系统性地对应**该模型自己的原生训练域与 3DGS/真实渲染风格之间的差异大小**——而不是所有模型对 3DGS 都表现出同等程度的、与训练域无关的通用失效。

**方法**:
1. 复用现有 3DGS 反事实场景集(P3 资产,若尚未产出,以最小可用子集先跑通管线,不新建 CARLA/Bench2Drive 依赖);
2. 分别在 SimLingo(CARLA 原生)与 DiffusionDrive(NAVSIM/真实数据原生)上跑同一批 3DGS 场景;
3. 用选型协议已定义的 $D_L$(跨域激活散度)与 $\cos(v_{domain}, v_{hazard})$(干涉角)公式计算 I 轴读数;
4. **核心读数**:两个模型的 $I_m$ 排序,与"3DGS 渲染风格相对各自训练域的直觉距离"(CARLA 更接近合成渲染 vs 真实数据训练模型可能对真实感渲染更适应,或反之——不预设方向,让数据说话)是否吻合;
5. 若只有一个模型可行(比如 DiffusionDrive 适配器来不及做),明确标注为单模型探索性读数,不做跨模型排序结论。

**注意**:这条不需要 CARLA 参考帧,不需要 Bench2Drive,只需要静态域配对帧——按 §3.5 已定稿的设计,不要重新引入已经排除的依赖。

---

## 5. T-C:失效集中度(复用为主)

**方法**:
1. 直接从 `scripts/ghosthead_infer/run_patching_ghosthead.py` 与 `run_cka_recovery_ghosthead.py` 已有产出中提取 DiffusionDrive 的逐层因果恢复数据,按选型协议公式计算 $C_m$ = top-2 层恢复占比 / 全层恢复之和——**这部分基本是整理现有数据,不需要重新跑因果实验**;
2. 补充:对同一批数据做一次 t-SNE(或 UMAP)可视化,按"失效 vs 正常"着色,检查视觉上是否聚类——这是辅助性、探索性的定性证据,报告里必须明确标注为**相关性证据,不能替代 patching 的因果结论**;
3. 若时间允许,对 SimLingo 也跑一版(哪怕数据量小),作为跨模型对照;不强制。

---

## 6. 输出规范(硬性要求:中英双语 + 期刊式描述与表格)

**每个子实验(T-G/T-F/T-I/T-C)各产出两个文件**:`results/<task>_report_zh.md` 与 `results/<task>_report_en.md`,内容对应(不是互译摘要,是同等信息量的两份完整报告),结构统一如下(仿照已有 `g_axis_positive_calibration_diffusiondrive.md` 的期刊式结构):

```
# <Title>

## Motivation / 动机
(1段:为什么做这个实验,对应因果链哪一环)

## Method / 方法
(数据、模型、公式、预注册主读数声明)

## Results / 结果
(Table 1,数字为主,附 95% CI 与 p;仪器侧/标本侧两行摘要)

## Discussion / 讨论
(对四轴因果链画像的更新;是否支撑或修正核心贡献主张;三态判定结论)
```

**Table 1 的绘制规范**:必须是可直接粘进论文的三线表结构(Markdown table,表头单位标注清楚,数值统一保留 3 位有效数字,CI 用方括号,p 值用科学计数法当 <0.001),表标题遵循期刊惯例(`Table 1. <一句话描述读数内容与比较对象>.`),不要用截图或非表格形式呈现数字。

**最后必须额外产出一份合成文档**:`results/four_axis_evidence_summary_zh.md` 与 `_en.md`——把 T-G/T-F/T-I/T-C 四份结果整合进**一张总表**(模型 × 四轴读数 × 判定),并在讨论段落明确回答 §0 的问题:这次的读数如何支撑"公开榜单排名 ≠ 真实部署排名,而四轴定位了脱节发生在哪一环"这个核心贡献。这份合成文档是给论文 Experiments 章节直接用的,写作时按 CVPR/期刊正文语域(不要口语化),但仍保留仪器侧/标本侧两行摘要的项目内部纪律作为附注。

---

## 7. 报告纪律(沿用项目一贯要求,不重复解释理由)

- 每个数字读数标注:仪器侧(效度依据)/ 标本侧(对模型说明什么)两行;
- 每个子实验预注册单一主读数,其余标记敏感性分析,维护分析账本防 p-hacking;
- scene/event 级 bootstrap,不做帧级重采样;
- 三态判定(PASS / FAIL / 不可估),不强行二元化;
- 任一环节偏离本工单设计,必须在报告"自我更正记录"章节写明原因,不得静默处理。

---

## 8. 产出物清单

```
results/
  g_positive_calibration_report_zh.md / _en.md
  f_axis_ttc_gradient_report_zh.md / _en.md
  i_axis_domain_pairing_report_zh.md / _en.md
  c_axis_concentration_report_zh.md / _en.md
  four_axis_evidence_summary_zh.md / _en.md   ← 最终交付,论文 Experiments 章节直接可用
  diffusiondrive_g1_adapter/                   ← T-G 所需适配器代码(T-I 复用)
```
