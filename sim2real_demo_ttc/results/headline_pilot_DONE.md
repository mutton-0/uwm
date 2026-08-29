# ✅ 完成标志：三线头号证据 + pilot post-train（第三轮）

**一句话总览**：三条线全部落地——公开分数**无法定义排名**（6.87 与 88.1 不在同一把尺子上）；
四轴矩阵在同一刺激集同一口径下补齐后，**只有 C 轴给出实质跨模型区分**
（SimLingo 的域失效自视觉接口级联、DiffusionDrive 的集中在深层融合层 L6，两个完全不同的修复处方）；
pilot post-train 证明 F 轴诊断所指的环节**可以被真实修好**（耦合量改动 3.3 倍），
**但固定小预算下不带来行为收益**（A2−A1 = +0.001 [−0.004, +0.007]，归因对照否掉了表面上 +0.051 的改善）。

---

## ⚠️ 回来后请先看这三条

1. **工单文件 `docs/headline_three_line_pilot_workorder.md` 不在磁盘上**（全盘检索无此文件，
   `docs/` 最后修改于上一轮同步时刻）。本轮按你消息里的规格执行，规格逐条抄录在
   `amendments.md` §HL/A23，**请与原工单比对**；若原工单含本消息未覆盖的设计细节
   （指定预算 / 指定行为分定义 / 指定对照臂），相应决策是我自行做的，均登记在 §HL/A26。
2. **轮询自匹配 bug 已记入** `amendments.md` §HL/A22（pgrep -f 匹配到轮询命令行自身），
   含三条下次的正确写法。
3. **本轮有 5 条修正案把已经拿到的阳性结果改回阴性或不可估**：A24（C 轴公式前提）、
   A25（F 轴仪器无分辨力）、A19（刺激集不等）、A16、A14。第一次跑出来的数字**不要引用**。

---

## 一、最终交付物（论文用）

| 路径 | 说明 |
| --- | --- |
| `results/paper_experiments_section_zh.md` | **本轮最重要交付物**：论文 Experiments 章节完整正文（中文），按 Setup / Main Results / Ablation-like / Limitations 重组，不是报告拼接 |
| `results/paper_experiments_section_en.md` | 同上，英文 |

## 二、三个子任务的中英双语报告

| 子任务 | 中文 | 英文 |
| --- | --- | --- |
| ① 轴矩阵补齐（F 补 DiffusionDrive、C 补 SimLingo） | `results/axis_matrix_completion_report_zh.md` | `results/axis_matrix_completion_report_en.md` |
| ② pilot post-train（SimLingo，耦合辅助损失） | `results/pilot_posttrain_report_zh.md` | `results/pilot_posttrain_report_en.md` |
| ③ 三线头号证据表 | `results/headline_three_line_report_zh.md` | `results/headline_three_line_report_en.md` |

## 三、数值真源（JSON / 权重 / 表格片段）

| 路径 | 内容 |
| --- | --- |
| `results/h1_three_line_evidence.json` / `.md` | 三线证据表的全部数字与可比性标注 |
| `results/f_axis_action_counterfactual.json` | F 轴行动层反事实测试（两模型，架构中立主验证） |
| `results/f_axis_dd_steer.json` | DiffusionDrive 注入四件套（主跑，20 seed 零分布） |
| `results/f_axis_dd_steer_brakedd.json` | **站内上界标定**：注入 DiffusionDrive 自己的行为定义轴 |
| `results/f_axis_dd_steer_escalate.json` | 剂量升级 α ∈ {4,8,16,32} |
| `results/v_brake_dd.npz` / `.json` | DiffusionDrive 的行为定义轴（逐层 held-out AUC 峰值 0.656 @L5） |
| `results/c_axis_simlingo.json` | SimLingo 逐层 activation patching（12 退化场景，patch-ALL = +1.004） |
| `results/c_axis_shape_diagnostics.json` | **恢复剖面形状诊断**（决定 top-2 占比公式是否适用） |
| `results/p2_pilot_posttrain.json` | pilot 三臂全部读数与对比 |
| `results/p2_head_A1_task_only.pt` / `p2_head_A2_task_plus_coupling.pt` | 训练后的 `speed_wps_head` 权重 |
| `variants/n1_d2/pilot_query_states.npz` | pilot 的特征缓存（2296 样本；已 gitignore，可重建） |

## 四、新增脚本

| 路径 | 作用 |
| --- | --- |
| `scripts/f_axis_action_counterfactual.py` | 行动层反事实测试（零 GPU，两模型共用） |
| `scripts/f_axis_dd_brake_axis.py` | 构造 DiffusionDrive 的站内上界方向 |
| `scripts/f_axis_dd_steer.py` | DiffusionDrive 注入四件套（支持 `--vec-npz` / `--alphas`） |
| `scripts/c_axis_simlingo_patch.py` | SimLingo 逐层 activation patching |
| `scripts/c_axis_shape.py` | 恢复剖面形状诊断（两模型统一，写回各自 JSON） |
| `scripts/p1_cache_query_states.py` | pilot 特征缓存（一遍前向） |
| `scripts/p2_pilot_posttrain.py` | pilot 三臂训练与评估 |
| `scripts/h1_three_line_table.py` | 三线证据表组装 |
| `results/diffusiondrive_g1_adapter/dd_adapter.py` | **已扩展**：新增 `set_steering` / `lateral_offset` / `comfort` |
| `scripts/simlingo_runner.py` | **已扩展**：新增 `set_patch` / `_apply_patch`（C 轴 patching） |

## 五、账本

`results/amendments.md` —— 本轮新增 **§HL/A22–A26** 五条（累计 A1–A26）：

| # | 内容 | 后果 |
| --- | --- | --- |
| A22 | 轮询脚本 `pgrep -f` 自匹配 bug | 无计算损失，浪费数小时等待；含下次正确写法 |
| A23 | **工单文件缺失**，按用户消息规格执行 | 规格逐条抄录，供回来核对 |
| A24 | C 轴 top-2 占比公式的**适用性判据** | SimLingo 的 C 由「PASS」改判 **不可估（公式前提不成立）** |
| A25 | DiffusionDrive 的 F 轴 steering **仪器无分辨力** | 零结果归因由标本侧改为**仪器侧**；F② 标为跨模型不可比 |
| A26 | pilot 的全部设计决策 + 耦合方向拟合口径修正 | 首版耦合方向 held-out ρ 符号是反的，已修正 |

---

## 六、三条线的核心数字（速查）

**线①公开榜单**：SimLingo CARLA LB2.0 Driving Score **6.87**；DiffusionDrive NAVSIM navtest PDMS **88.1**。
→ 基准/域/协议/指标定义全不同，**两数之间不存在有意义的大小关系**（这是论据，不是限制）。

**线②四轴矩阵**（同一刺激集、同一口径）：

| Axis | SimLingo | DiffusionDrive | 可比 | Verdict |
| --- | --- | --- | --- | --- |
| G（主读数−D2cV 地板） | +0.035 [−0.026, +0.097] | +0.009 [−0.047, +0.065] | 是 | 均不可估 |
| F①（行动层反事实 b-AUC） | 0.534 [0.469, 0.592] | 0.553 [0.486, 0.623] | 是 | 均不可估 |
| F①（b(A)，m/s） | **+0.307 [+0.112, +0.500]** | +0.010 [−0.030, +0.045] | 模型内 | 反应但不特异 vs 不反应 |
| F②（注入斜率） | −0.0405 | −0.00000（上界轴亦 +0.00002） | **否** | DD 侧仪器无分辨力 |
| I（表征端 $I_m$） | **0.606** | 0.203 | 是 | SimLingo > DD |
| I（行为端敏感度） | 0.346 | **0.115** | 是 | **相反** |
| C（剖面 Spearman） | **−0.997** 级联 | **+0.881** 内部峰 | 是 | 进入位置不同 |
| C（责任层 / 熵） | **L0** / 0.279 | **L6** / 0.685 | 是 | 视觉接口 vs 深层融合 |

**线③pilot post-train**（SimLingo，只训 229k 参数的 `speed_wps_head`，held-out scene）：

| Arm | b-AUC | 95% CI | Δ@1σ (m/s) | cos(g, v̂) |
| --- | --- | --- | --- | --- |
| A0 基线 | 0.544 | [0.461, 0.621] | +0.091 | +0.046 |
| A1 纯任务（归因对照） | 0.592 | [0.496, 0.680] | −0.076 | −0.016 |
| A2 任务+耦合 | 0.593 | [0.499, 0.679] | **−0.301** | **−0.068** |

→ **A2−A0 = +0.051 看着像成功，但 A1−A0 = +0.050、A2−A1 = +0.001 [−0.004, +0.007]**。
耦合量确实被改变（Δcos = −0.113，3.3 倍），行为分无增量。**判定：部分成立。**

---

## 七、本轮明确未做 / 未能证明的

1. **三线对比的第三条线未真正闭合**：pilot 是固定小预算、单模型、单方向的验证，
   不是"固定 post-train 预算下 held-out 域安全分增量"的完整测量。
   故本工作证明的是"四轴可测、互不冗余、所定位环节可干预"，**不是**"四轴可预测 post-train 收益"。
2. **未对 DiffusionDrive 做 pilot post-train**：其耦合损失依赖可测的注入响应 Δ@1σ，
   而该量在 DD 上恒为 0（§HL/A25），耦合项无梯度信号。架构决定，非预算决定。
3. **公开榜单分数原样引用，未在本轮独立复现**（未重跑 CARLA LB2.0 / NAVSIM navtest）。
   本文对这两个数字的唯一用法是论证其不可比，该论证不依赖数值准确性。
4. **G 轴仍只能报"读得出/读不出"**，瓶颈是 D2cV 负例样本量（212 条）。
5. 域配对是**世界模型真实感重绘**，不是 3DGS 重建。
