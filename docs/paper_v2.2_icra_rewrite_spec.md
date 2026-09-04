# 待办：新版 ICRA 论文整理规格（**等前面所有实验跑完、数据齐了再做**）

> 登记于 2026-09-04。**目标文件：`paper_v2.2_draft/`（ICRA 模板那份），不是 CVPR 那份。**
> 前置条件：arc_full 方法定稿 ✅、ghost 两语料收敛 ✅、
> 三类新场景挖矿 ⬜、benchmark vs deployment 排名指标 ⬜。
> **不得拿不完整/占位数据先写。**

## 硬性要求

1. **全部结果基于最新采样方法**：brake-first 挖矿 + 归因 + 分层 QA + GT 轴（`arc_full` 全轨迹口径）。
   **一律不进正文**的旧内容：旧的 288/282 事件语料（原挖矿，走廊判据已证伪）、
   旧的 `commanded_speed` 0.5 s 短窗口读数、旧的三态 PASS/FAIL/不可估判据体系、
   旧的 legacy G/F 轴定义、任何与新方法冲突的表述。
   能删直接删；删不掉的必要说明（如方法演进）**放附录**，不犹豫，先放进去。
   **正文绝对不能出现新旧冲突的内容。**
2. **采样与遮挡对齐**：四个场景的遮挡组定义、GT 轴窗口口径必须一致。
   场景特定的调整（如前车场景的遮挡目标类别不同）须在方法节**写清差异与理由**。

## 必含内容

| # | 内容 | 备注 |
| --- | --- | --- |
| 1 | 四轴举例说明图 | 更新 `gfic_vivid_en.png`；**F 轴例子换成新方法/新数据的真实案例**（如 `scene-0862`） |
| 2 | Framework/pipeline 图 | 检查是否因 GT 轴引入而需更新示意 |
| 3 | 四场景 × 四轴完整实验数据表 | 鬼探头 / 前车-静态路障 / 无保护左转 / 十字路口 |
| 4 | benchmark(nuScenes) vs deployment(NAVSIM) 对比图 | 箱线图或 t-SNE，四轴各一张或合并 |
| 5 | 典型异常案例配图 + 分析 | 不必 4×4 全做，但**每个场景至少一个** |
| 6 | 部署质量评分横向对比表 | PDM / TTC / 舒适度，六候选，benchmark 排名 vs 部署排名 |
| 7 | 所有公式完整写出 | GT 轴、`arc_full` 读数、归因 $a_{req}$、打分方式；正文核心公式 + 附录专章展开 |

## 执行方式

* 写完用已装好的 tectonic 重新编译验证：**无编译错误、无遗留旧数据/旧结论的文字表述**。
* 产出前给一份**变更摘要**（改了哪些 section、依据哪些报告的哪些数字），留痕便于复核。
* 不需等确认再编译，但摘要必须留痕。

## 数据来源（截至登记时已就绪的）

```
results/f3_gt_axis_method_v2.md               方法定义（arc_full + GT 轴）
results/ghost_scenario_final_zh.md            ghost 场景两语料结论
results/f3_gt_axis_full{,_navsim}_*.json      逐候选 GT 轴数据
results/brake_first_pool{,_navsim}_final.json 语料池 + QA 留痕
results/figures/f3_reversed/                  反向案例图
```

## 已知必须在论文中如实写明的限制

1. **六候选未跑齐**：Alpamayo / AutoVLA 绑死 nuScenes devkit，NAVSIM 无法运行；
   nuScenes 上需把多帧遮挡改造成遮挡组语义。
2. **样本量**：nuScenes 13 scene、NAVSIM 8 scene（ghost 场景），功效有限，
   多数格子落"不显著"。**不得把 null 读成模型性质。**
3. **未标注交通管制**（信号灯 / STOP）⇒ $b^{GT}$ 高估，无法用几何量剥离。
4. **走廊外未遮同类目标** ⇒ 响应比方向性低估。
5. **DDv2 点云删点为 0** 的事件 ⇒ 响应弱可能是"没点可删"而非"不用该实体"。
