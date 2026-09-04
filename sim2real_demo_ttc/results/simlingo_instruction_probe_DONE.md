# DONE：SimLingo 指令服从/拒绝探针

完成时间：2026-09-04

## 结论一句话

**探针工作正常，核心假设未获支持。** SimLingo 会显式拒绝加速类指令（2129 次），
拒绝率随图像内容变化（空白灰图上降到 0%），但**与被挖掘的那个行人在不在画面无关**：
在同一张帧上把该行人抹掉，危险类拒绝率变化 +1.4pp [−1.7, +4.5] / +1.8pp [−0.8, +4.9]（CI 跨 0），
而只在别处涂同面积灰斑的对照臂效应**相当甚至更大**（+2.8pp [+0.3, +5.5]）。
最尖锐的一条：只数 `"because of the pedestrian"` 这条理由，把行人抹掉后其出现率
**不降反升**（7.1% → 8.5%），在"画面里再无其它行人"的 63 事件子集上同样如此。

**不建议按当前设计扩大规模**——n=282/147 场景已把 CI 压到 ±3–5pp 而效应为 0。

## 工单各步对照

| 步骤 | 状态 | 说明 |
|---|---|---|
| 1. prompt 语序对齐 `dataset_dreamer.py:129-131` | 完成 | 指令置于 target waypoint 之后、去掉 `"Predict the waypoints."`；**子类化 runner 实现，未改 `simlingo_runner.py`** |
| 2. 构造探针（clean/ghost × 加速/减速指令） | 完成并**扩展** | clean 帧经查**并非"行人不在"**（81.9% 仍带 bbox），故补了同帧 `occ`/`ctrl` 两臂做真正的在场/不在场操作 |
| 3. 核心问题：ghost 上加速指令拒绝率是否显著高于 clean | **答案为否** | 见报告 §3.2/§3.3 |
| 4. 小样本先验证探针本身工作 | 完成 | 25 事件试点确认拒绝逻辑触发；再扩到 282 事件 |
| 5. 如实报告，不往好处解读 | 遵守 | 唯一方向为正的格子（+3.2pp，CI 下界压在 0.000，n=63）已明确标注为**不足以支撑结论** |

## 报告与数据

- 报告：`results/simlingo_instruction_probe_report_zh.md`
- 主结果：`results/simlingo_instruction_probe_occ.json`（n=282，clean/ghost/occ/ctrl）
- 选择性地板：`results/simlingo_instruction_probe_grey.json`（n=120）
- 分层依据：`results/simlingo_instruction_probe_scene_census.json`
- 汇总：`results/simlingo_instruction_probe_analysis.json`、`results/simlingo_instruction_probe_pedestrian_reason.json`
- 试点：`results/simlingo_instruction_probe_pilot.json`
- 目视核查图：`results/figures/instruction_probe_occ_check/`

## 撞车规避

本轮全部使用新建文件名（`simlingo_instruction_probe*`、`simlingo_probe_scene_census*`）。
**未改动**任何既有脚本；`f3_occlusion_necessity.py` 与 `simlingo_runner.py` 均为只读引用/子类化。
**未写入 `amendments.md`**（该文件当前有其它会话的未提交改动）——
若需要登记账本条目，请在合并时由账本持有方补一条，内容可直接引用本报告 §2.1（`clean` 窗口不等于"实体不在场"）
与 §2.2（`occ` 仅移除单一实体，77.4% 的帧仍有其它行人）这两条方法学更正。
