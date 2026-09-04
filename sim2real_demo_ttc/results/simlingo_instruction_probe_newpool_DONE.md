# DONE：SimLingo 指令探针 · brake-first 新语料版

完成时间：2026-09-04
语料：`results/brake_first_pool_final.json`（nuScenes 主分析集，13 事件 / 13 scene）
**只用这 13 个事件；未读取、未合并、未引用其它语料的任何数据或结论。**

## 一句话结论

**探针在新语料上工作正常（38 次危险类拒绝、0 次基线误拒、灰图地板 0.0%），
但核心问题在 n=13 下无法回答——是功效不足，不是"测出无效应"。**
整组抹掉走廊内 VRU 后，危险类拒绝率变化 +7.7 / 0.0 / +15.4 / +7.7 pp，CI 全部跨 0，
配对不一致对最多 3/13，精确 p ≥ 0.5；且**必需对照臂给出相同或相近的变化**
（`acc_target_speed` 主臂与对照臂同为 +7.7pp）。
`acc_target_speed` 的区间 **[−15.4pp, +30.8pp]** 同时容得下"无效应"与"强正效应"。

## 工单各点对照

| 要求 | 状态 |
|---|---|
| 1. 换数据源为 brake_first_pool_final.json 的 13 事件 | 完成，13/13 全部跑通，0 跳过 |
| 2. 四指令 / 五臂 / 理由分类 / 场景清点 / "因为那个行人"读数 照旧 | 完成（clean 臂需构造，见下） |
| 3. 如实报告功效不足，判定纪律不放宽 | 完成：四条 L1 全记"不可估"；另用精确二项补出 0/13 的 20.6% 上界，明确指出 bootstrap 的 `[0,0]` 是退化产物 |
| 4. 新报告独立、不改旧报告、互不引用数字 | 完成：`simlingo_instruction_probe_newpool_report_zh.md`，旧报告未改动 |
| 5. 遮挡逻辑协调 | **已协调，采用新语料的整组遮挡**，详见下 |

## 第 5 点：遮挡逻辑协调结果

**协调得上。采用 brake-first 的 `f3_mask_group` 整组遮挡，不用探针原先的单目标遮挡。**

- 13 个事件中 **10 个 `n_mask_group == 1`**，两套逻辑等价；另 3 个为 3/2/2 框，新逻辑遮得更干净。
- 投影 `g1_mine_events.frame_bbox()` + 涂色 `f3_occlusion_necessity.occlude()`，
  与该池出图脚本 `brake_first_export.py` 逐像素同一套，未改动、未近似。
- `ctrl` 对照臂相应改为**逐框镜像**（同面积、同离心率带、不与组内任何框及其它对照框重叠），
  保持"涂掉的总面积"在两臂可比。13/13 全部找到合法对照。
- 目视核查已做：`scene-0003_f196`（3 人横穿）三人全被覆盖、对照三块落在别处，多框路径正确。

## 两处必须知道的事实

1. **组外行人残留**：遮挡组之外仍可见的行人 均值 5.00 / 中位 4，
   仅 **3/13** 事件组外无行人。故本轮只排除"用到走廊内这组 VRU"，未排除组外行人与车辆。
   该 3 事件子集低于 bootstrap 门槛（<5 scene），**未出读数**，未硬凑。
2. **唯一说"because of the pedestrian"的事件 `scene-0917_f32`**，在 clean/ghost/occ/ctrl 四臂全部这么说，
   但**该帧组外还有 4 名行人清晰可见**，故**不能**当作"指称一个不存在的行人"的例证。
   本语料里没有找到干净的该类例证。

## 产出

- 报告：`results/simlingo_instruction_probe_newpool_report_zh.md`
- 主结果：`results/simlingo_instruction_probe_newpool.json`
- 汇总：`results/simlingo_instruction_probe_newpool_analysis.json`
- 精确检验：`results/simlingo_instruction_probe_newpool_exact.json`
- 目视核查：`results/figures/instruction_probe_newpool_qa/`
- 脚本：`scripts/simlingo_instruction_probe_newpool{,_analyze}.py`

全部为本轮新建文件。未修改 `simlingo_instruction_probe_occ.json` 等既有产物，
未修改 `brake_first_*` / `lane_path_*` 任何文件（只读引用），未写入 `amendments.md`。
