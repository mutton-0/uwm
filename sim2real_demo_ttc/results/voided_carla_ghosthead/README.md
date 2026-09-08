# 作废：C-domain（CARLA / ghosthead 那一套）

**理由同 P-6（I 轴弃用 CARLA）**，逐条对应：

1. **训练域在相反一侧**。SimLingo 训练在 CARLA（`sim` 侧在域内），TransFuser 系
   训练在真实数据（`real` 侧在域内）。任何"sim↔real 配对"的读数都混着
   "你的训练域在哪边"，跨族不可比。
2. **量的是渲染风格，不是域移**。`gh_001010__brake` 这类场景是 ghosthead 渲染，
   与本文四轴共用的 brake-first 事件**零重叠**。
3. **覆盖面不够做任何比较**：只有 DiffusionDrive 一个候选、12 个场景。

产物来源：`/data/ruolin/uwm/outputs/ghosthead_infer/patching/recovery.csv`（2026-07-14）。

**这批文件不被 `scripts/axes_master.py`、`scripts/make_paper_tables.py`
或 `paper_v2.3_icra/sec/*.tex` 中任何一处引用** —— 实测 grep 无命中，
从未进入论文数字。仅 `scripts/c_axis_shape.py` / `scripts/h1_three_line_table.py`
两个早期脚本读它，两者均已停用。保留仅供审计。

C 轴当期有效集见 `PRINCIPLES.md` §P-8。
