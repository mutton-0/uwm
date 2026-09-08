# 作废：C-hazard 第一代结果（2026-08-31 / 09-01）

两个独立的致命问题，**不可修复，只能作废**：

1. **按结果预选样本**。用的是 `c_axis_hazard_patch.py` 的默认路径
   （`--k 12/14`，按实测 |v_clean − v_ghost| 降序取前 K）。每个模型各自
   挑自己响应最大的 12 个场景 —— 所以 13 个 cell 的场景集合**互不相同**，
   跨模型的 C_m 根本不在同一批场景上，排名无意义。
2. **语料过期**。这批跑在 2026-09-04 定稿的 brake-first 主分析集之前，
   与 `brake_first_pool_*_final.json` 的候选集**交集为 0 或 1**（逐 cell 实测）。

替代品（均为 `--all-events`、`readout=arc_full`、100% 落在新池内）：

| 用途 | 文件 |
|---|---|
| 主分析集，两语料 × 两场景 × 四候选 | `results/c_new_{ghost,lead}_{nusc,navsim}_{dd,ltf,ddv2,simlingo}.json` |
| 大样本 benchmark 侧（维加斯+匹兹堡，实测 100% LHD） | `results/c_dep_{ghost,lead}_*.json` |
| 按 P-1 舵位重切的汇总 | `results/c_axis_by_side.json`（`scripts/c_recut_side.py`） |

消费方 `scripts/axes_master.py`、`scripts/make_paper_tables.py` 读的都是 `c_new_*`，
**这批作废文件从未进入任何图表或论文数字**。保留仅供审计。
