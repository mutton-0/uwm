# G4 一致性验收报告（Tier-M-longgap，pool_mode=vision_mean）

> Tier-M-longgap：估计集 501 / 真值集 1673 事件，统计功效已可支撑趋势判断，但仍受限于 nuScenes 的日夜与场景构成。

## 判定表

| # | 判据 | 通过标准 | 实测 | 判定 |
|---|---|---|---|---|
| V1 | 估计集 vs 真值集（总分） | 真值落在估计的 95% CI 内 | 估计 0.321 CI95=[0.279, 0.365]，真值 0.309 | ✅ PASS |
| V2 | 分层趋势 | 分层达标率排序 Spearman ≥ 0.8 | ρ=0.07（共同层 10 个） | ❌ FAIL |
| V3 | 探针有效性 | selectivity 置换 p<0.05 **且** AUC(正例 vs D) 显著 >0.5 | p_perm=0.041，AUC(pos/D)=0.613 (p=0.043) | ✅ PASS |
| V4 | 读出预测行为 | 投影-行为 ρ 显著非零 | ρ=-0.112，p=0.216 | ❌ FAIL |
| V4+ | 加强版（无标注预测器） | 估计集拟合 logistic → 真值集 AUC ≥ 0.65 | AUC=0.519 | ❌ FAIL |
| V5 | 域信号存在 | D_L 曲线非平凡 + 干涉角 | Tier-S 按手册 §0.5 砍掉 CARLA 参考帧 → D_L / 干涉角 / V5 记 N/A | ⏸ N/A |

## 探针读数（同一 v_hazard / L*，在三块 held-out 上分别报数）

v_hazard 由 S_dir（221 事件）的正例 δ=h_ghost−h_clean 逐层 PCA 提出；峰层 L*=4 由 S_sel（156 事件）选出，共 24 层 × 896 维。

| 集合 | n | ρ_TTC | selectivity | p(置换) | AUC(ghost vs clean) | AUC(正例 vs D) | ρ(投影,行为) |
|---|---|---|---|---|---|---|---|
| S_test(手册主读数) | 124 (40+/84−, 48 scene) | -0.185 | 0.112 | 0.041 | 0.503 | 0.613 (0.043) | -0.112 (0.216) |
| truth(次级：完全 held-out，未参与提方向/选层) | 1673 (585+/1088−, 615 scene) | -0.060 | 0.039 | 0.014 | 0.510 | 0.559 (0.000) | 0.014 (0.579) |
| S_test ∪ truth(合并，提高功效) | 1797 (625+/1172−, 663 scene) | -0.069 | 0.051 | 0.004 | 0.510 | 0.562 (0.000) | 0.005 (0.828) |

> S_test 只剩 1 个 scene / 1 个正例，统计上退化；truth 集从未参与提方向与选层，在其上重算 V3/V4 作为功效更高的次级判定（Tier-S 偏离记录）。  次级判定：V3=✅ PASS，V4=❌ FAIL。

## 失败形态（手册 §8：任一不过须报失败形态，不许只报“不显著”）

- **V2 失败形态**：共同分层 10 个，Spearman=0.07 未达 0.8。分层数足够，因此这是一个**实质性的不一致**：估计集与真值集在分层层面的达标率排序确实不同，需要逐层看 `behavior_scores` 里哪几层反号。
- **V4 失败形态**：投影-行为相关在三块 held-out 上分别为 -0.112 / 0.014 / 0.005，符号不一致。V4+ 的 logistic AUC=0.519（同样未达标）。

## 图

- `results/figures/layer_profile_vision_mean.png` — 层剖面（ρ_TTC / EVR1；D_L 缺席，V5=N/A）
- `results/figures/projection_behavior_vision_mean.png` — 投影-行为散点（S_test 与 truth 两块 held-out）
- `results/figures/consistency_vision_mean.png` — V1 总分一致性 + b_min 敏感性、V2 分层对比
- `results/figures/distributions_vision_mean.png` — 各类事件行为响应箱线 + 挖掘 TTC 分布
