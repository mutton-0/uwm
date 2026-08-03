# G4 一致性验收报告（Tier-S，pool_mode=vision_mean）

> Tier-S 冒烟读数，事件量为几十级，统计功效极低，不作结论。

## 判定表

| # | 判据 | 通过标准 | 实测 | 判定 |
|---|---|---|---|---|
| V1 | 估计集 vs 真值集（总分） | 真值落在估计的 95% CI 内 | 估计 0.393 CI95=[0.053, 0.611]，真值 0.179 | ✅ PASS |
| V2 | 分层趋势 | 分层达标率排序 Spearman ≥ 0.8 | ρ=-0.50（共同层 3 个） | ❌ FAIL |
| V3 | 探针有效性 | selectivity 置换 p<0.05 **且** AUC(正例 vs D) 显著 >0.5 | p_perm=0.145，AUC(pos/D)=0.500 (p=1.000) | ❌ FAIL |
| V4 | 读出预测行为 | 投影-行为 ρ 显著非零 | ρ=-0.143，p=0.760 | ❌ FAIL |
| V4+ | 加强版（无标注预测器） | 估计集拟合 logistic → 真值集 AUC ≥ 0.65 | AUC=0.730 | ✅ PASS |
| V5 | 域信号存在 | D_L 曲线非平凡 + 干涉角 | Tier-S 按手册 §0.5 砍掉 CARLA 参考帧 → D_L / 干涉角 / V5 记 N/A | ⏸ N/A |

## 探针读数（同一 v_hazard / L*，在三块 held-out 上分别报数）

v_hazard 由 S_dir（13 事件）的正例 δ=h_ghost−h_clean 逐层 PCA 提出；峰层 L*=7 由 S_sel（8 事件）选出，共 24 层 × 896 维。

| 集合 | n | ρ_TTC | selectivity | p(置换) | AUC(ghost vs clean) | AUC(正例 vs D) | ρ(投影,行为) |
|---|---|---|---|---|---|---|---|
| S_test(手册主读数) | 7 (1+/6−, 1 scene) | -0.643 | 0.296 | 0.145 | 0.730 | 0.500 (1.000) | -0.143 (0.760) |
| truth(次级：完全 held-out，未参与提方向/选层) | 28 (15+/13−, 3 scene) | 0.373 | 0.214 | 0.058 | 0.467 | 0.303 (0.080) | 0.154 (0.435) |
| S_test ∪ truth(合并，提高功效) | 35 (16+/19−, 4 scene) | 0.313 | 0.182 | 0.055 | 0.507 | 0.303 (0.049) | 0.105 (0.550) |

> S_test 只剩 1 个 scene / 1 个正例，统计上退化；truth 集从未参与提方向与选层，在其上重算 V3/V4 作为功效更高的次级判定（Tier-S 偏离记录）。  次级判定：V3=❌ FAIL，V4=❌ FAIL。

## 失败形态（手册 §8：任一不过须报失败形态，不许只报“不显著”）

- **V2 失败形态**：估计集与真值集的共同分层只有 3 个，Spearman 在这么少的点上只能取到少数几个离散值（实测 ρ=-0.50）。这不是“趋势相反”的证据，而是**分层数不足以估计趋势**。
- **V3 失败形态**：主读数 S_test 上 AUC(正例 vs D)=0.500 (p=1.000)，置换 p=0.145。该集合只有 1 个正例 / 1 个 scene，**统计上退化**，功效更高的 truth holdout（15 正/13 负）上 AUC(正例 vs D)=0.303(p=0.080)、ρ_TTC=0.373。**AUC 低于 0.5**——v_hazard 在无害负例上的投影反而更高，指向「方向编码的是画面变化幅度而非危险」这一混淆（见 `results/deconfound_ablation.md`）。
- **V4 失败形态**：投影-行为相关在三块 held-out 上分别为 -0.143 / 0.154 / 0.105，符号不一致。V4+ 的 logistic AUC=0.730（达标，但与 V4 的 ρ 不一致，不应单独解读为“无标注预测器成立”）。

## 图

- `results/figures/layer_profile_vision_mean.png` — 层剖面（ρ_TTC / EVR1；D_L 缺席，V5=N/A）
- `results/figures/projection_behavior_vision_mean.png` — 投影-行为散点（S_test 与 truth 两块 held-out）
- `results/figures/consistency_vision_mean.png` — V1 总分一致性 + b_min 敏感性、V2 分层对比
- `results/figures/distributions_vision_mean.png` — 各类事件行为响应箱线 + 挖掘 TTC 分布
