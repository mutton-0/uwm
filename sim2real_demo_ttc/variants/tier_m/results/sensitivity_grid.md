# Tier-M 敏感性网格

> 主读数（★）在看到任何本档结果之前已冻结，见 `results/tier_m_preregistration.md`。
> 其余行是敏感性分析。**任何一行更好看都不构成替换主结论的理由。**

所有读数取自 **truth holdout**（该集合从未参与提方向、选层、调参）与 **S_test**（手册指定主判定集）。

| | 方向法 | 解混淆 | 池化 | L* | S_test AUC(正例/D) | truth AUC(正例/D) | p | truth ρ_TTC | p(置换) | truth ρ(投影,行为) | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ★ | supervised | none | vision_mean | 8 | 0.542 | 0.515 | 0.281 | -0.045 | 0.061 | -0.036 | 0.130 |
|  | pca | d_baseline | last_token | 4 | 0.640 | 0.526 | 0.071 | -0.051 | 0.026 | -0.303 | 0.000 |
|  | pca | d_baseline | vision_mean | 22 | 0.500 | 0.519 | 0.185 | -0.030 | 0.201 | -0.209 | 0.000 |
|  | pca | none | last_token | 2 | 0.557 | 0.541 | 0.004 | -0.087 | 0.000 | -0.215 | 0.000 |
|  | pca | none | vision_mean | 22 | 0.570 | 0.499 | 0.955 | -0.018 | 0.427 | -0.140 | 0.000 |
|  | supervised | d_baseline | last_token | 23 | 0.544 | 0.535 | 0.015 | -0.045 | 0.064 | -0.184 | 0.000 |
|  | supervised | d_baseline | vision_mean | 23 | 0.514 | 0.512 | 0.409 | -0.045 | 0.054 | -0.060 | 0.011 |
|  | supervised | none | last_token | 20 | 0.580 | 0.510 | 0.498 | -0.028 | 0.240 | -0.176 | 0.000 |

## 判读

- truth holdout 有 **1773** 个事件，AUC 的标准误约 ±0.015，因此 0.5 附近的偏离在这个样本量下是**可以判定为“没有效应”**的，而不是“功效不足”。
- 2/8 个格子在 truth 上给出显著且方向正确的 AUC(正例 vs D)：[('pca', 'none', 'last_token'), ('supervised', 'd_baseline', 'last_token')]。
- 与 Tier-S 的对照见 `../../results/deconfound_ablation.md`。
