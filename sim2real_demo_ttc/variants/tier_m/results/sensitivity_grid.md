# Tier-M 敏感性网格

> 主读数（★）在看到任何本档结果之前已冻结，见 `results/tier_m_preregistration.md`。
> 其余行是敏感性分析。**任何一行更好看都不构成替换主结论的理由。**

所有读数取自 **truth holdout**（该集合从未参与提方向、选层、调参）与 **S_test**（手册指定主判定集）。

| | 方向法 | 解混淆 | 池化 | L* | S_test AUC(正例/D) | truth AUC(正例/D) | p | truth ρ_TTC | p(置换) | truth ρ(投影,行为) | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ★ | supervised | none | vision_mean | 8 | 0.542 | 0.515 | 0.281 | -0.045 | 0.062 | -0.026 | 0.268 |
|  | pca | d_baseline | last_token | 21 | 0.498 | 0.488 | 0.386 | 0.029 | 0.201 | 0.223 | 0.000 |
|  | pca | d_baseline | vision_mean | 22 | 0.500 | 0.519 | 0.185 | -0.030 | 0.202 | -0.257 | 0.000 |
|  | pca | none | last_token | 23 | 0.568 | 0.491 | 0.516 | 0.008 | 0.743 | -0.310 | 0.000 |
|  | pca | none | vision_mean | 22 | 0.570 | 0.499 | 0.954 | -0.018 | 0.427 | -0.149 | 0.000 |
|  | supervised | d_baseline | last_token | 5 | 0.529 | 0.496 | 0.794 | -0.028 | 0.244 | -0.085 | 0.000 |
|  | supervised | d_baseline | vision_mean | 23 | 0.514 | 0.512 | 0.408 | -0.045 | 0.054 | -0.054 | 0.023 |
|  | supervised | none | last_token | 5 | 0.538 | 0.498 | 0.904 | -0.030 | 0.196 | -0.114 | 0.000 |

## 判读

- truth holdout 有 **1773** 个事件，AUC 的标准误约 ±0.015，因此 0.5 附近的偏离在这个样本量下是**可以判定为“没有效应”**的，而不是“功效不足”。
- 0/8 个格子在 truth 上给出显著且方向正确的 AUC(正例 vs D)。
- 与 Tier-S 的对照见 `../../results/deconfound_ablation.md`。
