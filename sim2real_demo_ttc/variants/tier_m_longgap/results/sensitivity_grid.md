# Tier-M-longgap 敏感性网格

> 主读数（★）在看到任何本档结果之前已冻结，见 `results/tier_m_preregistration.md`。
> 其余行是敏感性分析。**任何一行更好看都不构成替换主结论的理由。**

所有读数取自 **truth holdout**（该集合从未参与提方向、选层、调参）与 **S_test**（手册指定主判定集）。

| | 方向法 | 解混淆 | 池化 | L* | S_test AUC(正例/D) | truth AUC(正例/D) | p | truth ρ_TTC | p(置换) | truth ρ(投影,行为) | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ★ | supervised | none | vision_mean | 4 | 0.613 | 0.559 | 0.000 | -0.060 | 0.015 | 0.055 | 0.023 |
|  | pca | d_baseline | last_token | 22 | 0.392 | 0.506 | 0.675 | -0.004 | 0.873 | 0.260 | 0.000 |
|  | pca | d_baseline | vision_mean | 23 | 0.471 | 0.497 | 0.844 | 0.011 | 0.635 | -0.128 | 0.000 |
|  | pca | none | last_token | 10 | 0.535 | 0.526 | 0.075 | -0.020 | 0.389 | 0.290 | 0.000 |
|  | pca | none | vision_mean | 8 | 0.640 | 0.542 | 0.005 | -0.040 | 0.101 | 0.076 | 0.002 |
|  | supervised | d_baseline | last_token | 2 | 0.549 | 0.535 | 0.019 | -0.060 | 0.012 | -0.044 | 0.069 |
|  | supervised | d_baseline | vision_mean | 4 | 0.618 | 0.556 | 0.000 | -0.057 | 0.020 | 0.051 | 0.035 |
|  | supervised | none | last_token | 3 | 0.463 | 0.549 | 0.001 | -0.063 | 0.006 | 0.002 | 0.919 |

## 判读

- truth holdout 有 **1673** 个事件，AUC 的标准误约 ±0.015，因此 0.5 附近的偏离在这个样本量下是**可以判定为“没有效应”**的，而不是“功效不足”。
- 5/8 个格子在 truth 上给出显著且方向正确的 AUC(正例 vs D)：[('supervised', 'none', 'vision_mean'), ('pca', 'none', 'vision_mean'), ('supervised', 'd_baseline', 'last_token'), ('supervised', 'd_baseline', 'vision_mean'), ('supervised', 'none', 'last_token')]。
- 与 Tier-S 的对照见 `../../results/deconfound_ablation.md`。
