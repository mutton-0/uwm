# DONE — 第二步：ghost 场景收敛

**报告**：`results/ghost_scenario_final_zh.md`

**结论**：四候选在两语料上都没有稳定的、足够幅度的危险响应。
响应比 nuScenes 0.150/0.091/0.005/−0.015、NAVSIM 0.986/0.048/0.186/0.191；
方向一致率八个格子没有一个明显偏离掷硬币（最高 9/13，最低 5/13）。

**唯二两个"显著"都不稳**：SimLingo/NAVSIM 的 CI 下界仅 +0.0004 且被 2 个事件主导
（其余 6 个 ≤0.65、3 个为负）；DDv2 在 NAVSIM 显著为正但在 nuScenes 是反向的 −0.0265，
两语料结论相反，在 8/13 scene 的功效下无法判断。

**GT 轴跨语料稳定**：nuScenes vs NAVSIM 的 b_GT 为 1.06 vs 1.00（SimLingo 时域）、
1.71 vs 1.64（DD 家族时域），两边都各自全正。

**候选只跑了四个，六候选未跑齐**：Alpamayo/AutoVLA 的数据加载绑死 nuScenes devkit，
NAVSIM 上无法运行；nuScenes 上可运行但需把多帧遮挡从"单个目标"改造成"遮挡组"。
如实列为待决，不是省略。

**失效案例**：nuScenes 最有说服力的是 scene-0862（人类停车、SimLingo 反而规划加速、
位置特异）；NAVSIM 侧没有同等干净的案例（反向数最多 2/4 且都带已登记诊断问题），如实说明。

**两语料角色**：nuScenes=benchmark、NAVSIM=deployment，完全独立不合并。
NAVSIM 功效更低（8 vs 13 scene），故"NAVSIM 响应比更高"不作跨语料优劣断言。
