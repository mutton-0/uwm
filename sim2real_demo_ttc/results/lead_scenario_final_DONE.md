# DONE — 第三步第 1 类：前车 / 静态路障场景

**报告**：`results/lead_scenario_final_zh.md`

**三个核心发现**：
1. **SimLingo 类别特异盲区**：同一模型/语料/方法，响应比 ghost(VRU) 0.150 不显著
   vs lead(车辆) 2.081 显著，差 14 倍；lead 在 NAVSIM 206 scene 上以 86% 方向一致率复现。
2. **benchmark vs deployment 排名倒置**：LTF 从 benchmark 第 2 掉到 deployment 第 4（末位），
   b 符号翻转（+0.4582 显著 -> -0.1720 跨0），方向一致率 49.3%（掷硬币）。
   Spearman ρ=+0.400 (p=0.600, n=4)。**功效更高的 deployment 侧（206 vs 16 scene）给出否定结论。**
3. **DDv2 无正向响应跨语料复现**：lead 场景 nuScenes -0.029、NAVSIM -0.009，两边都跨 0，
   是四候选中唯一如此的。

**QA 抓到场景特有失效模式并修正**：scene-0047 的"危险"是沿路边一整排护栏（12 个），
a_req 149.67 = 实测减速的 150 倍。两处场景特定调整：静态物走廊收紧到 1.0m
（缓冲是给会动的目标留的，静态物不会动）；explained_ratio 加上界 10
（几何与人类行为的一致性检查，非按模型结果筛样本）。已验证不污染 ghost 场景。

**如实列出的局限**：静态路障子类未构成有效样本（收紧后 nuScenes 归零）⇒ 本场景实质是
"前车"场景，标题应改；六候选未跑齐（两 VLA 绑死 nuScenes devkit）；
nuScenes 仅 16 scene，两语料功效差 13 倍；QA 为分层抽样非全量。
