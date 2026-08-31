# 工单:候选池扩展(Alpamayo-R1 补测 + TransFuser/LTF/DiffusionDriveV2/AutoVLA 接入)

> 承接 [headline_three_line_pilot_workorder.md](headline_three_line_pilot_workorder.md) 与 `results/headline_pilot_DONE.md`。**自主执行授权**,不需要逐步请示。

## 0. 方法纪律重申(不要跑偏)

**新候选一律用现有 G1 nuScenes 鬼探头语料 + N1 负例体系发现并测量 G/F/C**,不是"用各自的原生/仿真训练域发现方向、再迁移到 nuScenes 验证"——后者是之前明确讨论并否决过的设计("行为指纹法"),会把核心贡献从"真实榜单排名≠真实部署排名"退化成"sim2real gap"。唯一需要真实↔仿真域配对的是 **I 轴**,这是 I 轴定义本身要求的(不变性必须有两个域),不是通用策略。每个新候选的成本应该主要是**输入适配器**(把 G1 语料转成该模型的输入格式),复用 DiffusionDrive 的 `diffusiondrive_g1_adapter/` 作为参考实现范式。

## 1. 候选与已知状态

| 候选 | ckpt 状态 | 环境风险 | 优先级 |
| --- | --- | --- | --- |
| **Alpamayo-R1** | 已在本机(`/data/ruolin/alpamayo_ckpt`),已过 P1 | 已知可跑(P1 跑过) | **最高,零下载成本** |
| **LTF** | `ckpt/download_ckpts.sh` 一键下载(SimScale 官方,`LTF/ltf_sim_navtest.ckpt`) | 低——本仓库自带 `transfuser_agent.yaml` 配置与 `scripts/evaluation_navhard/run_transfuser_evaluation.sh` | 高 |
| **TransFuser** | 需先确认它与 LTF 是否共用 `transfuser_agent.yaml`(README 显示 LTF 条目直接链到该配置),若是同一套权重可直接复用,若不是需另找官方 ckpt | 低-中 | 高 |
| **DiffusionDriveV2** | 需从 `github.com/hustvl/DiffusionDriveV2` 拉取,官方称有开源权重 | 中——与 DiffusionDrive 同门,大概率复用现有 NAVSIM 环境,但需核实输入/输出接口是否变化 | 中 |
| **AutoVLA** | 需从 `github.com/ucla-mobility/AutoVLA` 拉取,HF 上有已合并 LoRA 的可直接推理权重 | 中高——全新代码库,VLA 架构(推测更接近 SimLingo 的 token 式输出而非 anchor/diffusion 头),需要新写适配器,不能假设复用 DiffusionDrive 适配器模式 | 中,给 1 天时间盒 |

**明确排除**:UniAD / VAD / SparseDrive——已在工单 P 确认 mmcv 1.x 与本机 sm_120 不兼容,HF 无 VAD 权重,不要重试。

## 2. 执行顺序(按成本递增,失败不阻塞后续)

1. **Alpamayo-R1**:直接测 G/F/C(复用现成 G1 语料 + 已有的 `alpamayo_runner.py` 适配器,应该只需要补 activation 读取接口用于 G/C,F 已有部分基础)。
2. **LTF**:下载 ckpt → 确认 `transfuser_agent.yaml` 推理接口 → 写 G1 输入适配器(参照 `diffusiondrive_g1_adapter/`)→ 测 G/F/C。
3. **TransFuser**:视步骤 2 的发现决定是否可直接复用同一权重/适配器,或需要单独下载官方 TransFuser ckpt。
4. **DiffusionDriveV2**:clone 仓库,尝试复用 DiffusionDrive 现有适配器的接口层,1 天时间盒,失败则跳过并记录原因。
5. **AutoVLA**:clone 仓库 + 下载 HF 权重,新写适配器(不要假设能复用其他模型的适配器),1 天时间盒,失败则跳过并记录原因。

每个候选测完后追加进 `headline_three_line_table_{zh,en}.md` 的矩阵(不要求所有候选都测满四轴,某候选若某操作化不适用,按 §4.3/§4.4.4 的既有纪律标"not applicable"而不是勉强凑数)。

## 3. 输出要求

同前几轮:每个候选一份中英双语报告(`results/axis_<candidate>_report_{zh,en}.md`),期刊 Methods/Results 结构,三线表;失败的候选也要出一份简短报告说明卡在哪一步、为什么放弃,不要静默跳过。全部完成后更新 `results/paper_experiments_section_{zh,en}.md` 的 Table 1/2,把新候选并入同一张矩阵。

## 4. 完成标志

`results/candidate_expansion_DONE.md`,列出每个候选的最终状态(完整测完 / 部分轴 not applicable / 放弃及原因)与所有产出物路径。
