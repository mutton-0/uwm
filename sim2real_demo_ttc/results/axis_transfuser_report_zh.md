# 候选扩展：TransFuser —— 判定为**与 LTF 同一可执行对象**，不单列

> 工单 §1 要求"先确认 TransFuser 是否与 LTF 共用 `transfuser_agent.yaml`"。
> 本文件是该项核查的结论报告。登记见 [`amendments.md`](amendments.md) §CE/A28。
> 本候选**没有产生独立的轴读数**，理由如下；其位置由 [`axis_ltf_report_zh.md`](axis_ltf_report_zh.md) 覆盖。

---

## Methods：核查了什么

三条独立核查，全部指向同一结论。

**① 仓库配置。** `navsim/planning/script/config/common/agent/transfuser_agent.yaml` 的内容为
`_target_: navsim.agents.transfuser.transfuser_agent.TransfuserAgent`，
配置体里写死 **`latent: True`**。`latent: True` 即 **Latent TransFuser**——
用可学习 BEV latent 顶替真实 lidar 分支。本仓库**只提供这一个 TransFuser 系配置**，
没有 `latent: False` 的变体。

**② 官方权重清单。** `ckpt/download_ckpts.sh` 列出的 SimScale 官方 ckpt 为
DiffusionDrive、GTRS_Dense、**LTF** 三个；TransFuser 系只有 `LTF/ltf_sim_navtest.ckpt` 一个，
**没有带 lidar 的 TransFuser 权重**。

**③ 刺激集约束（决定性的一条）。** 本工作线的共享刺激集是 nuScenes **单目相机** G1 语料
（经 4:1 裁剪去天空适配），**不含 lidar**。带 lidar 的 TransFuser 即便另行取得权重，
也无法按其训练口径在本刺激集上推理；喂零 lidar 等于把它换成另一个模型
（这一点在本轮 DiffusionDriveV2 上有直接的反面教训，见 §CE/A33）。

---

## Results

**Table 1. Determination for the TransFuser candidate under this repository and this stimulus set.**

| 核查项 | 事实 | 结论 |
| --- | --- | --- |
| 仓库配置 | `transfuser_agent.yaml` 写死 `latent: True` | 该配置**就是** LTF |
| 官方权重 | TransFuser 系仅 `ltf_sim_navtest.ckpt` | 无独立 TransFuser 权重可用 |
| 刺激集 | nuScenes 单目相机，无 lidar | 带 lidar 的 TransFuser **不可执行** |
| 最终状态 | — | **与 LTF 同一可执行对象，不单列** |

> **仪器侧**：三条核查互相独立（配置、权重、输入模态），任何一条单独成立都足以说明
> "本条件下 TransFuser 不是一个与 LTF 相区分的候选"；三条同时成立，结论无歧义。
> **标本侧**：TransFuser 系在本工作矩阵中的位置由 LTF 占据，其读数见 LTF 报告。

---

## Discussion

**1. 这不是"跳过"，而是"二者在本条件下是同一个可执行对象"。**
工单把 TransFuser 与 LTF 分列为两个候选，是基于 README 里 LTF 条目直接链到
`transfuser_agent.yaml` 这一观察提出的疑问；核查结果证实了该疑问：它们共用同一份配置与同一份权重。
把同一组权重跑两遍、在矩阵里写成两行，会**虚增候选数并制造独立性的假象**。

**2. 顺带得到的收获比单列一行更有价值。**
LTF 与 DiffusionDrive 共用同一个 `TransfuserBackbone(latent=True)` 编码器、只差动作头，
因此这一对构成了一次**同编码器、异动作头**的受控对照。
上一轮把 DiffusionDrive 的注入法不可测归因于其扩散动作头，
该对照直接否定了这一归因（见 §CE/A31 与 LTF 报告 Discussion §3）。
**如果当初把 TransFuser 硬凑成独立一行，这个对照反而不会被识别出来。**

**3. 若要真正把带 lidar 的 TransFuser 纳入。** 需要两件事同时满足：
①取得官方 lidar 版权重（不在 SimScale 发布清单内）；
②为本刺激集构造 nuScenes lidar BEV 输入（本轮已为 DiffusionDriveV2 实现，见 §CE/A29，技术上可复用）。
两者齐备后它才是一个**与 LTF 相区分**的候选。本轮不做，如实记录。

---

## 自我更正记录

1. 本报告是**核查结论**而非实验失败记录：TransFuser 没有"卡在某一步"，
   而是在本仓库与本刺激集条件下与 LTF 不可区分。按工单 §3"失败的候选也要出报告、不要静默跳过"的要求
   出具本文，但状态应读作 **"合并入 LTF"** 而非 **"放弃"**。
