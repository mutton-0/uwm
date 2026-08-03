# Sim2Real 差异的因果定位与失效机制探针 —— 方法与实验（Ghosthead × DiffusionDrive）

> 论文式组织：**设计依据 → 方法/指标 → 实验与数据分析 → 结论 → 局限**。
> 配合 `FINDINGS.md`（结论速览）与 `bokeh_ghosthead_sim2real.html`（交互图）。

## 摘要
在 CARLA 渲染(sim) 与世界模型生成真实感图(real) 两路输入下，对同一批 ghosthead 场景跑 DiffusionDrive，
系统比较其**规划后果、内部注意力/表征、内建感知头输出**。核心发现：sim↔real 的**内部表征差异真实存在**，
其对轨迹的**因果责任集中在深层融合层 L6**（而非注意力散度最大的 L7）；但**因果层强度不能预测后果退化幅度**，
且**"因果层 → 失效表现(BEV 语义/漏车) → 后果(碰撞)"的中介链尚未量化**——这是本工作的主要缺口。

---

## 1. 研究问题与设计依据

### 1.1 背景与目标
端到端规划器存在 sim2real gap。我们想回答的不只是"有没有 gap"，而是**gap 从模型哪一层进入、以什么失效
表现、最终如何变成后果**——即建立一条可干预、可验证的**机制链**，为低成本对齐（如因果定位的 LoRA）提供依据。

### 1.2 核心研究问题
- **Q1（后果）**：换域输入是否、以及多大程度改变规划**后果**（安全/进度）？
- **Q2（症状）**：内部表征在**哪一层**变化最大？
- **Q3（病因）**：这些变化里，**哪一层对轨迹改变负因果责任**？"变化最大"是否等于"因果责任"？
- **Q4（失效表现）**：换域下模型的**感知理解**（BEV 语义/它车检测）如何退化？
- **Q5（关联）**：因果层强度能否**预测**后果退化幅度？失效表现与因果层能否连成中介链？

### 1.3 设计逻辑链（为什么这样选指标 = 设计依据）
把 sim2real 失效拆成四层，每层配**对应问题的最小充分指标**，并刻意区分"相关/症状"与"因果/病因"：

```
输入(sim vs real)
   │  ①症状：表征哪变了      → 注意力 JS 散度 / 1-CKA 特征散度      (§3.2)  ——观测量，回答 Q2
   │  ②病因：谁造成轨迹改变  → 激活修补 recovery(L)                (§3.3)  ——干预量，回答 Q3
   │  ③失效表现：感知怎么错  → 内建 BEV 语义头/检测头 一致性探针    (§3.4)  ——免训读出，回答 Q4
   ▼  ④后果：规划好不好      → PDM-式总分 / ADE·FDE                (§3.1)  ——回答 Q1
   关联：因果强度 ↔ 后果幅度 → 相关性分析                          (§3.5)  ——回答 Q5
```

**关键设计依据**：
1. **症状≠病因，必须分开测**。可视化/散度只能说"哪层表征变了"，不能说"该改哪层"；因果主张需**干预**
   （激活修补：把某层换成参考值看后果是否恢复），而非相关。故 §3.2(症状) 与 §3.3(病因) 并列且刻意对照。
2. **复用模型自带头当失效探针（免训、自带 GT 侧）**。BEV 语义头/检测头是 ckpt 内置输出，argmax/sigmoid 即得，
   无需训练探针；在无外部 GT(CARLA 无 nuPlan 地图) 时改测 **sim-vs-real 一致性**。
3. **后果指标须与前三层可对齐**。同一场景、同一 GT、唯一变量是输入图，保证 §3.1 的后果差异可归因到"域"。
4. **不能用"与后果相关"验证因果**。§3.5 专门检验这一点（结果为负），提醒因果正确性只能靠干预判据。

---

## 2. 数据与实验设置
- **两路输入**：transfered=`renders/*/frames.mp4`(CARLA 渲染, sim)；origin=`ghosthead_result/*/seg1p0/*.mp4`(世界模型真实感, real)。1600×900/10fps/4s。
- **场景**：72 个 scene-variant（24 场景 × none/brake/swerve）；机制类实验取 real 退化 top-12。
- **输入构建**：单目居中裁 4:1 去天空(行250–650) → 等比 2048×512；模型只吃单帧；t=1/2/3/4s 各推一次。
- **坐标**：输入帧 ego 系（x 前向、y 左向、米）；预测轨迹 8 位姿@0.5s。
- **⚠️前提（未独立验证）**：ckpt(`diffusiondrive_sim_navhard`) "主要 sim 训练"→ sim 为 in-domain。方向性结论依赖它；
  §3.2–3.4 的**内部差异/因果**结论不依赖它。
- **复现**：固定随机种子（DDIM 采样有随机性）。

---

## 3. 方法与指标（设计依据 → 公式 → 测什么）

### 3.1 后果层：PDM-式总分（+ADE/FDE）
**设计依据**：需要一个同 GT 头对头的"规划好不好"标量；官方 PDMScorer 需 nuPlan 地图+metric_cache，
Town04(CARLA) 均无 → 自实现 PDM 式分，口径对齐官方但省略 drivable-area。ADE/FDE 作辅助但**对 sim 有偏袒**（GT 是 CARLA 定义）。

ego 足迹：矩形 L=4.6/W=1.9，中心=位姿+航向×1.461。actor 框由 `scene.json corners_world` 变到 ego 系凸包；超场景末尾常速外推。

| 分项 | 公式 |
|---|---|
| NoCollision(门) | 任一 i：ego框(i)∩actor框(i)≠∅ → 0，否则 1 |
| TTC(门) | 每步前推 0.95s 投影框相交 → 0，否则 1 |
| Comfort(二值) | \|lon_acc\|≤2.40 且 \|lat_acc=v·yaw_rate\|≤4.89 且 \|yaw_rate\|≤0.95 且 \|lon_jerk\|≤4.13 → 1 |
| Progress | clip(预测弧长 / GT ego 同段距离, 0,1)；GT<0.5m→1 |
| **total** | `NoCollision × TTC × (5·Progress + 2·Comfort)/7` |
| ADE / FDE | mean_i‖pred_i−gt_i‖ / ‖pred_last−gt_last‖（仅 cov>0 步）|

### 3.2 症状层：注意力 JS / 1-CKA 特征散度
**设计依据**：先定位"表征在哪层变化最大"作为**症状基线**，并用两种视角交叉验证——注意力分布(JS) 与
特征表征(1-CKA)；同时为 §3.3 的"症状≠病因"对照提供靶子。

- 注意力：每层 `att`(320×320,head 平均)，图像 key 重要度 `key_imp=att[:256,:256].mean(0)`(256)；归一后
  `JS(p,q)=½KL(p‖m)+½KL(q‖m), m=(p+q)/2`。
- 特征：层自注意力输出 X(reshape (320,C))，`CKA=‖Xtᵀ·Xo‖_F²/(‖Xoᵀ·Xo‖_F‖Xtᵀ·Xt‖_F)`（Kornblith 线性 CKA，中心化），散度=**1−CKA**。

### 3.3 病因层：激活修补 recovery(L) ★核心因果
**设计依据**：因果主张需**干预**而非相关。以 sim(in-domain) 为参考、real 为退化，把 real 前向某层输出**替换为 sim 的**，
看轨迹是否恢复到 sim——恢复=该层因果负责。这直接检验 Q3 并可证伪"按散度选层"。

- `gap=‖traj_real−traj_sim‖`（xy，8×2 L2）
- 逐层：real 前向第 L 层输出←sim 缓存 → traj_patchL；`recovery(L)=1−‖traj_patchL−traj_sim‖/gap`
- 充分性检查：patch 全 8 层 → recovery≈1。

### 3.4 失效表现层：内建头一致性探针（免训）
**设计依据**：机制链缺"感知怎么错"这一环。复用 ckpt 自带 BEV 语义头/检测头（免训），在无外部 GT 时测
**sim-vs-real 一致性**，把"表征变了"落到"看到的场景变了"。

- forward → `bev_semantic_map`(1,7,128,256)→argmax→(128,256)类别图；`agent_states`(30,5)、`sigmoid(agent_labels)`(30)。
- per-class IoU=`|mask_sim_c∩mask_real_c|/|∪|`（空-空计 1.0）；vehicle 像素=`(bev==5).sum()`；ghost 置信度=离 GT ghost 最近 slot 的 sigmoid。
- 几何：128×256 @0.25m≈64×32m ego BEV。backbone：图像 resnet34 + 可学习 latent resnet34 + GPT 8 层融合（见 FINDINGS 附3）。

### 3.5 关联层：因果强度 ↔ 后果退化
**设计依据**：直觉是"因果层越强→退化越重"；但更重要的是检验**能否用后果相关来验证因果定位**（若能，方法学更省）。
- y=`Δtotal(t=1s)=transfered_total−origin_total`
- 因果强度候选：`L6=max(rec[6],0)`、`deep_share=clip(rec,0)[4:7].sum()/clip(rec,0).sum()`、`gap·deep_share`、`gap·L6`、`gap`
- 报 Pearson/Spearman。

---

## 4. 实验结果与数据分析

### 4.1 后果（Q1）
| | transfered(sim) | origin(real) |
|---|---|---|
| total(n=288) | **0.568** | 0.538 |
| NoCollision | 0.795 | 0.753 |
| brake 变体 | **0.638** | 0.575（差最大 +0.063）|
| ADE / FDE | 2.24 / 4.38 | 2.65 / 5.17 |
**分析**：方向与"sim 稳、real 掉点"一致，**但幅度小、61% 场景两路 total 相同** → **弱证据**；差异集中在最吃感知的
brake 场景。ADE 方向同，但因 GT 偏袒 sim，仅作旁证。

### 4.2 症状（Q2）
JS 逐层均值峰 **L7 0.29**、L0 0.14，6/6 场景 argmax=L7；1-CKA 峰 **L7 0.77**。图像 vs lidar 注意力占比几乎不变(0.784≈0.792)。
**分析**：表征变化最大在**最深融合层 L7 与最浅 L0（首尾）**；不是图像/雷达注意力重分配，而是**深层图像注意力落点**变。

### 4.3 病因（Q3）★
patch-ALL=1.0(12/12)；recovery 均值峰 **L6 0.31**、L7 0.30、L5 0.19；argmax 分布 **L6:5/L7:3/L5:2**；深层 L4-6 主导 9/12。
**分析**：**因果责任在深层 L6/L4-6**，而非散度峰 L7/L0 →「**症状≠病因**」独立复现。低成本对齐应放 **L6/L5**。

### 4.4 症状 vs 病因 对照 + 对齐度
1-CKA↔recovery **Spearman +0.64** > JS↔recovery **+0.24** → 特征对齐比注意力散度更贴因果；但 1-CKA 的 argmax 仍误落 L7
→ **CKA 也非可靠单层定位器**，最终定位仍须激活修补。

### 4.5 失效表现（Q4）
BEV 语义 sim-vs-real IoU：**centerline 0.12**(最不一致)/road 0.43/vehicle 0.48；vehicle 像素 **161→100(−38%)**；
ghost 置信度 0.207 vs 0.171（逐场景混、inconclusive）。
**分析**：换域使模型**场景理解显著漂移**，尤其车道与它车；vehicle 像素普降为 brake 撞车提供**机制假说**（漏看正前车）——但见 §4.6，这条链**未因果闭合**。

### 4.6 关联（Q5）
因果强度候选对 Δtotal 全部 **相关≈0**（−0.08~+0.16）；仅 gap↔origin绝对分 −0.28。
**分析**：**位点稳定 ≠ 后果幅度可预测**——后果有符号且几何门控（偏进障碍才 Δ 大），gap/recovery 无符号；
方法学上**不能用"与后果相关"验证因果定位**。

---

## 5. 结论
1. **内部表征 sim↔real 差异真实存在**，且不依赖训练域前提。
2. **对轨迹的因果责任集中在深层 L6/L4-6**（干预证据），**≠ 散度最大的 L7/L0**（症状）；1-CKA 相关更好但仍非可靠单层定位器。
3. **后果层退化弱而方向一致**（sim 略优、brake 最明显），受二值门+GT 偏袒限制。
4. **失效表现**（BEV 语义漂移、vehicle 像素普降）清晰，但**与因果层未量化连接**。
5. **因果强度不预测后果幅度**，且"与后果相关"非验证因果的正确判据。

---

## 6. 局限与下一步（本工作主要缺口）
**核心缺口——因果→失效→后果的中介链未量化**：目前 recovery 只测"轨迹是否恢复"，没测 patch L6 后
**BEV vehicle 像素/语义是否也恢复、碰撞余量是否也恢复**。要闭合机制链，需做**中介(mediation)分析**：
```
patch L6 (real→sim) 后，同时观测：轨迹 recovery、BEV 语义/vehicle-像素 recovery、碰撞余量 recovery
若三者同步恢复 ⇒ 「L6 表征漂移 → BEV 漏车 → 碰撞」链条成立并可量化各环节贡献
```
其余：
- **后果指标偏粗**：二值 NoCollision/TTC 使 y 离散(46% 平局)。拟加**连续碰撞余量**
  `margin=min_i min_a distance(ego(i), actor(i,a))`（可带符号取穿透深度），提升 §3.5 灵敏度。
- **因果只做充分性**：需补**必要性(消融破坏该层)+控制组(随机层不恢复)**，排除旁路。
- **训练域前提**、**GT 为 CARLA 定义**、**样本量/选择性**（见 FINDINGS 局限）。
- **更干净的因果实验**：ghosthead 天然是 do(ghost) 干预，配 matched 反事实（有/无 ghost、远/近）可量化"模型是否因果地对危险物响应"。

---

## 7. 汇报要点（claim → 支撑数据 → 必带 caveat）
可直接照此讲，每条都配"一句话+数字+一个保留"，避免过度断言。

1. **sim2real 差异真实存在，但后果层退化是"弱而方向一致"**。
   支撑：total sim 0.568 vs real 0.538、brake 差最大 +0.063。
   caveat：61% 场景两路 total 相同、指标被二值门离散化 → 弱证据，且方向依赖"ckpt 主要 sim 训练"这一未验证前提。

2. **症状（表征变化最大）在首尾层 L7/L0**。
   支撑：JS 峰 L7 0.29、1-CKA 峰 L7 0.77，6/6 场景 argmax=L7。
   caveat：这是相关/症状，不能据此选微调层。

3. **病因（对轨迹的因果责任）在深层 L6/L4-6，≠ 症状层**——「症状≠病因」。
   支撑：激活修补 recovery 峰 L6 0.31、深层 L4-6 主导 9/12、argmax L6:5/12；patch-ALL=1.0 证明方法自洽。
   caveat：只做了充分性，未做必要性+控制组。

4. **特征对齐(1-CKA) 比注意力散度(JS) 更贴因果，但仍不能单独定位**。
   支撑：1-CKA↔recovery Spearman +0.64 > JS 的 +0.24；但 1-CKA argmax 仍误落 L7。
   caveat：定位仍须干预（激活修补）。

5. **换域下感知理解明显漂移（失效表现）**。
   支撑：BEV 语义 IoU centerline 0.12/road 0.43、vehicle 像素 161→100(−38%)。
   caveat：vehicle 像素是均值主导的机制假说、IoU 被空-空平凡 1.0 抬高、agent 置信度 inconclusive。

6. **因果层强度不能预测后果退化幅度**（重要方法学点）。
   支撑：因果强度各候选↔Δtotal 相关≈0。
   caveat：位点稳定≠后果可预测；且"与后果相关"本就不是验证因果的正确判据。

7. **主要缺口**：**因果层 → 失效表现 → 后果** 的中介链**未量化**（recovery 只测轨迹，未测 BEV/碰撞余量是否同步恢复）。

**一分钟版**：域切换真会让模型内部表征变化（首尾层最明显），但**真正驱动规划变化的是深层 L6**（散度大的层反而不因果）；
换域下模型的 BEV 感知（车道/它车）明显漂移、倾向漏看正前车，为 brake 撞车提供机制假说；不过**"L6 漂移→漏车→碰撞"
这条链还没被因果地量化连起来**，且因果强度不预测后果幅度——这是下一步（中介分析 + 连续碰撞余量 + 必要性/控制组）。

---

## 8. 全部结果一览（便于填汇报表）
| 层 | 指标 | sim(transfered) | real(origin) | 峰/要点 |
|---|---|---|---|---|
| 后果 | PDM total(n=288) | 0.568 | 0.538 | brake 差 +0.063 |
| 后果 | NoCollision 率 | 0.795 | 0.753 | — |
| 后果 | ADE / FDE (m) | 2.24 / 4.38 | 2.65 / 5.17 | GT 偏袒 sim |
| 症状 | JS 峰层 | — | — | **L7** 0.29 (L0 0.14) |
| 症状 | 1-CKA 峰层 | — | — | **L7** 0.77 |
| 症状 | 图像注意力占比 | 0.784 | 0.792 | 几乎不变 |
| 病因 | recovery 峰层 | — | — | **L6** 0.31 (L7 0.30) |
| 病因 | argmax 因果层分布 | — | — | L6:5 / L7:3 / L5:2 (n=12) |
| 病因 | 深层 L4-6 主导 | — | — | 9/12 场景 |
| 对齐 | 1-CKA↔rec / JS↔rec | — | — | **+0.64** / +0.24 (Spearman) |
| 失效 | BEV IoU road/cl/veh | — | — | 0.43 / **0.12** / 0.48 |
| 失效 | vehicle 像素均值 | 161 | 100 | −38%（假说）|
| 失效 | ghost 置信度 | 0.207 | 0.171 | inconclusive |
| 关联 | 因果强度↔Δtotal | — | — | ≈0（−0.08~+0.16）|

---

## 9. 复现命令（`conda activate simscale`，env 见 FINDINGS §1）
```bash
cd /data/ruolin/uwm
PY=/home/mut0/.conda/envs/simscale/bin/python
# 后果(§3.1/4.1)：72 场景推理+评分
$PY scripts/ghosthead_infer/run_ghosthead_infer.py --all
# 症状 JS + RGB 叠加(§3.2/4.2)
$PY scripts/ghosthead_infer/attn_diff_analysis.py --k 6
# 病因 recovery(§3.3/4.3)
$PY scripts/ghosthead_infer/run_patching_ghosthead.py --k 12
# 对齐 CKA vs recovery(§3.2/4.4)
$PY scripts/ghosthead_infer/run_cka_recovery_ghosthead.py --k 12
# 失效 内建头探针(§3.4/4.5)
$PY scripts/ghosthead_infer/run_head_probe_ghosthead.py --k 12
# 关联 因果↔PDM(§3.5/4.6)
$PY scripts/ghosthead_infer/run_causal_pdm_corr_ghosthead.py
# 汇总交互 HTML
$PY scripts/ghosthead_infer/build_bokeh_ghosthead_sim2real.py
```
产物目录：`outputs/ghosthead_infer/{<scene>,attn_diff,patching,cka_recovery,head_probe,causal_pdm}/` + `_summary_all.csv`。

---

## 10. 术语与符号
| 记号 | 含义 |
|---|---|
| sim / transfered | CARLA 渲染输入（in-domain，前提） |
| real / origin | 世界模型生成真实感输入（out-of-domain） |
| L0..L7 | 8 个编码器融合自注意力层 `encoder_selfatt`（4 尺度×2 Block，4 头，320 token=图256+lidar64） |
| JS | Jensen-Shannon 散度（注意力分布差异，症状） |
| CKA / 1−CKA | Kornblith 线性 Centered Kernel Alignment / 特征散度 |
| recovery(L) | 激活修补恢复度（因果责任，∈(−∞,1]，1=完全修复） |
| gap | ‖traj_real−traj_sim‖ 轨迹 L2 差 |
| ADE / FDE | 平均 / 终点位移误差（vs GT ego，米） |
| IoU | 交并比（此处 sim vs real 类别掩码重叠比） |
| Δtotal | transfered_total − origin_total（PDM 退化幅度，含符号） |
| cov | gt_coverage_s，该帧有真实未来 GT 的秒数 |
| patch-ALL | 同时替换全 8 层（充分割集检查，应≈1.0） |
