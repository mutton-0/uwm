# Ghosthead × DiffusionDrive 推理评分 — Findings

> 📊 **交互式总览（推荐）**：[`bokeh_ghosthead_sim2real.html`](./bokeh_ghosthead_sim2real.html) —— 7 tab 汇总所有对照分析（逐层 JS/CKA/因果、注意力叠加、BEV 语义、内建头、因果↔PDM），支持 hover/缩放/下拉切场景。生成脚本 `scripts/ghosthead_infer/build_bokeh_ghosthead_sim2real.py`。
> 📄 **方法与实验（论文式，供自读/汇报）**：[`METRICS.md`](./METRICS.md) —— 设计依据→方法/公式→实验数据分析→结论→局限，含汇报要点/结果一览/复现命令/术语表。

数据：`/data/Zhengyang/Auto_Eval/ghosthead_v1`，72 个 scene-variant（24 场景 × none/brake/swerve），
每个 8 次推理（transfered/origin × t=1/2/3/4s）。脚本：`scripts/ghosthead_infer/run_ghosthead_infer.py`。
ckpt：`diffusiondrive_sim_navhard.ckpt`。

> ⚠️ **关键前提（未独立验证）**：本报告"sim 是 in-domain、real 才退化"的方向性结论，**依赖"该 ckpt 主要在
> sim 域训练"这一前提**——它来自 ckpt 命名与使用者说明，我**没有独立核实**训练数据域。若该 ckpt 实为真实
> 域(navsim/OpenScene)训练，则整个方向翻转（应是 real=in-domain、sim 退化）。**下方所有 sim/real 谁更好的
> 解读都以此前提为条件**；而**因果定位、注意力/CKA/内建头分析（附1–5）不依赖该前提**（它们只比较两路输入
> 下模型的内部差异，与"谁是 in-domain"无关）。

## 两路输入的域
- **transfered = CARLA 渲染（sim，in-domain）** — 来源 `renders/<scene>/frames.mp4`（游戏引擎画面）。
- **origin = 世界模型生成的真实感图（real，out-of-domain）** — 来源
  `ghosthead_result/<scene>/seg1p0/<scene>_seg1p0.mp4`（写实光照/材质，同一构图重绘）。

## 命题
训练偏 sim → 在 sim 输入下模型 OK；搬到 real（真实感）输入才退化。

## 结论：数据**弱**支持"sim OK、real 退化"（小幅、方向一致；依赖上方前提）

安全 total 对两路用**同一个场景 GT**（同 actor、同 ego GT），唯一变量是输入图 → 不同预测轨迹 → 同口径打分。
碰撞项对两路检的是**同一批 actor box**，这一项公平；但整体 total 的 progress 仍以 CARLA 定义的 ego-GT 计、
且被碰撞/TTC 二值门离散化，并非完全域中立、也偏粗。故它比 ADE 干净，但只能作**弱证据**。

| 指标 | transfered (sim, in-domain) | origin (real, out-domain) |
|---|---|---|
| 安全 total（均值，n=288/源） | **0.568** | 0.538 |
| NoCollision 率 | **0.795** | 0.753（碰撞更多） |
| **brake 变体** total（n=96/源） | **0.638** | 0.575（差距最大 +0.063） |
| none 变体 total | 0.496 | 0.471 |
| swerve 变体 total | 0.570 | 0.567 |

→ 同一批场景，喂 sim 图规划碰撞略少、total 略高；差异在 brake 变体最大。方向与"sim 稳、real 掉点"一致，
**但 margin 很小**（0.568 vs 0.538）、**61% 场景两路 total 完全相同**——是**小幅、方向一致的暗示，非强证据**。

## 辅助指标 ADE/FDE（方向一致，但对 sim 有偏袒，不作主证据）

**ADE = Average Displacement Error**（预测轨迹与 GT 在各未来时刻欧氏距离的均值，米，越小越贴 GT）；
**FDE = Final Displacement Error**（只看 4s 终点的距离）。仅统计 cov>0（有真实未来 GT）的帧。

| | transfered (sim) | origin (real) |
|---|---|---|
| ADE vs GT | 2.24 m | 2.65 m |
| FDE vs GT | 4.38 m | 5.17 m |
| 配对：谁更偏离 GT | 39% | 61% |

⚠️ **混淆注意**：GT ego 轨迹本身来自 CARLA 仿真，transfered 又是同段 CARLA 的渲染 → 图像与 GT
天然对齐，**结构性夸大** sim 的优势。故 ADE/FDE 只作方向佐证，主证据用上面的安全 total（不受此混淆影响）。

## 口径与假设（复现须知）
- **非官方 PDM 分**：Town04=CARLA，无对应 nuPlan 地图、无 metric_cache，官方 `PDMScorer` 不可用；
  自实现 PDM 式分 = `NoCollision × TTC × (5·Progress + 2·Comfort)/7`，**省略 drivable-area/车道**（无路面几何）。
- 单目 16:9 居中裁 4:1 去天空（行 250–650）→ 等比放大 2048×512，无变形。
- 模型只吃单帧（`cameras[-1]`）；4 帧各独立推理一次。
- `driving_command` 用直行 one-hot（无路由信息，假设）。
- backbone 走进程内 monkeypatch（`pretrained=False`），权重由 ckpt 提供，不下载。
- 固定随机种子，结果可复现。
- t=4s 帧几乎无未来 GT（cov=0，靠常速外推），参考性最低。

## 附：real 退化场景的 encoder self-attention 分布差异

脚本 `scripts/ghosthead_infer/attn_diff_analysis.py`。对 real 退化最狠的 top-6 场景
（`gh_001010__brake` 最极端：sim 0.90 → real 0.00；及 001007/001021/001022__brake、001011/001015__none），
抓 DiffusionDrive 编码器 8 层 `encoder_selfatt`（每层 1×4×320×320，256 图 token + 64 lidar），
对 sim(transfered) vs real(origin) 输入比较图像 key 重要度分布。

> 本节量的是**注意力散度=症状**，不是退化的因果位点。散度峰在哪层 ≠ 退化从哪层因果进入（因果见附2，峰在 L6）。

**结论**：
1. **注意力散度峰在 L7**——逐层 JS(sim‖real) 呈 U 形：L7 峰值(mean 0.24)、L0 次高(0.12)、L1 及中层≈0，
   **6/6 场景 argmax JS 都在 L7**。与历史 sim2real 的散度结论一致（首尾层散度最大）。
2. **不是图像/雷达注意力重分配**：图像 key 注意力占比 sim 0.784 ≈ real 0.792（几乎不变）；
   变的是**深层图像注意力在全景内的落点**（见 `overlay_<scene>.png` L7 列 sim/real 明显不同）。
3. **含义**：real 输入使深层表征(尤 L7)变化最大，但那是症状；**该在哪层干预由因果决定（附2：L6/L4–L6，故 LoRA
   放 L6/L5，不是 L7/L0）**。散度与因果在此分离，正是"症状≠病因"。

产物 `outputs/ghosthead_infer/attn_diff/`：
- `attn_layer_js.png` — 逐层 JS(sim‖real) 曲线（均值 + 各场景）
- `attn_share_entropy.png` — 图像注意力占比 & 熵，sim vs real
- `overlay_<scene>.png` — 每场景 sim/real 各 8 层图像注意力叠加 RGB（2×8）
- `attn_diff_summary.csv` — 每场景每层 JS / 熵 / 图像占比

## 附2：因果定位（激活修补，复现 research_plan §4.3「症状≠病因」）

脚本 `scripts/ghosthead_infer/run_patching_ghosthead.py`（**复用原 `sim2real_analysis/activation_patching.py`
内核**，仅换 ghosthead 输入前端）。设定：参考=transfered(sim, in-domain)、退化=origin(real)；跑 real 前向、
把第 L 层 encoder SelfAttention 输出换成 sim 的，测轨迹回到 sim 的比例 `recovery(L)`。n=12 real 退化场景，t=1s。

**结论**：
1. **patch ALL = +1.00（12/12）** → 8 个 self-attn 是差异的充分割集，方法自洽。
2. **因果层是深层 L4–L6（尤 L6）**：各层 recovery 均值 L6=+0.31、L7=+0.30、L5=+0.19；
   argmax 分布 **L6:5 / L7:3 / L5:2** / L2:1 / L4:1；**深层 L4-6 主导 9/12 场景**。
3. **散度≠因果**（招牌结论，ghosthead 独立复现）：JS 症状峰在 **L7/L0**，而因果峰在 **L6**
   （L6 的 JS 近最低、recovery 却最高）。→ 低成本对齐应把 **LoRA 放 L6/L5 深层融合**，而非按散度动 L7/L0。
   注：ghosthead 上 L7 除散度高外也有可观因果(0.30)，比原 navsim 结论(L7 因果≈0)更"首尾兼有"，但 L6 仍单层最强。

产物 `outputs/ghosthead_infer/patching/`：`recovery_by_layer.png` + `recovery.csv`（每场景逐层 recovery + gap + ALL）。

## 附3：内建头一致性探针（training-free，plan §4.2 O2）

脚本 `scripts/ghosthead_infer/run_head_probe_ghosthead.py`。**不训练任何探针**，直接读 DiffusionDrive ckpt
**自带**的 BEV 语义头(7 类) + agent 检测头，比较 sim vs real 输入下模型"看到的场景"。无外部 GT（CARLA 无
nuPlan 地图），故测 sim-vs-real **一致性**而非绝对精度。n=12 real 退化场景，frames 1–3。

**结论**：
1. **模型场景理解对域高度敏感**：BEV 语义 sim-vs-real IoU — **centerline 0.12（最不一致）**、road 0.43、
   vehicles 0.48（后者含若干两路皆 0-车像素的平凡一致，实际更低）。
2. **real 输入下倾向少画它车（机制假说）**：vehicle 类 BEV 像素均值 sim 161 → real 100（−38%），为 brake 撞车
   提供一个机制假说（real 图让模型少画/漏画正前方车辆）。典型 `gh_001007__brake` sim 画两团 vehicle、real 只剩
   一团、道路几何还变形（见 `bev_sem_<scene>.png`）。**注意**：−38% 是均值、被少数高像素场景主导
   （如 gh_001013 sim 903/real 511），非每场景都降；vehicle 的 sim-vs-real IoU(0.48) 又被若干"两路皆 0 车像素"的
   平凡一致抬高。故此项是**方向性机制假说，非定量定论**。
3. **agent 检测头置信度：inconclusive**：ghost 处 conf 均值 sim 0.207 vs real 0.171，但逐场景是混的
   （gh_001021 sim0.34/real0.80、gh_001022 sim0.00/real0.65 均 real 更高），**不能支持"real 更没把握"**。
   稀疏检测头信号不可靠；场景理解退化的证据以第 2 条 BEV vehicle 像素为准（且仅作假说）。

产物 `outputs/ghosthead_infer/head_probe/`：`head_probe_bars.png` + `bev_sem_<scene>.png` + `head_probe_summary.csv`。

### BEV 语义头提取流程与 backbone

**这些是 ckpt 自带的输出头，普通 forward 一遍就在返回字典里，无需训练、无需 hook。**

编码 BEV 的 backbone（`TransfuserBackbone`，TransFuser 式双分支）：
- **图像分支**：`timm resnet34`（`image_architecture='resnet34'`, features_only）→ 多尺度图像特征(layer1–4)。
- **BEV/lidar 分支**：也是 `timm resnet34`（`lidar_architecture='resnet34'`），但 v2 (`latent=True`) **无真实 lidar**，
  输入换成一个**可学习 latent 张量** `self.lidar_latent`(nn.Parameter, 1×C×256×256)，由这条 resnet34 产出 BEV 特征。
- **融合**：GPT 式 transformer（4 尺度 × 2 Block = **8 个 `encoder_selfatt`**, 4 heads），逐尺度用 1×1 conv 互投影后
  跨分支自注意力融合图像 token(256) + BEV/lidar token(64) = 320。
- 融合后的 `bev_feature_upscale` 经 `_bev_semantic_head` 出 7 类语义图；`_agent_head` 出 30 个 agent 框。

提取（`run_head_probe_ghosthead.py::infer_heads`）：
```python
out  = model({"camera_feature": cam, "status_feature": status})
bev  = out["bev_semantic_map"][0].argmax(0).cpu().numpy()   # (1,7,128,256) logits -> argmax -> (128,256) 类别图
ag   = out["agent_states"][0].cpu().numpy()                 # (30,5) 框(x,y,heading,l,w)
conf = torch.sigmoid(out["agent_labels"][0]).cpu().numpy()  # (30,) 检测置信度
```
- 类别：0 bg / 1 road / 2 walkways(绿) / 3 centerline / 4 static / 5 vehicles(红) / 6 pedestrians。
- 几何：BEV 128×256 @ `bev_pixel_size=0.25 m` ≈ **64 m 宽 × 32 m** 的 ego 俯视范围。
- 派生量：某类像素数 `(bev==c).sum()`；sim-vs-real 一致性 = 两张类别图的 per-class IoU（空-空计 1.0，会抬高均值）。

## 附4：CKA-vs-recovery 三元对照（复用原 `cka_vs_recovery.py` 内核）

脚本 `scripts/ghosthead_infer/run_cka_recovery_ghosthead.py`。对每层比较 注意力 JS(症状) / 1-CKA 特征散度 /
recovery 因果，看哪种散度更贴因果。n=12。

**逐层均值**（峰值层）：JS→**L7**(0.29)、1-CKA→**L7**(0.77)、recovery→**L6**(0.31)。

**相关性（逐层均值向量）**：

| | JS ↔ recovery | **1-CKA ↔ recovery** |
|---|---|---|
| Pearson | +0.39 | **+0.63** |
| Spearman | +0.24 | **+0.64** |

**结论**：**特征对齐(1-CKA) 比注意力散度(JS) 明显更贴近因果**（复现 research_plan）；但 1-CKA 的 argmax 仍误落
**L7**、抓不准单层最强的 **L6** → CKA 相关性好但**非可靠单层因果定位器**，最终定位仍须用激活修补。
产物 `outputs/ghosthead_infer/cka_recovery/`：`cka_vs_recovery.png` + `cka_recovery.csv`。

## 附5：因果层强度 ↔ PDM 退化 相关性（plan §4.2 验收关卡 —— 负结果）

脚本 `scripts/ghosthead_infer/run_causal_pdm_corr_ghosthead.py`。全部 72 场景、t=1s：算 sim/real 轨迹 gap +
逐层激活修补 recovery（复用 `activation_patching` 内核），与该帧 PDM Δtotal(=transfered−origin) 做相关。

**结果：因果层强度基本不预测 PDM 退化幅度（相关≈0）。**

| 预测量 ↔ PDM Δtotal(t=1s) | Pearson | Spearman |
|---|---|---|
| 轨迹 gap ‖real−sim‖（全部 72） | +0.10 | −0.03 |
| gap（|Δtotal| 绝对变化） | +0.09 | +0.04 |
| L6 recovery(归一) / 深层L4-6占比 / gap×深层占比（n=62, gap>1） | ≈ 0（−0.08 ~ +0.16） | ≈ 0 |
| gap ↔ origin 绝对分 | **−0.28** | **−0.25** |

**解读**：
1. **因果"位点"稳定 ≠ 因果"后果强度"可预测**。修补告诉我们退化从**哪层**进入（L6，稳定），但**退多少**由
   **场景几何**决定——同样大小的表征漂移，轨迹偏进障碍就撞(Δ大)、偏到空地就没事(Δ≈0)。gap 无符号、后果有符号
   且几何门控，故不相关。唯一弱信号：域漂移越大、real 绝对分略低(−0.28)。
2. **方法论含义（重要）**：plan §4.2 把"探针/因果退化 ↔ PDM delta 相关"当验收关卡——此负结果表明
   **用"与后果相关"来验证因果定位并不可靠**。因果正确性须靠**干预**（充分性=patch 恢复、必要性=消融破坏、
   控制组=随机层不恢复），**不能**用散度、CKA、或与下游分数的相关替代。
3. 局限：本 y（PDM-式 total）被二值门离散化（46% 平局），压低了相关上限；换连续"碰撞余量"作 y 或许更灵敏。

产物 `outputs/ghosthead_infer/causal_pdm/`：`causal_pdm_scatter.png` + `causal_pdm.csv`。

---

## 稳健结论 vs 局限

**稳健结论（不依赖"训练域"前提，只比较两路输入下模型内部差异）**：
1. 8 个 encoder self-attn 是 sim/real 差异的**充分割集**（patch-ALL=1.0，12/12）。
2. **因果层 = L6 / L4–L6**：激活修补 argmax L6:5/12、深层主导 9/12；散度峰 L7、因果峰 L6 → **散度≠因果**。
3. **1-CKA 比 JS 更贴因果**（Spearman +0.64 vs +0.24），但 argmax 仍误落 L7，非可靠单层定位器。
4. sim↔real 下模型**内部场景理解确有显著差异**（BEV 语义 IoU 低、注意力落点变、深层表征散度大）。
5. **因果层强度不预测 PDM 退化幅度**（附5，相关≈0）：位点稳定 ≠ 后果强度可预测。

**局限（限制结论强度）**：
1. **训练域前提未独立验证**（见开头 ⚠️）：sim/real 谁是 in-domain 依赖"ckpt 主要 sim 训练"这一未核实前提；
   若翻转，"sim OK/real 退化"的方向随之翻转。稳健结论 1–5 不受影响。
2. **PDM-式 total 非官方且偏粗**：省略 drivable-area、被碰撞/TTC 二值门离散化（46% 平局），压低相关灵敏度。
3. **GT 为 CARLA 定义**：ADE/FDE 结构性偏袒 sim；安全 total 仅碰撞项域中立，progress 仍引用 CARLA-GT。
4. **样本与帧**：因果/探针主要在 t=1s、real 退化 top-12（附5 用全 72）；n 偏小、退化样本有选择性。
5. **机制假说未定量**：BEV vehicle 像素 −38% 为均值主导的方向性假说；agent 检测头置信度 inconclusive。
6. **验证判据**：不能用"与 PDM 相关"验证因果定位（附5）；因果正确性须靠干预（充分+必要+控制组）。

## 产物
- 总表：`outputs/ghosthead_infer/_summary_all.csv`（576 行）
- 每场景：`outputs/ghosthead_infer/<scene>/` → `bev_compare.png`(2×4) + `bev/` + `inputs/` + `scores.csv` + `pred_traj.json`
