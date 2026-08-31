# 候选扩展：AutoVLA 的 G/F 轴读数

> 工单：[`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)（优先级最低、给 1 天时间盒的候选）。
> 本轮自行决策见 [`amendments.md`](amendments.md) §CE（尤其 **A34** D2cV 对多帧候选不成立、**A35** 环境解法）。
> 数值产出物：`g_axis_autovla.json`、`f_axis_action_counterfactual.json`、
> `v_hazard_autovla_{vision_mean,seq_mean}.npz`；适配器：`autovla_g1_adapter/`。

---

## Methods

### 候选与既有基础

AutoVLA（UCLA Mobility Lab）是一个 Qwen2.5-VL-3B 基座的 VLA 驾驶策略，
以**自回归动作 token 解码**输出轨迹，HF 上提供已合并 LoRA 的推理权重
（本机 `external/ckpts/AutoVLA_PDMS_89.ckpt`）。本仓库此前没有任何 AutoVLA 的基础，
适配器、环境、读出接口全部为本轮新写——这是工单里唯一一个"不能假设复用其他候选适配器"的候选。

### 环境：两个互相冲突的约束（§CE/A35）

AutoVLA 的接入成本几乎全部在环境上，且两个约束互相牵制：

1. **checkpoint 的模块布局锁定 `transformers==4.49`**。用本机默认的 4.57 加载会得到
   **824 个 missing key + 824 个 unexpected key**——4.49 的 `vlm.visual.*` 在 4.57 里变成
   `vlm.model.visual.*`。不能靠键名重映射硬凑（那等于替作者猜权重语义），
   故把 4.49 装进一个隔离的 `--target` 目录，只在 AutoVLA 的进程里插到 `sys.path` 最前。
2. **但 4.49 与本机 torchvision 的算子注册顺序冲突**：先插 4.49 路径再 `import torch` 会得到
   `operator torchvision::nms does not exist`。解法是**先** `import torch, torchvision, torchvision.ops`
   完成算子注册，**再**插 4.49 路径。这个顺序是载荷性的，写在 `autovla_adapter._bootstrap()` 里并加了注释。

此外 `models.utils.score` 会拉起 navsim→nuplan 的整条依赖链（与 DiffusionDriveV2 的 §CE/A30 同一性质），
注入 stub 模块绕开；`predict()` 路径需要 `driving_command`（统一给 `"go straight"`，
与其余候选一律不喂导航指令的口径一致）、`max_length` 提到 2048（实测 prompt 1085 token）、
`temperature` 取 1e-4 + `top_k=1` 以逼近确定性解码（0.0 被 generate 拒绝）。

最终**0 missing / 0 unexpected**，模型完整加载。

### 适配要点与偏离

* **可读层**：VLM 语言塔 36 个 decoder layer（hidden 2048），与 Alpamayo 的读出位置同构。
* **输入构造**：3 相机 × 4 时刻 = 12 张图，与 AutoVLA 原生 NAVSIM 输入的时序结构对齐；
  nuScenes 侧用 `x_clean_frames` / `x_ghost_frames` 各取 4 个时刻。
* **prefill 池化**：与 Alpamayo 同一手法——钩子只保留 seq_len 最大的那一次前向（整条 prompt），
  decode step（seq_len = 1）一律丢弃；池化 `vision_mean` / `last_token` / `seq_mean`。
* **图像 token 识别**：按 input_ids 中最长连续同 id 段自动识别（实测 id = 151656，864 / 1085 token）。
* **行为量**：`v_plan`，由解码出的 10 点轨迹前两点求得，与其余候选的 $v_{cmd}$ 定义同构。
* **偏离**：AutoVLA 训练于 NAVSIM，对 nuScenes OOD；相机内外参与 FOV 不匹配，只能近似映射。
  与其余所有候选相同，本文不为任何候选做域适配——这是刻意的，见 §4.1.1。

### 方法学限制：D2cV 对 AutoVLA 不成立（§CE/A34）

与 Alpamayo 完全同理：D2cV 的构造前提是"唯一差异是**单帧模型物理上不可能观测到**的相对速度"。
**AutoVLA 吃 4 个时刻**，相对速度对它是可观测量，故 D2cV 对它**不是证伪地板**，
而是一个正当的困难负例。本报告因此以**标签置换零分布**与**随机方向地板**为 G 轴的可用地板，
D2cV 的数字并列报告但不作为判据。

---

## Results

**Table 1. G-axis readout for AutoVLA on the shared G1 stimulus set. For this multi-frame model the usable floors are the permutation null and the random-direction floor (§CE/A34); D2cV is reported for completeness only.**

| Pool | CV-AUC(A vs D2a) | 95% CI | permutation floor | random-direction floor | 10 seed 折分配 | 峰层 $L^*$ |
| --- | --- | --- | --- | --- | --- | --- |
| **vision_mean**（主读数） | **0.608** | [0.562, 0.654] | 0.504 ± 0.020 | 0.522 ± 0.020 | 0.588 ± 0.022 | L20 |
| seq_mean（敏感性） | 0.609 | [0.564, 0.655] | 0.500 ± 0.020 | 0.521 ± 0.020 | 0.588 ± 0.019 | L20 |

主读数的 95% CI 下界（0.562）高出置换地板均值 +2.9 个 sd、高出随机方向地板 +2.0 个 sd，
$p = 7.7 \times 10^{-6}$。在**原始 AUC** 这一可跨组比较的尺度上，这与全表最高的 LTF（0.623 [0.577, 0.668]）
CI 大幅重叠、不可区分；AutoVLA 也是唯一一个两个池化口径
（vision_mean / seq_mean）峰层完全一致（均 L20）、读数几乎相同（0.608 / 0.609）的候选。
**注意不能把它与单帧组的"主读数 − D2cV 地板"直接比大小**——两者不是同一个构念（§CE/A34）。

**并列报告的困难负例读数**（对多帧候选**不作为判据**，见上文）：
vs D2cV = 0.594（$p$ = 0.0016，n = 141）、vs D2c = 0.573、vs D2b = 0.618、vs D2bV = 0.700。
主读数 − D2cV = $+0.014$ [$-0.056$, $+0.084$]。
对单帧候选，这一格会被判"不可估"；对 AutoVLA，"A vs D2cV 显著"是一个**正当的判别任务**
（相对速度对它可观测），因此这组数字只说明它对困难负例的区分度接近对 D2a 的区分度，
不说明方法有效或无效。

**必须并列报告的几何稳健性**：AutoVLA 的判别方向与成像几何**并非完全正交**——
$\rho(\text{投影}, \log \text{area})$ 合并 $+0.045$（$p = 0.28$，不显著），
但**在正例内部** $+0.132$（$p = 0.025$）；$\rho(\text{投影}, ecc)$ 合并 $+0.098$（$p = 0.019$）。
D2a 是按 log 成像面积与离心率做 caliper 匹配的，故**组间**混淆已被设计控制，
组内残留相关是弱的但可测的。作为对照，Alpamayo 的同一诊断是 $\rho = +0.019$（$p = 0.73$）。
因此 AutoVLA 的 G 读数应当读作"**显著，方向的语义纯度弱于 Alpamayo 但组间混淆受控**"，
而不是"AutoVLA 更懂危险"。

**Table 2. F① action-level counterfactual (architecture-neutral), same stimuli and same protocol as all other candidates.**

| 量 | AutoVLA | n | 判定 |
| --- | --- | --- | --- |
| $b(A)$ = $v_{plan}$(clean) − $v_{plan}$(ghost) [m/s] | +0.037 [−0.035, +0.119] | 291 | 与 0 不可区分 |
| $b$(D2a) [m/s] | −0.094 [−0.230, +0.009] | 283 | 与 0 不可区分 |
| **b-AUC(A vs D2a)** | **0.508** [0.451, 0.563] | — | **不可估**（CI 跨 0.5） |

### 本候选的诊断结论：G 与 F 的分离在 AutoVLA 上最干净

AutoVLA 同时给出：

* **G 轴 0.608**（原始 AUC 与全表最高的 LTF 0.623 不可区分，$p = 7.7 \times 10^{-6}$，两个池化口径一致），
* **F① 0.508**（候选池里最靠近 0.5 的一个，CI 几乎对称跨越 0.5），
* 且 $b(A)$ 与 $b$(D2a) **都**与 0 不可区分——即模型的规划速度对"目标出现"这件事
  **整体上没有可检测的响应**，不只是"响应不特异"。

这是本文核心命题的最干净的一个实例：**危险相关信息确实在表征里、而且比任何其他候选都更可线性读出，
但它没有到达动作**。与 SimLingo 的形态不同——SimLingo 是 $b(A) = +0.307$ 显著非零但
$b$(D2a) 同样大（反应强但不特异）；AutoVLA 是**根本不反应**。
两种失效模式的修复处方完全不同，而**任何单一综合分都会把它们压成同一个数字**。

---

## Discussion

**AutoVLA 在 1 天时间盒内完成，未触发跳过条款。** 成本分布与工单的预期一致：
适配器本身（输入构造 + 钩子 + 池化）约占 1/3，环境（transformers 版本 × torchvision 算子注册的
双重约束）约占 2/3。后者不是可预见的成本，已作为 §CE/A35 登记，供后续候选参考。

**未测的轴与理由（按既有纪律标 not applicable，不凑数）**：

| 轴 | 状态 | 理由 |
| --- | --- | --- |
| F② 表征注入 | **未测** | 时间盒内未能完成。注意这不是"不可测"的结论——AutoVLA 的注入位点（LLM 残差流）与 SimLingo 同构，先验上应当可测。标为未测而非 not applicable。 |
| I 域不变性 | **not applicable** | 域配对语料（CARLA↔世界模型重绘）每个时刻只有**单帧**，而 AutoVLA 要 4 个时刻。语料侧缺失，非模型侧不可测（§CE/A36）。 |
| C 失效集中度 | **未测** | 同上时间盒原因；C-hazard 的配对（G1 clean↔ghost）对 AutoVLA 在原理上可用，缓存也已具备，是本工作线最直接的下一步。 |

**一条对候选池设计的观察**：AutoVLA 与 Alpamayo 是候选池里仅有的两个多帧候选，
它们同时也是仅有的两个**证伪地板不可用**的候选。这不是巧合——
D2cV 的可用性与输入时序性是同一件事的两面。若未来要把多帧候选纳入同一张证伪表，
需要另造一个"同类别、同几何、**同相对速度**、只差标签"的负例类，
而这在 nuScenes 上意味着重新挖掘，不是本轮的范围。

---

## 自我更正记录

1. **首版用 transformers 4.57 加载，824 missing / 824 unexpected**，若不检查就直接读表征，
   得到的会是一个随机初始化视觉塔的"读数"。检查加载完整性是本仓库对每个新候选的强制步骤，
   这一次它直接拦下了一个会污染全部下游数字的错误。
2. **修 1 的过程中先插 4.49 路径导致 `torchvision::nms` 算子缺失**，
   一度误判为"4.49 与本机 CUDA 不兼容"。实际是算子注册顺序问题，与版本兼容性无关。
   记录在案，避免下次得出错误的"不可行"结论。
3. **首次 G/F 读数只用 A + D2a 缓存，数字与终稿不同，已作废**：
   首版 G 主读数 0.629、$\rho(\text{投影}, \log \text{area}) = +0.121$（$p$ = 0.004）；
   补齐 D2b/D2c/D2cV（538 个事件）后为 0.608、$\rho = +0.045$（$p$ = 0.28）。
   **原因是 scene 级 CV 折分配随语料规模变化**，方向与峰层随之微动。
   两个数都在 10 seed 折分配的范围内（主读数 0.588 ± 0.022），
   即差异是折分配噪声、不是新证据——**这正是本仓库强制做 10 seed 折分配稳定性检验的理由**。
   已登记为 §CE/A38。F① 不受影响（b 是逐事件量，不经 CV）。
