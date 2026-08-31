# 候选扩展：Alpamayo-R1 的 G/F 轴读数

> 工单：[`../docs/candidate_pool_expansion_workorder.md`](../../docs/candidate_pool_expansion_workorder.md)（零下载成本、最高优先级候选）。
> 本轮全部自行决策见 [`amendments.md`](amendments.md) §CE（尤其 **A34**：D2cV 对多帧候选不成立）。
> 数值产出物：`g_axis_alpamayo.json`、`f_axis_action_counterfactual.json`、
> `v_hazard_alpa_{vision_mean,seq_mean}.npz`；适配器：`alpamayo_g1_adapter/`。

---

## Methods

### 候选与既有基础

Alpamayo-R1（NVIDIA，10B VLA，flow-matching 动作头 + VLM rollout 产生 Chain-of-Causation）。
权重已在本机（`/data/ruolin/alpamayo_ckpt`），此前在工单 P 的阳性对照资格赛里跑过
（`p1_alpamayo_clean.json`：b-AUC 0.446，**资格赛未过**）。
本轮**不重跑行为读数**，而是补上此前缺失的**表征端 G 轴**：
新写 `alpamayo_g1_adapter/alpa_g1_cache.py`，在**预填充（prefill）阶段**逐层池化并落盘。

### 适配要点与偏离

* **可读层**：VLM 语言塔 36 个 decoder layer（hidden 4096）。
* **prefill 选取**：Alpamayo 的推理是 VLM rollout（先生成 CoT 再采轨迹），钩子会被触发很多次；
  我们只保留**seq_len 最大的那一次**（整条 prompt 的前向），decode step（seq_len = 1）一律丢弃。
* **图像 token 识别**：按 input_ids 中**最长连续同 id 段**自动识别（实测 id = 151655，
  2880 / 3006 token 为图像），识别结果随缓存落盘供核查。
* **在钩子内当场池化**（6 路相机 ⇒ S ≈ 3000，全张量保不住）：`vision_mean` / `last_token` / `seq_mean`。
* **ego 运动史锚定到 clean 帧**（与 SimLingo 的 `prompt_anchor`、navsim 系的 status 锚定同构）。
* **覆盖预筛**：adapter 无条件构造 6.4 s 未来轨迹，贴近场景末尾的事件不可用 ⇒
  A 类 **188 / 291**、D2a **157 / 283** 可用（与 P1 的可用率完全一致，交叉核对通过）。
* **既有偏离沿用**（P1 已登记）：Alpamayo 训练于 NVIDIA PhysicalAI-AV，对 nuScenes 同样 OOD；
  相机 FOV 失配（期待 120° 广角 + 30° 长焦，nuScenes 六相机均为 70°，只能近似映射）。

### **一处必须先讲清楚的方法学限制（§CE/A34）**

D2cV 证伪地板的设计理由是"与正例同类别同几何、唯一差异是相对速度，而**单帧模型对目标运动结构性全盲**"。
**Alpamayo 是多帧模型**（1.6 s ego 运动史 + 多时刻图像），相对速度对它**是可观测的**，
因此 **D2cV 对 Alpamayo 不再是证伪地板**——"A vs D2cV 显著"是一个正当判别任务，不是泄漏警报。
故本报告对 Alpamayo 的 G 轴，**以标签置换零分布与随机方向地板为可用地板**，
D2cV 的数字并列报告但**不作为判据**。

---

## Results

**Table 1. G-axis readout for Alpamayo-R1 on the shared G1 stimulus set. For this multi-frame model the usable floors are the permutation null and the random-direction floor (see §CE/A34); the D2cV column is reported for completeness only.**

| Pooling | 主读数 CV-AUC(A vs D2a) | 95% CI | $p$ | 置换零分布 | 随机地板 | 主读数 − 置换地板 | 10 seed 上的主读数 | D2cV（**不作判据**） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vision_mean`（主口径） | **0.562** | **[0.502, 0.623]** | 0.046 | 0.504 | 0.508 | **+0.058** | 0.563 ± 0.029 | 0.517 |
| `seq_mean`（敏感性） | 0.565 | [0.504, 0.625] | 0.038 | 0.501 | 0.501 | +0.064 | 0.562 ± 0.027 | 0.522 |

n = 188 正例 / 157 负例（覆盖预筛后）。峰层 $L^\*$ = 18（36 层中的中段），两种池化一致。
几何稳健性：$\rho$(投影, log 面积) = +0.019（$p$ = 0.73）、$\rho$(投影, 离心率) = +0.082（$p$ = 0.13），
**两项均不显著**。
判定：**主读数显著高于置换零分布与随机地板**（CI 下界 0.502 > 0.5，两种池化一致，10 seed 稳定）；
但**按 D2cV 口径不可估**（+0.045 [−0.041, +0.129]），而该口径对多帧模型本就不适用。

**Table 2. F① action-level counterfactual (architecture-neutral), same stimuli as all other candidates.**

| Quantity | Value | 95% CI | $p$ | Verdict |
| --- | --- | --- | --- | --- |
| $b$(A)，危险帧引起的动作变化 [m/s] | **−0.080** | [−0.178, +0.017] | — | 与 0 不可区分，**且符号为负** |
| $b$(D2a) [m/s] | +0.025 | [−0.104, +0.160] | — | 与 0 不可区分 |
| **b-AUC(A vs D2a)** | **0.446** | [0.374, 0.519] | 0.082 | **不可估**（且在 0.5 以下） |
| b-AUC(A vs D2cV) | 0.456 | [0.364, 0.551] | 0.415 | 不可估 |

**交叉核对**：b-AUC = 0.4456 与工单 P 的 `p1_alpamayo_clean.json` 报告的 0.4456 **逐位一致**，
说明本轮的适配器与 P1 的行为读数口径完全一致，不是另起炉灶。

> **仪器侧**：可用率（188/291、157/283）与 P1 逐条一致；b-AUC 与 P1 逐位一致 ⇒ 适配器口径可信。
> G 轴的正结果在**两种池化、10 个折分配 seed 上一致**，且几何稳健性两项均不显著。
> **但对 Alpamayo 必须换地板**：D2cV 的证伪前提（单帧不可观测速度）对多帧模型不成立，
> 故本报告以置换/随机地板为判据，并明确标注这使 Alpamayo 的 G 轴**与单帧候选不完全同判据**。
> **标本侧**：Alpamayo 的**表征端能读出危险**（主读数显著高于置换地板 +0.058），
> 但**动作端不但没有相应响应，方向还是反的**（b(A) = −0.080，b-AUC = 0.446 < 0.5）。

---

## Discussion

**1. Alpamayo 给出四轴框架里最干净的一例「G 通、F 断」。**
表征端：主读数 0.562 [0.502, 0.623] 显著高于置换零分布（0.504）与随机地板（0.508），
两种池化、10 个 seed 一致，几何稳健性不显著 ⇒ **危险信息在表征里是可线性读出的**。
动作端：b(A) = −0.080（符号为负 = 危险帧里反而规划得更快），b-AUC = 0.446 < 0.5 ⇒
**动作不但没跟着危险走，还略微反向**。
这正是本工作线核心主张里"奠基了但未驱动动作（F 环断裂）"的教科书式实例，
而且它出现在一个**表征端明确有信号**的候选上——比 SimLingo 那种"两端都读不出"的情形更有说服力。

**2. 这条结论必须挂上两个限定，缺一不可。**
① **地板不同**：Alpamayo 的 G 轴是以置换/随机地板判定的，不是以 D2cV 证伪地板；
按 §CE/A34 这是多帧模型的必然，但它意味着 Alpamayo 的 G 轴阳性**强度弱于 LTF 的阳性**
（后者越过了更严的 D2cV 地板）。
② **域外性**：Alpamayo 训练于 PhysicalAI-AV，对 nuScenes 是 OOD，且相机 FOV 失配；
本读数不构成对其原生域表现的任何推断。

**3. 与 P1 阳性对照结论的关系：本轮补上了当时缺的那一半。**
工单 P 当时只测了行为端，得出"资格赛未过"（b-AUC 0.446），
但**无法区分**"模型没看见"与"看见了但没驱动动作"。
本轮补上表征端后答案是明确的：**它看见了**。
因此 P1 的"资格赛未过"应被重新理解为**F 环断裂**，而不是感知缺失——
这是一次对既有结论的**解释性修订**（不是数值修订，b-AUC 逐位未变）。

**4. F② 与 C 轴未测，原因如实记录。**
F②（表征注入）：Alpamayo 是 flow-matching 动作头 + VLM rollout，
单次前向 ~2.7 s，一次完整注入四件套（40 事件 × 169 条件 × 2 帧）需 ≈ 4 h GPU，
超出本轮候选扩展的预算分配；且按本轮 §CE/A31 的发现，注入可测性由编码器决定，
Alpamayo 的编码器与已测三族均不同，其结果无法从已有结论外推 ⇒ **标为未测，不标不适用**。
C 轴：需要域配对或 clean↔ghost patching，前者 Alpamayo 缺多相机域配对适配器，
后者同样受单次前向 2.7 s 的成本约束 ⇒ **未测**。

---

## 自我更正记录

1. **D2cV 对多帧候选不成立（§CE/A34）**：这是本轮扩池才暴露出来的前提失效——
   前三轮的候选恰好全是单帧模型。Alpamayo 的 G 轴判据因此从"减 D2cV 地板"改为
   "减置换/随机地板"，并在矩阵里单独标注。**这不是为了让 Alpamayo 好看**：
   按旧判据它是"不可估"，按新判据是"显著"，两个数字都在表里，判据切换的理由是物理的（多帧可观测速度），
   与结果无关。
2. **负例缓存范围一度过大**：首次按 `--types D2b D2c` 缓存未套用 matched 清单，
   候选事件达 1562 个（≈ 70 min）。改为按 `matched_{D2b,D2c,D2bV,D2cV}` 的并集（941 个）缓存，
   覆盖预筛后 570 个。此为效率修正，不影响口径。
3. **F②/C 轴标"未测"而非"不适用"**：按工单 §6，只有"该操作化在该架构上不适用"才标 not applicable；
   Alpamayo 的情况是**预算不足**，性质不同，故如实标"未测"。
