# 补充完成标志：Alpamayo-R1 / AutoVLA 的 C-hazard 补测

> 承接 [`candidate_expansion_DONE.md`](candidate_expansion_DONE.md) §4「未完成项」里的
> **「Alpamayo-R1 / AutoVLA 的 C-hazard｜未测｜预算」** 一行。本轮把这两格补上。
> 完成日期：2026-09-01。修正案：[`amendments.md`](amendments.md) §CE/A39。
> **后续**：§5 里那条可证伪预测已于同日检验完毕，见 [`c_hazard_prediction_check.md`](c_hazard_prediction_check.md) 与 §CE/A40。
> 本轮**不引入新候选、不改变任何既有轴的口径**。

---

## 1. 两格的最终读数与判定

口径与既有三个候选（DiffusionDrive / LTF / DiffusionDriveV2）逐条一致：
配对 = G1 clean↔ghost（配对真实输入互换，禁用噪声破坏）；指标 = 连续量 $v_{plan}$（禁用二值化）；
退化样本按 $|v_{clean} - v_{ghost}|$ 取前 12；patch 范围 = 整条 prompt 的残差流
（与 DD 系换全部 320 个融合 token 同构，§CE/A32）；patch 打在 **prefill** 前向上，
decode step 不动，经 KV cache 影响后续全部生成步。

| | **Alpamayo-R1** | **AutoVLA** |
| --- | --- | --- |
| 事件 / 场景 | 12 / 10 | 12 / 11 |
| 可读层 | 36 | 36 |
| patch-ALL（充分割集自检） | 均值 +0.965，中位数 **+1.001** ⇒ **通过** | +1.000 ⇒ **通过** |
| 剖面形状 | **阶跃**：L0 0.584、L1 0.814、L2–L16 ≈ 0.88 ~ 1.01、L18 起 ≈ 0.55、末段 ≈ 0.42 ~ 0.49 | **阶跃**：L0–L20 ≈ 1.0，L21 起 0.84 → 0.77 → 0.56 → 0.37 → 0.10 → 0.00 |
| Spearman(层号, recovery) | **−0.873** | **−0.859** |
| $C_m$（名义值，不作判据） | 0.091 [0.080, 0.106]（基线 0.056） | 0.079 [0.069, 0.094]（基线 0.056） |
| **承诺层**（mean recovery ≥ 0.9 的最深层） | **L16 / 36（深度 0.47）** | **L20 / 36（深度 0.58）** |
| 责任层众数 / 归一化熵 | L1 / 0.520 | L0 / 0.158 |
| 采样噪声地板 | sd 0.430 = 中位 gap 的 **0.330 倍**（未触发 0.5 门限） | 不适用（$top_k$=1 贪心解码，等价确定性） |
| **判定** | **不可估（公式前提不成立：剖面呈阶跃）** | **不可估（同上）** |

两格都**不是**"没测出来"：patch-ALL 充分割集自检均以 ≈ +1.00 通过，仪器工作正常。
判不可估的依据是 §4.4.4 的既有适用性判据（$\rho < -0.7$ 判级联 ⇒ top-2 占比公式前提不成立），
与 SimLingo 的 C-domain 级联同一处理，不是为这两个候选新设的标准。

## 2. 补测带来的实质结论（比这两格本身更重要）

**同一个操作化在两个架构族上退化到两端**：

| 架构族 | 成员 | 剖面 | 承诺层 | $C_m$ 公式 |
| --- | --- | --- | --- | --- |
| TransFuser 系 | DiffusionDrive / LTF / DiffusionDriveV2 | **递增**（内部峰 @L6，Spearman +0.881 / +0.929 / +0.810） | **不存在**（任何单层 recovery < 0.9） | 适用 |
| VLA 栈 | Alpamayo-R1 / AutoVLA | **阶跃**（Spearman −0.873 / −0.859） | L16 / L20（36 层） | **不适用** |

原因是结构性的：patch 第 $L$ 层输出后，**比 $L$ 更深的每一层都由 clean 侧重算**。
纯自回归 transformer 栈里残差流是唯一通路 ⇒ patch 任意早期层即完全恢复 ⇒ 剖面必然阶跃。
TransFuser 系逃过这一点，**只因为每个融合块都从 CNN 分支重新注入一份未被 patch 的图像特征**。

由此得到两条方向相反的结论，都已写进正文：

1. **向前**：$C_m$ 在 **TransFuser 系与 VLA 栈之间不可比**——
   与 §4.4.2 的「F② 不可跨编码器族比较」是同一性质的结论：**操作化的前提是架构相关的**。
   若照字面读 $C_m$，会把两个 VLA 的 0.079 / 0.091（36 层基线 0.056）
   与三个 TransFuser 成员的 0.789 ~ 0.845（8 层基线 0.250）排在一起，
   得到一个完全由层数与架构决定、与"失效是否集中"无关的排名。
2. **向后（对既有结论的修订）**：DD 与 LTF 在 **L6** 的一致定位，
   此前被当作一次收敛效度检验；现在必须补一句——
   **这个内部峰是 TransFuser 逐级重注入设计的属性，两者一致部分是因为共用同一种融合设计**。
   该收敛效度只在编码器族**内部**成立。

**换 patch 范围不能绕开**：另跑一次 `--tokens image`（只换图像 token 段），
AutoVLA 的剖面仍在 L0–L21 饱和于 1.0（Spearman −0.876）。故这不是范围选得不好，
是该操作化在这类栈上无分辨力。

## 3. 前后矛盾的表述已同步改掉

| 位置 | 原表述 | 现表述 |
| --- | --- | --- |
| §4.2.3 | DiffusionDriveV2 的 $C_m$ 名义值 0.845「是**全表最高**的」 | 「是**三个 TransFuser 系成员里**最高的（三者共用同一条 8 层弥散基线 0.250，故该比较合法）」 |
| §4.2.3 | DD/LTF 的 L6 一致是「唯一一处两个候选给出一致诊断的地方」 | 保留，但补一句「**它们一致，部分是因为共用同一种融合设计**」 |
| Table 1(b) | Alpamayo / AutoVLA 的 C 列为 `n.m.` | 填入剖面形状 + 承诺层，并加注⁵说明 $C_m$ 为何 n/a |
| §4.3 | 「14 格没有可用数字 / 11 格不可估 / 共 25 格」 | 「12 格 / 10 格 / 共 22 格」，并把 VLA 栈上的 $C_m$ 归入「操作化前提不成立」这一类 |
| §4.4.4 | 适用性判据只举 SimLingo 一例 | 升级为「某类架构必然是级联」，并给出五个候选按架构族的完整分离 |
| §4.5 限制 4 | 「C 的 top-2 占比公式在级联剖面上前提不成立」 | 「在**整个 VLA 栈**上结构性不成立」，并新增「向后影响」段落 |
| 修正案计数 | 38 条 | **39 条** |

## 4. 产出物路径

| 路径 | 说明 |
| --- | --- |
| `results/c_axis_hazard_alpa.json` | Alpamayo-R1 的 C-hazard 全部读数，含 `commitment_layer` 与 `sampling_noise_floor` |
| `results/c_axis_hazard_autovla.json` | AutoVLA 的 C-hazard 全部读数，含 `commitment_layer` |
| `results/alpamayo_g1_adapter/alpa_patch.py` | **新增**：Alpamayo 的 patching 接口（`PatchCapture` + `AlpaPatchRunner`），与 AutoVLA 侧逐条同构 |
| `results/autovla_g1_adapter/autovla_adapter.py`（改） | `PooledCapture` 增加 `capture_full` 与 `set_patch(d, tokens)`；不开启时行为与原来一字不变 |
| `scripts/c_axis_hazard_patch.py`（改） | `--model` 增加 `alpa` / `autovla`；新增 `--tokens {all,image}`、`--seed-floor N`；新增 `commitment_layer` 读数（已回填到全部五个候选的 json） |
| `results/axis_alpamayo_report_{zh,en}.md`（改） | 新增「Results（补测）：C-hazard」节 |
| `results/axis_autovla_report_{zh,en}.md`（改） | 新增「Results（补测）：C-hazard」节 |
| `results/paper_experiments_section_{zh,en}.md`（改） | Table 1(b) 补格 + §4.2.3 / §4.3 / §4.4.4 / §4.5 同步 |
| `results/amendments.md`（改） | 新增 §CE/A39 |

## 5. 仍未完成项（如实记录）

| 项 | 状态 | 理由 |
| --- | --- | --- |
| Alpamayo-R1 / AutoVLA 的 F② | 未测 | 预算。**不是"不可测"的结论**——两者注入位点（LLM 残差流）与 SimLingo 同构，先验上应当可测。 |
| ~~SimLingo 的 C-hazard~~ | **已补测（2026-09-01），预测证实** | 见 `c_hazard_prediction_check.md` 与 §CE/A40：Spearman −0.997，级联，承诺层 L3/24；六个候选按架构族完全分离（3/3 vs 3/3）。预测中"早段必然饱和于 1.0"那半句被证伪并已更正。 |
| LTF 的 C-domain | 未测 | 预算。 |
| 五个候选的 C-hazard 事件数 | 均为 12 | 承诺层的定位精度受此限制；Alpamayo 另受采样噪声限制（见 §1）。 |
