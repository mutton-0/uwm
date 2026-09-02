# paper_v2.2_draft 变更说明

基线：`/tmp/author-kit-v2.2-base.tar.gz`，对应 git commit `13117e0`（v2.1 快照，"只放最终结果"重写版）。
解压位置：`paper_v2.2_draft/author-kit/`（已清掉 tar 里的 macOS `._*` AppleDouble 文件）。

---

## 已完成

### 1. `sec/4_experiments.tex` — Validity Checks 表新增第 6 行「F-3 遮挡完整性」

**改了两处**：

| 位置 | 改动 |
|---|---|
| `\subsection{Validity Checks}` 引导段 | "summarizes **five** checks … All **five** numbers" → **six** / **six** |
| `\label{tab:validity}` 表格 | 在末行（full-population null enumeration）之后新增一行 |

新增行的三列内容与依据：

| 列 | 内容 | 数字来源 |
|---|---|---|
| Check | F-3 occlusion completeness (per-frame projection across the input window; non-camera sensor channels) | — |
| What it rules out | An F-3 necessity reading produced by an occlusion that leaves the entity present in the model's input | — |
| Verdict | 实体投影进 **4.00/4**（Alpamayo-R1）与 **3.44/4**（AutoVLA）窗口帧；点云候选的实体框内**中位 1 个点**（实体**中位纵距 27 m**）；逐帧遮挡 + 删除 3D 框内点云、**对照臂对称处理**后，**三个受影响的判定全部不变**；$b_{ghost}$ 按构造不受影响，lidar 通道的效应为 $\Delta b_{occ}=+0.005\ [-0.004,+0.015]$ | 见下表 |

**逐个数字的出处**：

| 数字 | 出处 |
|---|---|
| Alpamayo 4.00/4、AutoVLA 3.44/4 | `results/f3_multiframe_fix_report_zh.md` §1「逐帧投影统计」表（`f3_occlusion_{alpa,autovla}_mf.json` 的 `window_projection_stats`：396/396 与 409/476） |
| 中位 1 个点、中位纵距 27 m | 同报告 §A.4 表（`f3_occlusion_ddv2_lidar.json` 的 `lidar_removal.n_points_removed_occ.median` = 1；实体 `d_long_at_emergence` 中位 27.1 m） |
| 三个判定全部不变 | 同报告 §A.2 表与 §2 表：Alpamayo / AutoVLA / DDv2 修复前后都是 "不可估：无基线响应" |
| $b_{ghost}$ 按构造不受影响 | 同报告 §2.2 / §A.3：Alpamayo Δ$b_{ghost}$ 0/99 非零；DDv2 0/282 非零 |
| $\Delta b_{occ}=+0.005\ [-0.004,+0.015]$ | 同报告 §A.3 表（m = 0.25 主口径） |

**风格遵循**：写成"我们做了这项自查，结果是…"的验证性陈述，与既有五行同格式（三列、平实、给最终数字）。
**没有**写成"我们发现了 bug / 修复了 bug"的过程叙事，**没有**引入 amendments 编号或数量，
**没有**加进后面那段"Two of these checks produced a correction…"——因为这一项**没有**改变任何论文结论，
把它写进"改变了论断"的段落会是过度声称。

**编译验证**：本机装了 `tectonic 0.15.0`（静态二进制，装在 `~/bin/tectonic`，无需 root）。
基线与改后均编译通过，**12 页**，改后 `main.pdf` 已就地更新。
日志里 3 处 overfull hbox 全部位于第 91–121 行（方法节公式区），**基线本来就有，非本次引入**；
字体 warning（`TU/ptm` 等）同样是基线自带。

### 2. `sec/4_experiments.tex` — Table 4（`tab:generality`）F-3 格的脚注 2 + 对应正文句

**依据**：`results/f3_corpus_extension_report_{zh,en}.md` §4 与 `results/amendments.md` §FC/A61。

**改了两处**（同一件事的表内与正文两处措辞，保持一致）：

| 位置 | 改动 |
|---|---|
| `\label{tab:generality}` 下的脚注 2 | 原文只有一句「No baseline response … for either candidate with an F-3 signature elsewhere (SimLingo, LTF)」；改为先陈述主判据结论，再给 clean 臂的实测比例与原因，再给不依赖 clean 臂的同帧对照数字，末句限定 "untestable" 的所指 |
| 正文 "F-3's most portable finding is negative." 段末句 | 「F-3 could not be evaluated … not a failure of the method.」之后补两句：该语料上这条限制**部分是定义性的**（前车全程被跟踪），以及同帧对照显示两个候选确实对遮挡有响应 |

**表格单元格 `untestable ×6` 未动** —— 主判据（必要性比 $R$，门槛是 $b_{ghost}$ 显著）
在该语料上确实对六候选都判不可估，这一格如实。改的是脚注对这个词的限定。

**逐个数字的出处**：

| 数字 | 出处 |
|---|---|
| clean 帧上实体已可见 **103/103 = 100.0%**（前车急刹语料） | `results/f3_clean_arm_audit_leadbrake.json`（`frac_entity_visible_in_clean_frame`）；脚本 `scripts/f3_clean_arm_audit.py` |
| ghost/clean 成像面积比中位 **1.00** | 同上（`both_visible.area_ratio_ghost_over_clean.median`；p25 0.77 / p75 1.22，纵距中位 31.4 → 31.8 m） |
| SimLingo $d_{occ}=-0.510\,[-0.736,-0.307]$，对照 $-0.032\,[-0.117,+0.048]$ | `results/f3_within_frame_effect.json`（`f3_occlusion_leadbrake_simlingo.json` 那行） |
| LTF $d_{occ}=-0.028\,[-0.047,-0.011]$，对照 $+0.007\,[-0.001,+0.016]$ | 同上（`f3_occlusion_leadbrake_ltf.json` 那行） |

**为什么这么写**：
* **不写成"我们之前判断错了"**。主判据在该语料上的结论没有变，变的是"untestable"这个词
  需要一句限定——它指的是**必要性比无从计算**，不是"该候选对遮挡毫无反应"。
* **不把 $d_{occ}$ 提为主判据**。F-3 的 PASS 门槛是「$R$ 的 CI 下界 > 0.5」，即擦除要能移除
  **过半**原始响应，这需要一个有意义的分母；$d_{occ}$ 只回答"擦除有没有效应"。
  脚注里明写 "It supplies no denominator for $R$ and does not change the PASS threshold"。
* **给出 clean 臂不成立的原因而不只是比例**：该语料按定义就是"已被跟踪的前车开始急刹"，
  实体全程在场、成像面积几乎不变（比值 1.00）⇒ $b_{ghost}$ 对比的是**制动起始**而非**实体有无**。
  这是场景定义的直接后果，写清楚比只报一个百分比更有解释力。

**编译验证**：`~/bin/tectonic 0.15.0` 重编译通过，**12 页**（与基线一致），
`0` 个 LaTeX Error、`0` 个 undefined control sequence；
3 处 overfull hbox 全部位于 `1_intro:96` / `3_method:91` / `4_experiments:108–121`，
**均为基线自带、非本次引入**（本次改动落在 `4_experiments` 的 192 行与 225–235 行）。
`author-kit/main.pdf` 已就地更新。

---

## 未做（依赖的报告尚未产出）

工单的第 2、3 项依赖两份**目前不存在**的报告，按"不要提前用还没出的中间结果"的要求**未动**：

| 待办 | 依赖 | 现状 |
|---|---|---|
| 2. 多帧平均降噪结果 → 更新 `tab:main-matrix-v1`（Table 3）/ `tab:generality`（Table 4）及相关段落 | `results/f3_temporal_averaging_report_{zh,en}.md` | **进行中**（DD / LTF / DDv2 已出，SimLingo 在跑） |
| ~~3. 前车急刹 / NAVSIM 语料推广后的复现数字 → `tab:generality`~~ | `results/f3_corpus_extension_report_{zh,en}.md` | **已产出，见上文「已完成 2」** |

第 3 项已于本次完成（见「已完成 2」）。第 2 项待 `f3_temporal_averaging_report` 四个候选齐全后续做——
若显示判定基本不变，按要求写成"验证了原判定不是噪声 artifact"，不夸大；
若某候选判定改变，则需连带检查 Table 3 脚注 1 的适用范围、"The most informative cell is DiffusionDrive" 那段论证、
以及 Validity Checks 新增行里"三个判定全部不变"这句。

**一处需要留意的现有表述**：Table 3 里 Alpamayo-R1 / AutoVLA / DiffusionDriveV2 三格现为
"no baseline response"。若多帧平均降噪把其中任何一格改成"有小的真实响应"，
除了改 Table 3 的格子，还需要同步改：
- Table 3 脚注 1（"untestable, not a failed test"）的适用范围；
- 正文 "\textbf{The most informative cell is DiffusionDrive}" 那段（它的论证依赖 DDv2 侧
  "F-3 baseline response indistinguishable from zero"）；
- 本次新增的 Validity Checks 行里"三个受影响的判定全部不变"这句（届时需按新判定重述）。

---

## 文件清单

| 文件 | 状态 |
|---|---|
| `author-kit/sec/4_experiments.tex` | **已改**（引导段计数 + 新增表行） |
| `author-kit/main.pdf` | **已重编译**（tectonic 0.15.0，12 页） |
| 其余 `.tex` / `.bib` / `.sty` | 未动 |
