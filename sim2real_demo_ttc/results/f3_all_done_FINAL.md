# 完成标记：从"多帧遮挡漏洞"到"论文草稿更新完毕"整条链路

> **醒来后先读这一份。** 完成日期：2026-09-02。
> 工单：[`../docs/f3_final_cleanup_workorder.md`](../../docs/f3_final_cleanup_workorder.md)（两部分）。
> 承接：`f3_multiframe_occlusion_fix_workorder.md`（§FM/A56–A59）。
> 提交：`f94bdaa`（第一部分）、`b48d935`（论文 Table 4 脚注）、`b680201`（第二部分 + 论文三处）。
> 修正案累计 **62** 条（本轮新增 §FC/A60–A62）。

---

## 0. 一句话结论

**两轮修复（逐帧遮挡、lidar 遮挡）与一轮降噪（多帧平均）全部完成，
十格 F-3 判定一格没变；但过程中查出三处此前未知的方法学问题，其中两处影响面超出本工单，
已如实登记并折进论文——论文里没有任何一个判定格因此改动。**

---

## 1. 整条链路做了什么

| 轮次 | 做了什么 | 判定变化 | 提交 |
| --- | --- | --- | --- |
| 上一轮（§FM/A56–A59） | 修 VLA"只遮窗口末帧"漏洞；补 DDv2 的 lidar 通道遮挡。**只在 G1 验证** | G1 三格未变 | `aacaba1` / `bf5aa3b` |
| **本轮第一部分** | 把上述两处修复**推广到前车急刹 / NAVSIM** | **四格未变** | `f94bdaa` |
| **本轮第二部分** | F-3 四臂改成 **9.9 帧窗口平均**（G1 四个单帧候选） | **四格未变** | `b680201` |
| **论文折入** | Table 4 脚注 2 改写；Table 3 脚注 1、Validity Checks 第 6 行、DiffusionDrive 段各加确证性表述 | **未改任何判定格** | `b48d935` / `b680201` |

---

## 2. 最终判定表

### 2.1 F-3（遮挡必要性检验）——本轮涉及的十格

| 候选 | 语料 | 最终判定 | 本轮是否重跑 | 备注 |
| --- | --- | --- | --- | --- |
| SimLingo | G1 | 不可估（$R$ 的 CI 跨 0.5） | 是（多帧平均） | 唯一通过同帧特异性检验的候选 |
| LTF | G1 | **FAIL**（盲目泛化签名） | 是（多帧平均） | 多帧下 $R$ +0.113 → +0.058，FAIL 更彻底 |
| DiffusionDrive | G1 | 不可估（无基线响应） | 是（多帧平均） | 同帧擦除显著但**不特异** |
| DiffusionDriveV2 | G1 | 不可估（无基线响应） | 是（多帧平均 + lidar） | — |
| Alpamayo-R1 | G1 | 不可估（无基线响应） | 上一轮已修 | — |
| AutoVLA | G1 | 不可估（无基线响应） | 上一轮已修 | — |
| DiffusionDriveV2 | 前车急刹 | 不可估（无基线响应） | **是（+lidar）** | $b_{ghost}$ 逐位未变 |
| DiffusionDriveV2 | NAVSIM | 不可估（无基线响应） | **是（+lidar）** | $R$ 的 CI 半宽 0.48 → **0.19** |
| Alpamayo-R1 | 前车急刹 | 不可估（无基线响应） | **是（逐帧遮挡）** | 修复后 $R_{ctrl}$ 显著 ⇒ $R$ 不可解读 |
| AutoVLA | 前车急刹 | 不可估（无基线响应） | **是（逐帧遮挡）** | $R$ = +4.42 分母贴近 0，不可用 |

**十格判定，一格未变。**（LTF × NAVSIM 的 FAIL、LTF × 前车急刹的不可估未在本轮重跑范围内，
见 §5。）

### 2.2 论文里的最终数字（`paper_v2.2_draft`）

* Table 3（`tab:main-matrix-v1`）：**未改任何格**，脚注 1 加了两句限定与确证。
* Table 4（`tab:generality`）：**未改任何格**（F-3 那格仍是 `untestable ×6`），脚注 2 全文改写。
* Validity Checks 第 6 行：标题扩为 "occlusion completeness **and temporal sampling**"，末尾加窗口平均结论。
* `main.pdf` 已重编译：**12 页**、`0` LaTeX Error、`0` undefined control sequence；
  3 处 overfull hbox 均为基线自带。

---

## 3. 本轮查出的三处方法学问题（都不是工单要求查的，都影响面超出本轮）

### 3.1 lidar 泄漏的量级跨语料差 81 倍（§第一部分 §3.1）

同一套删点规则、同一目标类别（VRU），每事件删点中位：
**G1 = 1 个点**（42% 事件一个点都没删）→ **前车急刹 = 8** → **NAVSIM = 81**。
上一轮"泄漏暴露面很小"的结论**不能外推到别的数据源**（nuScenes 32 线单雷达 vs NAVSIM 稠密点云）。
堵上后 NAVSIM 的 $R$ 的 CI 半宽 0.48 → 0.19，$R_{ctrl}$ −0.111 → +0.003。

### 3.2 NAVSIM 3D 框朝向约定多减了一次 ego 航向（§FC/A60）

`gt_boxes[:,6]` 的 yaw 本就在 ego 系，而 `frame_bbox` 又减了一次 ψ。
**实测判定**（276 条车辆轨迹）：std 0.308 → 0.106，77.2% 一致。
对**已发布**数字的影响两组都测：
* A 类（全 VRU）IoU 中位 **0.899** ⇒ 影响小，**已发布 NAVSIM F-3 数字不撤回**；
* **车辆类 IoU 中位 0.779、中心位移中位 9.9 px** ⇒ 任何以车辆为目标的 NAVSIM 读数会被实质影响。

### 3.3 F-3 的 clean 臂并非"危险不在场"（§FC/A61，影响面最大）

* **G1：219/288 = 76.0%** 的事件在 clean 帧上实体已可见，成像面积比中位 **1.67**；
* **前车急刹：103/103 = 100.0%**，面积比中位 **1.00**（该语料按定义就是"已被跟踪的前车开始急刹"）。

⇒ $b_{ghost}$ 量的是"危险**走近 / 状态改变**"而非"危险**出现**"，而三态判定的门恰是它。

补了一条不依赖 clean 臂的同帧擦除效应 $d_{occ}$（只重新分析已落盘数据，**未重跑任何前向**）：
25 格里 **6 格** $b_{ghost}$ 不显著而 $d_{occ}$ 显著，**4 格**过 ctrl 特异性检验，
**只有 2 格**方向也符合预期（前车急刹 × SimLingo / LTF）。
**未改 F-3 主判定门**——$d_{occ}$ 不提供 $R$ 需要的分母。

---

## 4. 三条纪律在本轮各拦下了一次错误

| 纪律 | 本轮拦下了什么 |
| --- | --- |
| **ctrl 臂是必需对照** | DD 的 $d_{occ}$ 在多帧下变显著，若无 ctrl 臂会被读成"揭示了被噪声掩盖的危险响应"；配对检验显示它对别处等面积灰斑反应一样大 ⇒ 记为非特异扰动敏感性 |
| **分母守卫** | AutoVLA 前车急刹 $R$ = +4.42 [+0.067, +13.33]，过门槛事件仅 34、分母贴近 0 ⇒ 只读到"不可估"，未读成"必要性极高" |
| **"看不看得见"必须逐帧投影核实**（§FM/A57） | 同一条教训本轮在 **clean 臂**上再次命中（§3.3）；前车急刹上 AutoVLA 的漏遮率 99.7%，比 G1 的 86% 更高 |

---

## 5. 未完成项（按优先级）

| 项 | 为什么没做 | 建议 |
| --- | --- | --- |
| **用 $d_{occ}$ 重构 F-3 的门** | 属方法改动而非"换语料重跑"；须先想清分母该用什么 | 优先级最高——它决定"untestable"这类格子将来还算不算 untestable |
| **NAVSIM 车辆类读数按更正朝向重算** | 本轮 F-3 的 NAVSIM 正例全是 VRU（影响小）；车辆类读数在别的实验线 | 需单独排期；`scripts/ns_yaw_audit.py` 已给出影响量化 |
| **"实体真正不在场"子集的预注册重测** | 本轮该分组是事后的、n = 67、方向不一致（DDv2 变显著但 SimLingo 反向） | 作为下一轮**预注册**假设 |
| **前车急刹 × DDv2 的"反向擦除效应"** | 擦掉前车反而让规划**减速** 0.30 m/s，且特异；需单独机制实验 | 值得追——可能是"前方有车 = 可通行"的捷径签名 |
| 多帧平均推广到其余语料 / 两个 VLA | 本轮判定零变化、降噪幅度小且方向不一致 ⇒ 按工单不推广 | 除非先解决 §3.3 的门的问题 |
| 两个 VLA × NAVSIM 的 F-3 | **n/a**：语料侧接口缺失（依赖 nuScenes devkit / `sd_token`） | 需要 NAVSIM 侧适配器 |
| 窗口长度敏感性（L = 3 / 5 / 10） | 瓶颈已定位在事件间方差而非窗内噪声，改 L 不触及瓶颈 | 低优先级 |

---

## 6. 产出物清单

| 类别 | 路径 |
| --- | --- |
| 报告（中英各一份） | `results/f3_corpus_extension_report_{zh,en}.md`、`results/f3_temporal_averaging_report_{zh,en}.md` |
| DONE 标记 | `results/f3_corpus_extension_DONE.md`、本文件 |
| 第一部分读数 | `results/f3_occlusion_{leadbrake,navsim}_ddv2_lidar{,_m0}.json`、`results/f3_occlusion_leadbrake_{alpa,autovla}_mf.json` |
| 第二部分读数 | `results/f3_tavg_{dd,ltf,ddv2,simlingo}.json`、`results/f3_tavg_summary.json` |
| 顺带查出的问题 | `results/ns_yaw_audit{,_vehicle}.json`、`results/f3_clean_arm_audit_{g1,leadbrake}.json`、`results/f3_within_frame_effect.json` |
| 新增脚本 | `scripts/f3_temporal_average.py`、`scripts/f3_tavg_summary.py`、`scripts/ns_yaw_audit.py`、`scripts/f3_within_frame_effect.py`、`scripts/f3_clean_arm_audit.py` |
| 改动脚本 | `scripts/f3_window_boxes.py`（NAVSIM 后端 + 朝向分支，nuScenes 路径逐位不变）、`scripts/f3_occlusion_necessity.py`（`--navsim-split`） |
| 论文 | `paper_v2.2_draft/`（`CHANGELOG.md` 记了三条改动的逐个数字出处；`author-kit/main.pdf` 已重编译） |
| 修正案 | `results/amendments.md` §FC/A60–A62 |

**旧结果文件一个没删**：新结果一律按既有习惯另存 `_lidar` / `_mf` / `_tavg` 后缀。
