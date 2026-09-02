# 完成标志：F-3 遮挡修复推广到前车急刹 / NAVSIM

> 工单：[`../docs/f3_final_cleanup_workorder.md`](../../docs/f3_final_cleanup_workorder.md) 第一部分。
> 完成日期：2026-09-02。修正案：[`amendments.md`](amendments.md) §FC/A60–A61（累计 61 条）。
> 详细报告：`results/f3_corpus_extension_report_{zh,en}.md`。

---

## 1. 直接结论

| 问题 | 答案 |
| --- | --- |
| 四格都重跑了吗？ | **是**。DDv2 × {前车急刹, NAVSIM} 补 lidar（m=0.25 主 / m=0.0 敏感性）；Alpamayo-R1 / AutoVLA × 前车急刹 补逐帧遮挡 |
| 判定变了吗？ | **四格全部未变**，都仍是"不可估：基线响应本身与 0 不可区分" |
| 白跑了吗？ | **没有。** 泄漏量级跨语料差 **81 倍**（G1 中位 1 点 → NAVSIM 中位 81 点）；NAVSIM 上 $R$ 的 CI 半宽由 0.48 收到 **0.19**；Alpamayo 的对照臂**修复后才暴露**出显著响应 |
| 有没有发现新问题？ | **两处**：NAVSIM 3D 框朝向约定多减了一次 ego 航向（§FC/A60）；F-3 的 clean 臂在 76% 的事件里实体已可见（§FC/A61） |

## 2. 数字一览

**DDv2 补 lidar（判定三个语料全部未变）**

| 语料 | $b_{ghost}$ | $R$（仅 RGB → +lidar） | 每事件删点中位（occ/ctrl） |
| --- | --- | --- | --- |
| G1 | −0.180 [−0.349, +0.003]（逐位未变） | — | 1 / 0（42% 事件一个点没删） |
| 前车急刹 | −0.105 [−0.475, +0.210]（**逐位未变**） | +0.686 → +1.088 | 8 / 6 |
| NAVSIM | +0.075 → +0.062（偏移 0.0126，已定位并记录） | +0.311 [−0.087, +0.876] → **+0.252 [+0.082, +0.464]** | **81 / 14** |

**两个 VLA 补逐帧遮挡（前车急刹）**

| 候选 | 漏遮率 | $\Delta b_{ghost}$（噪声地板） | $\Delta b_{occ}$（修复效应） | 比值 |
| --- | --- | --- | --- | --- |
| Alpamayo-R1 (n=51) | **204/204 = 100%** | 0/51 非零 | 51/51 非零，均值 0.258 | ∞ |
| AutoVLA (n=88 共有) | **359/360 = 99.7%** | 3/88 非零，均值 0.0043 | 17/88 非零，均值 0.0999 | **23×** |

## 3. 三条必须并列写出的不利读数

1. **Alpamayo-R1 修复后对照臂显著**：$R_{ctrl}$ = +0.394 [+0.085, +0.778]，应接近 0。
   ⇒ 该候选对"画面多了几块灰斑"本身有反应，其 $R$ 即便过门也不可解读。**修复才暴露出来的问题。**
2. **AutoVLA 的 $R$ = +4.42 [+0.067, +13.33] 不是可用读数**：过门槛事件仅 34、分母贴近 0。
   按分母守卫纪律只读到"不可估"，**不得**读成"必要性极高"。
3. **AutoVLA 新旧事件数 90 vs 88 未完全解释**，逐事件对比只在 88 个共有事件上做。

## 4. 产出物

| 路径 | 说明 |
| --- | --- |
| `results/f3_occlusion_leadbrake_ddv2_lidar{,_m0}.json` | 前车急刹 DDv2 + lidar 遮挡（主 / 敏感性） |
| `results/f3_occlusion_navsim_ddv2_lidar{,_m0}.json` | NAVSIM DDv2 + lidar 遮挡（主 / 敏感性） |
| `results/f3_occlusion_leadbrake_{alpa,autovla}_mf.json` | 前车急刹两个 VLA 的逐帧遮挡修复版 |
| `results/ns_yaw_audit{,_vehicle}.json` | NAVSIM 朝向约定判定 + 对已发布 2D 框的影响量化 |
| `results/f3_within_frame_effect.json` | 25 格 F-3 的同帧擦除效应重新分析（§FC/A61） |
| `scripts/ns_yaw_audit.py`、`scripts/f3_within_frame_effect.py` | **新增** |
| `scripts/f3_window_boxes.py` | **改**：加 NAVSIM 后端 + 朝向约定分支（nuScenes 路径逐位不变） |
| `scripts/f3_occlusion_necessity.py` | **改**：`--navsim-split`，把 corpus 透传给 WindowBoxes |

**旧结果文件一个没删**：新结果按既有习惯另存 `_lidar` / `_mf` 后缀。

## 5. 未完成项

| 项 | 状态 | 理由 |
| --- | --- | --- |
| 两个 VLA × NAVSIM | **n/a** | 语料侧接口缺失（依赖 nuScenes devkit / `sd_token`），工单亦明确跳过 |
| 前车急刹 × DDv2 的"反向擦除效应" | 未追 | 需单独机制实验，超出本工单 |
| 用 $d_{occ}$ 重构 F-3 的门 | 未做 | 属方法改动而非换语料重跑；须先想清分母 |
| NAVSIM 车辆类读数按更正朝向重算 | 未做 | 本轮 F-3 的 NAVSIM 正例全是 VRU（影响小）；车辆类在别的实验线，需单独排期 |
