# SimLingo × nuScenes TTC 突变集 — Tier-S 闭环报告

> **Tier-S 冒烟读数，事件量为几十级，统计功效极低，不作结论。** Tier-S 的验收目标是**管线闭环 + 脚本参数化**，不是统计结论。
> 手册：`docs/remote_demo_simlingo_guide.md`。所有脚本以 `configs/tier_s.yaml` 为唯一参数入口，换 Tier-M/L 数据只需改 config 中的 `paths.nuscenes_*`。

## 1. 环境与版本

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition（本次单卡，与他人任务共享） |
| conda env | `/data/ruolin/envs/simlingo`（python 3.10.20） |
| torch / transformers / timm / numpy | 2.8.0+cu128 4.46.3 0.9.16 1.26.4 |
| SimLingo repo | `/data/ruolin/simlingo` @ `743b243` |
| 权重 | `RenzKa/simlingo` epoch=013 `pytorch_model.pt`，sha1(head16)=`3ff2eadbb919218d` |
| VLM 底座 | InternVL2-1B（24 层 decoder，hidden 896） |
| 数据 | nuScenes `v1.0-mini` @ `/data/dataset/nuscenes/v1.0-mini` |
| 预处理配置哈希 | `eeb470cbc2ad`（resize_keep_aspect_then_crop） |
| 本仓库 commit | `911ad01` |

**与官方 repo 的偏离（必须记录）**

1. repo `environment.yaml` pin 的是 `torch==2.2.0` + `flash-attn`，**在 Blackwell(sm_120) 上不可用**；改用 torch 2.8.0+cu128、transformers 4.46.3（与 repo pin 同版本）、无 flash-attn（InternVL 自动回退 eager attention）。
2. 推理路径用 `predict_language=True`（= `team_code/agent_simlingo.py` 的部署路径：先贪心生成语言，再把 `[prompt+生成文本 | driving queries]` 整条序列过一次 LM 得到 waypoints）。repo 的 `predict_language=False` 分支在 `split_outputs_by_adaptor` 处有 bug（官方 eval 未走过该分支）。
3. 模型输入缺口的统一默认值（手册 §4 要求声明）：
   - prompt：`Current speed: 5.0 m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.`（`Target waypoint:` 模式，正前方直行目标点 [[20.0, 0.0], [40.0, 0.0]]，clean/ghost 之间**完全一致**，符合手册 §10.4）；
   - ego speed：由 nuScenes ego_pose 差分得到（非 CAN bus；CAN 数据在库但本轮未接入，记为偏离）。
4. nuScenes(1600×900, hfov≈64°) → SimLingo(CARLA 1024×512, fov 110°) 的几何对齐：`resize_keep_aspect_then_crop` = 等比缩放到宽 1024 后裁上部 359 行（359 = CARLA 512 经 SimLingo 自身 4.8/16 底裁后的高度），再走 InternVL `dynamic_preprocess`（2 patch）。**FOV 差异按手册 §10.2 不做纠正**（它本身是待测的域差），但必须在结论中声明。

## 2. G0–G4 门控

| 门 | 标准 | 实测 | 结果 |
|---|---|---|---|
| G0 冒烟 | 输出合理 + 全层 hidden 可抓 + 双跑逐位一致 | waypoints 10×2 无 NaN；24 层 × 896 维，序列长 577；两次运行逐位一致 | ✅ PASS |
| G1 挖掘 | 事件量（Tier-S 20–50）+ 抽检语义正确率 ≥80% | 56 事件 {'A': 16, 'B': 5, 'C': 6, 'D': 29}；抽检 12/30 张，正确率 91.7% | ✅ PASS |
| G2 缓存 | 完整率 ≥99% | 56/56，完整率 100.0%，耗时 33s | ✅ PASS |
| G3 指标 | 全链路可算 | 两种池化口径（vision_mean / last_token）均跑通 | ✅ PASS |
| G4 验收 | V1–V5 逐条判定 | V1=✅ PASS V2=❌ FAIL V3=❌ FAIL V4=❌ FAIL V5=N/A | 见 §5 |

**阈值回调记录（手册 §5.3 要求每次放宽记录在案）**

| # | 项 | 手册值 | 本轮值 | 依据 |
|---|---|---|---|---|
| 1 | A 类 TTC 上限 | 3.0s | 5.0s | mini 上 TTC<3s 的 VRU 入走廊事件仅 1 例；实测分位 p25=4.6s |
| 2 | B 类 d_long / TTC / 横向速度 | 20m / 4s / 0.2 m/s | 25m / 6s / 0.1 m/s | 车辆入走廊 TTC 中位 5.95s |
| 3 | B 类判据的逻辑连接词 | `d_long<20 **或** TTC<4` | `d_long<25 **且** TTC<6` | OR 会收进 “ego 静止 + TTC 14.7s” 的无危险样本（抽检 scene-0553_001_B 抓到） |
| 4 | C 类 | drop≥2s 且降后<3s | drop≥1.5s 且降后<5s | 帧级 TTC 很少跌破 3s |
| 5 | D 类判据 | 不入走廊 **或** 全程 TTC>6s | 不入走廊 **且** 自身 TTC>6s **且** 帧级 TTC>6s | 原判据会把擦走廊边缘的横穿目标、以及“负例帧里还站着别的真危险目标”的帧收成负例，污染 §7.1 硬指标 |
| 6 | D 类每 scene 上限 | — | 6 | 使 D 与 A+B+C 量级相当（29 vs 27） |

**挖掘阶段抓到并修复的 bug**

1. 轨迹速度：对 nan_to_num 后的整条轨迹做中心差分，在有效区间边界产生上千 m/s 的假速度，导致 TTC 出现 ~0.007s 的假尖峰（scene-0061_001_A 抓到）。已改为只在有效区间内差分。
2. B 类判据：手册原文 'd_long<20 或 TTC<4' 的 OR 会收进 'ego 静止 + TTC 14.7s' 的无危险样本（scene-0553_001_B 抓到）。已改为 AND。
3. D 类判据：仅要求目标本身无害时，负例帧里可能仍存在别的真危险目标（帧级 TTC < 3s）。已追加帧级 TTC 安全约束。

## 3. 挖掘统计

- 10 个 scene（10 个纳入），共 **56 事件**：A(VRU 突现) 16、B(近距 cut-in) 5、C(TTC 骤降) 6、D(无害出现，负例) 29
- 日/夜 = 46/10
- min-TTC(1s 窗) 直方图（边界 [0, 1, 2, 3, 4, 6, 10, 100]）：[3, 2, 1, 5, 16, 15, 13]
- **与手册预期相反**：手册预计 A 类稀少、以 B 类为主力；nuScenes mini 是密集城区场景，实际 A(16) 远多于 B(5)。B 类稀少的原因是 mini 中真正的邻道切入极少，且多数“车辆入走廊”其实是 ego 自己逼近静止车辆。

**划分**（scene 级，杜绝泄漏）：估计集 28 事件 / 5 scene，真值集 28 事件 / 3 scene。估计集内部再三分：S_dir 13 / S_sel 8 / S_test 7 事件。

## 4. 指标结果

行为响应量 b = v_plan(clean) − v_plan(ghost)，其中 v_plan 复刻部署端 `control_pid` 的 `desired_speed = ||wp[0]−wp[2]||×2`（waypoint dt=0.25s），即模型真实下发的目标速度。

- 达标率（b > 0.5 m/s）：估计集 **0.393** （scene bootstrap 95% CI [0.053, 0.611]），真值集 **0.179**
- b_min 敏感性：0.25→估计 0.43/真值 0.18，0.5→估计 0.39/真值 0.18，1.0→估计 0.32/真值 0.07
- 峰层 L*=7/24（vision_mean 口径）；另一口径 last_token 选出 L*=2——**两种池化选出的峰层完全不同，是过拟合的直接证据**（S_sel 仅 8 个事件）。

详细读数与失败形态见 `results/consistency_report.md`；图见 `results/figures/`。

## 5. V1–V5 判定

| # | 判定 | 一句话 |
|---|---|---|
| V1 | ✅ PASS | 真值达标率 0.18 落在估计集 95% CI 内——但 CI 宽达 [0.05, 0.61]，**通过是因为 CI 太宽，不是因为估计准** |
| V2 | ❌ FAIL | 共同分层仅 3 个，趋势不可估 |
| V3 | ❌ FAIL | S_test 退化（1 正例）；truth holdout 上 AUC(正例 vs D)=0.303 **< 0.5**，方向不泛化 |
| V4 | ❌ FAIL | 投影-行为 ρ 符号在三块 held-out 间不稳定 |
| V4+ | ✅ PASS | AUC=0.730 达标，但与 V4 矛盾，n=28 下不可解读 |
| V5 | ⏸ N/A | Tier-S 按手册 §0.5 砍掉 CARLA 参考帧 |

**Tier-S 自身的验收**（手册 §0.5：端到端生成 + 全参数化 + 数字标注冒烟读数）：✅ 达成。

## 6. 效度威胁

1. **相机差异**：nuScenes CAM_FRONT hfov≈64° vs CARLA 训练相机 110°，内参/畸变/安装高度均不同。本轮按手册不做纠正（它是待测域差的一部分），但这意味着模型看到的目标尺度与训练分布系统性偏大。
2. **2Hz 标注插值**：TTC 曲线由关键帧标注线性插值到 12Hz sweeps，t_emergence 精度按 ±80ms 记；目标速度来自插值轨迹的差分，对突然起步/刹停的目标有系统性平滑。
3. **B 类替代 A 类的语义折扣反转**：手册预期 B 为主力，实测 A 为主力（见 §3），两类的“突现”语义强度不同，混在一起算达标率会稀释信号。
4. **开环 ≠ 闭环**：b 是单帧规划输出的差分，不是闭环减速；模型在闭环里的 PID 还会叠加刹车逻辑。
5. **样本量**：S_test 7 事件 / 1 scene，truth 28 事件 / 3 scene。scene 级 bootstrap 在 <2 scene 时退化为事件级（已在 JSON 中标记 `bootstrap_unit`）。**本轮所有 p 值与 CI 都只应被当作管线自检，不构成任何关于 SimLingo 表征的结论。**
6. **v_hazard 可能编码“画面变化幅度”而非“危险”**：clean/ghost 相隔 1.2–1.5s，两帧之间除了危险目标出现，还有自车位移带来的全局视角变化。D 类负例在 truth 上投影**更高**，与该混淆一致。Tier-M 必须加入配对更干净的对照（见 §7）。

## 7. 下一步建议

1. **先解决混淆，再放大规模**：当前 δ=h_ghost−h_clean 里混着 1.5s 的自车运动。两个低成本改法：(a) 用 D 类负例的 δ 做“时间基线”，从正例 δ 中回归掉；(b) 缩短 clean/ghost 间隔到 0.5s 并要求同一 scene 内配对。
2. **Tier-M 直接可跑**：trainval metadata + samples 已在本地（`/data/dataset/nuscenes/v1.0-trainval`，54G 已解压），只需把 config 的 `paths.nuscenes_*` 换掉；按 mini 的事件密度（5.6 事件/scene）外推，850 scene 可得 ~4700 事件，足以支撑 500/2k 划分。按本轮 0.59s/事件估算，G2 前向约 46 分钟。
3. **补 CAN bus ego 速度**：本轮用 ego_pose 差分，Tier-M 应接 CAN bus 并交叉校验（手册 §2 要求）。
4. **补 V5**：拿 SimLingo 官方训练数据抽 200 帧 CARLA 参考帧，把 D_L / 干涉角补齐。
5. **峰层 token 级分析暂不值得**：两种池化口径选出的峰层不一致，说明当前层选择本身不可靠，在 S_sel 扩到几百事件之前不要投入 token 级分析。
6. **其余候选模型**：管线已与模型解耦（`scripts/simlingo_runner.py` 是唯一模型相关文件），接 SimLingo-base / TransFuser++ 只需实现同样的 `infer(img, speed) -> waypoints + 每层 hidden` 接口。

## 8. 产出物清单

```
sim2real_demo_ttc/
├── configs/tier_s.yaml            # 唯一参数入口（路径/阈值/split/指标全在这）
├── scripts/
│   ├── simlingo_runner.py         # standalone 单帧推理 + 全层 hidden hook（唯一模型相关文件）
│   ├── g0_smoke.py                # G0 冒烟
│   ├── g1_mine_events.py          # G1 挖掘
│   ├── g1_visualize.py            # G1 人工抽检渲染
│   ├── g1_split.py                # 估计/真值划分 + S_dir/S_sel/S_test 三分
│   ├── g2_cache.py                # G2 批量前向与 h5 缓存（断点续跑）
│   ├── g3_metrics.py              # G3 指标 + G4 判定
│   ├── g4_figures.py              # 出图
│   └── make_report.py             # 本报告生成器
├── mining/  events_all.jsonl, mining_stats.json, splits.json, inspection_record.json, inspect/*.png
├── cache/   {event_id}.h5（19MB，未入库）
└── results/ g0_smoke.json, g2_cache_report.json, metrics_estimate_*.json, truth.json,
            consistency_report.md, figures/*.png
```
