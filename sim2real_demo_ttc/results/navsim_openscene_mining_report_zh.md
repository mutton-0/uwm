# NAVSIM / OpenScene 独立复现语料：可行性评估与挖掘报告

> 工单：[`../docs/navsim_openscene_independent_corpus_workorder.md`](../../docs/navsim_openscene_independent_corpus_workorder.md)。
> 脚本：`scripts/ns1_navsim_geometry.py`（数据源接入层 + 挖掘）、`scripts/n1_match.py`（复用，未改）。
> 配置：`configs/navsim_corpus.yaml`。核心交付见 [`cross_corpus_generality_report_zh.md`](cross_corpus_generality_report_zh.md)。
> 自行决策：[`amendments.md`](amendments.md) §NS/A45–A48。

---

## Methods

### 1. 可行性评估（工单 §3 的 1 天时间盒，实际用时 < 2 小时）

| 检查项 | 结论 | 证据 |
| --- | --- | --- |
| 原始日志在本机？ | **是** | `/data/dataset/navsim/dataset/navsim_logs/test/` 147 个 `.pkl` |
| 相机图像在本机？ | **是** | `sensor_blobs/test/<log>/CAM_F0/*.jpg`，1920×1080，共 219 GB |
| 点云在本机？ | **是** | `<log>/MergedPointCloud/*.pcd`（DDv2 必需） |
| 标注能支持挖掘判据？ | **是** | 每帧 `anns`：`gt_boxes[N,7]`、`gt_names`、**`gt_velocity_3d`**、`track_tokens` |
| 有可复用的读取接口？ | **有，但本轮没用它** | `navsim/common/dataclasses.py` 的 `Scene`/`Frame`/`Annotations`。本轮直接读 `.pkl`，因为挖掘只需要原始字段，走 dataclass 反而多一层转换 |
| 与 nuScenes 独立采集？ | **是（有一处必须限定，见下）** | nuPlan 车队，4 个地图：拉斯维加斯 / 波士顿 / 匹兹堡 / 新加坡 |

**可行性判定：通过。** 未触发止损条款。

**一处必须限定的"独立性"**：nuScenes 采于**波士顿与新加坡**，
本语料含 `us-ma-boston`（5337 事件）与 `sg-one-north`（964 事件），
**地理上与 nuScenes 有重叠**。独立的是：不同车队、不同传感器配置（1920×1080 vs 1600×900，
不同内参与畸变）、不同采集时间、不同标注管线。
因此准确表述是 **"独立采集"而非"地理不相交"**。
本报告并列给出**仅拉斯维加斯 + 匹兹堡**（与 nuScenes 地理不相交，8750 事件）的敏感性分析口径，
但主读数用全量——因为按地图切分会把样本量砍掉一半，反而引入功效问题。

### 2. 语料构造：只换数据源接入层

**判据一行未改。** `scripts/ns1_navsim_geometry.py` 只做一件事：
把 NAVSIM 日志变成 G1 挖掘管线认识的 `geo` 结构，然后

```python
evs = G1.detect_events(geo, scene, cfg)      # ← 判据、阈值一行未改
clean, ghost = G1.pick_frames(geo, t, cfg)   # ← 帧窗口一行未改
rec[...] = G1.frame_record(geo, j, o)        # ← 事件记录构造一行未改
```

`configs/navsim_corpus.yaml` 由 `configs/n1_d2.yaml` 派生，
**mining 段的全部阈值逐字段保持不动**，只改 `work_dir` 与图像根目录。
**为什么不能调阈值**：若为新数据源调判据，"同场景结构、换数据源"就变成了
"换场景又换数据源"，结论不可归因。

负例匹配直接复用 `scripts/n1_match.py`，**未改一行**。

### 3. 数据源差异逐项核对（跨数据源迁移的必查项）

| 项 | nuScenes | NAVSIM/OpenScene | 处理 |
| --- | --- | --- | --- |
| 帧率 | 2 Hz（keyframe） | 2 Hz | 一致，无需处理 |
| 目标位置 | 全局系 → 需 world→ego 变换 | **已在 lidar 系**，且 `lidar2ego` 恒等（实测 translation [0,0,0]、rotation [1,0,0,0]） | 直接用 |
| 目标速度 | **无标注，靠 2 Hz 差分** | **`gt_velocity_3d` 直接给出**（实测静物 \|v\| ≈ 0.001 m/s ⇒ 全局系绝对速度） | 直接用，转 ego 系 |
| 自车速度 | 位移差分 | `ego_dynamic_state[:2]`，**已在 ego 系**（与位移差分转 ego 系核对，差 < 0.15 m/s） | 直接用 |
| 相机 | CAM_FRONT 1600×900，主点 (800, 450)，无畸变系数 | CAM_F0 1920×1080，主点 **(960, 560)**，含 5 阶畸变 | **见 §NS/A46**，投影仍用针孔（登记为偏离） |
| 停驻标注 | `vehicle.parked` / `vehicle.stopped` 属性 | **无 attribute 字段** | 改按运动学（§NS/A45） |
| 天气/光照 | scene description 可解析 | **无标注** | `is_night` / `is_rain` 一律置 False 并声明缺失 |
| 类别 | 细粒度（`human.pedestrian.adult` 等） | 粗粒度 7 类 | 映射表见下，原生名落盘在 `object_class_native` |

**类别映射**（下游按 nuScenes 前缀取用，故必须映射；原生名不丢）：

| NAVSIM 原生 | 映射为 | G1 类 |
| --- | --- | --- |
| `pedestrian` / `bicycle` | `human.pedestrian.adult` / `vehicle.bicycle` | **vru** |
| `vehicle` | `vehicle.car` | vehicle |
| `traffic_cone` / `barrier` / `czone_sign` | `movable_object.*` | **static** |
| `generic_object` | `static_object.generic` | **static** |

### 4. 成本反转：确认，但来源不是模型侧

工单预期"DiffusionDrive/LTF/DDv2 是 NAVSIM 原生训练的，接入这批数据应该比当初接入 nuScenes 更顺"。
**结论：成本确实反转了，但反转发生在数据侧而不是模型侧。**

* **模型侧没有省事**：三个适配器吃的是**原始 RGB + 自车速度**，与数据源无关，
  接 nuScenes 和接 NAVSIM 的工作量相同（都是零）。"NAVSIM 原生"这件事在**推理路径上没有体现**——
  因为我们本来就没走 NAVSIM 的 dataloader，而是复用了自己写的适配器前端。
* **数据侧省了三处**（都在挖掘环节）：
  ① 目标速度直接给（nuScenes 侧靠 2 Hz 差分，那正是 §LB/A44 里 32 个 \|a\| > 10 m/s² 伪影的来源）；
  ② 位置已在 ego 系，省掉 world→ego 变换；
  ③ 类别里天然有 `traffic_cone` / `barrier` / `czone_sign` / `generic_object`，
  正是 D2a"按类别就无害的静物"所需的素材，比 nuScenes 的 `movable_object.*` 丰富得多——
  这直接体现在匹配质量上（见 Results）。
* **但多了一处成本**：§NS/A46 的裁剪主点行。它藏在**模型前端**的一个常量里，
  不逐项核对相机内参就会静默污染全部读数。**这一处抵消了数据侧省下的一部分。**

净结论：**跨数据源迁移的成本主要在"数据源差异逐项核对"，不在"模型是不是这个数据源原生训练的"。**

---

## Results

### 语料规模

| 量 | 值 |
| --- | --- |
| log / scene | 147 / **1880**（另有 146 个 scene 因帧数 < 20 被跳过） |
| 事件总数 | **15051**（1850 个 scene 有事件） |
| **A（VRU 突现，正例）** | **397**（163 scene） |
| B（近距 cut-in）/ C（TTC 骤降） | 178 / 1111 |
| D2a（几何匹配静物） | 2725 |
| D2c（同类入走廊、TTC 全程高） | 2255 |
| 地图 | 拉斯维加斯 6744 / 波士顿 5337 / 匹兹堡 2006 / 新加坡 964（按事件数） |
| A 的原生类别 | pedestrian 371 / bicycle 26 |
| D2a 的原生类别 | generic_object 1584 / traffic_cone 783 / barrier 338 / czone_sign 20 |

**A = 397 大于 G1 的 291。** 这一点对本轮的核心目标至关重要：
判决 LTF 的 G 轴阳性需要功效，而本语料的功效**不低于** G1。

### 负例匹配质检

**Table 1. N1 caliper 匹配（log 成像面积 ≤ 0.15 dex，离心率 ≤ 0.06），与 G1 同一口径、同一脚本。**

| 负类 | 池 | 配对 | 匹配率 | SMD 面积（前→后） | SMD 离心率（前→后） |
| --- | --- | --- | --- | --- | --- |
| D2a | 2678 | **382** | 99% | +1.29 → **+0.02** | +0.27 → **−0.01** |
| D2b | 3500 | 381 | 99% | −0.02 → +0.01 | −1.43 → +0.00 |
| D2c | 2222 | 383 | 99% | −0.20 → −0.01 | +0.29 → +0.00 |
| **D2cV（证伪地板）** | 915 | **296** | **77%** | −0.03 → **−0.01** | +0.35 → **+0.04** |

全部 |SMD| ≤ 0.04，**匹配质量优于 G1 与前车急刹两个语料**
（前车急刹的 LBv 是 −0.328，受负例池只有 37 个所限）。
原因正是 §成本反转 ③：NAVSIM 的静物类别更丰富，D2a 的候选池有 2678 个（G1 侧远少于此），
caliper 有充分的选择余地。

**D2cV 匹配后进入分析的 n = 134**（匹配清单 296 条含正负两侧，
且需与已缓存事件求交）。这是本轮唯一比 G1（212）小的量，
但由于 A（397 vs 291）与 D2a（382 vs 283）都更大，
主读数 − 地板的 CI 宽度实测为 **0.103**，反而**略窄于 G1 的 0.109**（见跨语料报告 §2）。

---

## Discussion

**本报告的作用是把"可比性"钉死。** 跨数据源比较最容易出的问题不是统计，
而是"两边其实测的不是同一件事"。本轮把可比性拆成三条，逐条给出可核查的证据：

1. **判据可比**：`detect_events` / `pick_frames` / `frame_record` 全部 import 自 `g1_mine_events`，
   阈值逐字段不动；负例匹配复用 `n1_match.py` 未改一行。
2. **几何可比**：相机内外参逐项核对，主点行差异已修正（§NS/A46）；
   畸变差异登记为偏离（组间无偏，绝对值不可跨语料比）。
3. **统计可比**：scene 级 bootstrap、预注册主读数、三态判定、10 seed 折分配稳定性
   全部沿用，且本轮的 CI 宽度与 G1 相当（0.103 vs 0.109），使"不复现"与"功效不足"可分。

**一处如实记录的不可比**：成像面积与离心率的**绝对值**跨语料不可比
（不同分辨率、不同主点、NAVSIM 有畸变而投影按针孔）。
这不影响本轮任何读数——所有读数都是**语料内**的组间对比，
且几何匹配是在**各自语料内**做的。

---

## 自我更正记录

1. **§NS/A46 是本轮最危险的一处，且它不报错。** `CROP_CENTER_ROW = 450` 藏在模型前端，
   语义是"相机主点行"，但值写死成了 nuScenes 的。沿用它会给三个模型喂偏高 110 px 的画面，
   静默降低全部读数。**教训已推广**：跨数据源迁移必须逐项核对相机内参，
   "看起来通用的前处理常量"往往编码了原数据源的几何。
2. **§NS/A45**：NAVSIM 无 attribute 标注，`is_parked` 改按运动学定义。
   这是定义偏离而非等价替换，已声明影响面（只进 D2aP，不进主读数）。
3. **"独立采集"不等于"地理不相交"**：本语料含波士顿与新加坡，与 nuScenes 地理重叠。
   已在 Methods §1 明确限定，并给出仅拉斯维加斯 + 匹兹堡的敏感性口径。
   **不把"独立数据源"写成比事实更强的说法。**
