# RHD 右舵扩量 —— 步骤 A 判定：**不可行，不要下传感器包**

回给交接方。按交接文档 §3A 的判定规则执行，§4 要求"若不可行只回报数字"，故**未进入步骤 B**，
不产出 `rhd_pool_{ghost,lead}.json`。

---

## 1. 结论一句话

**nuPlan 全库（train+val+test）里"发布了传感器的新加坡 scene"总共只有 1,355 个**，
而达到 ~110 个 lead 事件需要 ≥ 8,000 个。差 5.9 倍。
**val/test 没有增量，train 也没有** —— 后者你文档里已列为已确认事实，本次独立复核成立。

---

## 2. 判定所依据的数字

### 2.1 有传感器的 log 比例（三个 split 全查）

| split | 官方 log 总数 | 发布传感器的 log | 比例 |
|---|---|---|---|
| train | 13,180 | 1,085 | 8.2% |
| val | 1,381 | 225 | 16.3% |
| test | 1,349 | 147 | 10.9% |

（train 的 1,085 与你文档一致；val/test 系本次首次解析。）

### 2.2 有传感器的 scene，按城市（val + test，共 372 个 log 全扫）

| city | logs | scenes | frames |
|---|---|---|---|
| las_vegas | 111 | 1,785 | 668,470 |
| us-ma-boston | 131 | 1,372 | 496,220 |
| **sg-one-north** | **65** | **837** | **307,339** |
| us-pa-pittsburgh-hazelwood | 65 | 792 | 289,821 |
| **合计** | **372** | **4,786** | |

### 2.3 新加坡，按 split 拆开（这是判定的核心表）

| split | SG **原始** log | SG log（有传感器） | 发布率 | SG scene |
|---|---|---|---|---|
| train | 2,396 | 45 | **1.9%** | 518 |
| val | **255** | 52 | 20.4% | 629 |
| test | **258** | 13 | 5.0% | 208 |
| **总计** | **2,909** | **110** | **3.8%** | **1,355** |

> 2026-09-07 补：原表只有"有传感器"一列，缺分母。val/test 的原始 SG log 数
> （255 / 258）由 `scripts/rhd_a5_raw_city.py` 首次解出 —— val/test 的 zip
> **不按城市分包**（实测 `nuplan-v1.1_val_singapore.zip` 为 404，只有 train 分城市），
> 故对全部 2,730 个 val/test log 各 range-取压缩流前 96 KB 增量解压，
> 在解出的前缀里读 `location`（实测该字段在解压后第 ~16 KB 处）。
> 2,730 个 log 合计约 175 MB 流量，而非 190 GB。识别率 2,730/2,730，无未知。

**读法：分母不小，是发布率低。** 2,396 这个数不是"新加坡只有这么多数据"，
它是分母；判定卡在 1.9% 这个发布率上，不是卡在采集量上。

### 2.4 判定

现有产出率（你文档给的）：ghost 0.15%、lead 0.59%。

| 口径 | SG scene | 预期 ghost | 预期 lead |
|---|---|---|---|
| val+test（本次唯一可能的增量） | 837 | 1.3 | 4.9 |
| 再加上 train 全部 | 1,355 | 2.0 | 8.0 |
| **目标** | **≥8,000** | **110** | **110** |

**VERDICT: NOT FEASIBLE。** 判定阈值 8,000，实测 837（val+test）或 1,355（全库）。


---

## 2.6 补：nuPlan 官方**从未公布**的按城市小时数（本次自行算出）

**先说找不到的**：nuPlan 论文（arXiv 2106.11810）摘要只有
*"1500h of human driving data from 4 cities"*，**无按城市拆分**；
devkit 的 `dataset_setup.md` / `nuplan_schema.md` 只有目录与表结构，
**无任何小时数或城市统计**；S3 上也无统计文件。**官方城市级拆分表不存在。**

**所以自己算。** 依据：nuPlan 的 log 文件名尾部两个数字是起止秒偏移，
`end − start` **等于该 log 的精确时长**。实测 4 个本地 db，与
`max(lidar_pc.timestamp) − min(...)` 逐秒吻合：

| log | 文件名差 | db 实测 |
|---|---|---|
| `2021.06.07.11.59.52_veh-35_02283_02464` | 181 | 181.0 s |
| `2021.06.07.12.42.11_veh-38_01777_02078` | 301 | 301.0 s |
| `2021.06.07.12.42.11_veh-38_02445_02843` | 398 | 398.0 s |
| `2021.06.07.12.42.11_veh-38_03254_03455` | 201 | 201.0 s |

于是只读各 zip 的**中央目录**（几十 MB）就能算全库，不下任何 db。

| split | city | 原始 log | **原始小时** | 有传感器 log | **有传感器小时** | 发布率 |
|---|---|---|---|---|---|---|
| train | las_vegas | 7,577 | 646.9 | 757 | 67.5 | 10.4% |
| train | **sg-one-north** | **2,396** | **146.7** | **45** | **2.7** | **1.8%** |
| train | us-pa-pittsburgh | 1,560 | 107.8 | 97 | 7.4 | 6.8% |
| train | us-ma-boston | 1,647 | 79.2 | 186 | 8.9 | 11.3% |
| val | las_vegas | 760 | 59.5 | 61 | 5.3 | 8.8% |
| val | sg-one-north | 255 | 15.6 | 52 | 3.2 | 20.4% |
| val | us-pa-pittsburgh | 174 | 10.1 | 43 | 2.3 | 23.1% |
| val | us-ma-boston | 192 | 8.8 | 69 | 3.3 | 37.0% |
| test | las_vegas | 740 | 64.1 | 50 | 4.0 | 6.3% |
| test | sg-one-north | 258 | 15.0 | 13 | 1.1 | 7.3% |
| test | us-pa-pittsburgh | 176 | 11.9 | 22 | 1.7 | 14.1% |
| test | us-ma-boston | 175 | 8.6 | 62 | 3.6 | 42.1% |

**全库合计**

| city | 原始 log | **原始小时** | 有传感器 log | **有传感器小时** | 发布率 |
|---|---|---|---|---|---|
| las_vegas | 9,077 | 770.5 | 868 | 76.8 | 10.0% |
| **sg-one-north** | **2,909** | **177.3** | **110** | **6.9** | **3.9%** |
| us-pa-pittsburgh | 1,910 | 129.8 | 162 | 11.4 | 8.8% |
| us-ma-boston | 2,014 | 96.7 | 317 | 15.8 | 16.4% |
| **合计** | **15,910** | **1,174.3** | **1,457** | **110.9** | **9.4%** |

**自检**：train 四城 log 数相加 = 2,396+1,647+1,560+7,577 = **13,180**，
与官方 train 总数逐个吻合。

**一处对不上，如实记**：本表全库 **1,174.3 小时**，而论文说 1500 h，差 22%。
逐 log 时长是精确的（上表已验），所以差额只可能来自"1500 h 是约数"或
"含未公开发布的部分"。**未能证实，不硬凑。**

### 2.7 两个口径必须分开讲

| | 内容 | 由谁决定 |
|---|---|---|
| **原始 log** | GPS / 轨迹 / 3D 标注 / 地图，**无图像无点云** | nuPlan 采集了多少 |
| **传感器子集** | 相机 + 激光 | nuPlan **选择发布**了多少 |

**新加坡有 177.3 小时原始数据，其中只有 6.9 小时发布了传感器（3.9%）。**

遮挡实验必须构造"遮住危险物 / 不遮住"两条臂，**没有图像就构造不出来**，
F / G / C 三根轴全部依赖这一点。所以 §2.4 的 NOT FEASIBLE **不受原始数据量影响**，
它卡在发布率上。而新加坡恰好是四城里被卡得最狠的：3.9%，
对比波士顿 16.4%、维加斯 10.0%、匹兹堡 8.8%，低 2–4 倍。

**数据体量**：全部原始 log 的官方包体 = train 1,017.3 GB + val 97.0 + test 95.9
= **1,210 GB**（HEAD 实测）。传感器 blob 是另行发布的一批，量级更大。
若外界说"16 TB"，指的不可能是原始 log。

产物：`results/rhd_raw_city_valtest.json`、`results/rhd_city_hours.json`；
脚本 `scripts/rhd_a5_raw_city.py`、`scripts/rhd_a6_city_hours.py`。

---

## 3. 一条强内部一致性佐证

把全库 1,355 个 SG scene 代入现有产出率，得 **ghost 2.0 / lead 8.0**；
而你文档里 OpenScene 已挖满的结果是 **ghost 2 + lead 8**，且 OpenScene 侧的 SG scene 数记为 **1,354**。

两边在 scene 数（1,355 vs 1,354，差 1）与事件数（2/8 vs 2/8）上同时对上。
最自然的解释是：**OpenScene 的那 1,354 个新加坡 scene 就是 nuPlan 全库发布了传感器的
新加坡 scene 的全部**，你已经挖到了这个池子的底。剩余的 nuPlan 新加坡数据
（train 2,396 个 log 里的 98.1%）**根本没有发布图像**，不是"还没下"，是"官方没放出来"。

> 口径提醒：这条是**推断**，不是我直接核对了 OpenScene 的 scene 清单。
> 我比对过 45 个 SG train log 与现有 pool 的 `log_name`，只有 1 个重合——但那不能当证据，
> 因为 pool 只记录**出了事件的 log**，而 `brake_first_pool_*_trainval.json` 的
> 1,368 条候选里 `log_name` 全是 `None`，无法区分"没扫过"和"扫了没出事件"。
> 若要坐实这条，需要一份 OpenScene trainval 实际扫过的 scene/log 清单。
> **但这不影响判定**：1,355 个 scene 无论是否已挖过，都远低于 8,000。

---

## 4. 花了多少、放在哪

**没有下任何传感器包（4 TB 一个字节都没动）。**

| 内容 | 大小 | 位置 |
|---|---|---|
| val/test 三份 sensor 清单 + train 清单 | 60 KB | `/mnt/skylabNAS/nuplan_sg/meta/public_set_*_sensor.txt` |
| val/test zip 中央目录索引 | 0.4 MB | `/mnt/skylabNAS/nuplan_sg/meta/rhd_zip_index.json` |
| val+test 372 个有传感器 log 的 `.db` | 17.8 GB（压缩）/ 29.8 GB（解压） | `/mnt/skylabNAS/nuplan_sg/dbs/{val,test}/` |
| train 45 个 SG log 的 `.db` | 0.63 GB | `/mnt/skylabNAS/nuplan_sg/dbs/train_sg/` |
| **合计下载** | **约 18.5 GB** | 全部在 NAS |

省下的：val.zip + test.zip 全量是 **193 GB**，传感器包是 **4 TB**。

做法：S3 支持 byte range，所以先用 range 读两个 zip 的**中央目录**（各 0.4 MB，zip64），
拿到全部 `.db` 条目的偏移与大小，再只 range 取那 372 个有传感器的 log。
解析出的条目数 1,381 / 1,349 与官方 log 数完全吻合，可作为解析正确性的校验。

---

## 5. 复现入口（全部新建，`rhd_` 前缀）

| 脚本 | 作用 |
|---|---|
| `scripts/rhd_a1_zipdir.py` | range 读 val/test zip 中央目录（含 zip64），输出条目索引 |
| `scripts/rhd_a2_fetch_dbs.py` | 按清单只 range 取有传感器的 `.db` |
| `scripts/rhd_a3_city_scan.py` | 扫 `location` / scene 数，出城市表与判定 |
| `scripts/rhd_a4_train_sg.py` | 用 city-split 的 train_singapore.zip 中央目录核查 train 侧增量 |

数值真源：`results/rhd_stepA_feasibility.json`（含 372 条 per-log 记录）、
`results/rhd_train_sg_summary.json`。

> 过程说明：`rhd_a3` 第一次运行时 225 个 val `.db` 因 NFS 属性尚未同步而全部读失败，
> 脚本如实报了 `failed=225`，我发现后等文件落定重跑，第二次 `ok=372 failed=0`。
> 当前 JSON 是重跑结果。

---

## 6. 未做的事

- **没有**进入步骤 B（挖矿），因为 §4 要求判定不可行就停。
- **没有**复制或修改 `brake_first_miner.py` / `f3_*.py` / `c_axis_*.py` / `gvs*.py`。
- **没有**写 `/data/dataset/navsim/**`，**没有**动 `results/` 下非 `rhd_` 前缀的文件，
  **没有**用 cuda:1（本次全程未用 GPU）。
- `/mnt/skylabNAS/nuplan_sg/train_singapore.zip`（你之前下的 24.6 GB 半包）**未动**。
  本次判定不需要它——city-split zip 的中央目录就够了，不必下完 35 GB。

---

## 7. 如果还想要右舵样本，可能的方向（都不在本次范围内）

1. **换数据源**：nuScenes 新加坡三区已用尽（16 个事件），nuPlan 新加坡传感器只有 1,355 scene。
   要凑到 110，得引入 nuPlan/nuScenes 之外的右舵语料（如 Waymo 无右舵；
   香港/日本/英国的公开集需另评）。
2. **接受小样本并如实标注**：按你文档 §5 的纪律，右舵 vs 左舵以"不可估"报告，
   而不是放宽判据凑数。以现有 16 个事件，这是唯一诚实的写法。
3. **改用无图像的口径**：若某条轴的读数不需要图像（纯轨迹/标注），
   则新加坡可用 scene 数从 1,355 放大到 2,396 个 log 的全部——
   但那就与现有 deployment 池的口径不一致，跨池比较会失效。

---

## 8. 补充核查：本地 NavSim 全场景库（应"去 /data 下看看"的要求）

**结论：本地确实有一份完整的 NavSim/OpenScene 场景库，但它的新加坡内容与我上面统计的
是同一批，没有任何增量；判定不变。附带好消息是——这批数据已经在本地，要挖不用再下。**

### 8.1 本地有什么

| 路径 | 内容 | 大小 |
|---|---|---|
| `/data/dataset/navsim/dataset/sensor_blobs/trainval` | **1,280** 个 log 的传感器 | 486 GB |
| `/data/dataset/navsim/dataset/sensor_blobs/test` | **147** 个 log 的传感器 | 219 GB |
| `/data/dataset/navsim/dataset/navsim_logs/{trainval,test}` | 对应的 `.pkl` 元数据 | 13.9 GB / 983 MB |
| `/data/dataset/navsim/stage/openscene-v1.1/meta_datas/trainval` | OpenScene trainval 元数据，**1,310** 个 log | 14 GB |
| | **合计** | **约 705 GB** |

### 8.2 关键对账（这几条同时成立，基本锁死了结论）

1. **OpenScene trainval = nuPlan train + val 的传感器 log，精确相等**：
   本地 meta_datas/trainval 有 **1,310** 个 log；
   我从 S3 清单查到 nuPlan train 有传感器的 log **1,085** 个、val **225** 个，
   1,085 + 225 = **1,310**。一个不差。

2. **本地新加坡 log = 我统计的新加坡 log**：
   我的清单 110 个（train 45 + val 52 + test 13），
   本地 `sensor_blobs` 里能找到 **109** 个（trainval 96 + test 13），
   仅缺 1 个：`2021.10.05.04.38.41_veh-50_01202_01296`（5 个 scene）。
   → 本地可用新加坡 scene = **1,350**，我统计的全库上限是 **1,355**。

3. 因此"NavSim 有没有更大的场景库"这个问题的答案是**没有**：
   NavSim 的场景库就是 nuPlan 发布了传感器的那一部分，
   它的新加坡部分已经被你挖过（对应你文档里的 ghost 2 + lead 8）。

### 8.3 对判定的影响

**判定不变：仍然不可行。** 1,350 个可用 scene 代入现有产出率 → ghost 2.0 / lead 8.0，
与你已经拿到的结果一致，说明池子确实见底。

**但有一条操作上的更正**：我在 §4 里写"省下 4 TB 传感器下载"——
更准确的说法是，**那 4 TB 本来就不需要下，因为新加坡那部分（约 705 GB 中的一小块）已经在本地**。
如果将来要对新加坡重挖（比如换判据、或做无图像口径），
直接读 `/data/dataset/navsim/dataset/` 即可，**下载量为 0**。
注意该路径按工单归你，我只读未写。

---

## 9. 本地数据集实际数量（全量清点）

用 `scripts/rhd_b1_local_inventory.py` 把本地 NavSim 元数据全部读了一遍
（1,457 个 log 的 `.pkl`，按 `scene_token` 去重计 scene，按 `map_location` 归城市），
`ok=1457 failed=0`。

### 9.1 总量

| | logs | scenes | frames |
|---|---|---|---|
| trainval | 1,310 | 19,376 | 723,019 |
| test | 147 | 2,026 | 75,122 |
| **合计** | **1,457** | **21,402** | **798,141** |

### 9.2 按城市（两个 split 合并）

| city | logs | scenes | frames | 占比(scene) |
|---|---|---|---|---|
| us-nv-las-vegas-strip | 868 | 14,687 | 552,502 | 68.6% |
| us-ma-boston | 317 | 3,155 | 113,854 | 14.7% |
| us-pa-pittsburgh-hazelwood | 162 | 2,206 | 81,911 | 10.3% |
| **sg-one-north（右舵）** | **110** | **1,354** | **49,874** | **6.33%** |

### 9.3 这张表把前面的推断变成了事实

§3 里我把"OpenScene 的 1,354 个新加坡 scene 就是全部"标注为**推断**，
现在可以改成**已核实**：本地库的新加坡 scene 数**正好是 1,354**，
与你文档里写的数字一字不差，且 log 数 110 与我从 nuPlan S3 清单独立数出来的 110 完全一致。

（唯一的 1 个 scene 差异：nuPlan `.db` 侧我数到 1,355，本地 OpenScene 侧是 1,354。
这是两边 scene 切分口径的差异，不影响任何结论。另外 110 个 log 的**元数据**都在本地，
但 `sensor_blobs` 只有 109 个——缺 `2021.10.05.04.38.41_veh-50_01202_01296`。）

### 9.4 对右舵扩量的最终含义

**右舵可用池 = 1,354 个 scene，占整个本地库的 6.33%，且已挖满。**
代入现有产出率 → ghost 2.0 / lead 8.0，与你已有的 ghost 2 + lead 8 完全吻合。
要达到 ~110 个事件需要 ≥8,000 个右舵 scene，缺口 5.9 倍，
而 nuPlan 官方在新加坡就只发布了这么多传感器数据（2,396 个 SG train log 里仅 45 个有图像）。

**这不是"还没下载"的问题，是数据本身不存在。**

---

## 10. 现有挖掘数据量（已挖出的事件池清点）

脚本：`scripts/rhd_b2_pool_inventory.py`（清点）、`scripts/rhd_b3_pool_city_join.py`（补城市标签）。
输出：`results/rhd_mined_pool_inventory.json`、`results/rhd_pool_city_join.json`。

### 10.1 主线在用的池子

| 池 | 事件 | scene | log | 城市构成 |
|---|---|---|---|---|
| `deploy_pool_lead_vp.json`（**部署池 lead**） | **317** | 316 | 223 | vegas 268 / pittsburgh 49 |
| `deploy_pool_ghost_vp.json`（**部署池 ghost**） | **49** | 49 | 45 | vegas 44 / pittsburgh 5 |
| `deploy_pool_lead_all.json`（未限城市） | 385 | 384 | 280 | vegas 268 / boston 65 / pittsburgh 49 / **SG 3** |
| `deploy_pool_ghost_all.json`（未限城市） | 53 | 53 | 49 | vegas 44 / pittsburgh 5 / boston 3 / **SG 1** |

`vp` = Vegas+Pittsburgh，是论文"benchmark 与 deployment 无共同城市"设计所要求的过滤，
因此**部署池按设计就不含新加坡**（`pool_lead_trainval_vp` 里记录 `n_scenes_skipped_by_city=3243`）。

### 10.2 上游原始候选池

| 池 | 事件 | 其中新加坡 |
|---|---|---|
| `brake_first_pool_lead_trainval.json` | 1,368 | **17** |
| `pool_lead_trainval_vp.json`（已过滤城市） | 1,233 | 0 |
| `brake_first_pool_ghost_trainval.json` | 180 | **6** |
| `pool_ghost_trainval_vp.json`（已过滤城市） | 160 | 0 |
| `brake_first_pool_lead_navsim_fx.json`（test） | 132 | **1** |
| `brake_first_pool_navsim_fx.json`（test, ghost） | 13 | **2** |

nuScenes 侧的池（`*_isec_nusc_v2` 10 条、`*_leftturn_nusc_v2` 2 条、`brake_first_pool*` 13–18 条）
用的是 nuScenes 的 scene 命名，与 NavSim 的 `log-XXXX-scene-YYYY` 不同名空间，
本次未做城市 join（`matched=0`），不影响右舵结论——工单已说明 nuScenes 右舵事件是 16 个。

### 10.3 右舵产出率：三个口径，结论不变

| 口径 | ghost | lead | lead 率 | 达到 110 lead 所需 scene |
|---|---|---|---|---|
| 原始候选（本次实测） | 8 | 18 | 1.33% | 8,271 |
| 工单给的口径 | 2 | 8 | 0.59% | 18,644 |
| 最终入池（`deploy_pool_*_all`） | 1 | 3 | 0.22% | 50,000 |

**可用新加坡 scene 只有 1,354。** 即使用**最宽松**的原始候选率（1.33%），也还差 6.1 倍。
**判定不依赖于选哪个产出率口径。**

> 工单写的 ghost 2 + lead 8 落在"原始候选"与"最终入池"之间，
> 应是某个中间筛选阶段的计数；我没有去复原是哪一阶段，因为三个口径都指向同一结论。

### 10.4 一处我自己的错误（已修正，记录在案）

第一版 join 我用了**裸 `scene` 名**做键，得出 lead 池新加坡 26 条、`deploy_pool_lead_all` 5 条。
与该池自带的 `city_counts`（SG=3）对不上，说明我错了——正是工单坑 #2 警告的
"`scene_name` 跨 split 不唯一"。实测：`scene_name` 有 20,068 个，其中 **835 个跨城市**、
**1,334 个（6.6%）同时出现在 trainval 和 test**；而 `scene_token` 21,402 个**零碰撞**。
池子里存的恰好是 `scene_name`，所以必须用 `(split, scene)` 复合键。

改用复合键后，四个 `deploy_pool_*` 的城市分布与它们自带的 `city_counts` **逐格吻合**，
join 才算可信。上表 10.2/10.3 是修正后的数字。

---

## 11. 左舵 / 右舵总账（两个语料都 join 上了）

脚本 `scripts/rhd_b4_driveside.py`，输出 `results/rhd_driveside_join.json`。
nuScenes 用 `scene.json` → `log.json` 的 `location`；NavSim 用 `(split, scene_name)` 复合键。
两边 30 个池全部 join 成功（除 `f3_candidate_pool.json` 6 条无法归属，见下）。

口径：**右舵 RHD** = 方向盘在右、靠左行驶 = 新加坡；**左舵 LHD** = 美国三城。

### 11.1 语料层面：可用 scene（分母）

| 语料 | RHD scene | LHD scene | 合计 |
|---|---|---|---|
| nuScenes v1.0-trainval | **383**（onenorth 183 / queenstown 115 / hollandvillage 85） | 467（boston-seaport） | 850 |
| NavSim / OpenScene | **1,354**（sg-one-north） | 20,048（vegas 14,687 / boston 3,155 / pittsburgh 2,206） | 21,402 |
| **合计** | **1,737** | **20,515** | 22,252 |

**右舵只占全部可用 scene 的 7.8%。** 注意 nuScenes 的右舵占比很高（383/850 = 45%），
但它总量太小；NavSim 量大（21k）却只有 6.3% 是右舵。

### 11.2 事件层面：已挖出的事件（去重后的口径池）

| 语料 | 场景 | 池 | LHD | RHD |
|---|---|---|---|---|
| nuScenes | ghost | `brake_first_pool_final.json` | 3 | **10** |
| nuScenes | lead | `brake_first_pool_lead_final.json` | 10 | **6** |
| NavSim | ghost | `deploy_pool_ghost_all.json` | 52 | **1** |
| NavSim | lead | `deploy_pool_lead_all.json` | 382 | **3** |
| **合计** | | | **447** | **20** |

**右舵事件共 20 个，占已挖事件的 4.3%。**

其中 nuScenes 侧的 **ghost 10 + lead 6 = 16**，与你交接文档写的
"右舵事件只有 16 个（ghost 10 + lead 6），且全部来自 nuScenes 新加坡三区"**完全一致**，
这两个池也因此可确认为 nuScenes 侧的口径池。

NavSim 侧右舵只有 4 个（ghost 1 + lead 3）。你文档写的 OpenScene "ghost 2 + lead 8"
与此不符——我实测的三个阶段是：原始候选 ghost 8 / lead 18 → 最终入池 ghost 1 / lead 3，
"2 + 8" 落在两者之间，应是某个中间筛选阶段。**不影响任何结论**（见 §10.3）。

### 11.3 实际用于四轴的部署池：右舵为 0

| 池 | LHD | RHD |
|---|---|---|
| `deploy_pool_lead_vp.json` | 317 | **0** |
| `deploy_pool_ghost_vp.json` | 49 | **0** |

`vp` 按设计只保留 Vegas+Pittsburgh（论文要求 benchmark 与 deployment 无共同城市），
所以现在跑四轴的池子里**一个右舵事件都没有**。
右舵那 20 个事件目前散在 nuScenes 口径池和 `deploy_pool_*_all` 里，未进入部署池。

### 11.4 对"右舵 vs 左舵"这个对比的含义

即便把两个语料的右舵事件全部合并，也只有 **20 个**（ghost 11 / lead 9）。
按你文档 §5 的纪律，这个量级只能报"**不可估**"。
而要把右舵做到与左舵同等的统计强度（lead 317 / ghost 49 那个量级），
需要的右舵 scene 远超现有 1,737 个——这正是 §1–§9 判定不可行的原因。

> 一处未归属：`f3_candidate_pool.json` 的 6 条在两个命名空间里都匹配不上
> （`matched=0`），未计入上表。其余 30 个池全部 join 成功。
