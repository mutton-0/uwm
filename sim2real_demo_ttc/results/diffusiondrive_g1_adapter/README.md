# DiffusionDrive × G1 语料 输入适配器

> 工单产出物（`docs/four_axis_proof_experiment_workorder.md` §8）。
> 存在的唯一理由：G 轴正向校准要求 DiffusionDrive 跑在与 SimLingo **完全相同**的刺激集上
> （nuScenes 鬼探头挖掘语料 G1 + N1 的 D2a/D2b/D2c/D2cV 负例），
> **不得**用 DiffusionDrive 自己的 NAVSIM/navhard 评测集替代——否则失去可比性，
> 整个校准的意义就没了（`docs/g_axis_positive_calibration_diffusiondrive.md` §3）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `dd_adapter.py` | 输入适配（nuScenes 帧 → DiffusionDrive 张量）+ 8 层可读表征抽取 |
| `run_g1_cache.py` | 批量跑 G1 语料，逐事件缓存 `.npz` |

## 环境

DiffusionDrive 侧脚本用 **simscale** 环境（本仓库 navsim 已安装在其中）：

```bash
PY=/home/mut0/.conda/envs/simscale/bin/python
```

该环境无 `h5py`，故缓存格式为 `.npz` 而非 `.h5`（与 SimLingo 侧的 `.h5` 缓存并行，互不影响）。

## 用法

```bash
cd /data/ruolin/uwm/sim2real_demo_ttc
$PY results/diffusiondrive_g1_adapter/run_g1_cache.py \
    --types A D2a D2b D2c --device cuda:1
# -> variants/n1_d2/dd_cache/<event_id>.npz   （1112 事件 ≈ 113 s，0.10 s/event）
$PY scripts/g_axis_dd_readout.py               # -> results/g_positive_calibration_diffusiondrive.json
```

## 口径对齐（与 SimLingo 侧逐条同构，协议 §4）

| 项 | 做法 | 依据 |
| --- | --- | --- |
| 图像 | 居中裁 4:1 去天空（以行 450 为竖直中心）→ resize(2048, 512) | 与 `scripts/ghosthead_infer/run_ghosthead_infer.py` **逐像素同一套** |
| ego 状态 | `driving_command` 直行 one-hot + (v, a)，**两条件共用 clean 帧锚定的速度** | 与 SimLingo 的 `prompt_anchor: clean` **同构**（消除同一个已知混淆） |
| 可读层 | TransFuser 编码器 8 个 `SelfAttention` 的输出，每层 320 token = 256 图像 + 64 BEV latent | 协议 §4 规则 3「层按角色对齐，不按层号」 |
| 池化 | `vision_mean`（256 图像 token 均值，= SimLingo `vision_mean` 的同构物）、`region_mean`（bbox 内图像 token）、`bg_mean`、`lidar_mean`、`all_mean` | 同上 |
| bbox → token | 图像 token 网格为 **8 行 × 32 列**（`img_vert_anchors` = 256/32，`img_horz_anchors` = 1024/32），行主序，外扩 1 token 容错 | `navsim/agents/diffusiondrive/transfuser_config.py` |
| region 框 | 两条件**共用 ghost 帧的框** | 与 `scripts/g2_cache.py` 同一纪律（clean 帧目标常不可见，各用各的框会让 δ 无定义） |
| 行为量 | `commanded_speed = ‖traj[0]‖ / PRED_DT`（PRED_DT = 0.5 s） | 与 SimLingo 的 `commanded_speed(wp)` 同为"模型下发的目标速度" |

## 缓存 schema（每个事件一个 `.npz`）

```
meta                       事件账本 json 串（含 _n_region_tokens / _prompt_anchor_speed）
clean/<pool>/L<l>          [n_frames, C_l]    l = 0..7；C_l = 64,64,128,128,256,256,512,512
ghost/<pool>/L<l>          [n_frames, C_l]
commanded_speed_{clean,ghost}   [n_frames]
traj_{clean,ghost}              [n_frames, 8, 3]
```

**注意各层通道数不同**（4 个尺度 × 2 block），因此方向必须逐层独立拟合与归一化，
不能像 SimLingo 那样堆成一个 `[L, C]` 数组。`scripts/g_axis_dd_readout.py` 已按此处理。

## 复用

`scripts/i_axis_extract.py --model dd` 复用 `dd_adapter.DDRunner`，
把同一套抽取用在 ghosthead 域配对帧上（T-I 的 $v_{domain}$ / $D_L$ 与 T-C 的 t-SNE 共用原料）。
