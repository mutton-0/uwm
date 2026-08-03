# Ghosthead → DiffusionDrive 纯图片推理 步骤文档（待审核）

> 目标：把 ghosthead 场景视频抽帧，喂进 `diffusiondrive_sim_navhard.ckpt` 做 **standalone 推理**，
> 拿到预测轨迹，在 BEV 里对着 scene.json 的真实 actor/ego GT **计算 PDM 式分数并可视化**。
> **不碰 navmini / 不下 dataset**；只用图片输入，参考 navhard 的推理形态。
>
> ⚠️ navsim 官方 `PDMScorer` 在此不可用：Town04 是 CARLA 城、`NUPLAN_MAPS_ROOT` 无对应 nuPlan 地图，
> 且无 token/metric_cache。故**自实现一套等价的 PDM 式 BEV 评分**（见第 6 节），drivable-area 因无路面
> 几何而省略。

## 0. 本次范围（已确认）
- 只跑 **`gh_001000__brake`** 一个场景，跑通后再议扩量。
- 每帧各跑一次推理：t=1/2/3/4 s 共 **4 帧 → 4 条预测轨迹**（transfered 4 条 + origin 4 条 = 8 次推理）。
- 单目 16:9 **居中裁成 4:1（去天空，不拉伸）**再等比放大到 2048×512 当模型输入（模型只吃 `cameras[-1]` 单帧）。
- 输入源：
  - **transfered** = `/data/Zhengyang/Auto_Eval/ghosthead_v1/renders/gh_001000__brake/frames.mp4`
  - **origin** = `/data/Zhengyang/Auto_Eval/ghosthead_v1/ghosthead_result/gh_001000__brake/seg1p0/gh_001000__brake_seg1p0.mp4`
  - 两段都是 1600×900 / 10fps / 4s / 40 帧。

## 1. 环境变量（沿用两份文档口径，dataset 路径已确认存在，不新下载）
```bash
cd /data/ruolin/uwm
conda activate simscale
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset"
export NAVSIM_DEVKIT_ROOT="/data/ruolin/uwm"
export NAVSIM_EXP_ROOT="/data/ruolin/navsim_logs/exp"
export PYTHONPATH="/data/ruolin/uwm:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```
说明：这些只是让 `navsim` import 时 module-level 读 `OPENSCENE_DATA_ROOT/NUPLAN_MAPS_ROOT` 不为 None；
standalone 推理不会真的去加载 navsim 场景，所以**不需要任何 navmini 数据 / metric cache**。

## 2. 依赖：resnet34 backbone —— 不下载、不改共享代码（局部 monkeypatch）
实测：ckpt 内含全部 595 个 backbone（`image_encoder.*`）键，`agent.initialize()` 会 strict 覆盖加载。
因此 `pretrained=True/False` 只决定"建模那一刻被丢弃的初值"，**对推理结果零影响**。
方案：在 standalone 脚本里对 `timm.create_model` 做**进程内 monkeypatch**，强制 `pretrained=False`、
去掉 `pretrained_cfg_overlay`，只影响本次运行：
```python
import timm
_orig = timm.create_model
def _patched(*a, **k):
    k['pretrained'] = False
    k.pop('pretrained_cfg_overlay', None)
    return _orig(*a, **k)
timm.create_model = _patched          # 在 import/实例化 agent 之前执行
```
→ 无需 `wget`、无需 `bkb_path` 文件、不改 `transfuser_backbone.py`；backbone 权重仍由 ckpt 提供。
（`bkb_path` 在 config 里随便给个占位字符串即可，不会被真正读取。）

## 3. 抽帧（t=1/2/3/4 s → 帧号 10/20/30/39）
两段视频统一用 ffmpeg 抽同样帧号，存到输出目录：
```
outputs/ghosthead_infer/gh_001000__brake/inputs/
  transfered_t1.png ... transfered_t4.png   # 帧 10/20/30/39
  origin_t1.png     ... origin_t4.png
```
（transfered 也可直接用现成的 `renders/.../frames/0010.png` 等，但统一走 ffmpeg 抽帧更一致。）

## 4. 构建模型输入（复刻 feature builder，逐比特一致）
对每张输入帧：
- `camera_feature`：PIL 读 **RGB** → **居中裁 4:1（去天空）**：宽 1600 取高 400，以裁剪中心行
  `crop_center_row`（默认主点 450）取 `rows[center-200 : center+200]` → 1600×400
  → `cv2.resize(crop,(2048,512))`（等比、无变形）→ `transforms.ToTensor()` → `[1,3,512,2048]`
  （复用 `transfuser_features.py::_get_camera_feature` 的 resize+ToTensor 口径；三目拼接换成单目去天空裁剪）。
- `status_feature`：`concat(driving_command[4], ego_velocity[2], ego_acceleration[2])` → `[1,8]`
  - `ego_velocity / ego_acceleration`：从 `scene.json` 每帧 `ego_to_world` 做有限差分（dt=0.1s），
    把世界系位移旋到 ego 系（vx 前向、vy 左向；accel 同理）。取该输入帧对应时刻的值。
  - `driving_command`：默认 **直行 one-hot**（ghosthead 的 brake/none/swerve 不是路口转向）。**这是唯一假设**，
    会在输出里标注；如需按场景 maneuver 调整可后续改。

## 5. 加载 ckpt + 推理
```python
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
cfg = TransfuserConfig(bkb_path='unused_placeholder',   # monkeypatch 后不会被读取
                       plan_anchor_path=f'{DEVKIT}/traj_final/kmeans_navsim_traj_20.npy',
                       latent=True)  # trajectory_sampling: time_horizon=4, interval=0.5 → 8 poses
agent = TransfuserAgent(config=cfg, checkpoint_path='/data/ruolin/uwm/ckpt/diffusiondrive_sim_navhard.ckpt')
agent.initialize(); agent.eval().cuda()
with torch.no_grad():
    out = agent.forward({'camera_feature': cam, 'status_feature': status})
poses = out['trajectory'][0].cpu().numpy()   # (8,3): x前向, y左向(m), heading(rad)，ego 坐标系
```
同时保留 `out['agent_states']`(30,5) 与 `out['agent_labels']`(30) 用于 BEV 叠加检测框。

## 6. PDM 式 BEV 评分 + 可视化（自制，无 token id）
### 6.1 评分（对着 scene.json GT）
坐标：以**输入帧时刻**的 `ego_to_world` 为 ego 原点，把预测 8 点轨迹(ego 系)沿时间前滚；把各 actor 每
未来时刻 `corners_world` 变换到该 ego 系。ego 足迹用 navsim 默认车尺寸(≈4.6×1.9 m)。分项：
- **NoCollision（乘性门）**：任一时刻 ego 足迹与任一 actor box 重叠 → 该项 0（总分归 0）。
- **TTC（乘性门）**：前滚中最小到碰时间 < 阈值(默认 0.95s) → 罚。
- **Comfort（加权）**：预测轨迹的纵/横加速度、jerk、横摆率是否在 navsim 舒适边界内。
- **EgoProgress（加权）**：预测前进距离 / GT ego 同段行驶距离，clip 到 [0,1]。
- 总分 = `NoCollision × TTC × 加权平均(Progress, Comfort)`，输出 0–1。
- 未来超出场景末尾(4s)的 actor 用**末两帧常速外推**补满 horizon；每次推理记录**有效 GT 覆盖秒数**。

### 6.2 BEV 可视化
每次推理画一张俯视 BEV（ego 原点，x 前向朝上，y 左向朝左，matplotlib，网格 ±32 m）：
- **绿线**：预测轨迹 8 点（主产物）；红点=碰撞发生点(若有)。
- **灰/橙框**：`actors` GT box 前滚（occluder/ghost/lead），随时间渐变；ghost 摩托高亮。
- **青框**：`agent_states` 中 `sigmoid(agent_labels)>0.5` 的模型检测框（模型自己看到的它车）。
- 标题标注：**总分 + 各分项**、source、帧号、driving_command 假设、GT 覆盖秒数。
再拼一张 **2×4 总览** `bev_compare.png`：上排 transfered t1..t4，下排 origin t1..t4，每格标分，直接对比
sim 渲染 vs 世界模型生成图下的规划分数差异。

## 7. 输出产物
```
outputs/ghosthead_infer/gh_001000__brake/
  inputs/            transfered_t{1..4}.png, origin_t{1..4}.png   （裁剪后的 4:1 模型输入）
  pred_traj.json     8 次推理的 8×3 轨迹 + status 假设 + 各分项/总分 + GT 覆盖秒数
  scores.csv         8 行：source,frame,total,no_collision,ttc,comfort,progress,gt_coverage_s
  bev/               bev_transfered_t{1..4}.png, bev_origin_t{1..4}.png（每张带分数标题）
  bev_compare.png    2×4 对比总览（每格标分）
```

## 8. 落地方式
- 新增单文件脚本 `scripts/ghosthead_infer/run_ghosthead_infer.py`（含抽帧→构输入→推理→BEV 全流程，参数化
  场景名，方便后面扩量），**不改动任何 navsim 共享代码**。
- 执行顺序：`第1节 env` → `python scripts/ghosthead_infer/run_ghosthead_infer.py`（backbone 走脚本内 monkeypatch，无额外步骤）。

## 9. 已知假设 / 风险（先说清）
1. 单目居中裁 4:1（去天空、等比缩放、无变形）；代价是水平 FOV 仅 ~64.6°，比训练全景窄，无法补出左右视角。
2. `driving_command` 用直行 one-hot（无路由信息）。
3. 世界模型生成视频(origin)与真实相机的色彩/畸变分布 ≠ 训练分布，轨迹是"模型对该图的反应"，非精度基准。
4. **非 navsim 官方 PDM 分**：Town04(CARLA) 无对应 nuPlan 地图、无 metric_cache，官方 `PDMScorer` 不可用；
   这里是基于 scene.json GT 的**自实现 PDM 式分数**（省略 drivable-area），口径对齐但不等同官方数值。
5. 输入越靠后 GT 覆盖越少（t=4s 几乎无未来 GT），碰撞/progress 项以常速外推补足并标注覆盖秒数，
   靠后帧分数参考性下降。
```
```
