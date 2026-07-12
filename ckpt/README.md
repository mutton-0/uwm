# Checkpoints 下载与放置说明

本目录对应运行时的 ckpt 根目录（本机 `/data/ruolin_a6k/ckpt/`，即分析脚本里的
`CKPT_PATH` 所在目录）。其中 **`diffusiondrive_sim_navhard.ckpt`（分析主用，233M）已通过
Git LFS 随仓库入库**，clone 后 `git lfs pull` 即可取到；其余权重（.ckpt / .bin，几百 MB
起）不入 git，见下方链接，`download_ckpts.sh` 可一键拉取 SimScale 官方 ckpt。

分析脚本默认用的是 **DiffusionDrive**（`diffusiondrive_sim_navhard.ckpt`）+ ResNet34
backbone + `traj_final/kmeans_navsim_traj_20.npy` 轨迹先验。

## A. SimScale 官方 ckpt（本目录内）

来源：🤗 HuggingFace [`OpenDriveLab/SimScale`](https://huggingface.co/datasets/OpenDriveLab/SimScale/tree/main/SimScale_ckpts) ·
👾 ModelScope（国内）[`OpenDriveLab/SimScale`](https://www.modelscope.cn/datasets/OpenDriveLab/SimScale)

| 文件（放到本目录） | 大小 | 模型 | HF resolve 直链 |
|---|---|---|---|
| `diffusiondrive_sim_navhard.ckpt` | 233M | DiffusionDrive / navhard | `.../SimScale_ckpts/DiffusionDrive/diffusiondrive_sim_navhard.ckpt` |
| `gtrs_dense_resnet_sim_expert_navhard.ckpt` | 257M | GTRS-Dense(ResNet34) / navhard | `.../SimScale_ckpts/GTRS_Dense/gtrs_dense_resnet_sim_expert_navhard.ckpt` |
| `ltf_sim_navtest.ckpt` | 215M | LTF / navtest | `.../SimScale_ckpts/LTF/ltf_sim_navtest.ckpt` |

直链前缀均为 `https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main`。
其余变体（`*_navtest` / `*_reward_*` / `gtrs_dense_vov_*`）见仓库根 `README.md` 的下载表。

## B. Backbone 与轨迹先验

| 文件 | 放置路径 | 大小 | 说明 / 来源 |
|---|---|---|---|
| `resnet34_model.bin` | `$BKB_PATH`（本机 `my_dataset/models/`） | 84M | ResNet34 ImageNet 预训练权重的本地副本（timm `resnet34`）。原始 NAVSIM/DiffusionDrive 视觉 backbone。 |
| `kmeans_navsim_traj_20.npy` | 仓库 `traj_final/`（**已随仓库入库**，无需下载） | 2.7K | DiffusionDrive 20 模态轨迹 anchor |
| `8192.npy` / `16384.npy` | 仓库 `traj_final/`（**已入库**） | 3.8M / 7.6M | GTRS 轨迹词表 |

> `resnet34_model.bin` 若缺失：即标准 timm `resnet34` ImageNet-1k 权重，另存为 `.bin`
> 放到 `$BKB_PATH` 即可（配置见 `navsim/.../agent/my_diffusiondrive.yaml`）。

## C. 外部项目 ckpt（本目录内，非 SimScale 提供）

| 文件 | 大小 | 项目 / 来源 |
|---|---|---|
| `ReCogDrive_Diffusion_Planner_2B_RL.ckpt` | 525M | ReCogDrive（小米，ICLR 2026）· HF [`owl10/ReCogDrive-2B-RL`](https://huggingface.co/owl10/ReCogDrive-2B-RL/tree/main) · 代码 [xiaomi-research/recogdrive](https://github.com/xiaomi-research/recogdrive) |
| `ReCogDrive-VLM-2B/`（目录） | — | ReCogDrive VLM · HF [`owl10/ReCogDrive-VLM-2B`](https://huggingface.co/owl10/ReCogDrive-VLM-2B/tree/main) |

## 一键下载

```bash
# 拉取本目录 A 组 SimScale 官方 ckpt（约 0.7 GB）
bash ckpt/download_ckpts.sh            # 默认下载到 ./ckpt/
# 国内可改用 ModelScope，见脚本内 USE_MODELSCOPE 开关
```

放置完成后，运行分析脚本前设置环境变量（见 `scripts/sim2real_analysis/README.md`）：

```bash
export CKPT_PATH=$PWD/ckpt/diffusiondrive_sim_navhard.ckpt
export BKB_PATH=/path/to/my_dataset/models/resnet34_model.bin
```
