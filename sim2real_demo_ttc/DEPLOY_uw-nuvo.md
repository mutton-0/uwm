# 实车部署勘察 · uw-Nuvo（暂存，2026-09-14）

目的：验六家端到端策略能否在实车计算单元上**部署运行**。本轮只做到 Alpamayo-1.5 跑通，
其余五家卡在环境未建。过两天继续时从「下一步」那节接上。

## 连接

- 地址：`uw@uw-nuvo.taildc4d2b.ts.net`（Tailscale，tailnet `yrlautoall@`，MagicDNS 名 `taildc4d2b.ts.net`）
- **必须用 MagicDNS 名，不能用 IP `100.66.176.43`**——用 IP 会超时，`tailscale ping` 也一直超时，
  但 TCP 22 是通的（disco ping 与 TCP 走的路径不同，不用管）。
- 本机 `CEE-R018191` 为此切到了 `yrlautoall@` 这个 tailnet；要切回原账号：
  `sudo tailscale switch bwang0522@gmail.com`（profile 都在）。

## 写入约束（共享车机，务必遵守）

- 可写：`/home/uw/120`、`/media/uw/T7 Shield/120models`（后者用户 09-14 批准）
- 只读：`/home/uw` 下其余一切，含 `/home/uw/zhengyang`、`/home/uw/.cache/huggingface`
- 已建的重定向：`/home/uw/120/hf/hub/` 里是指向 `~/.cache/huggingface/hub/models--*` 的软链，
  配 `HF_HOME=/home/uw/120/hf` 使用——**读走共享权重，写入全落在 120**

## 硬件

| | |
|---|---|
| GPU | 1× RTX A6000，48 GB（47.4 GB 可用），驱动 555.42.06 / CUDA 12.5 |
| CPU / 内存 | 16 核 / 31 GB（可用 25 GB） |
| 根分区 | 879 G 已用 814 G，**仅剩 21 G（98%）** —— 环境和权重都不能装根分区 |
| 外挂盘 | `/media/uw/T7 Shield` 1.9 T，**空闲 582 G** |
| 系统 | Ubuntu 22.04.5，kernel 6.8 |

## 已验证：Alpamayo-1.5-10B ✅

车上现成资源（都不用搬）：
- 权重：`~/.cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B`（21 G，5 个 safetensors 分片完整）
- 环境：`/home/uw/zhengyang/alpamayo1.5/a1_5_venv/bin/python`（3.12.13，torch 2.8.0+cu128，**flash-attn 2.8.3 已装**）
- 源码：`/home/uw/zhengyang/alpamayo1.5/src`（`sys.path` 加进去即可）

实测（脚本 `/home/uw/120/deploy_test/t_alpamayo.py`，副本见 `deploy/t_alpamayo.py`）：

| 指标 | 值 |
|---|---|
| 加载 | 34.0 s |
| 显存 | 20.6 GB（峰值 21.6） |
| 推理延迟 | 首次 2.75 s，之后 **1.61 s/帧** |
| 输出 | 64 路点，形状合法；CoC 链正常（`Keep lane since the lane is clear ahead`） |

**两个坑**：
1. `attn_implementation="sdpa"` 会报 `Alpamayo1_5 does not support ... sdpa`（transformers 4.57.1）。
   **不要传这个参数**，用模型默认（flash-attn 2）。
2. 官方样例 clip 要联网下载，offline 下取不到，本轮**喂的是合成帧**——
   所以加载/延迟/显存是真的，**轨迹内容无意义**。要真轨迹得放开下载或用车上 rosbag 图像。

**结论：能加载能出轨迹，但 1.61 s/帧 ≈ 0.6 Hz，离实时（10 Hz）差一个数量级。**
「能部署」与「能实时」是两回事，这点本身值得写进论文的部署讨论。

## 其余五家：权重与环境清单

| 模型 | 权重 | 大小 | 车上状态 |
|---|---|---|---|
| Alpamayo-1.5 | HF 缓存 | 21 G | ✅ 已跑通 |
| DiffusionDrive | `120/uwm/ckpt/diffusiondrive_sim_navhard.ckpt` | 233 M | ✅ 随仓库已在 |
| LTF | `ltf_sim_navtest.ckpt` | 215 M | 搬运中 → T7 |
| DiffusionDriveV2 | `diffusiondrivev2_sel.ckpt` | 364 M | 搬运中 → T7 |
| SimLingo | `pytorch_model.pt` + `.hydra/config.yaml` + InternVL2-1B 底座 | 2.4 G + 底座 | 搬运中 → T7 |
| AutoVLA | `AutoVLA_PDMS_89.ckpt` | **16 G** | 未搬 |

exx 上的源路径（适配器里写死的）：
- `/data/ruolin/uwm/external/ckpts/AutoVLA_PDMS_89.ckpt`
- `/data/ruolin/uwm/external/ckpts/diffusiondrivev2_sel.ckpt`
- `/data/ruolin/uwm/ckpt/ltf_sim_navtest.ckpt`
- `/data/ruolin/simlingo_ckpt/simlingo/{checkpoints/epoch=013.ckpt/pytorch_model.pt,.hydra}`

搬运路线：exx（局域网）→ 本机（`/tmp/.../scratchpad/relay`）→ 实车 T7（Tailscale）。
车与 exx 之间不互通，必须本机中转。

**09-14 已搬到 T7**（`/media/uw/T7 Shield/120models/`）：`ltf_sim_navtest.ckpt` 215 M、
`diffusiondrivev2_sel.ckpt` 364 M、`simlingo/`（含 pytorch_model.pt 与 .hydra）2.4 G、
`models--OpenGVLab--InternVL2-1B` 1.8 G。AutoVLA 的 16 G 未搬。

**两个搬运期的坑**：
1. **T7 是 exFAT，不支持软链**。HF 缓存目录靠 `snapshots/ → ../../blobs/` 软链组织，
   rsync 到 T7 会报 `symlink ... Operation not permitted`，InternVL 那份是这么进去的、
   软链没建成。用 `rsync -L`（把软链解成实体文件）重传，或者干脆放 ext4 的 `/home/uw/120`。
2. **Tailscale 链路只有 0.6–1.3 MB/s**。2.4 G 花了约一小时；**AutoVLA 16 G 预计 5–6 小时**。
   要么提前挂后台，要么找个 U 盘直接拷。

## 下一步（从这里接上）

1. **建环境**。`/home/uw/miniconda3` 有 conda 但没有我们的环境；根分区放不下，必须重定向：
   `CONDA_ENVS_DIRS` / `CONDA_PKGS_DIRS` / `PIP_CACHE_DIR` / `HF_HOME` / `TORCH_HOME` / `XDG_CACHE_HOME`
   全部指到 `/media/uw/T7 Shield/120models` 或 `/home/uw/120` 下。**不要动 `/home/uw/miniconda3` 本体**。
   驱动是 CUDA 12.5，torch 选 cu121/cu124 轮子。
2. **按用户 09-14 的选择走「最小验证」路线**：每家只验「能加载 + 出一条轨迹 + 延迟与显存」，
   不搬 navsim 数据管线（那是几百 G）。每家照 `t_alpamayo.py` 的模板写一个 `t_<model>.py`。
3. 顺序建议：LTF → DiffusionDrive → DDv2（要 LiDAR 输入）→ SimLingo → AutoVLA（16 G 权重最后搬）。
4. 真实输入：车上有 rosbag（`/home/uw/rosbag2_*`，14–27 G）和 `live_bags`，可取实拍前视帧替掉合成帧。

## 与论文的关系

部署侧的数（加载时间、显存、延迟）不在现有六项检查里，属于**新增的一维**。
若要写进论文，最自然的位置是 §IX discussion 的 future work，或作为「体检之外还需要什么」的一句话。
**注意不要与现有结论混着说**：现有结论全部基于离线回放，与实车实时性无关。
