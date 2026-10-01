# sim2real_demo_ttc

研究端到端自动驾驶模型(DiffusionDrive / LTF / DiffusionDriveV2 / SimLingo / AutoVLA / Alpamayo)在
benchmark(左舵,Boston/Vegas/Pittsburgh) 与 deployment(右舵,Singapore) 之间的域差:
用表征分析(v_faith / v_decel / 亮度轴等方向)和因果注入(RepE)解释"模型看到危险场景会不会踩刹车",
产出面向 ICRA 投稿的实验结果。

维护人: mut0(主要维护,环境与主线脚本);协作: Zhengyang(Alpamayo/AutoVLA 环境)、ruolin(SimLingo 环境)
状态: active,研究进行中,结论随周迭代,**先看下面"必读"再动手**

## 必读(按顺序)

这个目录不是"装完环境就能跑"的工程代码,是活跃的研究过程,口径经常修正。新加入的人按这个顺序看:

1. `PRINCIPLES.md` —— 口径原则(P-1 ~ P-11)。**这个文件优先于所有脚本注释和历史结论**,里面记录了哪些旧划分/旧判据已经作废,不看这个会直接用错口径。
2. `HANDOFF_2026-09-08.md` —— 最近一轮交接:环境路径、本轮新增脚本、已确认的结论、已否定的假设、没做完的事。
3. `report.md` —— 一次完整跑通的例子(Tier-S 闭环报告),能看到一次分析从挖掘到出数字的全过程。

## 环境

不同模型用不同解释器,别用错:

| 用途 | 解释器 |
|---|---|
| DD / LTF / DDv2 / 大部分分析脚本 | `/home/mut0/.conda/envs/simscale/bin/python`(对应 `environment.yml`,Python 3.9) |
| SimLingo | `/data/ruolin/envs/simlingo/bin/python`(Python 3.10) |
| Alpamayo / AutoVLA | 见 `HANDOFF_2026-09-08.md` §1,venv 的 `bin/python` 目录别人 traverse 不进去,需要按文档里的方法绕过,不要直接改权限 |

主环境安装:
```bash
conda env create -f environment.yml
```

## 怎么跑

以 `report.md` 里跑通的 Tier-S 为例,所有脚本以 `configs/tier_s.yaml` 为唯一参数入口:

```bash
# 挖掘事件(A 类 VRU 突现 / B 类近距 cut-in / C 类 TTC 骤降 / D 类无害负例)
python scripts/<miner>.py --config configs/tier_s.yaml

# 换数据源只改 config 里的 paths.nuscenes_*,不要改脚本
```

其余分析入口按用途分散在 `scripts/` 下几十个脚本里,没有统一的 `main.py`。找脚本的方法:
先看 `HANDOFF_2026-09-08.md` 的"主要脚本"表,或者用文件名前缀猜(`f_*` 是 F 轴/v_faith,
`i_*` 是 I 轴/光照,`c_*` 是 C 轴/遮挡,`g_vs_*` 是行为响应量)。每个脚本的具体参数看它自己的 docstring。

实车部署验证见 `DEPLOY_uw-nuvo.md`(哪些模型能在车机上跑通、显存/延迟实测)。

## 数据 / 权重

| 模型 | 权重位置 |
|---|---|
| DiffusionDrive | `ckpt/diffusiondrive_sim_navhard.ckpt`(随仓库,`ckpt/download_ckpts.sh` 可重新下载) |
| LTF / DiffusionDriveV2 / SimLingo / AutoVLA | 不在本仓库,路径记在 `DEPLOY_uw-nuvo.md`(exx 机器上的绝对路径),按需去搬 |
| nuScenes / NAVSIM 数据集 | 走组内共享存储,不进 git |

`results/` 和 `results_5090/` 下是分析产出的报告(`.md`)和缓存,`cache/`、`frames/` 等按
`.gitignore` 不进 git,可以从原始数据重新生成,具体重建方式写在 `.gitignore` 的注释里。

## 已知问题 / TODO

- RepE 注入实验(`steer_repe.py`)因为机器被占满,进度停滞,是 A/C 两条结论的因果层证据,优先级最高
- AutoVLA 的 C 轴 lead 还没跑完(298/317)
- 右舵(deployment)侧危险事件样本量本身就很小(个位数到几十个),任何右舵结论都不能当排名用,必须带噪声地板说明,见 `PRINCIPLES.md` P-1 / P-9
- `results/` 与 `results_5090/` 内容高度重复(同一批报告在两块 GPU 上各跑了一份),目前没有一个索引说明哪份是当前有效结果,新人容易看错版本 —— 这个需要清理,不属于本次 README 范围但先记在这里

## 分支说明

当前分支 `sim2real/demo-v3-repe-mainline` 是主线开发分支。另有 `sim2real/eva_ckpts_by_pdm`,
从命名看是按 PDM 分数评估 checkpoint 的分支,具体内容未在本文档核实,使用前自己确认。
