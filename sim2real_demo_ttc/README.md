# sim2real_demo_ttc

研究端到端自动驾驶模型(DiffusionDrive / LTF / DiffusionDriveV2 / SimLingo / AutoVLA / Alpamayo)在 benchmark(左舵:Boston / Las Vegas / Pittsburgh)和 deployment(右舵:Singapore)之间的域差。方法是表征分析(v_faith / v_decel / 亮度轴等方向)加因果注入(RepE),回答"模型看到危险场景会不会踩刹车",结果用于 ICRA 投稿。

依赖本仓库根目录的 `navsim/` 包,不能脱离 uwm 仓库单独运行。

维护人(manager):待确认
状态:active,研究进行中,结论按周迭代

> 这是 `demo/sim2real-ttc-clean` 分支:按 sky-lab 仓库规范整理后的样子。原始研究分支是 `sim2real/demo-v3-repe-mainline`,没有改动。整理了什么,见最后一节。

## 必读(按顺序)

这里不是"装完环境就能跑"的工程代码,是还在进行的研究,口径经常修正。新加入的人按这个顺序看:

1. `PRINCIPLES.md`:口径原则(P-1 ~ P-11)。**优先级高于所有脚本注释和历史结论**,记录了哪些旧划分、旧判据已经作废。不看这个会直接用错口径。
2. `HANDOFF_2026-09-08.md`:最近一轮交接,包括环境路径、新增脚本、确认的结论、被否定的假设、没做完的事。
3. `report.md`:一次完整跑通的例子(Tier-S 闭环报告),能看到从挖掘到出数字的全过程。
4. `DEPLOY_uw-nuvo.md`:实车计算单元上的部署勘察(哪些模型能跑、显存和延迟)。

## 环境

不同模型用不同的解释器,别用错:

| 用途 | 解释器 | Python |
|---|---|---|
| DD / LTF / DDv2 / 大部分分析脚本 | `/home/mut0/.conda/envs/simscale/bin/python`(对应仓库根目录 `environment.yml`) | 3.9 |
| SimLingo,以及下面的 Tier-S 流程 | `/data/ruolin/envs/simlingo/bin/python` | 3.10 |
| Alpamayo / AutoVLA | 见 `HANDOFF_2026-09-08.md` §1。venv 的 `bin/python` 指向别人的私有目录,要按文档里的办法绕开,不要去改别人的权限 | 3.12 |

硬件:

| 机器 | GPU | 备注 |
|---|---|---|
| exx(主力分析机) | NVIDIA RTX PRO 6000 Blackwell(sm_120) | SimLingo 官方 pin 的 torch 2.2 + flash-attn 在这张卡上用不了,改用 torch 2.8.0+cu128,见 `report.md` §1 |
| uw-nuvo(实车) | RTX A6000 48 GB,驱动 555.42.06 / CUDA 12.5 | 根分区只剩约 21 GB,环境和权重放外挂盘,见 `DEPLOY_uw-nuvo.md` |

主环境安装(在 uwm 仓库根目录):
```bash
conda env create -f environment.yml
```

## 怎么跑

以 `report.md` 里跑通的 Tier-S 为例。7 个脚本**必须按顺序串行跑**,每一步读上一步的产出。所有参数都在 `configs/tier_s.yaml`,换数据只改里面的 `paths.nuscenes_*`,不要改脚本。

```bash
cd sim2real_demo_ttc
PY=/data/ruolin/envs/simlingo/bin/python

$PY scripts/g0_smoke.py         # G0 冒烟:模型能加载、能抓到每层 hidden、两次运行逐位一致
$PY scripts/g1_mine_events.py   # G1 从 nuScenes 挖 TTC 突变事件(只用 CPU)
$PY scripts/g1_split.py         # G1 按 scene 划分估计集 / 真值集
$PY scripts/g1_visualize.py     # G1 随机抽事件画图,人工抽检
$PY scripts/g2_cache.py         # G2 批量前向,缓存各层激活
$PY scripts/g3_metrics.py       # G3 指标 + G4 一致性验收
$PY scripts/g4_figures.py       # G4 出图
```

跑完的数字应该和 `report.md` §2 ~ §5 对得上。

其他分析入口分散在 `scripts/` 下的一百多个脚本里,没有统一的 `main.py`。找脚本:先看 `HANDOFF_2026-09-08.md` 的"主要脚本"表,或者按文件名前缀找(`f_*` 是 F 轴 / v_faith,`i_*` 是 I 轴 / 光照,`c_*` 是 C 轴 / 遮挡,`g_vs_*` 是行为响应量)。具体参数看各脚本的 docstring。

## 数据和中间结果

**这个分支里没有分析产出和缓存**(`results/`、`variants/`、`probe_v0/` 下的 json / jsonl / npy / npz / png / pdf / pt,共约 234 MB)。它们还在原分支上。需要时在 uwm 仓库根目录运行下面的命令,把文件恢复到本地;`.gitignore` 已经配好,恢复出来的文件不会被误提交:
```bash
git fetch origin
git restore --source origin/sim2real/demo-v3-repe-mainline --worktree -- \
    sim2real_demo_ttc/results sim2real_demo_ttc/variants sim2real_demo_ttc/probe_v0
```

长期应该把这些文件迁到组内 NAS(代码里已经有 `/mnt/skylabNAS` 的引用),再把存放路径写在这里。

| 东西 | 在哪 |
|---|---|
| nuScenes / NAVSIM 数据集 | 组内共享存储(代码里是 `/data/dataset/...`) |
| DiffusionDrive 权重 | 仓库根目录 `ckpt/diffusiondrive_sim_navhard.ckpt`(Git LFS) |
| LTF / DDv2 / SimLingo / AutoVLA 权重 | 不在仓库里,路径见 `DEPLOY_uw-nuvo.md` |
| 已有的分析结论(文字) | `results/*.md`,这个分支里保留了 |
| 论文源文件和 PDF | 原分支的 `paper_*/`,这个分支里移走了 |

## 本机专属路径(换机器前必看)

代码里还有 174 个文件写死了绝对路径。这个分支**没有改**这些路径:一是改完没法验证还能不能跑,二是原分支还在用。按根目录归类:

| 路径前缀 | 引用次数 | 是什么 |
|---|---|---|
| `/data/ruolin/...` | 约 240 | exx 上的 SimLingo 仓库、环境、部分权重 |
| `/data/dataset/...` | 约 92 | nuScenes / NAVSIM 数据集 |
| `/home/boyuewang/...` | 约 6 | 个人机器上的临时路径 |
| `/data/Zhengyang/...` | 约 22 | Alpamayo / AutoVLA 的环境 |
| `/mnt/skylabNAS/...` | 约 8 | 组内 NAS |
| `/home/uw/...` | 约 4 | 实车 uw-nuvo |

换机器时,先在代码里搜这几个前缀,改成自己机器上的路径。长期的做法是把这些根目录收进一个配置文件(比如 `configs/paths.yaml`)或者环境变量,脚本统一从那里读。这一项记在下面的已知问题里。

## 已知问题 / TODO

- **174 个文件写死了绝对路径**,见上一节。优先收成一个路径配置
- RepE 注入实验(`steer_repe.py`)因为机器被占满,进度停滞。它是 A / C 两条结论的因果层证据,优先补
- AutoVLA 的 C 轴 lead 还没跑完(298/317)
- 右舵(deployment)一侧的危险事件样本只有个位数到几十个,任何右舵结论都不能拿来排名,必须带噪声地板说明,见 `PRINCIPLES.md` P-1 / P-9
- `results/` 里有 157 份报告,中英文各一份、还有 `*_DONE.md`,没有索引说明哪份是当前有效的结论。建议加一个 `results/INDEX.md`
- 第二轮结果 `results_5090/` 是在另一台机器上产出的,从来没进过 git

## 这个分支整理了什么

对照 [sky-lab 仓库整理表](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_CHECKLIST.md):

| 改动 | 原因 |
|---|---|
| 新增这份 README | 原来没有 README,只有交接、原则、报告几份文档,新人不知道从哪看起 |
| 移走 `results/`、`variants/`、`probe_v0/` 里的分析产出和缓存(约 234 MB) | 规范:数据、结果缓存、图片不进 git |
| 移走 5 个 `paper_*/` 目录(论文 tex、PDF、pptx,约 15 MB) | 论文不放在代码仓库里 |
| 移走 `results/voided_*/` | `PRINCIPLES.md` 里已经写明作废 |
| 移走 `probe_v0/data`、`probe_v0/tb` 两个软链接 | 指向没进 git 的 `results_5090/`,别人 clone 下来是断的 |
| `.gitignore` 追加规则 | 防止上面这些文件以后又被提交回来 |

整理后:1507 个文件、约 254 MB → 440 个文件、约 4.8 MB(含这份 README)。

**没有动的**:代码(`scripts/`、`configs/`、`deploy/`、`remote_pkg/`、`results/` 里的适配器 `.py`)、所有说明文档、`mining/` 和 `variants/` 里的小文件(场景列表、划分)。代码里的绝对路径也没改,见上面。
