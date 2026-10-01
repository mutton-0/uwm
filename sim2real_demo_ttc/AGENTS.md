# AGENTS.md

## 项目

端到端自动驾驶模型在左舵(benchmark)和右舵(deployment)之间的域差研究:表征分析 + RepE 因果注入。依赖 uwm 仓库根目录的 `navsim/` 包。

先读,按优先级:
1. `PRINCIPLES.md`:**优先级高于一切**,包括脚本注释、论文正文和本文件。和它冲突的地方告诉用户,不要自己取舍
2. `README.md`
3. `HANDOFF_2026-09-08.md`(最近一轮的脚本和结论)

## 环境

| 用途 | 解释器 |
|---|---|
| DD / LTF / DDv2 / 大部分分析 | `/home/mut0/.conda/envs/simscale/bin/python`(Python 3.9) |
| SimLingo、Tier-S 流程 | `/data/ruolin/envs/simlingo/bin/python`(Python 3.10) |
| Alpamayo / AutoVLA | 见 `HANDOFF_2026-09-08.md` §1,不要改别人目录的权限 |

GPU:exx 上是 RTX PRO 6000 Blackwell,共享机器,跑之前先 `nvidia-smi` 看有没有被别人占满。

## 常用命令

Tier-S 闭环,7 步**串行**,每一步读上一步的产出,参数都在 `configs/tier_s.yaml`:
```bash
cd sim2real_demo_ttc
PY=/data/ruolin/envs/simlingo/bin/python
$PY scripts/g0_smoke.py && $PY scripts/g1_mine_events.py && $PY scripts/g1_split.py && \
$PY scripts/g1_visualize.py && $PY scripts/g2_cache.py && $PY scripts/g3_metrics.py && $PY scripts/g4_figures.py
```

需要原来的分析结果和缓存时,从研究分支恢复到工作区(不会被提交):
```bash
git restore --source origin/sim2real/demo-v3-repe-mainline --worktree -- \
    sim2real_demo_ttc/results sim2real_demo_ttc/variants sim2real_demo_ttc/probe_v0
```

## 改分析代码时必须遵守的口径(摘自 PRINCIPLES.md,细节以原文为准)

- **P-1**:域按舵位分,左舵 = benchmark,右舵 = deployment。舵位从 `results/driveside_map.json` 取,不要用文件名里的 `nusc` / `navsim` 当域标签
- **P-3 / P-9**:高维方向的任何夹角,都要同时给随机基线和噪声地板,不能裸报角度
- **P-4**:NAVSIM 一律用 `(split, scene)` 复合键;`gt_velocity_3d` 和 `per_obj` 已经是 ego 系,不要再乘旋转;SimLingo 加载时会 `os.chdir`,相对路径要在加载模型前 `.resolve()`
- **P-7**:遮挡填充用 `--mask-mode road_flat`,不加噪声
- **P-8**:C 轴一律 `--all-events` 或 `--limit N`,不用 `--k`(按结果选样)
- **P-10**:修一个 bug 时,把做同一件事的其他代码路径都找出来逐个核对

## 不能做的事

- 不要直接改 `main` 或研究分支 `sim2real/demo-v3-repe-mainline`,不要 push / 开 PR,除非用户明确要求
- 不要提交 `results/`、`variants/` 里的 json / npy / npz / png 等产出,`.gitignore` 已经配好
- 不要改写代码里的绝对路径(174 个文件),除非用户要求。路径清单在 README"本机专属路径"一节
- 不要修改 `PRINCIPLES.md` 里的口径;发现矛盾告诉用户
- 不要在报告或 README 里写没有实际跑出来的数字

## 自检

```bash
python3 <sky-lab-repo-audit 所在目录>/repo_check.py .
```

## 已知的坑

- 右舵危险事件只有个位数到几十个,任何右舵结论都不能拿来排名
- `results_5090/` 从来没进过 git,只在产出它的那台机器上
- `results/` 里 157 份报告没有索引,中英文各一份,先看 `HANDOFF` 里引用的那几份
