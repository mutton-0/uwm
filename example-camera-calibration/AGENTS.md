# AGENTS.md

## 项目

RealSense D435i 与机械臂末端的手眼标定。产出 `configs/extrinsics.yaml`,给 arm-setup 仓库的抓取流程用。

先读:`README.md`

## 环境

- 解释器:`python3`(Python 3.10)
- GPU:不需要
- 安装:`pip install -r requirements.txt`
- 没接相机、没装 pyrealsense2 的机器上,只能做离线计算,不能采集

## 常用命令

```bash
bash scripts/smoke_test.sh                                   # 改完代码先跑这个,几秒钟
python scripts/run_calib.py --input data/calib_0930/ --output configs/extrinsics.yaml
```

采集(`capture.py`)和计算(`run_calib.py`)按顺序串行执行,没有需要同时开着的进程。

## 目录

| 目录 | 放什么 | 能不能改 |
|---|---|---|
| `scripts/` | 采集、标定、自检脚本 | 可以 |
| `configs/` | 标定结果 | `extrinsics.yaml` 是标定产出,不要手改,也不进 git |
| `data/` | 标定板图像 | 不进 git |
| `outputs/` | 运行时的临时产出(调试图、日志),只保留空目录 | 不进 git |
| `.claude/skills/` | sky-lab 检查 skill。这个演示分支里是指向 `skills/` 的软链接,真实仓库里是复制的一份 | 不要改,规范更新时由 manager 整体替换 |

## 不能做的事

- 不要直接改 `main`,不要 push / 开 PR,除非用户明确要求
- 不要提交 `data/` 下的图像和 `configs/extrinsics.yaml`
- 不要改 README"硬件"一节里的安装位置和线材要求,那是实际测出来的,改之前问用户

## sky-lab 规范检查(所有 AI 工具通用)

用户说下面这些话,或者输入 `/sky-lab-repo-audit <参数>` 时,读 `.claude/skills/sky-lab-repo-audit/SKILL.md`,按里面对应的模式从头做到尾:

| 用户说 | 模式 |
|---|---|
| "sky-lab 检查" | 只检查,不改文件 |
| "sky-lab 整理" | 修复能直接改的问题 |
| "sky-lab pr" | 提 PR 前自查,并起草 PR 描述和 Sheet 那一行 |
| "sky-lab 开源" | 开源前检查 |

用户说要对 main 提 PR 时,先提醒他跑一次 "sky-lab pr"。

检查脚本(只读):`python3 .claude/skills/sky-lab-repo-audit/repo_check.py .`

这个路径读不到时(比如仓库里还没放 skill),去 sky-lab 规范仓库的 `skills/sky-lab-repo-audit/SKILL.md` 读同一份;还找不到就告诉用户,不要自己凭印象检查。

## 已知的坑

- 近距离(<0.3 m)畸变大,采集时标定板离相机远一点
- USB 2.0 口会掉帧,采集必须插 USB 3.0
