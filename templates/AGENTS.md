# AGENTS.md

<!--
给 AI 编程助手(Claude Code、Codex、Cursor、Copilot 等)看的项目须知。
复制到仓库根目录(子项目就放在子项目目录),把 <> 里的内容换成真实信息,用不到的段落删掉。
只写能从仓库里核实的内容。完整说明在 README,这里只写 AI 干活时必须知道的。
Claude Code 读的是 CLAUDE.md,所以旁边再放一个 CLAUDE.md,内容只有一行:@AGENTS.md
-->

## 项目

<一句话:做什么。依赖哪些仓库或包。>

先读:`README.md`<,以及优先级更高的文档,比如 PRINCIPLES.md>

## 环境

- 解释器:`<conda 环境名或解释器路径>`(Python <版本>)
- GPU:<型号、显存;不需要 GPU 就写"CPU 即可">
- 安装:`<安装命令>`

## 常用命令

```bash
bash scripts/smoke_test.sh            # 改完代码先跑这个,几十秒
<最小例子的命令>
<测试命令,没有就删掉这行>
```

<多个终端/进程的:写清楚哪些要同时开着、哪些要等上一步跑完。>

## 目录

| 目录 | 放什么 | 能不能改 |
|---|---|---|
| `src/` 或 `<包名>/` | 代码 | 可以 |
| `scripts/` | 入口脚本 | 可以 |
| `configs/` | 配置 | <外参、标定结果等只有用户要求时才改> |
| `data/`、`outputs/`、`checkpoints/` | 数据和产出,不进 git | 不要提交 |

## 不能做的事

- 不要直接改 `main`,也不要 push / 开 PR,除非用户明确要求。流程:新分支 → PR → manager 和 Owner 审批
- 不要提交数据、权重、日志、视频、`.env`(`.gitignore` 已配好,提交前看一眼 `git status`)
- 不要改写代码里的绝对路径,除非用户要求。本机专属路径列在 README
- <项目特有的禁区,比如:不要改 configs/extrinsics.yaml;不要动 /data/<别人> 下的环境>

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

- <比如:模型加载时会 os.chdir,相对路径要在加载前 resolve()>
