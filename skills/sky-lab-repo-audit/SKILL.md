---
name: sky-lab-repo-audit
description: 按 sky-lab 仓库规范检查、修复、整理一个代码仓库或仓库里的子项目。用户说"检查仓库""按规范整理""自查一下""repo audit""提 PR 前检查""manager 审核前过一遍"时使用;对 main 提 PR 之前、月度审核之前也应该用。
---

# sky-lab 仓库规范检查与整理

规范原文:
- 仓库规范 REPO_GUIDELINES:https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_GUIDELINES.md
- 逐项检查表 REPO_CHECKLIST(编号以这里为准):https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_CHECKLIST.md
- 合入流程 MERGE_WORKFLOW:https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/MERGE_WORKFLOW.md
- 通用 .gitignore:https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/.gitignore
- AGENTS.md 模板:https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/AGENTS.md

目标只有一个:**一个陌生人只看 README,在一台新机器上能把代码跑起来、把结果复现出来。**

## 先确定模式

开始前先确认用户要哪一种,没说就用"只检查":

| 模式 | 做什么 |
|---|---|
| 只检查 | 跑脚本 + 读代码,输出报告,**不改任何文件** |
| 修复 | 在只检查的基础上,按下面"可以直接改"的范围动手,其余列成问题问用户 |

修复模式下,先确认当前不在 `main` 上(`git branch --show-current`)。在 main 上就新建一个分支再改,比如 `chore/repo-audit-YYYY-MM`。

## 第一步:跑检查脚本

脚本和本文件在同一个目录,只读,只依赖 Python 3.8+ 和 git:

```bash
python3 <本目录>/repo_check.py <项目目录> --json      # 给你自己读
python3 <本目录>/repo_check.py <项目目录>             # 贴给用户看的版本
python3 <本目录>/repo_check.py <项目目录> --public    # 准备开源(转 public)时
```

- 项目目录可以是仓库根目录,也可以是仓库里的子项目。
- 每条结果有 `id`、`level`(error / warn / info)、`checklist`(检查表编号)、`details`、`fix`。
- 有 error 时退出码是 1。
- `manual_items` 是脚本查不了的项,第三步处理。

## 第二步:处理脚本查出的问题

按 error → warn → info 的顺序处理。

### 可以直接改的(修复模式下)

| id | 怎么改 |
|---|---|
| README_SECTIONS | 按模板补章节标题和内容。内容只能来自代码、配置和已有文档;找不到的写"待确认",**不要编造**版本号、路径、命令或结果数字 |
| README_HARDWARE | 从已有文档、代码里的 device 设置、报告里找 GPU 信息;找不到写"待确认" |
| README_COMMANDS / README_LINKS | **以代码为准改 README**,不要为了对上 README 去改代码 |
| GITIGNORE_MISSING / GITIGNORE_GAPS | 从通用 .gitignore 补规则。改完用 `git check-ignore -v <路径>` 确认没有误伤要保留的文件(代码、配置、README、`data/.gitkeep`) |
| ABS_PATHS(README 里没写的) | 在 README 加"本机专属路径"一节,逐个列出路径前缀、引用次数、是什么。只改 README,不改代码 |
| AGENT_DOC | 用 AGENTS.md 模板写,内容只写能从仓库里核实的;再加一个只有一行 `@AGENTS.md` 的 CLAUDE.md |
| SMOKE_TEST | 写 `scripts/smoke_test.sh`:只检查环境能 import、关键路径存在、能跑一次最小推理,几十秒内跑完。写完要**实际跑一次** |

### 必须先问用户的

| id | 问什么 |
|---|---|
| README_META、CODEOWNERS | manager 和副 manager 是谁(GitHub 用户名)。不要从 commit 作者或路径里猜 |
| LICENSE | 用 MIT 还是 Apache-2.0(基于 Apache-2.0 项目改的一般跟上游) |
| ENV_UNPINNED | 项目实际用的是哪个环境。拿到后用 `pip freeze` / `conda list` 查真实版本再填,不要猜 |
| LARGE_FILES、ARTIFACTS_TRACKED | 把清单给用户,确认后 `git rm --cached`(本地文件保留),补 .gitignore,在 README 写清楚存放位置 |
| SYMLINKS | 确认后删软链接,或改成 README 里说明的路径。**不要删软链接指向的数据** |
| ABS_PATHS(要改代码时) | 默认不改代码。用户同意了,才把路径收进一个配置文件(比如 `configs/paths.yaml`)或环境变量 |
| LAYOUT_ROOT_PY | 移动文件会影响 import 和 README 里的命令,先给方案,用户同意再动 |
| SECRETS、SECRET_FILES | **立即告诉用户**,回复里不要原样贴出密钥。密钥要作废重新生成;代码改成从 `.env` 或环境变量读 |
| HISTORY_LARGE | 只报告。重写 git 历史会影响所有人的 clone,由用户自己决定和执行 |

## 第三步:脚本查不了的项

`manual_items` 里的每一项,读代码和文档来判断。能核实的写结论,核实不了的列成问题问用户,**不要替用户填一个看起来合理的答案**。

- **#13 最小例子能跑通**:能运行就实际运行一次。不能运行(没有 GPU、数据不在本机)就说清楚为什么没跑。
- **#14 多终端**:找 launch 文件、`&`、多个常驻进程(ROS 节点、服务端/客户端)。有的话,README 要写清楚哪些要同时开着、哪些要等上一步。
- **#17 结果复现**:README 里每个结果数字,找到产出它的脚本、参数和随机种子。找不到的标出来。
- **#19–#22、#24 实车项目**:传感器摆放、外参来源、线材、供电支架这些只有做实验的人知道,全部列成问题问用户。代码里的外参、分辨率、采样率和 README 对不上的,把两边的数值都列出来。
- **#23 参数一致**:README 里写的参数名、默认值,和代码里 argparse / 配置文件的定义对一遍。

## 第四步:报告

最后给用户一份报告,包括:

1. 修复前后各跑一次脚本的 error / warn / info 数量
2. 改了哪些文件,每处一句话说为什么
3. 没改的,以及为什么没改
4. 需要用户回答或决定的问题,逐条列出
5. 第三步里每一项的结论

## 绝对不能做的事

- 不 push,不提 PR,不合并;用户明确要求了才做
- 不改 `main`,不 force push,不重写 git 历史(`filter-repo`、`rebase -i`、`reset --hard`)
- 不删除未被 git 跟踪的本地文件(数据、结果、缓存)。对已跟踪的大文件只用 `git rm --cached`
- 不改别人的环境、权限、共享目录
- 不在 README / AGENTS.md 里写没有核实过的版本、路径、命令或结果
- 批量删除或移动之前,先把清单给用户确认;每一步改完都看一眼 `git status`
- 项目里有 `PRINCIPLES.md`、`AGENTS.md` 之类写明优先级的文档时,以它为准;和本规范冲突的地方告诉用户,不要自己取舍
