# demo/code-review-guidelines

这个分支是给 sky-lab 同学看的规范展示分支。所有链接都是完整网址,可以直接复制发到群里。

## 阅读顺序

每个同学都既是自己仓库的 member,也可能是某个仓库的 manager,所以下面这些都要看。按顺序读,第一次大约 50 分钟。"交给 AI"一栏是读完之后实际干活时可以直接用的命令:

| 顺序 | 读什么 | 看完知道什么 | 交给 AI | 大约 |
|---|---|---|---|---|
| 1 | [仓库规范 REPO_GUIDELINES](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_GUIDELINES.md) 全文 | 仓库要做成什么样、为什么;开头的分工表写了哪些交给 AI、哪些必须你提供 | `/sky-lab-repo-audit 整理` 按这份规范整理仓库 | 15 分钟 |
| 2 | [示例项目 example-camera-calibration](https://github.com/mutton-0/uwm/tree/demo/code-review-guidelines/example-camera-calibration) 的 README、AGENTS.md、.gitignore | 按规范做出来的仓库长什么样 | 照着它的 [AGENTS.md](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/example-camera-calibration/AGENTS.md) 给自己的仓库写一份,或者让 AI 起草 | 5 分钟 |
| 3 | [合入流程 MERGE_WORKFLOW](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/MERGE_WORKFLOW.md) 第零到第五节 | 第一次要配什么、谁负责什么、分支怎么用、每月时间线、怎么开发和提 PR(带图) | 提 PR 前 `/sky-lab-repo-audit pr`,顺便写好 PR 描述和 Sheet 那一行 | 15 分钟 |
| 4 | 合入流程第六到第十一节 | 当 manager 时怎么审、组会怎么过、怎么合并打 tag、紧急修复、常见问题 | manager 审 PR 时 `/sky-lab-repo-audit` 只检查 | 10 分钟 |
| 5 | [检查 skill 使用说明](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/skills/sky-lab-repo-audit/README.md) | 四条命令怎么用、怎么装进仓库、其他 AI 工具怎么用、报告存在哪 | — | 5 分钟 |
| 6 | [逐项检查表 REPO_CHECKLIST](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_CHECKLIST.md) | 扫一眼有哪几类,不用背 | 检查脚本按这张表查,输出里的编号就是表里的编号 | 2 分钟 |
| 选读 | [研究项目例子 sim2real_demo_ttc](https://github.com/mutton-0/uwm/tree/demo/sim2real-ttc-clean/sim2real_demo_ttc)(另一个分支)的 README 和 AGENTS.md | 体量大、结论天天变的研究项目怎么按规范整理 | 它的 [AGENTS.md](https://github.com/mutton-0/uwm/blob/demo/sim2real-ttc-clean/sim2real_demo_ttc/AGENTS.md) 是研究项目写法的参考 | 5 分钟 |

合入流程第十二节是 Owner 的组织设置,同学可以跳过。

用 Codex、Cursor 等其他 AI 工具时,把命令换成一句话:"sky-lab 检查""sky-lab 整理""sky-lab pr""sky-lab 开源"。

**读完先做三件事:**
1. 按 [合入流程第零节](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/MERGE_WORKFLOW.md) 配好自己电脑上的 git(一次就行)
2. 在自己的仓库里输入 `/sky-lab-repo-audit 整理`,回答 AI 最后列出的问题。仓库里还没有 skill 的话,先按 [使用说明的"安装"一节](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/skills/sky-lab-repo-audit/README.md) 放进去
3. 看一遍 `git diff`,自己 commit。要合进 main 时输入 `/sky-lab-repo-audit pr`,把它写好的 PR 描述和 Sheet 那一行复制过去

**Owner** 看:[仓库规范](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_GUIDELINES.md) 全文;[合入流程](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/MERGE_WORKFLOW.md) 第一到第三节(角色和流程)、第七节(组会上怎么审)、第十二节(要做的组织设置:approvers Team、成员权限、组织规则)。组织设置要在 GitHub 网页上操作,这部分没法交给 AI。

## 给 AI 的文档

人不用读这些,AI 会自己找到。列在这里是方便检查和修改:

| 文件 | 作用 |
|---|---|
| [SKILL.md](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/skills/sky-lab-repo-audit/SKILL.md) | AI 的操作说明:四种模式(检查 / 整理 / pr / 开源)的步骤,哪些能直接改、哪些必须先问人、哪些绝对不能做 |
| [repo_check.py](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/skills/sky-lab-repo-audit/repo_check.py) | 只读检查脚本,人和 AI 都能跑:`python3 repo_check.py <仓库目录>` |
| [AGENTS.md 模板](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/AGENTS.md) | 每个仓库根目录放一份。Codex、Cursor、Copilot 等会自动读它;里面"sky-lab 规范检查"一节让这些工具听到"sky-lab 整理"时去读 SKILL.md |
| [CLAUDE.md 模板](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/CLAUDE.md) | 只有一行 `@AGENTS.md`,让 Claude Code 读同一份 AGENTS.md |
| [示例 AGENTS.md(小项目)](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/example-camera-calibration/AGENTS.md)、[示例 AGENTS.md(研究项目)](https://github.com/mutton-0/uwm/blob/demo/sim2real-ttc-clean/sim2real_demo_ttc/AGENTS.md) | 填好的样子 |

## 其他文件

- [templates/monthly_review_sheet.csv](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/monthly_review_sheet.csv):组里 Google Sheet `Sky-Lab_code_review_monthly_updates` 的表头(和现在用的表逐字一致)加几行按新流程填的示例。
- [templates/repo_checklist.csv](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/repo_checklist.csv):检查表的表格版,可以导入 Google Sheet 逐项打勾。
- [templates/.gitignore](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/.gitignore):组里项目通用的 `.gitignore`,建仓库时复制到仓库根目录。上半部分默认生效,只有部分项目用得到的规则在最下面,默认注释掉。这个分支根目录的 [.gitignore](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/.gitignore) 就是它的一份副本。
- [.github/pull_request_template.md](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/.github/pull_request_template.md)、[.github/CODEOWNERS](https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/.github/CODEOWNERS):PR 模板和自动指定 reviewer 的配置,放在真实路径下,直接抄。
- [合入流程里的示意图](https://github.com/mutton-0/uwm/tree/demo/code-review-guidelines/docs/img):开 PR、审核、合并时点哪里。

正式版会放到 `sky-lab-uw` 组织下的 `lab-hub` 和 `.github` 仓库,这里先出个效果给大家看。
