# demo/code-review-guidelines

这个分支是给 sky-lab 同学看的规范展示分支。

## 阅读顺序

每个同学都既是自己仓库的 member,也可能是某个仓库的 manager,所以下面这些都要看。按顺序读,第一次大约 50 分钟:

| 顺序 | 读什么 | 看完知道什么 | 大约 |
|---|---|---|---|
| 1 | [docs/REPO_GUIDELINES.md](docs/REPO_GUIDELINES.md) 全文 | 一个仓库要做成什么样,每条规矩为什么这么定 | 15 分钟 |
| 2 | [example-camera-calibration/](example-camera-calibration/) 的 README、AGENTS.md、.gitignore | 按规范做出来的仓库长什么样,自己的仓库照着抄 | 5 分钟 |
| 3 | [docs/MERGE_WORKFLOW.md](docs/MERGE_WORKFLOW.md) 第零到第五节 | 第一次要配什么、谁负责什么、分支怎么用、每月的时间线、怎么开发和提 PR(带图) | 15 分钟 |
| 4 | MERGE_WORKFLOW.md 第六到第十一节 | 当 manager 时怎么审、组会上怎么过、怎么合并打 tag、紧急修复、用 AI 检查、常见问题 | 10 分钟 |
| 5 | [skills/sky-lab-repo-audit/README.md](skills/sky-lab-repo-audit/README.md) | 怎么用检查脚本和 AI 检查自己的仓库,报告存在哪 | 5 分钟 |
| 6 | [docs/REPO_CHECKLIST.md](docs/REPO_CHECKLIST.md) | 扫一眼有哪几类就行,整理仓库时再逐条对照 | 2 分钟 |
| 选读 | [`demo/sim2real-ttc-clean`](https://github.com/mutton-0/uwm/tree/demo/sim2real-ttc-clean/sim2real_demo_ttc) 分支的 README 和 AGENTS.md | 一个体量大、结论天天变的研究项目怎么按规范整理 | 5 分钟 |

MERGE_WORKFLOW.md 第十二节是 Owner 的组织设置,同学可以跳过。

**读完先做三件事:**
1. 按 MERGE_WORKFLOW.md 第零节配好自己电脑上的 git(一次就行)
2. 对自己的仓库跑一次检查:`python3 skills/sky-lab-repo-audit/repo_check.py <你的仓库目录>`
3. 对照输出和 REPO_CHECKLIST 整理仓库,error 清零后按合入流程提 PR

**Owner** 看:REPO_GUIDELINES.md 全文、MERGE_WORKFLOW.md 第一到第三节(角色和流程)、第七节(组会上怎么审)、第十二节(要做的组织设置)。

## 其他文件

- [templates/monthly_review_sheet.csv](templates/monthly_review_sheet.csv):每月 Google Sheet 的表头和示例行,Google Sheet 里"文件 → 导入"就能用。
- [templates/repo_checklist.csv](templates/repo_checklist.csv):REPO_CHECKLIST 的表格版,可以导入 Google Sheet 逐项打勾。
- [templates/AGENTS.md](templates/AGENTS.md)、[templates/CLAUDE.md](templates/CLAUDE.md):给 AI 看的项目须知模板。CLAUDE.md 只有一行 `@AGENTS.md`,让 Claude Code 和其他工具读同一份内容。
- [templates/.gitignore](templates/.gitignore):组里项目通用的 `.gitignore`,manager 建仓库时直接复制到仓库根目录。上半部分默认生效,只有部分项目用得到的规则在最下面,默认注释掉。这个分支根目录的 [.gitignore](.gitignore) 就是它的一份副本。
- [.github/pull_request_template.md](.github/pull_request_template.md)、[.github/CODEOWNERS](.github/CODEOWNERS):PR 模板和自动指定 reviewer 的配置,放在真实路径下,直接抄。
- `example-camera-calibration/`:一个干净的小项目按规范填出来的例子(README、硬件、.gitignore、LICENSE、AGENTS.md、smoke test 都有),`repo_check.py` 跑下来 0 error。照着抄就行。
- 真实项目的例子在单独的分支 [`demo/sim2real-ttc-clean`](https://github.com/mutton-0/uwm/tree/demo/sim2real-ttc-clean/sim2real_demo_ttc):把 uwm 的 `sim2real_demo_ttc/` 按这套规范整理后的样子(补 README、移走 234 MB 分析产出和论文、配好 `.gitignore`),原研究分支没有动。一个体量大、结论天天变的研究项目怎么按规范写,看这个。

正式版会放到 `sky-lab-uw` 组织下的 `lab-hub` 和 `.github` 仓库,这里先出个效果给大家看。
