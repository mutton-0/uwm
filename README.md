# demo/code-review-guidelines

这个分支是给 sky-lab 同学看的规范展示分支。

## 先看哪个

| 你是 | 看这个 |
|---|---|
| 所有人 | [docs/MERGE_WORKFLOW.md](docs/MERGE_WORKFLOW.md):代码怎么合进 main(每月对 main 提 PR、登记 Sheet、manager 和 Owner 都批准),PR 页面点哪里(带图) |
| 要整理自己仓库的人 | [docs/REPO_CHECKLIST.md](docs/REPO_CHECKLIST.md):一张表列出仓库里该有什么、README 该写什么、开源前要做什么 |
| 想知道每条规矩为什么这么定 | [docs/REPO_GUIDELINES.md](docs/REPO_GUIDELINES.md):仓库规范全文 |
| Owner / 管理员 | MERGE_WORKFLOW.md 第十一节:approvers Team、成员权限、组织规则怎么设 |

## 其他文件

- `templates/monthly_review_sheet.csv`:每月 Google Sheet 的表头和示例行,Google Sheet 里"文件 → 导入"就能用。
- `templates/repo_checklist.csv`:REPO_CHECKLIST 的表格版,可以导入 Google Sheet 逐项打勾。
- `templates/.gitignore`:组里项目通用的 `.gitignore`,manager 建仓库时直接复制到仓库根目录。上半部分默认生效,只有部分项目用得到的规则在最下面,默认注释掉。
- `.github/pull_request_template.md`、`.github/CODEOWNERS`:PR 模板和自动指定 reviewer 的配置,放在真实路径下,直接抄。
- `example-camera-calibration/`:一个干净的小项目按规范填出来的例子,照着这个目录结构和 README 写法抄就行。
- 真实项目的例子在单独的分支 [`demo/sim2real-ttc-clean`](https://github.com/mutton-0/uwm/tree/demo/sim2real-ttc-clean/sim2real_demo_ttc):把 uwm 的 `sim2real_demo_ttc/` 按这套规范整理后的样子(补 README、移走 234 MB 分析产出和论文、配好 `.gitignore`),原研究分支没有动。一个体量大、结论天天变的研究项目怎么按规范写,看这个。

正式版会放到 `sky-lab-uw` 组织下的 `lab-hub` 和 `.github` 仓库,这里先出个效果给大家看。
