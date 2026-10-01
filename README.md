# demo/code-review-guidelines

这个分支不是 uwm 的项目代码,是给 sky-lab 同学看的规范展示分支,内容只有两块:

- `docs/REPO_GUIDELINES.md` —— 组内仓库规范:每个人的项目要包含什么、README 怎么写、命名和分支怎么定、什么东西不能提交、结果怎么证明能复现、车辆实验的硬件要求。
- `docs/PR_WORKFLOW.md` —— 组织下的仓库怎么提 PR:要不要 fork、分支保护怎么设、CODEOWNERS 和 PR 模板怎么配。
- `.github/pull_request_template.md`、`.github/CODEOWNERS` —— PR_WORKFLOW.md 里提到的模板和自动分配 reviewer 配置的实际样子,放在真实路径下,方便直接抄。
- `example-camera-calibration/` —— 一个干净小项目按规范填出来的例子,照着这个目录结构和 README 写法抄就行。
- `example-sim2real-ttc/` —— 一个真实的、体量大、还在快速迭代的研究分支(uwm 的 `sim2real_demo_ttc/`)按同一套规范补的 README,给"项目不整洁、结论天天变"这种情况打个样:该按规范写的照样写,写不了的地方(比如几十个分析脚本没法逐个列)就说清楚去哪找,而不是硬列全。

正式版会放到 `sky-lab-uw` 组织下的 `lab-hub` 和 `.github` 仓库,这里只是先出个效果给大家看。
