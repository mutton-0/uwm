# 怎么往 sky-lab 的仓库提 PR

写给 sky-lab-uw 组织下所有仓库用,流程都一样,跟 REPO_GUIDELINES.md 配合看。

## 要不要 fork

不需要。组织里的同学都在自己方向的 Team 里,对应 repo 有 write 权限,直接 clone 原仓库建分支就行,不用像给陌生人的开源项目贡献代码那样 fork 一份出去。

如果发现自己 clone 下来 push 没权限,是没被拉进对应 Team,找方向负责人或导师加一下,不要自己 fork 绕过去——fork 出来的仓库跟组织脱钩,别人搜不到,也不会出现在 lab-hub 的索引里。

## 具体步骤

1. clone 仓库,建分支(命名规则见 REPO_GUIDELINES.md 第三节:`dev/你的名字` 或 `feature/具体描述`):

       git clone git@github.com:sky-lab-uw/camera-calibration.git
       cd camera-calibration
       git checkout -b feature/charuco-support

2. 开发,小步提交,commit message 说清楚改了什么、为什么,不要全是"update"、"fix"这种看不出内容的话。

3. push 分支,开 PR,目标分支是 `main`:

       git push -u origin feature/charuco-support
       gh pr create --base main --title "支持 ChArUco 标定板" --body "..."

   不用 `gh` 命令行的话,push 完网页上会提示 "Compare & pull request",点一下效果一样。

4. PR 描述里过一遍 REPO_GUIDELINES.md 的检查清单,尤其这几条动了的话必须在描述里说明:
   - 改了传感器配置/外参 → README 对应描述有没有同步改
   - 改了依赖版本 → requirements.txt 是不是也更新了,版本钉死了没有
   - 报了新的结果数字 → 是否可复现,不能复现的话波动范围写了没有

5. 等 review。谁负责这个方向就 @ 谁(没配 CODEOWNERS 的话手动指定,见下面怎么配自动分配)。至少一人 approve 才能合并——`main` 分支设了保护规则,不允许直接 push,也不允许没有 approval 就合并。

6. 合并方式统一用 **Squash and merge**:把开发过程中那些"修 typo"、"改回来"的 commit 压成一条,main 的历史保持干净、每条 commit 对应一个完整的改动。合并后删掉分支(GitHub 页面上有按钮,或者 merge 时勾选自动删除)。

## main 分支保护怎么设(给 repo 的 admin/负责人)

每个新建的 repo,在 Settings → Branches 里给 `main` 加规则:

- Require a pull request before merging
- Require approvals(至少 1 人)
- 有 CI 或检查脚本(比如 REPO_GUIDELINES.md 里提到的一致性检查)的话,勾上 Require status checks to pass

这样大家不会手滑直接推到 main,review 也成了硬要求而不是口头约定。

## 自己的 dev 分支不用走 PR

`dev/你的名字` 这种自己独占的分支,自己 push 不需要开 PR 走 review——PR 是为了"进 main 之前有人看一眼",不是什么改动都要走流程。等要合并回 main 了再开。

## 跨方向贡献代码

比如你是做 VLA 的,想往 camera-calibration 提一个修复,流程完全一样:你本来就在组织里,对所有 repo 至少有 read 权限;如果没有 write 权限推不上分支,找 camera 方向的人临时加你进对应 Team,或者直接把改动发给负责人由他本人开分支提交——不建议为了一次性修改专门申请长期权限。

## CODEOWNERS:自动分配 reviewer

在 repo 根目录放一个 `.github/CODEOWNERS`,PR 改到对应路径时 GitHub 会自动请求那个人 review,不用每次手动 @:

```
# 格式:路径 负责人
/configs/        @boyue
/scripts/        @boyue
*                @boyue
```

## PR 模板

在 `sky-lab-uw/.github` 仓库里放一个通用模板(`.github/pull_request_template.md`),组织下所有仓库开 PR 时会自动带出来这几行,不用每次手打:

```markdown
## 这个 PR 做了什么

## 怎么测的

## 检查清单
- [ ] README 和代码改动是否同步(尤其传感器配置/外参/依赖版本)
- [ ] 新增结果是否可复现,或者写清楚了波动范围和原因
- [ ] 没有把数据集/权重/密钥/视频文件带进这次提交
```

有单独仓库需要不一样的模板(比如车辆实验的仓库想额外加一条"传感器摆放是否变化"),可以在该仓库自己的 `.github/pull_request_template.md` 覆盖,组织级模板只是没有自己模板时的默认值。
