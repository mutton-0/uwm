# 代码合入流程(成员手册)

**main 每个月只动一次。** 平时大家在自己的分支上开发;每月组会前,各 members 把这个月能合的内容整理好提 PR、把链接贴进 Google Sheet;周一组会上 Owner 审核通过,才合进 main。

仓库里该放什么、README 怎么写,见 [REPO_GUIDELINES.md](REPO_GUIDELINES.md);提 PR 前的逐项自查,见 [REPO_CHECKLIST.md](REPO_CHECKLIST.md)。

> 文中的图是示意图。按钮文字和位置跟 GitHub 页面一致,其他细节做了简化。

## 一、谁负责什么

| 角色 | 是谁 | GitHub 上的权限 | 做什么 |
|---|---|---|---|
| Owner | 导师 / 总负责人 | 组织 Owner | 组会上审批所有合进 main 的 PR |
| Manager | 每个仓库一个人,写在 README 顶部"维护人"和 Google Sheet 里 | 该仓库 Maintain | 维护本月整合分支,审成员的 PR |
| Member | 其他同学 | 所在方向的 members 给 Write | 在自己的分支开发,向本月整合分支提 PR，review 前提月度 PR 并登记 |

一个人可以是 A 仓库的 manager,同时是 B 仓库的 member。

## 二、分支怎么分

![分支模型](img/00-branch-model.svg)

| 分支 / tag | 谁建 | 用途 | 规则 |
|---|---|---|---|.| `main` | 建仓库时自动有 | 经过 Owner 审核的稳定版本 | 受保护:不能直接 push,只能通过 PR 合入,必须 Owner 批准 |
| `base-YYYY-MM`(tag) | manager | 每次月度合入后在 main 上打的标记,是下个月所有人开发的起点 | 打了就不再改 |
| `monthly/YYYY-MM` | manager | 本月要合进 main 的内容先汇总到这里。月份按"在哪个月的组会上合入"写 | 成员不要直接往里推,走 PR |
| `feature/具体描述`、`dev/你的名字` | 成员自己 | 日常开发 | 自己随便推。一个功能一个分支更好管 |

## 三、第一次:建立仓库的 base(每个仓库只做一次)

1. **Owner/manager 建仓库。** 建的时候勾上 **Add README**,这样 main 一开始就存在,后面才能对它开 PR。在仓库 **Settings → Collaborators and teams** 里把 manager 设为 **Maintain**,members 设为 **Write**。管理员的完整设置见第十节。
2. **Manager 整理代码。** 对照 [REPO_CHECKLIST.md](REPO_CHECKLIST.md),把"每个仓库"那几项过一遍。
3. **推到初始化分支**(main 已受保护,不能直接推):
   ```bash
   git clone git@github.com:sky-lab-uw/camera-calibration.git
   cd camera-calibration
   git checkout -b init/2026-10
   # 把整理好的代码拷进来,然后:
   git status                       # 先看一眼要提交哪些文件
   git add README.md requirements.txt configs/ scripts/ src/ .gitignore LICENSE
   git commit -m "init: camera-calibration 2026-10 base"
   git push -u origin init/2026-10
   ```
4. **开 PR:`init/2026-10` → `main`**(怎么开见第四节第 3 步),把链接贴进 Google Sheet。
5. **组会上 Owner 批准、合入后**,manager 打第一个 base tag:
   ```bash
   git checkout main
   git pull
   git tag -a base-2026-10 -m "camera-calibration 2026-10 组会后的 base"
   git push origin base-2026-10
   ```

之后每个月按第四到第七节循环。

## 四、成员:平时怎么开发

### 1. 从最新的 main 拉自己的分支

命令行:
```bash
git fetch origin
git checkout -b feature/charuco-support origin/main
```

网页:

![网页上新建分支](img/01-create-branch.svg)

### 2. 改代码、提交、推到自己的分支

```bash
git status
git add scripts/run_calib.py README.md        # 只加你改过的文件,别用 git add .
git commit -m "run_calib 支持 ChArUco 标定板"
git push -u origin feature/charuco-support    # 第一次推加 -u,之后直接 git push
```

在网页上直接改文件的话,点 **Commit changes** 后会弹出这个框:

![网页提交时的选项](img/02-commit-dialog.svg)

- 在 **main** 上点的编辑:第一项是灰的,选第二项新建分支。
- 已经切到**自己的分支**再编辑:第一项会变成 "Commit directly to the feature/xxx branch",直接选它。

### 3. 想让改动进本月的 main:提 PR 到本月整合分支

截止时间是**组会前那一周的周五**,之后提的进下个月。

push 完回到仓库首页,点黄色提示条上的按钮:

![Compare & pull request](img/03-compare-banner.svg)

进入开 PR 的页面。**注意 base 要改**:默认是 main,成员要改成本月的 `monthly/YYYY-MM`。

![开 PR 页面](img/04-open-pr.svg)

装了 GitHub CLI(`gh`)的话,命令行也可以:
```bash
gh pr create --base monthly/2026-11 --title "run_calib 支持 ChArUco 标定板" --reviewer boyue
```

### 4. manager 提了修改意见

在**同一个分支**上接着改、接着 push,PR 会自动更新,不用关掉重开。改完在 PR 里回复一句,再点右侧 Reviewers 里 manager 名字旁边的刷新图标,重新请求 review。

### 5. 没赶上,或者这个月不打算合

留在自己分支上,下个月再提。组会前跟 manager 说一声,他会在 Google Sheet 的备注里写上。

### 6. 每月合入之后:把新的 main 同步到自己的分支

```bash
git fetch origin
git checkout feature/charuco-support
git merge origin/main          # 有冲突就按提示改完,再 git add、git commit
git push
```
已经合进 main 的分支可以删掉,下一个功能从新的 main 重新拉。

## 五、Manager:每月要做的事

### 月初(上次组会后)

从 main 拉出本月的整合分支,群里通知大家:
```bash
git fetch origin
git checkout -b monthly/2026-11 origin/main
git push -u origin monthly/2026-11
```

### 月中:审成员的 PR

成员的 PR 会自动请求你 review(仓库的 `.github/CODEOWNERS` 里写的是你)。审核的操作跟 Owner 审 main 一样,见第六节的图。通过后合并到 monthly 分支,合并方式可以选 **Squash and merge**,把成员零碎的 commit 压成一条。

### 组会前的周末:提月度 PR,登记 Google Sheet

1. 切到 monthly 分支,对照 [REPO_CHECKLIST.md](REPO_CHECKLIST.md) 自查。最好开一个新的 conda 环境,按 README 从头跑一遍。
2. 开 PR:**base 选 `main`,compare 选 `monthly/2026-11`**。还是第四节那张图,只是 base 不一样。
   - 标题:`[2026-11 月度合入] camera-calibration`
   - 描述:本月合入了哪几个 PR(贴链接),各自一句话说明做了什么,还有遗留问题
   - Reviewer:组织规则会自动要求 `approvers`(Owner)审批,不用手动选
3. 把 PR 链接填进 Google Sheet 本月那一行。表头见 [templates/monthly_review_sheet.csv](../templates/monthly_review_sheet.csv)。
4. **本月没东西可合,也要在 Sheet 里填一行**,写"本月无合入"和原因或进度。每个仓库每个月都有记录,才看得出哪个仓库落灰了。

## 六、组会上:Owner 怎么审

按 Google Sheet 的顺序逐个打开 PR:

![审核 PR](img/05-review-approve.svg)

- 没问题:**Approve**
- 要改:**Request changes**,写清楚改什么。manager 当周改完再请求审核;或者当场决定下个月再合,在 Sheet 里记下

审完在 Sheet 的"Owner 审核结果"一栏填:通过 / 打回 / 下月再合。

## 七、合并和收尾(manager)

Owner Approve 之后,manager 在 PR 页面底部合并。**月度合入选 Create a merge commit**,保留这个月所有的 commit:

![合并 PR](img/06-merge-box.svg)

合并后:
1. 打 tag:
   ```bash
   git checkout main
   git pull
   git tag -a base-2026-11 -m "camera-calibration 2026-11 组会后的 base"
   git push origin base-2026-11
   ```
2. 在 Sheet 里补上 tag 名。
3. 从新的 main 拉下个月的 `monthly/2026-12`(见第五节"月初")。
4. 群里通知:"camera-calibration 11 月已合入,`git fetch` 后把 origin/main 合进自己的分支"。
5. 已合并的成员分支和上个月的 monthly 分支可以删掉(PR 合并后页面上有 **Delete branch** 按钮)。tag 还在,随时能回到当时的版本。

## 八、紧急修复

README 写错导致别人跑不起来、明显的 bug,这类等不了一个月的:manager 从 main 拉 `hotfix/具体描述`,改完直接开 PR 到 main,单独找 Owner 审批,在 Sheet 里加一行备注。不要攒到月度合入里。

## 九、常见问题

| 现象 | 原因 / 怎么办 |
|---|---|
| `git push origin main` 报 `GH013: Repository rule violations` | 正常,main 不允许直接推。推到自己的分支再开 PR |
| PR 一直显示 Review required,Merge 按钮是灰的 | 还没有 Owner Approve。月度 PR 要等组会 |
| 有人 Approve 了还是不能合 | 点 Approve 的人不在 `approvers` 里,或者对仓库没有 Write 权限,不计数 |
| Approve 之后又推了新 commit,批准没了 | 规则设置了"有新推送,旧批准作废",需要重新审 |
| 没看到黄色的 Compare & pull request 提示 | 点 **Pull requests → New pull request**,手动选 base 和 compare |
| PR 提示有冲突(This branch has conflicts) | 在自己分支上 `git merge origin/<base 分支>`,本地解决冲突后 push |
| 推自己的分支报 403 | 你对这个仓库只有 Read,没被加进对应 Team。找 manager 或 Owner |

## 十、管理员一次性设置(Owner 看)

成员不用看这一节。这些设置做一次,以后新建的仓库自动生效。

**1. Team**(`https://github.com/orgs/sky-lab-uw/teams`)
- `approvers`:Owner 们。月度 PR 必须由这个 Team 的人批准
- 各方向 Team:`camera`、`vla`、`robot` ……

**2. 成员权限**(组织 **Settings → Member privileges**)
- Base permissions:**Read**
- Repository creation:只勾 Private,或者都不勾(只由 Owner 建仓库)
- Repository deletion and transfer:**不勾**
- Repository visibility change:**不勾**

**3. 组织级规则**(`https://github.com/organizations/sky-lab-uw/settings/rules` → New branch ruleset)
- Enforcement status:Active;Bypass list:留空
- Target repositories:**All repositories**;Target branches:**Include default branch**
- 勾 Restrict deletions、Block force pushes
- 勾 Require a pull request before merging,展开后:
  - Required approvals:1
  - 勾 Dismiss stale pull request approvals when new commits are pushed
  - 勾 Require approval of the most recent reviewable push
  - Required reviewers:Team 选 `approvers`,文件匹配填 `*`
  - Allowed merge methods:Merge 和 Squash 都勾上
- 界面上没有 Required reviewers 的话,改勾 **Require review from Code Owners**,并在每个仓库的 `.github/CODEOWNERS` 里写 `* @sky-lab-uw/approvers`

**4. 每次新建仓库**
- 勾 Add README(或者用模板仓库建)
- Collaborators and teams:manager 给 Maintain,方向 Team 给 Write;`approvers` 里如果有不是组织 Owner 的人,`approvers` 也给 Write,否则他们的 Approve 不计数
- 仓库是成员自己建的,把他从 Admin 降成 Maintain 或 Write
- `.github/CODEOWNERS` 写 `* @manager 的用户名`,成员的 PR 会自动请求 manager review

**5. 注意**
- 组织级 Rulesets 和 private 仓库的分支保护需要 GitHub Team 计划。学校组织可以通过 GitHub Education 免费升级;免费组织只能对 public 仓库逐个设置。
- 仓库转 public 之前,把 REPO_CHECKLIST 里"开源前"那几项全部过一遍。
