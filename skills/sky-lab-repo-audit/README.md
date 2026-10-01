# 用 AI 检查和整理仓库(使用说明)

这个目录是组里的仓库检查工具,人和 AI 都能用。这份是给人看的使用说明;同目录的 `SKILL.md` 是写给 AI 的,一般不用读。

| 文件 | 给谁 | 是什么 |
|---|---|---|
| `README.md` | 人 | 这份说明 |
| `repo_check.py` | 人和 AI | 检查脚本。只读,不改任何文件,只依赖 Python 3.8+ 和 git |
| `SKILL.md` | AI | 告诉 AI 怎么检查、哪些问题可以直接改、哪些必须先问你、哪些事绝对不能做 |

检查的标准就是 [REPO_CHECKLIST.md](../../docs/REPO_CHECKLIST.md),输出里的编号对应那张表。

## 什么时候用

| 时候 | 谁 | 怎么用 |
|---|---|---|
| 对 main 提 PR 之前 | member | 跑一次脚本,把 error 修掉再提 |
| 组会前审核 PR | manager | 在 PR 分支上跑一次,error 没清零的打回 |
| 新建仓库、接手别人的仓库 | manager / 接手的人 | 让 AI 按"修复模式"整理一遍 |
| 准备开源(转 public)之前 | manager | 加 `--public` 再跑一次 |

## 方式一:不用 AI,直接跑脚本

```bash
python3 <本目录>/repo_check.py <项目目录>            # 平时
python3 <本目录>/repo_check.py <项目目录> --public   # 开源前,多查英文 README、CONTRIBUTING、git 历史里的大文件
python3 <本目录>/repo_check.py <项目目录> --json     # 输出 JSON,给 AI 或脚本用
```

项目目录可以是仓库根目录,也可以是仓库里的子项目(比如 `uwm/sim2real_demo_ttc`)。

**怎么看输出**:

```
[✗ error] #29 CODEOWNERS: 仓库没有 .github/CODEOWNERS(main 需要 manager 审批)
      → 写一行:*  @manager用户名 @副manager用户名。用户名问用户,不要猜
[! warn ] #27 PERSONAL_INFO: 有 4 处个人 / 网络信息(邮箱、内网 IP、Tailscale 主机名)
      DEPLOY_uw-nuvo.md:8(邮箱)
      DEPLOY_uw-nuvo.md:9(内网 / Tailscale IP)
      → 告诉用户位置和类型,回复里不要贴出原值。删除还是改成占位符由用户决定
[· info ] #6 SMOKE_TEST: 没有一键自检脚本(scripts/smoke_test.sh)
```

- 方括号里是级别:**error 必须修**,提 PR 前要清零;**warn 很可能有问题**,修不了的在 PR 描述里说明原因;**info 是建议**
- `#29` 是检查表编号,去 REPO_CHECKLIST 里能找到完整要求
- `→` 后面是怎么修
- 最后一段"需要人或 AI 读代码判断的项",是脚本查不了的(能不能真的跑通、结果能否复现、实车硬件等),要自己对照着过一遍
- 有 error 时退出码是 1

## 方式二:Claude Code

**安装一次,之后所有仓库都能用。** 推荐用软链接,规范更新后 `git pull` 一下就同步了:

```bash
# 在 sky-lab 规范仓库的本地副本里(现在是 uwm 的 demo/code-review-guidelines 分支)
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/sky-lab-repo-audit" ~/.claude/skills/sky-lab-repo-audit
```

不想用软链接,就直接复制 `cp -r skills/sky-lab-repo-audit ~/.claude/skills/`,规范更新后要重新复制一次。

**怎么说**:在要检查的仓库里打开 Claude Code,直接说下面这些话,它会自动用这个 skill。也可以输入 `/sky-lab-repo-audit` 直接调用。

| 你想要 | 这么说 |
|---|---|
| 只看问题,不改文件 | "按 sky-lab 规范检查一下这个仓库,只检查不修改" |
| 让它整理 | "按 sky-lab 规范整理这个仓库,能直接改的改掉,其他的列成问题问我" |
| 提 PR 前自查 | "我要对 main 提 PR 了,按 sky-lab 规范帮我自查一下这次的改动" |
| 检查子项目 | "按 sky-lab 规范检查 sim2real_demo_ttc 这个目录" |
| 开源前 | "准备把这个仓库开源,按 sky-lab 规范做开源前检查" |

## 方式三:其他 AI 工具(Codex、Cursor、Copilot 等)

把 SKILL.md 的位置告诉它:

> 读 `<本目录>/SKILL.md`,按里面的步骤检查当前仓库,先只检查不修改,最后按第四步的格式给我报告。

如果这个工具不能执行命令,就自己跑 `repo_check.py --json`,把输出贴给它,再让它按 SKILL.md 的第二、三步处理。

## AI 会做什么、不会做什么

**会做**(修复模式下):补 README 缺的章节、补 `.gitignore`、在 README 里列出写死的绝对路径、写 `AGENTS.md`、写 smoke test。内容只来自你的代码和已有文档,找不到的会写"待确认"。如果你在 main 上,它会先新建一个分支再改。

**会先问你**:manager 是谁、用什么 LICENSE、实际用的哪个环境、大文件要不要移出 git、断掉的软链接怎么处理、要不要改代码里的绝对路径、个人信息要不要删、实车硬件信息(摆放、外参、线材)。

**不会做**:push、开 PR、改 main、重写 git 历史、删除没被 git 跟踪的本地数据、改别人的环境和权限。

## AI 改完之后,你要做的

1. 回答它列出来的问题
2. 看改动:`git status`、`git diff`。重点看 README 里有没有写错的版本号、路径、命令
3. 再跑一次 `repo_check.py`,确认 error 清零
4. 自己 commit,按 [合入流程](../../docs/MERGE_WORKFLOW.md) 提 PR

## 给自己的仓库写 AGENTS.md

AI 打开一个仓库,第一件事是找项目须知。写一份能少很多返工:

1. 把 [templates/AGENTS.md](../../templates/AGENTS.md) 复制到仓库根目录(子项目就放在子项目目录),按里面的提示填
2. 旁边放一个 `CLAUDE.md`,内容只有一行 `@AGENTS.md`(Claude Code 读 CLAUDE.md,其他工具读 AGENTS.md,这样两边是同一份)
3. 重点写:用哪个环境、怎么自检、哪些文件不能动、项目特有的坑

参考:
- 小项目:[example-camera-calibration/AGENTS.md](../../example-camera-calibration/AGENTS.md)
- 研究项目:`demo/sim2real-ttc-clean` 分支的 [sim2real_demo_ttc/AGENTS.md](https://github.com/mutton-0/uwm/blob/demo/sim2real-ttc-clean/sim2real_demo_ttc/AGENTS.md),把 PRINCIPLES.md 里改代码时必须遵守的口径写成了约束

也可以让 AI 帮你起草:"按 sky-lab 规范给这个仓库写一份 AGENTS.md",写完自己核对一遍。

## 常见问题

| 情况 | 怎么办 |
|---|---|
| 子项目没有自己的 LICENSE / CODEOWNERS,但仓库根目录有 | 算通过,脚本会往上找到 git 根目录 |
| 我觉得某条 warn 是误报 | 在 PR 描述里说明为什么。如果是脚本的规则有问题,告诉维护这个工具的人 |
| 新加的文件还是报"没有" | 脚本只看被 git 跟踪的文件(别人 clone 下来能拿到的),先 `git add` |
| `ENV_UNPINNED` 一大堆,来自上游的 requirements | 在 PR 描述里说明;能钉的话,在实际环境里 `pip freeze` 查真实版本再填,不要猜 |
| 报 `PERSONAL_INFO` | 文件里有邮箱、内网 IP、主机名。平时是 warn,只提醒;加 `--public` 时是 error,开源前必须删掉,git 历史里的也要清掉 |
| 不是 git 仓库也能跑吗 | 能,但会把目录里所有文件都当成要提交的,大文件、软链接的检查不准 |
