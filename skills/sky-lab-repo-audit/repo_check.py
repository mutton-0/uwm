#!/usr/bin/env python3
"""sky-lab 仓库规范检查。只读,不修改任何文件。

用法:
    python3 repo_check.py [项目目录] [--json] [--public]

- 项目目录默认是当前目录。可以是 git 仓库根目录,也可以是仓库里的子项目(比如 uwm/sim2real_demo_ttc)。
- --json   输出 JSON,给 AI / CI 用
- --public 额外检查开源前的要求(英文 README、CONTRIBUTING、CITATION、git 历史里的大文件),
            并且个人 / 网络信息(PERSONAL_INFO)从 warn 升级为 error

退出码:有 error 返回 1,否则返回 0。
每条结果带 REPO_CHECKLIST.md 里的编号。脚本只查能机械判断的项,需要人判断的项列在输出最后。
只依赖 Python 3.8+ 标准库和 git。
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CHECKLIST_URL = "https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/docs/REPO_CHECKLIST.md"
GITIGNORE_TEMPLATE_URL = "https://github.com/mutton-0/uwm/blob/demo/code-review-guidelines/templates/.gitignore"

LARGE_FILE_MB = 5
TEXT_SCAN_LIMIT = 1_000_000

# 不该进仓库的文件类型(权重、传感器数据、视频、压缩包)
ARTIFACT_EXT = {
    ".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".engine", ".h5",
    ".bag", ".db3", ".mcap", ".mp4", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".tgz", ".7z",
}
# 体积超过 1 MB 才算问题的数组缓存
ARRAY_EXT = {".npy", ".npz"}
# 扫绝对路径的文件类型
CODE_EXT = {".py", ".sh", ".bash", ".yaml", ".yml", ".toml", ".cfg", ".ini",
            ".launch", ".xml", ".c", ".cc", ".cpp", ".h", ".hpp"}
ENV_FILES = ["requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml", "setup.py", "setup.cfg"]

ABS_PATH_RE = re.compile(
    r"(?<![\w./}\)-])"
    r"(/home/[A-Za-z0-9_.-]+|/data/[A-Za-z0-9_.-]+|/mnt/[A-Za-z0-9_.-]+|/media/[A-Za-z0-9_.-]+"
    r"|/Users/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+)"
)
SECRET_PATTERNS = [
    ("私钥", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("AWS key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("OpenAI / Anthropic key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}\b")),
    ("HuggingFace token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("WandB key", re.compile(r"WANDB_API_KEY\s*[=:]\s*['\"]?[0-9a-f]{40}")),
]
# 个人 / 网络信息:不是密钥,但不该进仓库(尤其开源前)。输出里只给位置和类型,不打印原值
PERSONAL_PATTERNS = [
    ("邮箱", re.compile(r"(?<![\w.%+-])(?!git@)[A-Za-z0-9._%+-]+@(?!(?:users\.)?noreply\.|example\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("内网 / Tailscale IP", re.compile(r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01])|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))\.\d{1,3}\.\d{1,3}(?![\d.])")),
    ("Tailscale 主机名", re.compile(r"\b[\w-]+\.[\w-]+\.ts\.net\b")),
]
# 这些文件里出现邮箱是正常的
PERSONAL_SKIP = re.compile(r"(^|/)(LICENSE[^/]*|COPYING|AUTHORS[^/]*|CITATION\.cff|CODEOWNERS|\.mailmap)$")
SECRET_FILENAMES = re.compile(r"(^|/)(\.env(\.[^/]*)?|id_rsa|id_ed25519|[^/]*\.pem|[^/]*\.key)$")

README_SECTIONS = {
    "环境": ["环境", "environment", "setup", "install", "安装", "requirement", "依赖"],
    "怎么跑": ["怎么跑", "运行", "使用", "用法", "usage", "quick start", "quickstart", "getting started", "run", "复现"],
    "数据 / 权重": ["数据", "data", "权重", "weight", "checkpoint", "模型"],
    "已知问题": ["已知问题", "known issue", "todo", "limitation", "问题"],
}

# 脚本查不了、需要人或 AI 读代码判断的项(编号对应 REPO_CHECKLIST.md)
MANUAL_ITEMS = [
    (8, "一句话说明是否准确,上下游依赖写清楚了没有"),
    (13, "最小例子能不能真的跑通(最好实际运行一次,或者跑 smoke test)"),
    (14, "有多个终端/进程时,是否写清楚哪些并行、哪些串行"),
    (17, "报了结果的,是否给了命令和随机种子;不能完全复现的,是否写了波动范围和原因"),
    (18, "已知问题是否写的是实话"),
    (19, "实车项目:传感器摆放(位置、朝向、高度,配图)"),
    (20, "实车项目:外参来源(哪次标定、日期、脚本)"),
    (21, "实车项目:线材(接口、最短长度、哪里拿)"),
    (22, "实车项目:供电、支架等其他物料"),
    (23, "README 里的参数、默认值和代码是否一致(脚本只查了文件是否存在)"),
    (24, "实车项目:代码里的外参、分辨率、采样率和 README 描述是否一致"),
]
MANUAL_ITEMS_PUBLIC = [
    (34, "拷贝或改写的第三方代码是否保留了原 LICENSE 和出处"),
    (35, "权重和数据是否放到了公开位置,写清下载方式和许可"),
]


def run(cmd, cwd):
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    return r.stdout if r.returncode == 0 else None


class Checker:
    def __init__(self, project, public):
        self.proj = Path(project).resolve()
        self.public = public
        self.results = []
        top = run(["git", "rev-parse", "--show-toplevel"], self.proj)
        self.git_root = Path(top.strip()).resolve() if top else None
        self.branch = (run(["git", "rev-parse", "--abbrev-ref", "HEAD"], self.proj) or "").strip() or None
        self.tracked = self._tracked_files()
        self.readme_path = self._find_readme()
        self.readme = self.readme_path.read_text(encoding="utf-8", errors="replace") if self.readme_path else ""

    # ---------- 工具 ----------
    def add(self, cid, level, checklist, title, details=None, fix=None):
        self.results.append({"id": cid, "level": level, "checklist": checklist, "title": title,
                             "details": details or [], "fix": fix or ""})

    def _tracked_files(self):
        """返回 [(mode, sha, 相对项目目录的路径, 大小)]。不是 git 仓库时退化为遍历目录。"""
        out = run(["git", "ls-files", "-s", "-z"], self.proj) if self.git_root else None
        if out is None:
            files = []
            for p in self.proj.rglob("*"):
                if ".git" in p.parts or not (p.is_file() or p.is_symlink()):
                    continue
                mode = "120000" if p.is_symlink() else "100644"
                size = 0 if p.is_symlink() else p.stat().st_size
                files.append((mode, None, p.relative_to(self.proj).as_posix(), size))
            return files
        entries = []
        for rec in out.split("\0"):
            if not rec:
                continue
            meta, path = rec.split("\t", 1)
            mode, sha, _stage = meta.split()
            entries.append((mode, sha, path))
        sizes = {}
        if entries:
            batch = "\n".join(e[1] for e in entries) + "\n"
            r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
                               cwd=str(self.proj), input=batch, capture_output=True, text=True)
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].isdigit():
                    sizes[parts[0]] = int(parts[1])
        return [(m, s, p, sizes.get(s, 0)) for m, s, p in entries]

    def _find_readme(self):
        for name in ("README.md", "readme.md", "README.MD", "README.rst", "README.txt", "README"):
            p = self.proj / name
            if p.is_file():
                return p
        return None

    def _find_up(self, names):
        """在项目目录找,找不到再往上找到 git 根目录。"""
        d = self.proj
        while True:
            for n in names:
                if (d / n).exists():
                    return d / n
            if self.git_root is None or d == self.git_root or d.parent == d:
                return None
            d = d.parent

    def _read_text(self, rel):
        p = self.proj / rel
        try:
            if p.is_symlink() or p.stat().st_size > TEXT_SCAN_LIMIT:
                return None
            data = p.read_bytes()
        except OSError:
            return None
        if b"\0" in data[:4096]:
            return None
        return data.decode("utf-8", errors="replace")

    # ---------- 检查项 ----------
    def check_readme(self):
        if not self.readme_path:
            self.add("README_MISSING", "error", 1, "项目目录下没有 README.md",
                     fix="按 REPO_GUIDELINES 第二节模板写 README:这是什么、环境、怎么跑、数据/权重、已知问题")
            return
        text = self.readme
        low = text.lower()
        meta_missing = []
        if not re.search(r"维护人|maintainer|manager|负责人", text, re.I):
            meta_missing.append("维护人(manager)")
        if not re.search(r"状态|status", text, re.I):
            meta_missing.append("状态(active / 交接中 / archived)")
        if meta_missing:
            self.add("README_META", "error", 9, "README 顶部缺:" + "、".join(meta_missing),
                     fix="在标题下面写一行维护人和状态。维护人是谁要问用户,不要猜")
        headings = [l.lstrip("#").strip().lower() for l in text.splitlines() if l.startswith("#")]
        missing = [name for name, keys in README_SECTIONS.items()
                   if not any(k in h for h in headings for k in keys)]
        if missing:
            self.add("README_SECTIONS", "warn", "10/13/15/18", "README 缺少这些章节:" + "、".join(missing),
                     fix="照模板补章节。内容从代码和已有文档里找,找不到的写'待确认',不要编造版本号、路径或结果")
        if not re.search(r"gpu|cuda|显卡|显存|cpu 即可|不需要 gpu|无需 gpu", low):
            self.add("README_HARDWARE", "warn", 11, "README 没写硬件环境(GPU 型号、显存、驱动;不需要 GPU 也要写明)",
                     fix="写 GPU 型号、显存、驱动/CUDA 版本。不知道的问用户")
        self._check_readme_commands()
        self._check_readme_links()

    def _check_readme_commands(self):
        bad = []
        bases = [self.proj] + ([self.git_root] if self.git_root else [])
        for block in re.findall(r"```[^\n]*\n(.*?)```", self.readme, re.S):
            cds = []
            for line in block.splitlines():
                s = line.split("#", 1)[0].strip()
                m = re.match(r"cd\s+([^\s;&|]+)", s)
                if m:
                    cds.append(m.group(1))
                    continue
                for tok in re.findall(r"(?:^|\s)((?:\./)?[\w][\w./-]*\.(?:py|sh))(?=\s|$)", s):
                    if "<" in tok or "*" in tok:
                        continue
                    cands = [b / tok for b in bases] + [b / c / tok for b in bases for c in cds]
                    if not any(c.exists() for c in cands):
                        bad.append(tok)
        if bad:
            self.add("README_COMMANDS", "warn", 23, "README 命令里提到的脚本在仓库里找不到",
                     details=sorted(set(bad))[:10],
                     fix="以代码为准修改 README(改文件名或删掉过时的命令),不要为了对上 README 去改代码")

    def _check_readme_links(self):
        bad = []
        for link in re.findall(r"\]\(([^)\s#]+)", self.readme):
            if re.match(r"[a-z]+://|mailto:", link):
                continue
            if not (self.readme_path.parent / link).exists():
                bad.append(link)
        if bad:
            self.add("README_LINKS", "warn", 23, "README 里的相对链接指向不存在的文件", details=sorted(set(bad))[:10],
                     fix="改成正确的相对路径,或者删掉链接")

    def check_env(self):
        found = self._find_up(ENV_FILES)
        if not found:
            self.add("ENV_FILE", "error", 2, "找不到环境文件(requirements.txt / environment.yml / pyproject.toml)",
                     fix="在项目真正使用的环境里导出依赖(问用户是哪个环境),版本用 == 钉死")
            return
        if found.parent != self.proj:
            self.add("ENV_FILE", "info", 2, f"环境文件在上级目录:{found.relative_to(self.git_root or found.parent)}")
        unpinned = []
        req_files = list(found.parent.glob("requirements*.txt"))
        for rf in req_files:
            for line in rf.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.split("#", 1)[0].strip()
                if not s or s.startswith(("-", "git+", "http")) or "@" in s:
                    continue
                if "==" not in s:
                    unpinned.append(f"{rf.name}: {s}")
        for ef in (found.parent / "environment.yml", found.parent / "environment.yaml"):
            if ef.is_file():
                for line in ef.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = line.split("#", 1)[0].strip()
                    if not s.startswith("- ") or s.endswith(":"):
                        continue
                    dep = s[2:].strip()
                    if dep.startswith(("-r", "-e", "pip")) or "=" in dep:
                        continue
                    unpinned.append(f"{ef.name}: {dep}")
        if unpinned:
            self.add("ENV_UNPINNED", "warn", 2, f"有 {len(unpinned)} 个依赖没有钉死版本(没写 ==)",
                     details=unpinned[:10],
                     fix="在项目实际使用的环境里用 pip freeze / conda list 查到真实版本再填,不要猜版本号")

    def check_gitignore(self):
        if not self._find_up([".gitignore"]):
            self.add("GITIGNORE_MISSING", "error", 3, "没有 .gitignore",
                     fix=f"从 {GITIGNORE_TEMPLATE_URL} 复制到仓库根目录")
            return
        if not self.git_root:
            return
        samples = {"权重 model.ckpt": "model.ckpt", "权重 model.pt": "model.pt", "日志 logs/train.log": "logs/train.log",
                   "rosbag drive.bag": "drive.bag", "视频 demo.mp4": "demo.mp4", "密钥 .env": ".env",
                   "数据 data/raw.bin": "data/raw.bin"}
        gaps = []
        for label, path in samples.items():
            r = subprocess.run(["git", "-c", f"core.excludesFile={os.devnull}", "check-ignore", "-q", "--no-index", path],
                               cwd=str(self.proj), capture_output=True)
            if r.returncode != 0:
                gaps.append(label)
        if gaps:
            self.add("GITIGNORE_GAPS", "warn", 3, ".gitignore 没有覆盖这些常见文件", details=gaps,
                     fix=f"参照 {GITIGNORE_TEMPLATE_URL} 补规则;项目确实需要提交的,用 ! 单独放行")

    def check_license(self):
        if not self._find_up(["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"]):
            self.add("LICENSE", "error", 4, "没有 LICENSE",
                     fix="默认 MIT;基于 Apache-2.0 项目改的用 Apache-2.0。选哪个问用户")

    def check_layout(self):
        root_py = [p for m, s, p, z in self.tracked if "/" not in p and p.endswith(".py")]
        if len(root_py) > 8:
            self.add("LAYOUT_ROOT_PY", "warn", 5, f"项目根目录直接堆了 {len(root_py)} 个 .py",
                     fix="按 src/、scripts/、configs/ 分开。移动文件会影响 import 和 README 里的命令,先和用户确认")

    def check_smoke_and_agent_docs(self):
        smoke = [p for m, s, p, z in self.tracked if re.search(r"(^|/)smoke_test\.(sh|py)$", p)]
        if not smoke:
            self.add("SMOKE_TEST", "warn" if self.public else "info", 6,
                     "没有一键自检脚本(scripts/smoke_test.sh)",
                     fix="写一个只检查环境、关键路径、能否跑一次最小推理的脚本,几十秒内跑完")
        if not self._find_up(["AGENTS.md", "CLAUDE.md"]):
            self.add("AGENT_DOC", "info", 7, "没有 AGENTS.md(给 AI 看的项目须知)",
                     fix="用 templates/AGENTS.md 的模板写;再放一个只有一行 @AGENTS.md 的 CLAUDE.md")

    def check_files(self):
        large, artifacts, secrets_files = [], [], []
        for mode, sha, path, size in self.tracked:
            ext = os.path.splitext(path)[1].lower()
            if mode != "120000" and size > LARGE_FILE_MB * 1024 * 1024:
                large.append((size, path))
            if ext in ARTIFACT_EXT or (ext in ARRAY_EXT and size > 1024 * 1024):
                artifacts.append(path)
            if SECRET_FILENAMES.search(path) and not path.endswith(".example"):
                secrets_files.append(path)
        if large:
            large.sort(reverse=True)
            self.add("LARGE_FILES", "error", 26, f"有 {len(large)} 个超过 {LARGE_FILE_MB} MB 的文件被 git 跟踪",
                     details=[f"{s / 1048576:.1f} MB  {p}" for s, p in large[:10]],
                     fix="列清单给用户确认后:git rm --cached <文件>(本地文件保留),补 .gitignore,在 README 写清去哪下载")
        if artifacts:
            self.add("ARTIFACTS_TRACKED", "error", 26, f"有 {len(artifacts)} 个权重 / rosbag / 视频 / 压缩包文件被 git 跟踪",
                     details=artifacts[:10],
                     fix="同上:先和用户确认,再 git rm --cached,补 .gitignore,README 里写存放位置")
        if secrets_files:
            self.add("SECRET_FILES", "error", 27, "疑似密钥文件被 git 跟踪", details=secrets_files[:10],
                     fix="马上告诉用户:密钥要作废重新生成;文件 git rm --cached 并加进 .gitignore。git 历史怎么处理由用户决定")

    def check_text_content(self):
        secret_hits, personal_hits, abs_refs, abs_files = [], [], defaultdict(int), defaultdict(set)
        for mode, sha, path, size in self.tracked:
            if mode == "120000":
                continue
            ext = os.path.splitext(path)[1].lower()
            text = self._read_text(path)
            if text is None:
                continue
            for name, pat in SECRET_PATTERNS:
                if pat.search(text):
                    secret_hits.append(f"{path}({name})")
            if not PERSONAL_SKIP.search(path):
                for lineno, line in enumerate(text.splitlines(), 1):
                    for name, pat in PERSONAL_PATTERNS:
                        if pat.search(line):
                            personal_hits.append(f"{path}:{lineno}({name})")
            if ext in CODE_EXT:
                for m in ABS_PATH_RE.finditer(text):
                    prefix = m.group(1)
                    abs_refs[prefix] += 1
                    abs_files[prefix].add(path)
        if secret_hits:
            self.add("SECRETS", "error", 27, "代码里疑似有密钥 / token", details=secret_hits[:10],
                     fix="马上告诉用户,不要在回复里原样贴出密钥。密钥要作废重新生成;代码改成从 .env 或环境变量读")
        if personal_hits:
            self.add("PERSONAL_INFO", "error" if self.public else "warn", 27, f"有 {len(personal_hits)} 处个人 / 网络信息(邮箱、内网 IP、Tailscale 主机名)",
                     details=personal_hits[:10],
                     fix="告诉用户位置和类型,回复里不要贴出原值。平时只是提醒;开源(--public)前必须删掉,已进 git 历史的也要清理")
        if abs_refs:
            all_files = set().union(*abs_files.values())
            undocumented = [p for p in abs_refs if p not in self.readme]
            details = [f"{p}: {abs_refs[p]} 处 / {len(abs_files[p])} 个文件"
                       + ("" if p in self.readme else "  ← README 里没提到")
                       for p in sorted(abs_refs, key=lambda k: -abs_refs[k])]
            if undocumented:
                self.add("ABS_PATHS", "warn", 16,
                         f"{len(all_files)} 个代码/配置文件写死了绝对路径,其中 {len(undocumented)} 个根目录没在 README 里说明",
                         details=details[:10],
                         fix="默认只在 README 的'本机专属路径'一节逐个列出。改代码(收进 configs/paths.yaml 或环境变量)要先问用户")
            else:
                self.add("ABS_PATHS", "info", 16,
                         f"{len(all_files)} 个代码/配置文件写死了绝对路径,README 里都已说明", details=details[:10])

    def check_symlinks(self):
        problems = []
        for mode, sha, path, size in self.tracked:
            if mode != "120000":
                continue
            target = os.readlink(self.proj / path) if (self.proj / path).is_symlink() else None
            if target is None and sha:
                target = run(["git", "cat-file", "-p", sha], self.proj)
            if not target:
                continue
            target = target.strip()
            if os.path.isabs(target):
                problems.append(f"{path} -> {target}(绝对路径,别的机器上不存在)")
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            in_git = run(["git", "ls-files", "--", resolved], self.proj)
            if not in_git:
                problems.append(f"{path} -> {target}(目标没进 git,别人 clone 下来是断的)")
        if problems:
            self.add("SYMLINKS", "warn", 16, "有软链接在别的机器上会断", details=problems[:10],
                     fix="告诉用户,确认后删掉软链接或改成 README 里说明的路径;不要删除软链接指向的数据")

    def check_codeowners(self):
        if not self.git_root:
            return
        co = None
        for rel in (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"):
            if (self.git_root / rel).is_file():
                co = self.git_root / rel
                break
        if co is None:
            self.add("CODEOWNERS", "error", 29, "仓库没有 .github/CODEOWNERS(main 需要 manager 审批)",
                     fix="写一行:*  @manager用户名 @副manager用户名。用户名问用户,不要猜")
            return
        owners = set()
        for line in co.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.split("#", 1)[0].split()
            if s and s[0] == "*":
                owners.update(t for t in s[1:] if t.startswith("@"))
        if not owners:
            self.add("CODEOWNERS", "error", 29, "CODEOWNERS 里没有覆盖全部文件的 * 规则")
        elif any(re.search(r"[^\x00-\x7f]|用户名|username", o) for o in owners):
            self.add("CODEOWNERS", "warn", 29, "CODEOWNERS 里还是占位符", details=sorted(owners),
                     fix="换成真实的 GitHub 用户名,问用户")
        elif len(owners) < 2:
            self.add("CODEOWNERS", "warn", 29, "CODEOWNERS 只有一个人,manager 自己提的 PR 没人能批",
                     details=sorted(owners), fix="加一个副 manager,问用户是谁")

    def check_tags(self):
        if self.git_root and not (run(["git", "tag", "-l", "base-*"], self.proj) or "").strip():
            self.add("BASE_TAG", "info", 30, "还没有 base-YYYY-MM tag(第一次月度合入后由 manager 打)")

    def check_public(self):
        if not self.public:
            return
        if self.readme:
            cjk = len(re.findall(r"[一-鿿]", self.readme))
            if cjk / max(len(self.readme), 1) > 0.15 and not self._find_up(["README_EN.md", "README.en.md", "README-en.md"]):
                self.add("PUBLIC_EN_README", "warn", 33, "README 主要是中文,开源前需要英文或中英双语版本")
        for cid, names, num, title in (("PUBLIC_CONTRIBUTING", ["CONTRIBUTING.md", ".github/CONTRIBUTING.md"], 36, "没有 CONTRIBUTING.md"),
                                       ("PUBLIC_CITATION", ["CITATION.cff"], 37, "没有 CITATION.cff(论文引用方式)")):
            if not self._find_up(names):
                self.add(cid, "warn", num, title)
        if self.git_root:
            out = run(["git", "rev-list", "--objects", "HEAD", "--", "."], self.proj) or ""
            objs = [l.split(" ", 1) for l in out.splitlines() if " " in l]
            if objs:
                batch = "\n".join(o[0] for o in objs) + "\n"
                r = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
                                   cwd=str(self.proj), input=batch, capture_output=True, text=True)
                names = dict((o[0], o[1]) for o in objs)
                big = []
                for line in r.stdout.splitlines():
                    sha, typ, size = line.split()
                    if typ == "blob" and int(size) > LARGE_FILE_MB * 1024 * 1024:
                        big.append((int(size), names.get(sha, sha)))
                if big:
                    big.sort(reverse=True)
                    self.add("HISTORY_LARGE", "error", 38,
                             f"git 历史里有 {len(big)} 个超过 {LARGE_FILE_MB} MB 的文件(当前分支已删掉的也算),转 public 前要清理",
                             details=[f"{s / 1048576:.1f} MB  {p}" for s, p in big[:10]],
                             fix="重写历史(git filter-repo)会影响所有人的 clone,只能由用户决定和执行")

    def run_all(self):
        for f in (self.check_readme, self.check_env, self.check_gitignore, self.check_license, self.check_layout,
                  self.check_smoke_and_agent_docs, self.check_files, self.check_text_content, self.check_symlinks,
                  self.check_codeowners, self.check_tags, self.check_public):
            f()
        order = {"error": 0, "warn": 1, "info": 2}
        self.results.sort(key=lambda r: order[r["level"]])
        manual = MANUAL_ITEMS + (MANUAL_ITEMS_PUBLIC if self.public else [])
        summary = {lv: sum(1 for r in self.results if r["level"] == lv) for lv in ("error", "warn", "info")}
        return {
            "project": str(self.proj),
            "git_root": str(self.git_root) if self.git_root else None,
            "branch": self.branch,
            "tracked_files": len(self.tracked),
            "tracked_size_mb": round(sum(z for m, s, p, z in self.tracked) / 1048576, 2),
            "checklist": CHECKLIST_URL,
            "summary": summary,
            "results": self.results,
            "manual_items": [{"checklist": n, "item": t} for n, t in manual],
        }


def print_human(report):
    mark = {"error": "✗ error", "warn": "! warn ", "info": "· info "}
    print(f"项目:{report['project']}")
    print(f"git 根目录:{report['git_root']}  分支:{report['branch']}")
    print(f"跟踪的文件:{report['tracked_files']} 个,{report['tracked_size_mb']} MB")
    print(f"检查表:{report['checklist']}\n")
    if not report["results"]:
        print("脚本能查的项全部通过。\n")
    for r in report["results"]:
        print(f"[{mark[r['level']]}] #{r['checklist']} {r['id']}: {r['title']}")
        for d in r["details"]:
            print(f"      {d}")
        if r["fix"]:
            print(f"      → {r['fix']}")
    print("\n需要人或 AI 读代码判断的项(脚本查不了):")
    for m in report["manual_items"]:
        print(f"  #{m['checklist']} {m['item']}")
    s = report["summary"]
    print(f"\n合计:{s['error']} error,{s['warn']} warn,{s['info']} info")


def main():
    ap = argparse.ArgumentParser(description="sky-lab 仓库规范检查(只读)")
    ap.add_argument("project", nargs="?", default=".", help="项目目录,默认当前目录")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--public", action="store_true", help="额外检查开源前的要求")
    args = ap.parse_args()
    if not Path(args.project).is_dir():
        print(f"不是目录:{args.project}", file=sys.stderr)
        return 2
    report = Checker(args.project, args.public).run_all()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 1 if report["summary"]["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
