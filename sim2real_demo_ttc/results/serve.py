#!/usr/bin/env python3
"""带正确编码的静态服务器 —— 用它代替 `python -m http.server`。

为什么需要：`python -m http.server` 给 .md 发的 Content-Type 是 `text/markdown`
但**不带 charset**，浏览器只能猜编码；中文环境下常猜成 Big5/GBK，
于是 UTF-8 的简体中文被按 Big5 解码，显示成繁体夹乱码。文件本身没问题。

本脚本做两件事：
  1. 所有文本类响应强制 `charset=utf-8`（根治乱码）；
  2. .md 默认渲染成 HTML 网页（报告里表格很多，纯文本读着累）；
     加 `?raw=1` 拿原始 markdown 文本。

用法：
    cd results && python serve.py            # 默认 8000
    cd results && python serve.py 8123       # 指定端口
然后访问 http://<host>:<port>/ 看索引页。
"""
from __future__ import annotations

import html
import http.server
import re
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXTY = (".md", ".json", ".html", ".txt", ".csv", ".js", ".css", ".py")

PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root{{--ink:#1b1f27;--muted:#5f6b7c;--rule:#e3e7ee;--bg:#fbfbfd;--card:#fff;--accent:#3558c4;--code:#f2f4f8}}
@media(prefers-color-scheme:dark){{:root{{--ink:#e6eaf1;--muted:#98a2b3;--rule:#2b313a;--bg:#14161b;--card:#1c1f26;--accent:#7d9fff;--code:#232833}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);line-height:1.72;font-size:15.5px;
 font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,-apple-system,"Segoe UI",sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:36px 20px 80px}}
.bar{{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:22px;
 padding-bottom:14px;border-bottom:1px solid var(--rule);font-size:13px}}
.bar a{{color:var(--accent);text-decoration:none}} .bar a:hover{{text-decoration:underline}}
h1{{font-size:26px;line-height:1.3;margin:.6em 0 .4em}} h2{{font-size:20px;margin:1.5em 0 .5em;
 padding-top:.4em;border-top:1px solid var(--rule)}} h3{{font-size:16.5px;margin:1.3em 0 .4em}}
h4{{font-size:15px;margin:1.1em 0 .3em}}
p,ul,ol{{margin:.7em 0}} li{{margin:.25em 0}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em;
 font-family:ui-monospace,SFMono-Regular,Menlo,"Noto Sans Mono",monospace}}
pre{{background:var(--code);padding:13px 15px;border-radius:9px;overflow-x:auto;line-height:1.55}}
pre code{{background:none;padding:0;font-size:12.6px}}
blockquote{{margin:.9em 0;padding:.1em 0 .1em 15px;border-left:3px solid var(--rule);color:var(--muted)}}
hr{{border:none;border-top:1px solid var(--rule);margin:1.8em 0}}
.tw{{overflow-x:auto;margin:1em 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;min-width:min(100%,520px)}}
th,td{{border:1px solid var(--rule);padding:7px 10px;text-align:left;vertical-align:top}}
th{{background:var(--card);font-weight:640;white-space:nowrap}}
td{{font-variant-numeric:tabular-nums}}
a{{color:var(--accent)}} del{{color:var(--muted)}}
</style></head><body><div class="wrap">
<div class="bar"><a href="/">← 索引</a><span style="color:var(--muted)">{title}</span>
<a href="?raw=1">原始 markdown</a></div>
{body}
</div></body></html>"""

INLINE = [
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{m.group(1)}</code>"),
    (re.compile(r"\*\*(.+?)\*\*"), lambda m: f"<strong>{m.group(1)}</strong>"),
    (re.compile(r"~~(.+?)~~"), lambda m: f"<del>{m.group(1)}</del>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), lambda m: f"<em>{m.group(1)}</em>"),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>'),
]


def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    for rx, fn in INLINE:
        t = rx.sub(fn, t)
    return t


def md_to_html(md: str) -> str:
    out, i, lines = [], 0, md.split("\n")
    while i < len(lines):
        ln = lines[i]
        # 代码围栏
        if ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1; continue
        # 表格：当前行含 | 且下一行是分隔行
        if "|" in ln and i + 1 < len(lines) and re.fullmatch(r"\s*\|?[\s:|-]+\|[\s:|-]*", lines[i + 1] or ""):
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]  # noqa: E731
            head = cells(ln); i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(cells(lines[i])); i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div>')
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            lv = len(m.group(1))
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>"); i += 1; continue
        # 水平线
        if re.fullmatch(r"\s*(-{3,}|\*{3,}|_{3,})\s*", ln):
            out.append("<hr>"); i += 1; continue
        # 引用块
        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip(">").strip()); i += 1
            out.append("<blockquote>" + "<br>".join(inline(x) for x in buf) + "</blockquote>"); continue
        # 列表
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = bool(re.match(r"\d+\.", m.group(2)))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    break
                items.append(mm.group(3)); i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        # 空行 / 段落
        if not ln.strip():
            i += 1; continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|```|>|\s*([-*+]|\d+\.)\s)", lines[i]) \
                and not ("|" in lines[i] and i + 1 < len(lines)
                         and re.fullmatch(r"\s*\|?[\s:|-]+\|[\s:|-]*", lines[i + 1] or "")):
            buf.append(lines[i]); i += 1
        if buf:
            out.append("<p>" + inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


class Handler(http.server.SimpleHTTPRequestHandler):
    def guess_type(self, path):
        t = super().guess_type(path)
        if any(str(path).endswith(e) for e in TEXTY) and "charset=" not in t:
            t += "; charset=utf-8"
        return t

    def do_GET(self):
        raw = "raw=1" in (self.path.split("?", 1)[1] if "?" in self.path else "")
        clean = self.path.split("?", 1)[0]
        if clean == "/":
            return self._index()
        if clean.endswith(".md") and not raw:
            f = ROOT / clean.lstrip("/")
            if f.is_file():
                body = md_to_html(f.read_text(encoding="utf-8"))
                return self._send(PAGE.format(title=f.name, body=body))
        return super().do_GET()

    def _index(self):
        mds = sorted(p.name for p in ROOT.glob("*.md"))
        others = sorted(p.name for p in ROOT.glob("*") if p.suffix in (".html", ".json", ".png"))
        li = lambda xs: "".join(f'<li><a href="/{x}">{x}</a></li>' for x in xs)  # noqa: E731
        body = (f"<h1>sim2real demo · results</h1><h2>报告</h2><ul>{li(mds)}</ul>"
                f"<h2>看板与数据</h2><ul>{li(others)}</ul>"
                "<p style=\"color:var(--muted);font-size:13px\">"
                "本服务器对所有文本响应强制 <code>charset=utf-8</code>；"
                "<code>python -m http.server</code> 不带 charset，中文会被浏览器猜成 Big5/GBK 而显示为繁体乱码。</p>")
        self._send(PAGE.format(title="results 索引", body=body))

    def _send(self, text: str):
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), Handler) as srv:
        print(f"serving {ROOT} at http://0.0.0.0:{port}/  (Ctrl-C 停止)")
        srv.serve_forever()
