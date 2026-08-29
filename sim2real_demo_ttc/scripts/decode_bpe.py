"""把 byte-level BPE token 串还原成可读文本（无需加载 tokenizer）。

A1 落盘的 `top_tokens` 是 GPT-2/Qwen 系的 byte-level BPE 记号（如 `å®Īä½ı`），
直接展示会是乱码。这里用标准的 bytes_to_unicode 反表还原，
并就地为 `candidate_directions.json` 补一个 `top_tokens_decoded` 字段、
重写 `priority_queue*.md` 的最近邻 token 列。
"""
from __future__ import annotations
import json, re
from pathlib import Path

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b); cs.append(2 ** 8 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


DEC = {v: k for k, v in bytes_to_unicode().items()}


def decode(tok: str) -> str:
    try:
        return bytes([DEC[c] for c in tok]).decode("utf-8", errors="replace")
    except KeyError:
        return tok


def main():
    p = RES / "candidate_directions.json"
    d = json.load(open(p))
    for c in d["candidates"]:
        c["top_tokens_decoded"] = [decode(t) for t in c["top_tokens"]]
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    idx = {(c["layer"], c["unit"]): c["top_tokens_decoded"] for c in d["candidates"]}
    print("[decode] candidate_directions.json 已补 top_tokens_decoded")

    for f in RES.glob("priority_queue*.md"):
        out = []
        for line in f.read_text().split("\n"):
            m = re.match(r"^\| (\d+) \| L(\d+) \| (\d+) \|(.*?)\| `([^`]*)` \|(.*)$", line)
            if m:
                key = (int(m.group(2)), int(m.group(3)))
                toks = idx.get(key)
                if toks:
                    line = f"| {m.group(1)} | L{m.group(2)} | {m.group(3)} |{m.group(4)}| " \
                           f"`{' '.join(t.strip() for t in toks[:3])}` |{m.group(6)}"
            line = line.replace(
                "> 阈值取 结构匹配零分布 与 各向同性零分布 的较严者（Bonferroni / 极值外推取大）。",
                "> 阈值取 结构匹配零分布（全 116,546 条 value vector 枚举）与 各向同性零分布 "
                "两者经验分位的**较严者**；α = 0.05/950。**不使用高斯尾外推**（见修正案 DV/A14）。")
            out.append(line)
        f.write_text("\n".join(out))
        print(f"[decode] 重写 {f.name}")


if __name__ == "__main__":
    main()
