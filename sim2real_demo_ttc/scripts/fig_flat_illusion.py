"""为什么 convergence.png 看起来「100 来个就平了」—— 那是纵轴被 n=2 撑开造成的。

同一份数据，只改横轴起点：上排从 n=2 起（原图），下排从 n=20 起。
下排的纵轴不再被最左端的爆炸值支配，于是能看出带子一直收到最后都没停。
量化：90% 带半宽在 n=2 是 n=100 的 8–9 倍，所以 n≥100 的部分被压成一条线；
但 n=200 的带宽仍只有 n=100 的一半 —— 还在按 1/√n 收，没有拐点。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/figures")
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2"}
AXCOL = {"F-3  $r$": "#2a78d6", "G-VS  selectivity": "#eb6834",
         "C  $C_m$": "#1baf7a", "I  $I_m$": "#4a3aa7"}
SHOW = [("simlingo", "F-3  $r$"), ("dd", "G-VS  selectivity"),
        ("ltf", "C  $C_m$")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
TXT = {"zh": dict(font="Noto Sans CJK JP",
                  a="从 n=2 起画（原图）", b="同一份数据，从 n=20 起画",
                  x="场景数",
                  t="「看起来 100 就平了」是纵轴造成的错觉\n"
                    "上：n=2 的带宽是 n=100 的 8–9 倍，把右半边压成一条线；"
                    "下：砍掉最左端后，带子一直收到最后"),
       "en": dict(font="DejaVu Sans",
                  a="drawn from $n{=}2$ (the original)",
                  b="same data, drawn from $n{=}20$", x="number of scenes",
                  t="The apparent plateau is an artefact of the vertical range\n"
                    "Top: the band at $n{=}2$ is 8--9$\\times$ the band at "
                    "$n{=}100$, flattening everything to its right. "
                    "Bottom: with the left edge dropped, the band keeps shrinking.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("zh", "en"), default="en")
    A = ap.parse_args(); T = TXT[A.lang]
    plt.rcParams.update({"font.family": T["font"], "axes.unicode_minus": False})
    C = {(c["model"], c["axis"]): c for c in
         json.load(open(RES / "convergence_curves.json"))["curves"]}

    fig, axes = plt.subplots(2, len(SHOW), figsize=(9.0, 4.3))
    for j, key in enumerate(SHOW):
        c = C[key]; col = AXCOL[key[1]]
        n = np.array(c["n"]); mn = np.array(c["mean"])
        lo, hi = np.array(c["ci90_lo"]), np.array(c["ci90_hi"])
        for i, n0 in enumerate((2, 20)):
            ax = axes[i, j]
            k = n >= n0
            ax.fill_between(n[k], lo[k], hi[k], color=col, alpha=0.20, lw=0)
            ax.plot(n[k], mn[k], color=col, lw=1.8)
            ax.axhline(c["final"], color=INK2, lw=0.9, ls="--", zorder=3)
            pad = (hi[k].max() - lo[k].min()) * 0.06
            ax.set_ylim(lo[k].min() - pad, hi[k].max() + pad)
            ax.set_xlim(n0 - 2, n[-1] + 4)
            ax.tick_params(labelsize=7, colors=INK2, length=2.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(f"{NAME[key[0]]} · {key[1]}", fontsize=8.5,
                             color=INK, pad=4)
            if j == 0:
                ax.set_ylabel(T["a"] if i == 0 else T["b"], fontsize=7.5,
                              color=INK2)
    fig.supxlabel(T["x"], fontsize=8.5, color=INK2, y=0.015)
    fig.suptitle(T["t"], fontsize=9.5, color=INK, x=0.008, ha="left", y=1.05)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"flat_illusion_{A.lang}.{e}", dpi=300,
                    bbox_inches="tight")
    print(f"[FIG] -> {OUT/f'flat_illusion_{A.lang}.png'}")


if __name__ == "__main__":
    main()
