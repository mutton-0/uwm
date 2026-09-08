"""加数据能换来多少精度 —— 误差棒相对数值本身有多大，随场景数怎么走。

纵轴 = 误差棒半宽 ÷ |数值| = 1.96·σ/√n / |θ̂|，σ = SE_N·√N 是每场景尺度，
与现有池子大小无关，所以横轴可以外推到手上没有的数据量。
这是与 when_stable 互补的另一个问法：那张问"少抽会不会变"（只在池子内部
有效），这张问"要把这个数报准，一般需要多少"。

横线 10% = 误差棒是数值的十分之一，即该数报到一位有效数字是稳的。
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
        "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
AXES = [("I  $I_m$", "I:  $I_m$"), ("C  $C_m$", "C:  $C_m$"),
        ("G-VS  selectivity", "G-VS:  selectivity"), ("F-3  $r$", "F-3:  $r$")]
# 族色（3 色，all-pairs 过）+ 线型作二次编码；6 色在色觉检验下不合格
FAM = {"dd": ("#2a78d6", "-"), "ltf": ("#2a78d6", "--"), "ddv2": ("#2a78d6", ":"),
       "simlingo": ("#eb6834", "-"), "alpa": ("#1baf7a", "-"),
       "autovla": ("#1baf7a", "--")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
TARGET = 0.10
NGRID = np.array([25, 35, 50, 75, 100, 150, 200, 300, 450, 600, 900, 1200,
                  1800, 2400, 3600])

TXT = {"zh": dict(font="Noto Sans CJK JP", x="场景数（对数轴）",
                  y="误差棒半宽 ÷ |数值|",
                  title="要把一个轴的数报准，需要多少场景\n"
                        "（横线 = 误差棒缩到数值的 10%；竖线 = 现有数据量）",
                  now="现有", far="{k} 条超出 3600"),
       "en": dict(font="DejaVu Sans", x="number of scenes (log)",
                  y="error-bar half-width $\\div$ |value|",
                  title="How much data does it take to quote an axis precisely?\n"
                        "(horizontal line: error bar down to 10% of the value; "
                        "vertical: what we have)",
                  now="have", far="{k} beyond 3600")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("zh", "en"), default="en")
    A = ap.parse_args(); T = TXT[A.lang]
    plt.rcParams.update({"font.family": T["font"], "axes.unicode_minus": False})

    S = json.load(open(RES / "sample_size.json"))["rows"]
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.1), sharey=True)
    for ax, (key, lab) in zip(axes, AXES):
        rs = [r for r in S if r["axis"] == key]
        Npool = rs[0]["N"] if rs else None
        cross, far = [], []
        for r in sorted(rs, key=lambda z: z["sigma"]):
            col, ls = FAM[r["model"]]
            val = r.get("final")
            if not val:                      # 值恰为 0 时相对误差无定义
                continue
            rel = 1.96 * r["sigma"] / np.sqrt(NGRID) / abs(val)
            ax.plot(NGRID, rel, color=col, ls=ls, lw=1.7,
                    label=NAME[r["model"]])
            # 对数-对数下都是斜率 -1/2 的直线，解析求过 10% 线的点
            n0 = (1.96 * r["sigma"] / (TARGET * abs(val))) ** 2
            if NGRID[0] <= n0 <= NGRID[-1]:
                ax.plot([n0], [TARGET], "o", ms=5.5, color=col,
                        mec="#fcfcfb", mew=1.0, zorder=5)
                cross.append((n0, col, NAME[r["model"]]))
            elif n0 > NGRID[-1]:
                far.append(NAME[r["model"]])
        ax.axhline(TARGET, color=INK2, lw=1.0, ls="--", zorder=1)
        # 交叉点的 n 竖排标在点下方，避免彼此重叠
        for k, (n0, col, nm) in enumerate(sorted(cross)):
            ax.annotate(f"{n0:.0f}", (n0, TARGET), fontsize=7, color=col,
                        fontweight="bold", ha="center", va="top",
                        xytext=(0, -4 - 9 * (k % 2)), textcoords="offset points")
        if far:
            ax.text(0.97, 0.03, T["far"].format(k=len(far)),
                    transform=ax.transAxes, fontsize=6.8, color=INK2,
                    ha="right", va="bottom")
        if Npool:
            ax.axvline(Npool, color=GRID, lw=1.4, zorder=0)
            ax.text(Npool * 1.06, 5.5, T["now"], fontsize=7, color=INK2,
                    rotation=90, va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(25, 3600); ax.set_ylim(0.0012, 8)
        ax.set_title(lab, fontsize=9.5, color=INK, pad=5)
        ax.set_xticks([50, 100, 300, 1000, 3000])
        ax.set_xticklabels(["50", "100", "300", "1k", "3k"])
        ax.set_yticks([0.002, 0.005, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5])
        ax.set_yticklabels(["0.2%", "0.5%", "2%", "5%", "10%", "25%", "50%",
                            "1$\\times$", "2$\\times$", "5$\\times$"])
        ax.tick_params(labelsize=7.5, colors=INK2, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.grid(True, which="major", color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel(T["y"], fontsize=8.5, color=INK2)
    # 共享图例：6 色在 all-pairs 色觉检验下不合格，故族色 + 线型复合编码
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=FAM[m][0], ls=FAM[m][1], lw=1.7, label=NAME[m])
          for m in ("dd", "ltf", "ddv2", "simlingo", "alpa", "autovla")]
    fig.legend(handles=hs, loc="lower center", ncol=6, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.055), labelcolor=INK2,
               handlelength=2.2)
    fig.supxlabel(T["x"], fontsize=8.5, color=INK2, y=0.02)
    fig.suptitle(T["title"], fontsize=10, color=INK, x=0.008, ha="left", y=1.06)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"precision_vs_n_{A.lang}.{e}", dpi=300,
                    bbox_inches="tight")
    print(f"[FIG] -> {OUT/f'precision_vs_n_{A.lang}.pdf'}")


if __name__ == "__main__":
    main()
