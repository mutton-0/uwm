"""「从多少个场景起，读数就跟用满全池一样了」——一张图。

判据：九成情况下 |θ̂_n − θ̂_N| < 0.1·|θ̂_N|，且此后不再超出。
用**相对**阈值而非绝对阈值：四根轴的量级差 20 倍以上，G-VS 的值本身只有
0.02–0.04，绝对 0.02 就等于整个数值；C_m 接近 0.9，同样的 0.02 只占 2%。

必须连着读的一件事：抽的 n 个场景本来就在这 N 个里面，n 越接近 N 越必然一致，
所以撞到池子上限的行读作"到用满都没稳"，不是"N 个就够了"。图里用空心
标记 + 灰色天花板把这件事画出来，避免被读成需求量。
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
# 轴色 = 调色板槽 7/3/2/1，all-pairs 全过；顺序 = 便宜→贵
AXES = [("I  $I_m$", "I:  $I_m$", "#4a3aa7"),
        ("C  $C_m$", "C:  $C_m$", "#1baf7a"),
        ("G-VS  selectivity", "G-VS:  selectivity", "#eb6834"),
        ("F-3  $r$", "F-3:  $r$", "#2a78d6")]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
TOL_REL = 0.10

# 中英双语：论文用 en，看结论用 zh（matplotlib 没有中文字形，需显式指定 Noto CJK）
TXT = {
    "zh": dict(font="Noto Sans CJK JP",
               ceil="用满 {N} 仍未稳", x="场景数",
               title="从多少个场景起，读数就跟用满全池一样了\n"
                     "（九成情况下差距 < 该值的 10%；灰底 = 该轴现有的池子大小）",
               note="注：抽的 n 个场景本来就在池子里，n 越接近池子上限越必然一致。"
                    "撞到天花板的行读作「到用满都没稳」，不是「这么多就够了」。"),
    "en": dict(font="DejaVu Sans",
               ceil="not settled at {N}", x="number of scenes",
               title="At what pool size does the reading match the full-pool value?\n"
                     "(within 10% of that value, nine times in ten; grey = the pool we have)",
               note="Note: the $n$ scenes are drawn from the pool itself, so agreement is "
                    "forced as $n\\to N$. A row that hits the ceiling means \"not settled "
                    "even at full pool\", not \"this many suffice\"."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("zh", "en"), default="en")
    A = ap.parse_args()
    T = TXT[A.lang]
    plt.rcParams.update({"font.family": T["font"], "axes.unicode_minus": False})
    D = json.load(open(RES / "convergence_curves.json"))
    by = {}
    for c in D["curves"]:
        n = np.array(c["n"]); e = np.array(c["err90_abs"])
        N = int(n[-1]); th = c["final"]
        tol = abs(th) * TOL_REL
        k = [i for i in range(len(n)) if (e[i:] <= tol).all()] if tol > 0 else []
        nn = int(n[k[0]]) if k else None
        # n = N 时误差恒等于 0（抽满就是全池本身），这不是收敛，是恒等式。
        # 只有在 n < N 处就达标才算真的稳下来了。
        by[(c["model"], c["axis"])] = {
            "n": nn if (nn is not None and nn < N) else None, "N": N,
            "final": th}

    rows = []                      # (label, axis_label, color, n, N, hit_ceiling)
    for ax, axlab, col in AXES:
        got = [(m, v) for (m, a), v in by.items() if a == ax]
        got.sort(key=lambda z: (z[1]["n"] is None, z[1]["n"] or 1e9))
        for m, v in got:
            rows.append((NAME[m], axlab, col, v["n"], v["N"], v["n"] is None))

    h = 0.34 * len(rows) + 1.9
    fig, ax = plt.subplots(figsize=(7.4, h))
    ys = np.arange(len(rows))[::-1]

    for y, (lab, axlab, col, n, N, ceil) in zip(ys, rows):
        # 灰色天花板 = 该轴现有的池子大小
        ax.plot([0, N], [y, y], color=GRID, lw=5.5, solid_capstyle="butt",
                zorder=1)
        if ceil:
            # 到用满都没稳：画满整条并在末端用空心箭头表示"还没到"
            ax.plot([0, N], [y, y], color=col, lw=5.5, alpha=0.30,
                    solid_capstyle="butt", zorder=2)
            ax.annotate("", xy=(N + 26, y), xytext=(N + 2, y),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4,
                                        mutation_scale=11))
            ax.text(N + 32, y, T["ceil"].format(N=N), va="center", ha="left",
                    fontsize=7.2, color=col)
        else:
            ax.plot([0, n], [y, y], color=col, lw=5.5, solid_capstyle="butt",
                    zorder=2)
            ax.plot([n], [y], "o", ms=7.5, color=col, mec="#fcfcfb", mew=1.2,
                    zorder=3)
            ax.text(n + 7, y, f"{n}", va="center", ha="left", fontsize=8,
                    color=col, fontweight="bold")
        ax.text(-8, y, lab, va="center", ha="right", fontsize=8, color=INK)

    # 轴分组标签 + 分隔线
    seen, start = None, 0
    for i, (_, axlab, col, *_r) in enumerate(rows):
        if axlab != seen:
            if seen is not None:
                ax.axhline(ys[i] + 0.5, color=GRID, lw=0.8, zorder=0)
            seen, start = axlab, i
        if i == len(rows) - 1 or rows[i + 1][1] != axlab:
            ymid = (ys[start] + ys[i]) / 2
            ax.text(-118, ymid, axlab, va="center", ha="left", fontsize=9,
                    color=col, fontweight="bold")

    ax.set_xlim(-125, 430); ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.tick_params(labelsize=8, colors=INK2, length=3)
    ax.set_xlabel(T["x"], fontsize=9, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)

    ax.set_title(T["title"], fontsize=10, color=INK, pad=12, loc="left", x=-0.165)
    fig.text(0.012, 0.012, T["note"], fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"when_stable_{A.lang}.{e}", dpi=300,
                    bbox_inches="tight")
    print(f"[FIG] -> {OUT/f'when_stable_{A.lang}.pdf'}")


if __name__ == "__main__":
    main()
