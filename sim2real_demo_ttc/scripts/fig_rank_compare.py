"""排名对比图：F-3 响应比 vs 官方部署分数（随机场景 / 危险事件）。

形式：并列点图 + 连线（slope chart 的变体）。这里的任务是"同一批候选在不同度量下
名次是否一致"，所以画的是**名次**而不是原始值 —— 原始值量纲不可比。
颜色按候选身份固定（调色板槽 1-3 已过 all-pairs 校验；第 4 个用灰阶避免超出安全槽位）。
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/figures")
# 排名图只画在**三种度量上都有数据**的候选；两个 VLA 没跑 PDM，不入图。
MODELS = ["simlingo", "dd", "ltf", "ddv2"]
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2"}
# 槽 1/2/3/7（蓝/橙/青绿/紫）：validate_palette --pairs all 全过（最差 CVD ΔE 9.2，
# 常视 16.3）。不用槽 4（黄）——它与槽 2（橙）在 all-pairs 下常视 ΔE 13.7 不达标；
# 也不用灰——chroma 太低会被读成"无类别"。每条线都有直接标注作为二次编码。
COL = {"simlingo": "#2a78d6", "dd": "#eb6834", "ltf": "#1baf7a", "ddv2": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def f3_r(tag, m):
    p = RES / f"f3_gt_{tag}_{m}.json"
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    bm = np.array([e["b_model__arc_full"] for e in ev])
    bg = np.array([e["b_gt__arc_full"] for e in ev])
    return float(bm.mean() / bg.mean())


def main():
    cols = []
    r = {m: f3_r("dep_lead", m) for m in MODELS}
    if all(v is not None for v in r.values()):
        cols.append(("GFIC F-3\n(hazard response)", r))
    p = RES / "deploy_rank_joint.json"
    if p.exists():
        S = json.load(open(p))["summary"]
        cols.append(("Official PDM\n(random scenes)",
                     {m: S[m]["pdm_score"]["mean"] for m in MODELS}))
    p = RES / "hazard_avoidance_dep_lead.json"
    if p.exists():
        S = json.load(open(p))["summary"]
        cols.append(("Official PDM\n(hazard events)",
                     {m: S[m]["pdm_score"]["mean"] for m in MODELS}))
    if len(cols) < 2:
        print("数据不足"); return

    ranks = []
    for lab, d in cols:
        order = sorted(MODELS, key=lambda m: -d[m])
        ranks.append({m: i + 1 for i, m in enumerate(order)})

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    xs = np.arange(len(cols))
    for m in MODELS:
        ys = [rk[m] for rk in ranks]
        ax.plot(xs, ys, "-o", color=COL[m], lw=2.0, ms=6,
                markeredgecolor="#fcfcfb", markeredgewidth=0.9, zorder=3)
        ax.annotate(NAME[m], (xs[-1] + 0.06, ys[-1]), color=COL[m], fontsize=8,
                    va="center", ha="left")
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cols], fontsize=8, color=INK2)
    ax.set_yticks([1, 2, 3, 4]); ax.set_yticklabels(["1st", "2nd", "3rd", "4th"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.25, len(cols) - 1 + 1.35)
    ax.tick_params(colors=INK2, length=3, labelsize=8)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"rank_compare.{e}", dpi=300, bbox_inches="tight")
    print(f"[FIG] -> {OUT/'rank_compare.pdf'}")
    for (lab, d), rk in zip(cols, ranks):
        print(f"  {lab.splitlines()[0]:<24}" + "  ".join(f"{NAME[m]}:{rk[m]}" for m in MODELS))


if __name__ == "__main__":
    main()
