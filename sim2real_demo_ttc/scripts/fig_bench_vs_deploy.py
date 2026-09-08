"""benchmark vs deployment 对比图：响应比 r 的点线图（带 scene 级 bootstrap CI）。

域的划分按 **P-1：左舵(LHD)=benchmark，右舵(RHD)=deployment**。两个语料
（nuScenes / NAVSIM）先合并再按舵位切，不再"nuScenes=benchmark"那样划。

形式选择：数据的任务是"带不确定度的量级 + 身份"，且 r 可为负 ——
柱状图配 CI 是反模式（柱从 0 起，负值与基线冲突），故用点+误差棒，
按场景分面，语料作为唯一的类别维度（2 类，取调色板槽 1/2，已过 CVD 校验）。
参考线：0 = 无响应，1 = 与人类减速幅度相当。
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
MODELS = ["simlingo", "dd", "ltf", "ddv2"]
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2"}
BLUE, ORANGE = "#2a78d6", "#eb6834"          # 调色板槽 1/2，validate_palette 全过
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

SRC = {("ghost", "nuscenes"): "f3_gt_axis_full_{m}.json",
       ("lead", "nuscenes"): "gt_lead_{m}.json",
       ("ghost", "navsim"): "f3_gt_dep_ghost_{m}.json",
       ("lead", "navsim"): "f3_gt_dep_lead_{m}.json"}
_MAP = json.load(open(RES / "driveside_map.json"))
SIDE = {"bench": "LHD", "dep": "RHD"}


def side_of(scene, corpus):
    if corpus == "nuscenes":
        return _MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):                 # 复合键，见 P-4
        s = _MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s
    return None


def cell(scn, side, m):
    """两语料合并后按舵位切。scene 键加语料前缀，避免跨语料同名场景被并成一个。"""
    bm, bg, sc = [], [], []
    for corpus in ("nuscenes", "navsim"):
        p = RES / SRC[(scn, corpus)].format(m=m)
        if not p.exists():
            continue
        for e in json.load(open(p))["per_event"]:
            if side_of(e["scene"], corpus) != SIDE[side]:
                continue
            bm.append(e["b_model__arc_full"]); bg.append(e["b_gt__arc_full"])
            sc.append(f"{corpus}|{e['scene']}")
    if len(bm) < 2:
        return None
    bm = np.array(bm); bg = np.array(bg); sc = np.array(sc)
    rng = np.random.default_rng(0); uq = np.unique(sc)
    ix = {s: np.where(sc == s)[0] for s in uq}
    boot = []
    for _ in range(5000):
        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])
        d = bg[sel].mean()
        if abs(d) > 1e-9:
            boot.append(bm[sel].mean() / d)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(bm.mean() / bg.mean()), float(lo), float(hi), len(bm)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharex=True)
    for ax, scn in zip(axes, ["ghost", "lead"]):
        ys = np.arange(len(MODELS))[::-1]
        for side, col, off, lab in ((("bench"), BLUE, +0.16, "benchmark (LHD)"),
                                    (("dep"), ORANGE, -0.16, "deployment (RHD)")):
            xs, los, his, yy = [], [], [], []
            for y, m in zip(ys, MODELS):
                c = cell(scn, side, m)
                if c is None:
                    continue
                r, lo, hi, _ = c
                xs.append(r); los.append(r - lo); his.append(hi - r); yy.append(y + off)
            if not xs:
                continue
            ax.errorbar(xs, yy, xerr=[los, his], fmt="o", ms=5.5, lw=2.0,
                        color=col, ecolor=col, capsize=0, label=lab, zorder=3,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8)
        ax.axvline(0, color=INK2, lw=1.0, ls="-", zorder=1)
        ax.axvline(1, color=GRID, lw=1.0, ls="--", zorder=1)
        ax.set_yticks(ys); ax.set_yticklabels([NAME[m] for m in MODELS], fontsize=8)
        ax.set_title(scn, fontsize=9, color=INK, pad=4)
        ax.tick_params(labelsize=8, colors=INK2, length=3)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
    axes[0].set_ylabel("")
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("response ratio $r$   (0 = no response, 1 = human magnitude)",
                  fontsize=8.5, color=INK2, y=0.02)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"bench_vs_deploy.{ext}", dpi=300, bbox_inches="tight")
    print(f"[FIG] -> {OUT/'bench_vs_deploy.pdf'}")


if __name__ == "__main__":
    main()
