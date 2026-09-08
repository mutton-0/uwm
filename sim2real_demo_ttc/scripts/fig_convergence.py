"""四轴取值随 scene 数增长的收敛曲线。

两张图：
  convergence.pdf       6 候选 × 4 轴 的小多图，每格自己的量纲，看**取值**是否趋平；
  convergence_need.pdf  每候选一格、四轴叠加，纵轴是**归一化的分位带宽**
                        w(n)/w(n_min) —— 尺度无关，四轴才可合法共轴
                        （原始量纲差 20 倍以上，直接叠加读不出东西，也等于隐性双轴）。
                        横轴上标出各轴降到阈值所需的 scene 数，取最大者即
                        「该候选四轴都稳所需的样本量」。

## 方法
对每个 n（scene 数），**随机抽 n 个 scene**、重复 B 次、在每次抽样上重算该轴的量，
画中位数与 [10,90] 分位带。用随机子集而不是"前 n 个"——后者的结果取决于
事件的排列顺序，是任意的。抽样单位是 **scene**，与全文的 bootstrap 口径一致。

## 每轴重算的量（都从逐单元原始量重算，不是把总值按比例缩放）
  F-3   r = mean(b_model)/mean(b_gt)                  逐事件
  G-VS  selectivity = mean(trained mIoU − random mIoU) 逐 scene
  C     C_m = top-2 层在 sum(max(rec,0)) 中的占比      逐事件 × 逐层
  I     I_m = 1 − D_{L*}（L* 为预登记层）              逐 pair

## 读法
曲线随 n 增大**趋平且分位带收窄** ⇒ 该格的估计已稳定；
仍在漂移或带很宽 ⇒ 样本量不足，该格的点估计不可当作模型性质。
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/figures")
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
B = 200


# ---------------- 各轴：返回 (units, scenes, fn) ----------------
def f3_units(m, tag="dep_lead"):
    p = RES / f"f3_gt_{tag}_{m}.json"
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    bm = np.array([e["b_model__arc_full"] for e in ev])
    bg = np.array([e["b_gt__arc_full"] for e in ev])
    sc = np.array([e["scene"] for e in ev])
    return (bm, bg), sc, lambda ix: bm[ix].mean() / bg[ix].mean()


def gvs_units(m, tag="dep_lead"):
    p = RES / f"gvs_{tag}_{m}.json"
    if not p.exists():
        return None
    a = json.load(open(p))["arms"]
    tr, rd = a["trained"]["per_scene_miou"], a["random_init"]["per_scene_miou"]
    ks = [k for k in tr if k in rd]
    v = np.array([tr[k] - rd[k] for k in ks])
    return v, np.array(ks), lambda ix: v[ix].mean()


def c_units(m, tag="dep_lead"):
    p = RES / f"c_{tag}_{m}.json"
    if not p.exists():
        return None
    d = json.load(open(p)); ev = d["per_event"]; nL = d["n_layers"]
    R = np.array([[e.get(f"L{l}", np.nan) for l in range(nL)] for e in ev], float)
    sc = np.array([e["scene"] for e in ev])
    ok = np.isfinite(R).all(1)
    R, sc = R[ok], sc[ok]

    def cm(ix):
        # 与 c_axis_hazard_patch.py:356 逐字同义：**逐事件**先算 top-2 占比再取均值，
        # 不是先平均剖面再取 top-2（两者不等，后者会低估）。
        r = np.clip(R[ix], 0, None)
        tot = r.sum(1)
        t2 = np.where(tot > 1e-9, np.sort(r, axis=1)[:, -2:].sum(1) / np.maximum(tot, 1e-12),
                      np.nan)
        return float(np.nanmean(t2))
    return R, sc, cm


def i_units(m):
    p = RES / "i_axis_domain.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    if m not in d["models"] or m not in d["summary_I_m"]:
        return None
    pool = d["models"][m]["pools"]["vision_mean"]
    Ls = d["summary_I_m"][m]["peak_layer"]
    v = np.array(pool["D_per_pair"][Ls], float)
    return v, np.array(pool["pair_scenes"]), lambda ix: 1.0 - v[ix].mean()


AXES = [("F-3  $r$", f3_units, BLUE), ("G-VS  selectivity", gvs_units, ORANGE),
        ("C  $C_m$", c_units, AQUA), ("I  $I_m$", i_units, VIOLET)]


def full_half_width(scenes, fn, rng, nb=2000):
    """全数据 scene-bootstrap 的半宽 h —— 该轴在这批材料上**自身的精度**。
    用它做容差单位，四轴才可比：偏差 0.5h 意味着"比材料本身的噪声还小"。"""
    uq = np.unique(scenes); idx = {s: np.where(scenes == s)[0] for s in uq}
    vals = []
    for _ in range(nb):
        ix = np.concatenate([idx[s] for s in rng.choice(uq, len(uq), replace=True)])
        try:
            v = fn(ix)
        except Exception:                                    # noqa: BLE001
            continue
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return None
    return float((np.percentile(vals, 97.5) - np.percentile(vals, 2.5)) / 2)


def curve(units, scenes, fn, rng):
    uq = np.unique(scenes)
    idx = {s: np.where(scenes == s)[0] for s in uq}
    N = len(uq)
    grid = sorted(set(np.unique(np.linspace(2, N, min(18, max(3, N - 1))).astype(int))))
    xs, lo, mid, hi, e50, e90 = [], [], [], [], [], []
    e85, e90b, e95 = [], [], []
    mean_ = []
    band = {q: ([], []) for q in (85, 90, 95)}   # 估计值本身的置信带
    for n in grid:
        vals = []
        for _ in range(B):
            pick = rng.choice(uq, n, replace=False)
            ix = np.concatenate([idx[s] for s in pick])
            try:
                v = fn(ix)
            except Exception:                                    # noqa: BLE001
                continue
            if np.isfinite(v):
                vals.append(v)
        if len(vals) < 10:
            continue
        xs.append(n); lo.append(np.percentile(vals, 10))
        mid.append(np.median(vals)); hi.append(np.percentile(vals, 90))
        mean_.append(float(np.mean(vals)))
        for q in (85, 90, 95):                   # 双侧等尾
            a = (100.0 - q) / 2.0
            band[q][0].append(float(np.percentile(vals, a)))
            band[q][1].append(float(np.percentile(vals, 100.0 - a)))
        # **误差**（单次抽样的值 − 终值）的绝对值：直接回答"只有 n 个 scene 时会偏多少"，
        # 且把偏倚与方差一并计入。注意不能用"子采样中位数 − 终值" ——
        # 那个近似无偏，任何 n 下都很小，量的是偏倚不是精度。
        fin_ = fn(np.arange(len(scenes)))
        err = np.abs(np.asarray(vals) - fin_)
        e50.append(float(np.median(err))); e90.append(float(np.percentile(err, 90)))
        # 常用置信水平下的误差界：|θ̂_n − θ̂_N| 的 85/90/95 分位
        for q, acc in ((85, e85), (90, e90b), (95, e95)):
            acc.append(float(np.percentile(err, q)))
    return (np.array(xs), np.array(lo), np.array(mid), np.array(hi),
            np.array(e50), np.array(e90),
            np.array(e85), np.array(e90b), np.array(e95),
            np.array(mean_), {q: (np.array(a), np.array(b))
                              for q, (a, b) in band.items()})


# 判据：**单次抽 n 个 scene** 时 |估计值 − 终值| 的 90 分位 <= h 且此后不再超出。
# 即"九成情况下，只用 n 个 scene 得到的答案与用满全部的差距不超过后者自身的精度"。
# h = 全数据 bootstrap 半宽，即"用满全部数据能达到的精度"。
# 曲线降到 1.0 ⇒ 只收集 n 个 scene 得到的数值，已和用满全部数据一样稳。
# 不用 |中位 − 终值|：多次子采样的中位数在任何 n 下都近似无偏，那量的是偏倚不是精度。
TOL = 1.0


def fig_need(curves, halves, finals):
    """每候选一格、四轴叠加：纵轴 = |median(n) - 终值| / h。

    这是"数值稳定"的直接度量 —— 曲线降到 TOL 以下并不再抬头，
    就是"再加数据也不会改变答案"。h 是该轴在这批材料上自身的精度，
    所以四个量纲差 20 倍以上的轴可以合法共用一个纵轴。
    """
    ms = [m for m in MODELS if curves.get(m)]
    fig, axes = plt.subplots(len(ms), 1, figsize=(4.6, 1.55 * len(ms)), sharex=True)
    if len(ms) == 1:
        axes = [axes]
    need_all = {}
    for ax, m in zip(axes, ms):
        needs = []
        for lab, col in [(a2[0], a2[2]) for a2 in AXES]:
            c = curves[m].get(lab)
            h = halves.get((m, lab)); fin = finals.get((m, lab))
            if c is None or not h:
                continue
            xs, lo, mid, hi, e50, e90, e85, e90b, e95, mn, bd = c
            dev = e90b / h                       # 主线：90% 置信
            band_lo, band_hi = e85 / h, e95 / h  # 带：85% / 95%
            ax.fill_between(xs, band_lo, band_hi, color=col, alpha=0.16, lw=0)
            ax.plot(xs, dev, color=col, lw=1.8, label=lab)
            ok = [i for i in range(len(xs)) if (dev[i:] <= TOL).all()]
            if ok:
                k = ok[0]; n = int(xs[k]); needs.append((n, lab, col))
                ax.plot([n], [dev[k]], "o", ms=5, color=col,
                        markeredgecolor="#fcfcfb", markeredgewidth=0.8, zorder=4)
        ax.axhline(TOL, color=GRID, lw=1.0, ls="--", zorder=0)
        if needs:
            nmax, lmax, cmax = max(needs, key=lambda z: z[0])
            need_all[m] = {"n_stable": nmax, "driven_by": lmax,
                           "per_axis": {l: n for n, l, _ in needs}}
            ax.axvline(nmax, color=cmax, lw=1.0, ls=":", alpha=0.85, zorder=1)
            ax.annotate(f"stable at {nmax}", (nmax, 5.2), fontsize=7, color=cmax,
                        ha="left", va="top", xytext=(3, 0), textcoords="offset points")
            ax.annotate("", (0, 0), (0, 0))
        ax.set_ylim(0, 6.0)
        ax.set_ylabel(NAME[m], fontsize=7, color=INK, rotation=0, ha="right",
                      va="center", labelpad=6)
        ax.tick_params(labelsize=7, colors=INK2, length=2)
        for sp2 in ("top", "right"):
            ax.spines[sp2].set_visible(False)
        for sp2 in ("left", "bottom"):
            ax.spines[sp2].set_color(GRID)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, 1.02))
    fig.supxlabel("number of scenes drawn at random", fontsize=8, color=INK2, y=0.005)
    axes[0].set_title("error of an $n$-scene study vs.\\ the reported value, in units "
                      "of the full-data precision $h$\n"
                      "line = 90% confidence, band = 85–95%   (dashed: $1h$)",
                      fontsize=7.5, color=INK2, pad=16)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.955))
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"convergence_need.{e}", dpi=300, bbox_inches="tight")
    print(f"[FIG] -> {OUT/'convergence_need.pdf'}")
    return need_all


def main():
    rng = np.random.default_rng(0)
    CURVES = defaultdict(dict); HALF = {}; FINAL = {}
    fig, axes = plt.subplots(len(MODELS), len(AXES),
                             figsize=(8.2, 1.35 * len(MODELS)))
    summary = {}
    LASTROW = {}
    for i, m in enumerate(MODELS):
        for j, (lab, fu, col) in enumerate(AXES):
            ax = axes[i, j]
            r = fu(m)
            if r is None:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        fontsize=7, color=INK2, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                continue
            units, sc, fn = r
            cur = curve(units, sc, fn, rng)
            xs = cur[0]
            if len(xs) == 0:
                continue
            CURVES[m][lab] = cur
            xs, lo, mid, hi, e50, e90, e85, e90b, e95, mn, bd = cur
            allix = np.arange(len(sc))
            try:
                FINAL[(m, lab)] = float(fn(allix))
            except Exception:                                # noqa: BLE001
                FINAL[(m, lab)] = float(mid[-1])
            HALF[(m, lab)] = full_half_width(sc, fn, np.random.default_rng(1))
            # 经典蒙特卡洛收敛图：均值线随 n 拉平，置信带随 n 收窄，
            # 最终裹住全量终值（虚线）。三层嵌套 = 85 / 90 / 95%。
            for q, al in ((95, 0.12), (90, 0.16), (85, 0.22)):
                a, b = bd[q]
                ax.fill_between(xs, a, b, color=col, alpha=al, lw=0)
            ax.plot(xs, mn, color=col, lw=1.8)
            ax.axhline(FINAL[(m, lab)], color=INK2, lw=0.9, ls="--", zorder=3)
            # 稳定性判据：末段 20% 的分位带宽 相对 全程带宽中位
            k = max(1, len(xs) // 5)
            w_end = float(np.median(hi[-k:] - lo[-k:]))
            w_all = float(np.median(hi - lo))
            summary[f"{m}|{lab}"] = {"n_max": int(xs[-1]), "value": float(mid[-1]),
                                     "band_end": w_end, "band_med": w_all,
                                     "shrink": w_end / w_all if w_all > 0 else None}
            ax.tick_params(labelsize=6, colors=INK2, length=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            if i == 0:
                ax.set_title(lab, fontsize=8, color=INK, pad=4)
            if j == 0:
                ax.set_ylabel(NAME[m], fontsize=7, color=INK, rotation=0,
                              ha="right", va="center", labelpad=6)
            LASTROW[j] = i               # 该列最后一个有数据的行
    for j in range(len(AXES)):
        for i in range(len(MODELS)):
            if CURVES[MODELS[i]].get(AXES[j][0]) is not None and i != LASTROW.get(j):
                axes[i, j].set_xticklabels([])
    # 图例：均值线 / 三层嵌套带 / 全量终值
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    g = "#52514e"
    hs = [Line2D([], [], color=g, lw=1.8, label="mean of subsamples"),
          Patch(facecolor=g, alpha=0.22, label="85% CI"),
          Patch(facecolor=g, alpha=0.16, label="90% CI"),
          Patch(facecolor=g, alpha=0.12, label="95% CI"),
          Line2D([], [], color=INK2, lw=0.9, ls="--",
                 label="full-pool value $\\hat\\theta_N$")]
    fig.legend(handles=hs, loc="upper center", ncol=5, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.55, 1.005))
    fig.supxlabel("number of scenes drawn at random (deployment / lead pool)",
                  fontsize=8, color=INK2, y=0.01)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.975))
    OUT.mkdir(parents=True, exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"convergence.{e}", dpi=300, bbox_inches="tight")
    # 曲线原始数据落盘，供交互版（bokeh）复用，避免重算
    dump = {"tolerance_h": TOL, "B": B, "curves": []}
    for m, per in CURVES.items():
        for lab, (xs, lo, mid, hi, e50, e90, e85, e90b, e95, mn, bd) in per.items():
            h = HALF.get((m, lab)); fin = FINAL.get((m, lab))
            if not h:
                continue
            dump["curves"].append({
                "model": m, "axis": lab, "h": h, "final": fin,
                "n": [int(x) for x in xs],
                "median": [float(v) for v in mid],
                "p10": [float(v) for v in lo], "p90": [float(v) for v in hi],
                "spread_h": [float((b - a) / 2 / h) for a, b in zip(lo, hi)],
                "err50_h": [float(v / h) for v in e50],
                "err85_h": [float(v / h) for v in e85],
                "err90_h": [float(v / h) for v in e90b],
                "err95_h": [float(v / h) for v in e95],
                "err85_abs": [float(v) for v in e85],
                "err90_abs": [float(v) for v in e90b],
                "err95_abs": [float(v) for v in e95],
                # 经典 MC 收敛图所需：估计值本身的均值与三层嵌套置信带
                "mean": [float(v) for v in mn],
                **{f"ci{q}_{side}": [float(v) for v in bd[q][k]]
                   for q in (85, 90, 95) for k, side in ((0, "lo"), (1, "hi"))},
            })
    (RES / "convergence_curves.json").write_text(
        json.dumps(dump, indent=1, ensure_ascii=False))
    print(f"[DUMP] wrote {RES/'convergence_curves.json'}")

    need = fig_need(CURVES, HALF, FINAL)
    ci_rows, ci_spread = fig_ci(CURVES, HALF, FINAL)
    (RES / "convergence_ci.json").write_text(json.dumps(
        {"rows": ci_rows,
         "effect_spread": {k: v for k, v in ci_spread.items()}},
        indent=1, ensure_ascii=False))
    print()
    print(f"{'候选/轴':<34}{'85%':>7}{'90%':>7}{'95%':>7}   "
          f"{'←达到 1h 所需 n':<18}{'85%':>7}{'90%':>7}{'95%':>7}  ←分辨相邻候选(Δ)所需 n")
    print("-" * 118)
    byk = defaultdict(dict)
    for r in ci_rows:
        byk[(r["model"], r["axis"])][r["conf"]] = r
    for (m, lab), d in byk.items():
        def g(q, k):
            v = d.get(q, {}).get(k)
            return f"{v:>7}" if v else f"{'>'+str(d[q]['n_max']):>7}"
        print(f"{NAME[m]+' / '+lab:<34}" + "".join(g(q, "n_1h") for q in (85, 90, 95))
              + " " * 21 + "".join(g(q, "n_effect") for q in (85, 90, 95)))
    (RES / "convergence_summary.json").write_text(
        json.dumps({"B": B, "tolerance_h": TOL,
                    "tolerance": TOL,
                    "note": "随机 scene 子集，重复 B 次；need = |中位 − 终值| 首次并持续 "
                            "<= TOL×h 所需的 scene 数（h = 全数据 bootstrap 半宽）",
                    "cells": summary, "need": need}, indent=2, ensure_ascii=False))
    print(f"[FIG] -> {OUT/'convergence.pdf'}")
    print(f"\n{'候选':<20}{'四轴数值都稳所需 scene':>24}{'  由哪一轴决定'}")
    print("-" * 62)
    for m, v in need.items():
        print(f"{NAME[m]:<20}{v['n_stable']:>24}   {v['driven_by']}"
              f"   逐轴 {v['per_axis']}")
    print(f"\n{'候选/轴':<34}{'n_max':>7}{'终值':>10}{'末段带宽':>10}{'带宽收缩比':>12}")
    print("-" * 74)
    for k, v in summary.items():
        m, lab = k.split("|")
        sh = v["shrink"]
        print(f"{NAME[m]+' / '+lab:<34}{v['n_max']:>7}{v['value']:>10.3f}"
              f"{v['band_end']:>10.3f}{(sh if sh else float('nan')):>12.2f}")




# ---------------------------------------------------------------- CI 版本 ----
CONF = [(85, ":", 1.0), (90, "-", 1.8), (95, "--", 1.1)]


def fig_ci(curves, halves, finals):
    """绝对误差（各轴原生单位）随 n 的变化，三个常用置信水平各一条。

    读法：横轴 n 个场景的一次研究，其读数与全量终值之差，在 85/90/95%
    的抽样中不超过纵轴值。与 convergence_need 的区别是这里不除以 h，
    所以可以直接对着轴本身的量纲判断"够不够准"。
    """
    ms = [m for m in MODELS if curves.get(m)]
    fig, axes = plt.subplots(len(ms), len(AXES),
                             figsize=(8.4, 1.5 * len(ms)), squeeze=False)
    # 每轴的"效应量" Δ = 候选终值排序后**相邻间隔的中位数**。
    # 用极差会被单个离群候选（如 SimLingo 的 F-3 r=1.51）撑大，
    # 使"分辨两个模型"看上去几乎不要数据；相邻间隔才是实际要分开的尺度。
    spread = {}
    for lab, _, _ in AXES:
        vs = sorted(finals[(m, lab)] for m in ms if (m, lab) in finals)
        spread[lab] = float(np.median(np.diff(vs))) if len(vs) > 1 else None
    rows = []
    last_row = {}
    for i, m in enumerate(ms):
        for j, (lab, _fu, col) in enumerate(AXES):
            ax = axes[i][j]
            c = curves[m].get(lab)
            if c is None:
                ax.text(0.5, 0.5, "not measured", ha="center", va="center",
                        fontsize=6.5, color=INK2, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if i == 0:
                    ax.set_title(lab, fontsize=8, color=INK, pad=4)
                continue
            xs, lo, mid, hi, e50, e90, e85, e90b, e95, mn, bd = c
            h = halves.get((m, lab)); d = spread.get(lab)
            for (q, ls, lw), e in zip(CONF, (e85, e90b, e95)):
                ax.plot(xs, e, color=col, lw=lw, ls=ls,
                        label=f"{q}%" if (i == 0 and j == 0) else None)
                # 达到目标所需 n：1h（精度口径）与 0.5Δ（效应量口径）
                rec = {"model": m, "axis": lab, "conf": q}
                for key, tgt in (("n_1h", h), ("n_effect", d if d else None)):
                    if not tgt:
                        rec[key] = None; continue
                    ok = [k for k in range(len(xs)) if (e[k:] <= tgt).all()]
                    rec[key] = int(xs[ok[0]]) if ok else None
                rec["err_abs_at_max"] = float(e[-1]); rec["n_max"] = int(xs[-1])
                rows.append(rec)
            if h:
                ax.axhline(h, color=GRID, lw=0.9, ls="--", zorder=0)
            # 左端 n≈2 的误差极大，会把"曲线穿过目标线"的区段压扁；
            # 顶端截到 8h 上下，被截掉的只是最左几个点
            top = float(e95[0]) * 1.05
            if h:
                top = min(top, 8.0 * h)
            ax.set_ylim(0, top)
            ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
            ax.tick_params(labelsize=6, colors=INK2, length=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            if i == 0:
                ax.set_title(lab, fontsize=8, color=INK, pad=4)
            if j == 0:
                ax.set_ylabel(NAME[m], fontsize=7, color=INK, rotation=0,
                              ha="right", va="center", labelpad=6)
            last_row[j] = i          # 该列最后一个有数据的行
    for j in range(len(AXES)):
        for i in range(len(ms)):
            if curves[ms[i]].get(AXES[j][0]) is not None and i != last_row.get(j):
                axes[i][j].set_xticklabels([])
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, 1.015), title="confidence",
               title_fontsize=7.5)
    fig.supxlabel("number of scenes drawn at random", fontsize=8,
                  color=INK2, y=0.005)
    fig.supylabel("absolute error vs. the full-pool value (axis\u2019 own units)",
                  fontsize=8, color=INK2, x=0.005)
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.965))
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"convergence_ci.{e}", dpi=300, bbox_inches="tight")
    print(f"[FIG] -> {OUT/'convergence_ci.pdf'}")
    return rows, spread


if __name__ == "__main__":
    main()
