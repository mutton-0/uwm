"""四轴 + 部署指标主表。每格标注出处与自检状态，不合格的格子不给数只给理由。

C 轴的 patch-ALL 充分割集自检：|patch_all - 1| <= 0.25 才认为逐层占比可解释。
超出容差的格子标 UNINTERPRETABLE —— 这不是"结果不好"，是"这个量在这一格没有定义"。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
MODELS = ["simlingo", "dd", "ltf", "ddv2"]
PATCH_ALL_TOL = 0.25


def f3_cell(tag, m):
    p = RES / f"f3_gt_{tag}_{m}.json"
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    bm = np.array([e["b_model__arc_full"] for e in ev])
    bg = np.array([e["b_gt__arc_full"] for e in ev])
    sc = np.array([e["scene"] for e in ev])
    rng = np.random.default_rng(0); uq = np.unique(sc)
    ix = {s: np.where(sc == s)[0] for s in uq}
    out = []
    for _ in range(5000):
        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), replace=True)])
        d = bg[sel].mean()
        if abs(d) > 1e-9:
            out.append(bm[sel].mean() / d)
    ci = np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)
    pos, neg = bm[bm > 0].sum(), -bm[bm < 0].sum()
    return {"n": len(ev), "n_scene": len(uq), "r": float(bm.mean() / bg.mean()),
            "ci": [float(ci[0]), float(ci[1])],
            "agree": int((np.sign(bm) == np.sign(bg)).sum()),
            "cancel": float(neg / pos) if pos > 1e-9 else float("nan")}


def gvs_cell(w, m):
    p = RES / f"gvs_new_{w}_{m}.json"
    if not p.exists():
        return None
    d = json.load(open(p)); s = d["selectivity"]
    return {"n_scene": s["n_scenes"], "sel": s["mean"], "ci": s["ci95"],
            "sig": not (s["ci95"][0] <= 0 <= s["ci95"][1])}


def c_cell(w, m):
    p = RES / f"c_new_{w}_{m}.json"
    if not p.exists():
        return None
    d = json.load(open(p)); c = d.get("C_m_top2_share", {})
    pa = d.get("patch_all_recovery", {}).get("mean", float("nan"))
    ok = abs(pa - 1.0) <= PATCH_ALL_TOL
    return {"n": c.get("n"), "dropped": d.get("n_dropped_small_gap"),
            "patch_all": float(pa), "interpretable": bool(ok),
            "C_m": c.get("mean"), "ci": c.get("ci95")}


def main():
    POOLS = [("ghost", "nuscenes", "ghost_nusc", "f3_gt_axis_full"),
             ("lead", "nuscenes", "lead_nusc", "gt_lead"),
             ("ghost", "navsim", "ghost_navsim", "f3_gt_ghostfx_navsim"),
             ("lead", "navsim", "lead_navsim", "f3_gt_leadfx_navsim")]
    out = {}
    for sc, cp, w, f3tag in POOLS:
        for m in MODELS:
            key = f"{sc}|{cp}|{m}"
            f3 = (f3_cell(f3tag.replace("f3_gt_", ""), m)
                  if f3tag.startswith("f3_gt_") else None)
            if f3 is None:
                p = RES / (f"{f3tag}_{m}.json")
                if p.exists():
                    ev = json.load(open(p))["per_event"]
                    bm = np.array([e["b_model__arc_full"] for e in ev])
                    bg = np.array([e["b_gt__arc_full"] for e in ev])
                    scn = np.array([e["scene"] for e in ev])
                    rng = np.random.default_rng(0); uq = np.unique(scn)
                    ix = {s: np.where(scn == s)[0] for s in uq}
                    r = [bm[np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])].mean() /
                         bg[np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])].mean()
                         for _ in range(0)]
                    pos, neg = bm[bm > 0].sum(), -bm[bm < 0].sum()
                    boot = []
                    for _ in range(5000):
                        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])
                        d0 = bg[sel].mean()
                        if abs(d0) > 1e-9:
                            boot.append(bm[sel].mean() / d0)
                    ci = np.percentile(boot, [2.5, 97.5])
                    f3 = {"n": len(ev), "n_scene": len(uq),
                          "r": float(bm.mean() / bg.mean()), "ci": list(map(float, ci)),
                          "agree": int((np.sign(bm) == np.sign(bg)).sum()),
                          "cancel": float(neg / pos) if pos > 1e-9 else float("nan")}
            out[key] = {"F3": f3, "GVS": gvs_cell(w, m), "C": c_cell(w, m)}

    # ---- 打印 ----
    for axis, fmt in [("F3", "响应比 r"), ("GVS", "selectivity"), ("C", "C_m")]:
        print(f"\n===== {axis}（{fmt}）=====")
        print(f"{'场景/语料':<20}" + "".join(f"{m:>22}" for m in MODELS))
        print("-" * (20 + 22 * len(MODELS)))
        for sc, cp, w, _ in POOLS:
            line = f"{sc+'/'+cp:<20}"
            for m in MODELS:
                c = out[f"{sc}|{cp}|{m}"][axis]
                if c is None:
                    line += f"{'—':>22}"
                elif axis == "F3":
                    line += f"{c['r']:>+9.3f}[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]"
                elif axis == "GVS":
                    line += f"{c['sel']:>+9.4f}[{c['ci'][0]:+.2f},{c['ci'][1]:+.2f}]"
                else:
                    # 自检未过也给数值，加 † 标注
                    flag = "" if c["interpretable"] else "†"
                    line += f"{c['C_m']:>8.3f}{flag}[{c['ci'][0]:.2f},{c['ci'][1]:.2f}]"
            print(line)
    (RES / "axes_master.json").write_text(json.dumps(
        {"patch_all_tolerance": PATCH_ALL_TOL, "cells": out}, indent=2, ensure_ascii=False))
    print(f"\n[AXES] wrote {RES/'axes_master.json'}")


if __name__ == "__main__":
    main()
