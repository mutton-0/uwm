"""四场景 × 两语料 × 四候选 的 GT 轴主表（arc_full 主口径）。

汇总所有已完成场景的 F-3 GT 轴结果，输出：
  1. 响应比 r = mean(b_model)/mean(b_GT) 及 scene-bootstrap CI
  2. 方向一致率（b_model 与 b_GT 同号的事件占比）
  3. 反向抵消率 = sum_{b<0}|b| / sum_{b>0} b
无任何按结果的样本剔除。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
RD = "arc_full"
MODELS = ["simlingo", "dd", "ltf", "ddv2"]
# (场景, 语料) -> 文件名模板
FILES = {
    ("ghost", "nuscenes"):     "f3_gt_axis_full_{m}.json",
    ("ghost", "navsim"):       "f3_gt_ghostfx_navsim_{m}.json",
    ("lead", "nuscenes"):      "gt_lead_{m}.json",
    ("lead", "navsim"):        "f3_gt_leadfx_navsim_{m}.json",
}


def boot_ratio(scenes, bm, bg, n=5000, seed=0):
    """scene-level bootstrap 的响应比 CI（分子分母同抽同一批 scene）。"""
    rng = np.random.default_rng(seed)
    uq = np.unique(scenes)
    idx = {s: np.where(scenes == s)[0] for s in uq}
    out = []
    for _ in range(n):
        pick = rng.choice(uq, len(uq), replace=True)
        sel = np.concatenate([idx[s] for s in pick])
        d = bg[sel].mean()
        if abs(d) < 1e-9:
            continue
        out.append(bm[sel].mean() / d)
    if not out:
        return (float("nan"), float("nan"))
    return tuple(np.percentile(out, [2.5, 97.5]))


def load(scen, corp, m):
    p = RES / FILES[(scen, corp)].format(m=m)
    if not p.exists():
        return None
    d = json.load(open(p))
    ev = d["per_event"]
    key_m, key_g = f"b_model__{RD}", f"b_gt__{RD}"
    if key_m not in ev[0]:                       # 旧文件可能只有单读数
        key_m, key_g = "b_model", "b_gt"
        if key_m not in ev[0]:
            return None
    bm = np.array([e[key_m] for e in ev], float)
    bg = np.array([e[key_g] for e in ev], float)
    sc = np.array([e["scene"] for e in ev])
    pos, neg = bm[bm > 0].sum(), -bm[bm < 0].sum()
    return dict(n=len(ev), n_scene=len(np.unique(sc)),
                bm=float(bm.mean()), bg=float(bg.mean()),
                ratio=float(bm.mean() / bg.mean()),
                ci=boot_ratio(sc, bm, bg),
                agree=int((np.sign(bm) == np.sign(bg)).sum()),
                cancel=float(neg / pos) if pos > 1e-9 else float("nan"))


def main():
    rows, out = [], {}
    for (scen, corp) in FILES:
        for m in MODELS:
            r = load(scen, corp, m)
            if r is None:
                continue
            out[f"{scen}|{corp}|{m}"] = r
            rows.append((scen, corp, m, r))
    hdr = (f"{'场景':<14}{'语料':<10}{'候选':<10}{'n(ev/sc)':>10}"
           f"{'b_GT':>9}{'b_model':>9}{'响应比 r':>11}"
           f"{'  r 的 95%CI':<22}{'方向一致':>10}{'反向抵消':>10}")
    print(f"\n主口径 = {RD}（全轨迹弧长 / 时域）；r>0 且 CI 不跨 0 ⇒ 有正向响应\n")
    print(hdr); print("-" * 118)
    last = None
    for scen, corp, m, r in rows:
        if last and last != (scen, corp):
            print()
        last = (scen, corp)
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        print(f"{scen:<14}{corp:<10}{m:<10}{r['n']:>4}/{r['n_scene']:<5}"
              f"{r['bg']:>+9.3f}{r['bm']:>+9.3f}{r['ratio']:>+11.3f}"
              f"  {ci:<20}{r['agree']}/{r['n']:<6}{r['cancel']:>9.1%}")
    (RES / "f3_master_table.json").write_text(
        json.dumps({"readout": RD, "cells": out}, indent=2, ensure_ascii=False))
    print(f"\n[MASTER] wrote {RES/'f3_master_table.json'}")


if __name__ == "__main__":
    main()
