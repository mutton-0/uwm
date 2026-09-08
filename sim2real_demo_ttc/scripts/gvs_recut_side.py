"""G-VS 按 P-1 重切：左舵=benchmark，右舵=deployment。逐 scene mIoU 重新分组，不重跑 GPU。

selectivity = mIoU(trained) − mIoU(random_init)，逐 scene 先作差再 scene 级 bootstrap
（与 gvs2_probe.py 同口径）。position_only 作为"只靠位置先验能到多少"的地板一并报。
"""
import json, glob, os
import numpy as np

RES = "results"
MAP = json.load(open(f"{RES}/driveside_map.json"))


def side_of(scene, corpus):
    if corpus == "nusc":
        return MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):
        s = MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s


def boot(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else None, "ci95": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
            "n": int(len(v))}


pool = {}
for p in sorted(glob.glob(f"{RES}/gvs_new_*.json")):
    b = os.path.basename(p)[:-5].split("_")      # gvs new <scen> <corpus> <model>
    scen, co, m = b[2], b[3], "_".join(b[4:])
    d = json.load(open(p))
    tr = d["arms"]["trained"]["per_scene_miou"]
    ri = d["arms"]["random_init"]["per_scene_miou"]
    po = d["arms"]["position_only"]["per_scene_miou"]
    for s in tr:
        sd = side_of(s, co)
        if sd is None or s not in ri:
            continue
        pool.setdefault((scen, m, sd), []).append(
            (tr[s] - ri[s], tr[s] - po.get(s, np.nan), co))

rows = []
print(f"{'场景':<7}{'候选':<10}{'benchmark (LHD)':>30}{'n':>5}"
      f"{'deployment (RHD)':>30}{'n':>5}")
for scen in ("ghost", "lead"):
    for m in ("simlingo", "dd", "ltf", "ddv2"):
        cell = {"scenario": scen, "model": m}
        txt = []
        for sd, dom in (("LHD", "benchmark"), ("RHD", "deployment")):
            v = pool.get((scen, m, sd), [])
            sel = boot([x[0] for x in v])
            cell[f"selectivity_{dom}"] = sel
            cell[f"n_{dom}"] = len(v)
            cell[f"corpus_mix_{dom}"] = {c: sum(1 for x in v if x[2] == c)
                                         for c in ("nusc", "navsim")}
            cell[f"vs_position_only_{dom}"] = boot([x[1] for x in v])
            txt.append((f"{sel['mean']:+.4f} [{sel['ci95'][0]:+.3f},{sel['ci95'][1]:+.3f}]"
                        if sel["ci95"] else "—").rjust(30))
            txt.append(f"{len(v):>5}")
        rows.append(cell)
        print(f"{scen:<7}{m:<10}" + "".join(txt))

json.dump({"principle": "P-1：左舵=benchmark，右舵=deployment；两语料合并后按舵位切",
           "metric": "selectivity = mIoU(trained) − mIoU(random_init)，逐 scene 作差后 scene 级 bootstrap",
           "note": "逐 scene mIoU 重新分组，未重跑 GPU",
           "cells": rows}, open(f"{RES}/gvs_axis_by_side.json", "w"),
          ensure_ascii=False, indent=1)
print(f"\n-> {RES}/gvs_axis_by_side.json")
