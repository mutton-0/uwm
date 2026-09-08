"""C 轴按 P-1 重切：左舵=benchmark，右舵=deployment。

c_new_* 是逐事件逐层的 recovery 记录，重切只是换分组，**不需要重跑 GPU**。
C_m 的算法逐字沿用 c_axis_hazard_patch.py：recovery 先 clip 至 ≥0，
逐事件算 top-2 层占比，再对事件做 bootstrap。
"""
import json, glob, os, re
import numpy as np

RES = "results"
MAP = json.load(open(f"{RES}/driveside_map.json"))


def boot(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else None, "ci95": None, "n": int(len(v))}
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def side_of(scene, corpus):
    if corpus == "nusc":
        return MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):
        s = MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s
    return None


def rows(p):
    d = json.load(open(p))
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "scene" in v[0]:
            return d, v
    return d, []


def cm_of(recs, nL):
    R = np.array([[r[f"L{l}"] for l in range(nL)] for r in recs], float)
    Xc = np.clip(R, 0, None)
    top2 = np.array([np.sort(x)[-2:].sum() / x.sum() if x.sum() > 1e-9 else np.nan for x in Xc])
    prof = Xc.mean(0)
    return boot(top2), prof, 2.0 / nL


out = {}
for p in sorted(glob.glob(f"{RES}/c_new_*.json")):
    b = os.path.basename(p)[:-5].split("_")
    sc, co, m = b[2], b[3], "_".join(b[4:])
    d, rs = rows(p)
    nL = d["n_layers"]
    for r in rs:
        r["_side"] = side_of(r["scene"], co)
        r["_corpus"] = co
    out.setdefault((sc, m), []).extend(rs)
    out.setdefault(("_nL", m), nL)

report = []
for (sc, m), rs in sorted(out.items()):
    if sc == "_nL":
        continue
    nL = out[("_nL", m)]
    row = {"scenario": sc, "model": m, "n_layers": nL}
    for side, dom in (("LHD", "benchmark"), ("RHD", "deployment")):
        sub = [r for r in rs if r["_side"] == side]
        row[f"n_{dom}"] = len(sub)
        row[f"corpus_mix_{dom}"] = {c: sum(1 for r in sub if r["_corpus"] == c)
                                    for c in ("nusc", "navsim")}
        if len(sub) >= 2:
            cm, prof, base = cm_of(sub, nL)
            row[f"C_m_{dom}"] = cm
            row[f"profile_{dom}"] = [float(x) for x in prof]
            row["diffuse_baseline"] = base
        else:
            row[f"C_m_{dom}"] = None
    row["n_unmapped"] = sum(1 for r in rs if r["_side"] is None)
    report.append(row)

json.dump({"principle": "P-1：左舵(LHD)=benchmark，右舵(RHD)=deployment；语料(nuScenes/NAVSIM)合并后按舵位切",
           "note": "逐事件记录重新分组，未重跑模型",
           "cells": report}, open(f"{RES}/c_axis_by_side.json", "w"),
          ensure_ascii=False, indent=1)

print(f"{'场景':<6}{'模型':<10}{'bench n':>8}{'C_m^bench':>22}{'dep n':>7}{'C_m^dep':>22}{'弥散':>7}")
for r in report:
    def f(d):
        c = r.get(f"C_m_{d}")
        if not c or c["ci95"] is None:
            return "—".rjust(22)
        return f"{c['mean']:.3f} [{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]".rjust(22)
    print(f"{r['scenario']:<6}{r['model']:<10}{r['n_benchmark']:>8}{f('benchmark')}"
          f"{r['n_deployment']:>7}{f('deployment')}{r.get('diffuse_baseline',float('nan')):>7.2f}")
    if r["n_unmapped"]:
        print(f"       ! {r['n_unmapped']} 个事件无舵位映射")
