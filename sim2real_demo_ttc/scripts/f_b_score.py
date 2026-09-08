"""F 轴第一部分：把 b 本身当分数报，**按左右舵分组**（PRINCIPLES P-1）。

    benchmark = 左舵 LHD（Boston / Las Vegas / Pittsburgh）
    deployment = 右舵 RHD（Singapore）

语料（nuScenes / NAVSIM）**不再是分组变量**，降级为协变量：两个语料的事件
先合并成一个池，再按舵向切。旧的"nuScenes=benchmark, NAVSIM=deployment"已作废。

## 为什么报 b 而不是 r
r = mean(b_model)/mean(b_GT) 把人类侧的噪声乘进来，且分母接近 0 时不稳。
b_model 是物理量（m/s，全轨迹弧长速度的变化），与 b_GT 并排放就能读出
"物理一致程度"，不必先除。

## 负值量必须与均值并排
mean(b)≈0 可能是"没反应"，也可能是"正负抵消"。故同时报 frac_neg 与
rho = Σ|b<0| / Σ(b>0)。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
RD = "arc_full"
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
# 四格材料按 (场景类型, 语料) 编址；语料只用来定位文件，不再是分组变量
SRC = {("ghost", "nuscenes"): "f3_gt_axis_full_{m}.json",
       ("lead", "nuscenes"): "gt_lead_{m}.json",
       ("ghost", "navsim"): "f3_gt_dep_ghost_{m}.json",
       ("lead", "navsim"): "f3_gt_dep_lead_{m}.json"}
_MAP = json.load(open(RES / "driveside_map.json"))


def side_of(scene, corpus):
    if corpus == "nuscenes":
        return _MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):                 # 复合键，见 P-4
        s = _MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s
    return None


def boot_scene(v, sc, n=5000, seed=0):
    v = np.asarray(v, float); sc = np.asarray(sc)
    uq = np.unique(sc); ix = [np.where(sc == s)[0] for s in uq]
    if len(uq) < 5:
        return None
    rng = np.random.default_rng(seed)
    out = [v[np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])].mean()
           for _ in range(n)]
    return {"mean": float(v.mean()),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n": int(len(v)), "n_scenes": int(len(uq))}


def collect(scen, m):
    """把两个语料的同类场景事件合并，逐事件贴上舵向标签。"""
    bm, bg, bc, sc, sd = [], [], [], [], []
    for corpus in ("nuscenes", "navsim"):
        p = RES / SRC[(scen, corpus)].format(m=m)
        if not p.exists():
            continue
        for e in json.load(open(p))["per_event"]:
            s = side_of(e["scene"], corpus)
            if s is None:
                continue
            bm.append(e[f"b_model__{RD}"]); bg.append(e[f"b_gt__{RD}"])
            bc.append(e.get(f"b_ctrl__{RD}", np.nan))
            sc.append(f"{corpus}|{e['scene']}"); sd.append(s)
    return (np.array(bm), np.array(bg), np.array(bc),
            np.array(sc), np.array(sd))


def cell(scen, m, side):
    bm, bg, bc, sc, sd = collect(scen, m)
    if len(bm) == 0:
        return None
    k = sd == side
    if k.sum() == 0:
        return None
    bm, bg, sc = bm[k], bg[k], sc[k]
    pos, neg = bm[bm > 0].sum(), -bm[bm < 0].sum()
    return {"b_model": boot_scene(bm, sc), "b_gt": boot_scene(bg, sc),
            "frac_neg": float((bm < 0).mean()),
            "rho": float(neg / pos) if pos > 1e-9 else None,
            "ratio": float(bm.mean() / bg.mean()) if abs(bg.mean()) > 1e-9 else None,
            "n": int(len(bm)), "n_scenes": int(len(np.unique(sc)))}


def main():
    out = {}
    for scen in ("ghost", "lead"):
        print(f"\n{'='*100}\n场景 = {scen}    b 单位 = m/s（全轨迹弧长速度）"
              f"    分组 = 舵向（P-1）\n{'='*100}")
        print(f"{'候选':<18}{'域':<20}{'n':>5}{'scene':>6}"
              f"{'b_model [CI]':>26}{'b_GT':>9}{'一致度':>8}{'负值':>7}{'ρ':>7}")
        print("-" * 100)
        for m in MODELS:
            for side, lab in (("LHD", "benchmark (LHD)"), ("RHD", "deployment (RHD)")):
                c = cell(scen, m, side)
                if c is None:
                    continue
                out[f"{m}|{side}|{scen}"] = c
                v = c["b_model"]
                s = (f"{v['mean']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]"
                     if v else f"{np.nan if not c else ''}不可估(<5 scene)")
                g = f"{c['b_gt']['mean']:+.3f}" if c["b_gt"] else "--"
                r = f"{c['ratio']:+.3f}" if c["ratio"] is not None else "--"
                rho = f"{c['rho']:.2f}" if c["rho"] is not None else "--"
                print(f"{NAME[m]:<18}{lab:<20}{c['n']:>5}{c['n_scenes']:>6}"
                      f"{s:>26}{g:>9}{r:>8}{c['frac_neg']*100:>6.0f}%{rho:>7}")
    (RES / "f_b_score.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {RES/'f_b_score.json'}")


if __name__ == "__main__":
    main()
