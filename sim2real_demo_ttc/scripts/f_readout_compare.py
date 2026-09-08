"""哪个 F 读数更有效？—— 用同一把尺子量 b、r 和 v_faith 夹角。

"有效"在这里有明确定义，不是看谁的数好看：

  σ    = 每场景尺度 = SE_N·√N，与池子大小无关
  Δ    = 候选终值排序后**相邻间隔的中位数**（不用极差，避免被单个离群候选撑大）
  σ/Δ  = 噪声与信号之比 —— **这个比值就是全部**，因为分辨相邻候选所需的
         n = (1.96√2·σ/Δ)²，只由它决定

σ/Δ 越小越有效：同样的数据量能分辨更细的差别。
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
        "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
RD = "arc_full"


def per_event(cellfile, m):
    p = RES / cellfile.format(m=m)
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    return (np.array([e[f"b_model__{RD}"] for e in ev]),
            np.array([e[f"b_gt__{RD}"] for e in ev]),
            np.array([e["scene"] for e in ev]))


def boot(vals, sc, fn, n=4000, seed=0):
    """scene 级 bootstrap，fn 把选中的下标映射成一个标量。"""
    uq = np.unique(sc); ix = [np.where(sc == s)[0] for s in uq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])
        v = fn(sel)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def main():
    cellfile = "f3_gt_dep_lead_{m}.json"        # lead / **benchmark 侧**：数据最多的一格。
    # 按 P-1 这批实测 100% 左舵（维加斯+匹兹堡），属 benchmark；文件名 "dep" 是旧划分遗留。
    READOUTS = {
        "b_model (m/s)": lambda bm, bg: (lambda s: bm[s].mean()),
        "r = b_model/b_GT": lambda bm, bg: (lambda s: bm[s].mean() / bg[s].mean()
                                            if abs(bg[s].mean()) > 1e-9 else np.nan),
    }
    res = {}
    for rname, mk in READOUTS.items():
        fin, se = {}, {}
        for m in NAME:
            d = per_event(cellfile, m)
            if d is None:
                continue
            bm, bg, sc = d
            f = mk(bm, bg)
            allix = np.arange(len(bm))
            fin[m] = float(f(allix))
            bs = boot(bm, sc, f)
            N = len(np.unique(sc))
            se[m] = {"se": float(np.std(bs)), "N": N,
                     "sigma": float(np.std(bs) * np.sqrt(N))}
        vs = sorted(fin.values())
        delta = float(np.median(np.diff(vs)))
        res[rname] = {"final": fin, "se": se, "delta": delta}

    print("lead / benchmark(左舵) 这一格（N≈304 scene）上，两种读数的分辨力对比")
    print(f"\n{'读数':<20}{'Δ(相邻间隔)':>13}{'σ 范围':>18}{'σ/Δ 范围':>14}"
          f"{'排名所需 n':>14}")
    print("-" * 82)
    for rname, r in res.items():
        sg = [v["sigma"] for v in r["se"].values()]
        rat = [s / abs(r["delta"]) for s in sg]
        need = [(1.96 * np.sqrt(2) * x) ** 2 for x in rat]
        print(f"{rname:<20}{r['delta']:>13.4f}"
              f"{f'{min(sg):.3f}–{max(sg):.3f}':>18}"
              f"{f'{min(rat):.1f}–{max(rat):.1f}':>14}"
              f"{f'{min(need):.0f}–{max(need):.0f}':>14}")

    print(f"\n{'候选':<18}" + "".join(f"{k:>26}" for k in res))
    print("-" * 72)
    for m in NAME:
        line = f"{NAME[m]:<18}"
        for rname, r in res.items():
            if m in r["final"]:
                line += f"{r['final'][m]:+.3f} ± {r['se'][m]['se']:.3f}".rjust(26)
            else:
                line += f"{'--':>26}"
        print(line)
    (RES / "f_readout_compare.json").write_text(json.dumps(res, indent=1))
    print(f"\n-> {RES/'f_readout_compare.json'}")


if __name__ == "__main__":
    main()
