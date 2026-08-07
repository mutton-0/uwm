"""配对覆盖审计：几何匹配是不是把信号最强的近距正例系统性筛掉了？

动机：S1 曲线显示缓存里 d_long<10m 只有 18 个事件，而全量挖掘有 332 个。
若近距正例大而居中、D2 池里没有同几何的无害对象可配，caliper 匹配就会把它们全丢掉——
那么"读不出"里有一部分是**"把信号最强的样本筛掉之后读不出"**，属于设计缺陷而非模型属性。

本脚本零算力，只查账：
  ① 逐 d_long 箱的**匹配存活率** = 该箱正例进入 matched_* 的比例；
  ② 同箱 D2 负例池的可用量与几何重叠（看是"没有可配的"还是"配了但被 caliper 拒了"）；
  ③ 存活与被淘汰正例的几何/危险度对比（被淘汰的是不是更近、更大、TTC 更小）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--neg-types", default="D2a,D2b,D2c,D2cV")
    ap.add_argument("--bins", default="0,10,15,20,25,30,40,50,200")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    by_id = {e["event_id"]: e for e in evs}
    pos = [e for e in evs if e["event_type"] == "A"]
    edges = [float(x) for x in args.bins.split(",")]

    print(f"[审计] 全量正例 A n={len(pos)}   d_long 分箱 {edges}")

    for nt in args.neg_types.split(","):
        f = work / "mining" / f"matched_{nt}.txt"
        if not f.exists():
            continue
        keep = set(f.read_text().split())
        src = nt.rstrip("V") if nt.endswith("V") else nt
        pool = [e for e in evs if e["event_type"] == src]

        print(f"\n{'='*86}\n负类 {nt}：池 n={len(pool)}，匹配清单里共 {len(keep)} 个 id")
        print(f"{'d_long 箱[m]':>14} {'正例 n':>7} {'存活':>6} {'存活率':>8} "
              f"{'同箱负例池':>11} {'正例中位面积':>13} {'负例中位面积':>13}")
        print("-" * 86)
        surv_all, elim_all = [], []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            P = [e for e in pos if e.get("d_long_at_emergence") is not None
                 and lo <= e["d_long_at_emergence"] < hi]
            if not P:
                continue
            S = [e for e in P if e["event_id"] in keep]
            N = [e for e in pool if e.get("d_long_at_emergence") is not None
                 and lo <= e["d_long_at_emergence"] < hi]
            ap_ = np.median([e["area_px"] for e in P if e.get("area_px")]) if P else np.nan
            an_ = np.median([e["area_px"] for e in N if e.get("area_px")]) if N else np.nan
            surv_all += S
            elim_all += [e for e in P if e["event_id"] not in keep]
            print(f"{f'{lo:.0f}–{hi:.0f}':>14} {len(P):>7} {len(S):>6} {len(S)/len(P):>8.2f} "
                  f"{len(N):>11} {ap_:>13.0f} {an_ if np.isfinite(an_) else float('nan'):>13.0f}")

        if not elim_all or not surv_all:
            continue
        print(f"\n  存活 {len(surv_all)} / 淘汰 {len(elim_all)}")
        print(f"  {'量':>12} {'存活中位':>10} {'淘汰中位':>10} {'p (MWU)':>10}")
        for k, lab in (("d_long_at_emergence", "d_long[m]"), ("area_px", "面积[px]"),
                       ("ecc", "离心率"), ("min_ttc_1s", "min TTC[s]")):
            a = np.array([e[k] for e in surv_all if e.get(k) is not None], dtype=float)
            b = np.array([e[k] for e in elim_all if e.get(k) is not None], dtype=float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 5 or len(b) < 5:
                continue
            p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
            print(f"  {lab:>12} {np.median(a):>10.2f} {np.median(b):>10.2f} {p:>10.3g}")

    # 缓存覆盖对照
    cached = {p.stem for p in (work / "cache").glob("*.h5")}
    pc = [e for e in pos if e["event_id"] in cached]
    d_all = np.array([e["d_long_at_emergence"] for e in pos
                      if e.get("d_long_at_emergence") is not None])
    d_cac = np.array([e["d_long_at_emergence"] for e in pc
                      if e.get("d_long_at_emergence") is not None])
    print(f"\n{'='*86}\n[缓存覆盖] 正例 全量 {len(d_all)} -> 已缓存 {len(d_cac)}")
    for thr in (10, 15, 20, 30):
        print(f"  d_long<{thr:>3}m: 全量 {(d_all<thr).sum():>4}  已缓存 {(d_cac<thr).sum():>4}  "
              f"覆盖率 {(d_cac<thr).sum()/max(1,(d_all<thr).sum()):.2f}")


if __name__ == "__main__":
    main()
