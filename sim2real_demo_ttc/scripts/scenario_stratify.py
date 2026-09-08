"""场景分层对照：lead 池按"路口上下文 / 左转上下文"分层，看上下文是否改变响应。

背景（2026-09-04 QA）：intersection 池 97.5%、left_turn 池 100% 是 lead 池的子集。
场景闸门只筛**scene 上下文**，归因+走廊仍然每次都挑正前方的前车，
所以这两类不是独立场景，只能作为 lead 场景的**分层**来读。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
RD = "arc_full"
MODELS = ["simlingo", "dd", "ltf", "ddv2"]


def ev_keys(f):
    return {(c["scene"], c["frame_idx"]) for c in
            json.load(open(RES / f))["candidates"] if c["a_vru_max"] >= 0.4}


def boot_ratio(scenes, bm, bg, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    uq = np.unique(scenes); idx = {s: np.where(scenes == s)[0] for s in uq}
    out = []
    for _ in range(n):
        sel = np.concatenate([idx[s] for s in rng.choice(uq, len(uq), replace=True)])
        d = bg[sel].mean()
        if abs(d) > 1e-9:
            out.append(bm[sel].mean() / d)
    return tuple(np.percentile(out, [2.5, 97.5])) if out else (np.nan, np.nan)


def main():
    isec = ev_keys("brake_first_pool_isec_navsim_v2.json")
    lt = ev_keys("brake_first_pool_leftturn_navsim_v2.json")
    print(f"\nlead/NAVSIM 分层：路口上下文 {len(isec)} 事件，左转上下文 {len(lt)} 事件"
          f"（后者 100% 含在前者与 lead 池内）\n")
    hdr = f"{'候选':<10}{'分层':<16}{'n(ev/sc)':>10}{'b_GT':>9}{'b_model':>9}{'响应比':>9}{'  95%CI':<22}{'方向一致':>10}"
    print(hdr); print("-" * 96)
    out = {}
    for m in MODELS:
        d = json.load(open(RES / f"gt_lead_navsim_{m}.json"))
        ev = d["per_event"]
        # lead 的 GT 文件里 frame_idx 未必存，退回用 scene 匹配路口分层
        isec_sc = {s for s, _ in isec}; lt_sc = {s for s, _ in lt}
        groups = {"全部 lead": lambda e: True,
                  "路口上下文": lambda e: e["scene"] in isec_sc,
                  "非路口": lambda e: e["scene"] not in isec_sc,
                  "左转上下文": lambda e: e["scene"] in lt_sc}
        for gname, sel in groups.items():
            sub = [e for e in ev if sel(e)]
            if len(sub) < 2:
                print(f"{m:<10}{gname:<16}{len(sub):>4}      —— 样本不足，不可估")
                continue
            bm = np.array([e[f"b_model__{RD}"] for e in sub])
            bg = np.array([e[f"b_gt__{RD}"] for e in sub])
            sc = np.array([e["scene"] for e in sub])
            r = bm.mean() / bg.mean(); ci = boot_ratio(sc, bm, bg)
            agree = int((np.sign(bm) == np.sign(bg)).sum())
            out[f"{m}|{gname}"] = dict(n=len(sub), n_scene=len(set(sc)),
                                       bg=float(bg.mean()), bm=float(bm.mean()),
                                       ratio=float(r), ci=list(ci), agree=agree)
            print(f"{m:<10}{gname:<16}{len(sub):>4}/{len(set(sc)):<5}{bg.mean():>+9.3f}"
                  f"{bm.mean():>+9.3f}{r:>+9.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"{'':<6}{agree}/{len(sub)}")
        print()
    (RES / "scenario_stratify.json").write_text(
        json.dumps({"readout": RD, "note": "intersection/left_turn 是 lead 的上下文分层，非独立场景",
                    "cells": out}, indent=2, ensure_ascii=False))
    print(f"[STRAT] wrote {RES/'scenario_stratify.json'}")


if __name__ == "__main__":
    main()
