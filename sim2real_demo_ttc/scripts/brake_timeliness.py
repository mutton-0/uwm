"""刹车及时性 + 危险场景避障成功率 —— 在**挖出的危险事件**上，不是随机场景。

## 及时性
brake-first 挖矿把查询帧定在**人类减速起点**，所以 t=0 就是人类开始刹车的时刻。
对规划轨迹算瞬时速度剖面

    v_k = |w_k - w_{k-1}| / dt   (k>=1),   v_0 = |w_0| / dt

取**首个** v_k <= v0 - DELTA 的时刻 t = (k+1)*dt 作为该计划的减速起始时刻。
GT 侧用自车真实未来路径按同一定义、同一 dt、同一 DELTA 算。

    timeliness = t_onset(model) - t_onset(GT)      正 = 比人类晚

**删失必须显式报告**：很多候选在时域内根本不减速到阈值，这类事件没有 onset，
不能当成"很晚"塞进均值，也不能悄悄丢掉 —— 分开报"曾减速比例"与"减速者的延迟"。

## 避障成功率
用官方 PDM 的 `no_at_fault_collisions` / `time_to_collision_within_bound`，
但打在**危险事件帧**上（origin 臂的规划轨迹），而不是随机抽的 200 个 scene。
这才是"避障"成功率；随机场景里绝大多数根本没有需要避的东西。
"""
from __future__ import annotations
import argparse, json, pickle, sys, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
WP_DT = {"simlingo": 0.25, "dd": 0.5, "ltf": 0.5, "ddv2": 0.5}
DELTA = 1.0            # 判定"开始刹车"的速度降幅阈值 m/s
# **统一读数栅格**。各候选原生 dt 与时域都不同（SimLingo 10x0.25s=2.5s，
# DD 家族 8x0.5s=4.0s），直接按原生栅格算 onset 不可跨模型比：
# 同一条人类 GT 轨迹在 0.25s 栅格上 onset=0.38s、在 0.5s 栅格上 onset=1.61s。
# 故一律把路径插值到共同栅格、共同时域后再读，模型侧与 GT 侧用同一套。
GRID_DT = 0.25
GRID_T = 2.5           # = min(2.5, 4.0)，两族共有的时域
MODELS = ["simlingo", "dd", "ltf", "ddv2"]


def resample(w, dt, grid_dt=GRID_DT, T=GRID_T):
    """把 ego 系路径插值到共同栅格 [0, T]，返回含原点的位置序列。

    只截断不外推：所有候选的原生时域都 >= T，故不需要外推。
    """
    w = np.asarray(w, float)[:, :2]
    t = np.arange(1, len(w) + 1) * dt
    tq = np.arange(0.0, T + 1e-9, grid_dt)
    P = np.column_stack([np.interp(tq, np.concatenate([[0.0], t]),
                                   np.concatenate([[0.0], w[:, d]])) for d in (0, 1)])
    return P


def speed_profile(P, dt):
    """位置序列（含原点）-> 瞬时速度剖面。"""
    P = np.asarray(P, float)
    return np.linalg.norm(np.diff(P, axis=0), axis=1) / dt


def onset(v, v0, dt, delta=DELTA):
    """首个降到 v0-delta 以下的时刻；时域内没有则返回 None（删失）。"""
    idx = np.nonzero(v <= v0 - delta)[0]
    return float((idx[0] + 1) * dt) if len(idx) else None


def boot_mean(a, n=5000, seed=0):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) < 2:
        return {"mean": float(a.mean()) if len(a) else float("nan"), "ci": [np.nan, np.nan],
                "n": int(len(a))}
    rng = np.random.default_rng(seed)
    bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)])
    return {"mean": float(a.mean()), "ci": [float(np.percentile(bs, 2.5)),
                                            float(np.percentile(bs, 97.5))], "n": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["ghost", "lead"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--f3-prefix", required=True, help="如 f3_leadfx_navsim")
    ap.add_argument("--skip-pdm", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import ns1_navsim_geometry as NS
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"), resolve=True)
    # 按 (split, scene) 建索引：scene_name 跨 split 不唯一，合并池必须逐事件取 split
    cache = defaultdict(list)
    for sp in ("test", "trainval"):
        d0 = NS.NS_ROOT / "navsim_logs" / sp
        if not d0.exists():
            continue
        for lf in sorted(d0.glob("*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                cache[(sp, f["scene_name"])].append(f)
    for k in cache:
        cache[k].sort(key=lambda z: z["timestamp"])

    P0 = json.load(open(RES / args.pool))["candidates"]
    pool = {(c["scene"], c["frame_idx"]): c for c in P0}
    SPLIT = {c["scene"]: c.get("split", "test") for c in P0}
    geo_cache = {}

    def geo_of(n):
        if n not in geo_cache:
            sp = SPLIT.get(n, "test")
            geo_cache[n] = NS.build_geo(cache[(sp, n)], cfg, sp)
        return geo_cache[n]

    # ---------- GT 侧的减速起始时刻（与模型同定义，逐 dt 各算一次） ----------
    gt_onset = {}
    for (sn, j), c in pool.items():
        g = geo_of(sn); gt = g["grid_t"]; xy = g["ego_xyz"][:, :2]
        v0 = float(g["ego_speed"][j])
        # GT 也走同一栅格：先按 grid_dt 取真实位姿，再同法求 onset
        idx = [int(np.argmin(np.abs(gt - (gt[j] + k * GRID_DT))))
               for k in range(int(round(GRID_T / GRID_DT)) + 1)]
        P = xy[idx] - xy[j]
        gt_onset[(sn, j)] = (onset(speed_profile(P, GRID_DT), v0, GRID_DT), v0, GRID_T)

    out = {}
    for m in MODELS:
        p = RES / f"{args.f3_prefix}_{m}_traj.json"
        if not p.exists():
            print(f"  跳过 {m}：缺 {p.name}"); continue
        dt = WP_DT[m]
        rows = []
        for e in json.load(open(p))["per_event"]:
            key = (e["scene"], e["frame_idx"])
            if key not in pool:
                continue
            v0 = float(e["ego_v0"])
            v = speed_profile(resample(e["traj_origin"], dt), GRID_DT)
            t_m = onset(v, v0, GRID_DT)
            t_g, _v0g, T = gt_onset[(e["scene"], e["frame_idx"])]
            rows.append({"scene": e["scene"], "frame_idx": e["frame_idx"], "v0": v0,
                         "horizon_s": T, "t_model": t_m, "t_gt": t_g,
                         "delay_s": (None if (t_m is None or t_g is None) else t_m - t_g),
                         "v_min_planned": float(v.min()),
                         "max_drop": float(v0 - v.min())})
        n = len(rows)
        braked = [r for r in rows if r["t_model"] is not None]
        gt_braked = [r for r in rows if r["t_gt"] is not None]
        paired = [r["delay_s"] for r in rows if r["delay_s"] is not None]
        out[m] = {
            "n_events": n, "horizon_s": rows[0]["horizon_s"] if rows else None,
            "delta_mps": DELTA,
            "model_braked": {"n": len(braked), "frac": len(braked) / n if n else None},
            "gt_braked": {"n": len(gt_braked), "frac": len(gt_braked) / n if n else None},
            "onset_model_s": boot_mean([r["t_model"] for r in braked]),
            "onset_gt_s": boot_mean([r["t_gt"] for r in gt_braked]),
            "delay_s_paired": boot_mean(paired),
            "max_drop_mps": boot_mean([r["max_drop"] for r in rows]),
            "per_event": rows,
        }
        b = out[m]
        print(f"\n[{m}] n={n}  统一栅格 {GRID_DT}s / 时域 {GRID_T}s  阈值 Δ={DELTA} m/s"
              f"  （原生 dt={dt}）")
        print(f"    计划中曾减速 {b['model_braked']['n']}/{n} = {b['model_braked']['frac']:.1%}"
              f"   （人类 GT {b['gt_braked']['n']}/{n} = {b['gt_braked']['frac']:.1%}）")
        print(f"    减速起始 模型 {b['onset_model_s']['mean']:.2f}s "
              f"[{b['onset_model_s']['ci'][0]:.2f},{b['onset_model_s']['ci'][1]:.2f}] (n={b['onset_model_s']['n']})"
              f"   GT {b['onset_gt_s']['mean']:.2f}s (n={b['onset_gt_s']['n']})")
        print(f"    配对延迟 {b['delay_s_paired']['mean']:+.2f}s "
              f"[{b['delay_s_paired']['ci'][0]:+.2f},{b['delay_s_paired']['ci'][1]:+.2f}]"
              f" (n={b['delay_s_paired']['n']}，正=比人类晚)")
        print(f"    计划内最大降速 {b['max_drop_mps']['mean']:.2f} m/s")

    op = Path(args.out or RES / f"timeliness_{args.scenario}_navsim.json")
    op.write_text(json.dumps({"scenario": args.scenario, "delta_mps": DELTA,
                              "pool": args.pool, "models": out}, indent=2, ensure_ascii=False))
    print(f"\n[TIMELINESS] wrote {op}")


if __name__ == "__main__":
    main()
