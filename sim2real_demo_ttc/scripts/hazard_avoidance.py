"""危险场景避障成功率：官方 PDM 指标打在**挖出的危险事件帧**上。

与 `deploy_rank.py` 的区别只有一处 —— 刺激集：
  deploy_rank      随机 200 个 NAVSIM scene（绝大多数根本没有需要避的东西）
  hazard_avoidance 挖出的 ghost/lead 事件帧（人类在此刻确实开始刹车、且已归因到具体危险）
指标定义、打分代码、四家联合 proposal 归一化全部相同。

轨迹用 F-3 的 **origin 臂**（未遮挡，即真实输入下的规划），这正是"它在这个危险面前会怎么开"。
"""
from __future__ import annotations
import argparse, json, pickle, sys, glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, "/data/ruolin/uwm")
CACHE = "/data/dataset/navsim/exp/metric_cache"
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
WP_DT = {"simlingo": 0.25, "dd": 0.5, "ltf": 0.5, "ddv2": 0.5,
         # Alpamayo 64 步 × 0.1s；AutoVLA 10 步 × 0.5s（与 c_axis_hazard_patch.py 一致）
         "alpa": 0.1, "autovla": 0.5}


def to_pdm_poses(wp, dt, horizon=4.0, step=0.5):
    w = np.asarray(wp, float)[:, :2]
    t = np.arange(1, len(w) + 1) * dt
    tq = np.arange(step, horizon + 1e-9, step)
    P = np.column_stack([np.interp(tq, t, w[:, d]) for d in (0, 1)])
    if tq[-1] > t[-1]:
        ext = tq > t[-1]
        v = (w[-1] - w[-2]) / dt if len(w) >= 2 else w[-1] / dt
        P[ext] = w[-1] + np.outer(tq[ext] - t[-1], v)
    d = P - np.vstack([[0.0, 0.0], P[:-1]])
    h = np.arctan2(d[:, 1], d[:, 0]); h[np.linalg.norm(d, axis=1) < 1e-3] = 0.0
    return np.column_stack([P, h])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["ghost", "lead"])
    ap.add_argument("--f3-prefix", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer, PDMScorerConfig)
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    # 按 (split, scene) 建索引：scene_name 跨 split 不唯一
    sc = defaultdict(list)
    for sp in ("test", "trainval"):
        for lf in sorted(glob.glob(f"/data/dataset/navsim/dataset/navsim_logs/{sp}/*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                sc[(sp, f["scene_name"])].append(f)
    for k in sc:
        sc[k].sort(key=lambda z: z["timestamp"])

    per = {}
    for m in MODELS:
        p = RES / f"{args.f3_prefix}_{m}_traj.json"
        if p.exists():
            per[m] = {(e.get("split", "test"), e["scene"], e["frame_idx"]): e
                      for e in json.load(open(p))["per_event"]}
    models = [m for m in MODELS if m in per]
    keys = sorted(set.intersection(*[set(per[m]) for m in models]))

    loader = MetricCacheLoader(Path(CACHE)); cached = set(loader.tokens)
    usable = []
    for (sp, s, j) in keys:
        fl = sc.get((sp, s), [])
        if j < len(fl) and fl[j]["token"] in cached:
            usable.append((sp, s, j, fl[j]["token"]))
    print(f"[HAZ/{args.scenario}] 四家共有事件 {len(keys)}，其中有 metric cache 的 {len(usable)}")

    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(ps); scorer = PDMScorer(ps, PDMScorerConfig())
    fs = TrajectorySampling(num_poses=8, interval_length=0.5)

    out = defaultdict(list); fails = defaultdict(int)
    for (sp, s, j, tok) in usable:
        try:
            mc = loader.get_from_token(tok)
            arr = []
            for m in models:
                poses = to_pdm_poses(per[m][(sp, s, j)]["traj_origin"], WP_DT[m])
                it = transform_trajectory(Trajectory(poses, fs), mc.ego_state)
                arr.append(get_trajectory_as_array(it, ps, mc.ego_state.time_point))
            res = scorer.score_proposals(sim.simulate_proposals(np.stack(arr, 0), mc.ego_state),
                                         mc.observation, mc.centerline,
                                         mc.route_lane_ids, mc.drivable_area_map)
            for k, m in enumerate(models):
                row = res[k].iloc[0].to_dict()
                out[m].append({"scene": s, "frame_idx": j,
                               **{kk: (float(v) if np.size(v) == 1 else float(np.mean(v)))
                                  for kk, v in row.items()}})
        except Exception as e:                                          # noqa: BLE001
            fails[f"{type(e).__name__}: {str(e)[:60]}"] += 1

    if not out:
        print("无成功样本", dict(fails)); return
    show = ["no_at_fault_collisions", "time_to_collision_within_bound",
            "drivable_area_compliance", "ego_progress", "pdm_score"]
    rng = np.random.default_rng(0); summ = {}
    n = len(out[models[0]])
    print(f"\n危险事件上的 PDM（n={n}，失败 {sum(fails.values())}）\n")
    print(f"{'指标':<34}" + "".join(f"{m:>21}" for m in models))
    print("-" * (34 + 21 * len(models)))
    for k in show:
        line = f"{k:<34}"
        for m in models:
            a = np.array([r[k] for r in out[m]], float)
            bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(5000)])
            ci = np.percentile(bs, [2.5, 97.5])
            summ.setdefault(m, {})[k] = {"mean": float(a.mean()), "ci": list(ci)}
            line += f"{a.mean():>9.3f}[{ci[0]:.2f},{ci[1]:.2f}]"
        print(line)
    op = Path(args.out or RES / f"hazard_avoidance_{args.scenario}.json")
    op.write_text(json.dumps({"scenario": args.scenario, "n_events": n, "models": models,
                              "protocol": "官方 PDMScorer；四家同批 proposal；轨迹取 F-3 origin 臂",
                              "summary": summ, "failures": dict(fails),
                              "per_model": {m: out[m] for m in models}},
                             indent=2, ensure_ascii=False))
    print(f"\n[HAZ] wrote {op}")


if __name__ == "__main__":
    main()
