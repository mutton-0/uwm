"""部署质量排名：四候选在**同一批 NAVSIM 刺激**上跑官方 PDM + 舒适度指标。

## 三段式（必须拆开的原因）
候选之间环境不兼容：DDv2 自带一份 navsim 且必须排在 sys.path 最前，
SimLingo 的环境里没有 nuplan。所以：

  plan   刺激集只算一次并落盘 -> 四候选共用同一批 scene/token（消除刺激差异）
  infer  在**候选自己的环境**里跑推理，只落盘轨迹（不 import nuplan/主仓库 navsim）
  score  在装了 nuplan 的环境里用**官方 PDMScorer** 统一打分（打分代码对四家逐字相同）

## 协议
- 刺激：NAVSIM test 随机抽 N 个 scene，每 scene 取 frame 3（官方查询帧）。
- 输入：前视 RGB + 自车速度；DDv2 另吃点云。与 F-3 的 origin 臂同源。
- 打分：官方 `PDMSimulator` + `PDMScorer`，不搬运各候选自己发布的分数。

## 两条必须声明的口径限制
1. **非反应式 PDM（v1 口径）**：本机 metric_cache 是 v1 结构（有 `_occupancy_maps`、
   无 `_detections_tracks`），v2 的 `LogReplayTrafficAgents` 会 AttributeError。
   故走 `scorer.score_proposals(...)` 直接对缓存 observation 打分。四候选口径一致、
   可互比，但**不能**与他人发布的 v2 反应式分数直接比。
2. **SimLingo 时域比 PDM 短**：它输出 10x0.25s = 2.5s，PDM 要 4s。这里重采样到 0.5s
   网格后按末段速度匀速外推到 4s —— 协议适配，不是它自己的输出。
   DD 家族 8x0.5s = 4s 原生匹配，不外推。
"""
from __future__ import annotations
import argparse, json, pickle, sys, glob
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
UWM = "/data/ruolin/uwm"          # 主仓库 navsim：**只在 score 段插入**
NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"
NS_LOGS = "/data/dataset/navsim/dataset/navsim_logs/test"
CACHE = "/data/dataset/navsim/exp/metric_cache"
QUERY_FRAME = 3
WP_DT = {"simlingo": 0.25, "dd": 0.5, "ltf": 0.5, "ddv2": 0.5}
STIM = RES / "deploy_stimuli.json"


def load_scenes():
    sc = defaultdict(list)
    for lf in sorted(glob.glob(f"{NS_LOGS}/*.pkl")):
        for f in pickle.load(open(lf, "rb")):
            sc[f["scene_name"]].append(f)
    for k in sc:
        sc[k].sort(key=lambda z: z["timestamp"])
    return sc


def to_pdm_poses(wp, dt, horizon=4.0, step=0.5):
    """ego 系 (x,y) 航点 -> PDM 需要的 (x, y, heading) @ 0.5s x 8。

    朝向由路径切线给出（候选都不输出朝向）。时域不足 4s 的按末段速度匀速外推。
    """
    w = np.asarray(wp, float)[:, :2]
    t = np.arange(1, len(w) + 1) * dt
    tq = np.arange(step, horizon + 1e-9, step)
    P = np.column_stack([np.interp(tq, t, w[:, d]) for d in (0, 1)])
    if tq[-1] > t[-1]:
        ext = tq > t[-1]
        v = (w[-1] - w[-2]) / dt if len(w) >= 2 else w[-1] / dt
        P[ext] = w[-1] + np.outer(tq[ext] - t[-1], v)
    d = P - np.vstack([[0.0, 0.0], P[:-1]])
    head = np.arctan2(d[:, 1], d[:, 0])
    head[np.linalg.norm(d, axis=1) < 1e-3] = 0.0
    return np.column_stack([P, head])


def comfort_metrics(poses, step=0.5):
    """我们自己的舒适度读数，补官方 history_comfort 之外的量纲信息。"""
    p = np.vstack([[0.0, 0.0], np.asarray(poses, float)[:, :2]])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / step
    a = np.diff(v) / step
    j = np.diff(a) / step if len(a) > 1 else np.array([0.0])
    return {"jerk_rms": float(np.sqrt(np.mean(j ** 2))),
            "max_decel": float(max(-a.min(), 0.0)) if len(a) else 0.0,
            "max_accel": float(max(a.max(), 0.0)) if len(a) else 0.0}


# ------------------------------------------------------------------ plan
def do_plan(args):
    sys.path.insert(0, UWM)
    from navsim.common.dataloader import MetricCacheLoader
    cached = set(MetricCacheLoader(Path(CACHE)).tokens)
    sc = load_scenes()
    CITIES = {x.strip() for x in args.cities.split(",") if x.strip()}
    usable = sorted(s for s, v in sc.items()
                    if len(v) > QUERY_FRAME and v[QUERY_FRAME]["token"] in cached
                    and (not CITIES or v[QUERY_FRAME]["map_location"] in CITIES))
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(usable, min(args.n_scenes, len(usable)), replace=False))
    stim = [{"scene": s, "frame_idx": QUERY_FRAME,
             "token": sc[s][QUERY_FRAME]["token"]} for s in pick]
    STIM.write_text(json.dumps({"seed": args.seed, "query_frame": QUERY_FRAME,
                                "cities": sorted(CITIES) or None,
                                "n_usable": len(usable), "n": len(stim),
                                "stimuli": stim}, indent=2))
    print(f"[PLAN] 可用 scene {len(usable)}，抽 {len(stim)} -> {STIM}")


# ------------------------------------------------------------------ infer
def do_infer(args, traj_p):
    stim = json.load(open(STIM))["stimuli"]
    sc = load_scenes()
    import cv2
    lidar = None
    if args.model == "simlingo":
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        runner = SimLingoRunner(cfg, capture_hidden=False)

        def infer(img, spd, pts=None):
            return np.asarray(runner.infer(img, spd, pool_modes=()).waypoints, float)
    else:
        for a in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
            sys.path.insert(0, str(RES / a))
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        if args.model == "ddv2":
            from ddv2_adapter import NavsimLidar
            lidar = NavsimLidar()

        def infer(img, spd, pts=None):
            kw = {} if lidar is None else {"lidar_xyz": pts}
            return np.asarray(runner.run(img, spd, **kw)["trajectory"], float)

    recs, fails = [], defaultdict(int)
    for i, st in enumerate(stim):
        fr = sc[st["scene"]][st["frame_idx"]]
        try:
            img = cv2.cvtColor(cv2.imread(
                str(Path(NS_BLOBS) / "test" / fr["cams"]["CAM_F0"]["data_path"])),
                cv2.COLOR_BGR2RGB)
            spd = float(np.linalg.norm(np.asarray(fr["ego_dynamic_state"][:2], float)))
            pts = lidar.ego_points(fr["lidar_path"]) if lidar is not None else None
            wp = infer(img, spd, pts)
            recs.append({"scene": st["scene"], "token": st["token"],
                         "ego_speed": round(spd, 4),
                         "poses": to_pdm_poses(wp, WP_DT[args.model]).tolist()})
        except Exception as e:                                          # noqa: BLE001
            fails[f"{type(e).__name__}: {str(e)[:70]}"] += 1
        if (i + 1) % 50 == 0:
            print(f"  [infer] {i+1}/{len(stim)}  成功 {len(recs)}", flush=True)
    traj_p.write_text(json.dumps(
        {"model": args.model, "n": len(recs), "wp_dt": WP_DT[args.model],
         "horizon_adaptation": ("resample + constant-velocity extrapolation to 4s"
                                if args.model == "simlingo" else "native 4s"),
         "failures": dict(fails), "records": recs}, indent=2))
    print(f"[INFER/{args.model}] n={len(recs)} 失败 {sum(fails.values())} -> {traj_p}")
    for k, v in list(fails.items())[:5]:
        print(f"    {v:4d}  {k}")


# ------------------------------------------------------------------ score
def do_score(args, traj_p):
    sys.path.insert(0, UWM)
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer, PDMScorerConfig)
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    d = json.load(open(traj_p))
    loader = MetricCacheLoader(Path(CACHE))
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(ps); scorer = PDMScorer(ps, PDMScorerConfig())
    fs = TrajectorySampling(num_poses=8, interval_length=0.5)

    recs, fails = [], defaultdict(int)
    for i, r in enumerate(d["records"]):
        try:
            poses = np.asarray(r["poses"], float)
            mc = loader.get_from_token(r["token"])
            it = transform_trajectory(Trajectory(poses, fs), mc.ego_state)
            states = get_trajectory_as_array(it, ps, mc.ego_state.time_point)[None, ...]
            out = scorer.score_proposals(sim.simulate_proposals(states, mc.ego_state),
                                         mc.observation, mc.centerline,
                                         mc.route_lane_ids, mc.drivable_area_map)
            row = out[0].iloc[0].to_dict()
            recs.append({"scene": r["scene"], "token": r["token"],
                         "ego_speed": r["ego_speed"],
                         **{k: (float(v) if np.size(v) == 1 else float(np.mean(v)))
                            for k, v in row.items()},
                         **comfort_metrics(poses)})
        except Exception as e:                                          # noqa: BLE001
            fails[f"{type(e).__name__}: {str(e)[:70]}"] += 1
        if (i + 1) % 50 == 0:
            print(f"  [score] {i+1}/{len(d['records'])}", flush=True)
    if not recs:
        print("[SCORE] 无成功样本", dict(fails)); return

    keys = [k for k in recs[0] if k not in ("scene", "token")]
    rng = np.random.default_rng(0); summ = {}
    print(f"\n[SCORE/{d['model']}] n={len(recs)}  失败 {sum(fails.values())}")
    print(f"{'指标':<34}{'均值':>10}   scene bootstrap 95%CI")
    for k in keys:
        a = np.array([r[k] for r in recs], float)
        bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(5000)])
        ci = np.percentile(bs, [2.5, 97.5])
        summ[k] = {"mean": float(a.mean()), "ci": list(ci)}
        print(f"{k:<34}{a.mean():>10.4f}   [{ci[0]:.4f},{ci[1]:.4f}]")
    out_p = Path(args.out or RES / f"deploy_rank_{d['model']}.json")
    out_p.write_text(json.dumps(
        {"model": d["model"], "n": len(recs), "seed": json.load(open(STIM))["seed"],
         "protocol": {"query_frame": QUERY_FRAME, "pdm": "non-reactive (v1 cache)",
                      "horizon_adaptation": d["horizon_adaptation"],
                      "scorer": "official PDMScorer.score_proposals",
                      "stimuli": str(STIM)},
         "summary": summ, "failures": dict(fails), "per_scene": recs},
        indent=2, ensure_ascii=False))
    print(f"\n[SCORE] wrote {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["plan", "infer", "score"])
    ap.add_argument("--model", choices=["simlingo", "dd", "ltf", "ddv2"])
    ap.add_argument("--n-scenes", type=int, default=200)
    ap.add_argument("--cities", default="",
                    help="逗号分隔 map_location 白名单。与 F 轴同域：deployment 限定为"
                         "维加斯+匹兹堡，才能和 nuScenes(波士顿+新加坡) 构成零重叠的域对。")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sl-config", default=str(ROOT / "configs" / "n1_d2.yaml"))
    ap.add_argument("--traj", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    traj_p = Path(args.traj or RES / f"deploy_traj_{args.model}.json")
    {"plan": lambda: do_plan(args),
     "infer": lambda: do_infer(args, traj_p),
     "score": lambda: do_score(args, traj_p)}[args.stage]()


if __name__ == "__main__":
    main()
