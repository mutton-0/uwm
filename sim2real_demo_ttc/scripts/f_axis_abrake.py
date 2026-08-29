"""T-F 前置|从 nuScenes ego_pose 重算每个事件的**真实人类刹车强度** a_brake。

工单依据:four_axis_proof_experiment_workorder.md §3 步骤 1-2 + 本轮预注册 FA.1 偏离 2。

为什么要重算:事件账本 `events_all.jsonl` 只存了 clean/ghost 各 2 帧的 `ego_speed_mps`,
不足以刻画"探头出现后人类踩了多重的刹车"。这里回到 ego_pose 原始位姿序列,
在 [t_emergence, t_emergence + horizon] 窗口内取纵向加速度的最小值(最强减速)。

**这是 Stage E 红线 2 要求的强 ground truth** —— 人类驾驶员的真实行为,
不是模型自身输出的代理量(b = v_plan 差)。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf


def speed_series(nusc, scene, camera="CAM_FRONT"):
    """该 scene 的 (t, speed) 序列 —— 用 CAM_FRONT 全部 sample_data(含 sweeps)的 ego_pose。"""
    first = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", first["data"][camera])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    ts, xy = [], []
    while True:
        p = nusc.get("ego_pose", sd["ego_pose_token"])
        ts.append(sd["timestamp"] * 1e-6); xy.append(p["translation"][:2])
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])
    t = np.array(ts); x = np.array(xy)
    o = np.argsort(t); t, x = t[o], x[o]
    # 中心差分求速率，再中心差分求纵向加速度；两步各做一次 3 点滑动平均去量化噪声
    def cdiff(y, tt):
        d = np.zeros_like(y, dtype=float)
        d[1:-1] = (y[2:] - y[:-2]) / (tt[2:] - tt[:-2])[..., None] if y.ndim > 1 else \
                  (y[2:] - y[:-2]) / (tt[2:] - tt[:-2])
        d[0] = d[1]; d[-1] = d[-2]
        return d
    def smooth(y, k=3):
        return np.convolve(y, np.ones(k) / k, mode="same")
    v = np.linalg.norm(cdiff(x, t), axis=1)
    v = smooth(v)
    a = smooth(cdiff(v, t))
    return t, v, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/results/f_axis_abrake.json")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    by_scene = defaultdict(list)
    for e in events:
        by_scene[e["scene_token"]].append(e)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"],
                    dataroot=cfg["paths"]["nuscenes_root"], verbose=False)
    scenes = {s["token"]: s for s in nusc.scene}

    out = {}
    for i, (stok, evs) in enumerate(sorted(by_scene.items())):
        if stok not in scenes:
            continue
        t, v, a = speed_series(nusc, scenes[stok])
        for e in evs:
            t0 = e["t_emergence"]; m = (t >= t0) & (t <= t0 + args.horizon)
            if m.sum() < 3:
                continue
            pre = (t >= t0 - 1.0) & (t < t0)
            out[e["event_id"]] = {
                "a_brake": float(a[m].min()),                 # 最强减速(负值)
                "a_mean": float(a[m].mean()),
                "v_at_emergence": float(np.interp(t0, t, v)),
                "dv_over_horizon": float(v[m][-1] - v[m][0]),
                "v_pre_mean": float(v[pre].mean()) if pre.sum() else float("nan"),
                "n_samples": int(m.sum()),
                "event_type": e["event_type"], "scene": e["scene_name"],
                "min_ttc_1s": e.get("min_ttc_1s"),
            }
        if (i + 1) % 50 == 0:
            print(f"[T-F/a_brake] {i+1}/{len(by_scene)} scenes, {len(out)} events")

    arr = np.array([v["a_brake"] for v in out.values()])
    print(f"[T-F/a_brake] {len(out)} 事件; a_brake 分布 "
          f"p5={np.percentile(arr,5):.2f} p50={np.percentile(arr,50):.2f} "
          f"p95={np.percentile(arr,95):.2f} m/s²")
    Path(args.out).write_text(json.dumps({"horizon_s": args.horizon, "n": len(out),
                                          "events": out}, indent=2, ensure_ascii=False))
    print(f"[T-F/a_brake] wrote {args.out}")


if __name__ == "__main__":
    main()
