"""LTF checkpoint 健康度基线探针|在与危险无关的普通帧上跑 LTF，看规划速度是否正常。

工单第 4 项：排除「这个 checkpoint 本身退化/卡死」这种比『盲动』更根本的可能。
取样刻意避开事件语料：从 nuScenes 里随机抽 scene，再在每个 scene 里随机取 CAM_FRONT
关键帧，且**排除**出现在 F-3 事件集里的 scene，保证是"普通行驶帧"。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2, numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(RES / "ltf_g1_adapter"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
    used = {r["scene"] for r in json.load(open(RES / "f3_occlusion_ltf.json"))["per_event"]}
    scenes = [s for s in nusc.scene if s["name"] not in used]
    rng = np.random.default_rng(7)
    picks = []
    for s in [scenes[i] for i in rng.choice(len(scenes), n, replace=False)]:
        sample = nusc.get("sample", s["first_sample_token"])
        for _ in range(int(rng.integers(3, 18))):          # 随机往后走几帧，避开起始帧
            if not sample["next"]:
                break
            sample = nusc.get("sample", sample["next"])
        sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        # ego 当前速度：用相邻 ego_pose 的位移差
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        nxt = nusc.get("sample_data", sd["next"]) if sd["next"] else None
        v0 = 0.0
        if nxt:
            ep2 = nusc.get("ego_pose", nxt["ego_pose_token"])
            dt = (ep2["timestamp"] - ep["timestamp"]) / 1e6
            if dt > 0:
                v0 = float(np.linalg.norm(np.array(ep2["translation"][:2])
                                          - np.array(ep["translation"][:2])) / dt)
        picks.append((s["name"], sd["filename"], v0))

    from ltf_adapter import LTFRunner
    runner = LTFRunner(device="cuda")
    out = []
    for name, fn, v0 in picks:
        img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / fn)), cv2.COLOR_BGR2RGB)
        r = runner.run(img, v0)
        tj = np.asarray(r["trajectory"], float)
        out.append({"scene": name, "file": fn, "ego_speed_in": v0,
                    "commanded_speed": float(r["commanded_speed"]),
                    "traj_arclen": float(np.linalg.norm(np.diff(tj[:, :2], axis=0), axis=1).sum()),
                    "traj_endpoint": [float(tj[-1, 0]), float(tj[-1, 1])]})
        print(f"  {name:14s} v_in {v0:5.2f} -> v_plan {out[-1]['commanded_speed']:6.3f}  "
              f"弧长 {out[-1]['traj_arclen']:6.2f}  终点 ({tj[-1,0]:+.1f},{tj[-1,1]:+.1f})")

    v = np.array([o["commanded_speed"] for o in out])
    vin = np.array([o["ego_speed_in"] for o in out])
    L = np.array([o["traj_arclen"] for o in out])
    lat = np.abs(np.array([o["traj_endpoint"][1] for o in out]))
    summ = {"n": len(out), "v_plan": {"mean": float(v.mean()), "std": float(v.std()),
                                      "min": float(v.min()), "max": float(v.max()),
                                      "n_unique": int(len(np.unique(np.round(v, 4))))},
            "corr_v_plan_vs_ego_speed_in": float(np.corrcoef(vin, v)[0, 1]),
            "traj_arclen": {"mean": float(L.mean()), "min": float(L.min()), "max": float(L.max())},
            "abs_endpoint_lateral": {"mean": float(lat.mean()), "max": float(lat.max()),
                                     "n_gt_1m": int((lat > 1).sum())},
            "degenerate_checks": {
                "all_same_speed": bool(len(np.unique(np.round(v, 3))) == 1),
                "all_zero_speed": bool(np.all(v < 0.05)),
                "all_straight": bool(np.all(lat < 0.2))}}
    print("\n" + json.dumps(summ, indent=2, ensure_ascii=False))
    (RES / "ltf_baseline_probe.json").write_text(
        json.dumps({"summary": summ, "per_frame": out}, indent=2, ensure_ascii=False))
    print(f"[LTF-PROBE] wrote {RES/'ltf_baseline_probe.json'}")


if __name__ == "__main__":
    main()
