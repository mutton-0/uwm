"""把 G1 语料喂给 DiffusionDriveV2，缓存 8 层可读表征（含 nuScenes LIDAR_TOP → TransFuser BEV 直方图）。

与其余 navsim 系候选唯一的差异是**多了一路真实 lidar 输入**（V2 的发布权重要求，见 ddv2_adapter 头注）。
图像/状态/token/池化/行为量口径与 DiffusionDrive、LTF 逐字段一致。
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import cv2

R = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
sys.path.insert(0, str(R / "ddv2_g1_adapter"))
from ddv2_adapter import DDV2Runner, bbox_to_tokens        # noqa: E402


class NuScenesLidar:
    """按 CAM_FRONT 的 sample_data token 取同 sample 的 LIDAR_TOP 点云，变换到 ego 系。"""

    def __init__(self, root, version="v1.0-trainval"):
        from nuscenes.nuscenes import NuScenes
        from pyquaternion import Quaternion
        self.Q = Quaternion
        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)
        self.root = Path(root)

    def ego_points(self, cam_sd_token):
        sd = self.nusc.get("sample_data", cam_sd_token)
        samp = self.nusc.get("sample", sd["sample_token"])
        lsd = self.nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
        pts = np.fromfile(self.root / lsd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
        cs = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        Rm = self.Q(cs["rotation"]).rotation_matrix
        return (pts @ Rm.T) + np.array(cs["translation"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--types", nargs="+", default=["A", "D2a", "D2b", "D2c"])
    ap.add_argument("--events", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus", default="nuscenes", choices=["nuscenes", "navsim"],
                    help="点云来源：nuScenes 走 devkit，navsim 直接读 MergedPointCloud/*.pcd")
    ap.add_argument("--crop-center-row", type=int, default=0,
                    help="4:1 裁剪的竖直中心 = 相机主点行；0 = 沿用默认(nuScenes 450)。NAVSIM 用 560（§NS/A46）")
    args = ap.parse_args()
    if getattr(args, "crop_center_row", 0):
        import dd_adapter as _DD; _DD.set_crop_center_row(args.crop_center_row)
        print(f"[crop] CROP_CENTER_ROW -> {args.crop_center_row}")

    work = Path(args.work); out_dir = work / "ddv2_cache"; out_dir.mkdir(exist_ok=True)
    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D", "D2a", "D2b", "D2c", "D2bV", "D2cV")
               if (work / "mining" / f"matched_{t}.txt").exists()}
    if args.events:
        want = set(Path(args.events).read_text().split())
        keep = [e for e in events if e["event_id"] in want]
    else:
        keep = [e for e in events if e["event_type"] in args.types
                and not (e["event_type"] in matched and e["event_id"] not in matched[e["event_type"]])]
    if args.limit:
        keep = keep[: args.limit]
    print(f"[DDV2-G1] {len(keep)} 事件待缓存 -> {out_dir}", flush=True)

    if args.corpus == "navsim":
        # NAVSIM 语料：点云键用 CAM_F0 的 data_path（filename 去掉 split 前缀），且天然在 ego 系
        from ddv2_adapter import NavsimLidar
        lidar = NavsimLidar()
    else:
        lidar = NuScenesLidar(args.nuscenes_root)
    runner = DDV2Runner(device=args.device)
    t0, done, fail = time.time(), 0, []
    for i, ev in enumerate(keep):
        p = out_dir / f"{ev['event_id']}.npz"
        if p.exists():
            continue
        try:
            reg = None
            for f_ in ev["x_ghost_frames"]:
                if f_.get("bbox_xyxy"):
                    reg = bbox_to_tokens(f_["bbox_xyxy"], f_["im_wh"]); break
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            store, extra = {}, {}
            for cond in ("clean", "ghost"):
                per_pool, spds, trajs = {}, [], []
                for f_ in ev[f"x_{cond}_frames"]:
                    img = cv2.cvtColor(cv2.imread(str(Path(args.nuscenes_root) / f_["filename"])),
                                       cv2.COLOR_BGR2RGB)
                    key = (f_["filename"].split("/", 1)[1] if args.corpus == "navsim"
                           else f_["sd_token"])
                    pts = lidar.ego_points(key)
                    r = runner.run(img, anchor, reg, lidar_xyz=pts)
                    spds.append(r["commanded_speed"]); trajs.append(r["trajectory"])
                    for pool, v in r["pooled"].items():
                        per_pool.setdefault(pool, []).append(v)
                for pool, lst in per_pool.items():
                    for l in range(len(lst[0])):
                        store[f"{cond}/{pool}/L{l}"] = np.stack([x[l] for x in lst]).astype(np.float32)
                extra[f"commanded_speed_{cond}"] = np.array(spds, np.float32)
                extra[f"traj_{cond}"] = np.stack(trajs).astype(np.float32)
            meta = {k: v for k, v in ev.items() if k != "ttc_curve"}
            meta["_n_region_tokens"] = len(reg or []); meta["_prompt_anchor_speed"] = anchor
            np.savez_compressed(p, meta=json.dumps(meta, ensure_ascii=False), **store, **extra)
            done += 1
        except Exception as exc:                                        # noqa: BLE001
            fail.append({"event_id": ev["event_id"], "error": f"{type(exc).__name__}: {exc}"})
            if len(fail) <= 5:
                print(f"[DDV2-G1] FAIL {ev['event_id']}: {exc}", flush=True)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[DDV2-G1] {i+1}/{len(keep)} done={done} fail={len(fail)} "
                  f"{el:.0f}s ({el/max(1,done):.2f}s/event)", flush=True)
    (work / "results" / "ddv2_g1_cache_report.json").write_text(json.dumps(
        {"n_events": len(keep), "cached": done, "failed": fail[:20], "n_failed": len(fail),
         "elapsed_s": time.time() - t0,
         "lidar_note": "nuScenes LIDAR_TOP（32 线）→ ego 系 → TransFuser BEV 直方图；"
                       "与 NAVSIM 多雷达合并点云的密度不同，属分布外输入，随结果声明"},
        indent=2, ensure_ascii=False))
    print(f"[DDV2-G1] 完成 {done}/{len(keep)}，失败 {len(fail)}")


if __name__ == "__main__":
    main()
