"""NS1|NAVSIM / OpenScene 独立复现语料：数据源接入层 + 事件挖掘。

工单：docs/navsim_openscene_independent_corpus_workorder.md。

**这个脚本只做一件事：把 NAVSIM 日志变成 G1 挖掘管线认识的 `geo` 结构。**
事件判据（`detect_events`）、帧选取（`pick_frames`）、事件记录构造
（`frame_record` / `frame_bbox`）、阈值、统计纪律**全部 import 自 `g1_mine_events`，一行未改**。
这正是工单 §1「只换数据源接入层，不重新设计方法论」的执行形式：
若判据需要为新数据源改写，"同场景结构跨数据源"这个检验就已经被污染了。

--------------------------------------------------------------------------
数据源事实（可行性评估的一手记录，见 navsim_openscene_mining_report_*）

  路径      /data/dataset/navsim/dataset/{navsim_logs,sensor_blobs}/test
  规模      147 个 log，~68k 帧，~1.8k 个 scene（2 Hz，scene ≈ 18 s）
  采集      nuPlan 车队，拉斯维加斯 / 波士顿 / 匹兹堡 / 新加坡；与 nuScenes **完全独立采集**
  标注      每帧 anns：gt_boxes[N,7]、gt_names、gt_velocity_3d、track_tokens
  相机      CAM_F0 前视 1920×1080，含内参与 sensor2lidar 外参

与 nuScenes 的三处**对本管线有利**的差异（成本反转的来源，须在报告中记录）：
  1. `gt_velocity_3d` **直接给出目标绝对速度** —— nuScenes 侧要靠 2 Hz 标注差分，
     那正是 §LB/A44 里 32 个 |a| > 10 m/s² 伪影的来源；
  2. `gt_boxes` 的位置**已经在 lidar 系**，而 `lidar2ego` 是恒等变换
     （translation [0,0,0]、rotation [1,0,0,0]，已实测核对）⇒ 位置无需 world→ego 变换；
  3. 类别里天然带 `traffic_cone` / `barrier` / `czone_sign` / `generic_object`
     —— 这正是 D2a「按类别就无害的静物」所需的素材，比 nuScenes 的 movable_object 更丰富。

一处**不利**差异（须登记为偏离）：
  NAVSIM 相机有畸变系数（distortion[5]），而 G1 的投影是纯针孔。
  本适配层沿用针孔投影（与 G1 逐字段同构），故成像面积/离心率/bbox 有小量畸变误差。
  这对**组间**比较无偏（正负例走同一条投影），但绝对值不可与 nuScenes 侧直接比较。
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_mine_events as G1                                             # noqa: E402

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
NS_ROOT = Path("/data/dataset/navsim/dataset")

# NAVSIM 原生类别 → nuScenes 风格类名。
# **为什么要映射**：下游（D2cV 的 VRU 子集筛选、报告里的 object_class）按 nuScenes 前缀取用。
# 原生名同时落盘在 object_class_native，不丢溯源。
CLASS_MAP = {
    "pedestrian": "human.pedestrian.adult",
    "bicycle": "vehicle.bicycle",
    "vehicle": "vehicle.car",
    "traffic_cone": "movable_object.trafficcone",
    "barrier": "movable_object.barrier",
    "czone_sign": "movable_object.czone_sign",
    "generic_object": "static_object.generic",
}


def build_geo(frames_raw, cfg, split="test"):
    """把一个 NAVSIM scene 的原始帧列表，变成 g1_mine_events 认识的 geo 结构。

    字段与 `g1_mine_events.compute_scene_geometry` 的返回值**逐个对齐**，
    使 detect_events / pick_frames / frame_record 可以原样复用。
    """
    mcfg = cfg["mining"]
    n = len(frames_raw)
    grid_t = np.array([f["timestamp"] / 1e6 for f in frames_raw], float)
    ego_xyz = np.stack([np.asarray(f["ego2global_translation"], float) for f in frames_raw])
    R_we = np.stack([Quaternion(f["ego2global_rotation"]).rotation_matrix for f in frames_raw])
    # ego 速度：NAVSIM 直接给（已实测确认是 **ego 系**，与位移差分转 ego 系一致到 0.15 m/s 内）
    ego_v_ego = np.stack([np.asarray(f["ego_dynamic_state"][:2], float) for f in frames_raw])
    ego_speed = np.linalg.norm(ego_v_ego, axis=1)

    cam0 = frames_raw[0]["cams"]["CAM_F0"]
    K = np.asarray(cam0["cam_intrinsic"], float)
    # lidar2ego 是恒等 ⇒ sensor2lidar 即 sensor2ego，与 G1 的 (R_ec, t_ec) 同义
    R_ec = np.asarray(cam0["sensor2lidar_rotation"], float)
    t_ec = np.asarray(cam0["sensor2lidar_translation"], float)
    im_w, im_h = 1920, 1080

    frames = [{"t": grid_t[j], "token": f["token"],
               "filename": str(Path(split) / f["cams"]["CAM_F0"]["data_path"]),
               "is_key_frame": True,          # NAVSIM 每帧都是标注帧（2 Hz），无 sweep/sample 之分
               "ego_t": ego_xyz[j], "ego_q": Quaternion(f["ego2global_rotation"]),
               "width": im_w, "height": im_h, "calib_token": "CAM_F0"}
              for j, f in enumerate(frames_raw)]

    # ---- 按 track_token 拼轨迹 ----
    raw = defaultdict(lambda: {"p": {}, "v": {}, "size": {}, "yaw": {}, "name": None})
    for j, f in enumerate(frames_raw):
        a = f["anns"]
        b = np.asarray(a["gt_boxes"], float)
        if b.size == 0:
            continue
        nm = list(a["gt_names"]); tk = list(a["track_tokens"])
        vel = np.asarray(a["gt_velocity_3d"], float)
        for i in range(len(tk)):
            r = raw[tk[i]]
            r["name"] = nm[i]
            r["p"][j] = b[i, :3]                       # 已在 lidar(=ego) 系
            r["v"][j] = vel[i]                         # 全局系绝对速度（静物实测 |v|≈0.001）
            r["size"][j] = (float(b[i, 4]), float(b[i, 3]), float(b[i, 5]))   # (w, l, h)
            r["yaw"][j] = float(b[i, 6])

    per_obj = {}
    for tok, r in raw.items():
        native = r["name"]
        cat = CLASS_MAP.get(native)
        if cat is None:
            continue
        cls = G1.obj_class(cat)
        if cls == "other":
            continue
        idx = sorted(r["p"])
        if len(idx) < 3:
            continue
        valid = np.zeros(n, bool); valid[idx] = True
        p_ego = np.full((n, 3), np.nan); v_glob = np.full((n, 3), np.nan)
        for j in idx:
            p_ego[j] = r["p"][j]; v_glob[j] = r["v"][j]
        # 目标绝对速度 → ego 系（与 G1 的 v_obj_ego 同义）
        v_obj_ego = np.full((n, 3), np.nan)
        v_obj_ego[valid] = np.einsum("tij,tj->ti", np.transpose(R_we[valid], (0, 2, 1)), v_glob[valid])
        v_ego_ego = np.zeros((n, 3)); v_ego_ego[:, :2] = ego_v_ego

        x, y = p_ego[:, 0], p_ego[:, 1]
        in_corridor = (np.abs(y) < mcfg["corridor_half_width_m"]) & \
                      (x > mcfg["corridor_x_min_m"]) & (x < mcfg["corridor_x_max_m"]) & valid

        d_xy = np.linalg.norm(p_ego[:, :2], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            los = p_ego[:, :2] / d_xy[:, None]
            v_c = np.sum((v_ego_ego[:, :2] - v_obj_ego[:, :2]) * los, axis=1)
            ttc = np.where(v_c > mcfg["min_closing_speed_mps"], x / v_c, np.inf)
        ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, np.inf)
        ttc[~valid] = np.inf

        p_cam = np.full((n, 3), np.nan)
        p_cam[valid] = np.einsum("ij,tj->ti", R_ec.T, p_ego[valid] - t_ec)
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = (K @ p_cam.T).T
            u = uv[:, 0] / uv[:, 2]; v_img = uv[:, 1] / uv[:, 2]
        visible = valid & (p_cam[:, 2] > 0.1) & (u > 0) & (u < im_w) & (v_img > 0) & (v_img < im_h)

        sz = r["size"][idx[0]]
        depth = np.clip(p_cam[:, 2], 0.5, None)
        area_px = float(K[0, 0] * K[1, 1]) * (sz[0] * sz[2]) / depth ** 2
        area_px = np.where(visible, area_px, np.nan)
        ecc = np.hypot((u - im_w / 2) / (im_w / 2), (v_img - im_h / 2) / (im_h / 2))
        ecc = np.where(visible, ecc, np.nan)

        # NAVSIM 无 nuScenes 的 attribute 标注 ⇒ "停驻"按运动学定义：
        # 车辆且全程绝对速率 < 0.5 m/s。这是**定义偏离**，已登记（§NS/A45）。
        spd = np.linalg.norm(np.nan_to_num(v_glob[:, :2]), axis=1)
        is_parked = bool(cls == "vehicle" and valid.sum() and np.nanmax(spd[valid]) < 0.5)

        per_obj[tok] = {
            "cat": cat, "cat_native": native, "cls": cls, "is_parked": is_parked,
            "valid": valid, "p_ego": p_ego, "v_obj_ego": v_obj_ego,
            "in_corridor": in_corridor, "ttc": ttc, "visible": visible,
            "d_long": x, "lat": y, "area_px": area_px, "ecc": ecc,
            "_size": [sz[0], sz[1], sz[2]],
            "_rot": [[np.cos(r["yaw"][j] / 2), 0, 0, np.sin(r["yaw"][j] / 2)] for j in idx],
            "_rot_t": [grid_t[j] for j in idx],
            "_p_cam": p_cam,
        }

    frame_ttc = np.full(n, np.inf)
    for o in per_obj.values():
        frame_ttc = np.minimum(frame_ttc, np.where(o["in_corridor"], o["ttc"], np.inf))

    return dict(frames=frames, grid_t=grid_t, ego_speed=ego_speed, per_obj=per_obj,
                frame_ttc=frame_ttc, ego_xyz=ego_xyz,
                K=K, R_ec=R_ec, t_ec=t_ec, im_w=im_w, im_h=im_h, R_we=R_we)


def main():
    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "navsim_corpus.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=20, help="scene 最少帧数（太短的 scene 无法构窗口）")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"]); out_dir = work / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = sorted((NS_ROOT / "navsim_logs" / args.split).glob("*.pkl"))
    if args.limit_logs:
        logs = logs[: args.limit_logs]
    print(f"[NS1] {len(logs)} 个 log，split={args.split}")

    all_events, stats = [], defaultdict(int)
    n_scene = 0
    for li, lf in enumerate(logs):
        frames = pickle.load(open(lf, "rb"))
        by_scene = defaultdict(list)
        for f in frames:
            by_scene[f["scene_token"]].append(f)
        for stok, fl in by_scene.items():
            fl = sorted(fl, key=lambda z: z["timestamp"])
            if len(fl) < args.min_frames:
                stats["skipped_short_scene"] += 1
                continue
            n_scene += 1
            sname = fl[0]["scene_name"]
            geo = build_geo(fl, cfg, args.split)
            scene = {"name": sname, "token": stok, "description": ""}
            evs = G1.detect_events(geo, scene, cfg)          # ← 判据一行未改
            for ei, e in enumerate(evs):
                clean, ghost = G1.pick_frames(geo, e["t_emergence"], cfg)
                if not clean or not ghost:
                    stats["skipped_no_frames"] += 1
                    continue
                o = geo["per_obj"].get(e["inst"]); k = e["idx"]
                rec = {
                    "event_id": f"{sname}_{ei:03d}_{e['kind']}",
                    "scene_token": stok, "scene_name": sname, "event_type": e["kind"],
                    "t_emergence": e["t_emergence"], "object_token": e["inst"],
                    "object_class": e["obj_class"],
                    "object_class_native": (o or {}).get("cat_native"),
                    "log_name": fl[0]["log_name"], "map_location": fl[0]["map_location"],
                    "min_ttc_1s": None if not np.isfinite(e["min_ttc_1s"]) else float(e["min_ttc_1s"]),
                    "ttc_at_emergence": float(min(geo["frame_ttc"][k], 99.0)),
                    "d_long_at_emergence": float(o["d_long"][k]) if o is not None else None,
                    "lat_at_emergence": float(o["lat"][k]) if o is not None else None,
                    "area_px": (lambda z: None if not np.isfinite(z) else float(z))(
                        np.nanmedian(o["area_px"][ghost]) if o is not None and len(ghost) else np.nan),
                    "ecc": (lambda z: None if not np.isfinite(z) else float(z))(
                        np.nanmedian(o["ecc"][ghost]) if o is not None and len(ghost) else np.nan),
                    "is_night": False, "is_rain": False,      # NAVSIM 日志无天气/光照标注（登记为缺失）
                    "ego_speed_mps": float(geo["ego_speed"][k]),
                    "x_clean_frames": [G1.frame_record(geo, j, o) for j in clean],
                    "x_ghost_frames": [G1.frame_record(geo, j, o) for j in ghost],
                }
                all_events.append(rec)
                stats[f"type_{e['kind']}"] += 1
        if (li + 1) % 20 == 0:
            print(f"[NS1] {li+1}/{len(logs)} log，scene {n_scene}，事件 {len(all_events)}", flush=True)

    with open(out_dir / "events_all.jsonl", "w") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {"split": args.split, "n_logs": len(logs), "n_scenes": n_scene,
               "n_events": len(all_events), "by_type": dict(stats),
               "n_event_scenes": len(set(e["scene_name"] for e in all_events)),
               "maps": sorted(set(e["map_location"] for e in all_events))}
    (out_dir / "navsim_mining_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[NS1] scene {n_scene}，事件 {len(all_events)}：" +
          "  ".join(f"{k[5:]}={v}" for k, v in sorted(stats.items()) if k.startswith("type_")))
    print(f"[NS1] 地图：{summary['maps']}")
    print(f"[NS1] wrote {out_dir/'events_all.jsonl'}")


if __name__ == "__main__":
    main()
