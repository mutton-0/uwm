"""G1 | nuScenes TTC 突变场景挖掘（纯 CPU，只用 metadata）。

手册 §5：
  - 逐帧 TTC：用 sweeps 时间网格（~12Hz）+ 关键帧标注线性插值
  - 事件 A(VRU 突现) / B(近距 cut-in) / C(通用 TTC 骤降) / D(无害出现，负例)
  - 产出 events_all.jsonl + 统计报告 + split

所有阈值与路径来自 config，换 Tier-M/L 数据零改动复跑。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from pyquaternion import Quaternion

VRU_PREFIXES = ("human.pedestrian.", "vehicle.bicycle", "vehicle.motorcycle")
VEHICLE_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
                    "vehicle.construction", "vehicle.emergency")


def obj_class(name: str) -> str:
    if name.startswith(VRU_PREFIXES):
        return "vru"
    if name.startswith(VEHICLE_PREFIXES):
        return "vehicle"
    return "other"


# --------------------------------------------------------------------------------------
# 场景时间网格与轨迹
# --------------------------------------------------------------------------------------
def scene_camera_frames(nusc, scene, camera: str):
    """返回该 scene 全部 CAM_FRONT sample_data（关键帧 + sweeps），按时间排序。"""
    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd = nusc.get("sample_data", first_sample["data"][camera])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    frames = []
    while True:
        pose = nusc.get("ego_pose", sd["ego_pose_token"])
        frames.append({
            "token": sd["token"],
            "t": sd["timestamp"] * 1e-6,
            "filename": sd["filename"],
            "is_key_frame": sd["is_key_frame"],
            "ego_t": np.array(pose["translation"]),
            "ego_q": Quaternion(pose["rotation"]),
            "calib_token": sd["calibrated_sensor_token"],
            "width": sd["width"], "height": sd["height"],
        })
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])
    frames.sort(key=lambda f: f["t"])
    return frames


def scene_tracks(nusc, scene):
    """按 instance 聚合关键帧标注：{instance_token: dict(t[], xyz[], cat, size)}。"""
    tracks = defaultdict(lambda: {"t": [], "xyz": [], "cat": None, "size": None,
                                  "ann_tokens": [], "visibility": []})
    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        t = sample["timestamp"] * 1e-6
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            tr = tracks[ann["instance_token"]]
            tr["t"].append(t)
            tr["xyz"].append(np.array(ann["translation"]))
            tr["cat"] = ann["category_name"]
            tr["size"] = ann["size"]
            tr["ann_tokens"].append(ann_token)
            tr["visibility"].append(ann["visibility_token"])
        sample_token = sample["next"]
    for tr in tracks.values():
        tr["t"] = np.asarray(tr["t"])
        tr["xyz"] = np.asarray(tr["xyz"])
    return dict(tracks)


def interp_track(tr, grid_t):
    """把轨迹线性插值到时间网格；返回 (xyz[T,3], valid[T])。"""
    t = tr["t"]
    valid = (grid_t >= t[0]) & (grid_t <= t[-1])
    xyz = np.full((len(grid_t), 3), np.nan)
    if valid.sum() == 0 or len(t) < 2:
        return xyz, valid & False
    for d in range(3):
        xyz[valid, d] = np.interp(grid_t[valid], t, tr["xyz"][:, d])
    return xyz, valid


def central_diff(x, t):
    """对 [T, D] 数组沿时间做中心差分（端点用单边）。"""
    v = np.full_like(x, np.nan)
    if len(t) < 2:
        return v
    v[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None]
    v[0] = (x[1] - x[0]) / (t[1] - t[0])
    v[-1] = (x[-1] - x[-2]) / (t[-1] - t[-2])
    return v


def track_velocity(xyz, valid, t):
    """只在轨迹有效区间内做差分。

    直接对 nan_to_num 后的整条数组差分，会在有效区间边界拿 (真实坐标 - 0)/dt，
    产生上千 m/s 的假速度 → TTC 出现 ~0s 的假尖峰。必须先切到有效段。
    """
    v = np.full_like(xyz, np.nan)
    idx = np.nonzero(valid)[0]
    if len(idx) < 2:
        return v
    v[idx] = central_diff(xyz[idx], t[idx])
    return v


# --------------------------------------------------------------------------------------
# 每帧几何量
# --------------------------------------------------------------------------------------
def compute_scene_geometry(nusc, scene, cfg):
    """返回该 scene 每帧、每目标在 ego 系下的位置/速度/TTC/走廊标志/可见性。"""
    mcfg = cfg["mining"]
    frames = scene_camera_frames(nusc, scene, mcfg["camera"])
    grid_t = np.array([f["t"] for f in frames])
    ego_xyz = np.stack([f["ego_t"] for f in frames])
    ego_v_global = central_diff(ego_xyz, grid_t)          # [T,3] global

    # ego 系旋转（world->ego）
    R_we = np.stack([f["ego_q"].rotation_matrix for f in frames])   # ego->world
    ego_speed = np.linalg.norm(ego_v_global[:, :2], axis=1)

    # 相机内外参（用于视野判定）
    calib = nusc.get("calibrated_sensor", frames[0]["calib_token"])
    K = np.array(calib["camera_intrinsic"])
    R_ec = Quaternion(calib["rotation"]).rotation_matrix    # cam->ego
    t_ec = np.array(calib["translation"])
    im_w, im_h = frames[0]["width"], frames[0]["height"]

    tracks = scene_tracks(nusc, scene)
    per_obj = {}
    for inst, tr in tracks.items():
        cls = obj_class(tr["cat"])
        if cls == "other":
            continue
        xyz_g, valid = interp_track(tr, grid_t)
        if valid.sum() < 3:
            continue
        v_g = track_velocity(xyz_g, valid, grid_t)

        # world -> ego
        rel = xyz_g - ego_xyz
        p_ego = np.einsum("tij,tj->ti", np.transpose(R_we, (0, 2, 1)), rel)
        v_obj_ego = np.einsum("tij,tj->ti", np.transpose(R_we, (0, 2, 1)), v_g)
        v_ego_ego = np.einsum("tij,tj->ti", np.transpose(R_we, (0, 2, 1)), ego_v_global)

        x, y = p_ego[:, 0], p_ego[:, 1]
        in_corridor = (np.abs(y) < mcfg["corridor_half_width_m"]) & \
                      (x > mcfg["corridor_x_min_m"]) & (x < mcfg["corridor_x_max_m"]) & valid

        # 连线方向上的接近速度
        d_xy = np.linalg.norm(p_ego[:, :2], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            los = p_ego[:, :2] / d_xy[:, None]
            v_c = np.sum((v_ego_ego[:, :2] - v_obj_ego[:, :2]) * los, axis=1)
            ttc = np.where(v_c > mcfg["min_closing_speed_mps"], x / v_c, np.inf)
        ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, np.inf)
        ttc[~valid] = np.inf

        # 相机可见性（中心点投影进画幅）
        p_cam = np.einsum("ij,tj->ti", R_ec.T, p_ego - t_ec)
        with np.errstate(invalid="ignore", divide="ignore"):
            uv = (K @ p_cam.T).T
            u = uv[:, 0] / uv[:, 2]
            v_img = uv[:, 1] / uv[:, 2]
        visible = valid & (p_cam[:, 2] > 0.1) & (u > 0) & (u < im_w) & (v_img > 0) & (v_img < im_h)

        per_obj[inst] = {
            "cat": tr["cat"], "cls": cls, "valid": valid, "p_ego": p_ego,
            "v_obj_ego": v_obj_ego, "in_corridor": in_corridor, "ttc": ttc,
            "visible": visible, "d_long": x, "lat": y,
        }

    frame_ttc = np.full(len(grid_t), np.inf)
    for o in per_obj.values():
        frame_ttc = np.minimum(frame_ttc, np.where(o["in_corridor"], o["ttc"], np.inf))

    return dict(frames=frames, grid_t=grid_t, ego_speed=ego_speed, per_obj=per_obj,
                frame_ttc=frame_ttc, ego_xyz=ego_xyz)


# --------------------------------------------------------------------------------------
# 事件检测
# --------------------------------------------------------------------------------------
def rising_edges(mask):
    """返回 mask 由 False 变 True 的索引。"""
    m = mask.astype(int)
    return np.nonzero(np.diff(m) == 1)[0] + 1


def detect_events(geo, scene, cfg):
    mcfg = cfg["mining"]
    grid_t, per_obj, frame_ttc = geo["grid_t"], geo["per_obj"], geo["frame_ttc"]
    t0, t1 = grid_t[0], grid_t[-1]
    events = []

    def window_idx(t_e, lo, hi):
        return np.nonzero((grid_t >= t_e + lo) & (grid_t <= t_e + hi))[0]

    def lead_tail_ok(t_e):
        return (t_e - t0) >= mcfg["min_lead_time_s"] and (t1 - t_e) >= mcfg["min_tail_time_s"]

    # ---- A / B：按目标进入走廊 ----
    for inst, o in per_obj.items():
        for k in rising_edges(o["in_corridor"]):
            t_e = grid_t[k]
            if not lead_tail_ok(t_e):
                continue
            w = window_idx(t_e, 0.0, mcfg["event_A"]["window_s"])
            ttc_win = o["ttc"][w]
            min_ttc = float(np.min(ttc_win)) if len(w) else np.inf

            etype = None
            if o["cls"] == "vru" and mcfg["event_A"]["enabled"] and min_ttc < mcfg["event_A"]["ttc_max_s"]:
                etype = "A"
            elif o["cls"] == "vehicle" and mcfg["event_B"]["enabled"]:
                lat, v_lat = o["lat"][k], o["v_obj_ego"][k, 1]
                toward = (np.sign(v_lat) != np.sign(lat)) and abs(v_lat) > mcfg["event_B"]["min_lateral_speed_mps"]
                # 手册原文是 "d_long<20 或 TTC<4"；OR 会收进 "ego 静止 + TTC 15s" 这类无危险样本，
                # 故改为 AND（近距 *且* 时间紧迫），回调记录见 config 注释。
                close = (o["d_long"][k] < mcfg["event_B"]["d_long_max_m"]
                         and min_ttc < mcfg["event_B"]["ttc_max_s"])
                if toward and close:
                    etype = "B"
            if etype:
                events.append(dict(kind=etype, t_emergence=float(t_e), idx=int(k), inst=inst,
                                   min_ttc_1s=min_ttc, obj_class=o["cat"]))

    # ---- C：帧级 TTC 骤降（兜底）----
    if mcfg["event_C"]["enabled"]:
        dt = float(np.median(np.diff(grid_t)))
        win = max(1, int(round(mcfg["event_C"]["drop_window_s"] / dt)))
        capped = np.where(np.isfinite(frame_ttc), frame_ttc, 30.0)
        for k in range(win, len(grid_t)):
            drop = capped[k - win] - capped[k]
            if drop >= mcfg["event_C"]["min_drop_s"] and capped[k] < mcfg["event_C"]["ttc_after_max_s"]:
                t_e = grid_t[k]
                if not lead_tail_ok(t_e):
                    continue
                # 归因到当帧走廊内 TTC 最小的目标
                best, best_ttc = None, np.inf
                for inst, o in per_obj.items():
                    if o["in_corridor"][k] and o["ttc"][k] < best_ttc:
                        best, best_ttc = inst, o["ttc"][k]
                events.append(dict(kind="C", t_emergence=float(t_e), idx=int(k), inst=best,
                                   min_ttc_1s=float(best_ttc),
                                   obj_class=per_obj[best]["cat"] if best else "unknown"))

    # ---- D：无害出现（负例）----
    if mcfg["event_D"]["enabled"]:
        d_events = []
        for inst, o in per_obj.items():
            for k in rising_edges(o["visible"]):
                t_e = grid_t[k]
                if not lead_tail_ok(t_e):
                    continue
                w = window_idx(t_e, 0.0, 2.0)
                if len(w) == 0:
                    continue
                enters_corridor = bool(o["in_corridor"][w].any())
                min_ttc = float(np.min(o["ttc"][w]))
                # 严格判据三条同时成立：
                #  (1) 不入走廊 —— 手册原判据；
                #  (2) 自身 TTC 全程安全 —— 只用 (1) 会把擦着走廊边缘、TTC 很小的横穿目标当负例；
                #  (3) 帧级 TTC 也安全 —— 否则负例帧里还站着*别的*真危险目标，
                #      §7.1 的 "Acc(ghost vs D)" 硬指标就被污染了。
                frame_safe = bool(np.min(geo["frame_ttc"][w]) > mcfg["event_D"]["ttc_min_s"])
                harmless = (not enters_corridor) and (min_ttc > mcfg["event_D"]["ttc_min_s"]) and frame_safe
                if harmless:
                    d_events.append(dict(kind="D", t_emergence=float(t_e), idx=int(k), inst=inst,
                                         min_ttc_1s=min_ttc, obj_class=o["cat"],
                                         _d_long=float(o["d_long"][k])))
        # 优先保留近距的无害出现（更强的对照），控制数量
        d_events.sort(key=lambda e: e["_d_long"])
        for e in d_events[: mcfg["event_D"]["max_per_scene"]]:
            e.pop("_d_long")
            events.append(e)

    # 去重：同目标 0.8s 内多次触发只留优先级最高的（A > B > C > D）
    prio = {"A": 0, "B": 1, "C": 2, "D": 3}
    events.sort(key=lambda e: (prio[e["kind"]], e["t_emergence"]))
    kept = []
    for e in events:
        dup = any(k["inst"] == e["inst"] and abs(k["t_emergence"] - e["t_emergence"]) < 0.8 for k in kept)
        if not dup:
            kept.append(e)
    kept.sort(key=lambda e: e["t_emergence"])
    return kept


def pick_frames(geo, t_e, cfg):
    """选 clean / ghost 帧（相对 t_emergence 的窗口内均匀取 n 帧）。"""
    mcfg = cfg["mining"]
    grid_t, frames = geo["grid_t"], geo["frames"]
    n = int(mcfg["frames_per_condition"])

    def pick(lo, hi):
        idx = np.nonzero((grid_t >= t_e + lo) & (grid_t <= t_e + hi))[0]
        if len(idx) == 0:
            return []
        targets = np.linspace(t_e + lo, t_e + hi, n + 2)[1:-1] if n > 1 else [t_e + (lo + hi) / 2]
        chosen = []
        for tt in targets:
            j = idx[np.argmin(np.abs(grid_t[idx] - tt))]
            if j not in chosen:
                chosen.append(int(j))
        return chosen

    clean = pick(*mcfg["clean_window_s"])
    ghost = pick(*mcfg["ghost_window_s"])
    return clean, ghost


def frame_record(geo, j):
    f = geo["frames"][j]
    return {"idx": int(j), "t": float(geo["grid_t"][j]), "filename": f["filename"],
            "sd_token": f["token"], "is_key_frame": bool(f["is_key_frame"]),
            "ego_speed_mps": float(geo["ego_speed"][j])}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    from nuscenes.nuscenes import NuScenes

    paths, mcfg = cfg["paths"], cfg["mining"]
    work = Path(paths["work_dir"])
    out_dir = work / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=paths["nuscenes_version"], dataroot=paths["nuscenes_root"], verbose=False)
    scenes = nusc.scene[: args.limit_scenes] if args.limit_scenes else nusc.scene
    print(f"[G1] {len(scenes)} scenes, version={paths['nuscenes_version']}")

    all_events, stats = [], defaultdict(int)
    ttc_all = []
    for si, scene in enumerate(scenes):
        geo = compute_scene_geometry(nusc, scene, cfg)
        evs = detect_events(geo, scene, cfg)
        is_night = "night" in scene["description"].lower()
        rain = "rain" in scene["description"].lower()

        for ei, e in enumerate(evs):
            clean, ghost = pick_frames(geo, e["t_emergence"], cfg)
            if not clean or not ghost:
                stats["skipped_no_frames"] += 1
                continue
            o = geo["per_obj"].get(e["inst"])
            k = e["idx"]
            w = np.nonzero((geo["grid_t"] >= e["t_emergence"] - 1.5) &
                           (geo["grid_t"] <= e["t_emergence"] + 1.5))[0]
            ttc_curve = [[float(geo["grid_t"][j] - e["t_emergence"]),
                          float(min(geo["frame_ttc"][j], 99.0))] for j in w]
            rec = {
                "event_id": f"{scene['name']}_{ei:03d}_{e['kind']}",
                "scene_token": scene["token"],
                "scene_name": scene["name"],
                "event_type": e["kind"],
                "t_emergence": e["t_emergence"],
                "object_token": e["inst"],
                "object_class": e["obj_class"],
                "min_ttc_1s": None if not np.isfinite(e["min_ttc_1s"]) else float(e["min_ttc_1s"]),
                "ttc_at_emergence": float(min(geo["frame_ttc"][k], 99.0)),
                "d_long_at_emergence": float(o["d_long"][k]) if o is not None else None,
                "lat_at_emergence": float(o["lat"][k]) if o is not None else None,
                "ttc_curve": ttc_curve,
                "is_night": bool(is_night),
                "is_rain": bool(rain),
                "ego_speed_mps": float(geo["ego_speed"][k]),
                "x_clean_frames": [frame_record(geo, j) for j in clean],
                "x_ghost_frames": [frame_record(geo, j) for j in ghost],
            }
            all_events.append(rec)
            stats[f"type_{e['kind']}"] += 1
            stats["night" if is_night else "day"] += 1
            if rec["min_ttc_1s"] is not None:
                ttc_all.append(rec["min_ttc_1s"])
        print(f"[G1] {scene['name']:>12}  frames={len(geo['frames']):4d}  events={len(evs):3d}  "
              f"{'NIGHT' if is_night else 'day  '}  {scene['description'][:44]}")

    ev_path = out_dir / "events_all.jsonl"
    with open(ev_path, "w") as f:
        for e in all_events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    hist_edges = [0, 1, 2, 3, 4, 6, 10, 100]
    hist = np.histogram(np.clip(ttc_all, 0, 99), bins=hist_edges)[0].tolist() if ttc_all else []
    summary = {
        "n_events": len(all_events),
        "n_scenes": len(scenes),
        "by_type": {k[5:]: v for k, v in sorted(stats.items()) if k.startswith("type_")},
        "day": stats["day"], "night": stats["night"],
        "skipped_no_frames": stats["skipped_no_frames"],
        "ttc_hist_edges": hist_edges,
        "ttc_hist_counts": hist,
        "config_mining": mcfg,
    }
    (out_dir / "mining_stats.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[G1] events={len(all_events)}  by_type={summary['by_type']}  day={stats['day']} night={stats['night']}")
    print(f"[G1] TTC(min_1s) hist edges={hist_edges} counts={hist}")
    print(f"[G1] wrote {ev_path} + mining_stats.json")


if __name__ == "__main__":
    main()
