"""M1 | CARLA in-domain 挖掘 adapter（guide §12.2-M1）。

产出与 nuScenes 挖掘**完全相同的事件 schema**，下游 g2_cache / n1_match / n1_cv 零改动复用。

相对 nuScenes 的三点增强（in-domain 数据本身给的）：
  1. 图像 1024×512 = SimLingo 原生采集分辨率 → 预处理走 carla_native，只做 4.8/16 底裁；
  2. 目标 `speed` 直接给（nuScenes 靠 2Hz 标注插值估计）；
  3. `speed_reduced_by_obj_id/type/distance` = **专家自己的减速归因**，可作金标准危险标签。

事件类型（与 nuScenes 同名者判据一致，便于跨域对照）：
  A      VRU 入走廊且 1s 内 TTC < ttc_max          —— 与 nuScenes 同口径
  Aexp   专家把减速归因于某 walker 的首帧         —— in-domain 金标准正例
  D2a    静态 prop（vendingmachine 等）入走廊     —— 类别对照
  D2aP   static_car（停放车辆）入走廊             —— 类别对照之二
  D2cV   walker 入走廊但全程 TTC > d2c_ttc_min    —— 纯证伪地板（同类别、只差速度）
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

W_IMG, H_IMG, FOV = 1024, 512, 110
FOCAL = W_IMG / (2 * np.tan(np.radians(FOV / 2)))
K = np.array([[FOCAL, 0, W_IMG / 2], [0, FOCAL, H_IMG / 2], [0, 0, 1]])
DT = 0.25                       # 4 Hz（data_save_freq=5 @ 20 FPS）
CAM_DX, CAM_DZ = 1.5, 2.0       # 相机相对 ego 原点：后移 1.5m、抬高 2.0m


def corners_ego(pos, ext, yaw):
    l, w_, h_ = ext
    x = np.array([l, l, l, l, -l, -l, -l, -l])
    y = np.array([w_, -w_, -w_, w_, w_, -w_, -w_, w_])
    z = np.array([h_, h_, -h_, -h_, h_, h_, -h_, -h_])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * x - s * y, s * x + c * y, z], 1) + np.asarray(pos)


def project(pts):
    """ego 系 -> 图像。X_cam=y, Y_cam=CAM_DZ-z, Z_cam=x+CAM_DX（与 utils/projection.py 一致）。"""
    cam = np.stack([pts[:, 1], CAM_DZ - pts[:, 2], pts[:, 0] + CAM_DX], 1)
    if (cam[:, 2] <= 0.3).any():
        return None
    uv = (K @ cam.T).T
    return np.stack([uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]], 1)


def bbox_and_geom(box):
    p = project(corners_ego(box["position"], box["extent"], box["yaw"]))
    if p is None:
        return None
    u0, v0, u1, v1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
    cx, cy = (u0 + u1) / 2, (v0 + v1) / 2
    if not (0 < cx < W_IMG and 0 < cy < H_IMG):
        return None
    ecc = float(np.hypot((cx - W_IMG / 2) / (W_IMG / 2), (cy - H_IMG / 2) / (H_IMG / 2)))
    return {"bbox": [float(u0), float(v0), float(u1), float(v1)],
            "area": float(max(0.0, u1 - u0) * max(0.0, v1 - v0)), "ecc": ecc}


def load_route(route: Path):
    """读一条 route 的逐帧 measurements 与 boxes。"""
    fs = sorted((route / "boxes").glob("*.json.gz"))
    frames = []
    for f in fs:
        m = json.load(gzip.open(route / "measurements" / f.name))
        b = json.load(gzip.open(f))
        frames.append({"stem": f.name[:-8], "m": m, "boxes": b})
    return frames


def obj_kind(box):
    t = box.get("class")
    if t == "walker":
        return "vru"
    if t == "car":
        return "vehicle"
    if t == "static_car":
        return "parked"
    if t == "static":
        return "static"
    return None


def mine_route(route: Path, cfg, rel_root: Path):
    mc = cfg["mining"]
    frames = load_route(route)
    n = len(frames)
    if n < 20:
        return []

    # 逐目标建轨迹。动态目标（walker/car）有稳定的 actor id；
    # 静态物（static / static_car）**没有 id 字段**，用世界坐标合成稳定 ID
    #（ego_matrix 是 ego->world，实测其平移列 == pos_global）。
    tracks = defaultdict(lambda: {"t": [], "box": {}, "kind": None})
    for i, fr in enumerate(frames):
        em = np.array(fr["m"]["ego_matrix"])
        for b in fr["boxes"]:
            k = obj_kind(b)
            if k is None or "extent" not in b:
                continue
            if "id" in b:
                oid = b["id"]
            else:
                wp = em @ np.array(list(b["position"]) + [1.0])
                oid = f"s{round(wp[0] * 2) / 2}_{round(wp[1] * 2) / 2}"
            tr = tracks[(k, oid)]
            tr["t"].append(i); tr["box"][i] = b; tr["kind"] = k

    ego_speed = np.array([f["m"]["speed"] for f in frames])
    per_obj = {}
    for key, tr in tracks.items():
        idx = np.array(sorted(tr["t"]))
        if len(idx) < 3:
            continue
        pos = np.array([tr["box"][i]["position"] for i in idx])
        spd = np.array([tr["box"][i].get("speed", 0.0) or 0.0 for i in idx])
        x, y = pos[:, 0], pos[:, 1]
        in_cor = (np.abs(y) < mc["corridor_half_width_m"]) & \
                 (x > mc["corridor_x_min_m"]) & (x < mc["corridor_x_max_m"])
        # 接近速度：ego 沿 x 前进，目标速度取标量（CARLA 给的是速率）——保守用 ego−obj
        v_c = ego_speed[idx] - spd
        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(v_c > mc["min_closing_speed_mps"], x / v_c, np.inf)
        ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, np.inf)
        geom = [bbox_and_geom(tr["box"][i]) for i in idx]
        per_obj[key] = {"idx": idx, "kind": tr["kind"], "pos": pos, "ttc": ttc,
                        "in_cor": in_cor, "geom": geom, "box": tr["box"]}
    return frames, per_obj, ego_speed


def rising(mask):
    m = mask.astype(int)
    return np.nonzero(np.diff(m) == 1)[0] + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-routes", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    mc = cfg["mining"]
    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])          # CARLA 数据根（复用同名字段）
    scen = root / mc["carla_scenario_dir"]
    routes = sorted(p for p in scen.glob("*") if p.is_dir())
    if args.limit_routes:
        routes = routes[: args.limit_routes]
    (work / "mining").mkdir(parents=True, exist_ok=True)

    n_clean = int(round(abs(mc["clean_window_s"][0] - mc["clean_window_s"][1]) / DT)) or 1
    lead = int(round(mc["min_lead_time_s"] / DT))
    tail = int(round(mc["min_tail_time_s"] / DT))
    c_lo, c_hi = [int(round(v / DT)) for v in mc["clean_window_s"]]
    g_lo, g_hi = [int(round(v / DT)) for v in mc["ghost_window_s"]]
    nf = int(mc["frames_per_condition"])

    def frame_rec(route, frames, i, geom_i):
        f = frames[i]
        rec = {"idx": int(i), "t": float(i * DT),
               "filename": str((route / "rgb" / f"{f['stem']}.jpg").relative_to(root)),
               "sd_token": f"{route.name}:{f['stem']}", "is_key_frame": True,
               "ego_speed_mps": float(f["m"]["speed"]), "im_wh": [W_IMG, H_IMG]}
        rec["bbox_xyxy"] = geom_i["bbox"] if geom_i else None
        return rec

    events = []
    for route in routes:
        out = mine_route(route, cfg, root)
        if not out:
            continue
        frames, per_obj, ego_speed = out
        n = len(frames)
        exp_ids = [f["m"].get("speed_reduced_by_obj_id") for f in frames]

        for (kind, oid), o in per_obj.items():
            pos_of = {i: j for j, i in enumerate(o["idx"])}

            def make(i_e, etype):
                if i_e - lead < 0 or i_e + tail >= n:
                    return None
                cl = [i for i in range(i_e + c_lo, i_e + c_hi + 1) if 0 <= i < n]
                gh = [i for i in range(i_e + g_lo, i_e + g_hi + 1) if 0 <= i < n]
                if not cl or not gh:
                    return None
                cl = cl[:nf]; gh = gh[:nf]
                j = pos_of.get(i_e)
                g = o["geom"][j] if j is not None else None
                w = [pos_of[i] for i in range(i_e, min(n, i_e + 5)) if i in pos_of]
                mt = float(np.min(o["ttc"][w])) if w else np.inf
                return {
                    "event_id": f"{route.name}_{oid}_{i_e:04d}_{etype}",
                    "scene_token": route.name, "scene_name": route.name,
                    "event_type": etype, "t_emergence": float(i_e * DT),
                    "object_token": f"{kind}:{oid}",
                    "object_class": o["box"][i_e].get("type_id", kind) if i_e in o["box"] else kind,
                    "min_ttc_1s": None if not np.isfinite(mt) else mt,
                    "ttc_at_emergence": None if not np.isfinite(mt) else mt,
                    "d_long_at_emergence": float(o["pos"][j][0]) if j is not None else None,
                    "lat_at_emergence": float(o["pos"][j][1]) if j is not None else None,
                    "area_px": g["area"] if g else None, "ecc": g["ecc"] if g else None,
                    "ttc_curve": [], "is_night": False, "is_rain": False,
                    "ego_speed_mps": float(ego_speed[i_e]),
                    "x_clean_frames": [frame_rec(route, frames, i,
                                                 o["geom"][pos_of[i]] if i in pos_of else None) for i in cl],
                    "x_ghost_frames": [frame_rec(route, frames, i,
                                                 o["geom"][pos_of[i]] if i in pos_of else None) for i in gh],
                }

            # ---- A / D2cV：VRU 入走廊 ----
            if o["kind"] == "vru":
                for k in rising(o["in_cor"]):
                    i_e = int(o["idx"][k])
                    w = [j for j in range(k, min(len(o["idx"]), k + 5))]
                    mt = float(np.min(o["ttc"][w])) if w else np.inf
                    et = "A" if mt < mc["event_A"]["ttc_max_s"] else (
                        "D2cV" if mt > mc["event_D2"]["d2c_ttc_min_s"] else None)
                    if et and (e := make(i_e, et)):
                        events.append(e)
            # ---- D2a / D2aP：静态 prop / 停放车辆 入走廊 ----
            elif o["kind"] in ("static", "parked"):
                et = "D2a" if o["kind"] == "static" else "D2aP"
                for k in rising(o["in_cor"]):
                    if (e := make(int(o["idx"][k]), et)):
                        events.append(e)

            # ---- Aexp：专家把减速归因于该 walker 的首帧（in-domain 金标准）----
            if o["kind"] == "vru":
                hit = [i for i in range(n) if exp_ids[i] == oid]
                if hit:
                    for k, i in enumerate(hit):
                        if k == 0 or hit[k - 1] != i - 1:
                            if (e := make(i, "Aexp")):
                                events.append(e)
                            break

    # 去重
    seen = set(); kept = []
    for e in sorted(events, key=lambda x: {"Aexp": 0, "A": 1, "D2a": 2, "D2aP": 2, "D2cV": 3}[x["event_type"]]):
        k = (e["scene_name"], e["object_token"], round(e["t_emergence"], 1))
        if k in seen:
            continue
        seen.add(k); kept.append(e)

    p = work / "mining" / "events_all.jsonl"
    with open(p, "w") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(e["event_type"] for e in kept)
    stats = {"n_events": len(kept), "n_scenes": len(routes), "by_type": dict(c),
             "day": len(kept), "night": 0, "skipped_no_frames": 0,
             "ttc_hist_edges": [0, 1, 2, 3, 4, 6, 10, 100], "ttc_hist_counts": [],
             "config_mining": mc, "frame_rate_hz": 1 / DT}
    (work / "mining" / "mining_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"[M1] routes={len(routes)}  events={len(kept)}  by_type={dict(c)}")
    print(f"[M1] wrote {p}")


if __name__ == "__main__":
    main()
