"""LB1|第二场景类型：**前车急刹** —— 挖掘 + 几何/运动学匹配负例。

工单：docs/severity_gradient_and_second_scenario_workorder.md 任务二。

**为什么要这个场景**：G1 的全部三条正例判据（A/B/C）都属于同一种因果结构——
"新实体突然进入 / 逼近路径"。本场景的危险来源是**已跟踪实体的状态突变**
（一直看得见的前车突然大幅减速），因果结构不同。
它检验的是 G/F/I/C 四轴的**定义**能否迁移，而不是某组阈值能否迁移。

**复用而非新建**：几何计算（走廊、ego 系投影、成像面积/离心率、可见性）
直接 import `g1_mine_events.compute_scene_geometry`，一行未改；
帧选取沿用 `pick_frames`（clean/ghost 窗口定义与 G1 同构）。
本脚本只新增"事件判据 + 负例构造"。

--------------------------------------------------------------------------
事件族（与 G1 的 A / D2a / D2cV 逐条对应）

  **LB**   正例：走廊内已跟踪前车，在窗口内发生**大幅减速**（阈值按分位数定，见 §阈值回调）。
           clean = 刹车起始前的平稳跟车窗口；ghost = 刹车已发生的窗口。
  **LBn**  几何/运动学匹配负例（对应 D2a）：同为走廊内前车，**距离与车速**做 caliper 匹配，
           但窗口内**没有大幅减速**（平稳跟车）。
  **LBv**  该场景的"D2cV 等价物"（**证伪地板**）：同为走廊内前车，距离与车速匹配，
           **确实在减速但幅度小**（|a| 落在 [mild_lo, mild_hi]）。

  LBv 的构造逻辑与 D2cV 同构、但不是复制：
    D2cV  控制住"同类别 VRU + 同成像几何"，让唯一差异落在**相对速度**（单帧结构性不可见）；
    LBv   控制住"同为前车 + 同距离 + 同车速 + **同样在减速**"，
          让唯一差异落在**减速度的幅度**（同样需要帧间差分才能感知）。
  两者共用同一条判据：任何"危险判别力"若不能与证伪地板区分，就不构成危险概念的证据。

**刹车灯的处理（必须随结果一起报的适配偏离）**：
nuScenes **没有刹车灯标注**，无法按刹车灯状态做匹配。
工单允许"按刹车灯标注是否可得来具体设计"，故本脚本改用一条更强的替代构造：
**LBv 的负例本身也在减速**（只是幅度小），因此两臂的刹车灯**很可能都亮**，
差异被压到减速度幅度这一个维度上。
代价是**我们无法核实刹车灯确实亮**，这条记为限制，不作为已验证的控制。
--------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_mine_events import compute_scene_geometry, pick_frames          # noqa: E402

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
VEHICLE_PREFIX = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
                  "vehicle.construction", "vehicle.emergency")


def lead_series(geo, o, cfg):
    """返回该目标作为"走廊内前车"的逐帧量：速度、纵向加速度、距离。"""
    t = geo["grid_t"]
    v_obj = o["v_obj_ego"]                       # 目标**自身**速度（ego 朝向的坐标轴下）
    spd = np.linalg.norm(v_obj[:, :2], axis=1)
    # 纵向加速度 = 目标自身速率对时间的中心差分（负 = 减速）
    a = np.full_like(spd, np.nan)
    if len(t) >= 3:
        a[1:-1] = (spd[2:] - spd[:-2]) / (t[2:] - t[:-2])
        a[0] = (spd[1] - spd[0]) / (t[1] - t[0])
        a[-1] = (spd[-1] - spd[-2]) / (t[-1] - t[-2])
    # "前车"= 在走廊内、在 ego 前方、可见、且本身在动
    is_lead = o["in_corridor"] & o["visible"] & (o["d_long"] > 2.0) & (spd > 1.0)
    return spd, a, is_lead


def scan_scene(geo, scene, cfg, mc):
    """返回该 scene 里全部候选（不做阈值判定，阈值在全局分位数确定后再套）。"""
    t = geo["grid_t"]
    cands = []
    for inst, o in geo["per_obj"].items():
        if not any(o["cat"].startswith(p) for p in VEHICLE_PREFIX):
            continue
        if o.get("is_parked"):
            continue
        spd, a, is_lead = lead_series(geo, o, cfg)
        idx = np.nonzero(is_lead)[0]
        if len(idx) < mc["min_lead_frames"]:
            continue
        for k in idx:
            # 需要 clean 窗口（刹车前）与 ghost 窗口（刹车后）都在场景覆盖内
            if t[k] - t[0] < mc["min_lead_time_s"] or t[-1] - t[k] < mc["min_tail_time_s"]:
                continue
            w = np.nonzero((t >= t[k]) & (t <= t[k] + mc["window_s"]))[0]
            wpre = np.nonzero((t >= t[k] - mc["pre_window_s"]) & (t < t[k]))[0]
            if len(w) < 2 or len(wpre) < 2:
                continue
            if not (is_lead[w].all() and is_lead[wpre].all()):
                continue          # 全窗口都必须是"走廊内可见前车"，否则不是同一因果结构
            a_win = float(np.nanmin(a[w]))                    # 窗口内最强减速（负值）
            a_pre = float(np.nanmin(a[wpre]))                 # 刹车前是否已在减速
            if not (np.isfinite(a_win) and np.isfinite(a_pre)):
                continue
            cands.append({
                "scene_name": scene["name"], "scene_token": scene["token"], "inst": inst,
                "idx": int(k), "t_onset": float(t[k]), "object_class": o["cat"],
                "a_min_window": a_win, "a_min_pre": a_pre,
                "dv_window": float(np.nanmin(spd[w]) - spd[k]),
                "lead_speed_mps": float(spd[k]),
                "d_long_at_onset": float(o["d_long"][k]),
                "lat_at_onset": float(o["lat"][k]),
                "area_px": float(o["area_px"][k]) if np.isfinite(o["area_px"][k]) else None,
                "ecc": float(o["ecc"][k]) if np.isfinite(o["ecc"][k]) else None,
                "ego_speed_mps": float(geo["ego_speed"][k]),
            })
    # 同一目标只保留窗口内减速最强的那一个起始帧（避免同一次刹车被切成多个事件）
    best = {}
    for c in cands:
        key = c["inst"]
        if key not in best or c["a_min_window"] < best[key]["a_min_window"]:
            best[key] = c
    return list(best.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lead_brake.yaml"))
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--survey", action="store_true", help="只出分布，不定阈值不落盘事件")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    mc = cfg["lead_brake"]
    from nuscenes.nuscenes import NuScenes
    paths = cfg["paths"]
    work = Path(paths["work_dir"]); out_dir = work / "mining"; out_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=paths["nuscenes_version"], dataroot=paths["nuscenes_root"], verbose=False)
    scenes = nusc.scene[: args.limit_scenes] if args.limit_scenes else nusc.scene
    print(f"[LB1] {len(scenes)} scenes")

    pool, geos = [], {}
    for si, scene in enumerate(scenes):
        try:
            geo = compute_scene_geometry(nusc, scene, cfg)
        except Exception as exc:                                        # noqa: BLE001
            print(f"[LB1] skip {scene['name']}: {type(exc).__name__}: {exc}"); continue
        c = scan_scene(geo, scene, cfg, mc)
        for x in c:
            x["is_night"] = "night" in scene["description"].lower()
            x["is_rain"] = "rain" in scene["description"].lower()
        pool += c
        geos[scene["name"]] = geo
        if (si + 1) % 50 == 0:
            print(f"[LB1] {si+1}/{len(scenes)} 候选累计 {len(pool)}", flush=True)

    A = np.array([c["a_min_window"] for c in pool])
    print(f"\n[LB1] 候选前车窗口 {len(pool)} 个（{len(set(c['scene_name'] for c in pool))} scene）")
    qs = [1, 2, 5, 10, 25, 50, 75, 90]
    print("[LB1] a_min_window 分位（m/s²，负=减速）：" +
          "  ".join(f"p{q}={np.percentile(A, q):+.2f}" for q in qs))
    (out_dir / "lead_brake_survey.json").write_text(json.dumps(
        {"n_candidates": len(pool), "n_scenes": len(set(c["scene_name"] for c in pool)),
         "a_min_window_percentiles": {f"p{q}": float(np.percentile(A, q)) for q in qs},
         "lead_speed_percentiles": {f"p{q}": float(np.percentile(
             [c["lead_speed_mps"] for c in pool], q)) for q in qs},
         "d_long_percentiles": {f"p{q}": float(np.percentile(
             [c["d_long_at_onset"] for c in pool], q)) for q in qs}},
        indent=2, ensure_ascii=False))
    if args.survey:
        print(f"[LB1] survey 模式，已写 {out_dir/'lead_brake_survey.json'}"); return

    np.save(out_dir / "_lb_pool.npy", np.array(pool, dtype=object), allow_pickle=True)
    print(f"[LB1] 候选池落盘 {out_dir/'_lb_pool.npy'}（阈值与匹配由 lb2_build_events.py 完成）")


if __name__ == "__main__":
    main()
