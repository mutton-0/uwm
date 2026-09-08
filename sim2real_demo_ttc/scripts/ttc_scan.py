"""全量数据集按 TTC 筛危险场景 —— **不看人有没有刹车**。

## 为什么必须换口径
brake-first 挖矿找的是「人类先减速 → 再归因到前方物体」，查询帧取减速起始点。
入池前提是**司机看见了并且提前踩了刹车**，所以 TTC 在那一刻当然充裕
（实测中位 5.55 s，全库 1762 个事件 **TTC<1.5 s 的一个都没有**，最小 1.80 s）。
碰撞与险情恰恰是"人没来得及踩"的那些，被这套判据**从定义上排除**了。

本脚本反过来：扫全量场景的**每一帧**，取走廊内正在接近的物体的 TTC 最小值
（`geo["frame_ttc"]`，与挖掘期同一份定义），按阈值筛，不问人的反应。

阈值分档（依据见 results/rhd_stepA_report.md 与文献）：
    TTC < 1.0 s  严重 —— 对齐 NAVSIM PDM 自己的 future_collision_horizon_window
    TTC < 1.5 s  危险 —— 文献主流，且我们的 GT 是人类驾驶
    TTC < 2.0 s  偏保守筛选
    TTC < 3.0 s  文献临界区间上沿
"""
from __future__ import annotations
import argparse, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
import sys                                                            # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
MAP = json.load(open(RES / "driveside_map.json"))
CUTS = [1.0, 1.5, 2.0, 3.0]

# **必需的门**（首版无此门，实测出来的全是误检）：
#   走廊半宽 2.0 m、纵向下限 0.0 m ⇒ 并排经过的路侧护栏/锥桶会短暂落进走廊，
#   d_long≈0.1 m 除以接近速度得到 TTC≈0.1 s。实测右舵最危险的 12 个场景里，
#   |lat| 全在 1.7–1.9 m（贴走廊边缘）、d_long 全在 0.06–0.62 m（与车并排），
#   逐个调图确认：开阔道路 + 路侧施工护栏，前方无任何危险。
MIN_D_LONG = 5.0        # 物体必须**确实在前方**：0.5 m 处的东西不是"碰撞时间"，是已经在那了
MAX_ABS_LAT = 1.2       # 留出横向余量，贴走廊边缘的不算
MIN_RUN = 3             # 至少连续 3 帧低于阈值，排除单帧抖动
# 路侧静物：它们是道路家具，不是"司机必须为之刹车"的危险源
FURNITURE = ("barrier", "trafficcone", "debris", "pushable_pullable", "movable_object")
# **第二类误检**（逐个调图确认）：路口横穿 / 对向车流。
#   实测 scene-1100/0737/0065 的物体纵向速度是 −9.6 / −8.1 / −12.1 m/s，
#   即迎面而来；图上确认全是路口横穿的车，不是前车也不是静止障碍物。
#   要的是「前车或 obs」⇒ 接近必须**主要来自自车运动**：
MIN_V_OBJ_ALONG = -2.0  # 物体纵向速度下限（ego 系）：低于此为迎面/横穿，剔除
MAX_V_OBJ_LAT = 2.0     # 横向速度上限：横穿车的横向速度大
MIN_EGO_SPEED = 2.0     # 自车必须在动：停着的车没有"需要制动的危险"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["nuscenes", "navsim"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    rows = []

    if A.corpus == "navsim":
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"),
                                     resolve=True)
        n = 0
        for sp in ("test", "trainval"):
            d0 = NS.NS_ROOT / "navsim_logs" / sp
            if not d0.exists():
                continue
            for lf in sorted(d0.glob("*.pkl")):
                by = defaultdict(list)
                for f in pickle.load(open(lf, "rb")):
                    by[f["scene_name"]].append(f)
                for scn, fl in by.items():
                    side = MAP["navsim_side"].get(f"{sp}|{scn}")
                    try:
                        geo = NS.build_geo(sorted(fl, key=lambda z: z["timestamp"]), cfg, sp)
                    except Exception:                                  # noqa: BLE001
                        continue
                    rows.append(_row(geo, scn, sp, side))
                    n += 1
                    if n % 500 == 0:
                        print(f"  navsim {n} scene", flush=True)
                    if A.limit and n >= A.limit:
                        break
                if A.limit and n >= A.limit:
                    break
            if A.limit and n >= A.limit:
                break
    else:
        import g1_mine_events as G1
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"),
                                     resolve=True)
        nusc = NuScenes(version="v1.0-trainval",
                        dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
        G1.set_include_animal(True)
        for i, sc in enumerate(nusc.scene):
            if A.limit and i >= A.limit:
                break
            try:
                geo = G1.compute_scene_geometry(nusc, sc, cfg)
            except Exception:                                          # noqa: BLE001
                continue
            rows.append(_row(geo, sc["name"], "trainval",
                             MAP["nuscenes_side"].get(sc["name"])))
            if (i + 1) % 100 == 0:
                print(f"  nuscenes {i+1}/{len(nusc.scene)}", flush=True)

    out = Path(A.out) if A.out else RES / f"ttc_scan_{A.corpus}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False))
    _report(rows, A.corpus)
    print(f"\n-> {out}")


def _gated_ttc(geo):
    """重算逐帧 TTC，只计**确实在正前方、非路侧家具**的物体。"""
    n = len(geo["grid_t"])
    out = np.full(n, np.inf)
    who = [None] * n
    ego = np.asarray(geo["ego_speed"], float)
    for tk, o in geo["per_obj"].items():
        cat = str(o.get("cat_native") or o.get("cat") or "")
        if any(f in cat for f in FURNITURE):
            continue
        t = np.asarray(o["ttc"], float)
        v = np.asarray(o["v_obj_ego"], float)
        va = v[:, 0] if v.ndim > 1 else np.zeros(len(t))
        vl = v[:, 1] if v.ndim > 1 and v.shape[1] > 1 else np.zeros(len(t))
        ok = (np.asarray(o["in_corridor"], bool)
              & (np.asarray(o["d_long"], float) >= MIN_D_LONG)
              & (np.abs(np.asarray(o["lat"], float)) <= MAX_ABS_LAT)
              & (va >= MIN_V_OBJ_ALONG) & (np.abs(vl) <= MAX_V_OBJ_LAT)
              & (ego >= MIN_EGO_SPEED)
              & np.isfinite(t))
        cand = np.where(ok, t, np.inf)
        upd = cand < out
        out = np.minimum(out, cand)
        for j in np.where(upd)[0]:
            who[j] = (cat, float(o["d_long"][j]), float(o["lat"][j]))
    return out, who


def _runs_below(ft, c, k=MIN_RUN):
    """连续 k 帧以上低于 c 的最长游程长度。"""
    b = (ft < c).astype(int)
    best = cur = 0
    for x in b:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def _row(geo, scn, split, side):
    ft, who = _gated_ttc(geo)
    fin = ft[np.isfinite(ft)]
    r = {"scene": scn, "split": split, "side": side, "n_frames": int(len(ft)),
         "min_ttc": (float(fin.min()) if len(fin) else None)}
    for c in CUTS:
        r[f"n_frames_lt_{c}"] = int((ft < c).sum())
        r[f"run_lt_{c}"] = _runs_below(ft, c)
    if len(fin) and fin.min() < 3.0:
        j = int(np.nanargmin(np.where(np.isfinite(ft), ft, np.inf)))
        r["argmin_frame"] = j
        w = who[j]
        r["argmin_cat"] = (w[0] if w else None)
        r["argmin_d_long"] = (w[1] if w else None)
        r["argmin_lat"] = (w[2] if w else None)
        r["argmin_ego_speed"] = float(geo["ego_speed"][j])
    return r


def _report(rows, corpus):
    print(f"\n===== {corpus}：{len(rows)} 个场景 =====")
    print(f"门：d_long ≥ {MIN_D_LONG} m，|lat| ≤ {MAX_ABS_LAT} m，排除路侧家具，"
          f"物体纵向速度 ≥ {MIN_V_OBJ_ALONG} 且 |横向| ≤ {MAX_V_OBJ_LAT} m/s（排除横穿/对向），"
          f"自车 ≥ {MIN_EGO_SPEED} m/s，连续 ≥ {MIN_RUN} 帧")
    print(f"{'舵位':<6}{'场景数':>7}{'有TTC':>7}" + "".join(f"{'<'+str(c)+'s':>10}" for c in CUTS))
    print("-" * 60)
    for side in ("LHD", "RHD", None):
        g = [r for r in rows if r["side"] == side]
        if not g:
            continue
        h = [r for r in g if r["min_ttc"] is not None]
        cells = "".join(f"{sum(1 for r in h if r['min_ttc'] < c and r[f'run_lt_{c}'] >= MIN_RUN):>10}"
                        for c in CUTS)
        print(f"{str(side):<6}{len(g):>7}{len(h):>7}{cells}")


if __name__ == "__main__":
    main()
