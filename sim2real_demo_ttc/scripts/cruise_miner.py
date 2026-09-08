"""挖两组**同条件、只差刹不刹车**的帧：cruise（匀速/加速）与 brake（减速）。

这是 F 轴表征侧的第二条路：不做遮挡，直接用两个群体的隐藏层均值作差当减速方向。

    v_decel(band) = mean(h | brake, band) − mean(h | cruise, band)

## 为什么必须按速度分层
自车速度是这些模型的**显式输入**（status_feature / ego_dynamic_state）。
巡航帧速度高且稳、刹车帧是减速起点速度已偏低；不分层的话这个差里最大的成分
是那个速度标量，量到的是"速度方向"不是"减速方向"。**按速度分层后组内速度同分布，
这个混杂在构造上就没了**，且能顺带看方向在各速度段稳不稳。

## 筛选
两组共用的硬条件：
  * 光线：排除雨/夜。nuScenes 读 scene description；NAVSIM 无天气字段，
    只能用时间戳→当地小时筛白天，再加一道图像亮度/饱和度筛，两道各筛掉多少都报。
  * 视野：走廊内最近目标 ≥ MIN_CLEAR_M，且前方目标数 ≤ MAX_FRONT_OBJ。
    "视野越干净越好" —— 这条对两组同样施加，否则 cruise 天然更空，
    差里就混进"场景复杂度"。
分组条件：
  cruise  窗口内 |dv| ≤ CRUISE_DV 或 dv > 0（匀速或加速），且全程无危险归因
  brake   沿用 brake_first 的减速判据（dv < −0.5 且 |dv|/v0 > 0.3）

## 左右舵
benchmark = 左舵（Boston/Vegas/Pittsburgh），deployment = 右舵（新加坡）。
巡航帧两侧都很充裕；**危险帧右舵总共只有 20 个**，右舵那一侧的被减数很弱，
split-half 地板会把这件事暴露出来。
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import pickle
import sys
from collections import defaultdict as _dd
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))

rej_geo = collections.Counter()
SPEED_BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)]
CRUISE_DV = 0.3          # m/s，窗口内速度变化的容差
WIN_S = 2.0              # 判定窗口长度
MIN_CLEAR_M = 15.0       # 走廊内最近目标的纵向距离下限（视野开阔）
MAX_FRONT_OBJ = 6        # 前方 40 m 内的目标数上限
CORRIDOR_M = 2.0
# 城市时区：nuPlan 四城 + nuScenes 两城
TZ = {"us-ma-boston": -4, "us-pa-pittsburgh-hazelwood": -4,
      "us-nv-las-vegas-strip": -7, "sg-one-north": 8,
      "boston-seaport": -4, "singapore-onenorth": 8,
      "singapore-hollandvillage": 8, "singapore-queenstown": 8}
DAY_HOURS = (7, 18)
RHD_KEYS = ("singapore", "sg-")


def side(city):
    c = str(city or "").lower()
    return "RHD" if any(k in c for k in RHD_KEYS) else "LHD"


def band_of(v):
    for i, (a, b) in enumerate(SPEED_BANDS):
        if a <= v < b:
            return i
    return -1


def is_daytime(ts_us, city):
    """时间戳（微秒，UTC）→ 当地小时。NAVSIM 没有天气字段，这是唯一的光线线索。"""
    tz = TZ.get(city)
    if tz is None:
        return None
    h = (datetime.datetime.utcfromtimestamp(ts_us / 1e6).hour + tz) % 24
    return DAY_HOURS[0] <= h < DAY_HOURS[1]


def image_ok(path, lo=55, hi=215, min_sat=12):
    """一道很轻的图像筛：太暗（夜/隧道）、过曝、或饱和度极低（阴雨灰蒙）都剔掉。

    只在缩略图上算，成本可忽略。阈值故意宽松 —— 目的是剔掉明显不合格的，
    不是精确判天气；被它剔掉多少会单独报。
    """
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB").resize((64, 36))
        a = np.asarray(im, np.float32)
    except Exception:                                        # noqa: BLE001
        return None
    lum = a.mean()
    sat = (a.max(2) - a.min(2)).mean()
    return bool(lo <= lum <= hi and sat >= min_sat)


def clear_view(geo, j, corridor=CORRIDOR_M):
    """走廊内最近目标的纵向距离，以及前方 40 m 内的目标数。

    per_obj 的 d_long / lat **已经是 ego 系**（ns1_navsim_geometry.py:113），
    不能再乘一次 R_we —— 那正是 2026-09-04 修掉的速度坐标系同类错误。
    """
    near, n_front = np.inf, 0
    for tok, o in geo["per_obj"].items():
        if j >= len(o["valid"]) or not bool(o["valid"][j]):
            continue
        s, y = float(o["d_long"][j]), float(o["lat"][j])
        if not (np.isfinite(s) and np.isfinite(y)) or s <= 0 or s > 40.0:
            continue
        n_front += 1
        if abs(y) < corridor:
            near = min(near, s)
    return near, n_front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="navsim", choices=["nuscenes", "navsim"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--max-per-scene", type=int, default=2)
    ap.add_argument("--no-image-check", action="store_true")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    if A.corpus == "nuscenes":
        import g1_mine_events as G1
        scenes = G1.list_scenes()
        desc = G1.scene_descriptions() if hasattr(G1, "scene_descriptions") else {}

        def iter_scenes():
            for i, sn in enumerate(scenes):
                if A.limit_scenes and i >= A.limit_scenes:
                    return
                try:
                    yield sn, G1.build_geo(sn), None
                except Exception:                            # noqa: BLE001
                    continue
    else:
        import ns1_navsim_geometry as NS
        from omegaconf import OmegaConf
        # cfg 必须与 brake_first_miner 用同一份，否则走廊宽度等阈值不一致
        cfg = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"), resolve=True)
        logs = sorted((NS.NS_ROOT / "navsim_logs" / A.split).glob("*.pkl"))

        def iter_scenes():
            n = 0
            for lf in logs:
                by = _dd(list)
                for f in pickle.load(open(lf, "rb")):
                    by[f["scene_token"]].append(f)
                for stok, fl in by.items():
                    fl = sorted(fl, key=lambda z: z["timestamp"])
                    if len(fl) < A.min_frames:
                        continue
                    if A.limit_scenes and n >= A.limit_scenes:
                        return
                    n += 1
                    try:
                        yield fl[0]["scene_name"], NS.build_geo(fl, cfg, A.split), fl
                    except Exception as e:                   # noqa: BLE001
                        rej_geo[str(e)[:60]] += 1
                        continue

    out = {"cruise": [], "brake": []}
    rej = collections.Counter()
    for sn, geo, raw in iter_scenes():
        city = (raw[0].get("map_location") if raw else geo.get("location"))
        gt, es = geo["grid_t"], geo["ego_speed"]
        w = max(2, int(round(WIN_S / max(np.median(np.diff(gt)), 1e-3))))
        taken = {"cruise": 0, "brake": 0}
        for j in range(0, len(gt) - w):
            v0, v1 = float(es[j]), float(es[j + w])
            dv = v1 - v0
            b = band_of(v0)
            if b < 0:
                rej["speed_band"] += 1; continue
            if raw is not None:
                day = is_daytime(raw[j]["timestamp"], city)
                if day is False:
                    rej["night"] += 1; continue
            if abs(dv) <= CRUISE_DV or dv > 0:
                kind = "cruise"
            elif dv < -0.5 and abs(dv) / max(v0, 1e-6) > 0.3:
                kind = "brake"
            else:
                rej["neither"] += 1; continue
            near, nf = clear_view(geo, j)
            # **视野开阔只对 cruise 组施加**。对 brake 组同样要求走廊内 15 m 无物，
            # 会把绝大多数刹车事件筛光 —— 人正是因为走廊里有东西才刹的
            # （实测：两组同施加时 brake 为 0）。代价是这个非配对差里含着
            # "前方有无物体"这一项；但那恰是"减速"的定义本身，不是可去除的混杂。
            if kind == "cruise" and (near < MIN_CLEAR_M or nf > MAX_FRONT_OBJ):
                rej["cruise_view_not_clear"] += 1; continue
            if taken[kind] >= A.max_per_scene:
                continue
            fn = geo["frames"][j].get("filename") if geo.get("frames") else None
            if fn and not A.no_image_check:
                base = ("/data/dataset/navsim/dataset/sensor_blobs"
                        if A.corpus == "navsim" else "/data/dataset/nuscenes/v1.0-trainval")
                ok = image_ok(str(Path(base) / fn) if not Path(fn).is_absolute() else fn)
                if ok is False:
                    rej["image_light"] += 1; continue
            taken[kind] += 1
            out[kind].append({"scene": sn, "frame_idx": j, "split": A.split,
                              "city": city, "side": side(city), "band": b,
                              "v0": v0, "dv": dv,
                              "clear_m": None if not np.isfinite(near) else float(near),
                              "n_front": nf, "filename": fn})
    p = Path(A.out) if A.out else RES / f"cruise_pool_{A.corpus}_{A.split}.json"
    meta = {"corpus": A.corpus, "split": A.split, "bands": SPEED_BANDS,
            "win_s": WIN_S, "cruise_dv": CRUISE_DV, "min_clear_m": MIN_CLEAR_M,
            "max_front_obj": MAX_FRONT_OBJ, "rejects": dict(rej), **out}
    p.write_text(json.dumps(meta))
    for k in ("cruise", "brake"):
        c = collections.Counter((r["side"], r["band"]) for r in out[k])
        print(f"{k:7s} 共 {len(out[k]):5d}   " +
              "  ".join(f"{s}/band{b}={n}" for (s, b), n in sorted(c.items())))
    print("剔除原因:", dict(rej))
    if rej_geo:
        print("geo 构建失败:", dict(rej_geo.most_common(3)))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
