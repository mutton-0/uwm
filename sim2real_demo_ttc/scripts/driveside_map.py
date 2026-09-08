"""scene -> city -> 左舵/右舵 的映射，落盘供各轴复用。

右舵 (RHD, 方向盘在右、靠左行驶) = 新加坡三区 / sg-one-north
左舵 (LHD) = Boston / Las Vegas / Pittsburgh

nuScenes 用 scene 名直接索引；NavSim 的 scene 名按 split 各自编号，
必须用 (split, scene) 复合键 —— 单用 scene 名会把两个 split 的不同 log 混起来
（实测 1334/2026 重名）。此坑与 build_deploy_pool 一致。
"""
from __future__ import annotations
import glob
import json
import multiprocessing as mp
import pickle
from pathlib import Path

NUSC = Path("/data/dataset/nuscenes/v1.0-trainval/v1.0-trainval")
NAV = Path("/data/dataset/navsim/dataset/navsim_logs")
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = RES / "driveside_map.json"
RHD_KEYS = ("singapore", "sg-")


def side(city):
    if not city:
        return None
    c = str(city).lower()
    return "RHD" if any(k in c for k in RHD_KEYS) else "LHD"


def _nav_one(path):
    split = Path(path).parent.name
    try:
        with open(path, "rb") as f:
            frames = pickle.load(f)
    except Exception:                                        # noqa: BLE001
        return []
    return [(f"{split}|{fr['scene_name']}", fr.get("map_location"))
            for fr in frames if fr.get("scene_name")]


def build(workers=16):
    logs = {l["token"]: l["location"]
            for l in json.load(open(NUSC / "log.json"))}
    nusc = {s["name"]: logs.get(s["log_token"])
            for s in json.load(open(NUSC / "scene.json"))}
    files = []
    for s in ("trainval", "test"):
        files += sorted(glob.glob(str(NAV / s / "*.pkl")))
    nav = {}
    with mp.Pool(workers) as p:
        for d in p.imap_unordered(_nav_one, files, chunksize=4):
            nav.update(dict(d))
    return nusc, nav


def main():
    nusc, nav = build()
    out = {"nuscenes_city": nusc, "navsim_city": nav,
           "nuscenes_side": {k: side(v) for k, v in nusc.items()},
           "navsim_side": {k: side(v) for k, v in nav.items()}}
    OUT.write_text(json.dumps(out))
    import collections
    print(f"nuScenes {len(nusc)} scene, NavSim {len(nav)} (split|scene) 键")
    for nm, d in (("nuScenes", out["nuscenes_side"]), ("NavSim", out["navsim_side"])):
        c = collections.Counter(d.values())
        print(f"  {nm:9s} RHD {c['RHD']:6d}   LHD {c['LHD']:6d}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
