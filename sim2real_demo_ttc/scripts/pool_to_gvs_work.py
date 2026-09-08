"""把 brake-first 池转成 G-VS/C 轴流水线吃的 work 目录（旧 schema，新刺激）。

动机：G-VS 与 C 轴的**方法**不依赖速度或走廊判据，只依赖"在哪些帧上测"。
把刺激集从旧语料（288/282 事件，走廊判据已证伪）换成新的 brake-first 池，
方法一行不改，四根轴就落在同一批事件上。

产出 <work>/mining/events_all.jsonl，字段取 gvs1/gvs3 实际用到的那些：
  event_id / event_type / scene_name / x_ghost_frames[0].filename / x_clean_frames[*].ego_speed_mps
`x_ghost_frames` = 查询帧本身（危险在场）。新方法里没有"另一帧的 clean"，
`x_clean_frames` 只被用来取自车速度，故同样填查询帧。
"""
from __future__ import annotations
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", required=True, choices=["nuscenes", "navsim"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--window", type=int, default=0,
                    help="围绕查询帧取 ±window 帧。0 = 只取查询帧。"
                         "取窗口而非全 scene：既给线性探针样本量，又保住"
                         "『帧与被挖出的减速事件相关』这一点，不被无关帧稀释。")
    ap.add_argument("--stride", type=int, default=1, help="窗口内抽帧步长")
    args = ap.parse_args()

    G1.set_include_animal(True)
    cands = json.load(open(RES / args.pool))["candidates"]
    work = Path(args.work); (work / "mining").mkdir(parents=True, exist_ok=True)

    if args.corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval",
                        verbose=False)
        scmap = {s["name"]: s for s in nusc.scene}

        def geo_of(name, split=None):   # split 仅为签名对齐，nuScenes 不分 split
            return G1.compute_scene_geometry(nusc, scmap[name], cfg)
    else:
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"),
                                     resolve=True)
        # **按 (split, scene) 建索引**：scene_name 跨 split 不唯一（test 的 65.8%
        # 与 trainval 重名但指向不同 log），合并池必须逐事件用自带的 split 取数据。
        cache = defaultdict(list)
        for sp in ("test", "trainval"):
            d0 = NS.NS_ROOT / "navsim_logs" / sp
            if not d0.exists():
                continue
            for lf in sorted(d0.glob("*.pkl")):
                for f in pickle.load(open(lf, "rb")):
                    cache[(sp, f["scene_name"])].append(f)
        for k in cache:
            cache[k].sort(key=lambda z: z["timestamp"])

        def geo_of(name, split="test"):
            return NS.build_geo(cache[(split, name)], cfg, split)

    out = []
    for c in cands:
        geo = geo_of(c["scene"], c.get("split", "test"))
        j = c["frame_idx"]
        n = len(geo["frames"])
        idxs = (range(max(0, j - args.window), min(n, j + args.window + 1), args.stride)
                if args.window else [j])
        for k in idxs:
            fr = geo["frames"][k]
            spd = float(geo["ego_speed"][k])
            # DDv2 走点云，nuScenes 侧按 sd_token 索引、NAVSIM 侧按 data_path 索引。
            # 少了这个字段 gvs3 会 KeyError: None。
            lidar_key = (fr["filename"].split("/", 1)[1] if args.corpus == "navsim"
                         else fr.get("token"))
            out.append({
                "event_id": f"{c['scene']}_f{k}",
                "sd_token": lidar_key, "lidar_key": lidar_key,
                "event_type": "A",
                "scene_name": c["scene"],
                "query_frame_idx": j,
                "is_query_frame": bool(k == j),
                "ego_speed_mps": spd,
                "x_ghost_frames": [{"idx": k, "t": float(geo["grid_t"][k]),
                                    "filename": fr["filename"], "ego_speed_mps": spd,
                                    "sd_token": lidar_key, "lidar_key": lidar_key}],
                "x_clean_frames": [{"idx": k, "t": float(geo["grid_t"][k]),
                                    "filename": fr["filename"], "ego_speed_mps": spd,
                                    "sd_token": lidar_key, "lidar_key": lidar_key}],
            })
    # 同一 event_id 去重（同 scene 多个查询帧时可能撞）
    ded = {e["event_id"]: e for e in out}
    p = work / "mining" / "events_all.jsonl"
    with open(p, "w") as f:
        for e in ded.values():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    ns = len({e["scene_name"] for e in ded.values()})
    print(f"[POOL2WORK] {args.pool} -> {p}\n"
          f"            {len(ded)} 帧 / {ns} scene"
          f"  （查询帧 ±{args.window}，步长 {args.stride}）")


if __name__ == "__main__":
    main()
