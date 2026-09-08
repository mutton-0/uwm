"""为 C 轴构造新方法的 work 目录：同帧 origin ↔ 遮挡 配对。

C 轴脚本的配对完全由两个图像**文件路径**决定：
  x_ghost_frames[0].filename  -> 危险在场的输入
  x_clean_frames[0].filename  -> 危险移除的输入
旧口径里 clean 是**另一帧**（危险尚未出现），已被新方法废除
（同帧遮挡才是干预，跨帧比较混入了自车运动与场景变化）。

这里把遮挡后的图**预渲染到磁盘**，让 clean 指向它。
于是 C 轴脚本的配对逻辑一行不用改，就变成同帧 origin↔遮挡；
遮挡组 = 归因组全体，与 F-3 逐字同源。
"""
from __future__ import annotations
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                             # noqa: E402
from f3_occlusion_necessity import occlude                              # noqa: E402
from f3_brakefirst import box3d_from_geo                               # noqa: E402

NUSC = "/data/dataset/nuscenes/v1.0-trainval"
NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", required=True, choices=["nuscenes", "navsim"])
    ap.add_argument("--work", required=True)
    args = ap.parse_args()

    G1.set_include_animal(True)
    cands = json.load(open(RES / args.pool))["candidates"]
    work = Path(args.work)
    (work / "mining").mkdir(parents=True, exist_ok=True)
    (work / "masked").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)

    if args.corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
        scmap = {s["name"]: s for s in nusc.scene}
        root = Path(NUSC)

        def geo_of(n, split=None):      # split 仅为签名对齐
            return G1.compute_scene_geometry(nusc, scmap[n], cfg)
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
        root = Path(NS_BLOBS)

        def geo_of(n, split="test"):
            return NS.build_geo(cache[(split, n)], cfg, split)

    out, skipped = [], 0
    for c in cands:
        geo = geo_of(c["scene"], c.get("split", "test"))
        j = c["frame_idx"]; fr = geo["frames"][j]
        img = cv2.imread(str(root / fr["filename"]))
        if img is None:
            skipped += 1; continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = []
        for g in c["f3_mask_group"]:
            o = geo["per_obj"].get(g["token"])
            bb = G1.frame_bbox(geo, o, j) if o is not None else None
            if bb is not None:
                boxes.append(list(bb))
        if not boxes:
            skipped += 1; continue                 # 无可遮挡框 ⇒ 干预不成立，跳过
        masked = rgb.copy()
        for bb in boxes:
            masked = occlude(masked, bb)
        eid = f"{c['scene']}_f{j}"
        mp = work / "masked" / f"{eid}.jpg"
        cv2.imwrite(str(mp), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        spd = float(geo["ego_speed"][j]); t = float(geo["grid_t"][j])
        # 点云键 + 遮挡组 3D 框：DDv2 的 clean 臂必须**同时删点**，否则危险仍在雷达里，
        # 图像 token 的 patch 恢复不了 —— C 轴的 patch-ALL 充分割集自检会直接判负。
        lidar_key = (fr["filename"].split("/", 1)[1] if args.corpus == "navsim"
                     else fr.get("token"))
        b3 = []
        for g in c["f3_mask_group"]:
            bb3 = box3d_from_geo(geo, g["token"], j)
            if bb3 is not None:
                cc, sz, yaw = bb3
                b3.append({"center": list(map(float, cc)), "size": list(map(float, sz)),
                           "yaw": float(yaw)})
        out.append({
            "event_id": eid, "event_type": "A", "scene_name": c["scene"],
            # VLA 侧走 NAVSIM 输入时需要按 (split, scene, frame) 定位
            "split": c.get("split", "test"), "query_frame_idx": j,
            "n_mask_boxes": len(boxes), "a_req": c["a_vru_max"],
            # ghost = 原图（危险在场）；clean = 同帧遮挡后（危险移除）
            "lidar_key": lidar_key, "sd_token": lidar_key, "mask_boxes3d": b3,
            "x_ghost_frames": [{"idx": j, "t": t, "filename": fr["filename"],
                                "ego_speed_mps": spd, "sd_token": lidar_key,
                                "lidar_key": lidar_key}],
            # 绝对路径：遮挡图落在 work 目录，不在语料根下
            "x_clean_frames": [{"idx": j, "t": t, "filename": str(mp.resolve()),
                                "ego_speed_mps": spd, "sd_token": lidar_key,
                                "lidar_key": lidar_key, "delete_boxes3d": b3}],
        })
    p = work / "mining" / "events_all.jsonl"
    with open(p, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"[POOL2C] {args.pool} -> {p}\n"
          f"          {len(out)} 事件（跳过 {skipped}）；遮挡图 -> {work/'masked'}")


if __name__ == "__main__":
    main()
