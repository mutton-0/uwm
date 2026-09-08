"""右舵鬼探头 10 场景：灰斑 vs 路面色(road_flat) 两种填充下的 b 对照。

夜景是灰斑最可能出问题的条件 —— 过曝背景里灰斑是个突兀的暗块。
P-7 已把默认改为 road_flat，但这批池的遮挡图是早先用灰斑渲的。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(RES))
import g1_mine_events as G1                                          # noqa: E402
from f3_occlusion_necessity import occlude                           # noqa: E402
from c_axis_hazard_patch import arc_full, WP_DT                      # noqa: E402
NUSC = "/data/dataset/nuscenes/v1.0-trainval"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=["dd", "ltf", "ddv2"])
ap.add_argument("--device", default="cuda:1")
A = ap.parse_args()

from nuscenes.nuscenes import NuScenes                               # noqa: E402
cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
nusc = NuScenes(version="v1.0-trainval", dataroot=NUSC, verbose=False)
G1.set_include_animal(True)
SC = {s["name"]: s for s in nusc.scene}
MAP = json.load(open(RES / "driveside_map.json"))["nuscenes_side"]
pool = json.load(open(RES / "brake_first_pool_final.json"))
cands = [x for x in (pool.get("candidates") or []) + (pool.get("deferred_low_a_req") or [])
         if MAP.get(x["scene"]) == "RHD"]
evs = {json.loads(l)["scene_name"]: json.loads(l)
       for l in open(ROOT / "work_c_new/ghost_nusc/mining/events_all.jsonl")}

for d_ in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
    sys.path.insert(0, str(RES / d_))
if A.model == "dd":
    from dd_adapter import DDRunner as R_
elif A.model == "ltf":
    from ltf_adapter import LTFRunner as R_
else:
    from ddv2_adapter import DDV2Runner as R_
runner = R_(device=A.device)
lidar = None
if A.model == "ddv2":
    from ddv2_adapter import NuScenesLidar
    lidar = NuScenesLidar(NUSC)


def run(img, spd, tok):
    o = runner.run(img, spd) if lidar is None else runner.run(img, spd, lidar_xyz=lidar.ego_points(tok))
    return arc_full(np.asarray(o["trajectory"], float), WP_DT[A.model])


out = []
for c in sorted(cands, key=lambda z: -z["a_vru_max"]):
    s_, j = c["scene"], c["frame_idx"]
    ev = evs.get(s_)
    if ev is None:
        continue
    geo = G1.compute_scene_geometry(nusc, SC[s_], cfg)
    bbs = []
    for g in c["f3_mask_group"]:
        o = geo["per_obj"].get(g["token"])
        bb = G1.frame_bbox(geo, o, j) if o else None
        if bb:
            bbs.append(list(bb))
    if not bbs:
        continue
    fr = geo["frames"][j]
    img = np.asarray(Image.open(f"{NUSC}/{fr['filename']}").convert("RGB"))
    tok = fr.get("token")
    spd = float(c["ego_v0"])
    vo = run(img, spd, tok)
    res = {"scene": s_, "a_req": c["a_vru_max"], "v0": spd,
           "luminance": float(img.mean()), "v_origin": vo, "n_mask": len(bbs)}
    for mode in ("mean", "road_flat"):
        m2 = img.copy()
        for bb in bbs:
            m2 = occlude(m2, bb, mode=mode)
        res[f"v_clean_{mode}"] = run(m2, spd, tok)
        res[f"b_{mode}"] = res[f"v_clean_{mode}"] - vo
    out.append(res)
    print(f"  {s_}  b(灰斑)={res['b_mean']:+.4f}  b(路面色)={res['b_road_flat']:+.4f}", flush=True)

json.dump(out, open(RES / f"rhd_ghost_refill_{A.model}.json", "w"), ensure_ascii=False, indent=1)
b1 = np.array([r["b_mean"] for r in out]); b2 = np.array([r["b_road_flat"] for r in out])
print(f"\n[{A.model}] n={len(out)}  b 均值: 灰斑 {b1.mean():+.4f} -> 路面色 {b2.mean():+.4f}"
      f"   b>0: {(b1>0).mean():.0%} -> {(b2>0).mean():.0%}")
