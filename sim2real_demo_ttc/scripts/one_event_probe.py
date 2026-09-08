"""单事件三臂测量：给定场景+帧，量 v_origin / b(灰斑) / b(路面色) / 投影分位。"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(RES))
import g1_mine_events as G1
from f3_occlusion_necessity import occlude
from c_axis_hazard_patch import arc_full, WP_DT
NUSC="/data/dataset/nuscenes/v1.0-trainval"
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True,choices=["dd","ltf","ddv2"])
ap.add_argument("--scene",required=True); ap.add_argument("--pool",required=True)
ap.add_argument("--device",default="cuda:1")
A=ap.parse_args()
from nuscenes.nuscenes import NuScenes
cfg=OmegaConf.to_container(OmegaConf.load(ROOT/"configs/n1_d2.yaml"),resolve=True)
nusc=NuScenes(version="v1.0-trainval",dataroot=NUSC,verbose=False)
G1.set_include_animal(True); SC={s["name"]:s for s in nusc.scene}
p=json.load(open(RES/A.pool))
c=[x for x in (p.get("candidates") or [])+(p.get("deferred_low_a_req") or []) if x["scene"]==A.scene][0]
geo=G1.compute_scene_geometry(nusc,SC[A.scene],cfg); j=c["frame_idx"]
bbs=[]
for g in c["f3_mask_group"]:
    o=geo["per_obj"].get(g["token"]); bb=G1.frame_bbox(geo,o,j) if o else None
    if bb: bbs.append(list(bb))
fr=geo["frames"][j]
img=np.asarray(Image.open(f"{NUSC}/{fr['filename']}").convert("RGB"))
for d_ in ("diffusiondrive_g1_adapter","ltf_g1_adapter","ddv2_g1_adapter"): sys.path.insert(0,str(RES/d_))
if A.model=="dd": from dd_adapter import DDRunner as R_
elif A.model=="ltf": from ltf_adapter import LTFRunner as R_
else: from ddv2_adapter import DDV2Runner as R_
r=R_(device=A.device); lid=None
if A.model=="ddv2":
    from ddv2_adapter import NuScenesLidar; lid=NuScenesLidar(NUSC)
def go(a):
    o=r.run(a,float(c["ego_v0"])) if lid is None else r.run(a,float(c["ego_v0"]),lidar_xyz=lid.ego_points(fr.get("token")))
    return arc_full(np.asarray(o["trajectory"],float),WP_DT[A.model])
vo=go(img); out={"model":A.model,"scene":A.scene,"v_origin":vo,"v0":float(c["ego_v0"])}
for mode in ("mean","road_flat"):
    m2=img.copy()
    for bb in bbs: m2=occlude(m2,bb,mode=mode)
    out[f"b_{mode}"]=go(m2)-vo
print(f"{A.model:6s} v0={out['v0']:.2f}  v_origin={vo:.3f}  b(灰斑)={out['b_mean']:+.4f}  b(路面色)={out['b_road_flat']:+.4f}")
json.dump(out,open(RES/f"oneevent_{A.scene}_{A.model}.json","w"))
