"""刹车优先候选池出图|左=遮挡组全部框，右=实际遮挡后图像。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
OUT=RES/"figures"/"brake_first"; NUSC="/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0,str(ROOT/"scripts"))
import g1_mine_events as G1
from f3_occlusion_necessity import occlude

NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    corpus = sys.argv[2] if len(sys.argv) > 2 else "nuscenes"
    from omegaconf import OmegaConf
    global OUT
    if corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg=OmegaConf.to_container(OmegaConf.load(ROOT/"configs/n1_d2.yaml"),resolve=True)
        nusc=NuScenes("v1.0-trainval",dataroot=NUSC,verbose=False)
        sc={s['name']:s for s in nusc.scene}
        pool=json.load(open(RES/"brake_first_pool.json"))["candidates"]
        root=NUSC
        def build(c): return G1.compute_scene_geometry(nusc, sc[c["scene"]], cfg)
    else:
        import pickle
        from collections import defaultdict as _dd
        import ns1_navsim_geometry as NS
        OUT = RES/"figures"/"brake_first_navsim"
        cfg=OmegaConf.to_container(OmegaConf.load(ROOT/"configs/navsim_corpus.yaml"),resolve=True)
        pool=json.load(open(RES/"brake_first_pool_navsim.json"))["candidates"]
        root=NS_BLOBS
        _cache={}
        def build(c):
            if not _cache:
                for lf in sorted((NS.NS_ROOT/"navsim_logs"/"test").glob("*.pkl")):
                    for f in pickle.load(open(lf,"rb")):
                        _cache.setdefault(f["scene_name"], []).append(f)
            fl=sorted(_cache[c["scene"]], key=lambda z: z["timestamp"])
            return NS.build_geo(fl, cfg, "test")
    pool=sorted(pool,key=lambda z:-z["a_vru_max"])[:n]
    OUT.mkdir(parents=True,exist_ok=True); meta=[]
    for c in pool:
        geo=build(c)
        j=c["frame_idx"]; fr=geo["frames"][j]
        img=cv2.cvtColor(cv2.imread(str(Path(root)/fr["filename"])),cv2.COLOR_BGR2RGB)
        boxes=[]
        for g in c["f3_mask_group"]:
            ob=geo["per_obj"].get(g["token"])
            bb=G1.frame_bbox(geo,ob,j) if ob is not None else None
            if bb is not None: boxes.append((list(bb),g))
        occ=img.copy()
        for bb,_ in boxes: occ=occlude(occ,bb)
        left=img.copy()
        for bb,g in boxes:
            x0,y0,x1,y1=[int(round(v)) for v in bb]
            cv2.rectangle(left,(x0,y0),(x1,y1),(0,235,0),3)
            cv2.putText(left,f"{g['s_m']}m a={g['a_req']}",(x0,max(18,y0-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,235,0),2)
        pair=np.hstack([left,occ]); bar=np.zeros((150,pair.shape[1],3),np.uint8)
        for i,t in enumerate([
          f"{c['scene']} frame {j}   BRAKE-FIRST candidate   mask group = {c['n_mask_group']}",
          f"human: v0 {c['ego_v0']} -> {c['ego_vmin']} m/s  dv {c['dv']}  a_obs {c['a_obs']} m/s2"
          f"  over {c['brake_dur_s']}s",
          f"VRU a_req {c['a_vru_max']} vs non-VRU {c['a_non_vru_max']}"
          f"   class share {c['vru_class_share']}   explains {c['explained_ratio']}",
          f"lead VRU {c['lead_vru']['cat']} at {c['lead_vru']['s_m']}m"
          f" lat {c['lead_vru']['lat_m']}m  TTC {c['ttc_s']}s   drawn {len(boxes)}/{c['n_mask_group']}"]):
            cv2.putText(bar,t,(12,30+34*i),cv2.FONT_HERSHEY_SIMPLEX,0.78,
                        (255,255,255) if i==0 else (180,230,255),2)
        p=OUT/f"{c['scene']}_f{j}.jpg"
        cv2.imwrite(str(p),cv2.cvtColor(np.vstack([bar,pair]),cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY,84])
        meta.append({**{k:c[k] for k in ("scene","frame_idx","ego_v0","dv","a_obs",
                    "a_vru_max","a_non_vru_max","vru_class_share","ttc_s","n_mask_group")},
                    "file":p.name,"n_boxes_drawn":len(boxes)})
        print(f"  {c['scene']:14s} f{j:4d}  a_vru {c['a_vru_max']:5.2f}  画出 {len(boxes)}/{c['n_mask_group']}")
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(f"\nwrote {OUT}")

if __name__=="__main__": main()
