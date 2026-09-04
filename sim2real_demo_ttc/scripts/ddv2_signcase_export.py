"""DDv2 正/负响应典型场景出图（lead/NAVSIM，功效最足的格子）。"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
OUT=RES/"figures"/"ddv2_signcases"; BLOBS="/data/dataset/navsim/dataset/sensor_blobs"
sys.path.insert(0,str(ROOT/"scripts"))
import g1_mine_events as G1
from f3_occlusion_necessity import occlude
from f3_brakefirst import control_boxes_for_group
K="arc_full"

def main():
    import pickle
    from omegaconf import OmegaConf
    import ns1_navsim_geometry as NS
    cfg=OmegaConf.to_container(OmegaConf.load(ROOT/"configs/navsim_corpus.yaml"),resolve=True)
    g={r['scene']:r for r in json.load(open(RES/"gt_lead_navsim_ddv2.json"))['per_event']}
    t={r['scene']:r for r in json.load(open(RES/"f3_lead_navsim_ddv2_traj.json"))['per_event']}
    pool={c['scene']:c for c in json.load(open(RES/"brake_first_pool_lead_navsim_final.json"))['candidates']}
    s=sorted([x for x in g if x in t], key=lambda x: g[x][f'b_model__{K}'])
    sel=[(x,'NEG') for x in s[:3]]+[(x,'POS') for x in s[-3:]]
    _fr={}
    for lf in sorted((NS.NS_ROOT/"navsim_logs"/"test").glob("*.pkl")):
        for f in pickle.load(open(lf,"rb")): _fr.setdefault(f["scene_name"],[]).append(f)
    OUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(0); meta=[]
    for name,tag in sel:
        c=pool[name]; geo=NS.build_geo(sorted(_fr[name],key=lambda z:z["timestamp"]),cfg,"test")
        j=c['frame_idx']
        img=cv2.cvtColor(cv2.imread(str(Path(BLOBS)/geo["frames"][j]["filename"])),cv2.COLOR_BGR2RGB)
        boxes=[]
        for gg in c['f3_mask_group']:
            o=geo['per_obj'].get(gg['token']); bb=G1.frame_bbox(geo,o,j) if o is not None else None
            if bb is not None: boxes.append(list(bb))
        cb=control_boxes_for_group(img,boxes,rng)
        clean=img.copy(); ctrl=img.copy()
        for b in boxes: clean=occlude(clean,b)
        for b in cb: ctrl=occlude(ctrl,b)
        left=img.copy()
        for b in boxes:
            x0,y0,x1,y1=[int(round(v)) for v in b]
            cv2.rectangle(left,(x0,y0),(x1,y1),(0,235,0) if tag=='POS' else (60,60,255),3)
        pair=np.hstack([left,clean,ctrl]); bar=np.zeros((190,pair.shape[1],3),np.uint8)
        gr,tr=g[name],t[name]
        lines=[f"{name}   [{tag}]  DDv2 b(arc_full) = {gr[f'b_model__{K}']:+.4f}   b_GT = {gr[f'b_gt__{K}']:+.4f}",
               "LEFT origin (mask group outlined)   MID clean (masked, incl. lidar points deleted)   RIGHT ctrl",
               f"human {c['ego_v0']:.2f} -> {c['ego_vmin']:.2f} m/s   a_obs {c['a_obs']}   a_req {c['a_vru_max']}   "
               f"lead {c['lead_vru']['cat'].split('.')[-1]} @{c['lead_vru']['s_m']}m",
               f"lidar points deleted: clean {tr['n_lidar_del_clean']}  ctrl {tr['n_lidar_del_ctrl']}   "
               f"mask group {tr['n_mask']}   err_origin {gr[f'err_origin__{K}']:+.2f}"]
        for i,x in enumerate(lines):
            cv2.putText(bar,x,(12,32+40*i),cv2.FONT_HERSHEY_SIMPLEX,0.82,
                        (255,255,255) if i==0 else (180,230,255),2)
        p=OUT/f"{tag}__{name}.jpg"
        cv2.imwrite(str(p),cv2.cvtColor(np.vstack([bar,pair]),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
        meta.append({"scene":name,"tag":tag,"file":p.name,"b_model":gr[f'b_model__{K}'],
                     "b_gt":gr[f'b_gt__{K}'],"n_lidar_del_clean":tr['n_lidar_del_clean'],
                     "n_lidar_del_ctrl":tr['n_lidar_del_ctrl'],"ego_v0":c['ego_v0'],
                     "a_req":c['a_vru_max'],"lead":c['lead_vru']})
        print(f"  [{tag}] {name:22s} b {gr[f'b_model__{K}']:+7.3f}  删点 {tr['n_lidar_del_clean']:5d}  画出 {len(boxes)}")
    (OUT/"meta.json").write_text(json.dumps({"corpus":"NAVSIM","scenario":"lead","readout":K,
        "note":"正负两端各 3 例；删点数中位 268、无零删点事件 ⇒ '没点可删'不成立",
        "events":meta},indent=2,ensure_ascii=False))
    print(f"\nwrote {OUT}")

if __name__=="__main__": main()
