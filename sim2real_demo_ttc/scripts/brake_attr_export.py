"""刹车归因结果出图|两组：单实例主导 vs VRU 群体主导。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
OUT=RES/"figures"/"brake_attribution"; NUSC="/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0,str(ROOT/"scripts"))
from lane_path_export import annotate

def main():
    d=json.load(open(RES/"brake_attribution.json"))
    solo=[o for o in d if o["vru_share"]>0.6 and o["explained_ratio"]>0.3]
    grp=[o for o in d if o["vru_class_share"]>0.6 and o["explained_ratio"]>0.3
         and o["vru_share"]<=0.6]
    ev={}
    ids={o["eid"] for o in solo+grp}
    for l in open(ROOT/"variants/n1_d2/mining/events_all.jsonl"):
        e=json.loads(l)
        if e["event_id"] in ids: ev[e["event_id"]]=e
    meta={"criterion":"刹车 case -> 归因 -> 行人占比",
          "a_req":"a_i = max(v_close,0)^2 / (2*max(s-2.0, 0.5))，s 沿真实未来路径的弧长位置",
          "groups":{}}
    for name,g,col in (("solo",solo,(0,235,0)),("group",grp,(0,180,255))):
        dd=OUT/name; dd.mkdir(parents=True,exist_ok=True)
        recs=[]
        for o in sorted(g,key=lambda z:z["d_long_ghost_m"]):
            e=ev[o["eid"]]; fg=e["x_ghost_frames"][0]; bb=fg["bbox_xyxy"]
            im=cv2.imread(str(Path(NUSC)/fg["filename"]))
            if im is None: continue
            rivals=", ".join(f"{c['cat'].split('.')[-1]}@{c['s_m']}m(a={c['a_req']})"
                             for c in o["top_competitors"][:3]) or "none"
            out=annotate(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),
                f"{o['eid']}  [{name}]  VRU share {o['vru_share']:.2f} (class {o['vru_class_share']:.2f})",[
                f"brake: v0 {o['ego_v0_mps']:.1f} -> {o['ego_vmin_mps']:.1f} m/s"
                f"  dv {o['ego_delta_v_mps']:+.2f}  a_obs {o['a_obs_mps2']:.2f} m/s2",
                f"VRU brake demand a_req = {o['a_req_vru']:.2f} m/s2"
                f"  -> explains {o['explained_ratio']*100:.0f}% of observed decel",
                f"VRU: {o['object_class']}  s={o['d_long_ghost_m']:.1f}m"
                f"  lat_to_path={o['lat_to_real_path_m']:.2f}m",
                f"rivals in corridor ({o['n_competitors']}): {rivals[:88]}",
            ],bb,col)
            p=dd/f"{o['eid']}.jpg"
            cv2.imwrite(str(p),cv2.cvtColor(out,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,86])
            recs.append({**{k:o[k] for k in ("eid","scene","pool","object_class","vru_share",
                "vru_class_share","explained_ratio","a_req_vru","a_obs_mps2","ego_v0_mps",
                "ego_delta_v_mps","d_long_ghost_m","lat_to_real_path_m","stopped_long_s",
                "n_competitors","top_competitors")},"file":p.name})
            print(f"[{name:5s}] {o['eid']:20s} share {o['vru_share']:.2f}/{o['vru_class_share']:.2f} "
                  f"d {o['d_long_ghost_m']:5.1f}m a_req {o['a_req_vru']:5.2f} a_obs {o['a_obs_mps2']:5.2f} "
                  f"expl {o['explained_ratio']*100:3.0f}%")
        meta["groups"][name]={"n":len(recs),"events":recs}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(f"\nwrote {OUT}")

if __name__=="__main__": main()
