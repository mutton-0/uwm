"""按舵位诊断。用法：diagnose_side.py LHD|RHD。只读对应舵位的单位；近距有干涉口径：
走廊行人（Set A corr 5–12 m）或接近序列（Set B），d ≤ 15 m。输出 profile_<side>.json（点估计 + bootstrap CI）。"""
import json,sys,numpy as np
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5/scripts")
from diag_profile import dims,boot
SIDE=sys.argv[1]
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
rows=json.load(open(f"{V5}/diag_units.json"))
# DROP_DETFAIL=1：剔掉"移除后独立检测器仍能找到目标"的帧（IoU≥0.5），用于检验残留是否影响结论
import os as _os
_BAD=set()
if _os.environ.get("DROP_DETFAIL"):
    _BAD={o["uid"] for o in json.load(open(f"{R5}/paper_icra_v4/det_validate.json"))
          if "target_box" in o and o["front_only"] and o["rm"]["target_iou"]>=0.5}
    print(f"剔除移除失败的帧 {len(_BAD)} 个")
sel=lambda z: (SIDE=="ALL" or z["side"]==SIDE) and z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["uid"] not in _BAD
out={}
for m in ["dd","ltf","ddv2","simlingo","autovla","alpamayo","alpamayo15"]:
    U=[z for z in rows if z["m"]==m and sel(z)]
    p=dims(U,IDX); ci=boot(U,IDX)
    out[m]={"point":{k:(None if (isinstance(v,float) and np.isnan(v)) else float(v)) for k,v in p.items()},"ci":ci,"n_frames":len({z["uid"] for z in U})}
    print(f"{m:9s} frames={out[m]['n_frames']:3d} units={p['n_units']:4d} need={p['n_need']:3d} act={p['n_act']:3d} | "+
          "  ".join(f"{k}={p[k]:+.3f}[{ci[k][0]:+.2f},{ci[k][1]:+.2f}]" for k in ["exposure","HS","HS_slope","SP","sep_gain","CFR","align","align_shuffled","night_sep"]))
_sfx="_nodetfail" if _os.environ.get("DROP_DETFAIL") else ""
json.dump(out,open(f"{V5}/profile_{SIDE}{_sfx}.json","w"),indent=1)
