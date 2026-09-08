"""在**危险事件池**上重算加权速度轴（对照：巡航池上的版本）。

问：危险场景里，自车速度的编码方式和匀速巡航时是否一样？
做法与 speed_axis_weighted.py 逐条一致 —— 层权重由「该层投影与真实 v0 的一致性」
在训练折上定，留出折评估；左右舵严格分开（左舵学，右舵只留出）。

激活取 h_ghost（原图臂，危险物在），与 v0 配对。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from i_ortho import side_of                                            # noqa: E402
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo"}
MAP=json.load(open(RES/"driveside_map.json"))
POOLS={("lead","navsim"):["deploy_pool_lead_vp.json","brake_first_pool_lead_navsim_fx_final.json"],
       ("lead","nuscenes"):["brake_first_pool_lead_final.json"],
       ("ghost","navsim"):["deploy_pool_ghost_vp.json","brake_first_pool_navsim_fx_final.json"],
       ("ghost","nuscenes"):["brake_first_pool_final.json"]}

def spear(x,y):
    if len(x)<4: return float("nan")
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else 0.0

def v0map(scen,corp):
    mp={}
    for f in POOLS.get((scen,corp),[]):
        p=RES/f
        if not p.exists(): continue
        d=json.load(open(p))
        for c in (d.get("candidates") or d.get("events") or [])+(d.get("deferred_low_a_req") or []):
            if c.get("ego_v0") is not None: mp[c["scene"]]=float(c["ego_v0"])
    return mp

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default="dd,ltf,ddv2,simlingo")
    A=ap.parse_args()
    print("危险事件池上的加权速度轴（目标=v0）。左舵 5 折 CV + 右舵留出\n")
    print(f"{'场景':<7}{'候选':<17}{'n左':>5}{'n右':>5}{'左舵CV:单层':>12}{'加权':>8}{'零假设':>9}"
          f"   {'右舵:单层':>10}{'加权':>8}")
    print("-"*88)
    for scen in ("lead","ghost"):
        for m in A.models.split(","):
            H=[];V=[];SD=[]
            nL=None
            for corp in ("navsim","nuscenes"):
                p=RES/f"vfaith_acts_{corp}_{scen}_{m}.npz"
                if not p.exists(): continue
                z=np.load(p,allow_pickle=True)
                nl=sum(1 for k in z.files if k.startswith("h_ghost__L"))
                nL=nl if nL is None else min(nL,nl)
                vm=v0map(scen,corp); sc=[str(x) for x in z["scene"]]
                keep=[i for i,s in enumerate(sc) if s in vm]
                if not keep: continue
                H.append(([z[f"h_ghost__L{l}"][keep].astype(float) for l in range(nl)]))
                V.append(np.array([vm[sc[i]] for i in keep]))
                SD.append(np.array([(MAP["nuscenes_side"].get(sc[i]) if corp=="nuscenes"
                                     else side_of(sc[i])) or "?" for i in keep]))
            if not H: continue
            D=[np.concatenate([h[l] for h in H]) for l in range(nL)]
            v0=np.concatenate(V); sd=np.concatenate(SD)
            L_=np.where(sd=="LHD")[0]; R_=np.where(sd=="RHD")[0]
            if len(L_)<20: continue
            rng=np.random.default_rng(0)
            def projs(tr):
                P=np.zeros((len(v0),nL))
                for l in range(nL):
                    X=D[l]; vc=v0[tr]-v0[tr].mean()
                    d=(X[tr]-X[tr].mean(0)).T@vc/max(vc@vc,1e-12)
                    n=np.linalg.norm(d); p=X@(d/n) if n>1e-12 else np.zeros(len(v0))
                    s=p[tr].std() or 1.0; P[:,l]=(p-p[tr].mean())/s
                return P
            fo=np.array_split(rng.permutation(len(L_)),5); cs=[];cw=[];cn=[]
            for f in fo:
                te=L_[f]; tr=np.setdiff1d(L_,te)
                P=projs(tr); rh=np.array([spear(P[tr,l],v0[tr]) for l in range(nL)])
                lb=int(np.argmax(np.abs(rh)))
                cs.append(spear(P[te,lb]*np.sign(rh[lb]),v0[te])); cw.append(spear(P[te]@rh,v0[te]))
                vp=v0[tr][rng.permutation(len(tr))]
                cn.append(spear(P[te]@np.array([spear(P[tr,l],vp) for l in range(nL)]),v0[te]))
            P=projs(L_); rh=np.array([spear(P[L_,l],v0[L_]) for l in range(nL)])
            lb=int(np.argmax(np.abs(rh)))
            rs=spear(P[R_,lb]*np.sign(rh[lb]),v0[R_]) if len(R_)>=4 else float("nan")
            rw=spear(P[R_]@rh,v0[R_]) if len(R_)>=4 else float("nan")
            print(f"{scen:<7}{NAME[m]:<17}{len(L_):>5}{len(R_):>5}{np.mean(cs):>+12.3f}"
                  f"{np.mean(cw):>+8.3f}{np.mean(cn):>+9.3f}   {rs:>+10.3f}{rw:>+8.3f}")
if __name__=="__main__":
    main()
