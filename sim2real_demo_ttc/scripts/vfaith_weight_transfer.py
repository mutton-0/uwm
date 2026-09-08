"""v_faith 的层权重能否从 benchmark(左舵) 迁移到 deployment(右舵)。

若成立 ⇒ 权重在有 GT 的 bench 上学一次，deploy 上只需跑前向、不需标注，
这正是「用少量 deploy 场景测量」所需的形态。

严格切分：方向与权重**全部**只用左舵事件估；右舵只作留出评估，绝不参与拟合。
对照：① 三重门单层  ② 过门层等权  ③ 左舵训练的加权  ④ 零假设（左舵打乱 b 再定权重）
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from f_vfaith_direction import part_ratio                              # noqa: E402
from i_ortho import side_of                                            # noqa: E402
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo"}
MAP=json.load(open(RES/"driveside_map.json"))

def spear(x,y):
    if len(x)<4: return float("nan")
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else 0.0

def side_any(s,corp):
    return MAP["nuscenes_side"].get(s) if corp=="nuscenes" else side_of(s)

def load(scen,m):
    """两语料合并后按舵位切；返回 逐层 δ、b、舵位。"""
    parts=[]
    for corp in ("navsim","nuscenes"):
        p=RES/f"vfaith_acts_{corp}_{scen}_{m}.npz"
        if not p.exists(): continue
        z=np.load(p,allow_pickle=True)
        nL=sum(1 for k in z.files if k.startswith("h_clean__L"))
        sc=[str(x) for x in z["scene"]]
        sd=np.array([side_any(s,corp) or "?" for s in sc])
        parts.append({"z":z,"nL":nL,"sd":sd})
    if not parts: return None
    nL=min(x["nL"] for x in parts)
    D=[np.concatenate([x["z"][f"h_ghost__L{l}"]-x["z"][f"h_clean__L{l}"] for x in parts]).astype(float)
       for l in range(nL)]
    b=np.concatenate([(x["z"]["v_clean"]-x["z"]["v_ghost"]) for x in parts]).astype(float)
    sd=np.concatenate([x["sd"] for x in parts])
    return D,b,sd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--models",default="dd,ltf,ddv2,simlingo")
    A=ap.parse_args()
    print("方向与权重**只用左舵**估；右舵留出评估。Spearman(投影, b)\n")
    print(f"{'场景':<7}{'候选':<17}{'nL':>4}{'n左':>5}{'n右':>5}"
          f"{'左舵CV加权':>11}{'右舵:单层':>10}{'右舵:等权':>10}{'右舵:加权':>10}{'右舵:零假设':>11}")
    print("-"*94)
    out=[]
    for scen in ("lead","ghost"):
        for m in A.models.split(","):
            r=load(scen,m)
            if r is None: continue
            D,b,sd=r; nL=len(D)
            L_=sd=="LHD"; R_=sd=="RHD"
            if R_.sum()<4 or L_.sum()<20: continue
            gate=[l for l in range(nL) if part_ratio(D[l][L_])>=3] or list(range(nL))
            L0=gate[len(gate)//2]
            rng=np.random.default_rng(0)
            def projs(tr):
                P=np.zeros((len(b),nL))
                for l in range(nL):
                    v=D[l][tr].mean(0); n=np.linalg.norm(v)
                    p=D[l]@(v/n) if n>1e-12 else np.zeros(len(b))
                    s=p[tr].std() or 1.0; P[:,l]=(p-p[tr].mean())/s
                return P
            # 左舵内部 5 折 CV
            li=np.where(L_)[0]; fo=np.array_split(rng.permutation(len(li)),5); cv=[]
            for f in fo:
                te=li[f]; tr=np.setdiff1d(li,te)
                P=projs(tr); rh=np.array([spear(P[tr,l],b[tr]) for l in range(nL)])
                cv.append(spear(P[te]@rh,b[te]))
            # 右舵留出：权重来自左舵全体
            P=projs(li); rh=np.array([spear(P[li,l],b[li]) for l in range(nL)])
            bp=b[li][rng.permutation(len(li))]
            rn=np.array([spear(P[li,l],bp) for l in range(nL)])
            ri=np.where(R_)[0]
            row=dict(scen=scen,model=m,nL=nL,n_l=int(L_.sum()),n_r=int(R_.sum()),
                     cv=float(np.mean(cv)),
                     rhd_single=spear(P[ri,L0],b[ri]),
                     rhd_equal=spear(P[ri][:,gate].mean(1),b[ri]),
                     rhd_w=spear(P[ri]@rh,b[ri]), rhd_null=spear(P[ri]@rn,b[ri]))
            out.append(row)
            print(f"{scen:<7}{NAME[m]:<17}{nL:>4}{row['n_l']:>5}{row['n_r']:>5}"
                  f"{row['cv']:>+11.3f}{row['rhd_single']:>+10.3f}{row['rhd_equal']:>+10.3f}"
                  f"{row['rhd_w']:>+10.3f}{row['rhd_null']:>+11.3f}")
    json.dump(out,open(RES/"vfaith_weight_transfer.json","w"),ensure_ascii=False,indent=1)
    print(f"\n-> {RES}/vfaith_weight_transfer.json")
if __name__=="__main__":
    main()
