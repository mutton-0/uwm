"""按「投影与 b 的一致性」给层加权 —— **必须交叉验证**，否则是按结果选层。

朴素做法（选与 b 最相关的层）会把噪声当信号：315 个事件、8–24 个层，
总有一层碰巧相关。故：权重只在训练折上定，相关只在留出折上报。

对照组：
  ① 三重门单层（现行做法）
  ② 过门层等权平均
  ③ 训练折上最优单层（同样只在留出折评估）
  ④ **零假设**：训练折上把 b 打乱再定权重，其余流程不变
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from f_vfaith_direction import load_by_side, part_ratio, split_half_angle   # noqa: E402
from i_ortho import side_of                                                 # noqa: E402
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo"}

def spear(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else 0.0

def proj_layers(D_by_L, tr):
    """每层：用训练折的均值方向，给全体事件算投影（z 标准化）。"""
    out=[]
    for D in D_by_L:
        v=D[tr].mean(0); n=np.linalg.norm(v)
        p=D@(v/n) if n>1e-12 else np.zeros(len(D))
        s=p[tr].std() or 1.0
        out.append((p-p[tr].mean())/s)
    return np.stack(out,1)          # [n_events, n_layers]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scen",default="lead"); ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--models",default="dd,ltf,ddv2,simlingo")
    A=ap.parse_args()
    print(f"{A.folds}-折交叉验证：权重只在训练折定，Spearman 只在留出折报\n")
    print(f"{'候选':<17}{'过门层':>8}{'①单层':>9}{'②等权':>9}{'③训练最优单层':>14}"
          f"{'④加权(主)':>11}{'零假设':>9}")
    print("-"*82)
    for m in A.models.split(","):
        z=np.load(RES/f"vfaith_acts_navsim_{A.scen}_{m}.npz",allow_pickle=True)
        b_=load_by_side(A.scen,m)["LHD"]
        nL=sum(1 for k in z.files if k.startswith("h_clean__L"))
        sc=np.array([str(x) for x in z["scene"]]); k=np.array([side_of(x)=="LHD" for x in sc])
        Ds=[(z[f"h_ghost__L{l}"]-z[f"h_clean__L{l}"])[k].astype(float) for l in range(nL)]
        bb=(z["v_clean"]-z["v_ghost"])[k].astype(float)
        rng=np.random.default_rng(0)
        gate=[l for l in range(nL) if part_ratio(Ds[l])>=3]
        if not gate: gate=list(range(nL))
        L0=gate[len(gate)//2]
        n=len(bb); idx=rng.permutation(n); folds=np.array_split(idx,A.folds)
        acc={k_:[] for k_ in ("single","equal","best","weighted","null")}
        for f in folds:
            te=np.zeros(n,bool); te[f]=True; tr=~te
            P=proj_layers(Ds,tr)
            acc["single"].append(spear(P[te,L0],bb[te]))
            acc["equal"].append(spear(P[te][:,gate].mean(1),bb[te]))
            rh=np.array([spear(P[tr,l],bb[tr]) for l in range(nL)])
            lb=int(np.argmax(np.abs(rh)))
            acc["best"].append(spear(P[te,lb]*np.sign(rh[lb]),bb[te]))
            w=np.where(np.abs(rh)>0, rh, 0.0)            # 带符号权重
            acc["weighted"].append(spear(P[te]@w,bb[te]))
            bp=bb[tr][rng.permutation(tr.sum())]
            rn=np.array([spear(P[tr,l],bp) for l in range(nL)])
            acc["null"].append(spear(P[te]@rn,bb[te]))
        g=lambda k_: float(np.mean(acc[k_]))
        print(f"{NAME[m]:<17}{len(gate):>8}{g('single'):>+9.3f}{g('equal'):>+9.3f}"
              f"{g('best'):>+14.3f}{g('weighted'):>+11.3f}{g('null'):>+9.3f}")
if __name__=="__main__":
    main()
