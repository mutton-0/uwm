"""多层加权速度轴：层权重由「该层投影与真实 v0 的一致性」定，**交叉验证**。

动机（用户 2026-09-07）：三重门是纯表征侧判据（SNR / 有效维度 / 噪声地板），
它选出的层估得最准，但未必携带目标信息。对 v_faith 做同样的事时，
DD 从 −0.036 → +0.154、LTF 从 −0.111 → +0.179（5 折 CV，零假设 +0.076 / −0.028）。

这里把同一手法用到速度轴上。目标变量是 v0（自车速度），本身就是标注，
不存在"用要预测的量去选层"那种循环 —— 但仍然全程交叉验证。

**左右舵严格分开**（P-1）：权重只在左舵(benchmark)巡航样本上学，
右舵(deployment)只作留出评估，绝不参与拟合。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2"}

def spear(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry))
    return float(rx@ry/d) if d>0 else 0.0

def fit_dir(X,v):
    vc=v-v.mean(); return (X-X.mean(0)).T@vc/max(vc@vc,1e-12)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folds",type=int,default=5)
    ap.add_argument("--models",default="dd,ltf,ddv2")
    A=ap.parse_args()
    print(f"{A.folds}-折 CV（左舵内部）+ 右舵留出。目标 = 自车速度 v0\n")
    print(f"{'候选':<17}{'层数':>5}{'①单层(最优)':>12}{'②等权':>9}{'③加权':>9}{'零假设':>9}"
          f"   {'右舵留出:单层':>13}{'加权':>8}")
    print("-"*92)
    store={}
    for m in A.models.split(","):
        z=np.load(RES/f"vdecel_acts_navsim_test_{m}.npz",allow_pickle=True)
        nL=sum(1 for k in z.files if k.startswith("h__L"))
        cru=z["kind"]=="cruise"
        L_=z["side"]=="LHD"; R_=z["side"]=="RHD"
        trL=np.where(cru&L_)[0]; teR=np.where(cru&R_)[0]
        v0=z["v0"].astype(float)
        H=[z[f"h__L{l}"].astype(float) for l in range(nL)]
        rng=np.random.default_rng(0)
        idx=rng.permutation(len(trL)); folds=np.array_split(idx,A.folds)
        acc={k:[] for k in ("single","equal","weighted","null")}
        for f in folds:
            te=trL[f]; tr=np.setdiff1d(trL,te)
            P=np.stack([ (lambda d: H[l]@(d/max(np.linalg.norm(d),1e-12)))(fit_dir(H[l][tr],v0[tr]))
                         for l in range(nL)],1)
            for l in range(nL):
                s=P[tr,l].std() or 1.0; P[:,l]=(P[:,l]-P[tr,l].mean())/s
            rh=np.array([spear(P[tr,l],v0[tr]) for l in range(nL)])
            lb=int(np.argmax(np.abs(rh)))
            acc["single"].append(spear(P[te,lb]*np.sign(rh[lb]),v0[te]))
            acc["equal"].append(spear(P[te].mean(1),v0[te]))
            acc["weighted"].append(spear(P[te]@rh,v0[te]))
            vp=v0[tr][rng.permutation(len(tr))]
            rn=np.array([spear(P[tr,l],vp) for l in range(nL)])
            acc["null"].append(spear(P[te]@rn,v0[te]))
        # 右舵留出：权重与方向全部来自左舵全体
        P=np.stack([ (lambda d: H[l]@(d/max(np.linalg.norm(d),1e-12)))(fit_dir(H[l][trL],v0[trL]))
                     for l in range(nL)],1)
        for l in range(nL):
            s=P[trL,l].std() or 1.0; P[:,l]=(P[:,l]-P[trL,l].mean())/s
        rh=np.array([spear(P[trL,l],v0[trL]) for l in range(nL)])
        lb=int(np.argmax(np.abs(rh)))
        rs=spear(P[teR,lb]*np.sign(rh[lb]),v0[teR]); rw=spear(P[teR]@rh,v0[teR])
        g=lambda k: float(np.mean(acc[k]))
        print(f"{NAME[m]:<17}{nL:>5}{g('single'):>+12.3f}{g('equal'):>+9.3f}{g('weighted'):>+9.3f}"
              f"{g('null'):>+9.3f}   {rs:>+13.3f}{rw:>+8.3f}")
        store[m]={"weights":rh.tolist(),"best_layer":lb,"n_lhd":int(len(trL)),"n_rhd":int(len(teR)),
                  "cv_single":g("single"),"cv_equal":g("equal"),"cv_weighted":g("weighted"),
                  "cv_null":g("null"),"rhd_single":rs,"rhd_weighted":rw}
    json.dump(store,open(RES/"speed_axis_weighted.json","w"),ensure_ascii=False,indent=1)
    print(f"\n-> {RES}/speed_axis_weighted.json")
if __name__=="__main__":
    main()
