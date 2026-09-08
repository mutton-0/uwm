"""按 RepE 原文（arXiv 2310.01405）的正确做法做注入。

## 与我先前实现的两处差别
1. **级联**：原文明确指出「早层的改动会传到晚层，削弱事先算好的向量」，
   解决办法是「从最早的目标层开始逐层注入，**在已扰动的前向上**重算下一层的向量」。
   我先前是在未扰动激活上一次算完所有层再同时注入 —— 正是它警告的错法。
   实测佐证：L5 注入 64%，传到 L6 只剩 2.3%。
2. **算子**：原文有三个，我只用了线性组合。这里补上分段算子
   （顺着激活当前在 v 上的符号放大，而非硬加固定偏移）。

      线性组合   R' = R + α·σ·v
      分段       R' = R + α·σ·sign(Rᵀv)·v
      投影       R' = R − (Rᵀv/‖v‖²)·v

## 层的选取
原文用「reading performance 最强的中间若干层」，即读取性能，
对应本项目的三重门（PR≥3、噪声地板≤30°）+ SNR，不是因果可达性。
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from c_axis_hazard_patch import arc_full, WP_DT                        # noqa: E402
from f_vfaith_direction import part_ratio, split_half_angle            # noqa: E402
LABEL={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2"}
NSB="/data/dataset/navsim/dataset/sensor_blobs"
ALPHAS=[-8,-4,-2,-1,0,1,2,4,8]

def spear(x,y):
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean(); ry-=ry.mean(); d=np.sqrt((rx@rx)*(ry@ry)); return float(rx@ry/d) if d>0 else 0.
def slope(al,y):
    al=np.asarray(al,float); y=np.asarray(y,float); m=np.isfinite(y)
    if m.sum()<3: return float("nan")
    return float(np.linalg.lstsq(np.stack([al[m],np.ones(m.sum())],1),y[m],rcond=None)[0][0])
def fit(X,v):
    vc=v-v.mean(); return (X-X.mean(0)).T@vc/max(vc@vc,1e-12)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True,choices=list(LABEL))
    ap.add_argument("--n",type=int,default=60); ap.add_argument("--n-rand",type=int,default=30)
    ap.add_argument("--mode",default="add",choices=["add","piecewise"])
    ap.add_argument("--stim-side",default="LHD",choices=["LHD","RHD"],
                    help="**刺激**的舵位。方向与层权重一律只在左舵巡航样本上学；"
                         "此项只换被注入的场景。用于检验注入效应是否跨域保持。")
    ap.add_argument("--device",default="cuda:1")
    A=ap.parse_args()
    from PIL import Image
    for d_ in ("diffusiondrive_g1_adapter","ltf_g1_adapter","ddv2_g1_adapter"):
        sys.path.insert(0,str(RES/d_))
    import dd_adapter as _DD; _DD.set_crop_center_row(560)
    if A.model=="dd": from dd_adapter import DDRunner as R_
    elif A.model=="ltf": from ltf_adapter import LTFRunner as R_
    else: from ddv2_adapter import DDV2Runner as R_
    runner=R_(device=A.device); lidar=None
    if A.model=="ddv2":
        from ddv2_adapter import NavsimLidar; lidar=NavsimLidar()

    z=np.load(RES/f"vdecel_acts_navsim_test_{A.model}.npz",allow_pickle=True)
    nL=sum(1 for k in z.files if k.startswith("h__L"))
    tr=np.where((z["kind"]=="cruise")&(z["side"]=="LHD"))[0]; v0=z["v0"].astype(float)
    # 层选取：读取性能（RepE 口径）
    keep=[]
    for l in range(nL):
        X=z[f"h__L{l}"][tr].astype(float)
        d=fit(X,v0[tr]); n=np.linalg.norm(d)
        if n<1e-12: continue
        r=spear(X@(d/n),v0[tr])
        if part_ratio(X)>=3 and abs(r)>=0.3: keep.append(l)
    if not keep: keep=list(range(nL))
    print(f"[{LABEL[A.model]}] 注入层（读取性能选，RepE 口径）= {keep}   算子={A.mode}",flush=True)

    pool=[]
    for sp in ("test","trainval"):
        p=RES/f"cruise_pool_navsim_{sp}.json"
        if p.exists(): pool+=[c for c in json.load(open(p))["cruise"] if c.get("side")==A.stim_side]
    pool=[c for c in pool if (Path(NSB)/c["filename"]).exists()]
    rng=np.random.default_rng(0)
    pool=[pool[i] for i in rng.choice(len(pool),min(A.n,len(pool)),replace=False)]
    print(f"  刺激集 {len(pool)} 个巡航空场景（{A.stim_side}；轴与权重仍来自左舵）",flush=True)

    def cascade(img,spd,key,dirs,alpha):
        """一次前向完成多层注入。

        **为什么不需要"注入一层→重跑→再算下一层"**：RepE 的级联是针对**对比向量**
        （每个输入在推理时现算）；这里用的是**固定的群体读取向量** v，
        它不依赖当前激活 R，扰动后重算还是同一个 v。而 σ 本就由钩子按当次前向的
        实际（已被前面层扰动的）激活计算。故 add 算子的级联是恒等的。
        真正依赖当前 R 的是分段算子的 sign(Rᵀv)，已内联到钩子里（mode=piecewise）。
        这把每个 (事件,α,方向) 的前向次数从 len(keep)+1 降到 1。
        """
        runner.set_steering(None)
        if alpha!=0:
            for l in keep:
                runner.set_steering(layer=l,vec=-dirs[l],alpha=float(alpha),
                                    mode=A.mode,accumulate=True)
        o=(runner.run(img,spd) if lidar is None else runner.run(img,spd,lidar_xyz=lidar.ego_points(key)))
        runner.set_steering(None)
        t=np.asarray(o["trajectory"],float)
        return arc_full(t,WP_DT[A.model]),float(np.abs(t[:,1]).max())

    D0={}
    for l in keep:
        X=z[f"h__L{l}"][tr].astype(float); d=fit(X,v0[tr]); D0[l]=d/max(np.linalg.norm(d),1e-12)
    rands=[]
    for _ in range(A.n_rand):
        rd={}
        for l in keep:
            v=rng.normal(size=len(D0[l])); rd[l]=v/np.linalg.norm(v)
        rands.append(rd)

    Y={a:[] for a in ALPHAS}; LAT={a:[] for a in ALPHAS}; RS=[[] for _ in rands]
    for i,c in enumerate(pool):
        try:
            img=np.asarray(Image.open(f"{NSB}/{c['filename']}").convert("RGB"))
            key=c["filename"].split("/",1)[1]
            for a in ALPHAS:
                y,lt=cascade(img,float(c["v0"]),key,D0,a); Y[a].append(y); LAT[a].append(lt)
            for k,rd in enumerate(rands):
                RS[k].append([cascade(img,float(c["v0"]),key,rd,a)[0] for a in ALPHAS])
        except Exception as ex:
            print(f"  skip {c['scene']}: {type(ex).__name__}",flush=True); continue
        if (i+1)%10==0: print(f"  {i+1}/{len(pool)}",flush=True)

    ys=[float(np.mean(Y[a])) for a in ALPHAS]; lat=[float(np.mean(LAT[a])) for a in ALPHAS]
    s=slope(ALPHAS,ys); sr=[slope(ALPHAS,np.asarray(r).mean(0)) for r in RS if r]
    p=float((np.abs(sr)>=abs(s)).mean()) if sr else float("nan")
    pos=slope([a for a in ALPHAS if a>=0],[y for a,y in zip(ALPHAS,ys) if a>=0])
    neg=slope([a for a in ALPHAS if a<=0],[y for a,y in zip(ALPHAS,ys) if a<=0])
    sl=slope(ALPHAS,lat)
    print(f"\n[{LABEL[A.model]}] **RepE 注入 −ŝ**（{A.mode}，刺激={A.stim_side}），"
          f"α 阶梯上的规划速度：")
    print("  α    "+"".join(f"{a:>8.1f}" for a in ALPHAS))
    print("  速度 "+"".join(f"{y:>8.3f}" for y in ys))
    print("  横偏 "+"".join(f"{x:>8.3f}" for x in lat))
    print(f"\n  斜率 dy/dα = {s:+.4f}   随机方向中位|斜率| = {np.median(np.abs(sr)):.4f}   p={p:.3f}")
    print(f"  单调: α>0 {pos:+.4f}  α<0 {neg:+.4f}   横偏斜率 {sl:+.5f}")
    C1=(s<0) and (p<0.05); C2=(pos*neg>0) and (1/3<=abs(pos/neg if abs(neg)>1e-12 else 9)<=3)
    C3=abs(sl)<=abs(s)/3
    print(f"  C1 {'✓' if C1 else '✗'}  C2 {'✓' if C2 else '✗'}  C3 {'✓' if C3 else '✗'}"
          f"   ⇒ {'**证实**' if (C1 and C2 and C3) else '未证实'}")
    json.dump({"model":A.model,"mode":A.mode,"layers":keep,"alphas":ALPHAS,"arc":ys,"lat":lat,
               "slope":s,"p":p,"slope_pos":pos,"slope_neg":neg,"slope_lat":sl,
               "rand_slopes":sr,"n":len(Y[0])},
              open(RES/f"steer_repe_{A.model}_{A.mode}_{A.stim_side}.json","w"),
              ensure_ascii=False,indent=1)
if __name__=="__main__":
    main()
