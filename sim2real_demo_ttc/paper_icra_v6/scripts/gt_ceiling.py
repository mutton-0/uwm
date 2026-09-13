"""危险敏感度的**真值参照**：把"看得见行人"那一侧换成真人当时实际开出来的轨迹。
对每个需要反应的单位：人类轨迹 H 与同一单位的盲规划 Q 各自对行人真实未来算四项读数，
HS_gt = ¼[ΔA+ΔC+ΔT+tanh(ΔS)]，Δ=(·)(H)−(·)(Q)。按 a_req 分箱取均值。
读数定义与 diag_units 完全一致：c_t=‖X_t−F_t‖−1.0−0.4，0–2.5 s 按 0.1 s 插值。
另报"构造上限"（把 H 换成完美规划 A=C=T=1、S=10）作对照。输出 gt_ceiling.json"""
import json,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"; R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
HP=json.load(open(f"{V5}/human_path.json")); rows=json.load(open(f"{V5}/diag_units.json"))
TT=np.round(np.arange(0,2.51,0.1),2)
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def readouts(X,F):
    c=np.linalg.norm(X-F,axis=1)-1.0-0.4
    A=float(c.min()>0); C=float(np.clip(c.min()/1.0,0,1))
    i=np.where(c<1.0)[0]; T=float(TT[i[0]]/2.5) if len(i) else 1.0
    S=float(np.mean(np.minimum(c,10.0)))
    return A,C,T,S
sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0 and z["need"]
BINS=[(0,0.5),(0.5,1),(1,2),(2,4),(4,99)]; XL=["<0.5","0.5-1","1-2","2-4",">4"]
gt={b:[] for b in range(len(BINS))}; ideal={b:[] for b in range(len(BINS))}
for z in rows:
    if not sel(z): continue
    u=z["uid"]; x=IDX.get(u); h=HP.get(u)
    if x is None or h is None or len(h)<5 or not x.get("ped_future_ego"): continue
    p0=np.mean(np.asarray(x["corners_ego"])[:,:2],0)
    F=lin(np.r_[0,np.arange(1,6)*0.5],np.vstack([p0,np.asarray(x["ped_future_ego"])[:5,:2]]))
    H=lin(np.r_[0,np.arange(1,6)*0.5],np.vstack([[0,0],np.asarray(h)[:5]]))
    A,C,T,S=readouts(H,F); q=z["Q"]
    v=0.25*((A-q["A"])+(C-q["C"])+(T-q["T"])+np.tanh(S-q["S"]))
    w=0.25*((1-q["A"])+(1-q["C"])+(1-q["T"])+np.tanh(10-q["S"]))
    for b,(a_,b_) in enumerate(BINS):
        if a_<=z["a_req"]<b_: gt[b].append(v); ideal[b].append(w)
res={"bins":XL,"gt":[],"ideal":[],"n":[]}
print(f"{'危险等级':10s} {'n':>5s} {'真人参照':>9s} {'构造上限':>9s}")
for b,lab in enumerate(XL):
    n=len(gt[b]); g=float(np.mean(gt[b])) if n>=8 else float("nan"); i_=float(np.mean(ideal[b])) if n>=8 else float("nan")
    res["gt"].append(g); res["ideal"].append(i_); res["n"].append(n)
    print(f"  {lab:8s} {n:5d} {g:9.2f} {i_:9.2f}")
allg=[v for b in gt for v in gt[b]]
print(f"\n全体：真人参照 {np.mean(allg):+.2f}（n={len(allg)}）")
# 再按 TTC0 分箱算一遍（Fig. 2 用）
TB=[(2,99),(1.5,2),(1,1.5),(0,1)]; TL=[">2","1.5-2","1-1.5","<1"]
g2={b:[] for b in range(len(TB))}
for z in rows:
    if not sel(z): continue
    u=z["uid"]; x=IDX.get(u); h=HP.get(u)
    if x is None or h is None or len(h)<5 or not x.get("ped_future_ego"): continue
    p0=np.mean(np.asarray(x["corners_ego"])[:,:2],0)
    F=lin(np.r_[0,np.arange(1,6)*0.5],np.vstack([p0,np.asarray(x["ped_future_ego"])[:5,:2]]))
    H=lin(np.r_[0,np.arange(1,6)*0.5],np.vstack([[0,0],np.asarray(h)[:5]]))
    A,C,T,S=readouts(H,F); q=z["Q"]
    v=0.25*((A-q["A"])+(C-q["C"])+(T-q["T"])+np.tanh(S-q["S"]))
    for b,(a_,b_) in enumerate(TB):
        if a_<=z["ttc0"]<b_: g2[b].append(v)
res["ttc_bins"]=TL; res["gt_ttc"]=[float(np.mean(g2[b])) if len(g2[b])>=8 else float("nan") for b in range(len(TB))]
res["n_ttc"]=[len(g2[b]) for b in range(len(TB))]
print("\n按 TTC0 分箱的真人参照:")
for lab,v,n in zip(TL,res["gt_ttc"],res["n_ttc"]): print(f"  {lab:8s} n={n:4d}  {v:.2f}")
json.dump(res,open(f"{V5}/gt_ceiling.json","w"),indent=1)
