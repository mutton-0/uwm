"""方法论可迁移性检验：用新域（右舵）的**少量案例**重新量轴，能否推出该域大样本上的行为？比沿用旧域（左舵）的数更准吗？
大样本真值：右舵 254 个近行人场景，每个模型三条规划（原图 / 移除行人 / 夜化）。
  轴：F = mean_t‖clean−rm‖，I = mean_t‖clean−night‖（0–2.5 s）；CFR = mean F / mean I
  行为：对行人真实未来的最小净间隙、最小 TTC；P-F（F<0.5 ⇒ 移除后结果不变）、P-I（CFR<1 ⇒ 夜化影响 ≥ 移除影响）
估计器 A：随机抽 k 个右舵场景重新量（k=5,10,20,40,80），重复 200 次
估计器 B：直接沿用左舵体检的数（profile_LHD.json 的 CFR）
比较：(1) 对大样本 CFR 的绝对误差；(2) 六模型排名与大样本排名的 Spearman；(3) 两条通式的命中率能否用小样本估出。"""
import json,os,numpy as np
from scipy.stats import spearmanr
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
AX={m:json.load(open(f"{V5}/rhd_axes_{m}.json")) for m in M if os.path.exists(f"{V5}/rhd_axes_{m}.json")}
PF_=json.load(open(f"{V5}/nv_ped_future.json")); PL=json.load(open(f"{V5}/profile_LHD.json"))
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def outc(w,F):
    X=lin(T8,np.vstack([[0,0],np.asarray(w)[:,:2]])); c=np.linalg.norm(X-F,axis=1)-1.4
    if c.min()<=0: return float(c.min()),0.0
    cl=-np.gradient(c,TT); ok=cl>0.05
    return float(c.min()),(float(min(5,np.min(c[ok]/cl[ok]))) if ok.any() else 5.0)
def D(a,b):
    A=lin(T8,np.vstack([[0,0],np.asarray(a)[:,:2]])); B=lin(T8,np.vstack([[0,0],np.asarray(b)[:,:2]]))
    return float(np.mean(np.linalg.norm(A-B,axis=1)))
U={}
for m,d in AX.items():
    rows=[]
    for t,a in d.items():
        p=PF_.get(t)
        if p is None or "rm" not in a or any(z is None for z in p["fut"][:5]): continue
        F=lin(T6,np.vstack([p["p0"],np.asarray(p["fut"][:5])]))
        c0,t0=outc(a["clean"],F); c1,t1=outc(a["rm"],F); c2,t2=outc(a["night"],F)
        rows.append(dict(tok=t,v0=p["v0"],F=D(a["clean"],a["rm"]),I=D(a["clean"],a["night"]),clr=c0,clr_rm=c1,clr_night=c2,ttc=t0,ttc_rm=t1,ttc_night=t2))
    U[m]=rows
def agg(rows):
    F=np.array([r["F"] for r in rows]); I=np.array([r["I"] for r in rows])
    mv=[r for r in rows if r["v0"]>=1.0]
    # 两条通式改为双向判定：前提不成立时预测相反的结果，于是每个场景都有预测
    def _chg(r): return abs(r["clr_rm"]-r["clr"])>=0.5 or ((r["clr"]<0)!=(r["clr_rm"]<0))
    pf =[not _chg(r) for r in rows if r["F"]<0.5]      # 正向：F≈0 ⇒ 结果不变
    pfr=[_chg(r)     for r in rows if r["F"]>=0.5]     # 反向：F 大 ⇒ 结果会变
    cfr=F.mean()/I.mean() if I.mean()>0 else np.nan
    rc=[r for r in rows if r["I"]>1e-6]
    pi =[abs(r["clr_night"]-r["clr"])>=abs(r["clr_rm"]-r["clr"]) for r in rc if r["F"]/r["I"]<1]
    pir=[abs(r["clr_rm"]-r["clr"])>=abs(r["clr_night"]-r["clr"]) for r in rc if r["F"]/r["I"]>=1]
    return dict(n=len(rows),CFR=float(cfr),F_med=float(np.median(F)),I_med=float(np.median(I)),
                ttc_viol=float(100*np.mean([r["ttc"]<1.5 for r in mv])) if mv else np.nan,
                PF=float(100*np.mean(pf)) if pf else np.nan, PI=float(100*np.mean(pi)) if pi else np.nan,
                PFrev=float(100*np.mean(pfr)) if pfr else np.nan, PIrev=float(100*np.mean(pir)) if pir else np.nan,
                nPF=len(pf), nPFrev=len(pfr), nPI=len(pi), nPIrev=len(pir),
                PFall=float(100*np.mean(pf+pfr)) if (pf+pfr) else np.nan,
                PIall=float(100*np.mean(pi+pir)) if (pi+pir) else np.nan)
G={m:agg(r) for m,r in U.items()}
print("每家在自己全部可用场景上的真值（dd/ltf/ddv2/SimLingo 254 个，AutoVLA/Alpamayo1.5 80 个）:")
for m in M:
    if m in G: g=G[m]; print(f"  {m:11s} n={g['n']:3d} CFR={g['CFR']:.2f} F中位={g['F_med']:.2f} I中位={g['I_med']:.2f} TTC违规={g['ttc_viol']:.1f}%  P-F {g['PF']:.0f}%/{g['PFrev']:.0f}% (n={g['nPF']}/{g['nPFrev']})  P-I {g['PI']:.0f}%/{g['PIrev']:.0f}% (n={g['nPI']}/{g['nPIrev']})")
if len(G)<6: print("（还有模型在跑，未齐）"); raise SystemExit

# ---- 口径统一：六家共有 token 上重算真值，供排名/抽样实验使用（避免 n 不同带来的假差异）
COM=sorted(set.intersection(*[{r["tok"] for r in U[m]} for m in M]))
GC={m:agg([r for r in U[m] if r["tok"] in COM]) for m in M}
print(f"\n六家共有 {len(COM)} 个场景上的真值（口径统一）:")
for m in M:
    g=GC[m]; print(f"  {m:11s} CFR={g['CFR']:.2f} F中位={g['F_med']:.2f} I中位={g['I_med']:.2f} TTC违规={g['ttc_viol']:.1f}%  P-F {g['PF']:.0f}%/{g['PFrev']:.0f}% (n={g['nPF']}/{g['nPFrev']})  P-I {g['PI']:.0f}%/{g['PIrev']:.0f}% (n={g['nPI']}/{g['nPIrev']})")

def sweep(models,pool,truthG,KS,tag,rng):
    truth=np.array([truthG[m]["CFR"] for m in models]); tPF=np.array([truthG[m]["PF"] for m in models])
    out={}
    print(f"\n{tag}（真值 n={truthG[models[0]]['n']}，{len(models)} 家；每个 k 重抽 200 次）:")
    for k in KS:
        err=[];rho=[];pfe=[];ok=[]
        for _ in range(200):
            s=set(rng.choice(pool,k,replace=False))
            est=np.array([agg([r for r in U[m] if r["tok"] in s])["CFR"] for m in models])
            pf=np.array([agg([r for r in U[m] if r["tok"] in s])["PF"] for m in models])
            err.append(np.nanmean(np.abs(est-truth))); rr=spearmanr(est,truth)[0]
            rho.append(rr); ok.append(rr>=0.6); pfe.append(np.nanmean(np.abs(pf-tPF)))
        out[k]=dict(err=float(np.nanmean(err)),rho=float(np.nanmean(rho)),rho_hit=float(np.mean(ok)),pf_err=float(np.nanmean(pfe)))
        print(f"  k={k:3d}  CFR 绝对误差 {out[k]['err']:.3f}   排名 ρ 均值 {out[k]['rho']:+.2f}（ρ≥0.6 的比例 {100*out[k]['rho_hit']:.0f}%）  P-F 命中率误差 {out[k]['pf_err']:.1f} pt")
    lhd=np.array([PL[m]["point"]["CFR"] for m in models])
    le=float(np.nanmean(np.abs(lhd-truth))); lr=float(spearmanr(lhd,truth)[0])
    print(f"  沿用左舵体检的数：CFR 绝对误差 {le:.3f}   排名 ρ {lr:+.2f}")
    return out,le,lr

rng=np.random.default_rng(0)
r6,le6,lr6=sweep(M,COM,GC,[5,10,20,40],"A. 六家 / 共有场景",rng)
M4=["dd","ltf","ddv2","simlingo"]; POOL4=sorted(set.intersection(*[{r["tok"] for r in U[m]} for m in M4]))
r4,le4,lr4=sweep(M4,POOL4,G,[5,10,20,40,80,160],"B. 四家 / 全部 254 场景（小样本→大体量）",rng)
json.dump({"truth_full":G,"truth_common":GC,"n_common":len(COM),
           "sweep6":r6,"lhd6":{"err":le6,"rho":lr6},
           "sweep4":r4,"lhd4":{"err":le4,"rho":lr4}},
          open(f"{V5}/domain_transfer.json","w"),indent=1,default=float)
print("\nwrote domain_transfer.json")

# ---- 行为层：左舵体检的判读能否预测右舵大样本上的"实际危险行为"次序？小样本能否排出这个次序？
BEH="ttc_viol"   # 右舵近行人场景里最小 TTC<1.5 s 的比例（对行人真实未来轨迹算）
bt=np.array([GC[m][BEH] for m in M])
for key in ["exposure","SP","CFR"]:
    lv=np.array([PL[m]["point"][key] for m in M])
    print(f"  左舵 {key:8s} → 右舵 TTC 违规率  ρ={spearmanr(lv,bt)[0]:+.2f}")
print("  右舵自身 CFR → 右舵 TTC 违规率  ρ=%+.2f"%spearmanr([GC[m]["CFR"] for m in M],bt)[0])
print("  右舵自身 F 中位 → 右舵 TTC 违规率  ρ=%+.2f"%spearmanr([GC[m]["F_med"] for m in M],bt)[0])
print("  右舵自身 I 中位 → 右舵 TTC 违规率  ρ=%+.2f"%spearmanr([GC[m]["I_med"] for m in M],bt)[0])
beh={}
print(f"\nC. 小样本能否排出右舵的行为次序（真值 = 79 场景的 TTC 违规率）:")
for k in [5,10,20,40]:
    rho=[];err=[]
    for _ in range(200):
        ss=set(rng.choice(COM,k,replace=False))
        est=np.array([agg([r for r in U[m] if r["tok"] in ss])[BEH] for m in M])
        rho.append(spearmanr(est,bt)[0]); err.append(np.nanmean(np.abs(est-bt)))
    beh[k]=dict(rho=float(np.nanmean(rho)),rho_hit=float(np.mean(np.array(rho)>=0.6)),err=float(np.nanmean(err)))
    print(f"  k={k:3d}  排名 ρ 均值 {beh[k]['rho']:+.2f}（ρ≥0.6 的比例 {100*beh[k]['rho_hit']:.0f}%）  违规率绝对误差 {beh[k]['err']:.1f} pt")
lex=np.array([PL[m]["point"]["exposure"] for m in M])
J=json.load(open(f"{V5}/domain_transfer.json")); J["behaviour"]=beh
J["lhd_to_behaviour"]={key:float(spearmanr([PL[m]["point"][key] for m in M],bt)[0]) for key in ["exposure","SP","CFR"]}
J["rhd_axes_to_behaviour"]={key:float(spearmanr([GC[m][key] for m in M],bt)[0]) for key in ["CFR","F_med","I_med"]}
json.dump(J,open(f"{V5}/domain_transfer.json","w"),indent=1,default=float)
print("\nupdated domain_transfer.json")

