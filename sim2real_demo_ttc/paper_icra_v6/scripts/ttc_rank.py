"""行人 TTC 排名：左舵（nuScenes 波士顿近行人帧 + NAVSIM 美国三城近行人场景）vs 右舵（NAVSIM 新加坡近行人场景），
再与我们的行人敏感度（HS）、光照不敏感度（CFR；|夜化速度变化|）排名比较。
TTC 定义（两个数据集同一口径）：公共窗口 0–2.5 s、0.1 s 网格；c_t = ‖规划_t − 行人_t‖ − 1.4 m；
  接近速度 = −dc/dt；TTC_t = c_t / 接近速度（仅在接近时）；场景最小 TTC（接触记 0，从不接近记 5 s）
  违规 = 最小 TTC < 1.5 s。低速规则：自车初速 < 1 m/s 的场景不进主结果（另报全部）。
NAVSIM 另报官方 TTC 子项（time_to_collision_within_bound，4 s，全部物体）。"""
import json,glob,numpy as np
from scipy.stats import spearmanr,kendalltau
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo"}
TT=np.round(np.arange(0,2.51,0.1),2)
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def min_ttc(X,F):
    c=np.linalg.norm(X-F,axis=1)-1.4
    if c.min()<=0: return 0.0
    close=-np.gradient(c,TT); ok=close>0.05
    return float(min(5.0,np.min(c[ok]/close[ok]))) if ok.any() else 5.0
# ---- NAVSIM ----
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def nv_units(m,fname):
    T=json.load(open(fname)); out=[]
    for tok,w in T.items():
        p=PF.get(tok)
        if p is None or any(z is None for z in p["fut"][:5]): continue
        w=np.asarray(w,float)[:,:2]; X=lin(np.r_[0,np.arange(1,len(w)+1)*0.5],np.vstack([[0,0],w]))
        F=lin(np.r_[0,0.5,1.0,1.5,2.0,2.5],np.vstack([p["p0"],np.asarray(p["fut"][:5])]))
        out.append(dict(tok=tok,v0=p["v0"],ttc=min_ttc(X,F),src="navsim"))
    return out
import os as _os
def _pick(*c):
    for f in c:
        if _os.path.exists(f): return f
def nvfile(m,tag):
    if tag=="rhd": return _pick(f"{R5}/nvtraj_{m}_sg-one-north_closevru_nav.json",f"{R5}/nvtraj_{m}_sg-one-north_closevru.json")
    return _pick(f"{R5}/nvtraj_{m}_any_lhdclose_nav.json",f"{R5}/nvtraj_{m}_any_lhdclose.json")
# ---- nuScenes 左舵（风险卡近行人帧）----
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
def nu_units(m,side):
    C={**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")}; out=[]
    for u,r in C.items():
        x=IDX[u]
        if x["side"]!=side or not x["front_only"] or "actual" not in r: continue
        if float(x["dep"])>15 or not (x["set"]=="B" or x.get("grp")=="corr"): continue
        f=x.get("ped_future_ego") or []
        if len(f)<5 or any(z is None for z in f[:5]): continue
        X=lin(np.r_[0,0.5,1.0,1.5,2.0,2.5],np.vstack([[0,0],np.asarray(r["actual"]["clean"])]))
        F=lin(np.r_[0,0.5,1.0,1.5,2.0,2.5],np.vstack([np.mean(np.asarray(x["corners_ego"])[:,:2],0),np.asarray(f[:5])]))
        out.append(dict(tok=u,v0=float(x["v_act"]),ttc=min_ttc(X,F),src="nuscenes"))
    return out
def summ(U,moving=True):
    U=[z for z in U if (z["v0"]>=1.0)==moving] if moving is not None else U
    t=np.array([z["ttc"] for z in U]); return dict(n=len(U),viol=float(100*np.mean(t<1.5)) if len(t) else None,med=float(np.median(t)) if len(t) else None)
res={"LHD":{},"RHD":{},"LHD_nusc":{},"LHD_navsim":{}}
for m in M:
    lh_nv=nv_units(m,nvfile(m,"lhd")); rh_nv=nv_units(m,nvfile(m,"rhd")); lh_nu=nu_units(m,"LHD")
    res["LHD"][m]={"moving":summ(lh_nv+lh_nu),"all":summ(lh_nv+lh_nu,None)}
    res["LHD_nusc"][m]={"moving":summ(lh_nu),"all":summ(lh_nu,None)}; res["LHD_navsim"][m]={"moving":summ(lh_nv),"all":summ(lh_nv,None)}
    res["RHD"][m]={"moving":summ(rh_nv),"all":summ(rh_nv,None)}
# 官方 TTC 子项
def off(m,tag):
    f=_pick(f"{R5}/nvscore_{m}_sg-one-north_closevru_nav.json",f"{R5}/nvscore_{m}_sg-one-north_closevru.json") if tag=="rhd" else _pick(f"{R5}/nvscore_{m}_any_lhdclose_nav.json",f"{R5}/nvscore_{m}_any_lhdclose.json")
    rows=json.load(open(f))["rows"]; return float(np.mean([r["time_to_collision_within_bound"] for r in rows])),float(np.mean([r["pdm_score"] for r in rows]))
for m in M:
    res["RHD"][m]["official_ttc"],res["RHD"][m]["epdms"]=off(m,"rhd"); res["LHD"][m]["official_ttc"],res["LHD"][m]["epdms"]=off(m,"lhd")
# 我们的排名依据
PR=json.load(open(f"{V5}/profile_ALL.json")); CF=json.load(open(f"{R5}/paper_icra_v4/cfr_same_frame.json")); NS=json.load(open(f"{V5}/night_speed.json"))
ours={m:dict(HS=PR[m]["point"]["HS"],CFR=CF[m]["corr"]["CFR"],night_abs=abs(NS[m]["dv_rel"]),exposure=PR[m]["point"]["exposure"]) for m in M}
res["ours"]=ours
def rank(vals,higher_better): 
    o=sorted(M,key=lambda m:-vals[m] if higher_better else vals[m]); return {m:o.index(m)+1 for m in M}
R={"LHD TTC":rank({m:res["LHD"][m]["moving"]["viol"] for m in M},False),
   "RHD TTC":rank({m:res["RHD"][m]["moving"]["viol"] for m in M},False),
   "LHD official TTC":rank({m:res["LHD"][m]["official_ttc"] for m in M},True),
   "RHD official TTC":rank({m:res["RHD"][m]["official_ttc"] for m in M},True),
   "HS (ours)":rank({m:ours[m]["HS"] for m in M},True),
   "lighting CFR (ours)":rank({m:ours[m]["CFR"] for m in M},True),
   "lighting |Δspeed| (ours)":rank({m:ours[m]["night_abs"] for m in M},False),
   "exposure (ours, low=safe)":rank({m:ours[m]["exposure"] for m in M},False)}
res["ranks"]=R
keys=list(R); rho={a:{b:float(spearmanr([R[a][m] for m in M],[R[b][m] for m in M])[0]) for b in keys} for a in keys}
res["spearman"]=rho
json.dump(res,open(f"{V5}/ttc_rank.json","w"),indent=1)
print("TTC 违规率（最小 TTC<1.5 s，自车在走的场景）:")
for m in M:
    L=res["LHD"][m]; Rr=res["RHD"][m]
    print(f"  {NAME[m]:9s} 左舵 {L['moving']['viol']:5.1f}% (n={L['moving']['n']}; nuSc {res['LHD_nusc'][m]['moving']['viol']:.1f}% n={res['LHD_nusc'][m]['moving']['n']}, NAVSIM {res['LHD_navsim'][m]['moving']['viol']:.1f}% n={res['LHD_navsim'][m]['moving']['n']})"
          f" | 右舵 {Rr['moving']['viol']:5.1f}% (n={Rr['moving']['n']}) | 官方TTC 左 {L['official_ttc']:.3f} 右 {Rr['official_ttc']:.3f} | EPDMS 左 {L['epdms']:.3f} 右 {Rr['epdms']:.3f}")
print("排名:"); [print(f"  {k:26s}"+" ".join(f"{NAME[m]}:{v}" for m,v in R[k].items())) for k in keys]
print("Spearman vs 左舵 TTC:",{b:round(rho['LHD TTC'][b],2) for b in keys}); print("Spearman vs 右舵 TTC:",{b:round(rho['RHD TTC'][b],2) for b in keys})
