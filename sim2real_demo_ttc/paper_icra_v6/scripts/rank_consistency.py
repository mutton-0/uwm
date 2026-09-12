"""哪种评测能预测模型在"右舵近行人场景"里的实际表现排名？
目标（NAVSIM 新加坡 260 个近行人场景，原图，不编辑；自车在走 v0≥1 m/s 的场景）：
  TTC 违规率（0–2.5 s 最小 TTC<1.5 s）、与行人最小净间隙中位（0–4 s，越大越安全）、碰撞率（间隙<0）、近行人 EPDMS（全部 260）
预测者（都不用右舵 NAVSIM 数据）：
  A  同一指标在 nuScenes 波士顿（左舵）近行人帧上      —— "同样的考法、换个域"
  A2 同一指标在 NAVSIM 美国三城（左舵）近行人场景上
  B  nuScenes 标准开环指标（波士顿近行人帧）：L2、前方碰撞、撞行人
  C  NAVSIM 榜单分（新加坡 783 场景 EPDMS）与左舵近行人 EPDMS
  D  我们的体检（波士顿诊断）：暴露度、危险敏感度、特异度、光照 CFR、|夜里提速|、起步率
全部换成"1 = 最安全/最好"的名次，再算与目标名次的 Spearman（n=6，|ρ|≥0.89 双侧 5% 显著）。"""
import json,glob,os,numpy as np
from scipy.stats import spearmanr
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
def pick(*c):
    for f in c:
        if os.path.exists(f): return f
TRR={m:json.load(open(pick(f"{R5}/nvtraj_{m}_sg-one-north_closevru_nav.json",f"{R5}/nvtraj_{m}_sg-one-north_closevru.json"))) for m in M}
TRL={m:json.load(open(pick(f"{R5}/nvtraj_{m}_any_lhdclose_nav.json",f"{R5}/nvtraj_{m}_any_lhdclose.json"))) for m in M}
SCR={m:json.load(open(pick(f"{R5}/nvscore_{m}_sg-one-north_closevru_nav.json",f"{R5}/nvscore_{m}_sg-one-north_closevru.json")))["rows"] for m in M}
SCL={m:json.load(open(pick(f"{R5}/nvscore_{m}_any_lhdclose_nav.json",f"{R5}/nvscore_{m}_any_lhdclose.json")))["rows"] for m in M}
SCA={m:json.load(open(pick(f"{R5}/nvscore_{m}_sg-one-north_nav.json",f"{R5}/nvscore_{m}_sg-one-north_fix.json" if m=="simlingo" else f"{R5}/nvscore_{m}_sg-one-north.json")))["rows"] for m in M}
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def lin(TT,T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def readout(X_pts,Xt,F_pts,Ft,H):
    TT=np.round(np.arange(0,H+1e-6,0.1),2); X=lin(TT,Xt,X_pts); F=lin(TT,Ft,F_pts); c=np.linalg.norm(X-F,axis=1)-1.4
    m25=TT<=2.5; cc=c[m25]; t25=TT[m25]
    if cc.min()<=0: ttc=0.0
    else:
        cl=-np.gradient(cc,t25); ok=cl>0.05; ttc=float(min(5,np.min(cc[ok]/cl[ok]))) if ok.any() else 5.0
    return ttc,float(c.min())
def nv_metrics(TR):
    out={}
    for m in M:
        r=[]
        for t,w in TR[m].items():
            p=PF.get(t)
            if p is None or p["v0"]<1.0 or any(z is None for z in p["fut"]): continue
            w=np.asarray(w,float)[:,:2]
            r.append(readout(np.vstack([[0,0],w]),np.r_[0,np.arange(1,len(w)+1)*0.5],np.vstack([p["p0"],np.asarray(p["fut"])]),np.r_[0,np.arange(1,9)*0.5],4.0))
        r=np.array(r); out[m]=dict(n=len(r),ttc_viol=float(100*np.mean(r[:,0]<1.5)),clear_med=float(np.median(r[:,1])),coll=float(100*np.mean(r[:,1]<0)))
    return out
RHD=nv_metrics(TRR); LHDN=nv_metrics(TRL)
for m in M: RHD[m]["epdms"]=float(np.mean([r["pdm_score"] for r in SCR[m]])); LHDN[m]["epdms"]=float(np.mean([r["pdm_score"] for r in SCL[m]]))
# nuScenes 波士顿近行人帧（风险卡），同一读数（0–2.5 s；行人未来只有 2.5 s）
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
NUL={}
for m in M:
    C={**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")}; r=[]
    for u,rr in C.items():
        x=IDX[u]
        if x["side"]!="LHD" or not x["front_only"] or "actual" not in rr or float(x["v_act"])<1.0: continue
        if float(x["dep"])>15 or not (x["set"]=="B" or x.get("grp")=="corr"): continue
        f=x.get("ped_future_ego") or []
        if len(f)<5 or any(z is None for z in f[:5]): continue
        T6=np.r_[0,0.5,1,1.5,2,2.5]
        r.append(readout(np.vstack([[0,0],rr["actual"]["clean"]]),T6,np.vstack([np.mean(np.asarray(x["corners_ego"])[:,:2],0),f[:5]]),T6,2.5))
    r=np.array(r); NUL[m]=dict(n=len(r),ttc_viol=float(100*np.mean(r[:,0]<1.5)),clear_med=float(np.median(r[:,1])),coll=float(100*np.mean(r[:,1]<0)))
NU=json.load(open(f"{V5}/nusc_openloop.json")); PL=json.load(open(f"{V5}/profile_LHD.json")); CF=json.load(open(f"{R5}/paper_icra_v4/cfr_same_frame.json"))
NS=json.load(open(f"{V5}/night_speed.json")); DEP=json.load(open(f"{V5}/depart_when_waiting.json"))
def rank(v,hb):  # 1 = 最好
    o=sorted(M,key=lambda m:-v[m] if hb else v[m]); return [o.index(m)+1 for m in M]
T={"TTC 违规率":rank({m:RHD[m]["ttc_viol"] for m in M},False),"离行人最小间隙":rank({m:RHD[m]["clear_med"] for m in M},True),
   "撞行人率":rank({m:RHD[m]["coll"] for m in M},False),"近行人 EPDMS":rank({m:RHD[m]["epdms"] for m in M},True)}
P={"A nuScenes 左舵·TTC 违规率":rank({m:NUL[m]["ttc_viol"] for m in M},False),"A nuScenes 左舵·最小间隙":rank({m:NUL[m]["clear_med"] for m in M},True),
   "A nuScenes 左舵·撞行人率":rank({m:NUL[m]["coll"] for m in M},False),
   "A2 NAVSIM 左舵·TTC 违规率":rank({m:LHDN[m]["ttc_viol"] for m in M},False),"A2 NAVSIM 左舵·最小间隙":rank({m:LHDN[m]["clear_med"] for m in M},True),
   "A2 NAVSIM 左舵·近行人 EPDMS":rank({m:LHDN[m]["epdms"] for m in M},True),
   "B nuScenes 开环·L2":rank({m:NU[m]["clean_LHD"]["L2_avg"] for m in M},False),"B nuScenes 开环·前方碰撞":rank({m:NU[m]["clean_LHD"]["col_front"] for m in M},False),
   "B nuScenes 开环·撞行人":rank({m:NU[m]["clean_LHD"]["col_ped"] for m in M},False),
   "C NAVSIM 榜单 EPDMS（783）":rank({m:float(np.mean([r["pdm_score"] for r in SCA[m]])) for m in M},True),
   "D 体检·暴露度（小=安全）":rank({m:PL[m]["point"]["exposure"] for m in M},False),"D 体检·危险敏感度":rank({m:(PL[m]["point"]["HS"] if PL[m]["point"]["HS"] is not None else -9) for m in M},True),
   "D 体检·特异度":rank({m:PL[m]["point"]["SP"] for m in M},True),"D 体检·光照 CFR":rank({m:CF[m]["corr"]["CFR"] for m in M},True),
   "D 体检·|夜里提速|":rank({m:abs(NS[m]["dv_rel"]) for m in M},False),"D 体检·起步率":rank({m:DEP[m]["depart"] for m in M},False)}
tab={p:{t:float(spearmanr(pv,tv)[0]) for t,tv in T.items()} for p,pv in P.items()}
json.dump({"targets_RHD":RHD,"nusc_LHD":NUL,"navsim_LHD":LHDN,"ranks_targets":T,"ranks_predictors":P,"spearman":tab},open(f"{V5}/rank_consistency.json","w"),indent=1,ensure_ascii=False)
print("右舵 NAVSIM 近行人（原图）实际表现:")
for m in M: print(f"  {m:9s} n={RHD[m]['n']:3d} TTC违规 {RHD[m]['ttc_viol']:5.1f}%  最小间隙中位 {RHD[m]['clear_med']:5.2f} m  撞 {RHD[m]['coll']:4.1f}%  近行人EPDMS {RHD[m]['epdms']:.3f}")
print("nuScenes 波士顿近行人:")
for m in M: print(f"  {m:9s} n={NUL[m]['n']:3d} TTC违规 {NUL[m]['ttc_viol']:5.1f}%  最小间隙中位 {NUL[m]['clear_med']:5.2f} m  撞 {NUL[m]['coll']:4.1f}%")
print("\n"+f"{'预测者 / 右舵目标':34s}"+"".join(f"{t:>14s}" for t in T))
for p,row in tab.items(): print(f"{p:34s}"+"".join(f"{v:+14.2f}" for v in row.values()))
