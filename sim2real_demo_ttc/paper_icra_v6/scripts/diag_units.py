"""诊断单位表（v2 口径，用户 2026-09-11 确认方向）：所有读数都看整条规划轨迹相对目标行人的远离程度，不用速度/减速。
单位 = (帧, 输入车速)；Set A 只有实际车速，Set B 另有反事实车速扫描。只用前视可见帧（VLA 只改了前视图）。
轨迹与行人真实未来都插值到 0.1 s（t=0 自车在原点、行人在当前 3D 框中心），到 2.5 s。
  clear_t(X) = ‖X_t − F_t‖ − 1.0（自车半宽）− 0.4（行人半径）
  分离度  S(X)  = mean_t min(clear_t, 10)           —— 整条轨迹离行人多远（>10 m 视为已无关）
  接触    A(X)  = 1[min_t clear_t > 0]                —— 避让成功
  安全距  C(X)  = clip(min_t clear_t / 1 m, 0, 1)      —— 距离安全
  TTC     T(X)  = 首次 clear_t < 1 m 的时刻（没有则 2.5 s），归一 T/2.5 —— 贴近前还有多少时间
危险等级（与模型无关）：a_req = v²/(2·max(d−2,0.5))、TTC0 = max(d−2,0)/v
需要反应 need = C(Q) < 1（看不到行人时的规划会进入 1 m 以内）
  敏感度 HS = ¼[ΔA + ΔC + ΔT + tanh(ΔS/1 m)]，Δ = 看得到 − 看不到，∈[−1,1]
  特异度 SP = exp(−D(P,Q)/0.5 m)（非 need 单位：不需要时轨迹保持不动，最小变化原则）
低速规则：输入车速 < 1 m/s 的单位不参与（自车几乎停着，远离与否由不得它）。
光照：实际车速单位额外记 Δ_I = plan(夜) − plan(原图)、Δ_F = plan(移除) − plan(原图)（10 维），供纯粹度用。"""
import json,glob,numpy as np
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo","alpamayo15"]
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
TT=np.round(np.arange(0,2.51,0.1),2); TQ=np.array([0,0.5,1.0,1.5,2.0,2.5])
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
def lin(a): a=np.asarray(a,float); return np.stack([np.interp(TT,TQ,a[:,0]),np.interp(TT,TQ,a[:,1])],1)
def plan(p): return lin(np.vstack([[0,0],np.asarray(p,float)]))
def fut(x):
    f=x.get("ped_future_ego") or []
    if len(f)<5 or any(z is None for z in f[:5]): return None
    return lin(np.vstack([np.mean(np.asarray(x["corners_ego"])[:,:2],0)[None],np.asarray(f[:5],float)]))
def feats(X,F):
    c=np.linalg.norm(X-F,axis=1)-1.4
    hit=np.nonzero(c<1.0)[0]; t1=TT[hit[0]] if len(hit) else 2.5
    return dict(S=float(np.mean(np.minimum(c,10))),A=float(c.min()>0),C=float(np.clip(c.min(),0,1)),T=float(t1/2.5),cmin=float(c.min()))
def D(p,q): return float(np.mean(np.linalg.norm(np.asarray(p)-np.asarray(q),axis=1)))
rows=[]
for m in M:
    C={**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")}
    N={**load(f"{R5}/card_night_{m}.json"),**load(f"{R5}/card_night_{m}_*.json")}
    for u,r in C.items():
        x=IDX[u]
        if not x["front_only"]: continue
        F=fut(x)
        if F is None: continue
        d=float(x["dep"])
        for sv in ("actual","2","4","6","8"):
            if sv not in r: continue
            v=float(x["v_act"]) if sv=="actual" else float(sv)
            fp,fq=feats(plan(r[sv]["clean"]),F),feats(plan(r[sv]["rm"]),F)
            z=dict(m=m,uid=u,set=x["set"],grp=x.get("grp","B"),side=x["side"],sv=sv,v=v,d=d,
                   a_req=v*v/(2*max(d-2,0.5)),ttc0=max(d-2,0)/max(v,1e-3),
                   P=fp,Q=fq,Dm=D(r[sv]["clean"],r[sv]["rm"]),need=fq["C"]<1.0)
            z["HS"]=0.25*((fp["A"]-fq["A"])+(fp["C"]-fq["C"])+(fp["T"]-fq["T"])+np.tanh(fp["S"]-fq["S"]))
            z["SP"]=float(np.exp(-z["Dm"]/0.5))
            if sv=="actual" and u in N:
                c0=np.asarray(r["actual"]["clean"],float)
                z["dI"]=(np.asarray(N[u]["night"],float)-c0).ravel().tolist()
                z["dF"]=(np.asarray(r["actual"]["rm"],float)-c0).ravel().tolist()
                z["fN"]=feats(plan(N[u]["night"]),F)
            rows.append(z)
json.dump(rows,open(f"{V5}/diag_units.json","w"))
import collections
print(collections.Counter((z["m"],z["side"],z["v"]>=1.0) for z in rows if z["m"] in("dd","autovla")))
