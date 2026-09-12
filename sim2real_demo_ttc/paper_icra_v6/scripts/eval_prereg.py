"""按 PREREG_rhd_personality.md 的判定规则逐条判定。"""
import json,numpy as np
from scipy.stats import spearmanr
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
L=json.load(open(f"{V5}/profile_LHD.json")); R=json.load(open(f"{V5}/profile_RHD.json")); rows=json.load(open(f"{V5}/diag_units.json"))
import os as _os
# 预注册登记的是六家（含当时发布的 Alpamayo 1）。NVIDIA 发布带导航的 1.5 后全文换用 1.5，
# 而 1.5 的左右舵数是同时算出来的、不满足预注册纪律，故主表只判定登记时的其余五家；
# 设 PREREG_MODELS=six 可复现"含 Alpamayo 1"的原始版本，用于核对去掉它是否改变结论。
M=["dd","ltf","ddv2","simlingo","autovla"] if _os.environ.get("PREREG_MODELS","five")=="five" else ["dd","ltf","ddv2","simlingo","autovla","alpamayo"]
res={}
p1=all(R[m]["ci"]["CFR"][1]<1 for m in M); res["P1"]=dict(pass_=p1,detail={m:R[m]["ci"]["CFR"] for m in M})
el=[m for m in M if R[m]["point"]["n_need"]>=10]
p2=all(R[m]["ci"]["HS"][1]<0.15 for m in el); res["P2"]=dict(pass_=p2,detail={m:R[m]["ci"]["HS"] for m in el})
ex=lambda P: [P[m]["point"]["exposure"] for m in M]
rho3=spearmanr(ex(L),ex(R))[0]; top=M[int(np.argmax(ex(R)))]; bot=M[int(np.argmin(ex(R)))]
res["P3"]=dict(pass_=bool(rho3>=0.6 and top=="simlingo" and bot=="dd"),rho=rho3,top=top,bottom=bot)
sp=lambda P: [P[m]["point"]["SP"] for m in M]
rho4=spearmanr(sp(L),sp(R))[0]; low=M[int(np.argmin(sp(R)))]
res["P4"]=dict(pass_=bool(rho4>=0.6 and low=="simlingo"),rho=rho4,lowest=low)
hit=[];cells={}
for m in M:
    for k in ("exposure","SP","CFR"):
        lo,hi=L[m]["ci"][k]; v=R[m]["point"][k]; h=bool(lo<=v<=hi); hit.append(h); cells[f"{m}.{k}"]=dict(L=L[m]["point"][k],ci=[lo,hi],R=v,hit=h)
res["P5"]=dict(pass_=bool(np.mean(hit)>=0.7),rate=float(np.mean(hit)),cells=cells)
a=[abs(L[m]["point"]["align"]) for m in M]; dc=[abs(R[m]["point"]["CFR"]-L[m]["point"]["CFR"]) for m in M]
rho6=spearmanr(a,dc)[0]; res["P6"]=dict(pass_=bool(rho6>0),rho=rho6,abs_align_L=dict(zip(M,a)),dCFR=dict(zip(M,dc)))
# P7 右舵实际车速碰撞
col={}
for m in M:
    U=[z for z in rows if z["m"]==m and z["side"]=="RHD" and z["sv"]=="actual" and z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr")]
    see=np.mean([z["P"]["A"]==0 for z in U]); blind=np.mean([z["Q"]["A"]==0 for z in U]); col[m]=dict(n=len(U),see=float(see),blind=float(blind))
topc=max(M,key=lambda m:col[m]["see"]); gap=max(abs(col[m]["see"]-col[m]["blind"]) for m in M)
res["P7"]=dict(pass_=bool(topc=="simlingo" and gap<=0.03),top=topc,max_gap=float(gap),detail=col)
# 用户要的：偏移程度与正交程度变化
shift={m:{k:(R[m]["point"][k]-L[m]["point"][k]) if (R[m]["point"][k] is not None and L[m]["point"][k] is not None) else None
          for k in ("exposure","HS","SP","CFR","align","night_sep")} for m in M}
res["shift"]=shift
json.dump(res,open(f"{V5}/prereg_result{'' if len(M)==5 else '_six'}.json","w"),indent=1,default=float)
for k in ["P1","P2","P3","P4","P5","P6","P7"]:
    d=res[k]; print(k,"通过" if d["pass_"] else "否定",{kk:vv for kk,vv in d.items() if kk not in ("pass_","cells","detail","abs_align_L","dCFR")})
print("P7 碰撞:",{m:(round(v['see'],3),round(v['blind'],3),v['n']) for m,v in col.items()})
print("P5 逐格:",{k:v["hit"] for k,v in cells.items()})
print("偏移 RHD−LHD:"); [print(" ",m,{k:(None if v is None else round(v,3)) for k,v in s.items()}) for m,s in shift.items()]
