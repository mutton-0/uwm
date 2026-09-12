"""小样本拟合（v2 口径）。六家共同帧（前视可见、有行人未来、左右舵合并），按帧无放回子采样 n 帧，重复 400 次。
每个维度：SD(n) 拟合 a·√(1/n − 1/N)；判定所需帧数 n* = (a/δ)²，δ = |估计 − 判定阈| / 2（2 SD 离开阈值）；
排序：六家排序与全样本的 Spearman ≥ 0.6 的概率。"""
import json,sys,numpy as np
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5/scripts")
from diag_profile import dims
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
rows=json.load(open(f"{V5}/diag_units.json"))
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr")
by={m:{} for m in M}
for z in rows:
    if sel(z) and z["m"] in by: by[z["m"]].setdefault(z["uid"],[]).append(z)
U=np.array(sorted(set.intersection(*[set(by[m]) for m in M]))); N=len(U)
KEYS={"exposure":1.0,"CFR":1.0,"SP":None,"HS":0.0,"align":0.0}
full={m:dims([z for u in U for z in by[m][u]],IDX) for m in M}
NS=[n for n in (10,15,25,40,60,90,130) if n<N]; rng=np.random.default_rng(0)
est={k:{m:{n:[] for n in NS} for m in M} for k in KEYS}
for n in NS:
    for _ in range(400):
        us=U[rng.choice(N,n,replace=False)]
        for m in M:
            d=dims([z for u in us for z in by[m][u]],IDX)
            for k in KEYS: est[k][m][n].append(d[k])
out={"N":N,"ns":NS,"dims":{}}
for k,thr in KEYS.items():
    D={"full":{m:float(full[m][k]) for m in M},"fit":{},"n_star":{},"rank_p":{}}
    for m in M:
        y=np.array([np.nanstd(est[k][m][n]) for n in NS]); x=np.array(NS,float); ok=~np.isnan(y)
        f=lambda n,a: a*np.sqrt(np.clip(1/n-1/N,0,None))
        try:
            (a,),_=curve_fit(f,x[ok],y[ok],p0=[y[ok][0]*np.sqrt(x[ok][0])]); r2=1-np.sum((y[ok]-f(x[ok],a))**2)/np.sum((y[ok]-y[ok].mean())**2)
        except Exception: a,r2=np.nan,np.nan
        D["fit"][m]=dict(sd=y.tolist(),a=float(a),r2=float(r2))
        if thr is not None and not np.isnan(full[m][k]):
            dlt=abs(full[m][k]-thr)/2; D["n_star"][m]=float((a/dlt)**2) if dlt>0 else float("inf")
    for n in NS:
        arr=np.array([est[k][m][n] for m in M]).T; fv=np.array([full[m][k] for m in M]); good=~np.isnan(arr).any(1)&~np.isnan(fv).any()
        rr=[spearmanr(r,fv)[0] for r in arr[good]]; D["rank_p"][n]=float(np.mean(np.array(rr)>=0.6)) if rr else None
    out["dims"][k]=D
    print(f"== {k}: 全样本 "+" ".join(f"{m}={full[m][k]:.3f}" for m in M))
    print("   a,R²: "+" ".join(f"{m}:{D['fit'][m]['a']:.2f}/{D['fit'][m]['r2']:.2f}" for m in M))
    if D["n_star"]: print("   n*: "+" ".join(f"{m}:{v:.0f}" for m,v in D["n_star"].items()))
    print("   P(排序ρ≥0.6): "+" ".join(f"n={n}:{p:.2f}" for n,p in D["rank_p"].items() if p is not None))
json.dump(out,open(f"{V5}/sample_size_v2.json","w"),indent=1,default=float)
