"""F 轴动了，是往哪边动？
对每个近行人场景：clr = 原图规划与行人真实未来的最小净间隙；clr_rm = 抹掉行人后同一读数。
  Δ = clr − clr_rm > 0  → 看到行人让它离得更远 = 真避让
  Δ < 0                 → 看到行人反而让它离得更近 = 反向
只统计 F ≥ 0.5 m（规划确实被行人改动了）的场景，看这些改动花在哪。
输出 avoid_sign.json"""
import json,os,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
PF_=json.load(open(f"{V5}/nv_ped_future.json"))
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def clr(w,F):
    X=lin(T8,np.vstack([[0,0],np.asarray(w)[:,:2]])); return float((np.linalg.norm(X-F,axis=1)-1.4).min())
def D(a,b):
    A=lin(T8,np.vstack([[0,0],np.asarray(a)[:,:2]])); B=lin(T8,np.vstack([[0,0],np.asarray(b)[:,:2]]))
    return float(np.mean(np.linalg.norm(A-B,axis=1)))
out={}
for m in M:
    p=f"{V5}/rhd_axes_{m}.json"
    if not os.path.exists(p): continue
    d=json.load(open(p)); big=[]; allsc=[]
    for t,a in d.items():
        q=PF_.get(t)
        if q is None or "rm" not in a or any(z is None for z in q["fut"][:5]): continue
        F=lin(T6,np.vstack([q["p0"],np.asarray(q["fut"][:5])]))
        f=D(a["clean"],a["rm"]); c0=clr(a["clean"],F); c1=clr(a["rm"],F)
        allsc.append((f,c0-c1))
        if f>=0.5: big.append(c0-c1)
    b=np.array(big); n=len(b)
    out[m]=dict(n_all=len(allsc), n_big=n,
        far=int((b>=0.5).sum()) if n else 0, near=int((b<=-0.5).sum()) if n else 0,
        flat=int((abs(b)<0.5).sum()) if n else 0,
        med=float(np.median(b)) if n else None)
    g=out[m]
    print(f"  {m:11s} F≥0.5 的场景 {g['n_big']:3d}/{g['n_all']:3d} → 离更远 {g['far']:2d}  离更近 {g['near']:2d}  间隙没变 {g['flat']:2d}"
          + (f"  Δ中位 {g['med']:+.2f} m" if g['med'] is not None else ""))
T=[v for m in out for v in [out[m]]]
print(f"\n合计：F≥0.5 的 {sum(v['n_big'] for v in T)} 个场景里，真的离更远 {sum(v['far'] for v in T)} 个、"
      f"反而更近 {sum(v['near'] for v in T)} 个、间隙没变 {sum(v['flat'] for v in T)} 个")
json.dump(out,open(f"{V5}/avoid_sign.json","w"),indent=1)
