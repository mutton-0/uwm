"""EPDMS 模型间方差的 Shapley 分解（783 个新加坡场景，同一 scorer）。SimLingo 用修正轨迹（_fix：0.25 s 路点正确重采样 + 2.5 s 后匀速外推）。
EPDMS = NC·DAC·DDC·TLC × (5EP+5TTC+2LK+2HC)/14。"""
import json,itertools,math,numpy as np
from scipy.stats import spearmanr
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
import os as _os
def fn(m):
    for f in (f"{R5}/nvscore_{m}_sg-one-north_nav.json",f"{R5}/nvscore_{m}_sg-one-north_fix.json",f"{R5}/nvscore_{m}_sg-one-north.json"):
        if _os.path.exists(f): return f
R={m:{r["token"]:r for r in json.load(open(fn(m)))["rows"]} for m in M}
toks=sorted(set.intersection(*[set(v) for v in R.values()]))
F=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","traffic_light_compliance"]
def score(r,neut=()):
    mult=np.prod([1.0 if f in neut else r[f] for f in F]); return mult*(5*r["ego_progress"]+5*r["time_to_collision_within_bound"]+2*r["lane_keeping"]+2*r["history_comfort"])/14
tab={m:dict(pdms=float(np.mean([score(R[m][t]) for t in toks])),**{f:float(np.mean([R[m][t][f] for t in toks])) for f in F+["ego_progress","time_to_collision_within_bound","lane_keeping","history_comfort"]}) for m in M}
G={"NC":[F[0]],"DAC":[F[1]],"DDC":[F[2]],"TLC":[F[3]]}; cache={}
def V(act):
    k=frozenset(act)
    if k not in cache:
        neut=tuple(f for g,fs in G.items() if g not in act for f in fs); cache[k]=float(np.var([np.mean([score(R[m][t],neut) for t in toks]) for m in M]))
    return cache[k]
sh={}
for g in G:
    o=[h for h in G if h!=g]; sh[g]=0.0
    for k in range(len(o)+1):
        for S_ in itertools.combinations(o,k): sh[g]+=math.factorial(k)*math.factorial(len(G)-k-1)/math.factorial(len(G))*(V(set(S_)|{g})-V(set(S_)))
tot=V(set(G))
out={"tokens":len(toks),"table":tab,"shapley_dvar_over_full":{g:x/tot for g,x in sh.items()},"var_no_mult":V(set()),"var_full":tot,"note":"simlingo uses _fix trajectories"}
json.dump(out,open(f"{R5}/pdms_decomp_sg.json","w"),indent=1)
print("n",len(toks)); [print(m,{k:round(v,3) for k,v in tab[m].items() if k in ("pdms","no_at_fault_collisions","drivable_area_compliance","ego_progress")}) for m in M]
print("Shapley",{g:round(x/tot,2) for g,x in sh.items()},"weighted-only share",round(V(set())/tot,2))
print("ρ(EPDMS,DAC)=%.2f"%spearmanr([tab[m]["pdms"] for m in M],[tab[m]["drivable_area_compliance"] for m in M])[0])
