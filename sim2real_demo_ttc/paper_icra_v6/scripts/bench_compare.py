"""我们的体检 vs 两套标准评测，同六个模型：
 (1) nuScenes 开环（UniAD/VAD 口径）在我们同一批近行人帧上：L2、前方碰撞率（不计追尾——日志车不反应，慢了就被追尾）、行人碰撞率；原图 vs 移除
 (2) NAVSIM EPDMS 在新加坡"行人很近"场景（走廊净间隙 ≤1 m、纵向 ≤20 m，260 个）与全体 783 场景
一致性：跨模型 Spearman；综合性：标准指标对"移除行人"的敏感度、EPDMS 分差的 Shapley 来源。
SimLingo 的 NAVSIM 用修正后的轨迹（0.25 s 路点正确重采样）。"""
import json,itertools,math,numpy as np
from scipy.stats import spearmanr
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NU=json.load(open(f"{V5}/nusc_openloop.json")); PR=json.load(open(f"{V5}/profile_ALL.json"))
rows=json.load(open(f"{V5}/diag_units.json"))
VRU=json.load(open(f"{R5}/nv_sg_vru_brake.json"))
import os as _os
def nvfile(m,tag):
    cands=[f"{R5}/nvscore_{m}_sg-one-north{tag}_nav.json",f"{R5}/nvscore_{m}_sg-one-north{tag}.json"] if tag else \
          [f"{R5}/nvscore_{m}_sg-one-north_nav.json",f"{R5}/nvscore_{m}_sg-one-north_fix.json",f"{R5}/nvscore_{m}_sg-one-north.json"]
    for f in cands:
        if _os.path.exists(f): return f
F=["no_at_fault_collisions","drivable_area_compliance","driving_direction_compliance","traffic_light_compliance"]
def score(r,neut=()):
    mult=np.prod([1.0 if f in neut else r[f] for f in F]); w=(5*r["ego_progress"]+5*r["time_to_collision_within_bound"]+2*r["lane_keeping"]+2*r["history_comfort"])/14
    return mult*w
def navsim(tag,moving=None):
    D={m:json.load(open(nvfile(m,tag))) for m in M}
    R={m:{r["token"]:r for r in D[m]["rows"]} for m in M}
    toks=sorted(set.intersection(*[set(v) for v in R.values()]))
    if moving is not None: toks=[t for t in toks if t in VRU and ((VRU[t]["v0"]>=1.0)==moving)]
    out={}
    for m in M:
        rr=[R[m][t] for t in toks]; ped={h["token"] for h in D[m]["hits"] if "PEDESTRIAN" in h["type"]}
        out[m]=dict(n=len(toks),epdms=float(np.mean([r["pdm_score"] for r in rr])),**{k:float(np.mean([r[k] for r in rr])) for k in F+["ego_progress","time_to_collision_within_bound"]},
                    ped_col=float(100*np.mean([t in ped for t in toks])))
    # Shapley：四个乘法门对模型间方差的贡献
    G={"NC":[F[0]],"DAC":[F[1]],"DDC":[F[2]],"TLC":[F[3]]}; cache={}
    def V(act):
        k=frozenset(act)
        if k not in cache:
            neut=tuple(f for g,fs in G.items() if g not in act for f in fs)
            cache[k]=float(np.var([np.mean([score(R[m][t],neut) for t in toks]) for m in M]))
        return cache[k]
    sh={}
    for g in G:
        o=[h for h in G if h!=g]; sh[g]=0.0
        for k in range(len(o)+1):
            for S_ in itertools.combinations(o,k):
                sh[g]+=math.factorial(k)*math.factorial(len(G)-k-1)/math.factorial(len(G))*(V(set(S_)|{g})-V(set(S_)))
    tot=V(set(G)); return out,{g:(x/tot if tot>0 else None) for g,x in sh.items()},len(toks)
ALL,shA,nA=navsim(""); CV,shC,nC=navsim("_closevru"); CVm,shCm,nCm=navsim("_closevru",moving=True); CVs,_,nCs=navsim("_closevru",moving=False)
ours={m:PR[m]["point"] for m in M}
def coll(m,sv="actual"):
    U=[z for z in rows if z["m"]==m and z["sv"]==sv and z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr")]
    return 100*np.mean([z["P"]["A"]==0 for z in U])
res={"nusc":NU,"navsim_all":ALL,"navsim_close":CV,"navsim_close_moving":CVm,"navsim_close_stopped":CVs,
     "shapley_all":shA,"shapley_close":shC,"shapley_close_moving":shCm,"n":{"all":nA,"close":nC,"close_moving":nCm,"close_stopped":nCs}}
v=lambda d,k: [d[m][k] for m in M]
pairs={
 "exposure~nusc_L2avg":(v(ours,"exposure"),[NU[m]["clean"]["L2_avg"] for m in M]),
 "|exposure-1|~nusc_L2avg":([abs(ours[m]["exposure"]-1) for m in M],[NU[m]["clean"]["L2_avg"] for m in M]),
 "exposure~nusc_frontcol":(v(ours,"exposure"),[NU[m]["clean"]["col_front"] for m in M]),
 "our_ped_coll~nusc_pedcol":([coll(m) for m in M],[NU[m]["clean"]["col_ped"] for m in M]),
 "our_ped_coll~navsim_close_pedcol":([coll(m) for m in M],v(CV,"ped_col")),
 "exposure~navsim_close_EP":(v(ours,"exposure"),v(CV,"ego_progress")),
 "HS~navsim_close_NC":(v(ours,"HS"),v(CV,"no_at_fault_collisions")),
 "HS~navsim_close_EPDMS":(v(ours,"HS"),v(CV,"epdms")),
 "SP~navsim_close_EPDMS":(v(ours,"SP"),v(CV,"epdms")),
 "navsim_all_EPDMS~navsim_close_EPDMS":(v(ALL,"epdms"),v(CV,"epdms")),
 "navsim_close_EPDMS~close_DAC":(v(CV,"epdms"),v(CV,"drivable_area_compliance")),
 "navsim_close_EPDMS~close_NC":(v(CV,"epdms"),v(CV,"no_at_fault_collisions")),
}
res["spearman"]={k:float(spearmanr(a,b)[0]) for k,(a,b) in pairs.items()}
res["removal_sensitivity"]={m:dict(dL2=NU[m]["rm"]["L2_avg"]-NU[m]["clean"]["L2_avg"],dcol_front=NU[m]["rm"]["col_front"]-NU[m]["clean"]["col_front"],
                                   dcol_ped=NU[m]["rm"]["col_ped"]-NU[m]["clean"]["col_ped"]) for m in M}
json.dump(res,open(f"{V5}/bench_compare.json","w"),indent=1)
print("n:",res["n"]); print("Shapley 全体",{k:round(x,2) for k,x in shA.items()},"近行人",{k:(None if x is None else round(x,2)) for k,x in shC.items()})
for m in M:
    a=ALL[m]; c=CV[m]; cm=CVm[m]; nu=NU[m]
    print(f"{m:9s} NAVSIM 全体 {a['epdms']:.3f} | 近行人 {c['epdms']:.3f} (NC {c['no_at_fault_collisions']:.3f} DAC {c['drivable_area_compliance']:.3f} EP {c['ego_progress']:.2f} 撞行人 {c['ped_col']:.1f}%) | 近行人且在走 {cm['epdms']:.3f} "
          f"| nuScenes L2 {nu['clean']['L2_avg']:.2f}/{nu['rm']['L2_avg']:.2f} 前方碰撞 {nu['clean']['col_front']:.1f}/{nu['rm']['col_front']:.1f} 撞行人 {nu['clean']['col_ped']:.1f}/{nu['rm']['col_ped']:.1f}")
for k,x in res["spearman"].items(): print(f"  ρ {k}: {x:+.2f}")
