"""反事实速度下的 need 集与避让率（NAVSIM 右舵语料）。
为什么要这个：该语料 65% 自车停着、人类中位 2.5 s 只走 0 m，盲规划仅 3.6% 的 cell 会进到行人 1 m 内，
所以「位移≈0」多半是正确行为而非忽视行人，原先在全体 cell 上报"真避让率"分母不对。
照 nuScenes 体检的做法注入自车速度把危险造出来，再只在 need 集上读数。
档位与 nuScenes 体检对齐到 {actual, 4, 8}（AutoVLA / Alpamayo 在 nuScenes 上只有 4 和 8 两档，
六家并排比较必须同一个单位池；已验证 SP 与 scaling 的排名对档位集合不敏感，ρ=+1.000）。
实际跑的是 {2,4,6,8} 四档，2 与 6 保留在 rhd_cf_*.json 里作稳健性备查，主结果不用。
need 判据与 §III 的 C(R)<1 一致：盲规划（rm）对行人真实未来的全程最小净间隙 < 1 m
（已扣自车半宽 1.0 + 行人半径 0.4）。
读数：F=mean_t‖clean−rm‖；ΔS=clr(clean)−clr(rm)；弧长变化=arc(clean)−arc(rm)（负=看见行人后计划少走，即减速）。
输入 rhd_cf_<model>.json（BEV 三家本机、VLA 三家 exx），输出 cf_need.json。"""
import json,os,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME=dict(dd="DiffusionDrive",ltf="LTF",ddv2="DiffusionDriveV2",simlingo="SimLingo",
          autovla="AutoVLA",alpamayo15="Alpamayo-1.5")
SPD=["4","8"]          # 与 nuScenes 对齐；四档原始数据仍在文件里
USE_ACTUAL=True        # actual 档从 rhd_axes4_<m>.json 取
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def path(w): return lin(T8,np.vstack([[0,0],np.asarray(w,float)[:,:2]]))
def arc(X): return float(np.sum(np.linalg.norm(np.diff(X,axis=0),axis=1)))
CF={m:json.load(open(f"{V5}/rhd_cf_{m}.json")) for m in M if os.path.exists(f"{V5}/rhd_cf_{m}.json")}
AX={m:json.load(open(f"{V5}/rhd_axes4_{m}.json")) for m in CF} if USE_ACTUAL else {}
if not CF: raise SystemExit("还没有任何 rhd_cf_*.json")
good=lambda v: all(s in v and all(k in v[s] for k in ("clean","rm")) for s in SPD)
COM=sorted(set.intersection(*[{t for t,v in d.items() if good(v)} for d in CF.values()])
           & {t for t in PF if not any(z is None for z in PF[t]["fut"][:5])}
           & (set.intersection(*[{t for t,v in d.items() if "rm" in v} for d in AX.values()]) if AX else set(PF)))
FUT={t:lin(T6,np.vstack([PF[t]["p0"],np.asarray(PF[t]["fut"][:5])])) for t in COM}
def clr(X,t): return float(np.min(np.linalg.norm(X-FUT[t],axis=1))-1.4)
res={}; tot_need=0; tot_real=0
print(f"已跑完 {list(CF)}   共有场景 {len(COM)}   速度档 {SPD}\n")
print(f"{'模型':17s} {'need cell':>9s} {'占比':>6s} | {'F中位':>6s} {'ΔS中位':>7s} {'ΔS>=0.5':>8s} {'弧长Δ中位':>9s} {'真避让':>6s}")
for m,d in CF.items():
    F=[];dS=[];dA=[];I=[]
    for t in COM:
        for v in [d[t][s] for s in SPD]+([AX[m][t]] if USE_ACTUAL else []):
            if clr(path(v["rm"]),t)>=1.0: continue              # 只留 need
            F.append(float(np.mean(np.linalg.norm(path(v["clean"])-path(v["rm"]),axis=1))))
            dS.append(clr(path(v["clean"]),t)-clr(path(v["rm"]),t))
            dA.append(arc(path(v["clean"]))-arc(path(v["rm"])))
            I.append(float(np.mean(np.linalg.norm(path(v["clean"])-path(v["night"]),axis=1))) if "night" in v else np.nan)
    F=np.array(F);dS=np.array(dS);dA=np.array(dA);I=np.array(I)
    real=(F>=0.5)&(dS>=0.5)&(np.isnan(I)|(F>I))
    n=len(F); tot=len(COM)*(len(SPD)+(1 if USE_ACTUAL else 0)); tot_need+=n; tot_real+=int(real.sum())
    res[m]=dict(n_cell=tot,n_need=n,need_pct=100*n/tot,F_med=float(np.median(F)),dS_med=float(np.median(dS)),
                dS_big=100*float(np.mean(dS>=0.5)),dArc_med=float(np.median(dA)),
                n_real=int(real.sum()),real_pct=100*float(real.mean()))
    r=res[m]
    print(f"  {NAME[m]:15s} {n:9d} {r['need_pct']:5.1f}% | {r['F_med']:6.2f} {r['dS_med']:+7.2f} "
          f"{r['dS_big']:7.0f}% {r['dArc_med']:+9.2f} {r['n_real']:4d}({r['real_pct']:.0f}%)")
res["_all"]=dict(n_scene=len(COM),n_speed=len(SPD)+(1 if USE_ACTUAL else 0),models=list(CF),
                 n_need=tot_need,n_real=tot_real,real_pct=100*tot_real/max(tot_need,1))
print(f"\n合计：need cell {tot_need} 个，其中真避让 {tot_real} 个（{100*tot_real/max(tot_need,1):.1f}%）")
json.dump(res,open(f"{V5}/cf_need.json","w"),indent=1)
