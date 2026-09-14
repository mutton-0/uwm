"""黄昏 vs 夜化的光照混淆比 CFR（NAVSIM 右舵语料）。
F  = mean_t‖clean − rm‖，I = mean_t‖clean − night|dusk‖，0–2.5 s 重采样到 0.1 s；CFR = mean F / mean I。
筛选与 domain_transfer.py 完全一致：行人真实未来前 5 步必须完整，且取六家共有的 token，
这样重算的 CFR_night 应当复现 tab_axes 的 "Sing." 列，黄昏才有可比性。
两条光照支路来自 rhd_axes4_<m>.json（clean/rm/night/dusk 同进程同图源），
夜化列同时给出四条件重跑值与表中已发布值，便于核对。
黄昏的意义：它几乎不损行人可检出性（88.9% vs 夜化 66.7%，见 dusk_check.json），
所以若 CFR_dusk 仍 < 1，就不能用"夜化把行人一起抹掉了"来解释。输出 dusk_cfr.json。"""
import json,os,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME=dict(dd="DiffusionDrive",ltf="LTF",ddv2="DiffusionDriveV2",simlingo="SimLingo",
          autovla="AutoVLA",alpamayo15="Alpamayo-1.5")
PUB=dict(dd=0.24,ltf=0.19,ddv2=0.52,simlingo=0.19,autovla=0.46,alpamayo15=0.26)  # tab_axes 已发布的 Sing. 列
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]
def lin(P):
    P=np.vstack([[0,0],np.asarray(P,float)[:,:2]])
    return np.stack([np.interp(TT,T8,P[:,0]),np.interp(TT,T8,P[:,1])],1)
def D(a,b): return float(np.mean(np.linalg.norm(lin(a)-lin(b),axis=1)))
AX={m:json.load(open(f"{V5}/rhd_axes4_{m}.json")) for m in M}
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def ok(t,v):
    p=PF.get(t)
    return (p is not None and not any(z is None for z in p["fut"][:5])
            and all(k in v for k in ("clean","rm","night","dusk")))
COM=set.intersection(*[{t for t,v in d.items() if ok(t,v)} for d in AX.values()])
res={}
print(f"六家共有且行人未来完整的场景: {len(COM)}")
print(f"{'模型':17s} {'F':>6s} {'I_夜':>6s} {'I_黄':>6s} {'CFR_夜':>7s} {'已发布':>6s} {'CFR_黄':>7s}")
for m in M:
    d=AX[m]; toks=sorted(COM)
    F=np.array([D(d[t]["clean"],d[t]["rm"])    for t in toks])
    IN=np.array([D(d[t]["clean"],d[t]["night"])for t in toks])
    ID=np.array([D(d[t]["clean"],d[t]["dusk"]) for t in toks])
    r=dict(n=len(toks),F=float(F.mean()),I_night=float(IN.mean()),I_dusk=float(ID.mean()),
           cfr_night=float(F.mean()/IN.mean()),cfr_dusk=float(F.mean()/ID.mean()),
           cfr_night_published=PUB[m],
           frac_F_gt_I_night=float(np.mean(F>IN)),frac_F_gt_I_dusk=float(np.mean(F>ID)))
    res[m]=r
    print(f"  {NAME[m]:15s} {r['F']:6.3f} {r['I_night']:6.3f} {r['I_dusk']:6.3f} "
          f"{r['cfr_night']:7.2f} {PUB[m]:6.2f} {r['cfr_dusk']:7.2f}")
lo=min(r['cfr_dusk'] for r in res.values()); hi=max(r['cfr_dusk'] for r in res.values())
print(f"\nCFR_黄昏 区间 {lo:.2f}--{hi:.2f}（全部 < 1）")
json.dump(res,open(f"{V5}/dusk_cfr.json","w"),indent=1)
