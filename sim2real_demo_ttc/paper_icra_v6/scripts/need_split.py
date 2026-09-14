"""检验"位移≈0 即忽视行人"这条结论是否被"行人本来就不挡路"混淆。
把右舵 246 个场景按【盲规划是否真会撞上行人】拆开，分别看模型的反应：
  need   : 盲规划（移除行人后的规划，rm）对行人真实未来的全程最小净间隙 < 阈值 —— 行人确实在碰撞路线上
  no-need: 盲规划本来就绕开了 —— 此时不动才是正确行为
每组报：F 中位、F>=0.5 m 的比例、ΔS=clr(clean)-clr(rm) 中位（看见行人换来多少间隙）、
以及规划弧长变化 clean-rm（负值=看见行人后计划少走，即减速意愿）。
净间隙已扣掉自车半宽 1.0 与行人半径 0.4。数据：rhd_axes4_<m>.json + nv_ped_future.json。"""
import json,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME=dict(dd="DiffusionDrive",ltf="LTF",ddv2="DiffusionDriveV2",simlingo="SimLingo",
          autovla="AutoVLA",alpamayo15="Alpamayo-1.5")
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def path(w): return lin(T8,np.vstack([[0,0],np.asarray(w,float)[:,:2]]))
def arc(w):
    X=path(w); return float(np.sum(np.linalg.norm(np.diff(X,axis=0),axis=1)))
AX={m:json.load(open(f"{V5}/rhd_axes4_{m}.json")) for m in M}
def okt(t,v):
    p=PF.get(t)
    return p is not None and not any(z is None for z in p["fut"][:5]) and all(k in v for k in ("clean","rm"))
COM=sorted(set.intersection(*[{t for t,v in d.items() if okt(t,v)} for d in AX.values()]))
FUT={t:lin(T6,np.vstack([PF[t]["p0"],np.asarray(PF[t]["fut"][:5])])) for t in COM}
def clr(w,t): return float(np.min(np.linalg.norm(path(w)-FUT[t],axis=1))-1.4)
for THR,lab in ((0.0,"盲规划会接触 (clr_rm < 0)"),(1.0,"盲规划进到 1 m 内 (clr_rm < 1)")):
    print(f"\n══════ 判据：{lab}   共有场景 {len(COM)} ══════")
    print(f"{'模型':17s} | {'n_need':>6s} {'F中位':>6s} {'F>=.5':>6s} {'ΔS中位':>7s} {'ΔS>=.5':>7s} {'弧长Δ':>7s} | {'n_ok':>5s} {'F中位':>6s} {'F>=.5':>6s}")
    for m in M:
        d=AX[m]; need=[];ok=[]
        for t in COM:
            v=d[t]; cr=clr(v["rm"],t)
            (need if cr<THR else ok).append((t,v,cr))
        def st(rows,full=False):
            if not rows: return None
            F=np.array([float(np.mean(np.linalg.norm(path(v["clean"])-path(v["rm"]),axis=1))) for _,v,_ in rows])
            if not full: return len(rows),np.median(F),100*np.mean(F>=0.5)
            dS=np.array([clr(v["clean"],t)-cr for t,v,cr in rows])
            dA=np.array([arc(v["clean"])-arc(v["rm"]) for _,v,_ in rows])
            return len(rows),np.median(F),100*np.mean(F>=0.5),np.median(dS),100*np.mean(dS>=0.5),np.median(dA)
        a=st(need,True); b=st(ok)
        A=f"{a[0]:6d} {a[1]:6.2f} {a[2]:5.0f}% {a[3]:+7.2f} {a[4]:6.0f}% {a[5]:+7.2f}" if a else f"{'--':>6s}"+" "*34
        B=f"{b[0]:5d} {b[1]:6.2f} {b[2]:5.0f}%" if b else " "*19
        print(f"  {NAME[m]:15s} | {A} | {B}")
