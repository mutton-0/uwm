"""人类参照：右舵 246 个场景里，人类司机自己的 2.5 s 实际轨迹，与各策略的规划对照。
沿 sample_next 链出后续帧，用 ego2global 变回 t0 自车系（x 前 y 左），与行人真实未来同系。
报三组量（全体 / 盲规划会进 1 m 的 need 子集）：
  clr    = 全程最小净间隙（扣自车半宽 1.0 + 行人半径 0.4）—— 人类留了多少，策略留了多少
  arc    = 2.5 s 走过/计划走过的弧长
  decel  = 末速 - 初速（负=减速）；人类取链末帧的 ego_dynamic_state[0]，策略取规划末段速度
最后做一次**与策略无关**的配对比较：危险集按人类自己的最小净间隙定义（<2 m / <1.5 m），
六家在同一批场景上与人类比间隙，并报"比人类更近"的场景占比。
结论（2026-09-14）：该语料 65% 自车停着、人类中位弧长 0 m，只有 12/246 个场景人类真的靠近过行人；
在这 12 个场景上五家与人类持平（比人类近的占 42--58%，即掷硬币），Alpamayo 反而更远（33%），
只有 SimLingo 明显更差（92%，间隙 0.33 m）。故"策略反应不如人类激烈"这一说法在本语料上不成立。
输出 human_ref_rhd.json"""
import json,glob,pickle,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
PF=json.load(open(f"{V5}/nv_ped_future.json"))
AX={m:json.load(open(f"{V5}/rhd_axes4_{m}.json")) for m in M}
ok=lambda t,v: PF.get(t) and not any(z is None for z in PF[t]["fut"][:5]) and "rm" in v
COM=sorted(set.intersection(*[{t for t,v in d.items() if ok(t,v)} for d in AX.values()]))
SC={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())): SC[s["token"]]=s
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def path(w): return lin(T8,np.vstack([[0,0],np.asarray(w,float)[:,:2]]))
def arc(X): return float(np.sum(np.linalg.norm(np.diff(X,axis=0),axis=1)))
def human(t):
    """沿 sample_next 取 5 个后续帧（NAVSIM 2 Hz ⇒ 2.5 s），变到 t0 自车系"""
    s=SC.get(t)
    if s is None: return None,None
    E0=np.asarray(s["ego2global"],float); inv=np.linalg.inv(E0)
    pts=[[0.0,0.0]]; cur=s; v_end=float(s["ego_dynamic_state"][0])
    for _ in range(5):
        nx=cur.get("sample_next")
        if not nx or nx not in SC: return None,None          # 链不满 2.5 s 的丢掉
        cur=SC[nx]; g=np.asarray(cur["ego2global"],float)
        p=(inv@g)[:3,3]; pts.append([float(p[0]),float(p[1])]); v_end=float(cur["ego_dynamic_state"][0])
    return lin(T6,np.asarray(pts)), v_end-float(s["ego_dynamic_state"][0])
FUT={t:lin(T6,np.vstack([PF[t]["p0"],np.asarray(PF[t]["fut"][:5])])) for t in COM}
def clr(X,t): return float(np.min(np.linalg.norm(X-FUT[t],axis=1))-1.4)
H={}
for t in COM:
    X,dv=human(t)
    if X is not None: H[t]=(X,dv)
print(f"能链出完整 2.5 s 人类轨迹的场景: {len(H)} / {len(COM)}")
def endspeed(w):
    X=path(w); return float(np.linalg.norm(X[-1]-X[-3])/0.2)-float(np.linalg.norm(X[2]-X[0])/0.2)
for lab,sel in (("全部",lambda t,m:True),
                ("need：盲规划进 1 m 内",lambda t,m:clr(path(AX[m][t]["rm"]),t)<1.0)):
    print(f"\n══ {lab} ══")
    print(f"{'':16s} {'n':>4s} {'clr中位':>7s} {'arc中位':>7s} {'Δv中位':>7s}")
    hs=[t for t in H if sel(t,'dd')]
    print(f"  {'人类（参照）':14s} {len(hs):4d} {np.median([clr(H[t][0],t) for t in hs]):7.2f} "
          f"{np.median([arc(H[t][0]) for t in hs]):7.2f} {np.median([H[t][1] for t in hs]):+7.2f}")
    for m in M:
        S=[t for t in H if sel(t,m)]
        if not S: print(f"  {m:14s} {0:4d}"); continue
        print(f"  {m:14s} {len(S):4d} {np.median([clr(path(AX[m][t]['clean']),t) for t in S]):7.2f} "
              f"{np.median([arc(path(AX[m][t]['clean'])) for t in S]):7.2f} "
              f"{np.median([endspeed(AX[m][t]['clean']) for t in S]):+7.2f}")
HC={t:clr(H[t][0],t) for t in H}
for THR in (2.0,1.5):
    S=[t for t in H if HC[t]<THR]
    print(f"\n══ 与策略无关的危险集：人类自己最小净间隙 < {THR} m   n={len(S)} ══")
    print(f"{'':16s} {'clr中位':>7s} {'比人类近':>8s} {'arc中位':>7s}")
    print(f"  {'人类（参照）':14s} {np.median([HC[t] for t in S]):7.2f} {'--':>8s} {np.median([arc(H[t][0]) for t in S]):7.2f}")
    for m in M:
        c=np.array([clr(path(AX[m][t]['clean']),t) for t in S]); h=np.array([HC[t] for t in S])
        a=np.array([arc(path(AX[m][t]['clean'])) for t in S])
        print(f"  {m:14s} {np.median(c):7.2f} {100*np.mean(c<h):7.0f}% {np.median(a):7.2f}")
json.dump({t:{"clr":HC[t],"arc":arc(H[t][0]),"dv":H[t][1]} for t in H},
          open(f"{V5}/human_ref_rhd.json","w"),indent=1)
