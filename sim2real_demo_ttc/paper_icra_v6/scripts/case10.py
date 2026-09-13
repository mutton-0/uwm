"""NAVSIM 新加坡近行人场景，原图直接跑（不编辑），抽 10 个逐个对照左舵诊断的预测。
抽样：固定种子，8 个自车在走（v0≥2 m/s、行人未来 4 s 完整）+ 2 个真人停着等（v0<0.5）。
逐模型读数（0–4 s，NAVSIM 规划 8×0.5 s；行人用日志真实未来）：
  exposure_i = 规划 4 s 路程 / 真人 4 s 路程（真人停着时直接报规划路程）
  minclr = 整条规划与行人的最小净间隙（−1.4 m 车半宽+行人半径；<0 = 碰撞）；minTTC（0–2.5 s）
左舵诊断给出的预测（profile_LHD.json + 起步检查）：
  仓位：DD、LTF 小（<0.8）；DDv2、AutoVLA、Alpamayo 接近人类（0.8–1.3）；SimLingo 大（>1.5）
  风控：六家都不随行人调整 → 行人场景里的仓位与其一般仓位一致、离行人最近的是仓位最大的
  耐心：按左舵 depart_when_waiting.json 的起步率给每家一个预测——>50% 预测"会起步"，否则预测"不会起步"
        （DDv2 81%、LTF 45%、SimLingo 15%、AutoVLA 9%、DD 2.8%、Alpamayo 1.5 0%）"""
import json,glob,pickle,numpy as np
import os as _os
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SUF={m:("_sg-one-north_closevru_nav" if _os.path.exists(f"{R5}/nvtraj_{m}_sg-one-north_closevru_nav.json") else "_sg-one-north_closevru") for m in M}
TR={m:json.load(open(f"{R5}/nvtraj_{m}{SUF[m]}.json")) for m in M}
PF=json.load(open(f"{V5}/nv_ped_future.json"))
TOK={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())): TOK[s["token"]]=s
def human(t):
    s=TOK[t]; inv=np.linalg.inv(np.asarray(s["ego2global"])); s1=s; out=[]
    for k in range(8):
        n=s1.get("sample_next"); s1=TOK.get(n) if n else None
        if s1 is None: return None
        out.append((inv@np.asarray(s1["ego2global"])[:,3])[:2])
    return np.array(out)
TT=np.round(np.arange(0,4.01,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]
def lin(P): P=np.asarray(P,float); return np.stack([np.interp(TT,T8,P[:,0]),np.interp(TT,T8,P[:,1])],1)
def arc(w): w=np.vstack([[0,0],np.asarray(w)[:,:2]]); return float(np.linalg.norm(np.diff(w,axis=0),axis=1).sum())
cands=[t for t in json.load(open("/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad/sg_closevru_tok.json"))["tokens"]
       if t in PF and all(z is not None for z in PF[t]["fut"]) and all(t in TR[m] for m in M) and human(t) is not None]
# 只保留目标 VRU 在前视相机里可见、因而能做"移除"反事实的场景：
# 260 个近行人场景里有 6 个目标只出现在侧视（如 028613e1 的自行车只在 CAM_R0），
# 这些场景拿不到 F 轴，逐例预测会整例空着，故排除。
RMOK={t for t,v in json.load(open(f"{V5}/navsim_rm_meta.json")).items() if "err" not in v}
cands=[t for t in cands if t in RMOK]
mov=[t for t in cands if PF[t]["v0"]>=2.0]; wait=[t for t in cands if PF[t]["v0"]<0.5]
rng=np.random.default_rng(2026)
KEEP=f"{V5}/case10_tokens.json"
if _os.path.exists(KEEP): pick=json.load(open(KEEP))          # 固定下来的 10 例，保证图和表对得上
else:
    prev=[r["token"] for r in json.load(open(f"{V5}/case10.json"))] if _os.path.exists(f"{V5}/case10.json") else []
    keep=[t for t in prev if t in cands]                       # 原来合格的按原顺序留下
    nm=8-sum(1 for t in keep if PF[t]["v0"]>=2.0); nw=2-sum(1 for t in keep if PF[t]["v0"]<0.5)
    pool_m=[t for t in mov if t not in keep]; pool_w=[t for t in wait if t not in keep]
    pick=keep+list(rng.choice(pool_m,nm,replace=False))+list(rng.choice(pool_w,nw,replace=False))
    json.dump(list(pick),open(KEEP,"w"),indent=1)
PRED_EXP={"dd":"小","ltf":"小","ddv2":"人类级","simlingo":"大","autovla":"人类级","alpamayo15":"人类级"}
DEP=json.load(open(f"{V5}/depart_when_waiting.json"))
PAT={m:("会起步" if DEP[m]["depart"]>50 else "不会起步") for m in M}   # 左舵起步率 → 逐家预测，不留空
def cat(x): return "小" if x<0.8 else ("大" if x>1.5 else "人类级")
out=[]
for t in pick:
    p=PF[t]; h=human(t); F=lin(np.vstack([p["p0"],np.asarray(p["fut"])])); ha=arc(h)
    rec={"token":t,"v0":p["v0"],"ped":p["cat"],"ped_x":p["p0"][0],"ped_y":p["p0"][1],"human_arc":ha,"waiting":p["v0"]<0.5,"models":{}}
    for m in M:
        w=np.asarray(TR[m][t])[:,:2]; X=lin(np.vstack([[0,0],w])); c=np.linalg.norm(X-F,axis=1)-1.4
        cl=-np.gradient(c,TT); ok=(cl>0.05)&(TT<=2.5); ttc=0.0 if c.min()<=0 else (float(min(5,np.min(c[ok]/cl[ok]))) if ok.any() else 5.0)
        e=arc(w)/ha if ha>2 else None
        rec["models"][m]=dict(plan_arc=arc(w),exposure=e,minclr=float(c.min()),ttc=ttc)
    out.append(rec)
json.dump(out,open(f"{V5}/case10.json","w"),indent=1)
for i,r in enumerate(out):
    print(f"\n#{i+1} {r['token']}  自车 {r['v0']:.1f} m/s  {r['ped']} 在前方 {r['ped_x']:.1f} m、横向 {r['ped_y']:+.1f} m  真人 4 s 走 {r['human_arc']:.1f} m {'（真人在等）' if r['waiting'] else ''}")
    for m in M:
        d=r["models"][m]; e=d["exposure"]
        if r["waiting"]:
            pred=PAT[m]; act="起步" if d["plan_arc"]>2 else "停着"
            verdict="" if pred=="—" else ("✓" if (pred=="会起步")==(act=="起步") else "✗")
            print(f"   {m:9s} 规划 {d['plan_arc']:5.1f} m → {act:3s}  预测 {pred:4s} {verdict}  最小间隙 {d['minclr']:+.1f} m")
        else:
            v=cat(e); verdict="✓" if v==PRED_EXP[m] else "✗"
            print(f"   {m:9s} 规划/真人 {e:4.2f}（{v}，预测{PRED_EXP[m]} {verdict}）  最小间隙 {d['minclr']:+5.1f} m  TTC {d['ttc']:.1f}s {'撞' if d['minclr']<0 else ''}")

# ---- 逐例预测命中统计（每个模型每一例都有预测，不留空）  TALLY
import collections
hit=collections.Counter(); tot=collections.Counter()
for r in out:
    for m,d in r["models"].items():
        if r["waiting"]:
            ok=(PAT[m]=="会起步")==(d["plan_arc"]>2)
        else:
            ok=(cat(d["exposure"])==PRED_EXP[m]) if d["exposure"] is not None else None
        if ok is None: continue
        hit[m]+=ok; tot[m]+=1
print("\n逐例预测命中（仓位档位 8 例 + 耐心 2 例）:")
for m in M: print(f"  {m:11s} {hit[m]}/{tot[m]}")
print(f"  合计 {sum(hit.values())}/{sum(tot.values())}")
json.dump({"hit":sum(hit.values()),"tot":sum(tot.values()),"per":{m:[hit[m],tot[m]] for m in M}},
          open(f"{V5}/case10_tally.json","w"),indent=1)
