"""10 例逐场景的 F 轴 / I 轴，以及"用这两根轴预测本例表现"的一致性检验。
每个模型每个场景三条规划：clean（原图）/ rm（抹掉目标行人）/ night（全图夜化），公共窗口 0–2.5 s。
  F 轴 D_ped = mean_t‖clean−rm‖        行人在不在，对这条规划的影响
  I 轴 D_I   = mean_t‖clean−night‖     天色变了，对这条规划的影响
  CFR_case = D_ped / D_I               <1 = 这一例里光照比行人更能改变它
结果层读数（对行人真实未来，0–2.5 s）：最小净间隙 clr、最小 TTC；三种输入各算一遍。
两条用轴做的预测，逐例判定：
  P-F：若 D_ped < 0.5 m（行人几乎没改变规划）→ 预测"把行人删掉，结果层也几乎不变"：|Δclr| < 0.5 m 且不改变撞/不撞
  P-I：若 CFR_case < 1（光照影响更大）→ 预测"夜化对结果层的影响 ≥ 移除行人的影响"：|Δclr(night)| ≥ |Δclr(rm)|
另报每例的 F/I 与该模型左舵平均值的偏差（倍数），用于分析左右舵测量偏差。"""
import json,os,numpy as np
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
# 轴的来源：优先用右舵全量 rhd_axes_<m>（254 场景，含这 10 例），没有再退回单跑的 case10_axes_<m>
AX={}
for m in M:
    d={}
    for p in (f"{V5}/case10_axes_{m}.json", f"{V5}/rhd_axes_{m}.json"):
        if os.path.exists(p): d.update(json.load(open(p)))
    if d: AX[m]=d
PF=json.load(open(f"{V5}/nv_ped_future.json")); C=json.load(open(f"{V5}/case10.json"))
DET=json.load(open(f"{V5}/navsim_det_validate.json"))
PL=json.load(open(f"{V5}/profile_LHD.json"))
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def outcome(w,F):
    X=lin(T8,np.vstack([[0,0],np.asarray(w)[:,:2]])); c=np.linalg.norm(X-F,axis=1)-1.4
    if c.min()<=0: return float(c.min()),0.0
    cl=-np.gradient(c,TT); ok=cl>0.05
    return float(c.min()),(float(min(5,np.min(c[ok]/cl[ok]))) if ok.any() else 5.0)
def D(a,b):
    A=lin(T8,np.vstack([[0,0],np.asarray(a)[:,:2]])); B=lin(T8,np.vstack([[0,0],np.asarray(b)[:,:2]]))
    return float(np.mean(np.linalg.norm(A-B,axis=1)))
def arc(w): w=np.vstack([[0,0],np.asarray(w)[:,:2]]); return float(np.linalg.norm(np.diff(w,axis=0),axis=1).sum())
res=[]; pf=[0,0]; pi=[0,0]; pf2=[0,0]; pi2=[0,0]
for r in C:
    t=r["token"]; p=PF[t]; F=lin(T6,np.vstack([p["p0"],np.asarray(p["fut"][:5])]))
    row={"token":t,"v0":p["v0"],"waiting":r["waiting"],"det":DET.get(t,{}),"models":{}}
    for m in M:
        a=AX.get(m,{}).get(t)
        if not a or "rm" not in a: continue
        dped=D(a["clean"],a["rm"]); di=D(a["clean"],a["night"])
        c0,t0=outcome(a["clean"],F); c1,t1=outcome(a["rm"],F); c2,t2=outcome(a["night"],F)
        d=dict(D_ped=dped,D_I=di,cfr=dped/di if di>1e-6 else None,
               arc_clean=arc(a["clean"]),d_arc_rm=arc(a["rm"])-arc(a["clean"]),d_arc_night=arc(a["night"])-arc(a["clean"]),
               clr=c0,clr_rm=c1,clr_night=c2,ttc=t0,ttc_rm=t1,ttc_night=t2)
        # P-F（双向）：F<0.5 ⇒ 预测"移除后结果不变"；F≥0.5 ⇒ 预测"移除后结果会变"
        chg=(abs(c1-c0)>=0.5) or ((c0<0)!=(c1<0))
        d["PF_dir"]="same" if dped<0.5 else "change"
        ok=(not chg) if dped<0.5 else chg
        d["PF"]=bool(ok)
        (pf if dped<0.5 else pf2)[0]+=ok; (pf if dped<0.5 else pf2)[1]+=1
        # P-I（双向）：CFR<1 ⇒ 夜化影响 ≥ 移除影响；CFR≥1 ⇒ 移除影响 ≥ 夜化影响
        if d["cfr"] is not None:
            lo=d["cfr"]<1; d["PI_dir"]="night" if lo else "removal"
            ok=(abs(c2-c0)>=abs(c1-c0)) if lo else (abs(c1-c0)>=abs(c2-c0))
            d["PI"]=bool(ok); (pi if lo else pi2)[0]+=ok; (pi if lo else pi2)[1]+=1
        L=PL.get(m,{}).get("point",{})
        d["cfr_LHD"]=L.get("CFR"); d["cfr_ratio"]=(d["cfr"]/L["CFR"]) if (d["cfr"] and L.get("CFR")) else None
        row["models"][m]=d
    res.append(row)
json.dump({"cases":res,"PF":pf,"PI":pi,"PFrev":pf2,"PIrev":pi2},open(f"{V5}/case10_axes.json","w"),indent=1)
NM={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo1.5"}
for i,row in enumerate(res):
    print(f"\n#{i+1} {row['token'][:8]} v0={row['v0']:.1f}{' 真人在等' if row['waiting'] else ''}  移除{'有残留' if row['det'].get('rm_iou',0)>=0.5 else '干净'}，夜化后行人{'仍可见' if row['det'].get('night_iou',0)>=0.5 else '检不到'}")
    for m,d in row["models"].items():
        print(f"   {NM[m]:11s} F={d['D_ped']:.2f} I={d['D_I']:.2f} CFR={(d['cfr'] if d['cfr'] is not None else float('nan')):.2f}（左舵均值 {d['cfr_LHD'] if d['cfr_LHD'] is not None else float('nan'):.2f}，{d['cfr_ratio'] if d['cfr_ratio'] is not None else float('nan'):.1f}×） | 间隙 原{d['clr']:+.1f} 移除{d['clr_rm']:+.1f} 夜{d['clr_night']:+.1f} | TTC {d['ttc']:.1f}/{d['ttc_rm']:.1f}/{d['ttc_night']:.1f} | Δ弧长 移除{d['d_arc_rm']:+.2f} 夜{d['d_arc_night']:+.2f}"
          +f" | P-F[{d['PF_dir']}] {'✓' if d.get('PF') else '✗'}"+(f" P-I[{d['PI_dir']}] {'✓' if d.get('PI') else '✗'}" if 'PI_dir' in d else ""))
print(f"\nP-F 正向（F<0.5 ⇒ 结果不变）: {pf[0]}/{pf[1]}   反向（F≥0.5 ⇒ 结果会变）: {pf2[0]}/{pf2[1]}")
print(f"P-I 正向（CFR<1 ⇒ 夜化影响≥移除）: {pi[0]}/{pi[1]}   反向（CFR≥1 ⇒ 移除影响≥夜化）: {pi2[0]}/{pi2[1]}")
print(f"合计每格都有预测: P-F {pf[0]+pf2[0]}/{pf[1]+pf2[1]}   P-I {pi[0]+pi2[0]}/{pi[1]+pi2[1]}")
