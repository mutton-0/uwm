"""NAVSIM 右舵 246 场景上，黄昏(D)/夜间(N) 重打光对规划的影响：
planned mean speed 的相对变化、对行人真实未来的间距变化 ΔS，与表 II 的 dusk/night CFR 同一批场景。
只取自车在动的场景（原规划 2.5 s 弧长 ≥ 2.5 m，即均速 ≥ 1 m/s），与 nuScenes 的 v≥1 规则一致。
输出 relight_navsim.json。"""
import json,sys,os,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import fi_panels as fp
from scipy.stats import wilcoxon
def S_of(X,ped): return float(np.mean(np.minimum(np.linalg.norm(X-ped,axis=1)-1.4,10.0)))
def arc(X): return float(np.sum(np.linalg.norm(np.diff(X,axis=0),axis=1)))
out={}
for m in fp.M:
    d=fp.AX[m]; rows=[]
    for t,v in d.items():
        if not all(k in v for k in ("clean","dusk","night")) or t not in fp.PF: continue
        try: ped=fp.ped_xy(t)
        except Exception: continue
        C=fp.lin(v["clean"]); a0=arc(C)
        if a0<2.5: continue
        r={"tok":t,"v0":a0/2.5}
        for k in ("dusk","night"):
            X=fp.lin(v[k]); r["dv_"+k]=(arc(X)-a0)/2.5; r["dS_"+k]=S_of(X,ped)-S_of(C,ped)
        rows.append(r)
    o={"n":len(rows),"v0":float(np.mean([r["v0"] for r in rows]))}
    for k in ("dusk","night"):
        dv=np.array([r["dv_"+k] for r in rows]); dS=np.array([r["dS_"+k] for r in rows])
        o[k]=dict(dv=float(dv.mean()),dv_rel=float(dv.mean()/o["v0"]),p=float(wilcoxon(dv)[1]) if len(dv)>=10 else 1.0,
                  dS=float(dS.mean()),dS_p=float(wilcoxon(dS)[1]) if len(dS)>=10 else 1.0)
    out[m]=o
    print(f"{m:11s} n={o['n']:3d} v0={o['v0']:.2f}  dusk: dv {100*o['dusk']['dv_rel']:+.0f}% (p={o['dusk']['p']:.0e}) dS {o['dusk']['dS']:+.2f} (p={o['dusk']['dS_p']:.0e}) | night: dv {100*o['night']['dv_rel']:+.0f}% (p={o['night']['p']:.0e}) dS {o['night']['dS']:+.2f} (p={o['night']['dS_p']:.0e})")
json.dump(out,open(f"{fp.V5}/relight_navsim.json","w"),indent=1)
