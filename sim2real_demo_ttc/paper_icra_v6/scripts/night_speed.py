"""夜化对规划的有符号影响（只用输入车速 ≥1 m/s 的近距单位，与低速规则一致）：
  规划均速 = 2.5 s 规划弧长 / 2.5 s（原图 vs 夜化），分 5 段看是否一路加速
  ΔS_night = 夜化后整条轨迹与行人的平均间隙变化（负 = 更近）
  纵向：夜化后规划末端前伸量与 ΔS 的秩相关；横向：末端往行人一侧偏 >0.1 m 的帧比例"""
import json,glob,numpy as np
from scipy.stats import wilcoxon,spearmanr
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
rows=json.load(open(f"{V5}/diag_units.json"))
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
def seg(w): w=np.vstack([[0,0],np.asarray(w)]); return np.linalg.norm(np.diff(w,axis=0),axis=1)/0.5
sel=lambda z: z["sv"]=="actual" and "fN" in z and z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0
out={}
for m in ["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]:
    C={**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")}; N={**load(f"{R5}/card_night_{m}.json"),**load(f"{R5}/card_night_{m}_*.json")}
    U=[z for z in rows if z["m"]==m and sel(z)]
    a=np.array([seg(C[z["uid"]]["actual"]["clean"]) for z in U]); b=np.array([seg(N[z["uid"]]["night"]) for z in U])
    d=b.mean(1)-a.mean(1); dS=np.array([z["fN"]["S"]-z["P"]["S"] for z in U])
    dx=np.array([np.asarray(z["dI"]).reshape(5,2)[-1,0] for z in U])
    lat=[ (lambda dI,py: np.sign(dI[-1,1])==np.sign(py) and abs(dI[-1,1])>0.1)(np.asarray(z["dI"]).reshape(5,2),np.mean(np.asarray(IDX[z["uid"]]["corners_ego"])[:,1])) for z in U]
    out[m]=dict(n=len(U),v0=float(a.mean()),vN=float(b.mean()),dv=float(d.mean()),dv_rel=float(d.mean()/a.mean()),faster=float(np.mean(d>0.05)),p=float(wilcoxon(d)[1]),
                dv_seg=(b-a).mean(0).tolist(),dS=float(dS.mean()),dS_p=float(wilcoxon(dS)[1]),closer=float(np.mean(dS<-0.05)),
                rho_fwd=float(spearmanr(dx,dS)[0]),lat_toward=float(np.mean(lat)),
                contact_P=float(np.mean([z["P"]["A"]==0 for z in U])),contact_N=float(np.mean([z["fN"]["A"]==0 for z in U])))
    o=out[m]; print(f"{m:9s} n={o['n']} v {o['v0']:.2f}→{o['vN']:.2f} ({o['dv_rel']:+.0%}, p={o['p']:.0e}) ΔS={o['dS']:+.2f} (p={o['dS_p']:.0e}) ρfwd={o['rho_fwd']:+.2f} lat={o['lat_toward']:.0%} contact {o['contact_P']:.3f}→{o['contact_N']:.3f}")
json.dump(out,open(f"{V5}/night_speed.json","w"),indent=1)
