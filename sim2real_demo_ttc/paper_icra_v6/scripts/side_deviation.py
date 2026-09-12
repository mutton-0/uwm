"""左右舵测量偏差：同一套体检读数在波士顿（左舵）与新加坡（右舵）的对照，外加这 10 例的逐例 CFR 中位。
cfr_l / cfr_r = 两侧的模型级 CFR（profile_LHD / profile_RHD 的点估计）；ratio = 右/左
hs_*, sp_* = 两侧的危险敏感度与特异性
case_cfr = 这 10 例逐例 CFR 的中位（case10_axes.json，逐例 = D_ped/D_I，与大样本的聚合口径不同，故通常更小）
ttc = 右舵近行人场景里最小 TTC<1.5 s 的比例（domain_transfer 的共有场景口径）
输出 paper_icra_v5/side_deviation.json"""
import json,os,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
PL=json.load(open(f"{V5}/profile_LHD.json")); PR=json.load(open(f"{V5}/profile_RHD.json"))
CA=json.load(open(f"{V5}/case10_axes.json"))
DT=json.load(open(f"{V5}/domain_transfer.json")) if os.path.exists(f"{V5}/domain_transfer.json") else {}
out={}
for m in M:
    l=PL.get(m,{}).get("point",{}); r=PR.get(m,{}).get("point",{})
    if not l or not r: continue
    cs=[c["models"][m]["cfr"] for c in CA["cases"] if m in c["models"] and c["models"][m].get("cfr") is not None]
    out[m]=dict(cfr_l=l["CFR"],cfr_r=r["CFR"],ratio=r["CFR"]/l["CFR"] if l["CFR"] else None,
                hs_l=l.get("HS"),hs_r=r.get("HS"),sp_l=l.get("SP"),sp_r=r.get("SP"),
                case_cfr=float(np.median(cs)) if cs else None,
                ttc=DT.get("truth_common",{}).get(m,{}).get("ttc_viol"))
json.dump(out,open(f"{V5}/side_deviation.json","w"),indent=1)
from scipy.stats import spearmanr
print("模型   CFR左   CFR右   倍数   逐例中位   右舵TTC违规")
for m,d in out.items():
    print(f"  {m:11s} {d['cfr_l']:.2f}  {d['cfr_r']:.2f}  {d['ratio']:.2f}×  {d['case_cfr']:.2f}  {d['ttc'] if d['ttc'] is not None else float('nan'):.0f}%")
K=list(out)
print("CFR 左右舵排名 ρ=%+.2f   特异性 ρ=%+.2f"%(
  spearmanr([out[m]['cfr_l'] for m in K],[out[m]['cfr_r'] for m in K])[0],
  spearmanr([out[m]['sp_l'] for m in K],[out[m]['sp_r'] for m in K])[0]))
print("倍数范围 %.2f–%.2f×"%(min(out[m]['ratio'] for m in K),max(out[m]['ratio'] for m in K)))
