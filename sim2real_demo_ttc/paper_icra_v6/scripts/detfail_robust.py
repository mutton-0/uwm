"""残留检验：独立检测器在移除后仍能找到目标的帧（IoU≥0.5），剔掉它们后诊断是否改变。
对比 profile_ALL.json（全部 236 帧）与 profile_ALL_nodetfail.json（剔除 21 帧后）：
  1) 每个模型每个考项的点估计是否仍落在全样本的 bootstrap 95% CI 内
  2) 跨模型排序是否一致（Spearman）
输出 detfail_robust.json"""
import json
from scipy.stats import spearmanr
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
K=["exposure","HS","HS_slope","SP","CFR"]
A=json.load(open(f"{V5}/profile_ALL.json")); B=json.load(open(f"{V5}/profile_ALL_nodetfail.json"))
inside=0; tot=0; rows=[]
for k in K:
    a=[A[m]["point"][k] for m in M]; b=[B[m]["point"][k] for m in M]
    ok=[]
    for m in M:
        lo,hi=A[m]["ci"][k]; v=B[m]["point"][k]
        h=(lo<=v<=hi); ok.append(h); inside+=h; tot+=1
    rho=spearmanr(a,b)[0]
    rows.append(dict(exam=k,rho=float(rho),inside=sum(ok),n=len(ok)))
    print(f"  {k:10s} 排序 ρ={rho:+.2f}   点估计仍在原 CI 内 {sum(ok)}/{len(ok)}")
res=dict(rows=rows, inside=inside, total=tot, pct=100*inside/tot,
         rho_min=min(r["rho"] for r in rows), n_drop=len(A)and None)
print(f"\n合计 {inside}/{tot} 个模型-考项的点估计仍落在全样本 95% CI 内；最小排序 ρ={min(r['rho'] for r in rows):+.2f}")
json.dump(res,open(f"{V5}/detfail_robust.json","w"),indent=1)
