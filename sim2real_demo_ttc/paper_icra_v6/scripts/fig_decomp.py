"""Fig. 3（单栏）：位移不等于避让。
横轴 F = 移除行人引起的规划位移（对数），纵轴 ΔS = 因此换来的间隙变化。
  灰带 |ΔS|<0.5 m = "动了也没换来距离"；竖虚线 = F=0.5 m 判定门槛
  阴影域 = 按 F 分箱的 ΔS 10–90 分位（虚线边界）：F 小时收敛在 0 附近，越过门槛后向上下对称扇开
  空心圈 = 真避让（F≥0.5 且 F>I 且 ΔS≥0.5，绿）与动了反而更近（红）
输入 f_decomp_per_scene.json（六家 × 246 个右舵近行人场景）。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SH={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo"}
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.6,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
P=json.load(open(f"{V5}/f_decomp_per_scene.json")); EPS=3e-3
F=np.concatenate([[max(x["F"],EPS) for x in P[m]] for m in M])
I=np.concatenate([[x["I"] for x in P[m]] for m in M])
S=np.concatenate([[x["dS"] for x in P[m]] for m in M])
CI=np.concatenate([[COL[m]]*len(P[m]) for m in M])
fig,a=plt.subplots(figsize=(3.45,2.45))
a.axhspan(-0.5,0.5,color="#e9e8e2",lw=0,zorder=0)
a.axvline(0.5,color="#6b6a62",ls=(0,(3,2)),lw=0.8,zorder=3)
a.axhline(0,color="#9a998f",lw=0.5,zorder=1)
# 分位域：按 F 的对数分箱取 ΔS 的 10–90 分位
ed=np.logspace(np.log10(EPS),np.log10(F.max()*1.02),13); ctr=[];lo=[];hi=[]
for i in range(len(ed)-1):
    v=S[(F>=ed[i])&(F<ed[i+1])]
    if len(v)>=12: ctr.append(np.sqrt(ed[i]*ed[i+1])); lo.append(np.percentile(v,10)); hi.append(np.percentile(v,90))
a.fill_between(ctr,lo,hi,color="#1f5f73",alpha=0.13,lw=0,zorder=1)
a.plot(ctr,lo,color="#1f5f73",lw=0.8,ls=(0,(2.5,1.8)),alpha=0.8,zorder=2)
a.plot(ctr,hi,color="#1f5f73",lw=0.8,ls=(0,(2.5,1.8)),alpha=0.8,zorder=2)
real=(F>=0.5)&(F>I)&(S>=0.5); bad=(F>=0.5)&(S<=-0.5); rest=~(real|bad)
a.scatter(F[rest],S[rest],s=2.6,c=CI[rest],alpha=0.30,lw=0,zorder=4)
a.scatter(F[bad],S[bad],s=9,facecolor="none",edgecolor="#b3412c",lw=0.6,zorder=5)
a.scatter(F[real],S[real],s=9,facecolor="none",edgecolor="#2f7d4f",lw=0.6,zorder=5)
a.set_xscale("log"); a.set_xlim(EPS*0.85,25); a.set_ylim(-8.5,8.5)
a.set_xlabel(r"$F$: plan displacement on removing the pedestrian (m)",fontsize=6.6)
a.set_ylabel(r"$\Delta S$: clearance gained (m)",fontsize=6.6)
a.tick_params(labelsize=6)
a.text(0.62,7.4,"plan moved",fontsize=5.8,color="#52514e")
a.text(EPS*1.05,0.75,"no clearance bought",fontsize=5.8,color="#6b6a62")
a.scatter([],[],s=9,facecolor="none",edgecolor="#2f7d4f",lw=0.6,label="genuine avoidance")
a.scatter([],[],s=9,facecolor="none",edgecolor="#b3412c",lw=0.6,label="moved towards")
a.fill_between([],[],[],color="#1f5f73",alpha=0.13,label="10-90% of $\\Delta S$")
a.legend(frameon=False,fontsize=5.8,loc="lower left",bbox_to_anchor=(-0.015,-0.02),handletextpad=0.3,labelspacing=0.2,borderpad=0.1)
fig.savefig(f"{V5}/figures/decomp.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/decomp.png",dpi=230,bbox_inches="tight"); print("ok")
