"""Fig. 3：位移不等于避让。
(a) 逐场景散点：横轴 F（移除行人引起的规划位移，对数轴），纵轴 ΔS（换来的间隙变化）。
    灰带 = |ΔS|<0.5 m「动了也没换来距离」；竖线 = F=0.5 m 的判定门槛。
    绿色 = 真避让（F≥0.5 且 F>I 且 ΔS≥0.5），红色 = 动了反而更近。
(b) 每家在 F≥0.5 的场景里，位移去向的构成（离更远 / 没变 / 反而更近）。
输入 f_decomp_per_scene.json（六家 × 246 个右舵近行人场景）。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SH={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#008300"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.6,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
P=json.load(open(f"{V5}/f_decomp_per_scene.json"))
fig,ax=plt.subplots(1,2,figsize=(7.16,2.25),gridspec_kw={"wspace":0.26,"width_ratios":[1.55,1]})
a=ax[0]
a.axhspan(-0.5,0.5,color="#e9e8e2",lw=0,zorder=0)
a.axvline(0.5,color="#9a998f",ls=(0,(3,2)),lw=0.8,zorder=1)
a.axhline(0,color="#9a998f",lw=0.5,zorder=1)
EPS=3e-3
for m in M:
    r=P[m]; F=np.array([max(x["F"],EPS) for x in r]); dS=np.array([x["dS"] for x in r]); I=np.array([x["I"] for x in r])
    real=(F>=0.5)&(F>I)&(dS>=0.5); bad=(F>=0.5)&(dS<=-0.5); rest=~(real|bad)
    a.scatter(F[rest],dS[rest],s=4,color=COL[m],alpha=0.35,lw=0,zorder=2)
    a.scatter(F[bad],dS[bad],s=11,facecolor="none",edgecolor="#b3412c",lw=0.7,zorder=4)
    a.scatter(F[real],dS[real],s=11,facecolor="none",edgecolor="#2f7d4f",lw=0.7,zorder=4)
a.set_xscale("log"); a.set_xlim(EPS*0.8,30); a.set_ylim(-9,9)
a.set_xlabel(r"$F$: plan displacement on removing the pedestrian (m)")
a.set_ylabel(r"$\Delta S$: clearance gained (m)")
a.text(0.55,-8.4,"plan moved",fontsize=6,color="#52514e")
a.text(EPS*1.1,0.05,"no clearance bought",fontsize=6,color="#6b6a62")
a.scatter([],[],s=11,facecolor="none",edgecolor="#2f7d4f",lw=0.7,label="genuine avoidance")
a.scatter([],[],s=11,facecolor="none",edgecolor="#b3412c",lw=0.7,label="moved towards pedestrian")
a.legend(frameon=False,fontsize=6,loc="upper left",bbox_to_anchor=(0.0,0.99),handletextpad=0.3)
a.set_title("(a) displacement vs. separation, per scene",fontsize=7,loc="left",pad=2)
b=ax[1]; y=np.arange(len(M))[::-1]
for i,m in enumerate(M):
    r=P[m]; F=np.array([x["F"] for x in r]); dS=np.array([x["dS"] for x in r]); big=F>=0.5
    n=big.sum()
    if n==0: continue
    far=(dS[big]>=0.5).sum(); near=(dS[big]<=-0.5).sum(); flat=n-far-near
    l=0
    for v,c,lab in ((far,"#2f7d4f","away"),(flat,"#c9c8c0","unchanged"),(near,"#b3412c","towards")):
        b.barh(y[i],100*v/n,left=l,height=0.6,color=c,lw=0,label=lab if i==0 else None); l+=100*v/n
    b.text(101,y[i],f"n={n}",va="center",fontsize=6,color="#52514e")
b.set_yticks(y); b.set_yticklabels([SH[m] for m in M],fontsize=6.5); b.set_xlim(0,100); b.set_xlabel("share of scenes where the plan moved (%)")
b.legend(frameon=False,fontsize=6,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.22),handlelength=1.2,columnspacing=1.0)
b.set_title("(b) where the displacement went",fontsize=7,loc="left",pad=12)
b.spines["left"].set_visible(False); b.tick_params(axis="y",length=0)
fig.savefig(f"{V5}/figures/decomp.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/decomp.png",dpi=220,bbox_inches="tight"); print("ok")
