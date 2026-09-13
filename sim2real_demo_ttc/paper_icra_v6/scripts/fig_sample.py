"""Fig.（样本量）：估计标准差随帧数收敛，以及每条判定要多少帧。压成单栏一张扁图。
左：四项考试的估计标准差对帧数 n（每家一条细线，六家共 24 条），叠加拟合的有限总体律 a√(1/n−1/N)；
右：把拟合外推成"让估计离判定阈值两个标准差"所需的帧数，按判定分组。
数据：sample_size_v2.json（170 个六家共有帧，无放回子采样 400 次）。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.6,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
S=json.load(open(f"{V5}/sample_size_v2.json")); N=S["N"]
MIN=1/12.0   # 每帧 5 s（本语料 4 帧 / 20 s 场景）；横轴一律换算成分钟
ns=np.array(S["ns"],float); nn=np.linspace(ns[0],N-1,200)
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
EXC={"exposure":"#2a5db0","CFR":"#6a3d9a","SP":"#1baf7a","HS":"#c0392b"}
LAB={"exposure":"exposure","CFR":"lighting CFR","SP":"specificity","HS":"hazard sensitivity"}
fig,ax=plt.subplots(2,1,figsize=(3.45,1.72),gridspec_kw={"hspace":0.95})
a=ax[0]
for k,c in EXC.items():
    D=S["dims"][k]
    for m in M:
        f=D["fit"][m]
        a.plot(ns*MIN,np.array(f["sd"]),lw=0,marker="o",ms=1.3,color=c,alpha=0.55)
        a.plot(nn*MIN,f["a"]*np.sqrt(1/nn-1/N),color=c,lw=0.7,alpha=0.85)
a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([1,3,10]); a.set_xticklabels(["1","3","10"]); a.minorticks_off()
a.tick_params(labelsize=4.9,length=1.5,pad=1); a.set_xlabel("minutes of near-pedestrian driving",fontsize=5.4,labelpad=0.3)
a.set_ylabel("SD",fontsize=5.4,labelpad=1)
a.set_title("(a) spread of the estimate",fontsize=6.2,loc="left",pad=1.5)
h=[plt.Line2D([],[],color=c,lw=1.1,label=LAB[k]) for k,c in EXC.items()]
a.legend(handles=h,frameon=False,fontsize=4.7,ncol=4,loc="upper right",bbox_to_anchor=(1.02,1.13),handlelength=0.8,columnspacing=0.6,labelspacing=0.1,borderpad=0.05)
b=ax[1]; V=[("CFR","CFR < 1","#6a3d9a"),("exposure","exposure ≠ human","#2a5db0"),("HS","HS ≠ 0","#c0392b")]
XMAX=3000*1/12.0
for j_,(k,nm,c) in enumerate(V):
    y=len(V)-1-j_
    vals=[S["dims"][k]["n_star"].get(m) for m in M]
    vals=[v for v in vals if v and np.isfinite(v)]
    vals=[v*MIN for v in vals]
    ins=[v for v in vals if v<=XMAX]; out=[v for v in vals if v>XMAX]
    b.scatter(ins,[y]*len(ins),s=8,color=c,zorder=3)
    if out: b.scatter([XMAX*0.93],[y],s=10,marker=">",facecolor="white",edgecolor=c,lw=0.8,zorder=3)
    if ins: b.plot([min(ins),max(ins)],[y]*2,color=c,lw=0.7,alpha=0.5,zorder=2)
    b.text(0.10,y+0.22,nm,fontsize=5.0,color="#52514e",va="bottom")
b.set_yticks([]); b.set_ylim(-0.6,len(V)-0.25); b.set_xscale("log"); b.set_xlim(0.09,XMAX*1.2)
b.set_xticks([1,10,100]); b.set_xticklabels(["1","10","100"]); b.minorticks_off()
b.tick_params(labelsize=4.9,length=1.5,pad=1); b.set_xlabel("minutes",fontsize=5.4,labelpad=0.3)
b.set_title("(b) driving a verdict needs",fontsize=6.2,loc="left",pad=1.5)
b.spines["left"].set_visible(False); b.tick_params(axis="y",length=0)
fig.savefig(f"{V5}/figures/sample_size.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/sample_size.png",dpi=230,bbox_inches="tight"); print("ok")
