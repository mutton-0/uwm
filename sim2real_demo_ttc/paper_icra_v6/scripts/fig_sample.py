"""Fig.（样本量）：估计标准差随帧数收敛，以及每条判定要多少帧。压成单栏一张扁图。
左：四项考试的估计标准差对帧数 n（每家一条细线，六家共 24 条），叠加拟合的有限总体律 a√(1/n−1/N)；
右：把拟合外推成"让估计离判定阈值两个标准差"所需的帧数，按判定分组。
数据：sample_size_v2.json（170 个六家共有帧，无放回子采样 400 次）。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.6,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
S=json.load(open(f"{V5}/sample_size_v2.json")); N=S["N"]; ns=np.array(S["ns"],float); nn=np.linspace(ns[0],N-1,200)
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
EXC={"exposure":"#2a5db0","CFR":"#6a3d9a","SP":"#1baf7a","HS":"#c0392b"}
LAB={"exposure":"exposure","CFR":"lighting CFR","SP":"specificity","HS":"hazard sensitivity"}
fig,ax=plt.subplots(1,2,figsize=(3.45,1.55),gridspec_kw={"wspace":0.42,"width_ratios":[1.15,1]})
a=ax[0]
for k,c in EXC.items():
    D=S["dims"][k]
    for m in M:
        f=D["fit"][m]
        a.plot(ns,np.array(f["sd"]),lw=0,marker="o",ms=1.3,color=c,alpha=0.55)
        a.plot(nn,f["a"]*np.sqrt(1/nn-1/N),color=c,lw=0.7,alpha=0.85)
a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([10,30,100]); a.set_xticklabels(["10","30","100"]); a.minorticks_off()
a.tick_params(labelsize=5.8,length=2); a.set_xlabel("frames $n$",fontsize=6,labelpad=0.5)
a.set_ylabel("SD of estimate",fontsize=6,labelpad=1)
a.set_title("(a) spread falls as $1/\\sqrt{n}$",fontsize=6.5,loc="left",pad=2)
h=[plt.Line2D([],[],color=c,lw=1.1,label=LAB[k]) for k,c in EXC.items()]
a.legend(handles=h,frameon=False,fontsize=5.3,loc="lower left",bbox_to_anchor=(-0.02,-0.03),handlelength=1.1,labelspacing=0.12,borderpad=0.1)
b=ax[1]; V=[("CFR","CFR < 1","#6a3d9a"),("exposure","exposure ≠ human","#2a5db0"),("HS","HS ≠ 0","#c0392b")]
XMAX=3000
for j_,(k,nm,c) in enumerate(V):
    y=len(V)-1-j_
    vals=[S["dims"][k]["n_star"].get(m) for m in M]
    vals=[v for v in vals if v and np.isfinite(v)]
    ins=[v for v in vals if v<=XMAX]; out=[v for v in vals if v>XMAX]
    b.scatter(ins,[y]*len(ins),s=13,color=c,zorder=3)
    if out: b.scatter([XMAX*0.93],[y],s=16,marker=">",facecolor="white",edgecolor=c,lw=0.9,zorder=3)
    if ins: b.plot([min(ins),max(ins)],[y]*2,color=c,lw=0.7,alpha=0.5,zorder=2)
    b.text(1.25,y+0.30,nm,fontsize=5.6,color="#52514e",va="bottom")
b.set_yticks([]); b.set_ylim(-0.6,len(V)-0.25); b.set_xscale("log"); b.set_xlim(1.1,XMAX*1.15)
b.set_xticks([10,100,1000]); b.set_xticklabels(["10","100","1k"]); b.minorticks_off()
b.tick_params(labelsize=5.8,length=2); b.set_xlabel("frames a verdict needs",fontsize=6,labelpad=0.5)
b.set_title("(b) what each verdict costs",fontsize=6.5,loc="left",pad=2)
b.spines["left"].set_visible(False); b.tick_params(axis="y",length=0)
b.text(XMAX*0.55,-0.55,"off scale",fontsize=5.0,color="#8a8a84",ha="right")
fig.savefig(f"{V5}/figures/sample_size.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/sample_size.png",dpi=230,bbox_inches="tight"); print("ok")
