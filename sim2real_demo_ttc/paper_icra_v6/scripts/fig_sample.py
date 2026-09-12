"""Fig. 3：一次体检要多少帧。(a–d) 四个维度的估计标准差随帧数 n 的变化与 a·√(1/n−1/N) 拟合；(e) 各判定所需帧数。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
                     "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#008300"}
NAME={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo"}
S=json.load(open(f"{V5}/sample_size_v2.json")); N=S["N"]; ns=np.array(S["ns"],float); nn=np.linspace(ns[0],N-1,200)
M=list(COL)
fig=plt.figure(figsize=(7.16,2.1))
gs=fig.add_gridspec(1,6,width_ratios=[1,1,1,1,0.18,1.2],wspace=0.42,left=0.06,right=0.99,top=0.86,bottom=0.33)
TIT={"exposure":"exposure","CFR":"lighting CFR","SP":"specificity","HS":"hazard sensitivity"}
for i,k in enumerate(["exposure","CFR","SP","HS"]):
    ax=fig.add_subplot(gs[i]); D=S["dims"][k]
    for m in M:
        f=D["fit"][m]; y=np.array(f["sd"])
        ax.plot(ns,y,"o",ms=2.2,color=COL[m]); ax.plot(nn,f["a"]*np.sqrt(1/nn-1/N),color=COL[m],lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([10,30,100]); ax.set_xticklabels(["10","30","100"]); ax.minorticks_off()
    r2=[D["fit"][m]["r2"] for m in M]
    ax.set_title(f"({'abcd'[i]}) {TIT[k]}",fontsize=7,loc="left",pad=2)
    ax.text(0.03,0.04,f"fit R² {min(r2):.2f}–{max(r2):.2f}",transform=ax.transAxes,fontsize=5.8,color="#52514e")
    ax.set_xlabel("frames n",labelpad=1)
    if i==0: ax.set_ylabel("SD of estimate",labelpad=1)
ax=fig.add_subplot(gs[5])
V=[("CFR","CFR < 1"),("exposure","exposure ≠ human")]
y=np.arange(len(M))
for j,(k,lab) in enumerate(V):
    ns_=S["dims"][k]["n_star"]; vals=[min(ns_.get(m,np.nan),1e4) for m in M]
    ax.barh(y+(j-0.5)*0.36,vals,height=0.34,color=["#2a78d6","#eda100"][j],label=lab)
ax.set_yticks(y); ax.set_yticklabels([NAME[m] for m in M],fontsize=6.3); ax.set_xscale("log"); ax.set_xlim(0.8,1.2e4)
ax.axvline(N,color="#52514e",lw=0.7,ls=":"); ax.text(N*1.12,-0.75,f"pool\n({N})",fontsize=5.6,color="#52514e",va="top")
ax.set_xlabel("frames needed",labelpad=1); ax.legend(frameon=False,fontsize=5.8,loc="upper center",handlelength=1.2,bbox_to_anchor=(0.2,-0.3),ncol=2,columnspacing=0.6)
ax.set_title("(e) frames per verdict",fontsize=7,loc="left",pad=2); ax.invert_yaxis()
from matplotlib.lines import Line2D
fig.legend([Line2D([],[],color=COL[m],marker="o",ms=3,lw=1) for m in M],[NAME[m] for m in M],loc="lower left",ncol=6,frameon=False,fontsize=6.3,bbox_to_anchor=(0.04,0.0))
fig.savefig(f"{V5}/figures/sample_size.pdf"); fig.savefig(f"{V5}/figures/sample_size.png",dpi=220); print("ok")
