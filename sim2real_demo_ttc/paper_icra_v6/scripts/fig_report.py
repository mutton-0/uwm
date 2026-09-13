"""Fig.（体检报告单）：把表 II 的四项考试画成带参考区间的化验单。
每栏一项考试，纵轴六家；阴影 = 表 I 给出的"会让行的司机"应落在的范围；
点 = 测得值，横线 = 95% bootstrap 区间；落在参考区间外的点加红圈并在右侧标 ✗。
最右一栏用一句话给出该家的判读。数据来自 profile_ALL.json。"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SH={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.linewidth":0.6,"axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
A=json.load(open(f"{V5}/profile_ALL.json"))
# (键, 标题, 参考区间, x 范围, 对数?)
EX=[("exposure","exposure\nplanned / logged",(0.8,1.2),(0.3,2.9),False),
    ("HS","hazard sensitivity\nsafety gained",(0.15,1.05),(-0.06,0.75),False),
    ("HS_slope","scaling\ncorr. with hazard",(0.0,1.0),(-0.62,0.62),False),
    ("SP","specificity\nno needless motion",(0.8,1.02),(0.30,1.02),False),
    ("CFR","lighting CFR\npedestrian / night",(1.0,60),(0.08,60),True)]
fig,ax=plt.subplots(1,len(EX)+1,figsize=(7.16,2.05),gridspec_kw={"wspace":0.10,"width_ratios":[1]*len(EX)+[0.85]})
yy=np.arange(len(M))[::-1]
for j,(k,title,band,xlim,logx) in enumerate(EX):
    a=ax[j]
    a.axvspan(band[0],band[1],color="#dfeadf",lw=0,zorder=0)
    for i,m in enumerate(M):
        p=A[m]["point"].get(k); ci=A[m]["ci"].get(k)
        if p is None: continue
        ok=band[0]<=p<=band[1]
        if ci: a.plot([ci[0],ci[1]],[yy[i]]*2,color=COL[m],lw=1.0,solid_capstyle="butt",zorder=2)
        a.scatter([p],[yy[i]],s=26 if not ok else 20,facecolor=COL[m],
                  edgecolor="#b3412c" if not ok else "none",lw=1.1 if not ok else 0,zorder=3)
    if logx: a.set_xscale("log")
    a.set_xlim(*xlim); a.set_ylim(-0.7,len(M)-0.3)
    a.set_yticks(yy); a.set_yticklabels([SH[m] for m in M] if j==0 else []); a.tick_params(labelsize=6,length=2)
    a.set_title(title,fontsize=6.3,pad=2.5)
    a.grid(axis="y",color="#eceae4",lw=0.5,zorder=-1)
    for s_ in ("left",): a.spines[s_].set_visible(False)
    a.tick_params(axis="y",length=0)
# 末栏：8 m/s 下撞行人的比例，看得见(实心) vs 看不见(空心)——两者几乎重合
CO=json.load(open(f"{V5}/coll_speed.json")) if __import__("os").path.exists(f"{V5}/coll_speed.json") else None
a=ax[len(EX)]
if CO:
    for i,m in enumerate(M):
        p,q=CO[m]["vis"],CO[m]["rm"]
        a.plot([p,q],[yy[i]]*2,color="#c9c8c0",lw=1.0,zorder=1)
        a.scatter([p],[yy[i]],s=20,color=COL[m],zorder=3)
        a.scatter([q],[yy[i]],s=20,facecolor="white",edgecolor=COL[m],lw=1.0,zorder=3)
a.set_xlim(-2,46); a.set_ylim(-0.7,len(M)-0.3); a.set_yticks(yy); a.set_yticklabels([])
a.tick_params(labelsize=6,length=2); a.tick_params(axis="y",length=0)
a.set_title("collision at 8 m/s (%)\nvisible / removed",fontsize=6.3,pad=2.5)
a.grid(axis="y",color="#eceae4",lw=0.5,zorder=-1); a.spines["left"].set_visible(False)
h=[plt.Line2D([],[],marker="o",lw=0,ms=4,mfc="#8a8a84",mec="none",label="measured (95% CI)"),
   plt.Line2D([],[],marker="o",lw=0,ms=4.6,mfc="#8a8a84",mec="#b3412c",mew=1.1,label="outside normal range"),
   plt.Rectangle((0,0),1,1,fc="#dfeadf",ec="none",label="normal range (Table I)")]
fig.legend(handles=h,frameon=False,fontsize=6,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.10),handlelength=1.4,columnspacing=1.4)
fig.savefig(f"{V5}/figures/report.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/report.png",dpi=230,bbox_inches="tight"); print("ok")
