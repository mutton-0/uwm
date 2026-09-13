"""Fig. 2（碰撞随车速 + 危险敏感度的 TTC 版本）。
原 (a)"规划够得着才反应"的横截面图已删：只覆盖 DD/LTF 两家，且三个关键数字都在正文与表 IV 里。
(a) 碰撞率随输入车速上升，看得到 / 看不到行人两条线重合（v5 口径：0.1 s 插值、行人真实未来）
(b) 危险敏感度对 TTC0 = (d−2)/v（越左越危险）"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
                     "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#008300"}
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
M=list(COL); rng=np.random.default_rng(0)
def med_ci(v):
    i=rng.integers(0,len(v),(2000,len(v))); m=np.median(v[i],1); return np.median(v),np.percentile(m,2.5),np.percentile(m,97.5)
def mean_ci(v):
    i=rng.integers(0,len(v),(2000,len(v))); m=v[i].mean(1); return v.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
fig,ax=plt.subplots(1,2,figsize=(7.16,1.9),gridspec_kw={"wspace":0.30})
# (b)
rows=json.load(open(f"{V5}/diag_units.json")); b=ax[0]
B=[z for z in rows if z["set"]=="B"]
vact=float(np.median([z["v"] for z in B if z["sv"]=="actual"]))
for m in M:
    pts=[]
    for sv in ("actual","2","4","6","8"):
        U=[z for z in B if z["m"]==m and z["sv"]==sv]
        if len(U)<20: continue
        pts.append((np.median([z["v"] for z in U]) if sv=="actual" else float(sv),100*np.mean([z["P"]["A"]==0 for z in U]),100*np.mean([z["Q"]["A"]==0 for z in U])))
    pts.sort(); p=np.array(pts)
    b.plot(p[:,0],p[:,1],color=COL[m],lw=1.2,marker="o",ms=2.6,label=NAME[m])
    b.plot(p[:,0],p[:,2],color=COL[m],lw=0.9,ls=(0,(2,1.5)),marker="o",ms=2.6,mfc="white")
b.plot([],[],color="#52514e",lw=1.2,label="visible"); b.plot([],[],color="#52514e",lw=0.9,ls=(0,(2,1.5)),label="removed")
b.set_xlabel("input ego speed (m/s)"); b.set_ylabel("collision with logged pedestrian (%)"); b.set_xticks([2,4,6,8])
b.legend(frameon=False,fontsize=5.4,loc="upper left",ncol=1,handlelength=1.8,labelspacing=0.2)
b.set_title("(a) collisions vs. speed",fontsize=7,loc="left",pad=2)
# (c)
c=ax[1]; sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0 and z["need"]
TB=[(2,99),(1.5,2),(1,1.5),(0,1)]; TL=[">2","1.5–2","1–1.5","<1"]
for k,m in enumerate(M):
    need=[z for z in rows if z["m"]==m and sel(z)]; x=[];y=[];e0=[];e1=[]
    for i,(lo,hi) in enumerate(TB):
        v=np.array([z["HS"] for z in need if lo<=z["ttc0"]<hi])
        if len(v)>=8: mu,l,h=mean_ci(v); x.append(i+(k-2.5)*0.07); y.append(mu); e0.append(mu-l); e1.append(h-mu)
    c.errorbar(x,y,yerr=[e0,e1],color=COL[m],marker="o",ms=2.6,lw=1.1,elinewidth=0.6,capsize=0)
c.plot(range(len(TB)),[0.15,0.4,0.65,0.9],color="#9a998f",ls=(0,(3,2)),lw=0.9)
c.text(2.9,0.95,"yielding\ndriver",fontsize=5.8,color="#52514e",ha="center")
c.axhline(0,color="#c3c2b7",lw=0.5); c.set_xticks(range(len(TB))); c.set_xticklabels(TL,fontsize=6.3); c.set_ylim(-0.3,1.05)
c.set_xlabel(r"TTC$_0$ (s), more hazardous $\rightarrow$"); c.set_ylabel("hazard sensitivity HS")
c.set_title("(b) HS against time to collision",fontsize=7,loc="left",pad=2)
fig.savefig(f"{V5}/figures/reach.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/reach.png",dpi=220,bbox_inches="tight"); print("ok")
