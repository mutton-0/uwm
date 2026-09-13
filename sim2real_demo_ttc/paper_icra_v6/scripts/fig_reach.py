"""Fig. 2（碰撞随车速 + 危险敏感度的 TTC 版本）。
原 (a)"规划够得着才反应"的横截面图已删：只覆盖 DD/LTF 两家，且三个关键数字都在正文与表 IV 里。
(a) 碰撞率随输入车速上升，看得到 / 看不到行人两条线重合（v5 口径：0.1 s 插值、行人真实未来）
(b) 危险敏感度对 TTC0 = (d−2)/v（越左越危险）"""
import json,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
                     "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
M=list(COL); rng=np.random.default_rng(0)
def med_ci(v):
    i=rng.integers(0,len(v),(2000,len(v))); m=np.median(v[i],1); return np.median(v),np.percentile(m,2.5),np.percentile(m,97.5)
def mean_ci(v):
    i=rng.integers(0,len(v),(2000,len(v))); m=v[i].mean(1); return v.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
fig,ax=plt.subplots(1,2,figsize=(3.45,1.75),gridspec_kw={"wspace":0.34})
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
b.set_xlabel("input ego speed (m/s)",fontsize=5.9,labelpad=0.5); b.set_ylabel("collision with pedestrian (%)",fontsize=5.9,labelpad=1); b.set_xticks([2,4,6,8]); b.tick_params(labelsize=5.4,length=2)
b.legend(handles=[plt.Line2D([],[],color="#52514e",lw=1.2,label="visible"),plt.Line2D([],[],color="#52514e",lw=0.9,ls=(0,(2,1.5)),label="removed")],frameon=False,fontsize=5.4,loc="upper left",handlelength=1.5,labelspacing=0.15,borderpad=0.1)
b.set_title("(a) collision vs. input speed",fontsize=6.4,loc="left",pad=2)
# (b) 危险敏感度对 a_req（原图 1c 搬来）
c=ax[1]
sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0 and z["need"]
BINS=[(0,0.5),(0.5,1),(1,2),(2,4),(4,99)]; XL=["<0.5","0.5-1","1-2","2-4",">4"]
for m in M:
    need=[z for z in rows if z["m"]==m and sel(z)]
    xs=[];ys=[];lo=[];hi=[]
    for i,(a_,b_) in enumerate(BINS):
        v=np.array([z["HS"] for z in need if a_<=z["a_req"]<b_])
        if len(v)>=8:
            mu,l,h=mean_ci(v); xs.append(i+(M.index(m)-2.5)*0.07); ys.append(mu); lo.append(mu-l); hi.append(h-mu)
    c.errorbar(xs,ys,yerr=[lo,hi],color=COL[m],marker="o",ms=2.4,lw=1.0,elinewidth=0.55,capsize=0)
GC=json.load(open(f"{V5}/gt_ceiling.json"))
c.plot(range(len(BINS)),GC["gt"],color="#52514e",ls=(0,(3,2)),lw=1.0)
c.fill_between(range(len(BINS)),GC["gt_lo"],GC["gt_hi"],color="#9a998f",alpha=0.20,lw=0)
c.text(0.08,0.72,"logged human vs.\neach blind plan",transform=c.transAxes,fontsize=5.0,color="#52514e")
c.axhline(0,color="#c3c2b7",lw=0.6)
c.set_xticks(range(len(BINS))); c.set_xticklabels(XL,fontsize=5.0); c.set_ylim(-0.25,0.95)
c.set_xlabel(r"hazard $a_{\rm req}$ (m/s$^2$)",fontsize=5.9,labelpad=0.5)
c.set_ylabel("hazard sensitivity HS",fontsize=5.9,labelpad=1); c.tick_params(labelsize=5.4,length=2)
c.set_title("(b) hazard sensitivity",fontsize=6.4,loc="left",pad=2)
for s_ in ("top","right"): c.spines[s_].set_visible(False)
hh=[plt.Line2D([],[],color=COL[m],lw=1.2,label=NAME[m].replace("DiffusionDriveV2","DDv2").replace("DiffusionDrive","DD").replace("Alpamayo 1.5","Alpamayo")) for m in M]
fig.legend(handles=hh,frameon=False,fontsize=5.0,ncol=6,loc="lower center",bbox_to_anchor=(0.5,-0.13),handlelength=1.1,columnspacing=0.7)
fig.savefig(f"{V5}/figures/reach.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/reach.png",dpi=220,bbox_inches="tight"); print("ok")
