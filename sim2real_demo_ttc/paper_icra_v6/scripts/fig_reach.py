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
fig,ax=plt.subplots(1,1,figsize=(3.45,2.0))
# (b)
rows=json.load(open(f"{V5}/diag_units.json")); b=ax
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
b.legend(frameon=False,fontsize=5.6,loc="upper left",ncol=1,handlelength=1.6,labelspacing=0.18,borderpad=0.1)
b.set_title("collision vs. input speed",fontsize=7,loc="left",pad=2)
fig.savefig(f"{V5}/figures/reach.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/reach.png",dpi=220,bbox_inches="tight"); print("ok")
