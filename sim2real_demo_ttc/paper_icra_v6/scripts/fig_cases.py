"""Fig. 4：四个右舵案例，逐例检验左舵诊断给出的预测。
每格：自车在原点，走廊 ±1 m，目标行人/骑车人的真实未来（红色虚线 + 星号为起点），六家 4 s 规划。
选例：两例诊断命中（仓位档位、危险排序），两例含未命中（耐心那条），不挑好看的。
数据：case10.json（读数）+ nv_ped_future.json（行人真实未来）+ nvtraj_<m>_*_nav.json（规划）。"""
import json,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SH={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo"}
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.linewidth":0.6,"axes.edgecolor":"#52514e",
                     "xtick.color":"#52514e","ytick.color":"#52514e"})
def _pick(m):
    a=f"{R5}/nvtraj_{m}_sg-one-north_closevru_nav.json"; b=f"{R5}/nvtraj_{m}_sg-one-north_closevru.json"
    return json.load(open(a if os.path.exists(a) else b))
TR={m:_pick(m) for m in M}
PF=json.load(open(f"{V5}/nv_ped_future.json")); C={r["token"]:r for r in json.load(open(f"{V5}/case10.json"))}
CASES=[("b6cce8f28e405742","(a) group"),
       ("93a208914ea85781","(b) cyclist"),
       ("3346273c90155b64","(c) waiting"),
       ("e44df9ed23f45266","(d) crosswalk")]
fig,ax=plt.subplots(1,4,figsize=(3.45,2.05),gridspec_kw={"wspace":0.10})
for k,(tok,title) in enumerate(CASES):
    a=ax[k]; p=PF[tok]; r=C[tok]
    a.axvspan(-1,1,color="#eef1f6",lw=0,zorder=0)
    fut=np.array([p["p0"]]+[z for z in p["fut"] if z is not None],float)
    a.plot(-fut[:,1],fut[:,0],color="#e34948",ls=(0,(1.6,1.2)),lw=1.2,marker=".",ms=2.6,zorder=5)
    a.plot(-p["p0"][1],p["p0"][0],marker="*",ms=8,color="#e34948",mec="white",mew=0.5,zorder=6)
    for m in M:
        w=np.asarray(TR[m][tok],float)[:,:2]; w=np.vstack([[0,0],w])
        a.plot(-w[:,1],w[:,0],color=COL[m],lw=1.0,zorder=3)
    a.plot(0,0,marker="^",ms=5,color="#0b0b0b",zorder=6)
    if r["waiting"]:
        go=[(SH[m],r["models"][m]["plan_arc"]) for m in M if r["models"][m]["plan_arc"]>2]
        txt="departs: "+(", ".join(f"{n} {v:.1f} m" for n,v in go) if go else "none")
        if len(go)>2: txt=txt.replace(", "+f"{go[2][0]}"," \n"+f"{go[2][0]}",1)
        txt=txt.replace("departs: ","").replace(", ","\n")
        a.text(0.04,0.97,txt,transform=a.transAxes,fontsize=4.4,va="top",color="#b3412c",linespacing=1.2)
    a.set_xlim(-5.5,5.5); a.set_ylim(-2,26)
    a.set_xticks([-4,0,4]); a.set_xticklabels(["-4","0","4"],fontsize=5.0)
    a.set_yticks([0,10,20] if k==0 else []); a.tick_params(labelsize=5.2,length=1.6,pad=1)
    a.set_title(title,fontsize=6.0,loc="left",pad=1.5)
    if k==0: a.set_ylabel("ahead (m)",fontsize=5.6,labelpad=0.5)
    if k==0: a.set_xlabel("lateral (m)",fontsize=5.6,labelpad=0.5)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
h=[plt.Line2D([],[],color=COL[m],lw=1.4,label=SH[m]) for m in M]+[plt.Line2D([],[],color="#e34948",ls=(0,(1.6,1.2)),lw=1.2,marker="*",ms=6,label="VRU logged future")]
fig.legend(handles=h,frameon=False,fontsize=5.0,ncol=4,loc="lower center",bbox_to_anchor=(0.5,-0.15),handlelength=1.1,columnspacing=0.8,labelspacing=0.2)
fig.savefig(f"{V5}/figures/cases.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/cases.png",dpi=220,bbox_inches="tight"); print("ok")
