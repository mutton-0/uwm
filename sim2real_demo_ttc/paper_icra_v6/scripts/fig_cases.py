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
fig=plt.figure(figsize=(7.16,2.1))
gs=fig.add_gridspec(1,6,width_ratios=[1,1,1,1,0.30,3.05],wspace=0.12)
ax=[fig.add_subplot(gs[i]) for i in range(4)]
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
fig.legend(handles=h,frameon=False,fontsize=5.2,ncol=7,loc="lower center",bbox_to_anchor=(0.28,-0.13),handlelength=1.1,columnspacing=0.8)
# ---- 右半：逐场景 F 与 ΔS
P=json.load(open(f"{V5}/f_decomp_per_scene.json")); EPS=3e-3
MM=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
F=np.concatenate([[max(x["F"],EPS) for x in P[m]] for m in MM])
I=np.concatenate([[x["I"] for x in P[m]] for m in MM])
DS=np.concatenate([[x["dS"] for x in P[m]] for m in MM])
CI=np.concatenate([[COL[m]]*len(P[m]) for m in MM])
d=fig.add_subplot(gs[5])
d.axhspan(-0.5,0.5,color="#e9e8e2",lw=0,zorder=0)
d.axvline(0.5,color="#6b6a62",ls=(0,(3,2)),lw=0.8,zorder=3); d.axhline(0,color="#9a998f",lw=0.5,zorder=1)
ed=np.logspace(np.log10(EPS),np.log10(F.max()*1.02),13); ctr=[];lo=[];hi=[]
for i in range(len(ed)-1):
    v=DS[(F>=ed[i])&(F<ed[i+1])]
    if len(v)>=12: ctr.append(np.sqrt(ed[i]*ed[i+1])); lo.append(np.percentile(v,10)); hi.append(np.percentile(v,90))
d.fill_between(ctr,lo,hi,color="#1f5f73",alpha=0.13,lw=0,zorder=1)
d.plot(ctr,lo,color="#1f5f73",lw=0.8,ls=(0,(2.5,1.8)),alpha=0.8,zorder=2); d.plot(ctr,hi,color="#1f5f73",lw=0.8,ls=(0,(2.5,1.8)),alpha=0.8,zorder=2)
real=(F>=0.5)&(F>I)&(DS>=0.5); bad=(F>=0.5)&(DS<=-0.5); rest=~(real|bad)
d.scatter(F[rest],DS[rest],s=2.4,c=CI[rest],alpha=0.30,lw=0,zorder=4)
d.scatter(F[bad],DS[bad],s=8,facecolor="none",edgecolor="#b3412c",lw=0.6,zorder=5)
d.scatter(F[real],DS[real],s=8,facecolor="none",edgecolor="#2f7d4f",lw=0.6,zorder=5)
d.set_xscale("log"); d.set_xlim(EPS*0.85,25); d.set_ylim(-8.5,8.5)
d.set_xlabel("$F$: displacement on removing the pedestrian (m)",fontsize=5.8,labelpad=0.5)
d.set_ylabel("$\\Delta S$: clearance gained (m)",fontsize=5.8,labelpad=1)
d.tick_params(labelsize=5.2,length=2); d.set_title("(e) displacement and the separation it buys",fontsize=6.0,loc="left",pad=1.5)
d.scatter([],[],s=8,facecolor="none",edgecolor="#2f7d4f",lw=0.6,label="genuine avoidance")
d.scatter([],[],s=8,facecolor="none",edgecolor="#b3412c",lw=0.6,label="moved towards")
d.fill_between([],[],[],color="#1f5f73",alpha=0.13,label="10-90% of $\\Delta S$")
d.legend(frameon=False,fontsize=5.0,loc="lower left",bbox_to_anchor=(-0.015,-0.02),handletextpad=0.3,labelspacing=0.18,borderpad=0.1)
for s_ in ("top","right"): d.spines[s_].set_visible(False)
fig.savefig(f"{V5}/figures/cases.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/cases.png",dpi=220,bbox_inches="tight"); print("ok")
