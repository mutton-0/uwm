"""Fig. 2：两个例子格 + 全体格的 F–I 盘。
(a) 真避让格，(b) 朝行人移动格，(c) 所有策略×场景格在 F–I 平面上，(a)(b) 标在其中。
旧版（四个右舵案例 + BEV 语义头）：
每格：自车在原点，走廊 ±1 m，目标行人/骑车人的真实未来（红色虚线 + 星号为起点），六家 4 s 规划。
选例：两例诊断命中（仓位档位、危险排序），两例含未命中（耐心那条），不挑好看的。
数据：case10.json（读数）+ nv_ped_future.json（行人真实未来）+ nvtraj_<m>_*_nav.json（规划）。"""
import json,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
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
import sys; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fi_panels as fp
TT=fp.TT
def S_of(X,ped): return float(np.mean(np.minimum(np.linalg.norm(X-ped,axis=1)-1.4,10.0)))
cells=[]
for m in M:
    for t,v in fp.AX[m].items():
        if not all(k in v for k in ("clean","rm","night")) or t not in fp.PF: continue
        ped=fp.ped_xy(t); Fv=fp.disp(v["clean"],v["rm"]); Iv=fp.disp(v["clean"],v["night"])
        dS=S_of(fp.lin(v["clean"]),ped)-S_of(fp.lin(v["rm"]),ped)
        cR=float(np.min(np.linalg.norm(fp.lin(v["rm"])-ped,axis=1))-1.4); cO=float(np.min(np.linalg.norm(fp.lin(v["clean"])-ped,axis=1))-1.4)
        cells.append((m,t,Fv,Iv,dS,cR,cO))
EXCL={"34b62d7333845af3"}                                    # Fig.1(b) 已用的例子，不重复
good=[c for c in cells if c[2]>=0.5 and c[2]>c[3] and c[4]>=0.5 and c[1] not in EXCL and c[2]<3 and c[5]<1.0]   # need 格：盲规划进 1 m
bad=[c for c in cells if c[2]>=0.5 and c[4]<=-0.5 and c[1] not in EXCL and c[2]<3]
ca=max(good,key=lambda c:c[4]); cb=min(bad,key=lambda c:c[6])   # (b) 取原规划离行人最近的那格
print("(a) genuine:",ca); print("(b) towards:",cb)
fig=plt.figure(figsize=(3.45,3.3))
gs=fig.add_gridspec(2,2,height_ratios=[0.72,1.0],hspace=0.42,wspace=0.48)
for k,(c,lab) in enumerate(((ca,"(a) genuine avoidance"),(cb,"(b) moved towards"))):
    a=fig.add_subplot(gs[0,k]); fp.panel_extract(a,tok=c[1],mm=c[0],title=f"{lab}: {SH[c[0]]}",fs=1.05,legend=(k==0),ymax=11.5)
    a.text(0.02,0.72,f"$\\Delta S$ = {c[4]:+.2f} m",transform=a.transAxes,ha="left",fontsize=5.6,color="#2f7d4f" if c[4]>0 else "#b3412c")
MARK=[("a",ca),("b",cb)]
# ---- 右半：逐场景 F 与 ΔS
P=json.load(open(f"{V5}/f_decomp_per_scene.json")); EPS=3e-3
MM=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
F=np.concatenate([[max(x["F"],EPS) for x in P[m]] for m in MM])
I=np.concatenate([[x["I"] for x in P[m]] for m in MM])
DS=np.concatenate([[x["dS"] for x in P[m]] for m in MM])
CI=np.concatenate([[COL[m]]*len(P[m]) for m in MM])
# ---- (c) 避让区内每家：买到间距 / 之间 / 更靠近 的格数
c=fig.add_subplot(gs[1,0])
regm={m:np.array([(x["F"]>=0.5) and (x["F"]>x["I"]) for x in P[m]]) for m in MM}
cnt={m:(int(np.sum(regm[m]&(np.array([x["dS"] for x in P[m]])>=0.5))),
        int(np.sum(regm[m]&(np.abs(np.array([x["dS"] for x in P[m]]))<0.5))),
        int(np.sum(regm[m]&(np.array([x["dS"] for x in P[m]])<=-0.5)))) for m in MM}
ys=np.arange(len(MM))[::-1]
for y,m in zip(ys,MM):
    g,n,b=cnt[m]; left=0
    for v,col in ((g,"#2f7d4f"),(n,"#b8b7b0"),(b,"#b3412c")):
        c.barh(y,v,left=left,height=0.62,color=col,lw=0); left+=v
    c.text(left+0.6,y,f"{g}/{g+n+b}",va="center",ha="left",fontsize=5.0,color="#52514e")
c.set_yticks(ys); c.set_yticklabels([SH[m] for m in MM],fontsize=5.2)
c.tick_params(axis="y",length=0,pad=1.5); c.tick_params(axis="x",labelsize=5.2,length=2)
c.set_xlabel("cells in the avoidance region",fontsize=5.8,labelpad=0.5)
c.set_xlim(0,max(sum(v) for v in cnt.values())*1.32)
c.set_title("(c) how many bought clearance",fontsize=6.2,loc="left",pad=2)
for s_ in ("top","right"): c.spines[s_].set_visible(False)
d=fig.add_subplot(gs[1,1])
I=np.maximum(I,EPS)
d.fill_between([EPS,0.5],[0.5,0.5],[40,40],color="#e8f0e8",lw=0,zorder=0)      # F>=0.5 且 F>I
d.fill_between([0.5,40],[0.5,40],[40,40],color="#e8f0e8",lw=0,zorder=0)
d.plot([EPS,40],[EPS,40],color="#6b6a62",ls=(0,(3,2)),lw=0.7,zorder=2)
d.axhline(0.5,color="#9a998f",ls=(0,(1.6,1.6)),lw=0.7,zorder=2)
d.scatter(I,F,s=2.4,c=CI,alpha=0.30,lw=0,zorder=3)      # (d) 只按策略着色；ΔS 的拆分放在 (c)
LOFF={"autovla":(4.5,3.6),"ltf":(4.5,-3.0),"dd":(-4.5,0),"simlingo":(-2,5)}
for m in MM:
    fm=np.mean([x["F"] for x in P[m]]); im=np.mean([x["I"] for x in P[m]])
    d.plot(im,fm,marker="o",ms=4.0,color=COL[m],mec="white",mew=0.5,zorder=6)
    d.annotate(SH[m],(im,fm),textcoords="offset points",xytext=LOFF.get(m,(4.5,0)),ha="right" if m=="dd" else ("center" if m=="simlingo" else "left"),va="center",fontsize=4.8,color=COL[m])
d.set_xscale("log"); d.set_yscale("log"); d.set_xlim(EPS*0.9,55); d.set_ylim(EPS*0.9,30)
d.set_xlabel("$I$ (m)",fontsize=5.8,labelpad=0.5)
d.set_ylabel("$F$ (m)",fontsize=5.8,labelpad=1)
d.tick_params(labelsize=5.2,length=2); d.set_title("(d) all cells on the two axes",fontsize=6.2,loc="left",pad=2)
d.annotate("$F=I$",(0.02,0.02),textcoords="offset points",xytext=(-1,3.5),ha="right",va="bottom",fontsize=4.8,color="#6b6a62",rotation=45)
d.text(0.03,0.95,"avoidance region",transform=d.transAxes,fontsize=5.0,color="#3f6b45",va="top")
hh=[plt.Line2D([],[],marker="s",ms=4,color="#2f7d4f",ls="none",label="bought clearance, $\\Delta S\\geq0.5$ m"),
    plt.Line2D([],[],marker="s",ms=4,color="#b8b7b0",ls="none",label="in region, $|\\Delta S|<0.5$ m"),
    plt.Line2D([],[],marker="s",ms=4,color="#b3412c",ls="none",label="moved towards, $\\Delta S\\leq-0.5$ m"),
    plt.Line2D([],[],marker="o",ms=3.6,color="#52514e",mec="white",ls="none",label="policy mean")]
fig.legend(handles=hh,frameon=False,fontsize=4.8,loc="upper center",bbox_to_anchor=(0.5,0.02),ncol=2,handletextpad=0.3,labelspacing=0.2,columnspacing=1.2)
for s_ in ("top","right"): d.spines[s_].set_visible(False)
fig.savefig(f"{V5}/figures/cases.pdf",bbox_inches="tight",pad_inches=0.015); fig.savefig(f"{V5}/figures/cases.png",dpi=220,bbox_inches="tight",pad_inches=0.015); print("ok")
