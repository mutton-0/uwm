"""核查图：注入反事实速度后，规划到底变了没有。
左：该场景前视原图，红框为目标行人（由 3D 框投影得到）。
右：同一场景、同一模型在 v=2/4/6/8 m/s 下的 clean 规划（实线，越深越快）与 rm 规划（虚线），
    行人真实未来为红色点线，阴影为 ±1 m 走廊，圆点每 0.5 s 一个。
输出 figures/cf_check.png（仅供核查，不进论文）。"""
import json,sys,glob,pickle,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,"/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad")
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts")
from nv_proj import box2d,corners_from_box
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
MODEL=sys.argv[1] if len(sys.argv)>1 else "ddv2"
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
PF=json.load(open(f"{V5}/nv_ped_future.json")); CF=json.load(open(f"{V5}/rhd_cf_{MODEL}.json"))
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def path(w): return lin(T8,np.vstack([[0,0],np.asarray(w,float)[:,:2]]))
SPD=["2","4","6","8"]
ok=[t for t in CF if t in PF and not any(z is None for z in PF[t]["fut"][:5]) and all("rm" in CF[t][v] for v in SPD)]
def fut(t): return lin(T6,np.vstack([PF[t]["p0"],np.asarray(PF[t]["fut"][:5])]))
def clr(X,t): return float(np.min(np.linalg.norm(X-fut(t),axis=1))-1.4)
# 选例：v=8 时盲规划会进 1 m 内，v=2 时不会 —— 注入速度确实把危险造出来了
cand=[t for t in ok if clr(path(CF[t]["8"]["rm"]),t)<1.0 and clr(path(CF[t]["2"]["rm"]),t)>2.0]
TOK=cand[0] if cand else ok[0]
print("选例",TOK,"候选",len(cand))
SC=None
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"]==TOK: SC=s; break
    if SC: break
img=np.asarray(Image.open(f"/data/dataset/navsim/dataset/sensor_blobs/test/{SC['cams']['CAM_F0']['data_path']}").convert("RGB"))
a=SC["anns"]; p0=PF[TOK]["p0"]
j=int(np.argmin([np.hypot(b[0]-p0[0],b[1]-p0[1]) for b in a["gt_boxes"]]))
box=box2d(SC["cams"]["CAM_F0"],corners_from_box(a["gt_boxes"][j]))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8})
fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw=dict(width_ratios=[1.35,1]))
x0,y0,x1,y1=[int(z) for z in box]; pad=260
X0,Y0=max(0,x0-pad),max(0,y0-pad); X1,Y1=min(img.shape[1],x1+pad),min(img.shape[0],y1+pad)
ax[0].imshow(img[Y0:Y1,X0:X1]); ax[0].add_patch(plt.Rectangle((x0-X0,y0-Y0),x1-x0,y1-y0,fill=False,ec="#e34948",lw=2))
ax[0].set_xticks([]); ax[0].set_yticks([]); ax[0].set_title(f"{TOK[:10]}  front camera (red box = target pedestrian)",fontsize=9)
b=ax[1]; F=fut(TOK)
b.axvspan(-1,1,color="#eef1f6",lw=0,zorder=0)
b.plot(-F[:,1],F[:,0],color="#b3412c",ls=":",lw=1.6,zorder=5)
b.plot(-F[0,1],F[0,0],marker="*",ms=14,color="#b3412c",mec="white",zorder=6)
CO=["#bcd4ee","#7fb0e0","#3d7fc6","#14457f"]
for iv,(c,v) in enumerate(zip(CO,SPD)):
    for k,ls,lw in (("clean","-",2.0),("rm","--",1.2)):
        X=path(CF[TOK][v][k]); b.plot(-X[:,1],X[:,0],color=c,ls=ls,lw=lw,zorder=4)
        if k=="clean": b.plot(-X[::5,1],X[::5,0],ls="none",marker="o",ms=3.4,color=c,zorder=4)
    X=path(CF[TOK][v]["rm"])
    b.annotate(f"v={v} m/s   blind clearance {clr(X,TOK):+.2f} m",(-X[-1,1],X[-1,0]),textcoords="offset points",
               xytext=(8,(1.5-iv)*11),fontsize=7.5,color=c,va="center",
               arrowprops=dict(arrowstyle="-",lw=0.5,color=c,shrinkA=0,shrinkB=2))
b.plot(0,0,marker="^",ms=10,color="black",zorder=6)
A=np.concatenate([path(CF[TOK][v][k]) for v in SPD for k in ("clean","rm")]+[fut(TOK)])
b.set_xlim(min(-8,-A[:,1].max()-3),max(14,-A[:,1].min()+9)); b.set_ylim(-2,max(26,A[:,0].max()+3)); b.set_xlabel("lateral (m)"); b.set_ylabel("ahead (m)")
b.set_title(f"{MODEL}: 2.5 s plans at four injected speeds (solid = original, dashed = pedestrian removed)",fontsize=9)
b.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{V5}/figures/cf_check_{MODEL}.png",dpi=170,bbox_inches="tight")
print("ok")
