"""NAVSIM 近行人场景：BEV 感知结果 + 六模型轨迹图。
每张图 5 栏：前视图 | DD / LTF / DDv2 的 BEV 语义分割（argmax，7 类）叠加各自规划 | 真值目标框 + 行人真实未来 + 六模型规划。
BEV 地图几何：128×256，0.25 m/像素；第 r 行 = x = r·0.25 m（第 0 行在车头），第 c 列 = y = c·0.25 − 32（第 0 列在车右侧）。
画图坐标：横轴 = −y（车左侧在左），纵轴 = x（向前）。
文件名：<看到行人仍 TTC<1.5 s 的模型数>_<城市>_<token>.png，数字越大越该先看。"""
import os,sys,json,glob,pickle,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
OUT=_os.environ.get("BEV_OUT",f"{R5}/fill_examples/bev_navsim") if (_os:=__import__("os")) else None; os.makedirs(f"{OUT}/right_hand_SG",exist_ok=True); os.makedirs(f"{OUT}/left_hand_US",exist_ok=True)
ONLY=set(sys.argv[1].split(",")) if len(sys.argv)>1 else None
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
SEMC=ListedColormap(["#ffffff","#d9d9d9","#efe3c8","#8c8c8c","#a0785a","#6fa8dc","#e34948"])
SEMN=["background","road","walkway","centerline","static","vehicle","pedestrian"]
B={m:np.load(f"{V5}/bev_{m}.npz") for m in ("dd","ltf","ddv2")}
BI={m:{t:i for i,t in enumerate(B[m]["tokens"])} for m in B}
TR={}
import os as _os
def _pick(a,b): return b if _os.path.exists(b) else a     # 有带导航的版本就用带导航的
for m in M:
    TR[m]={**json.load(open(_pick(f"{R5}/nvtraj_{m}_sg-one-north_closevru.json",f"{R5}/nvtraj_{m}_sg-one-north_closevru_nav.json"))),
           **json.load(open(_pick(f"{R5}/nvtraj_{m}_any_lhdclose.json",f"{R5}/nvtraj_{m}_any_lhdclose_nav.json")))}
PF=json.load(open(f"{V5}/nv_ped_future.json"))
LOGS="/data/dataset/navsim/dataset/navsim_logs/test"; BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"
want=set(B["dd"]["tokens"]); sc={}
for f in sorted(glob.glob(f"{LOGS}/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in want: sc[s["token"]]=s
sys.path.insert(0,S); from nv_proj import box2d,corners_from_box
TT=np.round(np.arange(0,2.51,0.1),2)
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def min_ttc(w,p):
    if p is None or any(z is None for z in p["fut"][:5]): return None
    w=np.asarray(w,float)[:,:2]; X=lin(np.r_[0,np.arange(1,len(w)+1)*0.5],np.vstack([[0,0],w]))
    F=lin(np.r_[0,0.5,1,1.5,2,2.5],np.vstack([p["p0"],np.asarray(p["fut"][:5])])); c=np.linalg.norm(X-F,axis=1)-1.4
    if c.min()<=0: return 0.0
    cl=-np.gradient(c,TT); ok=cl>0.05
    return float(min(5.0,np.min(c[ok]/cl[ok]))) if ok.any() else 5.0
def rect(b):
    x,y,_,l,w,_,yaw=b[:7]; c,s_=np.cos(yaw),np.sin(yaw); d=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2],[l/2,w/2]])
    P=d@np.array([[c,s_],[-s_,c]])+[x,y]; return -P[:,1],P[:,0]
def draw_gt(ax,s,p,thin=False):
    a=s["anns"]
    for bb,nm in zip(a["gt_boxes"],a["gt_names"]):
        X,Y=rect(bb); isp=str(nm) in ("pedestrian","bicycle")
        ax.plot(X,Y,color="#e34948" if isp else "#3d3d3d",lw=0.9 if isp else (0.5 if thin else 0.7))
    if p is not None:
        fut=[p["p0"]]+[z for z in p["fut"] if z is not None]; F=np.asarray(fut)
        ax.plot(-F[:,1],F[:,0],color="#e34948",lw=1.2,ls=(0,(1.5,1)),marker=".",ms=3)
def draw_traj(ax,w,col,lw=1.6,label=None):
    w=np.vstack([[0,0],np.asarray(w,float)[:,:2]]); ax.plot(-w[:,1],w[:,0],color=col,lw=lw,marker="o",ms=2.2,label=label)
def setax(ax,title):
    ax.set_xlim(-18,18); ax.set_ylim(-4,32); ax.set_aspect("equal"); ax.set_title(title,fontsize=8,pad=2)
    ax.tick_params(labelsize=6,length=2); ax.plot(0,0,marker="^",color="black",ms=5)
for t,s in sc.items():
    if ONLY and t not in ONLY: continue
    p=PF.get(t); side="right_hand_SG" if s["map_location"]=="sg-one-north" else "left_hand_US"
    ttc={m:min_ttc(TR[m][t],p) for m in M if t in TR[m]}
    nbad=sum(1 for v in ttc.values() if v is not None and v<1.5)
    e=s["ego_dynamic_state"]; v0=float(np.hypot(e[0],e[1]))
    fig,axs=plt.subplots(1,5,figsize=(15,3.6),gridspec_kw={"width_ratios":[1.55,1,1,1,1],"wspace":0.12})
    img=Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"); axs[0].imshow(img); axs[0].axis("off")
    if p is not None:
        a=s["anns"]; j=int(np.argmin([np.hypot(bb[0]-p["p0"][0],bb[1]-p["p0"][1]) for bb in a["gt_boxes"]]))
        bx=box2d(s["cams"]["CAM_F0"],corners_from_box(a["gt_boxes"][j]))
        if bx is not None: axs[0].add_patch(plt.Rectangle((bx[0],bx[1]),bx[2]-bx[0],bx[3]-bx[1],fill=False,ec="#e34948",lw=1.5))
    axs[0].set_title(f"{s['map_location']}  ego {v0:.1f} m/s  target {p['cat'] if p else '?'} at {p['p0'][0]:.1f} m" if p else s["map_location"],fontsize=8,pad=2)
    for k,m in enumerate(("dd","ltf","ddv2")):
        ax=axs[1+k]; i=BI[m].get(t)
        if i is not None: ax.imshow(B[m]["sem"][i][:,::-1],cmap=SEMC,vmin=0,vmax=6,extent=[-32,32,0,32],origin="lower",interpolation="nearest")
        draw_gt(ax,s,p,thin=True); draw_traj(ax,TR[m][t],COL[m])
        v=ttc.get(m); setax(ax,f"{NAME[m]} BEV perception · TTC {'—' if v is None else f'{v:.1f}s'}")
    ax=axs[4]; draw_gt(ax,s,p)
    for m in M:
        if t in TR[m]:
            v=ttc.get(m); draw_traj(ax,TR[m][t],COL[m],lw=1.3,label=f"{NAME[m]} ({'—' if v is None else f'{v:.1f}s'})")
    setax(ax,"ground truth + six plans (4 s)"); ax.legend(fontsize=5.3,loc="upper left",frameon=False,handlelength=1.2,labelspacing=0.15)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=SEMC(i)) for i in range(1,7)],SEMN[1:],loc="lower center",ncol=6,fontsize=6.5,frameon=False,bbox_to_anchor=(0.6,-0.02))
    fig.savefig(f"{OUT}/{side}/{nbad}_{s['map_location'].split('-')[0]}_{t}.png",dpi=110,bbox_inches="tight"); plt.close(fig)
print("BEVPLOTDONE")
