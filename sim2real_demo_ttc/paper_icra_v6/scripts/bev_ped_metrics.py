"""用正确的 BEV 几何（第 r 行 = x = r·0.25，第 c 列 = y = c·0.25 − 32）重算"语义头是否看到行人"（NAVSIM 近行人 520 场景）：
  定位比   目标行人中心 ±1 m 内的行人通道峰值 / 全图行人通道峰值；对照：同距离、镜像到另一侧的位置
  argmax   目标行人框内被判为行人类的像素比例；另报全图行人类 argmax 像素数
  行人/车辆  全图行人通道峰值 / 全图车辆通道峰值"""
import json,glob,pickle,numpy as np
from matplotlib.path import Path
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
PF=json.load(open(f"{V5}/nv_ped_future.json"))
B={m:np.load(f"{V5}/bev_{m}.npz") for m in ("dd","ltf","ddv2")}
want=set(B["dd"]["tokens"]); sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in want: sc[s["token"]]=s
H,W=128,256; R,C=np.meshgrid(np.arange(H),np.arange(W),indexing="ij"); X=(R+0.5)*0.25; Y=(C+0.5)*0.25-32
out={}
for m,b in B.items():
    loc=[];loc_ctl=[];am=[];ratio=[];glob_ped=0
    for i,t in enumerate(b["tokens"]):
        p=PF.get(t); s=sc[t]
        if p is None: continue
        pp=b["p_ped"][i].astype(np.float32); pv=b["p_veh"][i].astype(np.float32); sem=b["sem"][i]
        x0,y0=p["p0"]
        if not (0<x0<32): continue
        near=(np.abs(X-x0)<=1)&(np.abs(Y-y0)<=1); ctl=(np.abs(X-x0)<=1)&(np.abs(Y+y0)<=1) if abs(y0)>2 else (np.abs(X-x0)<=1)&(np.abs(Y-(y0+4))<=1)
        g=pp.max()
        if g>0: loc.append(pp[near].max()/g); loc_ctl.append(pp[ctl].max()/g)
        a=s["anns"]; j=int(np.argmin([np.hypot(bb[0]-x0,bb[1]-y0) for bb in a["gt_boxes"]])); bb=a["gt_boxes"][j]
        x,y,_,l,w,_,yaw=bb[:7]; l,w=max(l,0.6),max(w,0.6); c_,s_=np.cos(yaw),np.sin(yaw)
        poly=Path(np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])@np.array([[c_,s_],[-s_,c_]])+[x,y])
        ins=poly.contains_points(np.c_[X.ravel(),Y.ravel()]).reshape(H,W)
        if ins.sum()>0: am.append(float((sem[ins]==6).mean()))
        ratio.append(pp.max()/max(pv.max(),1e-6)); glob_ped+=int((sem==6).sum())
    out[m]=dict(n=len(loc),loc_med=float(np.median(loc)),loc_ctl_med=float(np.median(loc_ctl)),loc_gt_ctl=float(np.mean(np.array(loc)>np.array(loc_ctl))),
                argmax_in_box=float(100*np.mean(am)),ped_veh_ratio=float(np.median(ratio)),ped_argmax_pixels_total=glob_ped)
    o=out[m]; print(f"{m:5s} n={o['n']} 定位比 {o['loc_med']:.2f}（镜像对照 {o['loc_ctl_med']:.2f}，高于对照 {100*o['loc_gt_ctl']:.0f}%） 行人框内 argmax=行人 {o['argmax_in_box']:.1f}%  行人/车辆峰值 {o['ped_veh_ratio']:.3f}  全图行人 argmax 像素 {o['ped_argmax_pixels_total']}")
json.dump(out,open(f"{V5}/bev_ped_metrics.json","w"),indent=1)
