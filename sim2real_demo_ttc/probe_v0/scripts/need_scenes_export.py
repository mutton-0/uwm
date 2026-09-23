"""need 场景整理与导出（结果在 results_5090/need_scenes/，不入 git）。
1) 按 cf_need.py 同一判据重算 787 个 need 格 → 211 个 need 场景，按行人相对位置/运动、自车速度分六类，写 results_5090/need_scenes.json
2) 每类挑代表场景，从 NAVSIM 日志导出前视帧 −2 s…+3 s（用 pkl 列表位置当时间轴：frame_idx 在 log 内会重复）、横条 strip.jpg、clip.gif、抹除图与 meta.json
本文件是 2026-09-23 会话中两段内联脚本的合并存档；直接运行即可重建整个目录。"""
import json,glob,pickle,os,shutil,collections,numpy as np
from PIL import Image,ImageDraw,ImageFont
R="/home/boyuewang/120/uwm/sim2real_demo_ttc"; V5=f"{R}/results_5090/paper_icra_v5"; M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"; RM="/data/dataset/navsim_rm"; OUT=f"{R}/results_5090/need_scenes"
PF=json.load(open(f"{V5}/nv_ped_future.json")); toks=json.load(open(f"{R}/probe_v0/data/probe_tokens.json"))
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]; T6=np.r_[0,0.5,1,1.5,2,2.5]
def lin(T,P): P=np.asarray(P,float); return np.stack([np.interp(TT,T,P[:,0]),np.interp(TT,T,P[:,1])],1)
def path(w): return lin(T8,np.vstack([[0,0],np.asarray(w,float)[:,:2]]))
CF={m:json.load(open(f"{V5}/rhd_cf_{m}.json")) for m in M}; AX={m:json.load(open(f"{V5}/rhd_axes4_{m}.json")) for m in M}
FUT={t:lin(T6,np.vstack([PF[t]["p0"],np.asarray(PF[t]["fut"][:5])])) for t in toks}
clr=lambda X,t: float(np.min(np.linalg.norm(X-FUT[t],axis=1))-1.4); S=lambda X,t: float(np.mean(np.minimum(np.linalg.norm(X-FUT[t],axis=1)-1.4,10)))
cells=[]
for m in M:
    for t in toks:
        for sv in ("actual","4","8"):
            v=AX[m][t] if sv=="actual" else CF[m][t].get(sv)
            if v is None or "rm" not in v: continue
            XO,XR=path(v["clean"]),path(v["rm"])
            if clr(XR,t)>=1.0: continue
            F=float(np.mean(np.linalg.norm(XO-XR,axis=1))); dS=S(XO,t)-S(XR,t); I=float(np.mean(np.linalg.norm(XO-path(v["night"]),axis=1))) if "night" in v else np.nan
            cells.append(dict(m=m,t=t,sv=sv,real=bool(F>=0.5 and dS>=0.5 and (np.isnan(I) or F>I)),toward=bool(F>=0.5 and dS<=-0.5)))
def desc(t):
    p=PF[t]; p0=np.array(p["p0"]); fut=[f for f in p["fut"][:5] if f is not None]; f=np.array(fut[-1]) if fut else p0; dx,dy=p0; vx,vy=(f-p0)/(0.5*len(fut)) if fut else (0,0)
    side="left" if dy>0.5 else ("right" if dy<-0.5 else "in-corridor"); lat="toward corridor" if (dy>0.5 and vy<-0.3) or (dy<-0.5 and vy>0.3) else ("away" if abs(vy)>0.3 else "static-lateral")
    lon="walking toward ego" if vx<-0.3 else ("walking away" if vx>0.3 else "static-longitudinal")
    return dict(dist=float(dx),lat=float(dy),side=side,lat_motion=lat,lon_motion=lon,ego_v=float(p["v0"]),ped_speed=float(np.hypot(vx,vy)))
per=collections.defaultdict(lambda:dict(nm=set(),sp=set(),real=[],toward=[]))
for c in cells:
    s=per[c["t"]]; s["nm"].add(c["m"]); s["sp"].add(c["sv"]); s["real"]+=[(c["m"],c["sv"])] if c["real"] else []; s["toward"]+=[(c["m"],c["sv"])] if c["toward"] else []
rows=sorted([dict(token=t,n_need_models=len(s["nm"]),speeds=sorted(s["sp"]),real=s["real"],toward=s["toward"],**desc(t)) for t,s in per.items()],key=lambda r:(-r["n_need_models"],r["dist"]))
os.makedirs(OUT,exist_ok=True); json.dump(rows,open(f"{R}/results_5090/need_scenes.json","w"),indent=1); rows={r["token"]:r for r in rows}
def match(r,side,lat,lon,mov): return r["side"]==side and r["lat_motion"]==lat and r["lon_motion"]==lon and ((r["ego_v"]>=1)==mov)
pick=lambda f,n:[r["token"] for r in rows.values() if f(r)][:n]
TYPES={"A_right_crossing_toward_corridor_ego_stopped":pick(lambda r:match(r,"right","toward corridor","walking toward ego",False),4),
       "B_left_static_near_corridor_ego_moving":pick(lambda r:match(r,"left","static-lateral","static-longitudinal",True),5),
       "C_right_static_curb_ego_stopped":pick(lambda r:match(r,"right","static-lateral","static-longitudinal",False),4),
       "D_in_corridor_static_ego_moving":pick(lambda r:match(r,"in-corridor","static-lateral","static-longitudinal",True),3),
       "E_left_walking_toward_ego_ego_stopped":pick(lambda r:match(r,"left","away","walking toward ego",False),3),
       "F_right_walking_away_ego_moving":pick(lambda r:match(r,"right","static-lateral","walking away",True),3)}
want={t for L in TYPES.values() for t in L}; loc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb")); L=d if isinstance(d,list) else list(d.values())
    for i,s in enumerate(L):
        if s["token"] in want: loc[s["token"]]=(L,i)
try: FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",28)
except Exception: FONT=None
index={}
for typ,tl in TYPES.items():
    for t in tl:
        L,i=loc[t]; s=L[i]; r=rows[t]; t0=s["timestamp"]; d=f"{OUT}/{typ}/{t}"; os.makedirs(d,exist_ok=True); small=[]
        for j in range(max(0,i-4),min(len(L),i+7)):
            q=L[j]; dt=(q["timestamp"]-t0)/1e6; sp=os.path.join(BLOB,q["cams"]["CAM_F0"]["data_path"])
            if q["log_name"]!=s["log_name"] or abs(dt-0.5*(j-i))>0.15 or not os.path.exists(sp): continue
            im=Image.open(sp).convert("RGB"); ImageDraw.Draw(im).text((20,20),f"t = {dt:+.1f} s"+("  <- query frame" if j==i else ""),fill=(255,60,60) if j==i else (255,255,255),font=FONT)
            im.save(f"{d}/f{j-i+4:02d}_t{dt:+.1f}s.jpg",quality=92); small.append(im.resize((960,540)))
        if os.path.exists(f"{RM}/{t}/CAM_F0_rm.jpg"): shutil.copy(f"{RM}/{t}/CAM_F0_rm.jpg",f"{d}/query_removed.jpg")
        if small:
            strip=Image.new("RGB",(480*len(small),270)); [strip.paste(im.resize((480,270)),(480*k,0)) for k,im in enumerate(small)]; strip.save(f"{d}/strip.jpg",quality=90)
            small[0].save(f"{d}/clip.gif",save_all=True,append_images=small[1:],duration=500,loop=0)
        meta={k:r[k] for k in ("dist","lat","side","lat_motion","lon_motion","ego_v","ped_speed","n_need_models","speeds","real","toward")}; meta.update(log=s["log_name"],query_image=s["cams"]["CAM_F0"]["data_path"],n_frames=len(small))
        json.dump(meta,open(f"{d}/meta.json","w"),indent=1); index.setdefault(typ,[]).append(dict(token=t,**meta))
json.dump(index,open(f"{OUT}/index.json","w"),indent=1); print({k:len(v) for k,v in index.items()})
