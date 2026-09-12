"""标准 nuScenes 开环评测（UniAD/VAD 口径）在我们同一批帧上：原图 vs 移除行人。
  L2@t：规划点与人类实际未来位置（当前 ego 系）的距离，t = 1.0 / 2.0 s，另报 0.5–2.5 s 平均
  碰撞率：自车框（4.084 × 1.85 m，中心在规划点、朝向取轨迹切向）在未来各关键帧（≈0.5 s 间隔）
          与该帧全部标注物体框（任何类别）是否相交；另单独报与行人框相交的比例；追尾（物体在自车后方）不计入前方碰撞与行人碰撞
只读 nuScenes 元数据表（逐表加载、筛完即释放，控内存）。"""
import json,glob,gc,numpy as np
from pyquaternion import Quaternion
from shapely.geometry import Polygon
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"; NU="/data/dataset/nuscenes/v1.0-trainval/v1.0-trainval"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); REC=[x for k in ("A","B") for x in MAN[k]]
S={s["token"]:s for s in json.load(open(f"{NU}/sample.json"))}
need=set(); chain={}
for x in REC:
    t=x["sample"]; c=[t]
    for _ in range(5):
        n=S[c[-1]]["next"]
        if not n: break
        c.append(n)
    chain[x["uid"]]=c; need|=set(c)
print("需要的样本",len(need),flush=True)
sd={}
for r in json.load(open(f"{NU}/sample_data.json")):
    if r["is_key_frame"] and r["sample_token"] in need and "LIDAR_TOP" in r["filename"]: sd[r["sample_token"]]=r["ego_pose_token"]
gc.collect(); print("sample_data 筛完",len(sd),flush=True)
ept=set(sd.values()); EP={}
for r in json.load(open(f"{NU}/ego_pose.json")):
    if r["token"] in ept: EP[r["token"]]=r
gc.collect(); print("ego_pose 筛完",len(EP),flush=True)
INS={r["token"]:r["category_token"] for r in json.load(open(f"{NU}/instance.json"))}
CAT={r["token"]:r["name"] for r in json.load(open(f"{NU}/category.json"))}
ANN={}
for r in json.load(open(f"{NU}/sample_annotation.json")):
    if r["sample_token"] in need: ANN.setdefault(r["sample_token"],[]).append((r["translation"],r["size"],r["rotation"],CAT[INS[r["instance_token"]]]))
gc.collect(); print("标注筛完",sum(len(v) for v in ANN.values()),flush=True)
def to_ego(p0,q0,pts): return (q0.inverse.rotation_matrix@(np.asarray(pts)-p0).T).T
def box_poly(cx,cy,yaw,l,w):
    c,s=np.cos(yaw),np.sin(yaw); d=np.array([[l/2,w/2],[l/2,-w/2],[-l/2,-w/2],[-l/2,w/2]])
    return Polygon(d@np.array([[c,s],[-s,c]])+[cx,cy])
GEO={}
for x in REC:
    c=chain[x["uid"]]
    if len(c)<6 or any(t not in sd for t in c): continue
    e0=EP[sd[c[0]]]; p0=np.array(e0["translation"]); q0=Quaternion(e0["rotation"]); t0=S[c[0]]["timestamp"]
    ts=np.array([(S[t]["timestamp"]-t0)/1e6 for t in c[1:]])
    hum=to_ego(p0,q0,[EP[sd[t]]["translation"] for t in c[1:]])[:,:2]
    boxes=[]
    for t in c[1:]:
        bb=[]
        for tr,sz,rot,cat in ANN.get(t,[]):
            ce=to_ego(p0,q0,[tr])[0]; yaw=(q0.inverse*Quaternion(rot)).yaw_pitch_roll[0]
            bb.append((box_poly(ce[0],ce[1],yaw,sz[1],sz[0]),cat.startswith("human"),cat))
        boxes.append(bb)
    GEO[x["uid"]]=dict(ts=ts,hum=hum,boxes=boxes)
print("几何就绪",len(GEO),flush=True)
TQ=np.array([0.5,1.0,1.5,2.0,2.5])
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
def evalp(plan,g):
    P=np.vstack([[0,0],np.asarray(plan)]); T=np.r_[0,TQ]
    at=lambda t: np.array([np.interp(t,T,P[:,0]),np.interp(t,T,P[:,1])])
    hum_at=lambda t: np.array([np.interp(t,g["ts"],g["hum"][:,0]),np.interp(t,g["ts"],g["hum"][:,1])])
    l2={t:float(np.linalg.norm(at(t)-hum_at(t))) for t in (1.0,2.0)}; l2["avg"]=float(np.mean([np.linalg.norm(at(t)-hum_at(t)) for t in TQ]))
    col=False; colp=False; colf=False; kinds=[]
    for k,t in enumerate(g["ts"]):
        if t>2.55: break
        a=at(max(t-0.05,0)); b=at(t+0.05) if t+0.05<=2.5 else at(t); p=at(t)
        yaw=np.arctan2(b[1]-a[1],b[0]-a[0]) if np.linalg.norm(b-a)>1e-3 else 0.0
        eb=box_poly(p[0],p[1],yaw,4.084,1.85)
        for poly,hum,cat in g["boxes"][k]:
            if eb.intersects(poly):
                col=True
                # 物体中心相对自车（规划点、朝向）的纵向位置：<0 = 在自车后方（追尾，用户明确不关注，只作参考）
                cxy=np.array(poly.centroid.coords[0])-p; lon=cxy@np.array([np.cos(yaw),np.sin(yaw)])
                kinds.append((cat.split(".")[0],"front" if lon>0 else "rear"))
                if lon>0: colf=True; colp=colp or hum      # 行人碰撞也只算前方
    return l2,col,colp,colf,kinds
MAN_IDX={x["uid"]:x for x in REC}
sel=lambda x: x["front_only"] and float(x["dep"])<=15 and (x["set"]=="B" or x.get("grp")=="corr")
out={}; KND={}
import collections
for m in ["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]:
    C={**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")}
    res={"clean":[],"rm":[]}; side={"LHD":{"clean":[],"rm":[]},"RHD":{"clean":[],"rm":[]}}
    for u,r in C.items():
        if u not in GEO or "actual" not in r or not sel(MAN_IDX[u]): continue
        for k in ("clean","rm"):
            l2,col,colp,colf,kinds=evalp(r["actual"][k],GEO[u]); res[k].append((l2[1.0],l2[2.0],l2["avg"],col,colp,colf)); KND.setdefault((m,k),[]).extend(kinds)
            side[MAN_IDX[u]["side"]][k].append((l2[1.0],l2[2.0],l2["avg"],col,colp,colf))
    o={}
    for k,v in res.items():
        a=np.array(v,float); o[k]=dict(n=len(a),L2_1=float(a[:,0].mean()),L2_2=float(a[:,1].mean()),L2_avg=float(a[:,2].mean()),col=float(100*a[:,3].mean()),col_ped=float(100*a[:,4].mean()),col_front=float(100*a[:,5].mean()))
    for sd,rr in side.items():
        a=np.array(rr["clean"],float); o[f"clean_{sd}"]=dict(n=len(a),L2_avg=float(a[:,2].mean()),col_front=float(100*a[:,5].mean()),col_ped=float(100*a[:,4].mean()))
    out[m]=o
    print("   碰撞对象（原图）:",collections.Counter(KND.get((m,"clean"),[])).most_common(6),f" 只算前方碰撞 {o['clean']['col_front']:.1f}/{o['rm']['col_front']:.1f}",flush=True)
    print(f"{m:9s} n={o['clean']['n']}  L2@1s {o['clean']['L2_1']:.2f}/{o['rm']['L2_1']:.2f}  L2@2s {o['clean']['L2_2']:.2f}/{o['rm']['L2_2']:.2f}  avg {o['clean']['L2_avg']:.2f}/{o['rm']['L2_avg']:.2f}  col% {o['clean']['col']:.1f}/{o['rm']['col']:.1f}  ped col% {o['clean']['col_ped']:.1f}/{o['rm']['col_ped']:.1f}",flush=True)
json.dump(out,open(f"{V5}/nusc_openloop.json","w"),indent=1)
