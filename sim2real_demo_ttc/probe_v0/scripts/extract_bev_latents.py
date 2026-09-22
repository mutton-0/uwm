"""潜空间探针 · 第 1 步的特征抽取（三家 BEV 规划器，本机 5090）。
对 246 个 NAVSIM 右舵近行人场景，在 {实际速度, 4, 8} m/s 三档、{clean, rm, night} 三条件下各前向一次，
记录 8 个 SelfAttention 层的 token 均值（all/vision/lidar 三种池化）与规划轨迹。
推理入口、seed、status 注入与 axes4_bev.py / axes_cf_bev.py 完全一致，所以轨迹可与 rhd_axes4_*.json、rhd_cf_*.json 逐点核对。
用法：extract_bev_latents.py dd|ltf|ddv2 ；输出 probe_v0/data/latents_{MODEL}.npz"""
import os,sys,json,glob,pickle,zlib,time,numpy as np,torch
from PIL import Image
MODEL=sys.argv[1]
ROOT="/data/ruolin/uwm/sim2real_demo_ttc"; RES=f"{ROOT}/results"; S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"; OUTD="/home/boyuewang/120/uwm/sim2real_demo_ttc/probe_v0/data"
sys.path.insert(0,f"{ROOT}/scripts"); sys.path.insert(0,S)
if MODEL=="ddv2": sys.path.insert(0,f"{RES}/ddv2_g1_adapter")
sys.path.insert(0,f"{RES}/ltf_g1_adapter"); sys.path.insert(0,f"{RES}/diffusiondrive_g1_adapter")
import status_fix as SF, dd_adapter as DD; DD.set_crop_center_row(560)
from dd_adapter import image_to_camera_feature, N_IMG_TOK
from appearance_transform import transform
from lidar_edit import remove_points
if MODEL=="dd":    from dd_adapter import DDRunner as R_
elif MODEL=="ltf": from ltf_adapter import LTFRunner as R_
else:              from ddv2_adapter import DDV2Runner as R_
runner=R_(device="cuda:0"); m=runner.agent._transfuser_model if hasattr(runner.agent,"_transfuser_model") else runner.agent
CUR=[None]
if MODEL=="ddv2":
    import ddv2_adapter; ddv2_adapter.status_feature=lambda *a,**k: CUR[0]
BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"; RM="/data/dataset/navsim_rm"
CASES=json.load(open(f"{S}/probe_tokens.json")); META=json.load(open(f"{RM}/meta.json"))
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in CASES: sc[s["token"]]=s
from pypcd4 import PointCloud
def pool():
    """runner._buf: 8 × [320, C_l]；前 256 图像 token，后 64 BEV/lidar latent token。"""
    A=[];Vv=[];L=[]
    for h in runner._buf:
        h=h.float().cpu().numpy(); A.append(h.mean(0)); Vv.append(h[:N_IMG_TOK].mean(0)); L.append(h[N_IMG_TOK:].mean(0))
    return A,Vv,L
def plan(s,img,lid,v):
    e=list(s["ego_dynamic_state"]); e[0]=v; e[1]=0.0 if v!="actual" else e[1]
    torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
    st=SF.status_feature_official(s.get("driving_command"),[e[0],e[1]],[e[2],e[3]]).to(runner.device)
    if MODEL=="ddv2":
        CUR[0]=st.cpu(); tr=np.asarray(runner.run(img,float(e[0]),lidar_xyz=lid)["trajectory"],float)[:8,:2]
    else:
        o=m.forward({"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st})
        tr=np.asarray(o["trajectory"][0].detach().cpu().numpy(),float)[:8,:2]
    return tr,pool()
recs=[]; t0=time.time()
for k,t in enumerate(CASES):
    s=sc[t]; a=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"))
    rp=f"{RM}/{t}/CAM_F0_rm.jpg"; r=np.asarray(Image.open(rp).convert("RGB")) if os.path.exists(rp) else None
    n=transform(a,kind="night",scope="global",seed=zlib.crc32(t.encode())%(2**31))
    lid=lid_rm=None
    if MODEL=="ddv2":
        lid=PointCloud.from_path(os.path.join(BLOB,s["lidar_path"])).numpy()[:,:3].astype(np.float32)
        cor=META.get(t,{}).get("corners"); lid_rm=remove_points(lid,[cor])[0] if cor else lid
    v_act=float(s["ego_dynamic_state"][0])
    for sv,v in (("actual",v_act),("4",4.0),("8",8.0)):
        for cond,img,ld in (("clean",a,lid),("night",n,lid),("rm",r,lid_rm)):
            if img is None: continue
            e=list(s["ego_dynamic_state"]); 
            vv=v
            # 与 axes4（actual 档保留横向速度）/ axes_cf（反事实档横向置零）一致
            if sv=="actual": e_use=e
            else: e_use=[v,0.0,e[2],e[3]]
            torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
            st=SF.status_feature_official(s.get("driving_command"),[e_use[0],e_use[1]],[e_use[2],e_use[3]]).to(runner.device)
            if MODEL=="ddv2":
                CUR[0]=st.cpu(); tr=np.asarray(runner.run(img,float(e_use[0]),lidar_xyz=ld)["trajectory"],float)[:8,:2]
            else:
                o=m.forward({"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st})
                tr=np.asarray(o["trajectory"][0].detach().cpu().numpy(),float)[:8,:2]
            A,Vv,L=pool()
            recs.append(dict(token=t,speed=sv,v=float(e_use[0]),cond=cond,traj=tr.astype(np.float32),
                             all_mean=[x.astype(np.float32) for x in A],vision_mean=[x.astype(np.float32) for x in Vv],lidar_mean=[x.astype(np.float32) for x in L]))
    if (k+1)%20==0: print(f"{MODEL} {k+1}/{len(CASES)}  {time.time()-t0:.0f}s",flush=True)
# 打包：每层一个数组
nl=len(recs[0]["all_mean"]); out={"token":np.array([r["token"] for r in recs]),"speed":np.array([r["speed"] for r in recs]),
     "v":np.array([r["v"] for r in recs],np.float32),"cond":np.array([r["cond"] for r in recs]),"traj":np.stack([r["traj"] for r in recs])}
for j in range(nl):
    for p in ("all_mean","vision_mean","lidar_mean"): out[f"{p}_L{j}"]=np.stack([r[p][j] for r in recs])
np.savez_compressed(f"{OUTD}/latents_{MODEL}.npz",**out)
print(MODEL,"records",len(recs),"layers",nl,"dims",[out[f"all_mean_L{j}"].shape[1] for j in range(nl)],"EXTRACTDONE",flush=True)
