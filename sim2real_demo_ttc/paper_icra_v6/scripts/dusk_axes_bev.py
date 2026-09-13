"""黄昏版 I 轴 · 三家 BEV 规划器（本机 5090）。用法：dusk_axes_bev.py dd|ltf|ddv2
对右舵 254 个近行人场景，同一策略跑两次：原图 clean 与黄昏 dusk（点云不变，天色不影响激光）。
移除版 rm 沿用已有的 rhd_axes_<m>.json，不重跑。
黄昏参数写死在此处，避免依赖 exx 上 appearance_transform 的版本：
  gamma 1.45 / gain 0.70 / sat 0.80 / 暖色 (1.06,0.97,0.92) / 噪声 2
输出 rhd_axes_dusk_<m>.json：{token: {"clean":[8x2], "dusk":[8x2]}}"""
import os,sys,json,glob,pickle,zlib,numpy as np,torch
from PIL import Image
MODEL=sys.argv[1]
ROOT="/data/ruolin/uwm/sim2real_demo_ttc"; RES=f"{ROOT}/results"
S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
sys.path.insert(0,f"{ROOT}/scripts"); sys.path.insert(0,S)
if MODEL=="ddv2": sys.path.insert(0,f"{RES}/ddv2_g1_adapter")
sys.path.insert(0,f"{RES}/ltf_g1_adapter"); sys.path.insert(0,f"{RES}/diffusiondrive_g1_adapter")
import status_fix as SF, dd_adapter as DD; DD.set_crop_center_row(560)
from dd_adapter import image_to_camera_feature
if MODEL=="dd":    from dd_adapter import DDRunner as R_
elif MODEL=="ltf": from ltf_adapter import LTFRunner as R_
else:              from ddv2_adapter import DDV2Runner as R_
DUSK=dict(gamma=1.45,sat=0.80,tint=(1.06,0.97,0.92),noise=2.0,gain=0.70)
def dusk(a,seed):
    rng=np.random.default_rng(seed); x=a.astype(np.float32)/255.0
    x=np.power(np.clip(x,0,1),DUSK["gamma"])*DUSK["gain"]
    lum=x@np.array([0.299,0.587,0.114],np.float32); x=lum[...,None]+(x-lum[...,None])*DUSK["sat"]
    x=x*np.asarray(DUSK["tint"],np.float32)[None,None,:]*255.0
    x=x+rng.normal(0,DUSK["noise"],x.shape)
    return np.clip(x,0,255).astype(np.uint8)
runner=R_(device="cuda:0"); m=runner.agent._transfuser_model if hasattr(runner.agent,"_transfuser_model") else runner.agent
BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"
want=list(json.load(open(f"{V5}/rhd_axes_dd.json")))
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in set(want): sc[s["token"]]=s
from pypcd4 import PointCloud
def plan(img,s):
    torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
    e=s["ego_dynamic_state"]
    st=SF.status_feature_official(s.get("driving_command"),[e[0],e[1]],[e[2],e[3]]).to(runner.device)
    f={"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st}
    if MODEL=="ddv2":
        from ddv2_adapter import lidar_histogram
        x=PointCloud.from_path(os.path.join(BLOB,s["lidar_path"])).numpy()[:,:3].astype(np.float32)
        f["lidar_feature"]=lidar_histogram(x).to(runner.device)
    with torch.no_grad():
        try: out=m.forward(f,cal_pdm=False) if MODEL=="ddv2" else m.forward(f)
        except Exception as ex:
            if type(ex).__name__!="_TrajReady": raise
            tr=ex.traj[0,-1].float().cpu().numpy()         # DDv2 官方出口：最后一次 fine 精化的轨迹
            return np.asarray(tr)[:8,:2].tolist()
    return np.asarray(out["trajectory"][0].detach().cpu().numpy())[:8,:2].tolist() if isinstance(out,dict) and "trajectory" in out else None
out={}
for k,t in enumerate(want):
    s=sc.get(t)
    if s is None: continue
    img=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"))
    a=plan(img,s); b=plan(dusk(img,zlib.crc32(t.encode())%(2**31)),s)
    if a is None or b is None: continue
    out[t]={"clean":a,"dusk":b}
    if (k+1)%50==0: print(f"  {k+1}/{len(want)}",flush=True); json.dump(out,open(f"{V5}/rhd_axes_dusk_{MODEL}.json","w"))
json.dump(out,open(f"{V5}/rhd_axes_dusk_{MODEL}.json","w")); print(MODEL,len(out),"DUSKDONE")
