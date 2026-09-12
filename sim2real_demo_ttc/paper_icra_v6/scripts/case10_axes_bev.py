"""10 例的 F / I 轴（本机三家 BEV）：同一场景跑三种输入 —— 原图 / 抹掉目标行人（SAM+LaMa，DDv2 同步删 3D 框内点云）/ 全图夜化。
输出 case10_axes_dd|ltf|ddv2.json：{token: {clean, rm, night}}，各 8×0.5 s 路点（NAVSIM ego 系）。"""
import os,sys,json,glob,pickle,zlib,numpy as np,torch
from PIL import Image
MODEL=sys.argv[1]
ROOT="/data/ruolin/uwm/sim2real_demo_ttc"; RES=f"{ROOT}/results"; S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
sys.path.insert(0,f"{ROOT}/scripts"); sys.path.insert(0,S)
if MODEL=="ddv2": sys.path.insert(0,f"{RES}/ddv2_g1_adapter")
sys.path.insert(0,f"{RES}/ltf_g1_adapter"); sys.path.insert(0,f"{RES}/diffusiondrive_g1_adapter")
import status_fix as SF, dd_adapter as DD; DD.set_crop_center_row(560)
from dd_adapter import image_to_camera_feature
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
CASES=json.load(open(os.environ["AXES_TOKENS"])) if os.environ.get("AXES_TOKENS") else [r["token"] for r in json.load(open(f"{V5}/case10.json"))]; META=json.load(open(f"{RM}/meta.json"))
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in CASES: sc[s["token"]]=s
from pypcd4 import PointCloud
def plan(s,img,lid):
    e=s["ego_dynamic_state"]; torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
    st=SF.status_feature_official(s.get("driving_command"),[e[0],e[1]],[e[2],e[3]]).to(runner.device)
    if MODEL=="ddv2":
        CUR[0]=st.cpu(); return np.asarray(runner.run(img,float(e[0]),lidar_xyz=lid)["trajectory"],float)[:8,:2].tolist()
    o=m.forward({"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st})
    return np.asarray(o["trajectory"][0].detach().cpu().numpy(),float)[:8,:2].tolist()
out={}
for t in CASES:
    s=sc[t]; a=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"))
    r=np.asarray(Image.open(f"{RM}/{t}/CAM_F0_rm.jpg").convert("RGB")) if os.path.exists(f"{RM}/{t}/CAM_F0_rm.jpg") else None
    n=transform(a,kind="night",scope="global",seed=zlib.crc32(t.encode())%(2**31))
    lid=lid_rm=None
    if MODEL=="ddv2":
        lid=PointCloud.from_path(os.path.join(BLOB,s["lidar_path"])).numpy()[:,:3].astype(np.float32)
        cor=META.get(t,{}).get("corners")
        lid_rm=remove_points(lid,[cor])[0] if cor else lid
    out[t]={"clean":plan(s,a,lid),"night":plan(s,n,lid)}
    if r is not None: out[t]["rm"]=plan(s,r,lid_rm)
json.dump(out,open(f"{V5}/{os.environ.get('AXES_TAG','case10')}_axes_{MODEL}.json","w"))
print(MODEL,len(out),"AXESDONE")
