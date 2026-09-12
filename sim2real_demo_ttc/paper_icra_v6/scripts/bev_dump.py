"""导出 BEV 感知结果（DD / LTF / DDv2 的 _bev_semantic_head，7 类 argmax）与各自规划，供画图。
场景：NAVSIM 近行人（新加坡右舵 260 + 美国左舵 260）。一模型一进程（dd/ltf 与 ddv2 用不同 navsim fork）。
输出 bev_<model>.npz：tokens, sem[uint8, N×H×W], p_ped/p_veh[float16, N×H×W], traj[N×8×2]
BEV 几何（navsim transfuser_features._coords_to_pixel + rot90[::-1] 推得，并用真值车框验证）：
  第 r 行 = 纵向 x = r·0.25 m（第 0 行在车头 0 m，往下越远），第 c 列 = 横向 y = c·0.25 − 32 m（第 0 列在车右侧）"""
import os,sys,json,glob,pickle,numpy as np,torch
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
runner=R_(device="cuda:0"); m=runner.agent._transfuser_model if hasattr(runner.agent,"_transfuser_model") else runner.agent
CAP=[None]; m._bev_semantic_head.register_forward_hook(lambda a,b,o: CAP.__setitem__(0,o))
LOGS="/data/dataset/navsim/dataset/navsim_logs/test"; BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"
want=json.load(open(f"{S}/sg_closevru_tok.json"))["tokens"]+json.load(open(f"{S}/lhd_closevru_tok.json"))["tokens"]
W=set(want); sc={}
for f in sorted(glob.glob(f"{LOGS}/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in W: sc[s["token"]]=s
from pypcd4 import PointCloud
toks=[];sem=[];trj=[];pp=[];pv=[]
for t in want:
    s=sc[t]; img=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB")); e=s["ego_dynamic_state"]
    torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
    st=SF.status_feature_official(s.get("driving_command"),[e[0],e[1]],[e[2],e[3]]).to(runner.device)
    f={"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st}
    if MODEL=="ddv2":
        from ddv2_adapter import lidar_histogram
        x=PointCloud.from_path(os.path.join(BLOB,s["lidar_path"])).numpy()[:,:3].astype(np.float32)
        f["lidar_feature"]=lidar_histogram(x).to(runner.device)
    CAP[0]=None; out=None
    with torch.no_grad():
        try: out=m.forward(f)
        except Exception as ex:
            if type(ex).__name__!="_TrajReady": raise
    if CAP[0] is None: continue
    L=CAP[0][0].float(); P=torch.softmax(L,0).cpu().numpy()
    sem.append(P.argmax(0).astype(np.uint8)); pp.append(P[6].astype(np.float16)); pv.append(P[5].astype(np.float16)); toks.append(t)
    trj.append(np.asarray(out["trajectory"][0].detach().cpu().numpy())[:8,:2] if isinstance(out,dict) and "trajectory" in out else np.full((8,2),np.nan))
np.savez_compressed(f"{V5}/bev_{MODEL}.npz",tokens=np.array(toks),sem=np.stack(sem),p_ped=np.stack(pp),p_veh=np.stack(pv),traj=np.stack(trj))
print(MODEL,len(toks),"BEVDONE",np.stack(sem).shape)
