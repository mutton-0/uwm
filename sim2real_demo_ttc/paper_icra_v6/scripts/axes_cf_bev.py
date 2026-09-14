"""反事实速度版 F/I 轴 · 三家 BEV 规划器（本机 5090）。
动机：NAVSIM 右舵语料 65% 自车停着，盲规划只有 3.6% 的 cell 会进到行人 1 m 内，
"位移≈0"多半是正确行为而非忽视行人。照 nuScenes 体检的做法，对同一批场景注入
反事实自车速度 {2,4,6,8} m/s，把危险造出来，再看策略是否避让。
每个 (场景, 速度) 跑三条输入：clean / rm（抹掉行人，DDv2 同步删框内点云）/ night。
黄昏只在实际速度下做过，这里不重复。速度入口：ego_dynamic_state[0]（DDv2 另有显式入参）。
用法：axes_cf_bev.py dd|ltf|ddv2；环境变量 AXES_TOKENS、AXES_TAG、CF_SPEEDS（默认 2,4,6,8）。
输出 {AXES_TAG}_cf_{MODEL}.json：{token: {v: {clean, rm, night}}}，各 8×0.5 s 路点。"""
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
CASES=json.load(open(os.environ["AXES_TOKENS"])) if os.environ.get("AXES_TOKENS") else []
META=json.load(open(f"{RM}/meta.json"))
SPD=[float(x) for x in os.environ.get("CF_SPEEDS","2,4,6,8").split(",")]
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in CASES: sc[s["token"]]=s
from pypcd4 import PointCloud
def plan(s,img,lid,v):
    e=list(s["ego_dynamic_state"]); e[0]=v; e[1]=0.0        # 注入纵向速度，横向置零（与 nuScenes 体检一致）
    torch.manual_seed(20260909); torch.cuda.manual_seed_all(20260909)
    st=SF.status_feature_official(s.get("driving_command"),[e[0],e[1]],[e[2],e[3]]).to(runner.device)
    if MODEL=="ddv2":
        CUR[0]=st.cpu(); return np.asarray(runner.run(img,float(e[0]),lidar_xyz=lid)["trajectory"],float)[:8,:2].tolist()
    o=m.forward({"camera_feature":image_to_camera_feature(img).to(runner.device),"status_feature":st})
    return np.asarray(o["trajectory"][0].detach().cpu().numpy(),float)[:8,:2].tolist()
OUT=f"{V5}/{os.environ.get('AXES_TAG','rhd')}_cf_{MODEL}.json"
out=json.load(open(OUT)) if os.environ.get("RESUME") and os.path.exists(OUT) else {}
for t in CASES:
    if t in out or t not in sc: continue
    s=sc[t]; a=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"))
    rp=f"{RM}/{t}/CAM_F0_rm.jpg"
    r=np.asarray(Image.open(rp).convert("RGB")) if os.path.exists(rp) else None
    n=transform(a,kind="night",scope="global",seed=zlib.crc32(t.encode())%(2**31))
    lid=lid_rm=None
    if MODEL=="ddv2":
        lid=PointCloud.from_path(os.path.join(BLOB,s["lidar_path"])).numpy()[:,:3].astype(np.float32)
        cor=META.get(t,{}).get("corners"); lid_rm=remove_points(lid,[cor])[0] if cor else lid
    rec={}
    for v in SPD:
        d={"clean":plan(s,a,lid,v),"night":plan(s,n,lid,v)}
        if r is not None: d["rm"]=plan(s,r,lid_rm,v)
        rec[f"{v:g}"]=d
    out[t]=rec
    if len(out)%25==0: print(len(out),"/",len(CASES),flush=True); json.dump(out,open(OUT,"w"))
json.dump(out,open(OUT,"w"))
print(MODEL,len(out),"场景 ×",len(SPD),"档速度","CFDONE")
