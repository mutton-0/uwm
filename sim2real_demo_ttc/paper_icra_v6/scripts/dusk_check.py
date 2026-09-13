"""黄昏 vs 夜化：在右舵 254 个近行人场景上，比较独立检测器还能不能看到目标 VRU。
夜化在 NAVSIM 上只保住 38.8% 的目标——"与行人无关的编辑"却把行人抹掉了一大半，
这会让 I 轴的解释力打折。黄昏参数更温和（gamma 1.45 / gain 0.70 / sat 0.80 / 暖色 / 噪声 2）。
检测器：Faster R-CNN（COCO，person 类，score≥0.5），与 §III 用的同一个。
输出 dusk_check.json：逐帧 orig/night/dusk 的目标 IoU。"""
import json,glob,pickle,sys,zlib,os,numpy as np,torch
from PIL import Image
S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
sys.path.insert(0,S); sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts")
from nv_proj import box2d,corners_from_box
from appearance_transform import transform
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"
PF=json.load(open(f"{V5}/nv_ped_future.json"))
want=set(json.load(open(f"{V5}/rhd_axes_dd.json")))
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in want: sc[s["token"]]=s
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
dev="cuda" if torch.cuda.is_available() else "cpu"
net=fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
def iou(a,b):
    x0,y0=max(a[0],b[0]),max(a[1],b[1]); x1,y1=min(a[2],b[2]),min(a[3],b[3])
    if x1<=x0 or y1<=y0: return 0.0
    i=(x1-x0)*(y1-y0); return i/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i)
@torch.no_grad()
def best_person(img,tgt):
    t=torch.from_numpy(img).permute(2,0,1).float().div(255)[None].to(dev)
    o=net(t)[0]
    m=(o["labels"]==1)&(o["scores"]>=0.5)
    return max([iou(tgt,b) for b in o["boxes"][m].cpu().numpy()],default=0.0)
out={}
for k,(t,s) in enumerate(sc.items()):
    p=PF.get(t)
    if p is None: continue
    a=s["anns"]; j=int(np.argmin([np.hypot(bb[0]-p["p0"][0],bb[1]-p["p0"][1]) for bb in a["gt_boxes"]]))
    b=box2d(s["cams"]["CAM_F0"],corners_from_box(a["gt_boxes"][j]))
    if b is None: continue
    img=np.asarray(Image.open(f"{BLOB}/{s['cams']['CAM_F0']['data_path']}").convert("RGB"))
    sd=zlib.crc32(t.encode())%(2**31)
    rec={"orig":best_person(img,b),
         "night":best_person(transform(img,kind="night",scope="global",seed=sd),b),
         "dusk":best_person(transform(img,kind="dusk",scope="global",seed=sd),b)}
    rp=f"/data/dataset/navsim_rm/{t}/CAM_F0_rm.jpg"
    if os.path.exists(rp):
        rimg=np.asarray(Image.open(rp).convert("RGB"))
        rec["removed"]=best_person(rimg,b)
        rec["removed_dusk"]=best_person(transform(rimg,kind="dusk",scope="global",seed=sd),b)
    out[t]=rec
    if (k+1)%50==0: print(f"  {k+1}/{len(sc)}",flush=True)
json.dump(out,open(f"{V5}/dusk_check.json","w"),indent=1)
import numpy as np
for kk in ("orig","removed","night","dusk","removed_dusk"):
    v=np.array([o[kk] for o in out.values() if kk in o])
    print(f"{kk:6s} 目标检出率(IoU≥0.5) {100*np.mean(v>=0.5):5.1f}%   IoU 中位 {np.median(v):.2f}")
