"""NAVSIM 右舵编辑的独立检测器核验（同 det_validate.py 口径）：目标框 = 原图里与投影框 IoU 最大的 person 检测；
移除后是否仍有 IoU≥0.5 的检测；夜化（同 transform、种子按 token）后是否仍检得到。"""
import json,os,sys,glob,pickle,zlib,numpy as np,torch,torchvision
from PIL import Image
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts"); from appearance_transform import transform
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"; OUT="/data/dataset/navsim_rm"; BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"
meta=json.load(open(f"{OUT}/meta.json")); ok={k:v for k,v in meta.items() if "err" not in v}
sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in ok: sc[s["token"]]=s
m=torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").eval().cuda()
def det(a):
    with torch.no_grad(): o=m([torch.from_numpy(a).permute(2,0,1).float().div(255).cuda()])[0]
    k=(o["labels"]==1)&(o["scores"]>=0.5); return o["boxes"][k].cpu().numpy()
def iou(b,c):
    x0,y0=max(b[0],c[0]),max(b[1],c[1]); x1,y1=min(b[2],c[2]),min(b[3],c[3]); i=max(0,x1-x0)*max(0,y1-y0)
    return i/((b[2]-b[0])*(b[3]-b[1])+(c[2]-c[0])*(c[3]-c[1])-i+1e-6)
res={}
for t,v in ok.items():
    a=np.asarray(Image.open(os.path.join(BLOB,sc[t]["cams"]["CAM_F0"]["data_path"])).convert("RGB")); r=np.asarray(Image.open(f"{OUT}/{t}/CAM_F0_rm.jpg").convert("RGB"))
    n=transform(a,kind="night",scope="global",seed=zlib.crc32(t.encode())%(2**31))
    B0=det(a)
    if not len(B0): res[t]={"target":False}; continue
    j=int(np.argmax([iou(b,v["box"]) for b in B0]))
    if iou(B0[j],v["box"])<0.2: res[t]={"target":False}; continue
    tb=B0[j]; res[t]={"target":True,"rm_iou":float(max([iou(tb,b) for b in det(r)],default=0)),"night_iou":float(max([iou(tb,b) for b in det(n)],default=0)),"front_only":v["front_only"]}
json.dump(res,open(f"{V5}/navsim_det_validate.json","w"))
T=[x for x in res.values() if x["target"]]
print("原图检出目标",len(T),"/",len(res),"移除后仍检出 %.1f%%  夜化后仍检出 %.1f%%"%(100*np.mean([x["rm_iou"]>=0.5 for x in T]),100*np.mean([x["night_iou"]>=0.5 for x in T])))
