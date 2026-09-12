"""NAVSIM 新加坡（右舵）近行人场景的反事实渲染：移除目标行人（投影 3D 框 → SAM 框+躯干点提示 → 裁到框内 → 膨胀 9 px → LaMa）。
与 nuScenes 体检同一套编辑。目标行人 = nv_ped_future 里的走廊最近 VRU。
另记该行人是否也出现在 CAM_L0 / CAM_R0（VLA 会看到侧前相机；只在前视可见的场景才进入 VLA 的主结果）。
输出 /data/dataset/navsim_rm/<token>/CAM_F0_rm.jpg + meta.json"""
import json,os,sys,glob,pickle,numpy as np,cv2
from PIL import Image
S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"; sys.path.insert(0,S)
from nv_proj import box2d,corners_from_box
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
OUT="/data/dataset/navsim_rm"; os.makedirs(OUT,exist_ok=True); CKPT="/data/dataset/ckpt/sam_vit_b_01ec64.pth"
from segment_anything import sam_model_registry, SamPredictor
from simple_lama_inpainting import SimpleLama
sam=sam_model_registry["vit_b"](checkpoint=CKPT).to("cuda"); sam.eval(); pred=SamPredictor(sam); lama=SimpleLama()
PF=json.load(open(f"{V5}/nv_ped_future.json"))
want=set(json.load(open(f"{S}/sg_closevru_tok.json"))["tokens"])
BLOB="/data/dataset/navsim/dataset/sensor_blobs/test"; sc={}
for f in sorted(glob.glob("/data/dataset/navsim/dataset/navsim_logs/test/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        if s["token"] in want: sc[s["token"]]=s
meta={}
for t,s in sc.items():
    p=PF.get(t)
    if p is None: continue
    a=s["anns"]; j=int(np.argmin([np.hypot(bb[0]-p["p0"][0],bb[1]-p["p0"][1]) for bb in a["gt_boxes"]])); cor=corners_from_box(a["gt_boxes"][j])
    b=box2d(s["cams"]["CAM_F0"],cor)
    side={c:(box2d(s["cams"][c],cor) is not None) for c in ("CAM_L0","CAM_R0") if c in s["cams"]}
    rec={"box":b,"front_only":not any(side.values()),"side_visible":side}
    if b is None: rec["err"]="not in CAM_F0"; meta[t]=rec; continue
    img=np.asarray(Image.open(os.path.join(BLOB,s["cams"]["CAM_F0"]["data_path"])).convert("RGB"))
    pred.set_image(img); pt=np.array([[(b[0]+b[2])/2,b[1]+0.72*(b[3]-b[1])]],np.float32)
    m,scs,_=pred.predict(point_coords=pt,point_labels=np.array([1]),box=np.array(b,np.float32),multimask_output=True)
    mm=m[int(np.argmax(scs))]; keep=np.zeros_like(mm); x0,y0,x1,y1=[int(round(v)) for v in b]; keep[max(0,y0):y1,max(0,x0):x1]=1; mm=mm&keep.astype(bool)
    if mm.sum()<20: rec["err"]="empty mask"; meta[t]=rec; continue
    mk=cv2.dilate(mm.astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)),iterations=1)
    r=np.asarray(lama(Image.fromarray(img),Image.fromarray(mk)).convert("RGB"))[:img.shape[0],:img.shape[1]]
    os.makedirs(f"{OUT}/{t}",exist_ok=True); Image.fromarray(r).save(f"{OUT}/{t}/CAM_F0_rm.jpg",quality=95)
    rec["maskfrac"]=float((mk>0).mean()); rec["corners"]=cor.tolist(); meta[t]=rec
json.dump(meta,open(f"{OUT}/meta.json","w"))
ok=[k for k,v in meta.items() if "err" not in v]
print("渲染",len(ok),"/",len(meta),"只在前视可见",sum(meta[k]["front_only"] for k in ok),"NVRMDONE")
