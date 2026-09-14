"""实车部署可行性测试 · Alpamayo-1.5-10B（uw-Nuvo, RTX A6000 48G）。
只读共享权重（HF_HOME 指向 /home/uw/120/hf，内含指向 ~/.cache 的软链），一切写入落在 /home/uw/120。
量三件事：加载耗时与显存、单帧推理延迟、输出轨迹是否合法。
优先用官方样例 clip；取不到就退化成同形状合成帧（只测加载与延迟，不看轨迹质量）。"""
import os,sys,time,json
os.environ.setdefault("HF_HOME","/home/uw/120/hf")
os.environ.setdefault("HF_HUB_OFFLINE","1")
sys.path.insert(0,"/home/uw/zhengyang/alpamayo1.5/src")
import torch,numpy as np
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5 import helper
def vram(): return torch.cuda.memory_allocated()/2**30, torch.cuda.max_memory_allocated()/2**30
print(f"torch {torch.__version__}  GPU {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB",flush=True)
t0=time.time()
model=Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B",dtype=torch.bfloat16).to("cuda").eval()
t_load=time.time()-t0
proc=helper.get_processor(model.tokenizer)
print(f"[加载] {t_load:.1f} s   显存 已分配 {vram()[0]:.1f} GB / 峰值 {vram()[1]:.1f} GB",flush=True)
data=None
try:
    import pandas as pd
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    cid=pd.read_parquet("/home/uw/zhengyang/alpamayo1.5/notebooks/clip_ids.parquet")["clip_id"].tolist()[774]
    data=load_physical_aiavdataset(cid); print("[数据] 官方样例 clip",cid,flush=True)
except Exception as e:
    print("[数据] 官方样例取不到（",type(e).__name__,"），改用合成帧",flush=True)
    C,T,H,W=4,4,504,896
    data={"image_frames":torch.randint(0,255,(C,T,3,H,W),dtype=torch.uint8),
          "camera_indices":torch.tensor([1,2,3,6][:C]),
          "ego_history_xyz":torch.zeros(1,1,16,3),"ego_history_rot":torch.eye(3).expand(1,1,16,3,3).clone()}
msgs=helper.create_message(data["image_frames"].flatten(0,1),camera_indices=data["camera_indices"])
inp=proc.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False,continue_final_message=True,return_dict=True,return_tensors="pt")
mi=helper.to_device({"tokenized_data":inp,"ego_history_xyz":data["ego_history_xyz"],"ego_history_rot":data["ego_history_rot"]},"cuda")
print(f"[输入] 序列长度 {tuple(inp.input_ids.shape)}  相机 {len(data['camera_indices'])}",flush=True)
lat=[]
for k in range(3):
    torch.cuda.synchronize(); t=time.time(); torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        xyz,rot,extra=model.sample_trajectories_from_data_with_vlm_rollout(
            data=mi,top_p=0.98,temperature=0.6,num_traj_samples=1,max_generation_length=256,return_extra=True)
    torch.cuda.synchronize(); lat.append(time.time()-t)
    print(f"[推理 {k+1}] {lat[-1]:.2f} s  轨迹 {tuple(xyz.shape)}",flush=True)
p=xyz[0,0,0].float().cpu().numpy()
print(f"[输出] 路点数 {p.shape[0]}  末点 ({p[-1,0]:+.1f}, {p[-1,1]:+.1f}) m  弧长 {np.sum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)):.1f} m")
print(f"[结果] 加载 {t_load:.1f}s | 延迟 中位 {np.median(lat):.2f}s | 峰值显存 {vram()[1]:.1f} GB")
cot=(extra or {}).get("cot",[""])[0]
print("[CoC 推理链前 200 字]",str(cot)[:200].replace("\n"," "))
json.dump({"model":"alpamayo15","load_s":t_load,"latency_s":lat,"vram_peak_gb":vram()[1],
           "traj_shape":list(xyz.shape),"ok":True},open("/home/uw/120/deploy_test/r_alpamayo.json","w"),indent=1)
