"""黄昏 vs 夜化的光照混淆比 CFR（右舵 254 场景，四条件同进程同图源）。
F  = mean_t‖clean − rm‖    行人本身能把规划推多远
I_N= mean_t‖clean − night‖ 夜化能把规划推多远
I_D= mean_t‖clean − dusk‖  黄昏能把规划推多远
CFR = F / I：< 1 表示"光照比路径上的行人更能改变规划"。
黄昏的意义：它几乎不损失行人可检出性（88.9% vs 夜化 66.7%，见 dusk_check.json），
所以若 CFR_dusk 仍 < 1，就不能用"夜化把行人也一起抹掉了"来解释这个现象。
输入 {TAG}_axes4_{model}.json（BEV 三家在 5090 上、VLA 三家在 exx 上生成）。输出 dusk_cfr.json。"""
import json,os,numpy as np
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]
def lin(P):
    P=np.asarray(P,float); return np.stack([np.interp(TT,T8,P[:,0]),np.interp(TT,T8,P[:,1])],1)
def D(a,b):
    A=lin(np.vstack([[0,0],np.asarray(a)[:,:2]])); B=lin(np.vstack([[0,0],np.asarray(b)[:,:2]]))
    return float(np.mean(np.linalg.norm(A-B,axis=1)))
MODELS=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME=dict(dd="DiffusionDrive",ltf="LTF",ddv2="DiffusionDriveV2",simlingo="SimLingo",autovla="AutoVLA",alpamayo15="Alpamayo-1.5")
res={}
print(f"{'模型':18s} {'n':>4s} {'F':>6s} {'I_夜':>6s} {'I_黄':>6s} {'CFR_夜':>7s} {'CFR_黄':>7s} {'I_黄/I_夜':>9s}")
for m in MODELS:
    p=f"{V5}/rhd_axes4_{m}.json"
    if not os.path.exists(p): print(f"  {NAME[m]:16s}  —— 尚无 rhd_axes4_{m}.json"); continue
    d=json.load(open(p)); F=[];IN=[];ID=[]
    for t,v in d.items():
        if not all(k in v for k in ("clean","rm","night","dusk")): continue
        F.append(D(v["clean"],v["rm"])); IN.append(D(v["clean"],v["night"])); ID.append(D(v["clean"],v["dusk"]))
    F=np.array(F); IN=np.array(IN); ID=np.array(ID)
    r=dict(n=len(F),F=float(F.mean()),I_night=float(IN.mean()),I_dusk=float(ID.mean()),
           cfr_night=float(F.mean()/IN.mean()),cfr_dusk=float(F.mean()/ID.mean()),
           F_med=float(np.median(F)),I_night_med=float(np.median(IN)),I_dusk_med=float(np.median(ID)),
           frac_F_gt_I_night=float(np.mean(F>IN)),frac_F_gt_I_dusk=float(np.mean(F>ID)))
    res[m]=r
    print(f"  {NAME[m]:16s} {r['n']:4d} {r['F']:6.3f} {r['I_night']:6.3f} {r['I_dusk']:6.3f} "
          f"{r['cfr_night']:7.2f} {r['cfr_dusk']:7.2f} {r['I_dusk']/r['I_night']:9.2f}")
if res:
    print("\n逐场景 F>I 的比例（该场景里行人比光照更能改变规划）：")
    for m,r in res.items():
        print(f"  {NAME[m]:16s} 对夜化 {r['frac_F_gt_I_night']:5.1%}   对黄昏 {r['frac_F_gt_I_dusk']:5.1%}")
    json.dump(res,open(f"{V5}/dusk_cfr.json","w"),indent=1)
