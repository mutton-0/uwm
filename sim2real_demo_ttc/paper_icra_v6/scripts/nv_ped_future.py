"""NAVSIM 近行人场景：取走廊里最近的行人/骑车人（净间隙 ≤1 m、纵向 ≤20 m 中纵向最近者），
用 track_token 在同 log 后续帧（0.5 s 间隔）里追它的位置，换到当前 ego 系，得到 0.5–4.0 s 的真实未来。"""
import glob,pickle,json,numpy as np
LOGS="/data/dataset/navsim/dataset/navsim_logs/test"
S="/tmp/claude-1001/-home-boyuewang-120-uwm/cd9145da-d85c-41bc-8d71-909dc71f6020/scratchpad"
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
want=set(json.load(open(f"{S}/sg_closevru_tok.json"))["tokens"])|set(json.load(open(f"{S}/lhd_closevru_tok.json"))["tokens"])
# 注意：frame_idx 是场景内序号、同一 log 里会重复，不能按 (log, frame_idx) 找后续帧 —— 用 sample_next 链
TOK={}; sc={}
for f in sorted(glob.glob(f"{LOGS}/*.pkl")):
    d=pickle.load(open(f,"rb"))
    for s in (d if isinstance(d,list) else list(d.values())):
        TOK[s["token"]]=s
        if s["token"] in want: sc[s["token"]]=s
def M(s): return np.asarray(s["ego2global"],float)
out={}
for t,s in sc.items():
    a=s["anns"]; B=np.asarray(a["gt_boxes"]); N=np.asarray(a["gt_names"]); TR=a["track_tokens"]
    best=None
    for i in range(len(N)):
        if str(N[i]) not in ("pedestrian","bicycle"): continue
        x,y,_,l,w,h,_=B[i][:7]
        if 0<x<=20 and abs(y)-w/2-1.0<=1.0 and (best is None or x<B[best][0]): best=i
    if best is None: continue
    tr=TR[best]; G0=M(s); inv=np.linalg.inv(G0); fut=[]
    s1=s
    for k in range(1,9):
        n=s1.get("sample_next") if s1 is not None else None
        s1=TOK.get(n) if n else None
        if s1 is None: fut.append(None); continue
        a1=s1["anns"]; TR1=list(a1["track_tokens"])
        if tr not in TR1: fut.append(None); continue
        j=TR1.index(tr); p=np.asarray(a1["gt_boxes"])[j][:3]
        g=M(s1)@np.r_[p,1.0]; c=inv@g; fut.append([float(c[0]),float(c[1])])
    e=s["ego_dynamic_state"]
    out[t]=dict(loc=s["map_location"],v0=float(np.hypot(e[0],e[1])),p0=[float(B[best][0]),float(B[best][1])],fut=fut,cat=str(N[best]))
json.dump(out,open(f"{V5}/nv_ped_future.json","w"))
ok=sum(1 for v in out.values() if all(z is not None for z in v["fut"][:5]))
print("场景",len(out),"前 2.5 s 轨迹完整",ok)
