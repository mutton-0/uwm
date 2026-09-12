"""诊断书：给定一组单位（按舵位/距离筛），算每个模型的画像。供左舵诊断、右舵检验共用。
维度（全部与模型无关的真值 + 模型自己的反事实）：
  exposure   人类同期弧长 >0.5 m 的实际车速帧：规划弧长 / 人类弧长（中位）—— 仓位
  HS         need 单位的危险敏感度均值（看见行人带来的安全增益）—— 风控
  HS_slope   HS 对 a_req 的秩相关（风险越大是否越敏感）—— 风险标度
  SP         非 need 单位的特异度（最小变化原则）—— 不过度反应
  sep_gain   全部单位的 tanh(ΔS) 均值 —— 整条轨迹因看见行人多远离了多少
  CFR        实际车速单位：mean ‖Δ_F‖ / mean ‖Δ_I‖（同帧）—— 光照不变性（幅度）
  align      实际车速单位：cos(Δ_I, Δ_F) 的中位（有符号）。>0 = 夜化把轨迹往"行人被移除"的方向推（光照冒充了因果）；
             0 = 正交（光照改变与行人反应无关，最纯粹）；<0 = 夜化像"多了个行人"。另报跨帧打乱的基线
  purity     1 − |align|
  night_sep  夜化前后整条轨迹与行人的分离度变化 S(夜) − S(原)（均值；负 = 夜里更贴近行人）
每个维度给 bootstrap 95% CI（按帧重采样）。"""
import json,numpy as np
from scipy.stats import spearmanr
def arc(w): w=np.vstack([[0,0],np.asarray(w)]); return float(np.linalg.norm(np.diff(w,axis=0),axis=1).sum())
def dims(units, man_idx, card_arc=None):
    out={}
    u1=[z for z in units if z["v"]>=1.0]
    need=[z for z in u1 if z["need"]]; free=[z for z in u1 if not z["need"]]
    out["HS"]=np.mean([z["HS"] for z in need]) if len(need)>=5 else np.nan
    out["HS_slope"]=spearmanr([z["a_req"] for z in need],[z["HS"] for z in need])[0] if len(need)>=8 else np.nan
    out["SP"]=np.mean([z["SP"] for z in free]) if free else np.nan
    out["sep_gain"]=np.mean([np.tanh(z["P"]["S"]-z["Q"]["S"]) for z in u1]) if u1 else np.nan
    act=[z for z in units if z["sv"]=="actual" and "dI" in z]
    nI=np.array([np.linalg.norm(z["dI"]) for z in act]); nF=np.array([np.linalg.norm(z["dF"]) for z in act])
    out["CFR"]=nF.mean()/nI.mean() if len(act)>=5 else np.nan
    ok=[z for z in act if np.linalg.norm(z["dI"])>0.05 and np.linalg.norm(z["dF"])>0.05]
    cs=[np.dot(z["dI"],z["dF"])/(np.linalg.norm(z["dI"])*np.linalg.norm(z["dF"])) for z in ok]
    out["align"]=float(np.median(cs)) if len(cs)>=5 else np.nan
    out["purity"]=1-abs(out["align"]) if len(cs)>=5 else np.nan
    if len(ok)>=5:   # 跨帧打乱：同一模型、不同帧的 Δ_I 与 Δ_F —— 只由"轨迹主要沿纵向变形"带来的对齐
        rr=np.random.default_rng(1); sh=[]
        for _ in range(20):
            p=rr.permutation(len(ok))
            sh+= [np.dot(ok[i]["dI"],ok[j]["dF"])/(np.linalg.norm(ok[i]["dI"])*np.linalg.norm(ok[j]["dF"])) for i,j in zip(range(len(ok)),p) if i!=j]
        out["align_shuffled"]=float(np.median(sh))
    else: out["align_shuffled"]=np.nan
    out["night_sep"]=float(np.mean([z["fN"]["S"]-z["P"]["S"] for z in act])) if act else np.nan
    ex=[]
    for z in act:
        h=man_idx[z["uid"]].get("human_arc_2p5")
        if h and h>0.5 and z.get("arcP") is not None: ex.append(z["arcP"]/h)
    out["exposure"]=np.median(ex) if len(ex)>=5 else np.nan
    out["n_units"]=len(u1); out["n_need"]=len(need); out["n_act"]=len(act)
    return out
def boot(units, man_idx, B=500, seed=0):
    rng=np.random.default_rng(seed); uids=sorted({z["uid"] for z in units})
    by={}
    for z in units: by.setdefault(z["uid"],[]).append(z)
    res=[]
    for _ in range(B):
        pick=rng.choice(len(uids),len(uids),replace=True)
        res.append(dims([z for i in pick for z in by[uids[i]]],man_idx))
    keys=[k for k in res[0] if not k.startswith("n_")]
    return {k:[float(np.nanpercentile([r[k] for r in res],2.5)),float(np.nanpercentile([r[k] for r in res],97.5))] for k in keys}
