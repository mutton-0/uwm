"""潜空间探针 · 第 1 步：从策略在原图 O 上的潜向量线性读出反事实读数 F / I / ΔS / need。
输入 probe_v0/data/latents_{MODEL}.npz（extract_bev_latents.py 抽出的 8 层 token 均值 + 轨迹）
     results_5090/paper_icra_v5/nv_ped_future.json（行人真实未来）
做法
  样本 = (场景, 速度档)，速度 ∈ {actual, 4, 8}；x = clean 条件下某层 all_mean（或 8 层拼接）；
  标签由同一进程抽出的 clean/rm/night 轨迹算出：F=D̄(clean,rm)、I=D̄(clean,night)、ΔS=S(clean)−S(rm)、need=[clr(rm)<1 m]
  探针 = Ridge（α 内层 CV）/ Logistic；外层 GroupKFold(5) 按场景分组，同一场景三档速度不跨折
  对照 = 只用速度 v 的岭回归；标签打乱的置换零分布（20 次）
  非线性对照 = 2 层 MLP（256 隐单元，AdamW，早停），TensorBoard 记录 train/val loss 与 val R²/ρ
输出 probe_v0/results/probe_{MODEL}.json、probe_v0/tb/{MODEL}/…、控制台摘要
用法 probe_train.py dd|ltf|ddv2 [--no-mlp]"""
import os,sys,json,numpy as np,torch,time
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
MODEL=sys.argv[1]; NOMLP="--no-mlp" in sys.argv
D="/home/boyuewang/120/uwm/sim2real_demo_ttc/probe_v0"; V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
Z=np.load(f"{D}/data/latents_{MODEL}.npz"); PF=json.load(open(f"{V5}/nv_ped_future.json"))
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]
def lin(P): P=np.vstack([[0,0],np.asarray(P,float)[:,:2]]); return np.stack([np.interp(TT,T8,P[:,0]),np.interp(TT,T8,P[:,1])],1)
def ped_xy(t):
    p=PF[t]; P=np.asarray([p["p0"]]+[z for z in p["fut"] if z is not None],float); tp=np.linspace(0,2.5,len(P))
    return np.stack([np.interp(TT,tp,P[:,0]),np.interp(TT,tp,P[:,1])],1)
def disp(a,b): return float(np.mean(np.linalg.norm(a-b,axis=1)))
def S_of(X,ped): return float(np.mean(np.minimum(np.linalg.norm(X-ped,axis=1)-1.4,10.0)))
def clr(X,ped): return float(np.min(np.linalg.norm(X-ped,axis=1))-1.4)
# ---------- 组样本 ----------
tok=Z["token"]; spd=Z["speed"]; cond=Z["cond"]; traj=Z["traj"]; vv=Z["v"]
NL=len([k for k in Z.files if k.startswith("all_mean_L")])
idx={}
for i,(t,s,c) in enumerate(zip(tok,spd,cond)): idx[(t,s,c)]=i
samples=[]
for (t,s,c),i in idx.items():
    if c!="clean" or (t,s,"rm") not in idx: continue
    ped=ped_xy(t); XO=lin(traj[i]); XR=lin(traj[idx[(t,s,"rm")]])
    F=disp(XO,XR); I=disp(XO,lin(traj[idx[(t,s,"night")]])) if (t,s,"night") in idx else np.nan
    dS=S_of(XO,ped)-S_of(XR,ped); need=int(clr(XR,ped)<1.0)
    samples.append(dict(i=i,token=t,speed=s,v=float(vv[i]),F=F,I=I,dS=dS,need=need,arc=float(np.sum(np.linalg.norm(np.diff(XO,axis=0),axis=1)))))
print(f"[{MODEL}] samples {len(samples)}  need {sum(s['need'] for s in samples)}  layers {NL}",flush=True)
groups=np.array([s["token"] for s in samples]); V=np.array([[s["v"]] for s in samples])
Y={"logF":np.log10(np.array([s["F"] for s in samples])+1e-3),"logI":np.log10(np.array([s["I"] for s in samples])+1e-3),
   "dS":np.array([s["dS"] for s in samples]),"logArc":np.log10(np.array([s["arc"] for s in samples])+1e-2)}
NEED=np.array([s["need"] for s in samples])
def feats(layer,pool="all_mean"):
    if layer=="concat": return np.concatenate([Z[f"{pool}_L{j}"][[s["i"] for s in samples]] for j in range(NL)],1)
    return Z[f"{pool}_L{layer}"][[s["i"] for s in samples]]
gkf=GroupKFold(n_splits=5); FOLDS=list(gkf.split(V,groups=groups))
def r2(y,p): return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
def ridge_cv(X,y):
    pred=np.zeros_like(y)
    for tr,te in FOLDS:
        sc=StandardScaler().fit(X[tr]); m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr]); pred[te]=m.predict(sc.transform(X[te]))
    return pred
def logit_cv(X,y):
    pred=np.zeros(len(y))
    for tr,te in FOLDS:
        if y[tr].min()==y[tr].max(): pred[te]=y[tr].mean(); continue
        sc=StandardScaler().fit(X[tr]); m=LogisticRegressionCV(Cs=8,cv=3,max_iter=2000,class_weight="balanced").fit(sc.transform(X[tr]),y[tr]); pred[te]=m.predict_proba(sc.transform(X[te]))[:,1]
    return pred
res={"model":MODEL,"n":len(samples),"n_need":int(NEED.sum()),"layers":{},"speed_only":{},"perm_null":{},"mlp":{}}
rng=np.random.default_rng(0)
# ---------- 速度对照 ----------
for k,y in Y.items():
    ok=~np.isnan(y); p=ridge_cv(V[ok],y[ok]); res["speed_only"][k]={"r2":r2(y[ok],p),"rho":float(spearmanr(y[ok],p)[0])}
res["speed_only"]["need"]={"auc":float(roc_auc_score(NEED,logit_cv(V,NEED)))}
# ---------- 逐层线性探针 ----------
for layer in list(range(NL))+["concat"]:
    X=feats(layer); r={}
    for k,y in Y.items():
        ok=~np.isnan(y); p=ridge_cv(X[ok],y[ok]); r[k]={"r2":r2(y[ok],p),"rho":float(spearmanr(y[ok],p)[0])}
        if layer=="concat": r[k]["pred"]=p.tolist(); r[k]["true"]=y[ok].tolist()
    r["need"]={"auc":float(roc_auc_score(NEED,logit_cv(X,NEED)))}
    res["layers"][str(layer)]=r; print(f"  L{layer}: "+"  ".join(f"{k} R²={r[k]['r2']:+.2f} ρ={r[k]['rho']:+.2f}" for k in Y)+f"  need AUC={r['need']['auc']:.2f}",flush=True)
# ---------- 置换零分布（concat 层）----------
X=feats("concat")
for k,y in Y.items():
    ok=~np.isnan(y); nulls=[]
    for _ in range(20):
        yp=y[ok].copy();
        # 按场景块打乱，保持三档速度成组
        g=groups[ok]; ug=np.unique(g); perm=dict(zip(ug,rng.permutation(ug)))
        order=np.argsort([perm[x] for x in g],kind="stable"); yp=yp[order]
        nulls.append(r2(y[ok],ridge_cv(X[ok],yp)))
    res["perm_null"][k]={"r2_mean":float(np.mean(nulls)),"r2_p95":float(np.percentile(nulls,95))}
# ---------- MLP 对照 + TensorBoard ----------
if not NOMLP:
    from torch.utils.tensorboard import SummaryWriter
    dev="cuda" if torch.cuda.is_available() else "cpu"
    for k,y in Y.items():
        ok=~np.isnan(y); Xk=X[ok]; yk=y[ok]; gk=groups[ok]; pred=np.zeros_like(yk); curves=[]
        for fi,(tr,te) in enumerate(list(GroupKFold(5).split(Xk,groups=gk))):
            # 内层再切一份 val 做早停（按场景）
            gtr=gk[tr]; ug=np.unique(gtr); rng2=np.random.default_rng(fi); vg=set(rng2.choice(ug,max(1,len(ug)//6),replace=False))
            va=np.array([g in vg for g in gtr]); tr_in=tr[~va]; tr_va=tr[va]
            sc=StandardScaler().fit(Xk[tr_in]); Xt=torch.tensor(sc.transform(Xk[tr_in]),dtype=torch.float32,device=dev); yt=torch.tensor(yk[tr_in],dtype=torch.float32,device=dev)
            Xv=torch.tensor(sc.transform(Xk[tr_va]),dtype=torch.float32,device=dev); yv=torch.tensor(yk[tr_va],dtype=torch.float32,device=dev)
            torch.manual_seed(fi); net=torch.nn.Sequential(torch.nn.Linear(Xt.shape[1],256),torch.nn.GELU(),torch.nn.Dropout(0.3),torch.nn.Linear(256,64),torch.nn.GELU(),torch.nn.Linear(64,1)).to(dev)
            opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-2); w=SummaryWriter(f"{D}/tb/{MODEL}/{k}/fold{fi}")
            best=(1e9,None,0); ymu,ysd=float(yt.mean()),float(yt.std()+1e-6)
            for ep in range(300):
                net.train(); perm=torch.randperm(len(Xt),device=dev)
                for b in range(0,len(Xt),64):
                    j=perm[b:b+64]; loss=torch.nn.functional.mse_loss(net(Xt[j]).squeeze(1),(yt[j]-ymu)/ysd); opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    ptr=net(Xt).squeeze(1)*ysd+ymu; pv=net(Xv).squeeze(1)*ysd+ymu
                    ltr=float(torch.mean((ptr-yt)**2)); lv=float(torch.mean((pv-yv)**2)); r2v=float(1-torch.sum((yv-pv)**2)/torch.sum((yv-yv.mean())**2))
                    rhov=float(spearmanr(yv.cpu().numpy(),pv.cpu().numpy())[0]) if len(yv)>2 else 0.0
                w.add_scalar("loss/train_mse",ltr,ep); w.add_scalar("loss/val_mse",lv,ep); w.add_scalar("val/r2",r2v,ep); w.add_scalar("val/spearman",rhov,ep)
                curves.append(dict(fold=fi,ep=ep,train=ltr,val=lv,r2=r2v,rho=rhov))
                if lv<best[0]: best=(lv,{kk:v.clone() for kk,v in net.state_dict().items()},ep)
                elif ep-best[2]>40: break
            w.close(); net.load_state_dict(best[1]); net.eval()
            with torch.no_grad(): pred[te]=(net(torch.tensor(sc.transform(Xk[te]),dtype=torch.float32,device=dev)).squeeze(1)*ysd+ymu).cpu().numpy()
        res["mlp"][k]={"r2":r2(yk,pred),"rho":float(spearmanr(yk,pred)[0]),"curves":curves,"pred":pred.tolist(),"true":yk.tolist()}
        print(f"  MLP {k}: R²={res['mlp'][k]['r2']:+.2f} ρ={res['mlp'][k]['rho']:+.2f}",flush=True)
res["samples"]=[{kk:s[kk] for kk in ("token","speed","v","F","I","dS","need","arc")} for s in samples]
json.dump(res,open(f"{D}/results/probe_{MODEL}.json","w")); print(MODEL,"PROBEDONE")
