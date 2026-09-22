"""探针结果出图：逐层 R²、置换零分布、速度对照、pred-vs-true 散点、need AUROC、MLP 训练曲线（TensorBoard 记录的同一份数据）。
输入 probe_v0/results/probe_{dd,ltf,ddv2}.json；输出 probe_v0/figures/*.pdf|png 与 results/summary.json"""
import json,os,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
D="/home/boyuewang/120/uwm/sim2real_demo_ttc/probe_v0"; FIG=f"{D}/figures"; os.makedirs(FIG,exist_ok=True)
plt.rcParams.update({"font.family":"Nimbus Roman","mathtext.fontset":"stix","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6})
M=[m for m in ("dd","ltf","ddv2") if os.path.exists(f"{D}/results/probe_{m}.json")]
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2"}; COL={"dd":"#8fa3b8","ltf":"#5b9bd5","ddv2":"#1f3f6e"}
TG=[("logF","$\\log_{10} F$"),("logI","$\\log_{10} I$"),("dS","$\\Delta S$"),("logArc","$\\log_{10}$ arc")]
R={m:json.load(open(f"{D}/results/probe_{m}.json")) for m in M}
# ---- Fig 1: 逐层 R²（线性探针）+ 置换零分布 95 分位 + 速度对照 ----
fig,ax=plt.subplots(1,4,figsize=(7.16,1.8),sharey=True)
for a,(k,lab) in zip(ax,TG):
    for m in M:
        L=R[m]["layers"]; xs=list(range(8)); ys=[L[str(j)][k]["r2"] for j in xs]
        a.plot(xs,ys,marker="o",ms=3,lw=1.0,color=COL[m],label=NAME[m])
        a.plot([8.6],[L["concat"][k]["r2"]],marker="s",ms=4,color=COL[m])
        a.axhline(R[m]["perm_null"][k]["r2_p95"],color=COL[m],lw=0.6,ls=(0,(2,1.5)),alpha=0.8)
        a.axhline(R[m]["speed_only"][k]["r2"],color=COL[m],lw=0.6,ls=(0,(1,1)),alpha=0.8)
    a.set_title(lab,fontsize=8,loc="left"); a.set_xticks(list(range(8))+[8.6]); a.set_xticklabels([f"L{j}" for j in range(8)]+["all"],fontsize=6.5)
    a.set_ylim(-0.3,1.0); a.axhline(0,color="#9a998f",lw=0.5)
ax[0].set_ylabel("held-out $R^2$ (5-fold, by scene)")
h=[plt.Line2D([],[],color=COL[m],marker="o",ms=3,lw=1,label=NAME[m]) for m in M]+[plt.Line2D([],[],color="#52514e",lw=0.6,ls=(0,(2,1.5)),label="permutation null, 95th pct"),plt.Line2D([],[],color="#52514e",lw=0.6,ls=(0,(1,1)),label="speed-only ridge"),plt.Line2D([],[],color="#52514e",marker="s",ms=4,ls="none",label="all layers concatenated")]
fig.legend(handles=h,frameon=False,fontsize=6.5,ncol=6,loc="lower center",bbox_to_anchor=(0.5,-0.08))
fig.savefig(f"{FIG}/fig_probe_layers.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{FIG}/fig_probe_layers.png",dpi=220,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
# ---- Fig 2: pred vs true（拼接层线性探针），按速度档着色 ----
fig,ax=plt.subplots(len(M),4,figsize=(7.16,1.75*len(M)),squeeze=False)
SC={"actual":"#b8c2cc","4":"#5b9bd5","8":"#0b0b14"}
for r,m in enumerate(M):
    sp=[s["speed"] for s in R[m]["samples"]]
    for c,(k,lab) in enumerate(TG):
        a=ax[r,c]; L=R[m]["layers"]["concat"][k]; y=np.array(L["true"]); p=np.array(L["pred"])
        spk=[s["speed"] for s in R[m]["samples"] if not (k=="logI" and np.isnan(s["I"]))]
        for s_ in ("actual","4","8"):
            i=[j for j,x in enumerate(spk) if x==s_]; a.scatter(y[i],p[i],s=4,alpha=0.55,color=SC[s_],lw=0,label=f"v={s_}" if s_!="actual" else "logged v")
        lo,hi=min(y.min(),p.min()),max(y.max(),p.max()); a.plot([lo,hi],[lo,hi],color="#9a998f",lw=0.6,ls=(0,(2,1.5)))
        a.set_title(f"{NAME[m]}: {lab}  $R^2$={L['r2']:.2f}, $\\rho$={L['rho']:.2f}",fontsize=7,loc="left"); a.tick_params(labelsize=6.5)
        if r==len(M)-1: a.set_xlabel("measured",fontsize=7)
        if c==0: a.set_ylabel("linear probe",fontsize=7)
ax[0,0].legend(frameon=False,fontsize=6,loc="upper left",handletextpad=0.2,markerscale=1.6)
fig.tight_layout(h_pad=0.6,w_pad=0.4); fig.savefig(f"{FIG}/fig_probe_scatter.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{FIG}/fig_probe_scatter.png",dpi=220,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
# ---- Fig 3: need AUROC 逐层 + MLP vs 线性 ----
fig,ax=plt.subplots(1,2,figsize=(7.16,1.8))
a=ax[0]
for m in M:
    L=R[m]["layers"]; a.plot(range(8),[L[str(j)]["need"]["auc"] for j in range(8)],marker="o",ms=3,lw=1,color=COL[m],label=NAME[m]); a.plot([8.6],[L["concat"]["need"]["auc"]],marker="s",ms=4,color=COL[m])
    a.axhline(R[m]["speed_only"]["need"]["auc"],color=COL[m],lw=0.6,ls=(0,(1,1)))
a.axhline(0.5,color="#9a998f",lw=0.5); a.set_ylim(0.4,1.0); a.set_xticks(list(range(8))+[8.6]); a.set_xticklabels([f"L{j}" for j in range(8)]+["all"],fontsize=6.5)
a.set_title("(a) need set, $C(X^R)<1$: held-out AUROC",fontsize=8,loc="left"); a.set_ylabel("AUROC"); a.legend(frameon=False,fontsize=6.5,loc="lower right")
a=ax[1]; w=0.25; x=np.arange(4)
for i,m in enumerate(M):
    lin=[R[m]["layers"]["concat"][k]["r2"] for k,_ in TG]; mlp=[R[m]["mlp"].get(k,{}).get("r2",np.nan) for k,_ in TG]
    a.bar(x+i*w-w,lin,w,color=COL[m],alpha=0.55,label=f"{NAME[m]} linear"); a.bar(x+i*w-w,mlp,w,fill=False,edgecolor=COL[m],lw=1.0,label=f"{NAME[m]} MLP")
a.set_xticks(x); a.set_xticklabels([l for _,l in TG]); a.set_ylabel("held-out $R^2$"); a.axhline(0,color="#9a998f",lw=0.5); a.set_ylim(-0.3,1.0)
a.set_title("(b) linear vs. MLP probe, all layers",fontsize=8,loc="left"); a.legend(frameon=False,fontsize=5.6,ncol=2,loc="upper left")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{FIG}/fig_probe_need_mlp.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{FIG}/fig_probe_need_mlp.png",dpi=220,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
# ---- Fig 4: MLP 训练曲线（TensorBoard 同源）：train/val MSE 与 val R²，按 fold 平均 ----
fig,ax=plt.subplots(len(M),4,figsize=(7.16,1.6*len(M)),squeeze=False)
for r,m in enumerate(M):
    for c,(k,lab) in enumerate(TG):
        a=ax[r,c]; cur=R[m]["mlp"].get(k,{}).get("curves",[])
        if not cur: a.axis("off"); continue
        for fi in range(5):
            cf=[q for q in cur if q["fold"]==fi]
            a.plot([q["ep"] for q in cf],[q["train"] for q in cf],color="#8fa3b8",lw=0.6,alpha=0.7); a.plot([q["ep"] for q in cf],[q["val"] for q in cf],color="#1f3f6e",lw=0.6,alpha=0.7)
        a.set_yscale("log"); a.set_title(f"{NAME[m]}: {lab}",fontsize=7,loc="left"); a.tick_params(labelsize=6.5)
        b=a.twinx();
        for fi in range(5):
            cf=[q for q in cur if q["fold"]==fi]; b.plot([q["ep"] for q in cf],[q["r2"] for q in cf],color="#c0392b",lw=0.5,alpha=0.5)
        b.set_ylim(-0.5,1.0); b.tick_params(labelsize=6,colors="#c0392b"); b.spines["right"].set_visible(True); b.spines["right"].set_color("#c0392b")
        if r==len(M)-1: a.set_xlabel("epoch",fontsize=7)
        if c==0: a.set_ylabel("MSE (z-scored)",fontsize=7)
        if c==3: b.set_ylabel("val $R^2$",fontsize=7,color="#c0392b")
h=[plt.Line2D([],[],color="#8fa3b8",lw=1,label="train MSE"),plt.Line2D([],[],color="#1f3f6e",lw=1,label="val MSE (early-stop)"),plt.Line2D([],[],color="#c0392b",lw=1,label="val $R^2$")]
fig.legend(handles=h,frameon=False,fontsize=6.5,ncol=3,loc="lower center",bbox_to_anchor=(0.5,-0.05)); fig.tight_layout(h_pad=0.6,w_pad=1.2)
fig.savefig(f"{FIG}/fig_probe_curves.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{FIG}/fig_probe_curves.png",dpi=220,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
# ---- 汇总表 ----
S={}
for m in M:
    L=R[m]["layers"]; best={k:max(range(8),key=lambda j:L[str(j)][k]["r2"]) for k,_ in TG}
    S[m]={"n":R[m]["n"],"n_need":R[m]["n_need"],
          "concat":{k:{"r2":round(L["concat"][k]["r2"],3),"rho":round(L["concat"][k]["rho"],3)} for k,_ in TG},
          "best_layer":{k:{"layer":best[k],"r2":round(L[str(best[k])][k]["r2"],3)} for k,_ in TG},
          "speed_only":{k:round(R[m]["speed_only"][k]["r2"],3) for k,_ in TG},
          "perm_p95":{k:round(R[m]["perm_null"][k]["r2_p95"],3) for k,_ in TG},
          "need_auc":{"concat":round(L["concat"]["need"]["auc"],3),"speed_only":round(R[m]["speed_only"]["need"]["auc"],3),"best":round(max(L[str(j)]["need"]["auc"] for j in range(8)),3)},
          "mlp":{k:round(R[m]["mlp"][k]["r2"],3) for k in R[m]["mlp"]}}
json.dump(S,open(f"{D}/results/summary.json","w"),indent=1); print(json.dumps(S,indent=1))
# ---- Fig 5: 三种输入（只潜向量 / 潜向量⊕速度 / 只速度）的线性探针对比 + 场景级（logged 速度）----
E=json.load(open(f"{D}/results/probe_extra.json"))
fig,ax=plt.subplots(1,len(M)+1,figsize=(7.16,1.9))
for i,m in enumerate(M):
    a=ax[i]; x=np.arange(4); w=0.26
    lat=[R[m]["layers"]["concat"][k]["r2"] for k,_ in TG]; lv=[E[m]["latent_plus_v"][k]["r2"] for k,_ in TG]; so=[R[m]["speed_only"][k]["r2"] for k,_ in TG]
    a.bar(x-w,lat,w,color=COL[m],alpha=0.45,label="latent only"); a.bar(x,lv,w,color=COL[m],label="latent $\\oplus$ speed"); a.bar(x+w,so,w,fill=False,edgecolor="#52514e",lw=0.9,label="speed only")
    a.set_xticks(x); a.set_xticklabels([l for _,l in TG],fontsize=6.5); a.set_ylim(-0.2,1.0); a.axhline(0,color="#9a998f",lw=0.5)
    a.set_title(NAME[m],fontsize=8,loc="left")
    if i==0: a.set_ylabel("held-out $R^2$"); a.legend(frameon=False,fontsize=5.8,loc="upper left")
a=ax[-1]; x=np.arange(len(M)); w=0.26
a.bar(x-w,[R[m]["layers"]["concat"]["need"]["auc"] for m in M],w,color=[COL[m] for m in M],alpha=0.45)
a.bar(x,[E[m]["latent_plus_v"]["need_auc"] for m in M],w,color=[COL[m] for m in M]); a.bar(x+w,[R[m]["speed_only"]["need"]["auc"] for m in M],w,fill=False,edgecolor="#52514e",lw=0.9)
a.axhline(0.5,color="#9a998f",lw=0.5); a.set_ylim(0.4,1.0); a.set_xticks(x); a.set_xticklabels([NAME[m].replace("Diffusion","D") for m in M],fontsize=6.5); a.set_title("need set: AUROC",fontsize=8,loc="left")
fig.tight_layout(w_pad=0.8); fig.savefig(f"{FIG}/fig_probe_inputs.pdf",bbox_inches="tight",pad_inches=0.02); fig.savefig(f"{FIG}/fig_probe_inputs.png",dpi=220,bbox_inches="tight",pad_inches=0.02); plt.close(fig)
print("fig5 ok")
