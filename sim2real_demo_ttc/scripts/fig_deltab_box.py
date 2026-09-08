"""Δb 分布的箱型图 + 逐事件散点。

形式选择：数据的任务是**比较四个分布的形状**，而不是比较四个数值 ——
均值柱状图恰恰会掩盖本图要说的事（SimLingo/DDv2 的均值不代表典型事件）。
故用箱型（中位/四分位/须/离群）+ 抖动散点（露出双峰）。
调色板经 validate_palette.js 校验（light 模式全部 PASS，CVD 最差相邻对 ΔE 8.9）。
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from i_ortho import side_of                                            # noqa: E402
OUT=ROOT/"paper_v2.3_icra/figures"
MOD=["dd","ltf","simlingo","ddv2"]
SHORT={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo"}
PAL={"dd":"#2a78d6","ltf":"#eb6834","simlingo":"#1f9d76","ddv2":"#8b5fc9"}
INK,INK2,GRID,SURF="#0b0b0b","#52514e","#d8d7d2","#fcfcfb"

D={}
for m in MOD:
    z=np.load(RES/f"vbright_acts_navsim_lead_{m}_night_global.npz",allow_pickle=True)
    sc=np.array([str(x) for x in z["scene"]]); k=np.array([side_of(x)=="LHD" for x in sc])
    D[m]=(z["v_alt"]-z["v_orig"])[k].astype(float)

fig,ax=plt.subplots(figsize=(3.4,2.5))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
rng=np.random.default_rng(0)
for i,m in enumerate(MOD):
    d=D[m]; x=i+1
    ax.scatter(x+rng.normal(0,.075,len(d)),d,s=3.2,color=PAL[m],alpha=.20,lw=0,zorder=2)
    bp=ax.boxplot([d],positions=[x],widths=.46,showfliers=False,patch_artist=True,zorder=3,
                  medianprops=dict(color=INK,lw=1.8),
                  boxprops=dict(facecolor="none",edgecolor=PAL[m],lw=1.6),
                  whiskerprops=dict(color=PAL[m],lw=1.2),capprops=dict(color=PAL[m],lw=1.2))
    ax.plot([x],[d.mean()],marker="D",ms=4.5,color=PAL[m],mec=SURF,mew=1.0,zorder=4)
ax.axhline(0,color=INK2,lw=.9,ls="--",zorder=1)
ax.set_xticks(range(1,len(MOD)+1)); ax.set_xticklabels([SHORT[m] for m in MOD],fontsize=8)
ax.set_ylabel(r"$\Delta b$  (m/s)",fontsize=8.5)
ax.tick_params(labelsize=7.5,colors=INK2,length=2)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.grid(axis="y",color=GRID,lw=.6,alpha=.7,zorder=0)
# 效应量直接标在各箱下方（不与图例争顶部空间）；均值/中位用图内符号说明
lo,hi=ax.get_ylim(); ax.set_ylim(lo-(hi-lo)*0.13,hi)
for i,m in enumerate(MOD):
    d=D[m]
    ax.annotate(f"{abs(d.mean())/d.std():.2f}",(i+1,ax.get_ylim()[0]),
                xytext=(0,4),textcoords="offset points",ha="center",va="bottom",
                fontsize=7,color=PAL[m],fontweight="bold")
ax.text(.5,-0.235,"effect size |m|/sd    (diamond = mean, bar = median)",
        transform=ax.transAxes,fontsize=7,color=INK2,ha="center",va="top")
plt.tight_layout(pad=.4)
for ext in ("pdf","png"): fig.savefig(OUT/f"deltab_box.{ext}",dpi=300,bbox_inches="tight")
print("统计：")
for m in MOD:
    d=D[m]
    print(f"  {SHORT[m]:<9} n={len(d)}  mean {d.mean():+.3f}  med {np.median(d):+.3f}  "
          f"|mu|/sd {abs(d.mean())/d.std():.2f}  IQR [{np.percentile(d,25):+.2f},{np.percentile(d,75):+.2f}]")
print(f"-> {OUT}/deltab_box.pdf")
