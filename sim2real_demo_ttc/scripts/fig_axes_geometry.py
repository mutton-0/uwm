"""三轴几何示意图：把「两族」这件事画出来。

形式选择：数据的任务是**极性与身份**（三根方向分成两族，且第三根在对侧），
不是量级比较。故用极坐标上的方向示意 + 每根轴的噪声地板扇区，
让读者一眼看到「地板很窄、夹角远大于地板、亮度轴越过 90°」这三件事。
调色板同 fig_deltab_box（已过 validate_palette.js）。
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES=ROOT/"results"
sys.path.insert(0,str(ROOT/"scripts"))
from f_vfaith_direction import load_by_side, part_ratio                # noqa: E402
from i_ortho import side_of                                            # noqa: E402
OUT=ROOT/"paper_v2.3_icra/figures"
MOD=["dd","ltf","ddv2"]
SHORT={"dd":"DD","ltf":"LTF","ddv2":"DDv2"}
C_F,C_D,C_B="#2a78d6","#1f9d76","#eb6834"
INK,INK2,GRID,SURF="#0b0b0b","#52514e","#d8d7d2","#fcfcfb"
def u(x):
    n=np.linalg.norm(x); return x/n if n>1e-12 else x
def ang(a,b): return float(np.degrees(np.arccos(np.clip(u(a)@u(b),-1,1))))
def half(D,rng,B=200):
    return float(np.median([ang(D[p[:len(D)//2]].mean(0),D[p[len(D)//2:]].mean(0))
                            for p in (rng.permutation(len(D)) for _ in range(B))]))

rng=np.random.default_rng(0)
fig,axes=plt.subplots(1,3,figsize=(6.9,1.72),subplot_kw={"projection":"polar"})
fig.patch.set_facecolor(SURF)
for ax,m in zip(axes,MOD):
    b=load_by_side("lead",m)["LHD"]
    zb=np.load(RES/f"vbright_acts_navsim_lead_{m}_night_global.npz",allow_pickle=True)
    nL=min(b["nL"],sum(1 for k in zb.files if k.startswith("h_orig__L")))
    cand=[l for l in range(nL) if part_ratio(b["d"][l])>=3]; L=cand[len(cand)//2]
    vf=b["d"][L].mean(0)
    sc=np.array([str(x) for x in zb["scene"]]); k=np.array([side_of(x)=="LHD" for x in sc])
    Db=(zb[f"h_alt__L{L}"]-zb[f"h_orig__L{L}"])[k]; vb=Db.mean(0)
    z=np.load(RES/f"vdecel_acts_navsim_test_{m}.npz",allow_pickle=True); s=(z["side"]=="LHD")
    H=z[f"h__L{L}"].astype(float)
    vd=H[s&(z["kind"]=="brake")].mean(0)-H[s&(z["kind"]=="cruise")].mean(0)
    a_d, a_b = ang(vf,vd), ang(vf,vb)
    h_f, h_b = half(b["d"][L],rng), half(Db,rng)
    ax.set_facecolor(SURF)
    ax.set_theta_zero_location("E"); ax.set_theta_direction(1)
    ax.set_thetamin(0); ax.set_thetamax(180); ax.set_rlim(0,1.05)
    ax.set_yticks([]); ax.set_xticks(np.radians([0,45,90,135,180]))
    ax.set_xticklabels(["$0^\\circ$","","$90^\\circ$","","$180^\\circ$"],fontsize=6.6,color=INK2)
    ax.grid(color=GRID,lw=.55,alpha=.8)
    # 90° 随机基线
    ax.plot([np.pi/2]*2,[0,1.0],color=INK2,lw=1.0,ls=(0,(3,2)),zorder=2)
    for a,c,lab,hh in ((0,C_F,r"$v_{\rm faith}$",h_f),(np.radians(a_d),C_D,r"$v_{\rm decel}$",None),
                       (np.radians(a_b),C_B,r"$v_{\rm bright}$",h_b)):
        ax.plot([a,a],[0,1.0],color=c,lw=2.4,solid_capstyle="round",zorder=4)
        if hh:   # 噪声地板扇区
            ax.fill_between(np.linspace(a-np.radians(hh),a+np.radians(hh),40),0,1.0,
                            color=c,alpha=.16,lw=0,zorder=3)
    ax.set_title(f"{SHORT[m]}   f–d {a_d:.0f}°   b–f {a_b:.0f}°",fontsize=7.6,color=INK,pad=5)
fig.text(.5,.015,r"$v_{\rm faith}$ (blue) $\cdot$ $v_{\rm decel}$ (green) $\cdot$ "
         r"$v_{\rm bright}$ (orange);   shaded = split-half floor,   dashed = $90^\circ$ random",
         fontsize=7,color=INK2,ha="center",va="bottom")
plt.tight_layout(pad=.35,rect=(0,.10,1,1))
for ext in ("pdf","png"): fig.savefig(OUT/f"axes_geometry.{ext}",dpi=300,bbox_inches="tight")
print(f"-> {OUT}/axes_geometry.pdf")
