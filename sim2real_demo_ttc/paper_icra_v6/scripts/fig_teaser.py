"""Fig. 1：一帧、两种干预、六个模型 + 全体走廊帧上的 D_ped–D_I 散点。数据全部从 card_*.json / card_night_*.json 直读。"""
import json,glob,sys,zlib,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts")
from appearance_transform import transform
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; P=f"{R5}/paper_icra_v4"; V5=f"{R5}/paper_icra_v5"; UID="A_0095"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7,"axes.linewidth":0.6,"axes.edgecolor":"#52514e",
                     "xtick.color":"#52514e","ytick.color":"#52514e","xtick.major.width":0.5,"ytick.major.width":0.5})
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
CARD={m:{**load(f"{R5}/card_{m}.json"),**load(f"{R5}/card_{m}_*.json")} for m in M}
NIGHT={m:{**load(f"{R5}/card_night_{m}.json"),**load(f"{R5}/card_night_{m}_*.json")} for m in M}
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
DV={o["uid"]:o for o in json.load(open(f"{P}/det_validate.json"))}
x=IDX[UID]
fig=plt.figure(figsize=(7.16,2.55))
G=GridSpec(1,4,figure=fig,width_ratios=[1.42,1.95,0.16,1.42],wspace=0.12,left=0.005,right=0.985,top=0.88,bottom=0.14)
gA=G[0].subgridspec(3,1,hspace=0.12); gB=G[1].subgridspec(3,3,wspace=0.16,hspace=0.42,height_ratios=[0.02,1,1])
# (a) 三张图
a=np.asarray(Image.open(x["img"]).convert("RGB")); r=np.asarray(Image.open(f"/data/dataset/risk_card/{x['rm_name']}").convert("RGB"))
n=transform(a,kind="night",scope="global",seed=zlib.crc32(UID.encode())%(2**31))
y0,y1=330,740; reg=DV[UID]["region"]
for i,(im,lab) in enumerate(((a,"original"),(r,"pedestrian removed"),(n,"night rendering"))):
    ax=fig.add_subplot(gA[i,0]); ax.imshow(im[y0:y1]); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    if i==0:
        ax.add_patch(plt.Rectangle((reg[0]-6,reg[1]-y0-6),reg[2]-reg[0]+12,reg[3]-reg[1]+12,fill=False,ec="#e34948",lw=1.0))
    ax.text(8,22,lab,color="white",fontsize=6.5,va="top",bbox=dict(fc="black",alpha=0.55,lw=0,pad=1.2))
    if i==0: ax.set_title("(a) one frame, two interventions",fontsize=7.5,loc="left",pad=3)
# (b) 上：一张 BEV，六家规划各一条（原图输入）；下：同一帧的 F 与 I 逐家对比
cen=np.mean(np.asarray(x["corners_ego"])[:,:2],0)
def _d(u,v):
    u=np.asarray(u,float)[:,:2]; v=np.asarray(v,float)[:,:2]; n=min(len(u),len(v))
    return float(np.mean(np.linalg.norm(u[:n]-v[:n],axis=1)))
# (b) 六个模型各一格：原图 / 移除行人 / 夜化 三条规划，用三种颜色而非三种线型区分
cen=np.mean(np.asarray(x["corners_ego"])[:,:2],0)
CC={"clean":"#2a5db0","rm":"#c0392b","night":"#8a8a84"}
for k,m in enumerate(M):
    ax=fig.add_subplot(gB[1+k//3,k%3])
    ax.set_xlim(-3.2,3.2); ax.set_ylim(-1,21); ax.set_xticks([])
    ax.set_yticks([0,10,20] if k%3==0 else []); ax.tick_params(labelsize=5.4,length=2)
    ax.axvspan(-1.0,1.0,color="#eef1f6",lw=0,zorder=0)
    ax.plot(-cen[1],cen[0],marker="*",ms=6,color="#e34948",zorder=5,mec="white",mew=0.4)
    ax.plot(0,0,marker="^",ms=4,color="#0b0b0b")
    ax.set_title(NAME[m],fontsize=6.1,pad=1.5,color="#0b0b0b")
    c=CARD[m].get(UID); nn=NIGHT[m].get(UID)
    if not c or "actual" not in c:
        ax.text(0,10,"n/a",ha="center",color="#9a998f"); continue
    for key,lw in (("clean",1.5),("rm",1.2)):
        w=np.vstack([[0,0],np.asarray(c["actual"][key])]); ax.plot(-w[:,1],w[:,0],color=CC[key],lw=lw,zorder=3 if key=="clean" else 4)
    if nn:
        w=np.vstack([[0,0],np.asarray(nn["night"])]); ax.plot(-w[:,1],w[:,0],color=CC["night"],lw=1.2,zorder=4)
    for s_ in ("top","right"): ax.spines[s_].set_visible(False)
hh=[plt.Line2D([],[],color=CC["clean"],lw=1.5,label="original"),
    plt.Line2D([],[],color=CC["rm"],lw=1.2,label="pedestrian removed"),
    plt.Line2D([],[],color=CC["night"],lw=1.2,label="night"),
    plt.Line2D([],[],color="#e34948",marker="*",ms=6,lw=0,label="pedestrian")]
tb=fig.add_subplot(gB[0,:]); tb.axis("off")
tb.set_title("(b) plans over 2.5 s under each edit",fontsize=7.5,loc="left",pad=3)
fig.legend(handles=hh,frameon=False,fontsize=5.9,ncol=4,loc="lower center",bbox_to_anchor=(0.455,-0.035),handlelength=1.5,columnspacing=1.1)
# (c) 危险敏感度曲线（v2 口径，左右舵合并，近距有干涉单位）
rng=np.random.default_rng(0)
ax=fig.add_subplot(G[3])
rows=json.load(open(f"{V5}/diag_units.json"))
sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0 and z["need"]
BINS=[(0,0.5),(0.5,1),(1,2),(2,4),(4,99)]; XL=["<0.5","0.5–1","1–2","2–4",">4"]
for m in M:
    need=[z for z in rows if z["m"]==m and sel(z)]
    xs=[];ys=[];lo=[];hi=[]
    for i,(a_,b_) in enumerate(BINS):
        v=np.array([z["HS"] for z in need if a_<=z["a_req"]<b_])
        if len(v)>=8:
            bb=[rng.choice(v,len(v)).mean() for _ in range(1000)]
            xs.append(i+(M.index(m)-2.5)*0.07); ys.append(v.mean()); lo.append(v.mean()-np.percentile(bb,2.5)); hi.append(np.percentile(bb,97.5)-v.mean())
    ax.errorbar(xs,ys,yerr=[lo,hi],color=COL[m],marker="o",ms=3,lw=1.1,elinewidth=0.6,capsize=0,label=NAME[m])
# 真值参照：把"看得见行人"那一侧换成真人当时实际开出来的轨迹（gt_ceiling.py），
# 对同一个盲规划算 HS。虚线 = 真人，浅灰带 = 完美让开的构造上限，两者基本重合。
GC=json.load(open(f"{V5}/gt_ceiling.json"))
ax.plot(range(len(BINS)),GC["gt"],color="#52514e",ls=(0,(3,2)),lw=1.1,label="human driver (logged)")
ax.fill_between(range(len(BINS)),GC["gt"],GC["ideal"],color="#9a998f",alpha=0.18,lw=0)
ax.axhline(0,color="#c3c2b7",lw=0.6)
ax.set_xticks(range(len(BINS))); ax.set_xticklabels(XL); ax.set_xlabel(r"hazard level $a_{\rm req}=v^2/2d$ (m/s$^2$)")
ax.set_ylabel("hazard sensitivity HS",labelpad=1); ax.set_ylim(-0.25,1.0)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
ax.legend(frameon=False,fontsize=5.3,loc="upper left",handlelength=1.6,labelspacing=0.25)
ax.set_title("(c) hazard-sensitivity curve",fontsize=7.5,loc="left",pad=3)
rng=np.random.default_rng(0)
fig.savefig(f"{V5}/figures/teaser.pdf"); fig.savefig(f"{V5}/figures/teaser.png",dpi=220)
print("ok")
