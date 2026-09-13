"""Fig. 1（方法总览）：一帧 → 两种反事实编辑 → 同一策略三次查询 → 四项读数 → 体检报告。
左栏用真实图像（原图 / 抹掉行人 / 夜化，带独立检测器的核验结果）；
中栏是同一帧三条规划与行人真实未来，并标出净间隙 c_t；右栏是四项读数如何汇成各项考试。
数据与图 2 同源（card_*.json / card_night_*.json / risk_card_manifest.json）。"""
import json,glob,sys,zlib,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from PIL import Image
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts")
from appearance_transform import transform
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; P=f"{R5}/paper_icra_v4"; V5=f"{R5}/paper_icra_v5"; UID="A_0095"; MM="ddv2"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.4,"axes.linewidth":0.6,"axes.edgecolor":"#52514e"})
def load(pat):
    o={}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)):
            if "err" not in r: o[r["uid"]]=r
    return o
CARD={**load(f"{R5}/card_{MM}.json"),**load(f"{R5}/card_{MM}_*.json")}
NIGHT={**load(f"{R5}/card_night_{MM}.json"),**load(f"{R5}/card_night_{MM}_*.json")}
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
DV={o["uid"]:o for o in json.load(open(f"{P}/det_validate.json"))}
x=IDX[UID]; c=CARD[UID]; nn=NIGHT.get(UID)
fig=plt.figure(figsize=(7.16,1.62))
G=fig.add_gridspec(1,3,width_ratios=[1.30,1.05,1.45],wspace=0.06,left=0.004,right=0.996,top=0.90,bottom=0.04)
CC={"clean":"#2a5db0","rm":"#c0392b","night":"#8a8a84"}
# ---- (a) 一帧，两种编辑
gA=G[0].subgridspec(3,1,hspace=0.14)
a0=np.asarray(Image.open(x["img"]).convert("RGB")); r0=np.asarray(Image.open(f"/data/dataset/risk_card/{x['rm_name']}").convert("RGB"))
n0=transform(a0,kind="night",scope="global",seed=zlib.crc32(UID.encode())%(2**31))
y0,y1=330,740; reg=DV[UID]["region"]
TAG=[("logged frame","detector: pedestrian found"),("pedestrian removed","detector: not found"),("night rendering","detector: still found")]
for i,(im,(lab,det)) in enumerate(zip((a0,r0,n0),TAG)):
    ax=fig.add_subplot(gA[i,0]); ax.imshow(im[y0:y1]); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    if i==0: ax.add_patch(plt.Rectangle((reg[0]-6,reg[1]-y0-6),reg[2]-reg[0]+12,reg[3]-reg[1]+12,fill=False,ec="#e34948",lw=0.9))
    ax.text(6,16,lab,color="white",fontsize=4.8,va="top",bbox=dict(fc=CC[["clean","rm","night"][i]],alpha=0.85,lw=0,pad=1.0))
    ax.text(6,im[y0:y1].shape[0]-8,det,color="white",fontsize=4.2,va="bottom",bbox=dict(fc="black",alpha=0.5,lw=0,pad=0.8))
    if i==0: ax.set_title("(a) one frame, two counterfactual edits",fontsize=6.6,loc="left",pad=2.5)
# ---- (b) 三次查询 → 三条规划
b=fig.add_subplot(G[1])
cen=np.mean(np.asarray(x["corners_ego"])[:,:2],0)
b.axvspan(-1,1,color="#eef1f6",lw=0,zorder=0)
for key,lw in (("clean",1.5),("rm",1.2)):
    w=np.vstack([[0,0],np.asarray(c["actual"][key])]); b.plot(-w[:,1],w[:,0],color=CC[key],lw=lw,zorder=3)
if nn:
    w=np.vstack([[0,0],np.asarray(nn["night"])]); b.plot(-w[:,1],w[:,0],color=CC["night"],lw=1.2,zorder=3)
b.plot(-cen[1],cen[0],marker="*",ms=7,color="#e34948",mec="white",mew=0.4,zorder=5)
b.annotate("",xy=(-cen[1],cen[0]),xytext=(0.35,cen[0]-0.6),arrowprops=dict(arrowstyle="<->",lw=0.7,color="#52514e"))
b.text(-cen[1]/2+0.15,cen[0]+0.9,"$c_t$",fontsize=6.2,color="#52514e",ha="center")
b.plot(0,0,marker="^",ms=5,color="#0b0b0b",zorder=5)
b.set_xlim(-4.2,4.2); b.set_ylim(-1.5,21); b.set_xticks([]); b.set_yticks([])
for s_ in ("top","right","left","bottom"): b.spines[s_].set_visible(False)
b.set_title("(b) same policy, three queries",fontsize=6.6,loc="left",pad=2.5)
b.text(0.5,-0.04,"clearance to the pedestrian's logged future",transform=b.transAxes,ha="center",va="top",fontsize=4.9,color="#52514e")
# ---- (c) 读数 → 考试
d=fig.add_subplot(G[2]); d.axis("off"); d.set_xlim(0,1); d.set_ylim(0,1)
d.set_title("(c) four readouts, five exams",fontsize=6.6,loc="left",pad=2.5)
box=lambda xx,yy,w_,h_,t,fc: (d.add_patch(FancyBboxPatch((xx,yy),w_,h_,boxstyle="round,pad=0.012",fc=fc,ec="#c9c8c0",lw=0.5,transform=d.transAxes)),
                              d.text(xx+w_/2,yy+h_/2,t,transform=d.transAxes,ha="center",va="center",fontsize=5.0,color="#1b2226"))
box(0.02,0.62,0.40,0.30,"contact $A$  ·  clearance $C$\ntime $T$  ·  separation $S$","#eef1f6")
box(0.02,0.18,0.40,0.30,"read on $P$, $Q$, $N$\nover the common 2.5 s","#f6f1ea")
for yy,t in ((0.80,"exposure"),(0.62,"hazard sensitivity"),(0.44,"scaling"),(0.26,"specificity"),(0.08,"lighting CFR")):
    d.add_patch(FancyBboxPatch((0.60,yy-0.055),0.38,0.115,boxstyle="round,pad=0.008",fc="#e9efe9",ec="#b9cbb9",lw=0.5,transform=d.transAxes))
    d.text(0.79,yy,t,transform=d.transAxes,ha="center",va="center",fontsize=5.2,color="#1b2226")
for y_ in (0.80,0.62,0.44,0.26,0.08):
    d.add_patch(FancyArrowPatch((0.44,0.50),(0.585,y_),transform=d.transAxes,arrowstyle="-|>",mutation_scale=4.5,lw=0.5,color="#9a998f",shrinkA=0,shrinkB=1))
for x0,x1_ in ((0.335,0.365),(0.625,0.655)):
    fig.add_artist(FancyArrowPatch((x0,0.5),(x1_,0.5),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=6,lw=0.8,color="#52514e"))
fig.savefig(f"{V5}/figures/method.pdf",bbox_inches="tight"); fig.savefig(f"{V5}/figures/method.png",dpi=230,bbox_inches="tight"); print("ok")
