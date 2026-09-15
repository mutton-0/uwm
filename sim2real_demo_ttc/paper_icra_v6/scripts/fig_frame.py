"""Fig. 1（方法总览，一个走通的实例）：场景编辑 → 诊断 → 体检结论。
(a) 一帧三种输入：原图 O / 抹掉行人 R / 重打光 N，抠图由独立检测器核验（找到 / 找不到 / 仍找到）。
(b) 同一策略对三种输入各问一次，读出两个数：F（行人造成的位移）与 I（光照造成的位移），
    右侧把六家在 F–I 平面上的读数一并给出。两个面板由 fi_panels.py 提供，与 fig_fi_dash 同源。
(c) 两个数连同其余读数汇成五项检查。
用法：fig_frame.py [wide|col]
  wide → figures/frame.pdf     跨栏 \textwidth 横排（三段左右并列）
  col  → figures/frame1c.pdf   单栏 \columnwidth 竖排（三段上下叠，字号不缩）
版式一律按英寸推导，内容正好填满画布，四周不留空。"""
import json,sys,zlib,numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from PIL import Image
MODE=sys.argv[1] if len(sys.argv)>1 else "wide"
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts"); sys.path.insert(0,f"{V5}/scripts")
from appearance_transform import transform
from fi_panels import panel_extract,panel_readings
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.4,"axes.linewidth":0.6,"axes.edgecolor":"#52514e",
                     "xtick.color":"#52514e","ytick.color":"#52514e"})
UID="A_0095"
MAN=json.load(open(f"{R5}/risk_card_manifest.json")); IDX={x["uid"]:x for k in ("A","B") for x in MAN[k]}
DV={o["uid"]:o for o in json.load(open(f"{R5}/paper_icra_v4/det_validate.json"))}
x=IDX[UID]; reg=DV[UID]["region"]
a0=np.asarray(Image.open(x["img"]).convert("RGB"))
r0=np.asarray(Image.open(f"/data/dataset/risk_card/{x['rm_name']}").convert("RGB"))
n0=transform(a0,kind="night",scope="global",seed=zlib.crc32(UID.encode())%(2**31))
d0=transform(a0,kind="dusk", scope="global",seed=zlib.crc32(UID.encode())%(2**31))   # 同帧同 seed 的黄昏档
cx=(reg[0]+reg[2])//2; cy=(reg[1]+reg[3])//2               # 以检测框为中心裁到 1.55:1，去掉上下无关留白
CH=int((reg[3]-reg[1])*1.55); CW=int(CH*1.55)
X0=max(0,min(a0.shape[1]-CW,cx-CW//2)); Y0=max(0,min(a0.shape[0]-CH,cy-CH//2))
CROP=(slice(Y0,Y0+CH),slice(X0,X0+CW)); AR=CW/CH
IMGS=(a0,r0,n0,d0); CC=["#2a5db0","#c0392b","#8a8a84","#b9814a"]; NIMG=len(IMGS)
LAB=[("logged $O$","detector: pedestrian found"),
     ("removed $R$","detector: not found"),
     ("re-lit $N$, night","detector: still found"),
     ("re-lit $N$, dusk","detector: still found")]   # A_0095 黄昏档 IoU 0.88，与 dusk_check 同判据
EX=["exposure","hazard\nsensitivity","scaling","specificity","lighting\n$\\mathrm{CFR}$"]

def draw_img(fig,rect,i,fs=1.0):
    ax=fig.add_axes(rect); ax.imshow(IMGS[i][CROP]); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    if i==0: ax.add_patch(plt.Rectangle((reg[0]-X0-5,reg[1]-Y0-5),reg[2]-reg[0]+10,reg[3]-reg[1]+10,
                                        fill=False,ec="#e34948",lw=0.9))
    ax.text(5,13,LAB[i][0],color="white",fontsize=4.6*fs,va="top",bbox=dict(fc=CC[i],alpha=0.88,lw=0,pad=1.0))
    ax.text(5,CH-6,LAB[i][1],color="white",fontsize=4.0*fs,va="bottom",bbox=dict(fc="black",alpha=0.52,lw=0,pad=0.8))

if MODE=="wide":
    FW=7.16; IW_in=0.93; IH_in=IW_in/AR
    GAP_in=0.04; PAD_in=0.05; TIT_in=0.16
    FH=TIT_in+PAD_in+NIMG*IH_in+(NIMG-1)*GAP_in+PAD_in+0.03
else:                                       # 单栏竖版：三段上下叠，字号与跨栏版一致
    FW=3.45; IW_in=(FW-2*0.05-(NIMG-1)*0.04-2*0.03)/NIMG; IH_in=IW_in/AR
    GAP_in=0.04; PAD_in=0.05; TIT_in=0.145
    H_A=TIT_in+PAD_in+IH_in+PAD_in
    H_B=TIT_in+1.30
    H_C=TIT_in+0.31
    ARR=0.15
    FH=H_A+ARR+H_B+ARR+H_C+0.03
fig=plt.figure(figsize=(FW,FH))
def fx(v): return v/FW
def fy(v): return v/FH
bg=fig.add_axes([0,0,1,1]); bg.set_xlim(0,1); bg.set_ylim(0,1); bg.axis("off")
def box(x0,y0,x1,y1):
    bg.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=0.012",
                                fc="white",ec="#c9c7c0",lw=0.8,zorder=0))
def arrow(p0,p1,lw=2.2,ms=9):
    bg.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=ms,lw=lw,
                                 color="#2f5d94",shrinkA=0,shrinkB=0,zorder=3))
def chip(x0,y0,x1,y1,lab,fs):
    bg.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=0.010",
                                fc="#e8efe8",ec="#b9c9b9",lw=0.7,zorder=3))
    bg.text((x0+x1)/2,(y0+y1)/2,lab,fontsize=fs,ha="center",va="center",color="#243024",linespacing=1.05)
def clipboard(NX,NY,NW,NH,fs):                 # 问诊节点：一张体检单
    bg.add_patch(FancyBboxPatch((NX-NW/2,NY-NH/2),NW,NH,boxstyle="round,pad=0,rounding_size=0.010",
                                fc="#f4f6f4",ec="#4a4a46",lw=1.0,zorder=4))
    bg.add_patch(FancyBboxPatch((NX-NW*0.22,NY+NH/2-NH*0.16),NW*0.44,NH*0.21,
                                boxstyle="round,pad=0,rounding_size=0.005",fc="#4a4a46",ec="#4a4a46",lw=0.8,zorder=5))
    for j in range(4):
        yy=NY+NH/2-NH*0.32-j*NH*0.20
        bg.plot([NX-NW*0.14,NX+NW*0.33],[yy,yy],color="#9a998f",lw=0.8,zorder=5,solid_capstyle="round")
        bg.plot([NX-NW*0.37,NX-NW*0.29,NX-NW*0.19],[yy,yy-NH*0.056,yy+NH*0.056],color="#2f7d4f",lw=0.9,
                zorder=5,solid_capstyle="round",solid_joinstyle="round")
    bg.text(NX,NY-NH/2-fy(0.03),"check-up report",fontsize=fs,ha="center",va="top",color="#52514e")

if MODE=="wide":
    IW=fx(IW_in); IH=fy(IH_in); GAP=fy(GAP_in); PAD=fy(PAD_in)
    TOP=1-fy(TIT_in); ITOP=TOP-PAD; BOT=ITOP-NIMG*IH-(NIMG-1)*GAP-PAD
    MID=(TOP+BOT)/2; TY=TOP+fy(0.035)
    AX0=fx(0.03); AX1=AX0+IW+2*fx(0.035)
    box(AX0,BOT,AX1,TOP); bg.text(AX0+fx(0.01),TY,"(a) scenario edit",fontsize=7.4,weight="bold",color="#2b2b28")
    for i in range(NIMG): draw_img(fig,[AX0+fx(0.035),ITOP-(i+1)*IH-i*GAP,IW,IH],i)
    arrow((AX1+fx(0.04),MID),(AX1+fx(0.18),MID))
    BX0=AX1+fx(0.22); BX1=BX0+fx(3.42)
    box(BX0,BOT,BX1,TOP); bg.text(BX0+fx(0.04),TY,"(b) diagnosis",fontsize=7.4,weight="bold",color="#2b2b28")
    PB=BOT+fy(0.30); PT=TOP-fy(0.16)
    panel_extract(fig.add_axes([BX0+fx(0.30),PB,fx(0.88),PT-PB]),title="three queries, one scene")
    panel_readings(fig.add_axes([BX0+fx(1.84),PB,fx(1.47),PT-PB]),title="the two readings")
    arrow((BX1+fx(0.04),MID),(BX1+fx(0.18),MID))
    CX0=BX1+fx(0.22)
    bg.text(CX0,TY,"(c) prognosis",fontsize=7.4,weight="bold",color="#2b2b28")
    NX,NY=CX0+fx(0.33),MID; NW,NH=fx(0.36),fy(0.62)
    clipboard(NX,NY,NW,NH,5.2)
    EBX0=CX0+fx(1.06); EBX1=1-fx(0.03); EH=(TOP-BOT-4*fy(0.035))/5
    for k,lab in enumerate(EX):
        yc=TOP-EH/2-k*(EH+fy(0.035)); chip(EBX0,yc-EH/2,EBX1,yc+EH/2,lab.replace("\n$","  $"),6.2)
        bg.add_patch(FancyArrowPatch((NX+NW/2+fx(0.02),NY+(yc-NY)*0.10),(EBX0-fx(0.03),yc),arrowstyle="-|>",
                                     mutation_scale=6,lw=0.9,color="#52514e",shrinkA=1,shrinkB=1,zorder=2))
    OUT="frame"
else:
    IW=fx(IW_in); IH=fy(IH_in); PAD=fy(PAD_in); TIT=fy(TIT_in); ARRf=fy(ARR)
    L,R=fx(0.03),1-fx(0.03)
    # (a) 三张图横排
    aT=1-fy(0.01); aB=aT-TIT-PAD-IH-PAD
    box(L,aB,R,aT); bg.text(L+fx(0.05),aT-fy(0.115),"(a) scenario edit",fontsize=7.0,weight="bold",color="#2b2b28")
    for i in range(NIMG): draw_img(fig,[L+fx(0.05)+i*(IW+fx(0.04)),aB+PAD,IW,IH],i,fs=0.88)
    arrow(((L+R)/2,aB-fy(0.022)),((L+R)/2,aB-ARRf+fy(0.022)),lw=1.6,ms=6)
    # (b) 两个面板
    bT=aB-ARRf; bB=bT-TIT-1.30/FH
    box(L,bB,R,bT); bg.text(L+fx(0.05),bT-fy(0.115),"(b) diagnosis",fontsize=7.0,weight="bold",color="#2b2b28")
    PB=bB+fy(0.28); PT=bT-fy(0.27)
    panel_extract(fig.add_axes([L+fx(0.30),PB,fx(0.86),PT-PB]),title="three queries, one scene",compact=True)
    panel_readings(fig.add_axes([L+fx(1.72),PB,fx(1.50),PT-PB]),title="the two readings",compact=True)
    arrow(((L+R)/2,bB-fy(0.022)),((L+R)/2,bB-ARRf+fy(0.022)),lw=1.6,ms=6)
    # (c) 五个检查横排
    cT=bB-ARRf; cB=cT-TIT-0.34/FH
    bg.text(L+fx(0.05),cT-fy(0.105),"(c) prognosis",fontsize=7.0,weight="bold",color="#2b2b28")
    CW_=(R-L-4*fx(0.035))/5
    for k,lab in enumerate(EX):
        x0=L+k*(CW_+fx(0.035)); chip(x0,cB,x0+CW_,cT-TIT-fy(0.01),lab,4.7)
    OUT="frame1c"
fig.savefig(f"{V5}/figures/{OUT}.pdf")
fig.savefig(f"{V5}/figures/{OUT}.png",dpi=300)   # 版式已填满画布，不用 tight，免得裁出不一致的边
print(OUT,"ok",round(FW,2),"x",round(FH,2),"in")
