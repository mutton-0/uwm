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
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Wedge
from PIL import Image
MODE=sys.argv[1] if len(sys.argv)>1 else "wide"
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
sys.path.insert(0,"/home/boyuewang/120/uwm/sim2real_demo_ttc/scripts"); sys.path.insert(0,f"{V5}/scripts")
from appearance_transform import transform
from fi_panels import panel_extract,panel_readings,frame_extract,frame_readings
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
IMGS=(a0,r0,d0,n0); CC=["#2a5db0","#c0392b","#b9814a","#8a8a84"]; NIMG=len(IMGS)
LAB=[("logged $O$","detector: pedestrian found"),
     ("removed $R$","detector: not found"),
     ("re-lit $D$","detector: still found"),
     ("re-lit $N$","detector: still found")]   # A_0095 黄昏档 IoU 0.88，与 dusk_check 同判据
EX=["exposure","hazard\nsensitivity","scaling","specificity","lighting"]
PR=["orderings\ntransfer","verdicts\ntransfer","point values\ndo not","priced\nin frames"]

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
else:                                       # 单栏竖版：四段上下叠 + 体检单
    FW=3.45; IW_in=(FW-2*0.05-(NIMG-1)*0.04-2*0.03)/NIMG; IH_in=IW_in/AR
    GAP_in=0.04; PAD_in=0.05; TIT_in=0.145
    H_A=TIT_in+PAD_in+IH_in+PAD_in
    H_B=TIT_in+1.12
    H_C=TIT_in+0.30+2*PAD_in
    H_D=TIT_in+0.72+2*PAD_in
    ARR=0.14
    FH=H_A+ARR+H_B+ARR+H_C+ARR+H_D+0.04
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
    def stage(top,h,lab):
        bot=top-fy(h); box(L,bot,R,top); bg.text(L+fx(0.05),top-fy(0.115),lab,fontsize=7.0,weight="bold",color="#2b2b28"); return bot
    def down(y,txt):
        arrow(((L+R)/2,y-fy(0.018)),((L+R)/2,y-ARRf+fy(0.018)),lw=1.6,ms=6)
        bg.text((L+R)/2+fx(0.08),y-ARRf/2,txt,fontsize=5.0,va="center",color="#2f5d94"); return y-ARRf
    # (a) 配对数据
    aT=1-fy(0.01); aB=stage(aT,H_A,"(a) paired data")
    for i in range(NIMG): draw_img(fig,[L+fx(0.05)+i*(IW+fx(0.04)),aB+PAD,IW,IH],i,fs=0.88)
    bT=down(aB,"query $\\pi$ on $O,R,D,N$")
    # (b) 两条轴
    bB=stage(bT,H_B,"(b) the two axes")
    PB=bB+fy(0.24); PT=bT-fy(0.26)
    ax1=fig.add_axes([L+fx(0.10),PB,fx(1.25),PT-PB]); frame_extract(ax1); ax1.set_title("three queries, one scene",fontsize=6.0,loc="left",pad=2)
    ax2=fig.add_axes([L+fx(1.80),PB,fx(1.42),PT-PB]); frame_readings(ax2); ax2.set_title("the two readings",fontsize=6.0,loc="left",pad=2)
    cT=down(bB,"read $F$, $I$ per unit")
    # (c) 五项检查
    cB=stage(cT,H_C,"(c) the five exams")
    CW_=(R-L-2*fx(0.05)-4*fx(0.035))/5
    for k,lab in enumerate(EX):
        x0=L+fx(0.05)+k*(CW_+fx(0.035)); chip(x0,cB+PAD,x0+CW_,cB+PAD+fy(0.30),lab,4.7)
    dT=down(cB,"diagnose on one side, test on the other")
    # (d) 预后 + 体检单
    dB=stage(dT,H_D,"(d) prognosis")
    GW=(R-L)*0.60; CW2=(GW-fx(0.05)-fx(0.035))/2; CH2=fy(0.33)
    for k,lab in enumerate(PR):
        r_,c_=divmod(k,2); x0=L+fx(0.05)+c_*(CW2+fx(0.035)); y1=dT-TIT-fy(0.02)-r_*(CH2+fy(0.045))
        chip(x0,y1-CH2,x0+CW2,y1,lab,4.7)
    # 箭头指向体检单
    mid=(dT-TIT+dB)/2
    arrow((L+fx(0.05)+GW-fx(0.02),mid),(L+fx(0.05)+GW+fx(0.16),mid),lw=1.4,ms=6)
    NX=L+fx(0.05)+GW+fx(0.62); NY=mid; NW=fx(0.46); NH=fy(0.56)
    clipboard(NX,NY,NW,NH,4.8)
    # 医生小人：头 + 肩 + 胸前十字 + 听诊器
    DX=NX+NW/2+fx(0.30); DY=mid+fy(0.02)
    bg.add_patch(plt.Circle((DX,DY+fy(0.19)),fx(0.075),fc="#f1d7c2",ec="#4a4a46",lw=0.8,zorder=5))
    bg.add_patch(plt.Circle((DX,DY+fy(0.19)),fx(0.075),fc="none",ec="#4a4a46",lw=0.8,zorder=6))
    bg.add_patch(Wedge((DX,DY-fy(0.07)),fx(0.15),0,180,fc="white",ec="#4a4a46",lw=0.8,zorder=5))
    bg.plot([DX-fx(0.03),DX+fx(0.03)],[DY+fy(0.02)]*2,color="#c0392b",lw=1.2,zorder=6); bg.plot([DX]*2,[DY-fy(0.005),DY+fy(0.045)],color="#c0392b",lw=1.2,zorder=6)
    th=np.linspace(np.pi*0.15,np.pi*0.95,30)
    bg.plot(DX+fx(0.09)*np.cos(th)-fx(0.02),DY-fy(0.02)+fy(0.09)*np.sin(th)-fy(0.06),color="#2f5d94",lw=1.0,zorder=6)
    bg.add_patch(plt.Circle((DX-fx(0.02)+fx(0.09)*np.cos(th[0]),DY-fy(0.08)+fy(0.09)*np.sin(th[0])),fx(0.018),fc="#2f5d94",ec="none",zorder=7))
    OUT="frame1c"
fig.savefig(f"{V5}/figures/{OUT}.pdf")
fig.savefig(f"{V5}/figures/{OUT}.png",dpi=300)   # 版式已填满画布，不用 tight，免得裁出不一致的边
print(OUT,"ok",round(FW,2),"x",round(FH,2),"in")
