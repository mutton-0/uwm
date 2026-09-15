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
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Wedge,Ellipse
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
IMGS=(a0,r0,d0,n0); CC=["#1f4e79","#2e75b6","#5b9bd5","#17233b"]; NIMG=len(IMGS)
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
                                        fill=False,ec="#5b9bd5",lw=0.9))
    ax.text(5,13,LAB[i][0],color="white",fontsize=4.6*fs,va="top",bbox=dict(fc=CC[i],alpha=0.88,lw=0,pad=1.0))
    ax.text(5,CH-6,LAB[i][1],color="white",fontsize=4.0*fs,va="bottom",bbox=dict(fc="black",alpha=0.52,lw=0,pad=0.8))

if MODE=="wide":                            # 跨栏扁横版：四段左右排
    FW=7.16; IW_in=0.66; IH_in=IW_in/AR
    GAP_in=0.035; PAD_in=0.045; TIT_in=0.15
    FH=TIT_in+PAD_in+2*IH_in+GAP_in+PAD_in+0.02+0.22
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
                                fc="white",ec="#b7c6d8",lw=0.8,zorder=0))
def arrow(p0,p1,lw=2.2,ms=9):
    bg.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=ms,lw=lw,
                                 color="#2f5d94",shrinkA=0,shrinkB=0,zorder=3))
def chip(x0,y0,x1,y1,lab,fs):
    bg.add_patch(FancyBboxPatch((x0,y0),x1-x0,y1-y0,boxstyle="round,pad=0,rounding_size=0.010",
                                fc="#e9f0f8",ec="#a9bfd8",lw=0.7,zorder=3))
    bg.text((x0+x1)/2,(y0+y1)/2,lab,fontsize=fs,ha="center",va="center",color="#17233b",linespacing=1.05)
def clipboard(NX,NY,NW,NH,fs):                 # 问诊节点：一张体检单
    bg.add_patch(FancyBboxPatch((NX-NW/2,NY-NH/2),NW,NH,boxstyle="round,pad=0,rounding_size=0.010",
                                fc="#f2f6fb",ec="#4a4a46",lw=1.0,zorder=4))
    bg.add_patch(FancyBboxPatch((NX-NW*0.22,NY+NH/2-NH*0.16),NW*0.44,NH*0.21,
                                boxstyle="round,pad=0,rounding_size=0.005",fc="#4a4a46",ec="#4a4a46",lw=0.8,zorder=5))
    for j in range(4):
        yy=NY+NH/2-NH*0.32-j*NH*0.20
        bg.plot([NX-NW*0.14,NX+NW*0.33],[yy,yy],color="#9a998f",lw=0.8,zorder=5,solid_capstyle="round")
        bg.plot([NX-NW*0.37,NX-NW*0.29,NX-NW*0.19],[yy,yy-NH*0.056,yy+NH*0.056],color="#2e75b6",lw=0.9,
                zorder=5,solid_capstyle="round",solid_joinstyle="round")
    bg.text(NX,NY-NH/2-fy(0.03),"check-up report",fontsize=fs,ha="center",va="top",color="#52514e")

if MODE=="wide":
    IW=fx(IW_in); IH=fy(IH_in); GAP=fy(GAP_in); PAD=fy(PAD_in); TIT=fy(TIT_in)
    TOP=1-fy(0.01); BOT=fy(0.01); CT=TOP-TIT; MID=(CT+BOT)/2
    ARW=fx(0.30); XPAD=fx(0.04)
    def stage(x0,w,lab):
        box(x0,BOT,x0+w,TOP); bg.text(x0+XPAD,TOP-fy(0.105),lab,fontsize=6.8,weight="bold",color="#2b2b28"); return x0+w
    def right(x,txt):
        arrow((x+fx(0.02),MID),(x+ARW-fx(0.02),MID),lw=1.5,ms=6)
        bg.text(x+ARW/2,MID+fy(0.045),txt,fontsize=4.0,ha="center",va="bottom",color="#2f5d94",linespacing=1.1); return x+ARW
    # (a) 配对数据：2×2
    x=fx(0.03); WA=2*XPAD+2*IW+fx(0.035); x1=stage(x,WA,"(a) paired data")
    for i in range(NIMG):
        r_,c_=divmod(i,2); draw_img(fig,[x+XPAD+c_*(IW+fx(0.035)),BOT+PAD+(1-r_)*(IH+GAP),IW,IH],i,fs=0.8)
    x=right(x1,"query $\\pi$ on\n$O,R,D,N$")
    # (b) 两条轴
    WB=fx(2.0); x1=stage(x,WB,"(b) the two axes"); PB=BOT+fy(0.20); PT=CT-fy(0.13)
    ax1=fig.add_axes([x+fx(0.05),PB,fx(0.74),PT-PB]); frame_extract(ax1,fs=0.85); ax1.set_title("three queries, one scene",fontsize=5.2,loc="left",pad=1.5)
    ax2=fig.add_axes([x+fx(1.06),PB+fy(0.02),fx(0.88),PT-PB-fy(0.02)]); frame_readings(ax2,fs=0.85); ax2.set_title("the two readings",fontsize=5.2,loc="left",pad=1.5)
    x=right(x1,"read\n$F$, $I$")
    # (c) 五项检查：两列
    WC=fx(0.90); x1=stage(x,WC,"(c) the five exams")
    cw=(WC-2*XPAD-fx(0.03))/2; ch=(CT-BOT-2*PAD-2*fy(0.03))/3
    for k,lab in enumerate(EX):
        c_,r_=divmod(k,3); x0=x+XPAD+c_*(cw+fx(0.03)); y1=CT-PAD-r_*(ch+fy(0.03))
        chip(x0,y1-ch,x0+cw,y1,lab,4.3)
    x=right(x1,"test on the\nother side")
    # (d) 预后 + 体检单
    WD=1-fx(0.03)-x; x1=stage(x,WD,"(d) prognosis")
    GW=fx(0.92); cw=(GW-fx(0.03))/2; ch=(CT-BOT-2*PAD-fy(0.03))/2
    for k,lab in enumerate(PR):
        r_,c_=divmod(k,2); x0=x+XPAD+c_*(cw+fx(0.03)); y1=CT-PAD-r_*(ch+fy(0.03))
        chip(x0,y1-ch,x0+cw,y1,lab,4.3)
    arrow((x+XPAD+GW+fx(0.02),MID),(x+XPAD+GW+fx(0.12),MID),lw=1.3,ms=5)
    NX=x+XPAD+GW+fx(0.33); NY=MID+fy(0.03); NW=fx(0.34); NH=fy(0.52)
    clipboard(NX,NY,NW,NH,4.3)
    DX=NX+NW/2+fx(0.22); DY=MID+fy(0.02)
    bg.add_patch(Ellipse((DX,DY+fy(0.17)),2*fx(0.065),2*fy(0.065),fc="#ffffff",ec="#4a4a46",lw=0.8,zorder=5))
    bg.add_patch(Wedge((DX,DY-fy(0.06)),fx(0.13),0,180,fc="white",ec="#4a4a46",lw=0.8,zorder=5))
    bg.plot([DX-fx(0.026),DX+fx(0.026)],[DY+fy(0.02)]*2,color="#2e75b6",lw=1.1,zorder=6); bg.plot([DX]*2,[DY-fy(0.003),DY+fy(0.043)],color="#2e75b6",lw=1.1,zorder=6)
    th=np.linspace(np.pi*0.15,np.pi*0.95,30)
    bg.plot(DX-fx(0.018)+fx(0.08)*np.cos(th),DY-fy(0.075)+fy(0.08)*np.sin(th),color="#2f5d94",lw=0.9,zorder=6)
    bg.add_patch(Ellipse((DX-fx(0.018)+fx(0.08)*np.cos(th[0]),DY-fy(0.075)+fy(0.08)*np.sin(th[0])),2*fx(0.016),2*fy(0.016),fc="#2f5d94",ec="none",zorder=7))
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
    bg.add_patch(Ellipse((DX,DY+fy(0.19)),2*fx(0.075),2*fy(0.075),fc="#ffffff",ec="#4a4a46",lw=0.8,zorder=5))
    bg.add_patch(Ellipse((DX,DY+fy(0.19)),2*fx(0.075),2*fy(0.075),fc="none",ec="#4a4a46",lw=0.8,zorder=6))
    bg.add_patch(Wedge((DX,DY-fy(0.07)),fx(0.15),0,180,fc="white",ec="#4a4a46",lw=0.8,zorder=5))
    bg.plot([DX-fx(0.03),DX+fx(0.03)],[DY+fy(0.02)]*2,color="#2e75b6",lw=1.2,zorder=6); bg.plot([DX]*2,[DY-fy(0.005),DY+fy(0.045)],color="#2e75b6",lw=1.2,zorder=6)
    th=np.linspace(np.pi*0.15,np.pi*0.95,30)
    bg.plot(DX+fx(0.09)*np.cos(th)-fx(0.02),DY-fy(0.02)+fy(0.09)*np.sin(th)-fy(0.06),color="#2f5d94",lw=1.0,zorder=6)
    bg.add_patch(Ellipse((DX-fx(0.02)+fx(0.09)*np.cos(th[0]),DY-fy(0.08)+fy(0.09)*np.sin(th[0])),2*fx(0.018),2*fy(0.018),fc="#2f5d94",ec="none",zorder=7))
    OUT="frame1c"
fig.savefig(f"{V5}/figures/{OUT}.pdf")
fig.savefig(f"{V5}/figures/{OUT}.png",dpi=300)   # 版式已填满画布，不用 tight，免得裁出不一致的边
print(OUT,"ok",round(FW,2),"x",round(FH,2),"in")
