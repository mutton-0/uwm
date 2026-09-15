"""F–I 两个面板的可复用画法，供 fig_fi_dash.py（独立图）与 fig_frame.py（方法总览图）共用。
panel_extract(ax)  : 一个场景三次问诊，逐时刻连线的长度均值即 F（对 rm）与 I（对重打光）。
panel_readings(ax) : F–I 平面读数盘；六家均值 + 逐场景淡点；六家四条件文件齐了才画 dusk。
数据：rhd_axes4_<m>.json（四条件）或 rhd_axes_<m>.json（三条件）+ nv_ped_future.json。"""
import json,os,numpy as np,matplotlib.pyplot as plt
V5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v5"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
SH={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo"}
COL={"dd":"#2a78d6","ltf":"#eb6834","ddv2":"#1baf7a","simlingo":"#eda100","autovla":"#e87ba4","alpamayo15":"#6a3d9a"}
LOFF={"autovla":(4.5,3.6),"ltf":(4.5,-3.0)}      # AutoVLA 与 LTF 读数贴得近，标签错开
TT=np.round(np.arange(0,2.51,0.1),2); T8=np.r_[0,np.arange(1,9)*0.5]
HAS4=all(os.path.exists(f"{V5}/rhd_axes4_{m}.json") for m in M)
AX={m:json.load(open(f"{V5}/rhd_axes{'4' if HAS4 else ''}_{m}.json")) for m in M}
PF=json.load(open(f"{V5}/nv_ped_future.json"))
def lin(P):
    P=np.vstack([[0,0],np.asarray(P,float)[:,:2]])
    return np.stack([np.interp(TT,T8,P[:,0]),np.interp(TT,T8,P[:,1])],1)
def disp(a,b): return float(np.mean(np.linalg.norm(lin(a)-lin(b),axis=1)))
def ped_xy(t):
    p=PF[t]; P=np.asarray([p["p0"]]+[z for z in p["fut"] if z is not None],float); tp=np.linspace(0,2.5,len(P))
    return np.stack([np.interp(TT,tp,P[:,0]),np.interp(TT,tp,P[:,1])],1)

def panel_extract(a,tok="34b62d7333845af3",mm="ddv2",title="(a) three queries, one scene",fs=1.0,legend=True,compact=False):
    """选例条件：盲规划会进 1.4 m 禁入范围而原规划没进，且 F>I —— 真避让，方向可解释。"""
    v=AX[mm][tok]; P={k:lin(v[k]) for k in ("clean","rm","night")}; ped=ped_xy(tok)
    a.axvspan(-1,1,color="#eef1f6",lw=0,zorder=0)
    a.plot(-ped[:,1],ped[:,0],color="#b3412c",ls=(0,(1.3,1.3)),lw=0.9,zorder=5)
    a.plot(-ped[0,1],ped[0,0],marker="*",ms=6*fs,color="#b3412c",mec="white",mew=0.4,zorder=6)
    for k,c in (("night","#8d8b84"),("rm","#e34948")):     # 每 0.25 s 一根连线，长度的时间均值就是 I / F
        for i in range(0,len(TT),3):
            a.plot([-P["clean"][i,1],-P[k][i,1]],[P["clean"][i,0],P[k][i,0]],color=c,lw=0.45,alpha=0.75,zorder=2)
    for k,c,lw,z in (("night","#8d8b84",0.9,3),("rm","#e34948",0.9,3),("clean","#2a78d6",1.3,4)):
        a.plot(-P[k][:,1],P[k][:,0],color=c,lw=lw,ls="-" if k=="clean" else (0,(2.6,1.6)),zorder=z)
        a.plot(-P[k][::5,1],P[k][::5,0],ls="none",marker="o",ms=1.6*fs,color=c,zorder=z+1)
    a.plot(0,0,marker="^",ms=4.5*fs,color="#0b0b0b",zorder=6)
    for k,c,dy,tl in (("rm","#e34948",15,"contact"),("clean","#2a78d6",-13,"clear")):   # 全程最小间隙，扣掉 1.0 自车半宽 + 0.4 行人半径
        g=float(np.min(np.linalg.norm(P[k]-ped,axis=1))-1.4)
        dx,dy=(9,int(dy*0.72)) if compact else (11,dy)      # 面板矮的时候标注要收，不然顶到标题
        tl="contact" if g<0 else "clear"
        a.annotate(f"{g:+.2f} m {tl}",(-P[k][-1,1],P[k][-1,0]),textcoords="offset points",xytext=(dx,dy),
                   ha="left",va="center",fontsize=4.6*fs,color=c,
                   arrowprops=dict(arrowstyle="-",lw=0.4,color=c,shrinkA=0.5,shrinkB=1.5))
    a.text(0.02,0.955,f"$F$ = {disp(v['clean'],v['rm']):.2f} m",transform=a.transAxes,ha="left",fontsize=5.4*fs,color="#b3412c")
    a.text(0.02,0.855,f"$I$ = {disp(v['clean'],v['night']):.2f} m",transform=a.transAxes,ha="left",fontsize=5.4*fs,color="#5e5c56")
    a.set_xlim(-6.0,5.6); a.set_ylim(-1.5,19.5 if compact else 17.5); a.set_xticks([-3,0,3]); a.set_yticks([0,5,10,15])
    a.tick_params(labelsize=5.0*fs,length=1.6,pad=1)
    a.set_xlabel("lateral (m)",fontsize=5.4*fs,labelpad=0.5); a.set_ylabel("ahead (m)",fontsize=5.4*fs,labelpad=0.5)
    if title: a.set_title(title,fontsize=6.0*fs,loc="left",pad=2)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
    if legend:
        h=[plt.Line2D([],[],color="#2a78d6",lw=1.3,label="original $O$"),
           plt.Line2D([],[],color="#e34948",lw=1.0,ls=(0,(2.6,1.6)),label="removed $R$"),
           plt.Line2D([],[],color="#8d8b84",lw=1.0,ls=(0,(2.6,1.6)),label="re-lit $N$")]
        a.legend(handles=h,frameon=False,fontsize=4.3*fs,loc="lower left",bbox_to_anchor=(-0.05,-0.02),
                 handlelength=1.0,handletextpad=0.3,labelspacing=0.14,borderpad=0.05)

def panel_readings(b,title="(b) the two readings",fs=1.0,compact=False,schematic=False):
    EPS=4e-3; FI=[]
    if schematic:   # 框架图：只画判据几何，不放实测点（实测见 fig_cases e）
        b.fill_between([EPS,0.5],[0.5,0.5],[40,40],color="#e8f0e8",lw=0,zorder=0)
        b.fill_between([0.5,40],[0.5,40],[40,40],color="#e8f0e8",lw=0,zorder=0)
        b.plot([EPS,40],[EPS,40],color="#6b6a62",ls=(0,(3,2)),lw=0.7,zorder=2)
        b.axhline(0.5,color="#9a998f",ls=(0,(1.6,1.6)),lw=0.7,zorder=2)
        b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(EPS*0.9,55); b.set_ylim(EPS*0.9,30)
        b.set_xlabel("$I$ from re-lighting (m)",fontsize=5.4*fs,labelpad=0.5)
        b.set_ylabel("$F$ from the pedestrian (m)",fontsize=5.4*fs,labelpad=1)
        b.set_xticks([0.01,0.1,1,10]); b.set_yticks([0.01,0.1,1,10]); b.tick_params(labelsize=5.0*fs,length=1.8)
        if title: b.set_title(title,fontsize=6.0*fs,loc="left",pad=2)
        b.annotate("$F=I$",(0.02,0.02),textcoords="offset points",xytext=(-1,3.5),ha="right",va="bottom",fontsize=4.8*fs,color="#6b6a62",rotation=45)
        b.text(0.06,0.93,"avoidance:\n$F\\geq0.5$ m, $F>I$",transform=b.transAxes,fontsize=4.8*fs,color="#3f6b45",va="top",linespacing=1.2)
        b.text(0.97,0.12,"jitter: $F\\leq I$",transform=b.transAxes,fontsize=4.8*fs,color="#6b6a62",va="bottom",ha="right")
        for m,d in AX.items():        # 每家一个点：F、I 的场景均值（夜间 I），不画逐场景云
            F=[];IN=[]
            for t,v in d.items():
                if not all(k in v for k in ("clean","rm","night")): continue
                F.append(disp(v["clean"],v["rm"])); IN.append(disp(v["clean"],v["night"]))
            fm,inm=np.mean(F),np.mean(IN)
            b.plot(inm,fm,marker="o",ms=4.6*fs,mfc="white",mec=COL[m],mew=1.1,zorder=5)
            off={"dd":(-5,0,"right"),"autovla":(4.5,-3.8,"left"),"ltf":(4.5,-3.0,"left")}.get(m,(4.5,0,"left"))
            b.annotate(SH[m],(inm,fm),textcoords="offset points",xytext=off[:2],ha=off[2],va="center",fontsize=4.8*fs,color=COL[m])
        b.text(0.97,0.55,"one point\nper policy",transform=b.transAxes,fontsize=4.4*fs,color="#6b6a62",va="bottom",ha="right",linespacing=1.2)
        for s_ in ("top","right"): b.spines[s_].set_visible(False)
        return
    for m,d in AX.items():
        for t,v in d.items():
            if all(k in v for k in ("clean","rm","night")):
                FI.append((max(disp(v["clean"],v["rm"]),EPS),max(disp(v["clean"],v["night"]),EPS),COL[m]))
    Fs=np.array([x[0] for x in FI]); Is=np.array([x[1] for x in FI]); Cs=[x[2] for x in FI]
    b.fill_between([EPS,0.5],[0.5,0.5],[40,40],color="#e8f0e8",lw=0,zorder=0)     # F≥0.5 且 F>I
    b.fill_between([0.5,40],[0.5,40],[40,40],color="#e8f0e8",lw=0,zorder=0)
    b.plot([EPS,40],[EPS,40],color="#6b6a62",ls=(0,(3,2)),lw=0.7,zorder=2)
    b.axhline(0.5,color="#9a998f",ls=(0,(1.6,1.6)),lw=0.7,zorder=2)
    b.scatter(Is,Fs,s=1.6*fs,c=Cs,alpha=0.22,lw=0,zorder=1)
    for m,d in AX.items():
        F=[];IN=[];ID=[]
        for t,v in d.items():
            if not all(k in v for k in ("clean","rm","night")): continue
            if HAS4 and "dusk" not in v: continue
            F.append(disp(v["clean"],v["rm"])); IN.append(disp(v["clean"],v["night"]))
            if HAS4: ID.append(disp(v["clean"],v["dusk"]))
        fm,inm=np.mean(F),np.mean(IN); idm=np.mean(ID) if HAS4 else inm
        if HAS4:
            b.annotate("",(idm,fm),(inm,fm),zorder=4,          # 同一条水平线：F 不变，只有 I 换了
                       arrowprops=dict(arrowstyle="-|>",lw=0.7,color=COL[m],shrinkA=2.2,shrinkB=2.2,mutation_scale=4))
            b.plot(idm,fm,marker="s",ms=3.6*fs,mfc="white",mec=COL[m],mew=0.9,zorder=5)
        b.plot(inm,fm,marker="o",ms=4.2*fs,color=COL[m],mec="white",mew=0.5,zorder=5)
        b.annotate(SH[m],(max(idm,inm),fm),textcoords="offset points",xytext=LOFF.get(m,(4.5,0)),ha="left",
                   va="center",fontsize=4.8*fs,color=COL[m])
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(EPS*0.9,55); b.set_ylim(EPS*0.9,30)
    b.set_xlabel("$I$ from re-lighting (m)" if compact else "$I$: displacement from re-lighting (m)",
                 fontsize=5.4*fs,labelpad=0.5)
    b.set_ylabel("$F$ from the pedestrian (m)" if compact else "$F$: displacement from the pedestrian (m)",
                 fontsize=5.4*fs,labelpad=1)
    b.tick_params(labelsize=5.0*fs,length=1.8)
    if title: b.set_title(title,fontsize=6.0*fs,loc="left",pad=2)
    b.annotate("$F=I$",(0.02,0.02),textcoords="offset points",xytext=(-1,3.5),ha="right",va="bottom",
               fontsize=4.8*fs,color="#6b6a62",rotation=45)
    b.text(0.055,0.93,"$F\\geq0.5$ m and $F>I$",transform=b.transAxes,fontsize=4.8*fs,color="#3f6b45",va="top")
    if HAS4:
        b.plot([],[],marker="s",ms=3.2*fs,mfc="white",mec="#52514e",mew=0.8,ls="none",label="$I$ from dusk")
        b.plot([],[],marker="o",ms=3.6*fs,color="#52514e",ls="none",label="$I$ from night")
        b.legend(frameon=False,fontsize=4.6*fs,loc="upper left",bbox_to_anchor=(0.02,0.84),handletextpad=0.25,
                 labelspacing=0.16,borderpad=0.05)
    b.text(0.98,0.16,"one $F$ per policy;\nthe arrow changes only $I$" if HAS4 else "one $F$ and one $I$\nper policy",
           transform=b.transAxes,fontsize=4.4*fs,color="#6b6a62",va="top",ha="right",linespacing=1.25)
    for s_ in ("top","right"): b.spines[s_].set_visible(False)


# ---------------- 框架图专用：示意风格（无刻度、无数字，只表达 F、I 的含义与判据几何） ----------------
def frame_extract(a,fs=1.0):
    """示意：三条理想化的规划（同一场景问三次），F、I 分别是 O–R、O–N 的距离，不用真实数据。"""
    t=np.linspace(0,1,60)
    O=np.stack([ 1.6*t**2.2, 15*t],1)                     # 看见行人：向右让开
    R=np.stack([ 0.0*t,       16*t],1)                     # 抹掉行人：直行，穿过行人路径
    N=np.stack([-1.3*t**1.5,  15.5*t],1)                   # 夜化：无缘由地偏了一点
    ped=np.array([[-2.6,11.5],[0.6,13.4]])                 # 行人从左侧走向走廊
    a.axvspan(-1,1,color="#eef1f6",lw=0,zorder=0)
    a.plot(ped[:,0],ped[:,1],color="#b3412c",ls=(0,(1.3,1.3)),lw=1.0,zorder=5)
    a.plot(ped[0,0],ped[0,1],marker="*",ms=7*fs,color="#b3412c",mec="white",mew=0.4,zorder=6)
    for X,c,lw,ls,z in ((N,"#8d8b84",1.1,(0,(2.6,1.6)),3),(R,"#e34948",1.1,(0,(2.6,1.6)),3),(O,"#2a78d6",1.6,"-",4)):
        a.plot(X[:,0],X[:,1],color=c,lw=lw,ls=ls,zorder=z)
    a.plot(0,0,marker="^",ms=6*fs,color="#0b0b0b",zorder=6)
    for X,c,lab,i,off in ((R,"#e34948","$F$",-1,(0.0,-1.3)),(N,"#8d8b84","$I$",30,(-1.1,0.9))):   # F 在末端，I 在半程：都是同时刻两条规划的距离
        p0=tuple(O[i]); p1=tuple(X[i])
        a.annotate("",p1,p0,arrowprops=dict(arrowstyle="<->",lw=0.9,color=c,shrinkA=0,shrinkB=0),zorder=7)
        a.annotate(lab,((p0[0]+p1[0])/2+off[0],(p0[1]+p1[1])/2+off[1]),ha="center",va="center",fontsize=6.6*fs,color=c,weight="bold",zorder=8)
    a.set_xlim(-4.5,4.5); a.set_ylim(-1.2,18.5); a.set_xticks([]); a.set_yticks([])
    for s_ in ("top","right","left","bottom"): a.spines[s_].set_visible(False)
    h=[plt.Line2D([],[],color="#2a78d6",lw=1.6,label="plan on $O$"),
       plt.Line2D([],[],color="#e34948",lw=1.1,ls=(0,(2.6,1.6)),label="plan on $R$"),
       plt.Line2D([],[],color="#8d8b84",lw=1.1,ls=(0,(2.6,1.6)),label="plan on $N$"),
       plt.Line2D([],[],color="#b3412c",lw=1.0,ls=(0,(1.3,1.3)),marker="*",ms=5,label="pedestrian")]
    a.legend(handles=h,frameon=False,fontsize=4.6*fs,loc="lower right",bbox_to_anchor=(1.04,-0.02),
             handlelength=1.0,handletextpad=0.3,labelspacing=0.14,borderpad=0.05)

def frame_readings(b,fs=1.0):
    """F–I 平面示意：箭头坐标轴、F=I 对角线、避让区/抖动区，六家各一个点（对数坐标下的均值位置，无刻度）。"""
    EPS=4e-3
    b.fill_between([EPS,0.5],[0.5,0.5],[40,40],color="#e8f0e8",lw=0,zorder=0)
    b.fill_between([0.5,40],[0.5,40],[40,40],color="#e8f0e8",lw=0,zorder=0)
    b.plot([EPS,40],[EPS,40],color="#6b6a62",ls=(0,(3,2)),lw=0.8,zorder=2)
    b.axhline(0.5,color="#9a998f",ls=(0,(1.6,1.6)),lw=0.7,zorder=2)
    for m,d in AX.items():
        F=[];IN=[]
        for t,v in d.items():
            if not all(k in v for k in ("clean","rm","night")): continue
            F.append(disp(v["clean"],v["rm"])); IN.append(disp(v["clean"],v["night"]))
        b.plot(np.mean(IN),np.mean(F),marker="o",ms=4.6*fs,color=COL[m],mec="white",mew=0.6,zorder=5)
    b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(EPS*0.9,55); b.set_ylim(EPS*0.9,30)
    b.set_xticks([]); b.set_yticks([]); b.minorticks_off()
    for s_ in ("top","right","left","bottom"): b.spines[s_].set_visible(False)
    # 箭头坐标轴
    b.annotate("",(55,EPS*0.9),(EPS*0.9,EPS*0.9),arrowprops=dict(arrowstyle="-|>",lw=0.7,color="#52514e",shrinkA=0,shrinkB=0),zorder=3)
    b.annotate("",(EPS*0.9,30),(EPS*0.9,EPS*0.9),arrowprops=dict(arrowstyle="-|>",lw=0.7,color="#52514e",shrinkA=0,shrinkB=0),zorder=3)
    b.text(0.5,-0.05,"$I$ (re-lighting)",transform=b.transAxes,ha="center",va="top",fontsize=5.8*fs,color="#52514e")
    b.text(-0.04,0.5,"$F$ (pedestrian)",transform=b.transAxes,ha="right",va="center",rotation=90,fontsize=5.8*fs,color="#52514e")
    b.annotate("$F=I$",(0.02,0.02),textcoords="offset points",xytext=(-1,4),ha="right",va="bottom",fontsize=5.0*fs,color="#6b6a62",rotation=45)
    b.text(0.05,0.94,"avoidance\n$F\\geq0.5$ m, $F>I$",transform=b.transAxes,fontsize=5.0*fs,color="#3f6b45",va="top",linespacing=1.2)
    b.text(0.97,0.10,"jitter: $F\\leq I$",transform=b.transAxes,fontsize=5.0*fs,color="#6b6a62",va="bottom",ha="right")
    b.text(0.97,0.50,"one point\nper policy",transform=b.transAxes,fontsize=4.6*fs,color="#6b6a62",va="bottom",ha="right",linespacing=1.2)
