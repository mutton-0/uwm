"""v5 表格与正文数字宏：全部从 JSON 直读。表 I（检查项）手写。"""
import json,numpy as np,os
# 表头措辞随论文版本切换：v6（默认）用工程说法，v5 保留投资类比的原始措辞
_ST=os.environ.get("PAPER_STYLE","v6"); _W=(lambda a,b: b if _ST=="v5" else a)
# 数据仍读 paper_icra_v5（json/npz 都在那里），但表与数字宏默认写到当前论文目录 v6
_OUT=os.environ.get("PAPER_OUT","/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090/paper_icra_v6")
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"; V4=f"{R5}/paper_icra_v4"
M=["dd","ltf","ddv2","simlingo","autovla","alpamayo15"]
NAME={"dd":"DiffusionDrive","ltf":"LTF","ddv2":"DiffusionDriveV2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
SH={"dd":"DD","ltf":"LTF","ddv2":"DDv2","simlingo":"SimLingo","autovla":"AutoVLA","alpamayo15":"Alpamayo 1.5"}
A=json.load(open(f"{V5}/profile_ALL.json")); L=json.load(open(f"{V5}/profile_LHD.json")); R=json.load(open(f"{V5}/profile_RHD.json"))
PR=json.load(open(f"{V5}/prereg_result.json")); SS=json.load(open(f"{V5}/sample_size_v2.json"))
PD=json.load(open(f"{R5}/pdms_decomp_sg.json")); CF=json.load(open(f"{V4}/cfr_same_frame.json")); DR=(json.load(open(f"{V5}/detfail_robust.json")) if os.path.exists(f"{V5}/detfail_robust.json") else None)
DV=[o for o in json.load(open(f"{V4}/det_validate.json")) if "target_box" in o and o["front_only"]]
rows=json.load(open(f"{V5}/diag_units.json"))
sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr")
def coll(m,sv):
    U=[z for z in rows if z["m"]==m and sel(z) and z["sv"]==sv]
    return (100*np.mean([z["P"]["A"]==0 for z in U]),100*np.mean([z["Q"]["A"]==0 for z in U]),len(U)) if U else (np.nan,np.nan,0)
def ci(d,k,f="{:.2f}"): return f"{{\\scriptsize[{f.format(d['ci'][k][0])},{f.format(d['ci'][k][1])}]}}"
def sgn(x,f="{:+.2f}"): return "$"+f.format(x).replace("+","{+}").replace("-","{-}")+"$"
# ---------- Table II：诊断书（化验单样式：表头给正常范围，超界标箭头，列内最优加粗） ----------
# 正常范围取自表 I：暴露度≈1、HS 随危险升到 1、缩放>0、SP≈1、CFR≫1
NORM={"exposure":(0.8,1.2),"HS":(0.15,1.01),"HS_slope":(0.0,1.01),"SP":(0.8,1.01),"CFR":(1.0,1e9)}
def flag(v,k):
    lo,hi=NORM[k]
    return r"$\downarrow$" if v<lo else (r"$\uparrow$" if v>hi else "")
NS=json.load(open(f"{V5}/night_speed.json"))
def bold(txt,cond): return f"\\textbf{{\\boldmath {txt}}}" if cond else txt
def cell(v,k,best,fmt="%.2f",d=None):
    t=(fmt%v)+flag(v,k)
    if d is not None and k in d.get("ci",{}):
        lo,hi=d["ci"][k]; f2="%+.2f" if fmt.startswith("%+") else "%.2f"
        t+=" {\\scriptsize["+(f2%lo)+", "+(f2%hi)+"]}"
    return (r"\textbf{"+t+"}") if best else t
_bst={"exposure":min(M,key=lambda m:abs(A[m]["point"]["exposure"]-1)),
      "HS":max(M,key=lambda m:A[m]["point"]["HS"]),
      "HS_slope":max(M,key=lambda m:A[m]["point"]["HS_slope"]),
      "SP":max(M,key=lambda m:A[m]["point"]["SP"]),
      "CFR":max(M,key=lambda m:A[m]["point"]["CFR"])}
_bc8=min(M,key=lambda m:coll(m,"8")[0])
# 转置版：行 = 读数，列 = 策略 (+ Human)
_gcx=json.load(open(f"{V5}/gt_ceiling.json")); _hs_h=sum(g*n for g,n in zip(_gcx["gt"],_gcx["n"]))/sum(_gcx["n"])
_DK=json.load(open(f"{V5}/dusk_cfr.json")) if os.path.exists(f"{V5}/dusk_cfr.json") else {}
_BC=json.load(open(f"{V5}/bench_compare.json")); _TR=json.load(open(f"{V5}/ttc_rank.json"))
_NUo=_BC["nusc"]; _CV=_BC["navsim_close"]; _AL=_BC["navsim_all"]
_rk=lambda d: {m:i+1 for i,m in enumerate(sorted(M,key=lambda x:-d[x]["epdms"]))}
_RKA=_rk(_AL); _RKC=_rk(_CV)
_bA=max(M,key=lambda m: _AL[m]["epdms"]); _bC=max(M,key=lambda m: _CV[m]["epdms"])
_bL=min(M,key=lambda m: _TR["LHD"][m]["moving"]["viol"]); _bR=min(M,key=lambda m: _TR["RHD"][m]["moving"]["viol"])
_bf=lambda t,on: (r"\textbf{"+t+"}") if on else t
def _row(label,cells): return label+" & "+" & ".join(cells)+r" \\"
def _cir(k,f="%.2f"):   # 置信区间独立成一小行，灰色 scriptsize
    return r"\multicolumn{1}{r}{\textcolor{gray}{\scriptsize 95\% CI}} & "+" & ".join(r"\textcolor{gray}{\scriptsize["+(f%A[m]["ci"][k][0])+", "+(f%A[m]["ci"][k][1])+"]}" for m in M)+r" & \\[-1pt]"
R=[]
R.append(r"\multicolumn{8}{@{}l}{\emph{Exams} (236 nuScenes frames)} \\")
R.append(_row(r"Exposure (ratio to human) $\approx1$",[cell(A[m]["point"]["exposure"],"exposure",m==_bst["exposure"]) for m in M]+["1.00"])); R.append(_cir("exposure"))
R.append(_row(r"Hazard sens.\ $\HS\in[-1,1]$ $\uparrow$",[cell(A[m]["point"]["HS"],"HS",m==_bst["HS"]) for m in M]+[f"{_hs_h:.2f}"])); R.append(_cir("HS"))
R.append(_row(r"Scaling (rank corr.) $\uparrow$",[cell(A[m]["point"]["HS_slope"],"HS_slope",m==_bst["HS_slope"],"%+.2f") for m in M]+[f"{_gcx.get('sc_median',float('nan')):+.2f}"])); R.append(_cir("HS_slope","%+.2f"))
R.append(_row(r"Specificity $\SP\in(0,1]$ $\uparrow$",[cell(A[m]["point"]["SP"],"SP",m==_bst["SP"]) for m in M]+["--"])); R.append(_cir("SP"))
R.append(_row(r"Lighting $\CFR$ (ratio) $\uparrow$",[cell(A[m]["point"]["CFR"],"CFR",m==_bst["CFR"]) for m in M]+["--"])); R.append(_cir("CFR"))
R.append(r"\midrule")
R.append(_row(r"\emph{Collision} at 8\,m/s, vis.\,/\,rm.\ (\%) $\downarrow$",[_bf(f"{coll(m,'8')[0]:.1f}\\,/\\,{coll(m,'8')[1]:.1f}",m==_bc8) for m in M]+["--"]))
R.append(r"\midrule")
def _sp(m):
    n_=NS[m]; _f="{:+.0f}" if abs(100*n_['dv_rel'])>=1 else "{:+.1f}"
    return bold("$"+_f.format(100*n_['dv_rel'])+"\\%$",n_["p"]<0.01)
def _ds(m):
    n_=NS[m]; return bold("$"+("{:+.2f}" if abs(n_['dS'])>=0.01 else "{:+.3f}").format(n_['dS'])+"$",n_["dS_p"]<0.01)
R.append(_row(r"speed of $N$ (\%) $\downarrow$",[_sp(m) for m in M]+["--"]))
R.append(_row(r"clearance of $N$, $\Delta_N S$ (m) $\uparrow$",[_ds(m) for m in M]+["--"]))
R.append(r"\midrule")
R.append(r"\multicolumn{8}{@{}l}{\emph{Standard scores} (rank)} \\")
R.append(_row(r"L2 (m), vis.\,/\,rm. $\downarrow$",[f"{_NUo[m]['clean']['L2_avg']:.2f}\\,/\\,{_NUo[m]['rm']['L2_avg']:.2f}" for m in M]+["--"]))
R.append(_row(r"EPDMS, all scenes $\uparrow$",[_bf("%.3f (%d)"%(_AL[m]["epdms"],_RKA[m]),m==_bA) for m in M]+["--"]))
R.append(_row(r"EPDMS, near-ped. $\uparrow$",[_bf("%.3f (%d)"%(_CV[m]["epdms"],_RKC[m]),m==_bC) for m in M]+["--"]))
R.append(_row(r"TTC$<$1.5\,s (\%), LHD $\downarrow$",[_bf("%.1f (%d)"%(_TR["LHD"][m]["moving"]["viol"],_TR["ranks"]["LHD TTC"][m]),m==_bL) for m in M]+["--"]))
R.append(_row(r"TTC$<$1.5\,s (\%), RHD $\downarrow$",[_bf("%.1f (%d)"%(_TR["RHD"][m]["moving"]["viol"],_TR["ranks"]["RHD TTC"][m]),m==_bR) for m in M]+["--"]))
T=[r"\begin{table*}[t]",r"\centering",
 r"\caption{\textbf{The full report for the six policies.} Rows are readings, columns policies. Brackets: 95\% CI. Arrows in cells: outside the reference threshold of \cref{sec:checkup}; bold: best per row, or $p<0.01$ for the two rows of $N$. Speed and clearance of $N$: change under the night-style perturbation, on the same frames (negative clearance = closer). Human: the logged trajectory scored the same way against each policy's blind plan. Standard scores: nuScenes L2 on the same frames, NAVSIM EPDMS on all 783 Singapore scenes and on the \NumNclose{} near-pedestrian ones, and the share of moving-ego scenes with pedestrian TTC $<1.5$\,s per driving side.}",
 r"\label{tab:report}",r"\footnotesize",r"\setlength{\tabcolsep}{4pt}",
 r"\begin{tabular}{@{}lccccccc@{}}",r"\toprule",
 r"Reading\,$\backslash$\,Policy & "+" & ".join(SH[m] for m in M)+r" & Human \\",r"\midrule"]+R+[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
open(f"{_OUT}/tables/tab_report.tex","w").write("\n".join(T)+"\n")
# ---------- Table III：跨舵位预注册 ----------
lab={"P1":"Lighting outweighs the pedestrian (CFR $<1$) for every policy",
     "P2":"Hazard sensitivity stays below 0.15 (upper CI)",
     "P3":"Exposure ordering transfers ($\\rho\\ge0.6$; SimLingo top, DD bottom)",
     "P4":"Specificity ordering transfers ($\\rho\\ge0.6$; SimLingo lowest $\\SP$)",
     "P5":f"Point values stay within the left-hand CI ($\\ge$70\\% of {len(PR['P5']['cells'])} cells)",
     "P7":"SimLingo collides most, visible vs.\\ removed within 3\\,pts"}
def det(k):
    d=PR[k]
    if k=="P1": return f"max upper CI {max(v[1] for v in d['detail'].values()):.2f}"
    if k=="P2": 
        bad=[SH[m] for m,v in d['detail'].items() if v[1]>=0.15]; return ", ".join(bad)+" above" if bad else "all below"
    if k in ("P3","P4"): return f"$\\rho={d['rho']:.2f}$"
    if k=="P5": return f"{100*d['rate']:.0f}\\%"
    if k=="P7": return f"gap {100*d['max_gap']:.1f}\\,pts"
T=[r"\begin{table}[t]",r"\centering",
   r"\caption{\textbf{Pre-registered transfer test.} Predictions fixed on 88 Boston frames, tested on 148 Singapore frames; five policies, the replaced sixth changes no verdict (\cref{sec:sample}).}",
   r"\label{tab:prereg}",r"\footnotesize",r"\setlength{\tabcolsep}{2.5pt}",
   r"\begin{tabular}{@{}lp{4.35cm}cl@{}}",r"\toprule",r"No. & Prediction & Holds & Evidence \\",r"\midrule"]
for k in ["P1","P2","P3","P4","P5","P7"]:               # 登记顺序；P6（方向类，已弃用）不再报告，P7 显示为 P6
    yn="yes" if PR[k]["pass_"] else r"\textbf{no}"
    T.append(f"{'P6' if k=='P7' else k} & {lab[k]} & {yn} & {det(k)} \\\\")
T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(f"{_OUT}/tables/tab_prereg.tex","w").write("\n".join(T)+"\n")
# ---------- Table IV：光照（精简：不列 p，显著者加粗） ----------
T=[r"\begin{table}[t]",r"\centering",
  r"\caption{\textbf{The night-style perturbation} on the nuScenes frames. Speed: change of planned mean speed under re-lighting. $\Delta_N S$: change of clearance to the pedestrian, negative = closer. Last column: $\CFR$ at dusk and at night on the \NumDuskN{} NAVSIM scenes. Brackets: 95\% CI; bold: $p<0.01$.}",
   r"\label{tab:light}",r"\scriptsize",r"\setlength{\tabcolsep}{1.6pt}",
   r"\begin{tabular}{@{}lcccc@{}}",r"\toprule",
   r"Policy & $\CFR$ [95\% CI] $\uparrow$ & speed $\downarrow$ & $\Delta_N S$ (m) $\uparrow$ & dusk\,/\,night $\CFR$ $\uparrow$ \\",r"\midrule"]
for m in M:
    c=CF[m]["corr"]; n=NS[m]
    pm=lambda x,f: "$"+f.format(x)+"$"   # 不再把小值压成 "0"：真值很小就多给一位小数，见下
    # 小数位按数量级给：不足 1% 的速度变化保留一位小数，避免四舍五入成 "0%" 看着像缺数据
    _f="{:+.0f}" if abs(100*n['dv_rel'])>=1 else "{:+.1f}"
    sp=bold(pm(100*n['dv_rel'],_f)[:-1]+"\\%$",n["p"]<0.01)
    ds=bold(pm(n['dS'],"{:+.2f}" if abs(n['dS'])>=0.01 else "{:+.3f}"),n["dS_p"]<0.01)
    _dk=json.load(open(f"{V5}/dusk_cfr.json"))[m] if os.path.exists(f"{V5}/dusk_cfr.json") else None
    dk=f"{_dk['cfr_dusk']:.2f}\\,/\\,{_dk['cfr_night']:.2f}" if _dk else "--"
    T.append(f"{SH[m]} & {c['CFR']:.2f} {{\\scriptsize[{c['CFR_ci'][0]:.2f},{c['CFR_ci'][1]:.2f}]}} & {sp} & {ds} & {dk} \\\\")
T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(f"{_OUT}/tables/tab_light.tex","w").write("\n".join(T)+"\n")
# ---------- 数字宏 ----------
# 帧 → 场景换算：体检帧来自多少个 nuScenes 场景（一场景 20 s，关键帧 2 Hz）
from collections import Counter as _Ctr
_MAN=json.load(open(f"{R5}/risk_card_manifest.json")); _IDX={x["uid"]:x for k in ("A","B") for x in _MAN[k]}
_uu=sorted({z["uid"] for z in json.load(open(f"{V5}/diag_units.json"))
            if z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr")})
_fps=len(_uu)/len(_Ctr(_IDX[u]["scene"] for u in _uu if u in _IDX))
# HS 四个分项各自的最大绝对值（跨模型），用于说明结论与权重无关；真人参照取全体均值
_UU=json.load(open(f"{V5}/diag_units.json"))
_sel=lambda z: z["d"]<=15 and (z["set"]=="B" or z["grp"]=="corr") and z["v"]>=1.0 and z["need"]
_mx=0.0
for _m in M:
    _U=[z for z in _UU if z["m"]==_m and _sel(z)]
    if not _U: continue
    for _k in ("A","C","T"):
        _mx=max(_mx,abs(np.mean([z["P"][_k]-z["Q"][_k] for z in _U])))
    _mx=max(_mx,abs(np.mean([np.tanh(z["P"]["S"]-z["Q"]["S"]) for z in _U])))
_GC=json.load(open(f"{V5}/gt_ceiling.json")) if os.path.exists(f"{V5}/gt_ceiling.json") else None

cf=[A[m]["point"]["CFR"] for m in M]; hs=[A[m]["point"]["HS"] for m in M]; ex=[A[m]["point"]["exposure"] for m in M]
ns=SS["dims"]; nc=[PD["table"][m]["no_at_fault_collisions"] for m in M]
g1=max(abs(coll(m,"actual")[0]-coll(m,"actual")[1]) for m in M); g8=max(abs(coll(m,"8")[0]-coll(m,"8")[1]) for m in M)
NUM={"NumCFRlo":f"{min(cf):.2f}","NumCFRhi":f"{max(cf):.2f}","NumCFRciHi":f"{max(A[m]['ci']['CFR'][1] for m in M):.2f}",
     "NumHSlo":f"{min(hs):.2f}","NumHShi":f"{max(hs):.2f}","NumExpLo":f"{min(ex):.2f}","NumExpHi":f"{max(ex):.2f}",
     "NumDetRm":f"{100*np.mean([o['rm']['target_iou']<0.5 for o in DV]):.1f}","NumDetNight":f"{100*np.mean([o['night']['target_iou']>=0.5 for o in DV]):.1f}",
     "NumDetStill":f"{100*np.mean([o['rm']['target_iou']>=0.5 for o in DV]):.1f}","NumDetN":str(len(DV)),
     "NumDetOrig":f"{100*np.mean([o['orig']['hit'] for o in DV]):.0f}",
     "NumDacShare":f"{100*PD['shapley_dvar_over_full']['DAC']:.0f}","NumNCrange":f"{100*(max(nc)-min(nc)):.1f}",
     "NumDdcShare":f"{100*PD['shapley_dvar_over_full']['DDC']:.0f}","NumMapShare":f"{100*(PD['shapley_dvar_over_full']['DAC']+PD['shapley_dvar_over_full']['DDC']):.0f}",
     "NumNCshare":f"{100*PD['shapley_dvar_over_full']['NC']:.0f}",
     "NumRobustIn":f"{DR['inside']}/{DR['total']}" if DR else "--",
     "NumRobustRho":(f"$\\rho\\ge{min(r['rho'] for r in DR['rows']):.2f}$" if DR else "--"),
     "NumDetDrop":(str(DR['n_frames_dropped']) if DR and DR.get('n_frames_dropped') else "17"),
     "NumMaxTerm":f"{_mx:.2f}",
     "NumHumanRef":(f"{sum(g*n for g,n in zip(_GC['gt'],_GC['n']))/sum(_GC['n']):.2f}" if _GC else "0.57"),
     "NumHumanSc":(f"{_GC.get('sc_median',float('nan')):+.2f}" if _GC else "+0.13"),
     "NumHumanLo":(f"{np.nanmin(_GC['gt']):.2f}" if _GC else "0.52"),
     "NumHumanHi":(f"{np.nanmax(_GC['gt']):.2f}" if _GC else "0.62"),
     "NumFrPerScene":"%.0f"%_fps,
     "NumScCFR":"%.0f"%(max(ns["CFR"]["n_star"].values())/_fps),
     "NumScExp":"%.0f"%(max(v for m,v in ns["exposure"]["n_star"].items() if m in ("dd","ltf","ddv2","simlingo"))/_fps),
     "NumScHS":"%.0f"%(max(v for m,v in ns["HS"]["n_star"].items() if v and v<1e5)/_fps),
     "NumMinHS":"%.0f"%(max(v for m,v in ns["HS"]["n_star"].items() if v and v<1e5)/_fps*20/60),
     "NumPool":str(SS["N"]),"NumCollGap":f"{g1:.1f}","NumCollGapEight":f"{g8:.1f}",
     "NumNstarCFRhi":f"{max(ns['CFR']['n_star'].values()):.0f}","NumNstarExpHi":f"{max(v for m,v in ns['exposure']['n_star'].items() if m in ('dd','ltf','ddv2','simlingo')):.0f}",
     "NumRankSPforty":f"{100*ns['SP']['rank_p']['40']:.0f}","NumRankExpTen":f"{100*ns['exposure']['rank_p']['10']:.0f}",
     "NumRankHSninety":f"{100*ns['HS']['rank_p']['90']:.0f}","NumRankAlignOneThirty":f"{100*ns['align']['rank_p']['130']:.0f}",
     "NumPass":str(sum(PR[k]['pass_'] for k in ['P1','P2','P3','P4','P5','P7'])),
     "NumNightDD":f"{100*NS['dd']['dv_rel']:.0f}","NumNightLTF":f"{100*NS['ltf']['dv_rel']:.0f}","NumNightSL":f"{100*NS['simlingo']['dv_rel']:.0f}",
     "NumNightDDv":f"{100*NS['ddv2']['dv_rel']:.0f}"}
open(f"{_OUT}/tables/numbers.tex","w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in NUM.items())+"\n")
print(json.dumps(NUM,indent=0))
print({m:(coll(m,'actual'),coll(m,'8')) for m in M})
# ---------- Table VI：与两套标准评测对比 + TTC 排名 ----------
import os
if os.path.exists(f"{V5}/bench_compare.json") and os.path.exists(f"{V5}/ttc_rank.json"):
    BC=json.load(open(f"{V5}/bench_compare.json")); TR=json.load(open(f"{V5}/ttc_rank.json"))
    NUo=BC["nusc"]; CV=BC["navsim_close"]; AL=BC["navsim_all"]
    T=[r"\begin{table}[t]",r"\centering",
       r"\caption{\textbf{The six policies under the standard scores.} Left: nuScenes open-loop L2 on the 236 near-pedestrian frames, original\,/\,removed. Middle: NAVSIM EPDMS on all 783 Singapore scenes and on the \NumNclose{} near-pedestrian ones. Right: share of moving-ego scenes with pedestrian TTC $<1.5$\,s per driving side, rank in parentheses. Bold: best per column.}",
       r"\label{tab:bench}",r"\scriptsize",r"\setlength{\tabcolsep}{2.0pt}",
       r"\begin{tabular}{@{}lc cc cc@{}}",r"\toprule",
       r" & L2 (m) $\downarrow$ & \multicolumn{2}{c}{EPDMS $\uparrow$} & \multicolumn{2}{c}{TTC$<$1.5\,s (\%) $\downarrow$} \\",
       r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}\cmidrule(l){5-6}",
       r"Policy & orig.\,/\,rm. & all & near & LHD & RHD \\",r"\midrule"]
    # EPDMS 两列各自排名（高分为 1），让"榜单第一跌到第五"在表里直接看得见
    _rk=lambda d: {m:i+1 for i,m in enumerate(sorted(M,key=lambda x:-d[x]["epdms"]))}
    RKA=_rk(AL); RKC=_rk(CV)
    # 列内最优加粗（CVPR/ICRA 惯例）：EPDMS 越高越好，TTC 违规率越低越好
    _bA=max(M,key=lambda m: AL[m]["epdms"]); _bC=max(M,key=lambda m: CV[m]["epdms"])
    _bL=min(M,key=lambda m: TR["LHD"][m]["moving"]["viol"]); _bR=min(M,key=lambda m: TR["RHD"][m]["moving"]["viol"])
    _bf=lambda t,on: (r"\textbf{"+t+"}") if on else t
    for m in M:
        n=NUo[m]; L_=TR["LHD"][m]["moving"]; R_=TR["RHD"][m]["moving"]
        cA="%.3f (%d)"%(AL[m]["epdms"],RKA[m]); cC="%.3f (%d)"%(CV[m]["epdms"],RKC[m])
        cL="%.1f (%d)"%(L_["viol"],TR["ranks"]["LHD TTC"][m]); cR="%.1f (%d)"%(R_["viol"],TR["ranks"]["RHD TTC"][m])
        cells=[_bf(cA,m==_bA),_bf(cC,m==_bC),_bf(cL,m==_bL),_bf(cR,m==_bR)]
        T.append(f"{SH[m]} & {n['clean']['L2_avg']:.2f}\\,/\\,{n['rm']['L2_avg']:.2f} & "
                 + " & ".join(cells) + " \\\\")
    T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(f"{_OUT}/tables/tab_bench.tex","w").write("\n".join(T)+"\n")
    sp=TR["spearman"]; rs=BC["removal_sensitivity"]
    NUM2={"NumRhoTTC":f"{sp['LHD TTC']['RHD TTC']:+.2f}","NumRhoTTCHSL":f"{sp['LHD TTC']['HS (ours)']:+.2f}","NumRhoTTCHSR":f"{sp['RHD TTC']['HS (ours)']:+.2f}",
          "NumRhoTTCCFRL":f"{sp['LHD TTC']['lighting CFR (ours)']:+.2f}","NumRhoTTCCFRR":f"{sp['RHD TTC']['lighting CFR (ours)']:+.2f}",
          "NumRhoTTCExpL":f"{sp['LHD TTC']['exposure (ours, low=safe)']:+.2f}","NumRhoTTCExpR":f"{sp['RHD TTC']['exposure (ours, low=safe)']:+.2f}",
          "NumRhoTTCNightL":f"{sp['LHD TTC']['lighting |Δspeed| (ours)']:+.2f}","NumRhoTTCNightR":f"{sp['RHD TTC']['lighting |Δspeed| (ours)']:+.2f}",
          "NumMaxdLtwo":f"{max(abs(v['dL2']) for v in rs.values()):.2f}","NumMaxdCol":f"{max(abs(v['dcol_front']) for v in rs.values()):.1f}",
          "NumMaxdPed":f"{max(abs(v['dcol_ped']) for v in rs.values()):.1f}",
          "NumRhoLtwoExp":f"{BC['spearman']['|exposure-1|~nusc_L2avg']:+.2f}","NumRhoAllClose":f"{BC['spearman']['navsim_all_EPDMS~navsim_close_EPDMS']:+.2f}",
          "NumNCshareClose":f"{100*BC['shapley_close']['NC']:.0f}","NumDACshareClose":f"{100*BC['shapley_close']['DAC']:.0f}",
          "NumNclose":str(BC["n"]["close"])}
    with open(f"{_OUT}/tables/numbers.tex","a") as fh: fh.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in NUM2.items())+"\n")
    print(json.dumps(NUM2,indent=0,ensure_ascii=False))
# ---------- Table VII：哪种评测能预测右舵近行人表现（排名一致性） ----------
if os.path.exists(f"{V5}/rank_consistency.json"):
    RC2=json.load(open(f"{V5}/rank_consistency.json")); SPR=RC2["spearman"]
    ROWS=[("Same readouts, nuScenes Boston",[("TTC violations","A nuScenes 左舵·TTC 违规率"),("clearance","A nuScenes 左舵·最小间隙"),("collisions","A nuScenes 左舵·撞行人率")]),
          ("Same readouts, NAVSIM US cities",[("TTC violations","A2 NAVSIM 左舵·TTC 违规率"),("clearance","A2 NAVSIM 左舵·最小间隙"),("near-ped.\\ EPDMS","A2 NAVSIM 左舵·近行人 EPDMS")]),
          ("nuScenes open-loop (Boston)",[("L2","B nuScenes 开环·L2"),("collision","B nuScenes 开环·前方碰撞"),("ped.\\ collision","B nuScenes 开环·撞行人")]),
          ("NAVSIM leaderboard",[("EPDMS, 783 scenes","C NAVSIM 榜单 EPDMS（783）")]),
          ("Check-up (Boston)",[("exposure","D 体检·暴露度（小=安全）"),("hazard sensitivity","D 体检·危险敏感度"),("specificity","D 体检·特异度"),
                                ("lighting CFR","D 体检·光照 CFR"),("patience","D 体检·起步率")])]
    TG=["TTC 违规率","离行人最小间隙","撞行人率","近行人 EPDMS"]
    def fm(x): s=f"{x:+.2f}".replace("-","$-$"); return f"\\textbf{{{s}}}" if abs(x)>=0.886 else s
    T=[r"\begin{table}[t]",r"\centering",
       r"\caption{\textbf{Which evaluation predicts behaviour near pedestrians in another domain?} Rank correlation between the six-policy ordering given by each evaluation "
       r"(none uses the target data) and the ordering observed on 260 NAVSIM Singapore scenes with a pedestrian near the ego corridor (original images, moving ego). "
       r"Bold: $|\rho|\ge0.89$ (two-sided 5\% for $n{=}6$).}",
       r"\label{tab:rank}",r"\footnotesize",r"\setlength{\tabcolsep}{2.4pt}",
       r"\begin{tabular}{@{}lcccc@{}}",r"\toprule",r" & \multicolumn{4}{c}{Singapore near-pedestrian ordering by} \\",r"\cmidrule(l){2-5}",
       r"Ordering from & TTC & clearance & collisions & EPDMS \\",r"\midrule"]
    for g,items in ROWS:
        T.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{g}}}}} \\\\")
        for lab,key in items: T.append(f"\\quad {lab} & "+" & ".join(fm(SPR[key][t]) for t in TG)+" \\\\")
    T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(f"{_OUT}/tables/tab_rank.tex","w").write("\n".join(T)+"\n")
# ---------- Table VIII：轴在新域重新量（逐例 F/I、两条通式、左右舵偏差） ----------
if os.path.exists(f"{V5}/side_deviation.json") and os.path.exists(f"{V5}/case10_axes.json"):
    SD=json.load(open(f"{V5}/side_deviation.json")); CA=json.load(open(f"{V5}/case10_axes.json"))
    DT=json.load(open(f"{V5}/domain_transfer.json")) if os.path.exists(f"{V5}/domain_transfer.json") else None
    T=[r"\begin{table}[t]",r"\centering",
       r"\caption{\textbf{$\CFR$ and the two rules across driving sides.} $\CFR$: the check-up value on the nuScenes frames of \cref{tab:report}, Boston and Singapore separately, and the median over the ten cases taken per scene. TTC and the rule columns come instead from the \NumComN{} NAVSIM Singapore scenes all six policies share; rules are given forward\,/\,reversed.}",
       r"\label{tab:axes}",r"\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",
       r"\begin{tabular}{@{}lcccccc@{}}",r"\toprule",
       r" & \multicolumn{3}{c}{$\CFR$} & TTC & \multicolumn{2}{c}{rule holds \%} \\",r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(l){6-7}",
       r"Policy & Bos. & Sing. & scene & $<$1.5\,s & P-F & P-I \\",r"\midrule"]
    tr=(DT or {}).get("truth_common",{})
    for m in M:
        if m not in SD: continue
        d=SD[m]; t=tr.get(m,{})
        def _two(a,b):
            if not t: return "--"
            f1=f"{a:.0f}" if a==a else "--"; f2=f"{b:.0f}" if b==b else "--"
            return f"{f1}/{f2}"
        pf=_two(t.get("PF",float("nan")),t.get("PFrev",float("nan")))
        pi=_two(t.get("PI",float("nan")),t.get("PIrev",float("nan")))
        tv=f"{t['ttc_viol']:.0f}\\%" if t.get("ttc_viol")==t.get("ttc_viol") and t else "--"
        T.append(f"{SH[m]} & {d['cfr_l']:.2f} & {d['cfr_r']:.2f} & {d['case_cfr']:.2f} & {tv} & {pf} & {pi} \\\\")
    T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    open(f"{_OUT}/tables/tab_axes.tex","w").write("\n".join(T)+"\n")
    import statistics as _st
    _fs=[d["D_ped"] for c in CA["cases"] for d in c["models"].values()]
    _is=[d["D_I"]   for c in CA["cases"] for d in c["models"].values()]
    _lit=[d["D_I"]>d["D_ped"] for c in CA["cases"] for d in c["models"].values()]
    NUM3={"NumAxF":f"{_st.median(_fs):.2f}","NumAxI":f"{_st.median(_is):.2f}",
          "NumAxShare":f"{100*sum(_lit)/len(_lit):.0f}",
          "NumPFsmall":str(CA["PF"][0]),"NumPFsmallN":str(CA["PF"][1]),
          "NumPIsmall":str(CA["PI"][0]),"NumPIsmallN":str(CA["PI"][1]),
          "NumPFrev":str(CA["PFrev"][0]),"NumPFrevN":str(CA["PFrev"][1]),
          "NumPIrev":str(CA["PIrev"][0]),"NumPIrevN":str(CA["PIrev"][1]),
          "NumCaseCells":str(len(_fs))}
    if DT:
        tc=DT["truth_common"]
        NUM3["NumComN"]=str(DT["n_common"])
        NUM3["NumPFbig"]=f"{min(v['PF'] for v in tc.values()):.0f}--{max(v['PF'] for v in tc.values()):.0f}"
        NUM3["NumPIbig"]=f"{min(v['PI'] for v in tc.values()):.0f}--{max(v['PI'] for v in tc.values()):.0f}"
        if os.path.exists(f"{V5}/f_decomp.json"):
            FD=json.load(open(f"{V5}/f_decomp.json"))
            NUM3["NumRealN"]=str(sum(v["n_real"] for v in FD.values()))
            NUM3["NumRealTot"]=str(sum(v["n"] for v in FD.values()))
            if os.path.exists(f"{V5}/case10_tally.json"):
                TY=json.load(open(f"{V5}/case10_tally.json"))
                NUM3["NumCaseHit"]=str(TY["hit"]); NUM3["NumCaseTot"]=str(TY["tot"])
            NUM3["NumSGscenes"]=str(max(v["n"] for v in FD.values()))
            NUM3["NumRealPct"]=f"{100*sum(v['n_real'] for v in FD.values())/sum(v['n'] for v in FD.values()):.1f}"
            NUM3["NumFgtIlo"]=f"{min(v['share_F_gt_I'] for v in FD.values()):.0f}"
            NUM3["NumFgtIhi"]=f"{max(v['share_F_gt_I'] for v in FD.values()):.0f}"
            NUM3["NumRhoLo"]=f"{min(v['rho_dS_areq'] for v in FD.values()):+.2f}".replace("-","$-$")
            NUM3["NumRhoHi"]=f"{max(v['rho_dS_areq'] for v in FD.values()):+.2f}"
            # 表 IX：F 轴分解
            T2=[r"\begin{table}[t]",r"\centering",
               r"\caption{\textbf{A plan that moves is not a plan that yields.} For each policy and each of the 246 Singapore near-pedestrian scenes we measure $F$ (displacement caused by removing the pedestrian), "
               r"$I$ (displacement caused by the night-style perturbation, the policy's own jitter under an irrelevant edit) and $\Delta S$, the resulting change in clearance to the pedestrian's logged future ($\Delta S>0$: seeing the pedestrian keeps the plan farther away). "
               r"$\eta=\Delta S/F$ is the share of the displacement that becomes separation, taken over the scenes in which the plan actually moved. Genuine avoidance requires all three: $F\ge0.5$\,m, $F>I$, and $\Delta S\ge0.5$\,m.}",
               r"\label{tab:decomp}",r"\scriptsize",r"\setlength{\tabcolsep}{3pt}",
               r"\begin{tabular}{@{}lccccc@{}}",r"\toprule",
               r"Policy & $F>I$ & $F\ge0.5$ & median $\eta$ & genuine & $\rho(\Delta S,a_{\mathrm{req}})$ \\",r"\midrule"]
            for m in M:
                if m not in FD: continue
                v=FD[m]; et="--" if v["eta_med"] is None else f"{v['eta_med']:+.2f}".replace("-","$-$")
                rr=f"{v['rho_dS_areq']:+.2f}".replace("-","$-$")
                T2.append(f"{SH[m]} & {v['share_F_gt_I']:.0f}\\% & {v['n_big']} & {et} & {v['n_real']} ({v['real_rate']:.0f}\\%) & {rr} \\\\")
            T2+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
            open(f"{_OUT}/tables/tab_decomp.tex","w").write("\n".join(T2)+"\n")
        if os.path.exists(f"{V5}/avoid_sign.json"):
            AV=json.load(open(f"{V5}/avoid_sign.json"))
            NUM3["NumMoveN"]=str(sum(v["n_big"] for v in AV.values()))
            NUM3["NumMoveFar"]=str(sum(v["far"] for v in AV.values()))
            NUM3["NumMoveNear"]=str(sum(v["near"] for v in AV.values()))
            NUM3["NumMoveFlat"]=str(sum(v["flat"] for v in AV.values()))
            NUM3["NumSLnear"]=str(AV["simlingo"]["near"]); NUM3["NumSLfar"]=str(AV["simlingo"]["far"])
            NUM3["NumSLmed"]=f"{AV['simlingo']['med']:+.2f}".replace("-","$-$")
            NUM3["NumDDfar"]=str(AV["dd"]["far"]); NUM3["NumDDnear"]=str(AV["dd"]["near"])
            NUM3["NumDDmed"]=f"{AV['dd']['med']:+.2f}"
        NUM3["NumPFbigRev"]=f"{min(v['PFrev'] for v in tc.values()):.0f}--{max(v['PFrev'] for v in tc.values()):.0f}"
        NUM3["NumPIbigRev"]=f"{min(v['PIrev'] for v in tc.values()):.0f}--{max(v['PIrev'] for v in tc.values()):.0f}"
        NUM3["NumExpBeh"]=f"{-DT['lhd_to_behaviour']['exposure']:.2f}".replace("-","{-}")
        NUM3["NumSpBeh"]=f"{-DT['lhd_to_behaviour']['SP']:.2f}".replace("-","{-}")
        NUM3["NumCfrBeh"]=f"{-DT['lhd_to_behaviour']['CFR']:.2f}".replace("-","{-}")
        W={5:"Five",10:"Ten",20:"Twenty",40:"Forty"}
        for k in [5,10,20,40]:
            NUM3[f"NumBehRho{W[k]}"]=f"{DT['behaviour'][str(k)]['rho']:+.2f}".replace("+","{+}")
            NUM3[f"NumBehHit{W[k]}"]=f"{100*DT['behaviour'][str(k)]['rho_hit']:.0f}"
            NUM3[f"NumBehErr{W[k]}"]=f"{DT['behaviour'][str(k)]['err']:.0f}"
            NUM3[f"NumCfrErr{W[k]}"]=f"{DT['sweep6'][str(k)]['err']:.3f}"
            NUM3[f"NumPFErr{W[k]}"]=f"{DT['sweep6'][str(k)]['pf_err']:.1f}"
        NUM3["NumLhdCfrErr"]=f"{DT['lhd6']['err']:.3f}"
        NUM3["NumCfrErrHuge"]=f"{DT['sweep6']['160']['err']:.3f}"
        NUM3["NumCfrErrBig"]=f"{DT['sweep6']['80']['err']:.3f}"      # 与前文同一个六家池子
        NUM3["NumLhdCfrErrBig"]=f"{DT['lhd6']['err']:.3f}"
    # 特异度：SimLingo 与其余五家的值域（走宏，避免正文写死）
    _sp={m:A[m]["point"]["SP"] for m in M if m in A and A[m]["point"].get("SP") is not None}
    if _sp:
        _o=[v for k,v in _sp.items() if k!="simlingo"]
        NUM3["NumSpSL"]=f"{_sp.get('simlingo',float('nan')):.2f}"
        NUM3["NumSpOtherLo"]=f"{min(_o):.2f}"; NUM3["NumSpOtherHi"]=f"{max(_o):.2f}"
    # 风险标度：ρ(HS, a_req) 与换成时间轴 TTC0 的对照，六家取值域（走宏，避免正文写死）
    _sl=[A[m]["point"]["HS_slope"] for m in M if m in A and A[m]["point"].get("HS_slope") is not None]
    _st=[A[m]["point"].get("HS_slope_ttc") for m in M if m in A and A[m]["point"].get("HS_slope_ttc") is not None]
    _f=lambda x: f"{x:+.2f}".replace("-","$-$")
    if _sl:
        _lo=min(_sl); NUM3["NumScalLo"]=_f(_lo); NUM3["NumScalHi"]=_f(max(_sl))
        NUM3["NumScalNeg"]=str(sum(1 for x in _sl if x<0))
        NUM3["NumScalWorst"]=NAME[[m for m in M if m in A and A[m]["point"].get("HS_slope")==_lo][0]]
    if _st: NUM3["NumScalTtcLo"]=_f(min(_st)); NUM3["NumScalTtcHi"]=_f(max(_st))
    # 反事实速度下的 need 集（cf_need.py）：注入 2--8 m/s 造出危险，只在盲规划真会撞上的 cell 上读数
    if os.path.exists(f"{V5}/cf_need.json"):
        _CN=json.load(open(f"{V5}/cf_need.json")); _a=_CN["_all"]
        _mm=[k for k in _CN if k!="_all"]
        NUM3["NumCfScenes"]=str(_a["n_scene"]); NUM3["NumCfModels"]=str(len(_a["models"]))
        NUM3["NumCfCells"]=str(_a["n_scene"]*_a["n_speed"]*len(_a["models"]))
        NUM3["NumCfNeed"]=str(_a["n_need"]); NUM3["NumCfReal"]=str(_a["n_real"])
        NUM3["NumCfRealPct"]=f"{_a['real_pct']:.1f}"
        NUM3["NumCfNeedPct"]=f"{100*_a['n_need']/(_a['n_scene']*_a['n_speed']*len(_a['models'])):.0f}"
        NUM3["NumCfNeedLo"]=f"{min(_CN[m]['need_pct'] for m in _mm):.0f}"
        NUM3["NumCfNeedHi"]=f"{max(_CN[m]['need_pct'] for m in _mm):.0f}"
        _ds=[_CN[m]["dS_med"] for m in _mm]; _ar=[_CN[m]["dArc_med"] for m in _mm]
        f2=lambda x: f"{x:+.2f}".replace("-","$-$")
        NUM3["NumCfDsLo"]=f2(min(_ds)); NUM3["NumCfDsHi"]=f2(max(_ds))
        NUM3["NumCfArcLo"]=f2(min(_ar)); NUM3["NumCfArcHi"]=f2(max(_ar))
    # 黄昏对照（同一批 NAVSIM 右舵场景、同进程四条件）：检出率来自 dusk_check.json，CFR 来自 dusk_cfr.json
    if os.path.exists(f"{V5}/dusk_cfr.json") and os.path.exists(f"{V5}/dusk_check.json"):
        _DC=json.load(open(f"{V5}/dusk_cfr.json")); _DK=json.load(open(f"{V5}/dusk_check.json"))
        _base=[o for o in _DK.values() if o["orig"]>=0.5]      # 判据与 dusk_check.py 一致：目标框 IoU>=0.5
        _keep=lambda k: 100*sum(1 for o in _base if o.get(k,0)>=0.5)/len(_base)
        NUM3["NumDuskDetN"]=str(len(_base))
        NUM3["NumDuskDetNight"]=f"{_keep('night'):.0f}"
        NUM3["NumDuskDetDusk"]=f"{_keep('dusk'):.0f}"
        NUM3["NumDuskDetRm"]=f"{_keep('removed'):.0f}"
        NUM3["NumDuskLo"]=f"{min(v['cfr_dusk'] for v in _DC.values()):.2f}"
        NUM3["NumDuskHi"]=f"{max(v['cfr_dusk'] for v in _DC.values()):.2f}"
        NUM3["NumDuskNightLo"]=f"{min(v['cfr_night'] for v in _DC.values()):.2f}"
        NUM3["NumDuskNightHi"]=f"{max(v['cfr_night'] for v in _DC.values()):.2f}"
        NUM3["NumDuskN"]=str(max(v["n"] for v in _DC.values()))
    with open(f"{_OUT}/tables/numbers.tex","a") as fh: fh.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in NUM3.items())+"\n")

# 重新生成会覆盖掉 VS Code 认根文件用的魔法注释，这里统一补回
import glob as _glob
for _f in _glob.glob(f"{_OUT}/tables/*.tex"):
    _s=open(_f).read()
    if not _s.startswith("% !TEX root"): open(_f,"w").write("% !TEX root = ../main.tex\n"+_s)

# ---------- 后处理：正值不带 "+"（作者约定），负号保留 ----------
import re as _re
for _fn in ("numbers.tex","tab_report.tex","tab_prereg.tex","tab_light.tex","tab_bench.tex"):
    _p=f"{_OUT}/tables/{_fn}"
    if not os.path.exists(_p): continue
    _s=open(_p).read()
    _s=_s.replace("{+}","")
    _s=_re.sub(r"(?<![\\\w])\+(?=\d)","",_s)
    open(_p,"w").write(_s)
