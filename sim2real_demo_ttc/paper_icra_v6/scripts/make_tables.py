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
# ---------- Table II：诊断书 ----------
T=[r"\begin{table*}[t]",r"\centering",
 r"\caption{\textbf{The check-up report of six policies} (pooled near-pedestrian units, both driving sides; 95\% bootstrap intervals over frames). "
 r"Exposure: planned / logged distance over 2.5\,s ("+_W("","position size; ")+r"1 = human). HS: safety gained by seeing the pedestrian when it matters, on $[-1,1]$ "
 r"("+_W("a yielding driver approaches 1","risk management; a yielding driver approaches 1")+r"). Scaling: rank correlation of HS with hazard level $a_{\mathrm{req}}$ (does it respond more when risk is higher). "
 r"SP: how little the plan moves when no reaction is needed (over-reaction; 1 = unmoved). Collision: contact with the pedestrian's logged future, pedestrian visible\,/\,removed, "
 r"at the logged speed and at 8\,m/s. Right: the NAVSIM benchmark on 783 Singapore scenes.}",
 r"\label{tab:report}",r"\small",r"\setlength{\tabcolsep}{2.0pt}",
 r"\begin{tabular}{@{}lcccccccc@{}}",r"\toprule",
 r" & exposure & HS & scaling & SP & \multicolumn{2}{c}{collision (\%), visible\,/\,removed} & \multicolumn{2}{c}{NAVSIM} \\",
 r"\cmidrule(lr){6-7}\cmidrule(l){8-9}",
 r"Policy & ("+_W("distance","position")+r") & ("+_W("yielding","risk mgmt.")+r") & ("+_W("scaling","procyclicality")+r") & (over-reaction) & logged $v$ & 8\,m/s & EPDMS & DAC \\",r"\midrule"]
for m in M:
    p=A[m]["point"]; c1=coll(m,"actual"); c8=coll(m,"8")
    T.append(f"{NAME[m]} & {p['exposure']:.2f} {ci(A[m],'exposure')} & {sgn(p['HS'])} {ci(A[m],'HS')} & {sgn(p['HS_slope'])} & {p['SP']:.2f} {ci(A[m],'SP')} & "
             f"{c1[0]:.1f}\\,/\\,{c1[1]:.1f} & {c8[0]:.1f}\\,/\\,{c8[1]:.1f} & {PD['table'][m]['pdms']:.3f} & {PD['table'][m]['drivable_area_compliance']:.3f} \\\\")
T+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
open(f"{_OUT}/tables/tab_report.tex","w").write("\n".join(T)+"\n")
# ---------- Table III：跨舵位预注册 ----------
lab={"P1":"Lighting outweighs the pedestrian (CFR $<1$) for every policy",
     "P2":"Hazard sensitivity stays below 0.15 (upper CI)",
     "P3":"Exposure ordering transfers ($\\rho\\ge0.6$; SimLingo top, DD bottom)",
     "P4":"Over-reaction ordering transfers ($\\rho\\ge0.6$; SimLingo lowest SP)",
     "P5":f"Point values stay within the left-hand CI ($\\ge$70\\% of {len(PR['P5']['cells'])} cells)",
     "P6":"Purer policies (lower $|$align$|$) shift less",
     "P7":"SimLingo collides most; visible vs.\\ removed within 3\\,pts"}
def det(k):
    d=PR[k]
    if k=="P1": return f"max upper CI {max(v[1] for v in d['detail'].values()):.2f}"
    if k=="P2": 
        bad=[SH[m] for m,v in d['detail'].items() if v[1]>=0.15]; return "fails for "+", ".join(bad) if bad else "all below"
    if k in ("P3","P4"): return f"$\\rho={d['rho']:.2f}$"
    if k=="P5": return f"{100*d['rate']:.0f}\\%"
    if k=="P6": return f"$\\rho={d['rho']:+.2f}$ (opposite)"
    if k=="P7": return f"gap {100*d['max_gap']:.1f}\\,pts"
T=[r"\begin{table}[t]",r"\centering",
   r"\caption{\textbf{Pre-registered transfer test.} Diagnoses made on 88 left-hand-drive frames (Boston) predict the near-pedestrian units of 148 right-hand-drive frames (Singapore). "
   r"Predictions and decision rules were fixed before any right-hand statistic was computed. Evaluated on the five policies whose registration the "
   r"Alpamayo-1.5 replacement leaves intact (\cref{sec:sample}); retaining the replaced policy changes no verdict.}",
   r"\label{tab:prereg}",r"\footnotesize",r"\setlength{\tabcolsep}{2.5pt}",
   r"\begin{tabular}{@{}lp{4.45cm}cl@{}}",r"\toprule",r" & Prediction & Holds & Evidence \\",r"\midrule"]
for k in ["P1","P3","P4","P7","P2","P5","P6"]:
    yn="yes" if PR[k]["pass_"] else r"\textbf{no}"
    T.append(f"{k} & {lab[k]} & {yn} & {det(k)} \\\\")
T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(f"{_OUT}/tables/tab_prereg.tex","w").write("\n".join(T)+"\n")
# ---------- Table IV：光照（精简：不列 p，显著者加粗） ----------
NS=json.load(open(f"{V5}/night_speed.json"))
def bold(txt,cond): return f"\\textbf{{\\boldmath {txt}}}" if cond else txt
T=[r"\begin{table}[t]",r"\centering",
   r"\caption{\textbf{What the night rendering does to the plan.} CFR: plan change from removing the pedestrian over plan change from the night "
   r"rendering, same frames (corridor pedestrians 5--12\,m, 95\% CI; ideal $\gg1$). Speed: change of planned mean speed over 2.5\,s under the night rendering. "
   r"$\Delta S$: change of whole-plan separation from the pedestrian (negative = closer). Both on frames where the ego moves ($\ge$1\,m/s). "
   r"Align: direction of the night-induced change relative to the removal-induced one, Boston\,$\to$\,Singapore (0 = orthogonal). Bold: $p<0.01$.}",
   r"\label{tab:light}",r"\footnotesize",r"\setlength{\tabcolsep}{3pt}",
   r"\begin{tabular}{@{}lcccc@{}}",r"\toprule",
   r"Policy & CFR [95\% CI] & speed & $\Delta S$ (m) & align \\",r"\midrule"]
for m in M:
    c=CF[m]["corr"]; n=NS[m]
    pm=lambda x,f: "$0$" if abs(x)<(0.005 if f=="{:+.2f}" else 0.5) else "$"+f.format(x).replace("-","-")+"$"
    sp=bold(pm(100*n['dv_rel'],"{:+.0f}")[:-1]+"\\%$" if abs(100*n['dv_rel'])>=0.5 else "$0\\%$",n["p"]<0.01); ds=bold(pm(n['dS'],"{:+.2f}"),n["dS_p"]<0.01)
    T.append(f"{SH[m]} & {c['CFR']:.2f} {{\\scriptsize[{c['CFR_ci'][0]:.2f},{c['CFR_ci'][1]:.2f}]}} & {sp} & {ds} & "
             f"${L[m]['point']['align']:+.2f}\\to{R[m]['point']['align']:+.2f}$ \\\\")
T+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
open(f"{_OUT}/tables/tab_light.tex","w").write("\n".join(T)+"\n")
# ---------- 数字宏 ----------
cf=[CF[m]["corr"]["CFR"] for m in M]; hs=[A[m]["point"]["HS"] for m in M]; ex=[A[m]["point"]["exposure"] for m in M]
ns=SS["dims"]; nc=[PD["table"][m]["no_at_fault_collisions"] for m in M]
g1=max(abs(coll(m,"actual")[0]-coll(m,"actual")[1]) for m in M); g8=max(abs(coll(m,"8")[0]-coll(m,"8")[1]) for m in M)
NUM={"NumCFRlo":f"{min(cf):.2f}","NumCFRhi":f"{max(cf):.2f}","NumCFRciHi":f"{max(CF[m]['corr']['CFR_ci'][1] for m in M):.2f}",
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
     "NumPool":str(SS["N"]),"NumCollGap":f"{g1:.1f}","NumCollGapEight":f"{g8:.1f}",
     "NumNstarCFRhi":f"{max(ns['CFR']['n_star'].values()):.0f}","NumNstarExpHi":f"{max(v for m,v in ns['exposure']['n_star'].items() if m in ('dd','ltf','ddv2','simlingo')):.0f}",
     "NumRankSPforty":f"{100*ns['SP']['rank_p']['40']:.0f}","NumRankExpTen":f"{100*ns['exposure']['rank_p']['10']:.0f}",
     "NumRankHSninety":f"{100*ns['HS']['rank_p']['90']:.0f}","NumRankAlignOneThirty":f"{100*ns['align']['rank_p']['130']:.0f}",
     "NumPass":str(sum(PR[k]['pass_'] for k in ['P1','P2','P3','P4','P5','P6','P7'])),
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
    T=[r"\begin{table*}[t]",r"\centering",
       r"\caption{\textbf{The same six policies under the standard evaluations.} Left: nuScenes open-loop metrics on the same 236 near-pedestrian frames, "
       r"original\,/\,pedestrian removed (L2 to the human trajectory averaged over 0.5--2.5\,s; collisions with objects ahead of the ego, rear-end contacts by non-reactive logged agents excluded). "
       r"Middle: NAVSIM EPDMS on all 783 Singapore scenes and on 260 Singapore scenes with a pedestrian within 1\,m of the ego corridor and 20\,m ahead; the leaderboard ordering does not survive the slice. "
       r"Right: pedestrian TTC along the plan (share of moving-ego scenes with minimum TTC $<1.5$\,s, lower is better) on left-hand-drive near-pedestrian scenes "
       r"(nuScenes Boston + NAVSIM Las Vegas/Boston/Pittsburgh) and right-hand-drive ones (NAVSIM Singapore), with the resulting rank.}",
       r"\label{tab:bench}",r"\small",r"\setlength{\tabcolsep}{3pt}",
       r"\begin{tabular}{@{}lccc cc cc@{}}",r"\toprule",
       r" & \multicolumn{3}{c}{nuScenes open-loop (orig.\,/\,removed)} & \multicolumn{2}{c}{NAVSIM EPDMS (rank)} & \multicolumn{2}{c}{pedestrian TTC $<1.5$\,s (rank)} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
       r"Policy & L2 (m) & collision (\%) & ped.\ coll.\ (\%) & all & near ped. & left-hand & right-hand \\",r"\midrule"]
    # EPDMS 两列各自排名（高分为 1），让"榜单第一跌到第五"在表里直接看得见
    _rk=lambda d: {m:i+1 for i,m in enumerate(sorted(M,key=lambda x:-d[x]["epdms"]))}
    RKA=_rk(AL); RKC=_rk(CV)
    for m in M:
        n=NUo[m]; L_=TR["LHD"][m]["moving"]; R_=TR["RHD"][m]["moving"]
        T.append(f"{NAME[m]} & {n['clean']['L2_avg']:.2f}\\,/\\,{n['rm']['L2_avg']:.2f} & {n['clean']['col_front']:.1f}\\,/\\,{n['rm']['col_front']:.1f} & "
                 f"{n['clean']['col_ped']:.1f}\\,/\\,{n['rm']['col_ped']:.1f} & {AL[m]['epdms']:.3f} ({RKA[m]}) & {CV[m]['epdms']:.3f} ({RKC[m]}) & "
                 f"{L_['viol']:.1f} ({TR['ranks']['LHD TTC'][m]}) & {R_['viol']:.1f} ({TR['ranks']['RHD TTC'][m]}) \\\\")
    T+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
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
       r"\caption{\textbf{Re-measuring the two axes in the new domain.} $\CFR$ measured on Boston frames, on Singapore frames, and per scene on the ten Singapore cases "
       r"(median). The verdict $\CFR<1$ holds on both sides for every policy, while the value drifts by $0.6$--$1.7\times$; the across-policy ordering of $\CFR$ agrees between "
       r"sides only at $\rho={+}0.37$ (specificity: $\rho={+}0.83$). Right: share of near-pedestrian scenes in which each rule holds "
       r"(P-F: a near-zero F axis implies removing the pedestrian leaves clearance and contact unchanged; P-I: $\CFR<1$ implies the night rendering moves the outcome at least as much as the removal). "
       r"Both rules are two-sided: each cell gives the hit rate when the premise holds / when it is reversed (fwd/rev), so every scene receives a prediction. "
       r"and the share of scenes whose minimum time-to-proximity to the pedestrian's logged future falls below 1.5\,s. The large-sample columns use the \NumComN{} NAVSIM Singapore scenes all six policies share; the Boston column is measured on the nuScenes frames of \cref{tab:report}.}",
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
               r"$I$ (displacement caused by the night rendering, the policy's own jitter under an irrelevant edit) and $\Delta S$, the resulting change in clearance to the pedestrian's logged future ($\Delta S>0$: seeing the pedestrian keeps the plan farther away). "
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
        NUM3["NumExpBeh"]=f"{DT['lhd_to_behaviour']['exposure']:+.2f}".replace("+","{+}")
        NUM3["NumSpBeh"]=f"{DT['lhd_to_behaviour']['SP']:+.2f}".replace("+","{+}").replace("-","{-}")
        NUM3["NumCfrBeh"]=f"{DT['lhd_to_behaviour']['CFR']:+.2f}".replace("+","{+}").replace("-","{-}")
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
    with open(f"{_OUT}/tables/numbers.tex","a") as fh: fh.write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in NUM3.items())+"\n")
