"""从结果 JSON 生成论文表格（.tex）。表格由数据生成，不手打，避免转录错误。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
OUT = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/tables")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
NAME = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
PATCH_TOL = 0.25


def ci_str(m, ci, sig_bold=True, fmt="{:+.3f}"):
    """均值用正文字号、区间降一号。六候选横排时这一步就省掉约 15% 宽度，
    比整表 \\scriptsize 或 \\resizebox 更容易读。"""
    sig = not (ci[0] <= 0 <= ci[1])
    mean = fmt.format(m)
    mean = f"\\textbf{{{mean}}}" if (sig and sig_bold) else mean
    return (f"{mean}\\,{{\\scriptsize[{fmt.format(ci[0])},"
            f"{fmt.format(ci[1])}]}}")


# 栏宽 252pt / 通栏 516pt（IEEEtran conference）。超过 252pt 的一律 table*，
# 并把 tabcolsep 从默认 6pt 收到 4pt —— 六列就省 20pt。
def topen(caption, label, colspec, wide=True, colsep=4):
    env = "table*" if wide else "table"
    return [f"\\begin{{{env}}}[!tb]", r"\centering", r"\footnotesize",
            r"\setlength{\tabcolsep}{%dpt}" % colsep,
            f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"]


def tclose(wide=True):
    return [r"\bottomrule", r"\end{tabular}",
            r"\end{table*}" if wide else r"\end{table}", ""]


def boot_ratio(bm, bg, sc, n=5000, seed=0):
    rng = np.random.default_rng(seed); uq = np.unique(sc)
    ix = {s: np.where(sc == s)[0] for s in uq}
    out = []
    for _ in range(n):
        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])
        d = bg[sel].mean()
        if abs(d) > 1e-9:
            out.append(bm[sel].mean() / d)
    return np.percentile(out, [2.5, 97.5])


def f3_cell(path):
    p = RES / path
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    bm = np.array([e["b_model__arc_full"] for e in ev])
    bg = np.array([e["b_gt__arc_full"] for e in ev])
    sc = np.array([e["scene"] for e in ev])
    pos, neg = bm[bm > 0].sum(), -bm[bm < 0].sum()
    return {"n": len(ev), "ns": len(np.unique(sc)), "r": bm.mean() / bg.mean(),
            "ci": boot_ratio(bm, bg, sc),
            "cancel": (neg / pos) if pos > 1e-9 else float("nan")}


def table_f3(rows, fname, caption, label):
    """六候选下横排会超宽，故**转置**：候选作行、语料/场景作列。"""
    cols = [(corp, scen, tmpl) for corp, scen, tmpl in rows]
    data = {m: [] for m in MODELS}
    ns = []
    for corp, scen, tmpl in cols:
        n = None
        for m in MODELS:
            c = f3_cell(tmpl.format(m=m))
            data[m].append(c)
            if c and n is None:
                n = c["n"]
        ns.append(n)
    # F-3 的格子同时带区间和 rho，自然宽 534pt > 通栏 516pt。列距收到 3pt
    # 后仍差一点，故再套一层 resizebox —— 缩放比约 0.97，肉眼无感，
    # 且是唯一一张需要缩放的表。
    lines = topen(caption, label, "l" + "c" * len(cols), colsep=3)
    lines.insert(lines.index(r"\begin{tabular}{l" + "c" * len(cols) + "}"),
                 r"\resizebox{\textwidth}{!}{%")
    hdr = " & ".join(f"{c[0].split()[0]}\\,{c[1]}" for c in cols)
    lines.append("Policy & " + hdr + r"\\")
    lines.append("& " + " & ".join(f"$n{{=}}{n}$" if n else "--" for n in ns) + r"\\")
    lines.append(r"\midrule")
    for m in MODELS:
        cs = []
        for c in data[m]:
            if c is None:
                cs.append("--")
            else:
                # 反向抵消率随均值一起给：均值近 0 可能是"弱"也可能是"正负相消"
                rho = c.get("cancel")
                tag = ("" if rho is None or not np.isfinite(rho)
                       else f"\\,{{\\scriptsize$\\rho{{=}}{rho:.2f}$}}")
                cs.append(ci_str(c["r"], c["ci"]) + tag)
        lines.append(f"{NAME[m]} & " + " & ".join(cs) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    (OUT / fname).write_text("\n".join(lines))
    print(f"  wrote {OUT/fname}")


def table_deploy(fname):
    p = RES / "deploy_rank_joint.json"
    if not p.exists():
        print("  跳过部署表：缺 deploy_rank_joint.json"); return
    d = json.load(open(p)); S = d["summary"]
    keys = [("pdm_score", "PDM score"), ("no_at_fault_collisions", "No at-fault collision"),
            ("time_to_collision_within_bound", "TTC within bound"),
            ("drivable_area_compliance", "Drivable area"),
            ("driving_direction_compliance", "Driving direction"),
            ("ego_progress", "Ego progress"), ("lane_keeping", "Lane keeping"),
            ("jerk_rms", "Jerk RMS (m/s$^3$)"), ("max_decel", "Max decel (m/s$^2$)")]
    have0 = [m for m in MODELS if m in S]
    lines = [r"\begin{table*}[!tb]", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Deployment quality on a common NAVSIM stimulus set "
             f"($n={d['n_tokens']}$ scenes), official PDM scorer, all four candidates scored "
             r"as one proposal batch. Progress is normalised within the batch (no PDM-Closed "
             r"baseline available), so these are \emph{relative} scores, comparable across "
             r"the four rows here but not with published absolute PDM numbers.}",
             r"\label{tab:deploy}",
             r"\begin{tabular}{l" + "c" * len(have0) + "}", r"\toprule",
             r"Metric & " + " & ".join(NAME[m] for m in have0) + r"\\", r"\midrule"]
    have = [m for m in MODELS if m in S]          # 部署指标只对跑了 PDM 的候选存在
    for k, lab in keys:
        vals = [S[m][k]["mean"] for m in have if k in S[m]]
        if not vals:
            continue
        best = max(vals) if k not in ("jerk_rms", "max_decel") else min(vals)
        cs = []
        for m in have:
            v = S[m][k]["mean"]
            cs.append(f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}")
        lines.append(f"{lab} & " + " & ".join(cs) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / fname).write_text("\n".join(lines))
    print(f"  wrote {OUT/fname}")


def main():
    print("生成论文表格：")
    table_f3([("nuScenes", "ghost", "f3_gt_axis_full_{m}.json"),
              ("nuScenes", "lead", "gt_lead_{m}.json"),
              ("NAVSIM", "ghost", "f3_gt_dep_ghost_{m}.json"),
              ("NAVSIM", "lead", "f3_gt_dep_lead_{m}.json")],
             "tab_f3.tex",
             "Faithfulness: response ratio $r$ with scene-level bootstrap 95\\% CI. "
             "$r{=}1$ matches the human's own deceleration magnitude. Bold = CI excludes 0.",
             "tab:f3")
    table_deploy("tab_deploy.tex")
    table_timeliness()
    table_hazard()
    table_gic()



def table_timeliness(fname="tab_timeliness.tex"):
    rows = []
    for sc, lab in [("dep_ghost", "ghost"), ("dep_lead", "lead"),
                    ("ghost", "ghost"), ("lead", "lead")]:
        p = RES / f"timeliness_{sc}_navsim.json"
        p2 = RES / f"timeliness_{sc}.json"
        q = p if p.exists() else (p2 if p2.exists() else None)
        if q:
            rows.append((lab, json.load(open(q))["models"]))
            if len(rows) == 2:
                break
    if not rows:
        print("  跳过及时性表：缺数据"); return
    L = [r"\begin{table*}[!tb]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Timeliness of the planned braking response, on a common $0.25$\,s "
         r"grid and horizon. `Plans a slowdown' is the fraction of events in which the "
         r"policy's own plan ever drops $1$\,m/s below the current speed; the delay is "
         r"computed only on events where both the policy and the human do, so it must "
         r"be read together with that fraction.}",
         r"\label{tab:timeliness}", r"\begin{tabular}{llcc}", r"\toprule",
         r"Scenario & Policy & Plans a slowdown & Onset delay vs.\ human (s)\\", r"\midrule"]
    for lab, M in rows:
        for m in MODELS:
            if m not in M:
                continue
            b = M[m]
            d = b["delay_s_paired"]
            L.append(f"{lab} & {NAME[m]} & {b['model_braked']['frac']*100:.1f}\\% & "
                     f"{d['mean']:+.2f}\\,[{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]" + r"\\")
        L.append(r"\midrule")
    L = L[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (OUT / fname).write_text("\n".join(L)); print(f"  wrote {OUT/fname}")


def table_hazard(fname="tab_hazard.tex"):
    got = {}
    for sc in ("dep_ghost", "dep_lead", "ghost", "lead"):
        p = RES / f"hazard_avoidance_{sc}.json"
        if p.exists():
            got[sc.replace("dep_", "")] = json.load(open(p))
    if not got:
        print("  跳过避障表：缺数据"); return
    dep = RES / "deploy_rank_joint.json"
    D = json.load(open(dep))["summary"] if dep.exists() else {}
    keys = [("no_at_fault_collisions", "No at-fault collision"),
            ("time_to_collision_within_bound", "TTC within bound"),
            ("pdm_score", "PDM score")]
    L = [r"\begin{table*}[!tb]", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\caption{The same official metrics on \emph{mined hazard events} versus on a "
         r"random sample of the same domain. The ordering is not the same.}",
         r"\label{tab:hazard}"]
    HV = sorted({m for d in got.values() for m in d["summary"]} | set(D),
                key=lambda x: MODELS.index(x))
    L += [r"\begin{tabular}{ll" + "c" * len(HV) + "}", r"\toprule",
          r"Stimuli & Metric & " + " & ".join(NAME[m] for m in HV) + r"\\", r"\midrule"]
    for sc, d in got.items():
        S = d["summary"]
        for k, lab in keys:
            cs = [f"{S[m][k]['mean']:.3f}" if m in S else "--" for m in HV]
            L.append(f"hazard ({sc}, $n{{=}}{d['n_events']}$) & {lab} & " + " & ".join(cs) + r"\\")
        L.append(r"\midrule")
    if D:
        for k, lab in keys:
            cs = [f"{D[m][k]['mean']:.3f}" if k in D.get(m, {}) else "--" for m in HV]
            L.append(f"random & {lab} & " + " & ".join(cs) + r"\\")
    L += tclose()
    (OUT / fname).write_text("\n".join(L)); print(f"  wrote {OUT/fname}")


def table_gic():
    """G-VS 与 C 拆成两张表。

    原来一张 14 列的合表自然宽度 1018pt，通栏只有 516pt，压根放不下。
    拆开后每张就是 tab_f3 的形状：**候选作行、语料/场景作列**（4 列），
    通栏可容。"""
    cols = [("nuScenes", "ghost", "ghost_nusc"), ("nuScenes", "lead", "lead_nusc"),
            ("NAVSIM", "ghost", "dep_ghost"), ("NAVSIM", "lead", "dep_lead")]

    def pick(*cands):
        for c in cands:
            if (RES / c).exists():
                return RES / c
        return None

    # ---- G-VS ----
    L = topen(r"Grounding: probe selectivity against a random-initialisation control. "
              r"Bold = CI excludes $0$. All six policies sit within $0.016$ of one "
              r"another, which is why the axis supports values but not a ranking "
              r"(Table~\ref{tab:samplesize}).",
              "tab:gvs", "l" + "c" * len(cols))
    L.append("Policy & " + " & ".join(f"{c[0]}\\,{c[1]}" for c in cols) + r"\\")
    L.append(r"\midrule")
    for m in MODELS:
        cs = []
        for _, _, w in cols:
            q = pick(f"gvs_new_{w}_{m}.json", f"gvs_{w}_{m}.json")
            if q is None:
                cs.append("--")
            else:
                d = json.load(open(q))["selectivity"]
                cs.append(ci_str(d["mean"], d["ci95"], fmt="{:+.3f}"))
        L.append(f"{NAME[m]} & " + " & ".join(cs) + r"\\")
    L += tclose()
    (OUT / "tab_gvs.tex").write_text("\n".join(L)); print(f"  wrote {OUT/'tab_gvs.tex'}")

    # ---- C ----
    L = topen(r"Concentration $C_m$: share of the recovery carried by the top two "
              r"layers, against a diffuse baseline of $2/n_{\mathrm{layers}}$. "
              r"$\dagger$ marks a cell where the patch-all sufficiency check failed "
              r"($|\text{patch-all}-1|>0.25$): the value is reported but the per-layer "
              r"share is not identified there, so it should not be compared.",
              "tab:c", "l" + "c" * len(cols))
    L.append("Policy & " + " & ".join(f"{c[0]}\\,{c[1]}" for c in cols) + r"\\")
    L.append(r"\midrule")
    for m in MODELS:
        cs = []
        for _, _, w in cols:
            q = pick(f"c_new_{w}_{m}.json", f"c_{w}_{m}.json")
            if q is None:
                cs.append("--")
            else:
                dd = json.load(open(q))
                pa = dd.get("patch_all_recovery", {}).get("mean", float("nan"))
                cm = dd["C_m_top2_share"]
                # 自检未过时**照给数值**，只加 dagger 标注 —— 盖掉数字会丢信息
                v = (f"{cm['mean']:.3f}\\,{{\\scriptsize[{cm['ci95'][0]:.3f},"
                     f"{cm['ci95'][1]:.3f}]}}")
                cs.append(v + ("$^\\dagger$" if abs(pa - 1.0) > PATCH_TOL else ""))
        L.append(f"{NAME[m]} & " + " & ".join(cs) + r"\\")
    L += tclose()
    (OUT / "tab_c.tex").write_text("\n".join(L)); print(f"  wrote {OUT/'tab_c.tex'}")


if __name__ == "__main__":
    main()
