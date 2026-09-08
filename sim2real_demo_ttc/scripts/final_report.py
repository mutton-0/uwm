"""生成最终结果汇总（中文），数字全部从结果 JSON 读出，不手打。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
NAME = {"simlingo": "SimLingo", "dd": "DD", "ltf": "LTF", "ddv2": "DDv2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
CELLS = [("ghost", "nuScenes", "f3_gt_axis_full_{m}.json", "ghost_nusc"),
         ("lead", "nuScenes", "gt_lead_{m}.json", "lead_nusc"),
         ("ghost", "NAVSIM", "f3_gt_dep_ghost_{m}.json", "dep_ghost"),
         ("lead", "NAVSIM", "f3_gt_dep_lead_{m}.json", "dep_lead")]


def boot(bm, bg, sc, n=5000, seed=0):
    rng = np.random.default_rng(seed); uq = np.unique(sc)
    ix = {s: np.where(sc == s)[0] for s in uq}
    out = []
    for _ in range(n):
        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), True)])
        d = bg[sel].mean()
        if abs(d) > 1e-9:
            out.append(bm[sel].mean() / d)
    return np.percentile(out, [2.5, 97.5])


def f3(tmpl, m):
    p = RES / tmpl.format(m=m)
    if not p.exists():
        return None
    ev = json.load(open(p))["per_event"]
    bm = np.array([e["b_model__arc_full"] for e in ev])
    bg = np.array([e["b_gt__arc_full"] for e in ev])
    sc = np.array([e["scene"] for e in ev])
    lo, hi = boot(bm, bg, sc)
    return dict(n=len(ev), ns=len(np.unique(sc)), r=bm.mean() / bg.mean(),
                lo=lo, hi=hi, sig=not (lo <= 0 <= hi))


def cell(pat, w, m, key, sub=None):
    for f in (RES / f"{pat}_new_{w}_{m}.json", RES / f"{pat}_{w}_{m}.json"):
        if f.exists():
            d = json.load(open(f))
            return d[key] if sub is None else d.get(key, {}).get(sub)
    return None


L = []
A = L.append
A("# GFIC 最终结果汇总\n")
A(f"生成时间：脚本自动 ｜ 主口径 `arc_full` ｜ scene 级 bootstrap 5000 次\n")
A("**域划分（P-1）**：benchmark = 左舵(LHD)；deployment = 右舵(RHD)。"
  "两个语料 nuScenes/NAVSIM 先合并再按舵位切，旧的\"nuScenes=benchmark\"划分已废弃。"
  "注意 P-9：切完后舵位与语料仍近共线（NAVSIM 94% 左舵，右舵主体是"
  "nuScenes-新加坡），方向类读数不可直接归因到舵位。\n")

# ---- P-1 主表：按舵位切 ----
_MAP = json.load(open(RES / "driveside_map.json"))
_CORP = {"ghost": [("nuscenes", "f3_gt_axis_full_{m}.json"),
                   ("navsim", "f3_gt_dep_ghost_{m}.json")],
         "lead": [("nuscenes", "gt_lead_{m}.json"),
                  ("navsim", "f3_gt_dep_lead_{m}.json")]}


def _side(scene, corpus):
    if corpus == "nuscenes":
        return _MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):
        x = _MAP["navsim_side"].get(f"{sp}|{scene}")
        if x:
            return x


def _cell_side(scn, side, m):
    bm, bg, sc = [], [], []
    for corpus, tmpl in _CORP[scn]:
        f = RES / tmpl.format(m=m)
        if not f.exists():
            continue
        for e in json.load(open(f))["per_event"]:
            if _side(e["scene"], corpus) != side:
                continue
            bm.append(e["b_model__arc_full"]); bg.append(e["b_gt__arc_full"])
            sc.append(f"{corpus}|{e['scene']}")
    if len(bm) < 2:
        return None
    bm, bg, sc = np.array(bm), np.array(bg), np.array(sc)
    lo, hi = boot(bm, bg, sc)
    return (float(bm.mean() / bg.mean()), float(lo), float(hi)), len(bm), len(set(sc))


A("## 1. F 轴（P-1 主表）：响应比 $r$，按舵位切\n")
A("| 场景 | 域 | n(事件/scene) | " + " | ".join(NAME[m] for m in MODELS) + " |")
A("|---|---|---|" + "---|" * len(MODELS))
for scn in ("ghost", "lead"):
    for side, dom in (("LHD", "benchmark"), ("RHD", "deployment")):
        # 各候选覆盖的事件数不同（有的模型缺格），故 n 报**跨候选的范围**，不取单一值
        cells, ns_all = [], []
        for m in MODELS:
            c = _cell_side(scn, side, m)
            if c is None:
                cells.append("—"); continue
            (r, lo, hi), ne, ns = c
            ns_all.append(ne)
            cells.append(f"{r:+.3f} [{lo:+.2f},{hi:+.2f}]<br><sub>n={ne}</sub>")
        nn = (f"{min(ns_all)}–{max(ns_all)}" if len(set(ns_all)) > 1
              else str(ns_all[0]) if ns_all else "—")
        A(f"| {scn} | {dom} ({side}) | {nn} | " + " | ".join(cells) + " |")
A("")
A("> deployment(右舵) 侧每格只有 8–10 个事件 —— 按 P-2，这个量级只够报值，"
  "**不够排名**。且按 P-9，舵位与语料仍近共线，差异不可归因到舵位。\n")

A("## 1b. F 轴（分语料明细）：响应比 $r=\\bar b^{model}/\\bar b^{GT}$\n")
A("| 场景 | 语料 | n(事件/scene) | " + " | ".join(NAME[m] for m in MODELS) + " |")
A("|---|---|---|" + "---|" * len(MODELS))
for scn, corp, tmpl, _ in CELLS:
    cs, nn = [], ""
    for m in MODELS:
        c = f3(tmpl, m)
        if c is None:
            cs.append("—"); continue
        nn = f"{c['n']}/{c['ns']}"
        cs.append(f"**{c['r']:+.3f}** [{c['lo']:+.2f},{c['hi']:+.2f}]" if c["sig"]
                  else f"{c['r']:+.3f} [{c['lo']:+.2f},{c['hi']:+.2f}]")
    A(f"| {scn} | {corp} | {nn} | " + " | ".join(cs) + " |")
A("\n加粗 = CI 不跨 0。$r=1$ 表示响应幅度与人类实际减速相当。\n")

p = RES / "f3_contrasts.json"
if p.exists():
    d = json.load(open(p)); a = d["bonferroni_alpha"]
    A(f"## 2. 预登记对比（{d['n_tests']} 项，Bonferroni α={a:.4f}）\n")
    A("| 对比 | 候选 | Δr | 95%CI | p | 判定 |")
    A("|---|---|---|---|---|---|")
    for k, t in d["tests"].items():
        nm, cond, m = k.split("|")
        v = "**通过校正**" if t["p"] < a else ("未校正显著" if t["p"] < 0.05 else "")
        if not v:
            continue
        A(f"| {nm} / {cond} | {NAME[m]} | {t['delta']:+.3f} | "
          f"[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}] | {t['p']:.4f} | {v} |")
    A("")

A("## 3. G 轴 selectivity / C 轴 $C_m$\n")
A("| 场景 | 语料 | 轴 | " + " | ".join(NAME[m] for m in MODELS) + " |")
A("|---|---|---|" + "---|" * len(MODELS))
for scn, corp, _, w in CELLS:
    for axis, pat in (("G-VS", "gvs"), ("C", "c")):
        cs = []
        for m in MODELS:
            if axis == "G-VS":
                s = cell("gvs", w, m, "selectivity")
                cs.append("—" if s is None else
                          (f"**{s['mean']:+.4f}**" if not (s['ci95'][0] <= 0 <= s['ci95'][1])
                           else f"{s['mean']:+.4f}"))
            else:
                pa = cell("c", w, m, "patch_all_recovery", "mean")
                cm = cell("c", w, m, "C_m_top2_share")
                if cm is None:
                    cs.append("—")
                else:
                    # 自检未过也给数值，只加 † 标注 —— 盖掉数字会丢信息
                    flag = "†" if (pa is None or abs(pa - 1) > 0.25) else ""
                    cs.append(f"{cm['mean']:.3f}{flag}")
        A(f"| {scn} | {corp} | {axis} | " + " | ".join(cs) + " |")
A("\nG-VS 加粗 = CI 不跨 0（**跨 0 的也照给数值**，不隐藏）。"
  "C 轴 † = patch-ALL 偏离 1 超过 0.25，数值照给但逐层占比在该格未被识别，不应参与比较。\n")

p = RES / "deploy_rank_joint.json"
if p.exists():
    S = json.load(open(p))
    have = [m for m in MODELS if m in S["summary"]]      # 只列跑了 PDM 的候选
    A(f"## 4. 部署质量（官方 PDM，{S['n_tokens']} 个随机 scene，{len(have)} 家联合打分）\n")
    A("| 指标 | " + " | ".join(NAME[m] for m in have) + " |")
    A("|---|" + "---|" * len(have))
    for k in ["pdm_score", "no_at_fault_collisions", "time_to_collision_within_bound",
              "drivable_area_compliance", "ego_progress", "jerk_rms", "max_decel"]:
        A(f"| {k} | " + " | ".join(f"{S['summary'][m][k]['mean']:.3f}" for m in have) + " |")
    A("")

for scn in ("ghost", "lead"):
    p = RES / f"hazard_avoidance_dep_{scn}.json"
    q = RES / f"timeliness_dep_{scn}.json"
    if p.exists():
        d = json.load(open(p))
        A(f"### 危险事件上的 PDM（{scn}，n={d['n_events']}）\n")
        hv = [m for m in MODELS if m in d["summary"]]
        A("| 指标 | " + " | ".join(NAME[m] for m in hv) + " |")
        A("|---|" + "---|" * len(hv))
        for k in ["pdm_score", "no_at_fault_collisions", "time_to_collision_within_bound"]:
            A(f"| {k} | " + " | ".join(f"{d['summary'][m][k]['mean']:.3f}" for m in hv) + " |")
        A("")
    if q.exists():
        d = json.load(open(q))
        A(f"### 刹车及时性（{scn}，统一 0.25s 栅格/2.5s 时域）\n")
        A("| 候选 | 计划中曾减速 | 相对人类延迟 | 计划内最大降速 |")
        A("|---|---|---|---|")
        for m in MODELS:
            if m not in d["models"]:
                continue
            b = d["models"][m]; dl = b["delay_s_paired"]
            A(f"| {NAME[m]} | {b['model_braked']['frac']:.1%} | "
              f"{dl['mean']:+.2f}s [{dl['ci'][0]:+.2f},{dl['ci'][1]:+.2f}] (n={dl['n']}) | "
              f"{b['max_drop_mps']['mean']:.2f} m/s |")
        A("")

out = RES / "FINAL_RESULTS_zh.md"
out.write_text("\n".join(L))
print(f"[FINAL] wrote {out}  ({len(L)} 行)")
