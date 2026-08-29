"""Stage D/E|把主方向的 ±α 整条阶梯斜率与**同层 ≥20 seed 随机方向零分布**合并,给出 E2 判定。

依据修正案 A3:随机对照必须用**整条 ±α 阶梯的斜率**与同层零分布比,
不能用单点 α 的效应量(单点分不开"无符号扰动"与"有符号方向效应",且零分布随层变化)。
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np

VAR = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2/results")
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")

MAIN = [  # (展示名, t2_steer tag, 注入层, token 集)
    ("v_hazard_clean @L9 (vision)",  "dv_hazard_clean",      9,  "vision"),
    ("v_hazard_carla @L8 (vision)",  "dv_hazard_carla",      8,  "vision"),
    ("A1 候选1 @L22 (vision)",       "dv_cand1",             22, "vision"),
    ("A1 候选2 @L18 (vision)",       "dv_cand2",             18, "vision"),
    ("A1 候选3 @L18 (vision)",       "dv_cand3",             18, "vision"),
    ("v_brake @L22 (query)",         "brake",                22, "query"),
    ("v_danger_lang @L10 (query)",   "lang2",                10, "query"),
    ("v_hazard_clean @L23 (query)",  "dvq_hazard_clean_L23", 23, "query"),
    ("v_hazard_clean @L9 (query)",   "dvq_hazard_clean_L9",  9,  "query"),
    ("v_hazard_carla @L8 (query)",   "dvq_hazard_carla_L8",  8,  "query"),
    ("A1 候选1 @L22 (query)",        "dvq_cand1_L22",        22, "query"),
]


def find_null(layer, tokens):
    """优先用同层同 token 集的零分布;其次同层任意 token 集(标注为降级)。"""
    cands = []
    for f in glob.glob(str(VAR / "t2_random_null*.json")):
        d = json.load(open(f))
        if d.get("layer") != layer:
            continue
        cands.append((d.get("tokens") == tokens, Path(f).name, d))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    exact, name, d = cands[0]
    sl = np.array([s["full_slope"] for s in d["slopes"]])
    return {"file": name, "layer": layer, "tokens": d.get("tokens"), "token_match": bool(exact),
            "n_seeds": len(sl), "mean": float(sl.mean()), "sd": float(sl.std(ddof=1)),
            "min": float(sl.min()), "max": float(sl.max()), "slopes": sl.tolist()}


def main():
    rows = []
    for disp, tag, L, tok in MAIN:
        p = VAR / f"t2_steer_region_mean_{tag}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        sl = d.get("dose_slope_full") or d.get("dose_slope")
        n = find_null(L, tok)
        row = {"direction": disp, "layer": L, "tokens": tok,
               "slope_full_ladder": sl["mean"], "slope_ci95": sl.get("ci95"),
               "n_events": d.get("n_events"),
               "dose_monotonic": bool(sl["mean"] < 0 and (sl.get("ci95") or [1, 1])[1] < 0),
               "termination": d.get("termination")}
        # 特异性检查（计划 Stage D 四件套之一）：注入后横向行为与舒适度不应劣化。
        # 报无量纲比值 |Δ横向| / |Δv_plan| 与 |Δ舒适度| / |Δv_plan|，在 α = ±2 上取。
        dz = d.get("dose", {})
        spec = {}
        for a in ("+2", "-2"):
            e = dz.get(a)
            if e and abs(e["dv"]) > 1e-9:
                spec[a] = {"dv": e["dv"], "d_lateral": e["d_lateral"], "d_comfort": e["d_comfort"],
                           "lateral_over_dv": abs(e["d_lateral"]) / abs(e["dv"]),
                           "comfort_over_dv": abs(e["d_comfort"]) / abs(e["dv"])}
        row["specificity"] = spec
        if n:
            z = (sl["mean"] - n["mean"]) / (n["sd"] + 1e-12)
            n_ge = int((np.abs(np.array(n["slopes"])) >= abs(sl["mean"])).sum())
            row["null"] = {k: v for k, v in n.items() if k != "slopes"}
            row["z_vs_null"] = float(z)
            row["n_null_ge"] = n_ge
            row["p_empirical"] = float((n_ge + 1) / (n["n_seeds"] + 1))
            row["exceeds_null"] = bool(n_ge == 0 and abs(z) > 2)
        rows.append(row)

    for r in rows:
        nl = r.get("null")
        s = (f"零分布 {nl['mean']:+.5f}±{nl['sd']:.5f} (n={nl['n_seeds']}, L{nl['layer']}/{nl['tokens']}"
             f"{'' if nl['token_match'] else ' ⚠️token 集不同,降级'}) "
             f"z={r['z_vs_null']:+.2f} p_emp={r['p_empirical']:.3f} "
             f"{'✅超出' if r['exceeds_null'] else '❌未超出'}") if nl else "零分布缺失"
        sp = r.get("specificity", {}).get("+2")
        sps = (f" | 特异性@α=+2: Δv={sp['dv']:+.4f} Δ横向={sp['d_lateral']:+.4f} "
               f"Δ舒适={sp['d_comfort']:+.4f} (比值 {sp['lateral_over_dv']:.2f} / {sp['comfort_over_dv']:.2f})"
               if sp else "")
        print(f"{r['direction']:30s} 斜率={r['slope_full_ladder']:+.5f} "
              f"[{r['slope_ci95'][0]:+.5f},{r['slope_ci95'][1]:+.5f}] n={r['n_events']:3d} | {s}{sps}")
    (RES / "e2_steering_verdicts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    md = ["**Table D2. 剂量单调性、同层随机方向零分布对照与 termination/recovery（Stage D 四件套之三）。**", "",
          "| 方向 | 注入层 | token 集 | n | ±α 全阶梯斜率 | 95% CI | 同层随机零分布 (n seed) | z | 经验 p | 超出零分布 | project_out Δv | recover Δv |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        n = r.get("null"); t = r.get("termination") or {}
        nul = (f"{n['mean']:+.5f} ± {n['sd']:.5f} ({n['n_seeds']})"
               + ("" if n.get("token_match") else f" ⚠️L{n['layer']}/{n['tokens']}")) if n else "缺失"
        z = f"{r['z_vs_null']:+.2f}" if n else "—"
        pe = f"{r['p_empirical']:.3f}" if n else "—"
        ex = ("✅" if r.get("exceeds_null") else "❌") if n else "—"
        po = f"{t['project_out']['dv']:+.4f}" if t else "—"
        rc = f"{t['recover']['dv']:+.4f}" if t else "—"
        md.append(f"| {r['direction']} | L{r['layer']} | {r['tokens']} | {r['n_events']} | "
                  f"{r['slope_full_ladder']:+.5f} | [{r['slope_ci95'][0]:+.5f}, {r['slope_ci95'][1]:+.5f}] | "
                  f"{nul} | {z} | {pe} | {ex} | {po} | {rc} |")
    md += ["", "**Table D3. 特异性检查（Stage D 四件套之四）：注入后横向行为与舒适度不应随之劣化。**",
           "在 α = +2 上报无量纲比值；比值 ≪ 1 = 效应主要落在纵向（特异），≈ 1 = 横向被同等推动（不特异）。", "",
           "| 方向 | 注入层 | token 集 | Δv_plan | Δ横向 | Δ舒适度 | \|Δ横向\|/\|Δv\| | \|Δ舒适\|/\|Δv\| | 特异性 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = r.get("specificity", {}).get("+2")
        if not sp:
            continue
        lab = "✅ 纵向特异" if sp["lateral_over_dv"] < 0.3 else ("⚠️ 中等" if sp["lateral_over_dv"] < 0.7 else "❌ 不特异")
        md.append(f"| {r['direction']} | L{r['layer']} | {r['tokens']} | {sp['dv']:+.4f} | "
                  f"{sp['d_lateral']:+.4f} | {sp['d_comfort']:+.4f} | {sp['lateral_over_dv']:.2f} | "
                  f"{sp['comfort_over_dv']:.2f} | {lab} |")
    (RES / "steering_results" / "e2_tables.md").write_text("\n".join(md) + "\n")
    print(f"\n[E2] wrote results/e2_steering_verdicts.json + steering_results/e2_tables.md  ({len(rows)} 条方向)")


if __name__ == "__main__":
    main()
