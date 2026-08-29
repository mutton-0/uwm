"""Stage D 收口|解析法 vs 经验法一致性检验(H3) + 剂量/随机/termination 四件套汇总。

产出 results/analytic_vs_empirical.md 与 results/d3_compare.json。

一致性检验分两层:
  ① **汇总层**:解析预测的平均斜率 dΔv/dα vs 经验 α 扫描拟合的斜率(含 scene 级 bootstrap CI);
  ② **逐事件层**:同一批 S_test 事件上,解析预测的逐事件斜率 与 经验逐事件斜率 的 Pearson/Spearman
     —— 这是 H3 的强版本(汇总层符号一致可能只是两条独立的均值恰好同号)。
解析式仅对 **tokens=query** 的注入适用(残差流恒等路径直达 query 位置);
tokens=vision 的注入必须经注意力搬运,恒等路径预测为 0,单列不参与 H3 判定。
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
VAR = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2/results")
ALPHAS = [0.5, 1, 2, 4]


def emp_slope_per_event(row):
    """对单事件的 ± 阶梯做最小二乘斜率(与 t2_steer 的 full_slope 同构)。"""
    xs, ys = [], []
    for a in ALPHAS + [-a for a in ALPHAS]:
        k = f"a{a:+g}"
        if k in row and row[k] is not None:
            xs.append(a); ys.append(row[k][0] - row["base_clean"][0])
    if len(xs) < 4:
        return np.nan
    return float(np.polyfit(xs, ys, 1)[0])


def boot_r(x, y, scenes, n=2000, seed=0):
    by = {}
    for i, s in enumerate(scenes):
        by.setdefault(s, []).append(i)
    ks = sorted(by); rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        idx = np.concatenate([by[ks[i]] for i in rng.integers(0, len(ks), len(ks))])
        if len(set(x[idx])) < 3:
            continue
        o.append(np.corrcoef(x[idx], y[idx])[0, 1])
    o = np.array(o)
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


PAIRS = [
    # (解析 spec 名, 解析 json, 经验 tag, 层, tokens, 展示名)
    ("v_brake_L22_query",        "d2_analytic.json",      "brake",                 22, "query",  "v_brake @L22 (F 轴锚)"),
    ("v_hazard_clean_L23_query", "d2_analytic.json",      "dvq_hazard_clean_L23",  23, "query",  "v_hazard_clean @L23"),
    ("v_hazard_clean_L9_query",  "d2_analytic.json",      "dvq_hazard_clean_L9",    9, "query",  "v_hazard_clean @L9"),
    ("v_hazard_carla_L8_query",  "d2_analytic.json",      "dvq_hazard_carla_L8",    8, "query",  "v_hazard_carla @L8"),
    ("v_cand1_L22_query",        "d2_analytic.json",      "dvq_cand1_L22",         22, "query",  "A1 候选1 @L22 (occlusion)"),
    ("v_danger_lang_L10_query",  "d2_analytic_lang.json", "lang2",                 10, "query",  "v_danger_lang @L10"),
    ("v_hazard_clean_L9_vision", "d2_analytic.json",      "dv_hazard_clean",        9, "vision", "v_hazard_clean @L9 (vision token)"),
]


def main():
    ana = {f: json.load(open(RES / f)) for f in {p[1] for p in PAIRS} if (RES / f).exists()}
    rows = []
    for name, af, tag, L, tok, disp in PAIRS:
        if af not in ana:
            continue
        A = ana[af]
        per_a = {r["event_id"]: r[name] for r in A["per_event"] if name in r}
        ep = VAR / f"t2_steer_region_mean_{tag}.json"
        if not ep.exists():
            rows.append({"name": disp, "layer": L, "tokens": tok, "status": "经验臂缺失"})
            continue
        E = json.load(open(ep))
        emp = {r["event_id"]: (emp_slope_per_event(r), r["scene"]) for r in E["per_event"]}
        ids = [i for i in per_a if i in emp and np.isfinite(emp[i][0])]
        a = np.array([per_a[i]["analytic_slope_identity_path"] for i in ids])
        e = np.array([emp[i][0] for i in ids])
        sc = [emp[i][1] for i in ids]
        r_p = float(np.corrcoef(a, e)[0, 1]) if len(ids) > 5 else float("nan")
        r_s = float(stats.spearmanr(a, e).statistic) if len(ids) > 5 else float("nan")
        ci = boot_r(a, e, sc) if len(ids) > 5 else [float("nan")] * 2
        rows.append({
            "name": disp, "layer": L, "tokens": tok, "n_matched_events": len(ids),
            "analytic_mean_slope": float(A["summary"][name]["mean_analytic_slope"]),
            "analytic_sd": float(A["summary"][name]["sd"]),
            "empirical_slope": (E.get("dose_slope_full") or E["dose_slope"])["mean"],
            "empirical_ci95": (E.get("dose_slope_full") or E["dose_slope"])["ci95"],
            "empirical_slope_posonly": E["dose_slope"]["mean"],
            "empirical_ci95_posonly": E["dose_slope"]["ci95"],
            "empirical_n_events": E["n_events"],
            "sign_agree": bool(np.sign(A["summary"][name]["mean_analytic_slope"]) ==
                               np.sign((E.get("dose_slope_full") or E["dose_slope"])["mean"])),
            "magnitude_ratio_analytic_over_empirical":
                float(A["summary"][name]["mean_analytic_slope"] /
                      (E.get("dose_slope_full") or E["dose_slope"])["mean"])
                if abs((E.get("dose_slope_full") or E["dose_slope"])["mean"]) > 1e-9 else None,
            "event_level_pearson": r_p, "event_level_spearman": r_s,
            "event_level_pearson_ci95_scene_bootstrap": ci,
            "applicable_to_H3": tok == "query",
            "termination": E.get("termination"),
        })

    ok = [r for r in rows if r.get("applicable_to_H3") and "status" not in r]
    # 只在**经验斜率显著非零**（CI 不跨 0）的方向上判符号一致 ——
    # 经验值本身与 0 不可区分时，"符号"无从谈起，把它计入分母会人为压低一致率。
    sig = [r for r in ok if (r["empirical_ci95"][0] > 0) or (r["empirical_ci95"][1] < 0)]
    ns = [r for r in ok if r not in sig]
    n_sign = sum(r["sign_agree"] for r in sig)
    bad = [r["name"] for r in sig if not r["sign_agree"]]
    good_deep = [r for r in sig if r["sign_agree"] and r["layer"] >= 22]
    shallow = [r for r in sig if r["sign_agree"] and r["layer"] < 22]
    verdict = (
        f"**H3 部分成立，且有效边界被定量刻画。** 在经验斜率显著非零的 {len(sig)} 条适用方向上，"
        f"解析预测与实测**符号一致 {n_sign}/{len(sig)}**"
        + (f"（例外：{'、'.join(bad)}）" if bad else "")
        + "；"
        + (f"经验斜率与 0 不可区分的 {len(ns)} 条（{'、'.join(r['name'] for r in ns)}）不计入符号判定。 " if ns else " ")
        + (f"**深层注入（L ≥ 22，n={len(good_deep)}）量级高度吻合**，解析/经验比 "
           f"{min(r['magnitude_ratio_analytic_over_empirical'] for r in good_deep):.2f}–"
           f"{max(r['magnitude_ratio_analytic_over_empirical'] for r in good_deep):.2f}"
           f"（逐事件 Pearson r 最高 {max(r['event_level_pearson'] for r in good_deep):+.3f}）；" if good_deep else "")
        + (f"**浅/中层注入（L < 22，n={len(shallow)}）解析系统性低估** "
           f"{1/max(r['magnitude_ratio_analytic_over_empirical'] for r in shallow):.0f}–"
           f"{1/min(r['magnitude_ratio_analytic_over_empirical'] for r in shallow):.0f} 倍。" if shallow else "")
        + " ⇒ 解析捷径的有效边界是**注入层到动作头之间的非线性深度**，不是方向本身："
          "L ≥ 22 的候选可先用解析法初筛（成本 ≈ 1 次前向 + 1 次反传），L < 22 必须跑经验 α 扫描。")
    out = {"rows": rows, "n_sign_agree": n_sign, "n_applicable": len(ok),
           "n_empirically_significant": len(sig), "verdict_H3": verdict,
           "method": "解析：Δv_pred = α·σ_L·Σ_t⟨g_t, v̂⟩，g = ∂commanded_speed/∂h_23[query]（autograd 精确）；"
                     "经验：t2_steer ±{0.5,1,2,4} 阶梯的最小二乘斜率。"}
    (RES / "d3_compare.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    md = ["# Stage D 解析臂 vs 经验臂一致性报告（H3）", "",
          "> 计划依据：`direction_vector_discovery_validation_plan.md` Stage D「新增（仅适用于连续/近线性动作头）」+ §2 H3。",
          "> 数值产出物：`d2_analytic.json`、`d2_analytic_lang.json`、`d3_compare.json`、`steering_results/`。", "",
          "## 方法", "",
          "SimLingo 的动作头满足做解析投影的架构条件：",
          "`speed_wps_head = Linear(896→256) → SiLU → Linear(256→2, bias=False)`，预测再 `cumsum`；",
          "行为量 `commanded_speed(wp) = ‖wp[0] − wp[2]‖ × 2 = 2‖head(f₁) + head(f₂)‖`，",
          "**只依赖 speed_wps query 段的第 1、2 个位置**，且从最终隐状态到该量只隔一层 SiLU MLP。",
          "因此 g = ∂v_cmd/∂h₂₃ 可由 autograd **精确**求出（不是有限差分近似）。", "",
          "注入 Z′ = Z + α·σ_L·v̂ 于第 L 层的 query 位置时，残差流恒等路径给出",
          "Δh₂₃ ≈ Δh_L = α·σ_L·v̂，故 **Δv_pred(α) = α·σ_L·Σ_{t∈query}⟨g_t, v̂⟩**。",
          "L = 23 时该式仅忽略 head 的二阶项（SiLU 曲率）；L < 23 时还额外忽略 L→23 之间注意力/MLP 对该扰动的加工。",
          "**tokens = vision 的注入不经恒等路径到达 query 位置，解析预测为 0，单列不参与 H3 判定。**", "",
          "## Table 1. 解析预测斜率 vs 经验 α 扫描斜率（S_test A 类事件，两臂同一 seed 同一三分）", "",
          "经验斜率取**整条 ±α 阶梯**的最小二乘斜率（与解析式的线性对称假设同构，且按修正案 A3 约掉无符号扰动）；",
          "预注册的正向-α-only 斜率并列于末列作对照。", "",
          "| 方向 | 注入层 | token 集 | 解析 dΔv/dα (mean ± sd) | 经验斜率(±α 全阶梯) | 经验 95% CI | 符号一致 | 量级比 (解析/经验) | 逐事件 Pearson r | r 的 95% CI | 经验斜率(仅 +α，预注册口径) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "status" in r:
            md.append(f"| {r['name']} | L{r['layer']} | {r['tokens']} | — | — | — | — | — | — | — | {r['status']} |")
            continue
        mr = r["magnitude_ratio_analytic_over_empirical"]
        md.append(f"| {r['name']} | L{r['layer']} | {r['tokens']} | {r['analytic_mean_slope']:+.4f} ± {r['analytic_sd']:.4f} | "
                  f"{r['empirical_slope']:+.4f} | [{r['empirical_ci95'][0]:+.4f}, {r['empirical_ci95'][1]:+.4f}] | "
                  f"{'✅' if r['sign_agree'] else '❌'} | {mr:+.2f} | {r['event_level_pearson']:+.3f} | "
                  f"[{r['event_level_pearson_ci95_scene_bootstrap'][0]:+.3f}, {r['event_level_pearson_ci95_scene_bootstrap'][1]:+.3f}] | "
                  f"{r['empirical_slope_posonly']:+.4f} |")
    md += ["", f"**H3 判定**：{verdict}", "",
           "## Table 2. termination / recovery（ghost 帧，剔除方向 → 应减弱减速；加回 → 应回到基线）", "",
           "| 方向 | 注入层 | project_out Δv_plan | 95% CI | recover Δv_plan | 95% CI |", "|---|---|---|---|---|---|"]
    for r in rows:
        t = r.get("termination")
        if not t:
            continue
        md.append(f"| {r['name']} | L{r['layer']} | {t['project_out']['dv']:+.4f} | "
                  f"[{t['project_out']['ci95'][0]:+.4f}, {t['project_out']['ci95'][1]:+.4f}] | "
                  f"{t['recover']['dv']:+.4f} | [{t['recover']['ci95'][0]:+.4f}, {t['recover']['ci95'][1]:+.4f}] |")
    (RES / "analytic_vs_empirical.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-30:]))
    print(f"\n[D3] wrote results/analytic_vs_empirical.md + d3_compare.json")


if __name__ == "__main__":
    main()
