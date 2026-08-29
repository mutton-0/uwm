"""三线头号证据表|公开榜单排名 vs G/F/I/C 四轴矩阵 vs 真实 post-train 结果。

纪律（工单明文）：**不做任意加权的单一复合分**。
本脚本只做三件事：把三条线的读数按同一张表对齐、逐格标注可比性、给出三态判定；
凡不可比的格子一律标注为什么不可比，不用任何权重把它们抹平成一个数。

第一条线（公开榜单）的关键论据不是"谁分高"，而是**两个分数根本不在同一把尺子上**：
  SimLingo        CARLA Leaderboard 2.0  Driving Score  6.87   （闭环、CARLA 仿真、0–100）
  DiffusionDrive  NAVSIM navtest         PDMS           88.1   （伪闭环、真实数据重放、0–100）
两者的基准、域、指标定义、评分区间全部不同 ⇒ **6.87 与 88.1 之间不存在有意义的大小关系**。
这本身就是核心主张的第一个论据：连"排名"都无法在公开分数上定义，
而 G/F/I/C 四轴是在**同一份刺激集、同一套口径**下对两个模型测出来的，因而可以对齐比较。
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")

PUBLIC = {
    "SimLingo": {"benchmark": "CARLA Leaderboard 2.0", "metric": "Driving Score",
                 "value": 6.87, "range": "0–100", "protocol": "闭环仿真（CARLA）",
                 "native_domain": "CARLA (sim)"},
    "DiffusionDrive": {"benchmark": "NAVSIM navtest", "metric": "PDMS",
                       "value": 88.1, "range": "0–100", "protocol": "伪闭环（真实数据重放，non-reactive）",
                       "native_domain": "NAVSIM / OpenScene (real)†"},
}


def g(dct, *keys, default=None):
    for k in keys:
        if dct is None:
            return default
        dct = dct.get(k) if isinstance(dct, dict) else None
    return default if dct is None else dct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=str(RES / "h1_three_line_evidence.json"))
    ap.add_argument("--out-md", default=str(RES / "h1_three_line_evidence.md"))
    args = ap.parse_args()

    J = {}
    for k, f in (("g", "g_positive_calibration_diffusiondrive.json"),
                 ("g_sl", "b1_v_hazard_clean_vision.json"),
                 ("f_act", "f_axis_action_counterfactual.json"),
                 ("f_sl", "f_axis_ttc_gradient.json"),
                 ("f_dd", "f_axis_dd_steer.json"),
                 ("f_dd_pc", "f_axis_dd_steer_brakedd.json"),
                 ("f_dd_esc", "f_axis_dd_steer_escalate.json"),
                 ("i", "i_axis_domain.json"),
                 ("c_dd", "c_axis_concentration.json"),
                 ("c_sl", "c_axis_simlingo.json"),
                 ("c_shape", "c_axis_shape_diagnostics.json"),
                 ("e2", "e2_steering_verdicts.json"),
                 ("pilot", "p2_pilot_posttrain.json")):
        p = RES / f
        J[k] = json.load(open(p)) if p.exists() else None

    OUT = {"line1_public_leaderboard": PUBLIC, "line2_four_axis": {}, "line3_pilot_posttrain": {},
           "composite_score": None,
           "no_composite_rationale":
               "工单明文要求不做任意加权的单一复合分。四轴各自的单位、判定状态与可比性都不同："
               "G 与 C 是模型内归一化读数，F 在两个模型上因动作头架构不同而不可同量纲比较，"
               "I 的表征端与行为端给出相反排序。任何权重都会把这些结构性差异抹平成一个无法追责的数。"}

    # ---------------- 第二条线：四轴矩阵 ----------------
    gp = g(J["g"], "arms", "vision_mean") or {}
    ax = {}
    ax["G"] = {
        "readout": "CV-AUC(A vs 几何平衡负例 D2a) 减去自身 D2cV 证伪地板（vision_mean，scene 级 4 折 CV）",
        "comparable": True, "comparability_note": "同一刺激集、同一负例体系、同一选层与报数规则",
        "SimLingo": {"main_auc": g(J["g_sl"], "main", "auc"),
                     "minus_floor": g(J["g_sl"], "main_minus_floor"),
                     "minus_floor_ci95": g(J["g_sl"], "main_minus_floor_bootstrap", "ci95"),
                     "stability_10seed": g(J["g_sl"], "stability_across_cv_seeds", "diff_mean"),
                     "verdict": "不可估"},
        "DiffusionDrive": {"main_auc": g(gp, "main_D2a", "auc"),
                           "minus_floor": g(gp, "main_minus_floor"),
                           "minus_floor_ci95": g(gp, "main_minus_floor_bootstrap", "ci95"),
                           "stability_10seed": g(gp, "stability_across_cv_seeds", "diff_mean"),
                           "verdict": "不可估"},
        "verdict": "两模型均不可估；且两种池化口径给出相反排序 ⇒ 本轮 G 不支持跨模型排序"}

    fa = g(J["f_act"], "models") or {}
    ax["F"] = {
        "readout": "① 行动层反事实测试 b-AUC(A vs D2a)（架构中立，协议 §3.5 指定的 F 主验证）；"
                   "② steering 通路利用率（架构相关，见可比性说明）",
        "comparable": False,
        "comparability_note":
            "① 可比（读出对象是动作，不依赖动作头形式）；② **不可比**：SimLingo 是连续回归头，"
            "±α 注入可产生可测的纵向响应；DiffusionDrive 是 anchored 扩散头，"
            "即便注入它自己的行为定义轴 v_brake_dd（held-out AUC 0.656）也推不动纵向输出 ⇒ "
            "steering 效应量在两模型间不同量纲，不得同表排序。",
        "SimLingo": {
            "action_counterfactual_b_auc": g(fa, "SimLingo", "b_auc_A_vs_D2a", "auc"),
            "action_counterfactual_ci95": g(fa, "SimLingo", "b_auc_A_vs_D2a", "ci95_scene_bootstrap"),
            "b_mean_A": g(fa, "SimLingo", "b_mean_A", "mean"),
            "steering_slope_v_brake": -0.04054,
            "steering_measurable": True,
            "cos_obs_vs_injection": g(J["f_sl"], "cos_main"),
            "cos_ci95": g(J["f_sl"], "cos_ci95_scene_bootstrap"),
            "verdict": "不可估（双方法互证未建立；共线上界 |cos| ≤ 0.064）"},
        "DiffusionDrive": {
            "action_counterfactual_b_auc": g(fa, "DiffusionDrive", "b_auc_A_vs_D2a", "auc"),
            "action_counterfactual_ci95": g(fa, "DiffusionDrive", "b_auc_A_vs_D2a", "ci95_scene_bootstrap"),
            "b_mean_A": g(fa, "DiffusionDrive", "b_mean_A", "mean"),
            "steering_slope_v_hazard": g(J["f_dd"], "dose_slope_full", "mean"),
            "steering_slope_v_brake_dd_positive_control": g(J["f_dd_pc"], "dose_slope_full", "mean"),
            "steering_measurable": False,
            "verdict": "不可估（仪器侧：站内上界方向亦推不动纵向输出）"},
        "verdict": "两模型的行动层反事实读数均不可估；steering 口径在两模型间不可比"}

    ii = g(J["i"], "summary_I_m") or {}
    ax["I"] = {
        "readout": "I_m = 1 − D_{L*}（表征端）与 行为端域敏感度 E|Δv_cmd|/mean（同一域配对）",
        "comparable": True,
        "comparability_note": "两者都是模型内归一化的比值（协议 §4 规则 1）；但剖面形态差异大，排序为条件结论",
        "SimLingo": {"I_m": g(ii, "simlingo", "I_m"), "D_at_peak": g(ii, "simlingo", "D_at_peak"),
                     "D_ci95": g(ii, "simlingo", "D_ci95"),
                     "behavioural_domain_sensitivity": g(ii, "simlingo", "behavioral_domain_sensitivity")},
        "DiffusionDrive": {"I_m": g(ii, "dd", "I_m"), "D_at_peak": g(ii, "dd", "D_at_peak"),
                           "D_ci95": g(ii, "dd", "D_ci95"),
                           "behavioural_domain_sensitivity": g(ii, "dd", "behavioral_domain_sensitivity")},
        "verdict": "表征端 SimLingo > DiffusionDrive；行为端 **相反**；两组 CI 均不重叠 ⇒ 单一标量无法承载"}

    cs = J["c_shape"] or {}
    ax["C"] = {
        "readout": "C_m = top-2 层 recovery 占比（协议 §3⑤），并附恢复剖面形状诊断",
        "comparable": False,
        "comparability_note":
            "**不可比**：C_m 的弥散基线随层数变化（8 层 0.250 vs 24 层 0.083），"
            "且 top-2 占比公式的前提是剖面存在内部峰。SimLingo 的剖面自 L0 单调递减（Spearman −0.997）⇒ "
            "公式前提不成立，其 C_m 不可读作集中度。可跨模型比较的是**形状量**（见下）。",
        "SimLingo": {"C_m": g(cs, "SimLingo", "C_m_top2_share"),
                     "diffuse_baseline": g(cs, "SimLingo", "diffuse_baseline"),
                     "profile_shape": g(cs, "SimLingo", "profile_shape"),
                     "spearman_layer_vs_recovery": g(cs, "SimLingo", "spearman_layer_vs_recovery"),
                     "layers_for_80pct_frac": g(cs, "SimLingo", "layers_for_80pct_frac"),
                     "argmax_layer_mode": g(cs, "SimLingo", "argmax_layer_mode"),
                     "argmax_entropy": g(cs, "SimLingo", "argmax_normalized_entropy"),
                     "patch_all_recovery": g(cs, "SimLingo", "patch_all_recovery"),
                     "verdict": "不可估（公式前提不成立）；实质结论：失效在视觉输入接口即已进入并向下游级联"},
        "DiffusionDrive": {"C_m": g(cs, "DiffusionDrive", "C_m_top2_share"),
                           "diffuse_baseline": g(cs, "DiffusionDrive", "diffuse_baseline"),
                           "profile_shape": g(cs, "DiffusionDrive", "profile_shape"),
                           "spearman_layer_vs_recovery": g(cs, "DiffusionDrive", "spearman_layer_vs_recovery"),
                           "layers_for_80pct_frac": g(cs, "DiffusionDrive", "layers_for_80pct_frac"),
                           "argmax_layer_mode": g(cs, "DiffusionDrive", "argmax_layer_mode"),
                           "argmax_entropy": g(cs, "DiffusionDrive", "argmax_normalized_entropy"),
                           "patch_all_recovery": g(cs, "DiffusionDrive", "patch_all_recovery"),
                           "verdict": "PASS（内部峰在 L6，top-2 占比 0.858 显著高于弥散基线 0.250）"},
        "verdict": "两模型的失效**进入位置不同**：SimLingo 在视觉输入接口（级联），DiffusionDrive 在深层融合段（内部峰）"}
    OUT["line2_four_axis"] = ax

    # ---------------- 第三条线：pilot post-train ----------------
    if J["pilot"]:
        p = J["pilot"]
        OUT["line3_pilot_posttrain"] = {
            "target_model": "SimLingo", "budget": "只训 speed_wps_head（229k 参数），其余全部冻结",
            "diagnosis_used": "F 轴：驱动动作的方向与编码危险的方向近乎正交",
            "intervention": "耦合辅助损失 λ·ReLU(Δ@1σ + margin)，Δ 与 steering 的 α=+1 剂量响应同量",
            "arms": p.get("arms"), "contrasts": p.get("contrasts"),
            "verdict": p.get("verdict"), "verdict_rule": p.get("verdict_rule"),
            "coupling_direction_heldout_rho": p.get("coupling_direction_heldout_rho_vs_abrake")}
    else:
        OUT["line3_pilot_posttrain"] = {"status": "未就绪"}

    Path(args.out_json).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))

    # ---------------- markdown 片段 ----------------
    L = ["# 三线头号证据表（公开榜单 / G-F-I-C 四轴 / 真实 post-train）", "",
         "> 生成脚本 `scripts/h1_three_line_table.py`；数值真源 `h1_three_line_evidence.json`。",
         "> **不做任意加权的单一复合分**——理由见表下「为什么不给一个总分」。", "",
         "## 第一条线：公开榜单分数（**两个分数不在同一把尺子上**）", "",
         "| Model | Benchmark | Metric | Score | Range | Protocol | Native domain |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for m, d in PUBLIC.items():
        L.append(f"| {m} | {d['benchmark']} | {d['metric']} | **{d['value']}** | {d['range']} | "
                 f"{d['protocol']} | {d['native_domain']} |")
    L += ["", "† DiffusionDrive 本仓库所用 ckpt（`diffusiondrive_sim_navhard.ckpt`）的训练域未经独立核实。", "",
          "**这两个数字之间不存在有意义的大小关系。** 基准不同（CARLA Leaderboard 2.0 vs NAVSIM navtest）、",
          "域不同（仿真 vs 真实数据重放）、协议不同（闭环 vs 非反应式伪闭环）、指标定义不同",
          "（Driving Score = 路线完成度 × 违规惩罚连乘 vs PDMS = 若干子分的加权组合）。",
          "把 6.87 与 88.1 并排放，唯一能读出的信息是：**公开分数体系连「排名」这件事都无法定义**。",
          "这正是本工作的出发点——若两个候选连排名都排不出来，选型就只能靠别的东西；",
          "而下面第二条线的四轴，是在**同一份 nuScenes 鬼探头刺激集、同一套负例体系与统计口径**下",
          "对两个模型分别测出来的，因而可以逐格对齐。", ""]

    L += ["## 第二条线：G/F/I/C 四轴矩阵（同一刺激集、同一口径）", "",
          "| Axis | Readout | SimLingo | DiffusionDrive | 跨模型可比 | Verdict |",
          "| --- | --- | --- | --- | --- | --- |"]
    def fmt(x, n=3):
        return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
    L.append(f"| **G** | 主读数 − 自身 D2cV 证伪地板 | {fmt(ax['G']['SimLingo']['minus_floor'])} "
             f"{np.round(ax['G']['SimLingo']['minus_floor_ci95'],3).tolist() if ax['G']['SimLingo']['minus_floor_ci95'] else ''} | "
             f"{fmt(ax['G']['DiffusionDrive']['minus_floor'])} "
             f"{np.round(ax['G']['DiffusionDrive']['minus_floor_ci95'],3).tolist() if ax['G']['DiffusionDrive']['minus_floor_ci95'] else ''} | "
             f"是 | 两者均不可估 |")
    L.append(f"| **F**① | 行动层反事实 b-AUC(A vs D2a) | {fmt(ax['F']['SimLingo']['action_counterfactual_b_auc'])} "
             f"{np.round(ax['F']['SimLingo']['action_counterfactual_ci95'],3).tolist()} | "
             f"{fmt(ax['F']['DiffusionDrive']['action_counterfactual_b_auc'])} "
             f"{np.round(ax['F']['DiffusionDrive']['action_counterfactual_ci95'],3).tolist()} | 是 | 两者均不可估 |")
    L.append(f"| **F**② | steering ±α 全阶梯斜率 (m/s per σ) | {fmt(ax['F']['SimLingo']['steering_slope_v_brake'],5)} "
             f"(v_brake) | {fmt(ax['F']['DiffusionDrive']['steering_slope_v_hazard'],5)} "
             f"(v_hazard_dd)；站内上界 v_brake_dd 亦仅 "
             f"{fmt(ax['F']['DiffusionDrive']['steering_slope_v_brake_dd_positive_control'],5)} | "
             f"**否**（动作头架构不同） | DiffusionDrive 侧仪器无分辨力 |")
    L.append(f"| **I** | I_m = 1 − D_{{L*}}（表征端） | {fmt(ax['I']['SimLingo']['I_m'])} | "
             f"{fmt(ax['I']['DiffusionDrive']['I_m'])} | 是 | SimLingo > DD |")
    L.append(f"| **I** | 行为端域敏感度 | {fmt(ax['I']['SimLingo']['behavioural_domain_sensitivity'])} | "
             f"{fmt(ax['I']['DiffusionDrive']['behavioural_domain_sensitivity'])} | 是 | **DD > SimLingo（相反）** |")
    L.append(f"| **C** | C_m = top-2 层 recovery 占比 | {fmt(ax['C']['SimLingo']['C_m'])} "
             f"(基线 {fmt(ax['C']['SimLingo']['diffuse_baseline'])}) | {fmt(ax['C']['DiffusionDrive']['C_m'])} "
             f"(基线 {fmt(ax['C']['DiffusionDrive']['diffuse_baseline'])}) | **否**（层数不同/公式前提不同） | 见形状量 |")
    L.append(f"| **C** | 恢复剖面 Spearman(层号, recovery) | {fmt(ax['C']['SimLingo']['spearman_layer_vs_recovery'])}"
             f"（级联） | {fmt(ax['C']['DiffusionDrive']['spearman_layer_vs_recovery'])}（内部峰） | 是 | "
             f"失效**进入位置不同** |")
    L.append(f"| **C** | 责任层 argmax 归一化熵 | {fmt(ax['C']['SimLingo']['argmax_entropy'])} "
             f"(众数 L{ax['C']['SimLingo']['argmax_layer_mode']}) | {fmt(ax['C']['DiffusionDrive']['argmax_entropy'])} "
             f"(众数 L{ax['C']['DiffusionDrive']['argmax_layer_mode']}) | 是 | SimLingo 更集中 |")
    L.append("")

    if J["pilot"]:
        p = J["pilot"]; a = p["arms"]; c = p["contrasts"]
        L += ["## 第三条线：pilot post-train（SimLingo，固定预算：只训 speed_wps_head）", "",
              "| Arm | 训练目标 | b-AUC(A vs D2a)，held-out | 95% CI | Δ@1σ (m/s) | cos(g, v̂_hazard) |",
              "| --- | --- | --- | --- | --- | --- |"]
        lab = {"A0_baseline": "不训练（基线）", "A1_task_only": "任务项 + 蒸馏正则（λ=0，归因对照）",
               "A2_task_plus_coupling": "任务项 + 蒸馏正则 + **耦合项**"}
        for k in ("A0_baseline", "A1_task_only", "A2_task_plus_coupling"):
            r = a[k]
            L.append(f"| {k} | {lab[k]} | {r['b_auc_A_vs_D2a_heldout']:.4f} | "
                     f"{np.round(r['b_auc_ci95'],4).tolist()} | {r['delta_at_1sigma_heldout_mean']:+.4f} | "
                     f"{r['cos_g_vhazard_heldout_mean']:+.4f} |")
        L += ["", f"**对比**：b-AUC A2−A0 = {c['b_auc_A2_minus_A0']['mean']:+.4f} "
                  f"{np.round(c['b_auc_A2_minus_A0']['ci95'],4).tolist()}；"
                  f"A2−A1（归因）= {c['b_auc_A2_minus_A1']['mean']:+.4f} "
                  f"{np.round(c['b_auc_A2_minus_A1']['ci95'],4).tolist()}；"
                  f"A1−A0 = {c['b_auc_A1_minus_A0']['mean']:+.4f}",
              f"**耦合量**：Δcos(A2−A0) = {c['coupling_cos_A2_minus_A0']:+.4f}；"
              f"ΔΔ@1σ(A2−A0) = {c['coupling_delta_A2_minus_A0']:+.4f} m/s；"
              f"Δcos(A1−A0) = {c['coupling_cos_A1_minus_A0']:+.4f}", "",
              f"**判定**：{p['verdict']}", ""]

    L += ["## 为什么不给一个总分", "", OUT["no_composite_rationale"], "",
          "具体到本表：G 与 I 的读数是模型内归一化的、可跨模型对齐的；",
          "F 的 steering 口径在两个模型间**不同量纲**（连续回归头 vs anchored 扩散头）；",
          "C 的 top-2 占比公式在 SimLingo 上**前提不成立**（剖面单调递减）。",
          "把这四个格子加权成一个数，等于把「不可比」和「不可估」当作 0 或当作中位数来处理，",
          "而这两种处理都会让最终排名的来源无法追责。"]
    Path(args.out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\n[H1] wrote {args.out_json} + {args.out_md}")


if __name__ == "__main__":
    main()
