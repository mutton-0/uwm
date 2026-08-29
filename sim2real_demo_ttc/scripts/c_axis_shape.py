"""C 轴|恢复剖面的**形状诊断**，并据此判定 top-2 占比公式是否适用（两个模型共用）。

**为什么需要这一步（本轮新发现，见 amendments.md HL/A24）**：
协议 §3⑤ 的 C_m = top-2 层 recovery 占比 / Σ_L recovery，其**隐含前提是恢复剖面存在内部峰**
（失效在某一层"进入"，patch 该层恢复最多，patch 其他层恢复少）。
DiffusionDrive 的剖面确实如此（L0/L1 近 0，峰在 L6）。
但 SimLingo 的剖面是**从 L0 单调递减到 L23**——这是"级联"签名：
失效在最前端（视觉→LLM 接口）就已进入，patch 越早恢复越多，不存在内部责任层。
剖面单调递减时，top-2 占比量到的不是"集中度"，而只是"1/L 的一个倍数"，**不可读作集中度**。

故本脚本给出三个与 L 无关、可跨模型比较的形状量，并据此给 C_m 加一个适用性判据：
  ① ρ_shape = Spearman(层序号, recovery)  —— 强负 = 级联；≈0 或正 = 内部峰
  ② L80/L   = 达到 80% 恢复质量所需层数占总层数的比例（越小越集中）
  ③ H_argmax = 责任层 argmax 分布的归一化熵（越小 = 跨场景越一致）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
FILES = {"DiffusionDrive": RES / "c_axis_concentration.json",
         "SimLingo": RES / "c_axis_simlingo.json"}


def frac_layers_for(profile, q=0.8):
    s = np.sort(profile)[::-1]
    if s.sum() <= 0:
        return float("nan")
    return int(np.searchsorted(np.cumsum(s) / s.sum(), q) + 1)


def main():
    summary = {}
    for name, f in FILES.items():
        if not f.exists():
            continue
        d = json.load(open(f))
        p = np.array(d["recovery_profile_mean"], float)
        L = len(p)
        rho, pv = stats.spearmanr(np.arange(L), p)
        n80 = frac_layers_for(p)
        shape = ("级联（cascade）：剖面自最前层单调递减，失效在输入接口即已进入，无内部责任层"
                 if rho < -0.7 else
                 ("内部峰（localized）：存在内部责任层，top-2 占比可读作集中度" if rho > -0.3 else
                  "混合/不明确"))
        applicable = bool(rho > -0.7)
        diag = {"n_layers": L, "spearman_layer_vs_recovery": float(rho), "p": float(pv),
                "layers_for_80pct_mass": int(n80), "layers_for_80pct_frac": float(n80 / L),
                "argmax_normalized_entropy": d.get("argmax_normalized_entropy"),
                "argmax_layer_mode": d.get("argmax_layer_mode"),
                "recovery_at_first_layer": float(p[0]), "recovery_at_last_layer": float(p[-1]),
                "profile_shape": shape,
                "top2_share_formula_applicable": applicable,
                "C_m_top2_share": d.get("primary_clipped", {}).get("mean"),
                "C_m_ci95": d.get("primary_clipped", {}).get("ci95"),
                "diffuse_baseline": d.get("diffuse_baseline_top2_share"),
                "patch_all_recovery": (d.get("patch_all_recovery", {}) or {}).get("mean")}
        if not applicable:
            diag["verdict_override"] = ("不可估（公式前提不成立）：恢复剖面单调递减，"
                                        "top-2 占比不构成集中度证据；实质结论改述为"
                                        "「失效在视觉输入接口即已进入并向下游级联」")
        d["shape_diagnostics"] = diag
        f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        summary[name] = diag
        print(f"\n===== {name} （{L} 层）=====")
        print(f"  剖面 首层 {p[0]:.3f} -> 末层 {p[-1]:.3f}；Spearman(层号, recovery) = {rho:+.3f} (p={pv:.3g})")
        print(f"  形状：{shape}")
        print(f"  80% 恢复质量所需层数 = {n80}/{L} = {n80/L:.3f}；"
              f"责任层 argmax 众数 L{d.get('argmax_layer_mode')}，归一化熵 {d.get('argmax_normalized_entropy'):.3f}")
        print(f"  C_m(top-2 占比) = {diag['C_m_top2_share']:.3f}（弥散基线 {diag['diffuse_baseline']:.3f}）"
              f" -> top-2 公式{'适用' if applicable else '**不适用**'}")
        if not applicable:
            print(f"  判定改写：{diag['verdict_override']}")
    (RES / "c_axis_shape_diagnostics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[C-shape] wrote {RES}/c_axis_shape_diagnostics.json")


if __name__ == "__main__":
    main()
