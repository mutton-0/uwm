"""与池子大小无关的样本量需求。

## 为什么需要这个脚本（fig_convergence.py 的判据错在哪）
fig_convergence 用「无放回子采样重算，看 |θ̂_n − θ̂_N| 何时降到 h 以下」。
问题是无放回子采样自带有限总体修正：

    Var(θ̂_n − θ̂_N) = σ²/n · (1 − n/N)，  而 h ≈ 1.96·σ/√N
  ⇒ err_q/h ≈ (z_q/1.96)·√(N/n − 1)      —— **只依赖 n/N**

于是"跨过 1h 的 n"必然是 N 的一个固定比例（实测 0.413–0.474，18 个格子，
跨 4 根轴、6 个候选、池子 72 与 316，比例都一样）。它回答的是
"抽掉一半数据答案变不变"（一个有用的自洽检验），**不是"要采多少数据"**。

## 这里用的口径
σ = SE_N·√N 是**每场景**的尺度，与池子大小无关；外推用 SE_n = σ/√n，
不含 FPC。两个各带 SE_n 的独立读数要在 95% 下分开，需 |Δ| ≥ 1.96·√2·SE_n，
即 n ≥ (2.77·σ/Δ)²。

前提：场景间独立同分布、SE ∝ n^{-1/2}。对比值型估计量在小 n 下只是近似，
但**向上外推**（从 N≈300 推到上千）是标准做法。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
TAB = Path("/data/ruolin/uwm/sim2real_demo_ttc/paper_v2.3_icra/tables")
NM = {"simlingo": "SimLingo", "dd": "DiffusionDrive", "ltf": "LTF",
      "ddv2": "DiffusionDriveV2", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
ORDER = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
SHORT = {"F-3  $r$": "F-3 $r$", "G-VS  selectivity": "G-VS",
         "C  $C_m$": "C $C_m$", "I  $I_m$": "I $I_m$"}
Z2 = 1.96 * np.sqrt(2.0)          # 两个独立读数相差 Δ 的 95% 可分辨门槛


def main():
    D = json.load(open(RES / "convergence_curves.json"))
    DL = json.load(open(RES / "convergence_ci.json"))["effect_spread"]
    rows, per_axis = [], {}
    for c in D["curves"]:
        N = c["n"][-1]
        se = c["h"] / 1.96                      # h 是 95% 半宽
        sig = se * np.sqrt(N)                   # 每场景尺度
        d = DL[c["axis"]]
        n_req = float((Z2 * sig / d) ** 2)
        rows.append({"model": c["model"], "axis": c["axis"], "N": N,
                     "final": c["final"],
                     "se_N": se, "sigma": sig, "delta": d, "n_rank": n_req,
                     "frac_1h": None})
        per_axis.setdefault(c["axis"], []).append(n_req)

    # 顺带把"n@1h 是 N 的固定比例"这件事量化，作为判据失效的证据
    fr = []
    for c in D["curves"]:
        n = np.array(c["n"]); e = np.array(c["err90_h"])
        k = [i for i in range(len(n)) if (e[i:] <= 1.0).all()]
        if k:
            fr.append(n[k[0]] / n[-1])
    diag = {"n_at_1h_over_N": {"min": float(min(fr)), "max": float(max(fr)),
                               "median": float(np.median(fr)),
                               "cv": float(np.std(fr) / np.mean(fr)),
                               "cells": len(fr)}}

    (RES / "sample_size.json").write_text(json.dumps(
        {"rows": rows, "per_axis_n_rank": {k: [min(v), max(v)]
                                           for k, v in per_axis.items()},
         "diagnostic": diag}, indent=1, ensure_ascii=False))

    print(f"n@1h / N: {min(fr):.3f}–{max(fr):.3f}  中位 {np.median(fr):.3f}  "
          f"CV {np.std(fr)/np.mean(fr):.3f}  ({len(fr)} 格)")
    print(f"\n{'候选 / 轴':<32}{'sigma':>8}{'SE_N':>9}{'Delta':>9}{'n(rank)':>9}{'N':>6}")
    print("-" * 74)
    for r in sorted(rows, key=lambda z: (ORDER.index(z["model"]), z["axis"])):
        print(f"{NM[r['model']]+' / '+SHORT[r['axis']]:<32}{r['sigma']:8.3f}"
              f"{r['se_N']:9.4f}{r['delta']:9.4f}{r['n_rank']:9.0f}{r['N']:6d}")

    # ---- LaTeX 表：按轴汇总，这是论文里真正该报的数 ----
    TAB.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Axis & $\Delta$ & $\sigma$ range & scenes to rank & have \\",
             r"\midrule"]
    NPOOL = {"F-3  $r$": 303, "G-VS  selectivity": 316,
             "C  $C_m$": 313, "I  $I_m$": 72}
    for ax in ("F-3  $r$", "G-VS  selectivity", "C  $C_m$", "I  $I_m$"):
        v = per_axis[ax]
        sg = [r["sigma"] for r in rows if r["axis"] == ax]
        lo, hi = min(v), max(v)
        lines.append(f"{SHORT[ax]} & {DL[ax]:.4f} & "
                     f"{min(sg):.2f}--{max(sg):.2f} & "
                     f"{lo:.0f}--{hi:.0f} & {NPOOL[ax]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_samplesize.tex").write_text("\n".join(lines) + "\n")
    print(f"\n[TAB] -> {TAB/'tab_samplesize.tex'}")


if __name__ == "__main__":
    main()
