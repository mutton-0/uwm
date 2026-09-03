"""F-3|必要性比 R 的估计量 B（汇总均值之比）—— 小范围确认：LTF 与 SimLingo。

工单：2026-09-03（第三份）。**纯重新分析**已落盘的速度版 F-3 结果，不跑任何推理。

## 两个估计量

R 的**逐事件定义**不变：$R_i = 1 - b_{occ,i}/b_{ghost,i}$。差别在怎么把它汇总：

* **估计量 A（现行，论文在用）**：逐事件先算 $R_i$，再对 $R_i$ 取 scene 级 bootstrap 均值。
  估的是「各事件必要性比的平均」$\;\\mathbb{E}[R_i]$。
  **弱点**：$b_{ghost,i}$ 逐事件可以很小 ⇒ $R_i$ 重尾，少数事件主导均值
  （分母守卫 $|b_{ghost}|\\ge$ min_b 只是部分缓解）。
* **估计量 B（本轮确认）**：先分别求 $\\overline{b_{occ}}$ 与 $\\overline{b_{ghost}}$，再算一次比值：

  $$R^{B} \;=\; 1-\\frac{\\overline{b_{occ}}}{\\overline{b_{ghost}}}
             \;=\; \\frac{\\overline{b_{ghost}}-\\overline{b_{occ}}}{\\overline{b_{ghost}}}$$

  估的是「平均响应里被遮挡移除的比例」$\;1-\\mathbb{E}[b_{occ}]/\\mathbb{E}[b_{ghost}]$。
  更贴合 PASS 门槛「遮挡要移除**过半**的响应」的字面意思，且不受个别事件小分母拉偏。

$\\mathbb{E}[X/Y]\\neq\\mathbb{E}[X]/\\mathbb{E}[Y]$，故两者一般不相等。

## 分母守卫在估计量 B 下是否还需要

估计量 A 的分母守卫是为了压住**逐事件**小分母；估计量 B 的分母是**汇总均值**，
不存在该问题 ⇒ 守卫在原理上不再必要。本模块两种口径都算：

* **B-gated**：沿用 A 的事件集（$|b_{ghost,i}|\\ge$ min_b）—— 与 A **逐事件同一批**，
  是 A/B 的 like-for-like 对照，也是与上一轮顺手对比的**复核口径**；
* **B-all**：用全部事件 —— 估计量 B 的**干净定义**。

两者判定不一致时如实并列，不合并。

## 不变的部分

三态门槛（$R$ 的 scene 级 CI 相对 0.5 的位置）、开门条件（$b_{ghost}$ 显著）、
ctrl 对照臂（同样用估计量 B 算一份 $R_{ctrl}$，应接近 0）—— 一条没放松。
bootstrap 一律 scene 级、5000 次、与既有实现同一套重抽方式。
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def ratio_of_means(recs, num_key, den_key="b_ghost", n=5000, seed=0):
    """R^B = mean(num)/mean(den)，scene 级 bootstrap（两个均值共用同一次重抽）。

    num 传 (b_ghost − b_occ) 或 (b_ghost − b_ctrl)；den 传 b_ghost。
    共用重抽是必需的：分子分母来自同一批事件，独立重抽会白扔配对功效（§FE/A66）。
    """
    d = defaultdict(lambda: ([], []))
    for r in recs:
        a, b = r[num_key], r[den_key]
        if np.isfinite(a) and np.isfinite(b):
            d[r["scene"]][0].append(a); d[r["scene"]][1].append(b)
    keys = sorted(d)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(keys), len(keys))
        A = [x for i in idx for x in d[keys[i]][0]]
        B = [x for i in idx for x in d[keys[i]][1]]
        if not A or not B or abs(np.mean(B)) < 1e-12:
            continue
        vals.append(np.mean(A) / np.mean(B))
    allA = [x for k in keys for x in d[k][0]]
    allB = [x for k in keys for x in d[k][1]]
    return {"mean": float(np.mean(allA) / np.mean(allB)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            "mean_numerator": float(np.mean(allA)), "mean_denominator": float(np.mean(allB)),
            "n_events": len(allA), "n_scenes": len(keys), "n_boot": len(vals)}


def verdict_of(R):
    if R is None:
        return "不可估：样本不足"
    c = R["ci95"]
    return ("PASS：遮住关键实体后动作显著退回基线" if c[0] > 0.5 else
            ("FAIL：遮住关键实体后动作基本不变（盲目泛化签名）" if c[1] < 0.5 else
             "不可估：必要性比的 scene 级 CI 跨 0.5"))


def tag(R):
    return "—" if R is None else ("PASS" if R["ci95"][0] > 0.5 else
                                  ("FAIL" if R["ci95"][1] < 0.5 else "不可估"))


def fmt(R):
    return "—" if not R else f"{R['mean']:+.4f} [{R['ci95'][0]:+.4f}, {R['ci95'][1]:+.4f}]"


CELLS = [("G1 × LTF", "f3_occlusion_ltf.json"),
         ("NAVSIM × LTF", "f3_occlusion_navsim_ltf.json"),
         ("G1 × SimLingo", "f3_occlusion_simlingo.json")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None,
                    help="形如 '名称:文件名'；缺省为本轮工单指定的三格")
    ap.add_argument("--out", default=str(RES / "f3_restimator_b.json"))
    args = ap.parse_args()
    cells = ([tuple(c.split(":", 1)) for c in args.cells] if args.cells else CELLS)

    rows = []
    for name, fn in cells:
        p = RES / fn
        if not p.exists():
            print(f"[RB] 缺 {fn}，跳过"); continue
        d = json.load(open(p))
        pe = d["per_event"]
        gate = float(d.get("min_b_gate", 0.02))
        # 逐事件补出估计量 B 需要的分子
        for r in pe:
            r["num_occ"] = r["b_ghost"] - r["b_occ"]
            r["num_ctrl"] = r["b_ghost"] - r["b_ctrl"]
        gated = [r for r in pe if abs(r["b_ghost"]) >= gate]

        row = {"cell": name, "src": fn, "n_events": len(pe), "min_b_gate": gate,
               "n_gated": len(gated),
               "baseline_response_significant": bool(d.get("baseline_response_significant")),
               "b_ghost": d.get("b_ghost"),
               "estimator_A": {"R": d.get("necessity_ratio"),
                               "R_ctrl": d.get("necessity_ratio_control"),
                               "n_used": d.get("n_used_for_R"),
                               "verdict": d.get("verdict")}}
        for key, recs in (("estimator_B_gated", gated), ("estimator_B_all", pe)):
            R = ratio_of_means(recs, "num_occ")
            Rc = ratio_of_means(recs, "num_ctrl")
            row[key] = {"R": R, "R_ctrl": Rc, "n_used": len(recs),
                        "verdict": (verdict_of(R) if row["baseline_response_significant"]
                                    else "不可估：基线响应本身与 0 不可区分（门未开）")}
        row["verdict_tags"] = {"A": tag(row["estimator_A"]["R"]),
                               "B_gated": tag(row["estimator_B_gated"]["R"]),
                               "B_all": tag(row["estimator_B_all"]["R"])}
        row["A_vs_B_gated_same_verdict"] = row["verdict_tags"]["A"] == row["verdict_tags"]["B_gated"]
        row["B_gated_vs_B_all_same_verdict"] = (row["verdict_tags"]["B_gated"]
                                                == row["verdict_tags"]["B_all"])
        rows.append(row)

        bg = row["b_ghost"]
        print(f"\n=== {name}  (n={len(pe)}，过守卫 {len(gated)}，守卫 {gate}) ===")
        print(f"  b_ghost {fmt(bg)}  {'显著 ⇒ 门开' if row['baseline_response_significant'] else '跨0 ⇒ 门关'}")
        for k, lab in (("estimator_A", "估计量A 逐事件比值均值"),
                       ("estimator_B_gated", "估计量B 汇总均值之比（沿用 A 的事件集）"),
                       ("estimator_B_all", "估计量B 汇总均值之比（全部事件）")):
            e = row[k]
            print(f"  {lab:34s} R={fmt(e['R']):34s} [{tag(e['R']):>5s}]  "
                  f"R_ctrl={fmt(e['R_ctrl'])}  n={e['n_used']}")
        print(f"  A vs B(同事件集) 判定一致: {row['A_vs_B_gated_same_verdict']}   "
              f"B(同事件集) vs B(全量) 判定一致: {row['B_gated_vs_B_all_same_verdict']}")
        print(f"  估计量B 最终判定：{row['estimator_B_gated']['verdict']}")

    out = {"design": "F-3 必要性比 R 的估计量 B（汇总均值之比）小范围确认：LTF 与 SimLingo",
           "no_new_inference": True,
           "estimator_A": "逐事件先算 R_i = 1 − b_occ,i/b_ghost,i，再取 scene 级 bootstrap 均值",
           "estimator_B": "R = 1 − mean(b_occ)/mean(b_ghost)，分子分母共用同一次 scene 级重抽",
           "threshold": "三态门槛不变：R 的 scene 级 CI 相对 0.5 的位置；开门条件仍是 b_ghost 显著",
           "cells": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[RB] wrote {args.out}")


if __name__ == "__main__":
    main()
