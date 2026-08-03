"""Q4 prompt_anchor 敏感性：锚定 clean / ghost / 两帧均值，长间隔臂的三个信号是否稳。

背景：手册 §10.4 要求 clean/ghost 之间 prompt 完全一致，而 prompt 含 `Current speed: X m/s`。
锚定 clean 意味着 ghost 帧的速度条件是"错的"（写的是 clean 时刻的速度）。
差分读数（δ 与 b 都是两条件之差）理论上对锚点的选择不敏感——但这需要验证，
因为锚点会改变**两个条件共同的**工作点（模型在不同速度下的规划行为不同）。

三个信号（均在 truth holdout 上，主设定 supervised + none + vision_mean）：
  1. 表征 AUC(正例 vs D)
  2. ρ(投影, 行为)
  3. 行为 b-AUC(A 类 vs D)
另报 corr(Δego, b) 作为"污染是否复现"的对照。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import (hazard_directions, hazard_directions_supervised, load_cache,  # noqa: E402
                        project, select_peak_layer, subset, time_baseline_basis)


def analyse(cfg_path, pool_mode="vision_mean"):
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    mcfg = cfg["metrics"]
    work = Path(cfg["paths"]["work_dir"])
    items = load_cache(work, pool_mode)
    splits = json.loads((work / "mining" / "splits.json").read_text())
    est = subset(items, (work / "mining" / "split_estimate.txt").read_text().split())
    truth = subset(items, (work / "mining" / "split_truth.txt").read_text().split())
    S = {k: [e for e in est if e["scene"] in splits["probe_split_scenes"][k]] for k in ("dir", "sel", "test")}

    basis = None
    if mcfg.get("deconfound", "none") == "d_baseline":
        basis, _ = time_baseline_basis(S["dir"], k=int(mcfg.get("deconfound_k", 2)))
    if mcfg.get("direction_method", "pca") == "supervised":
        v, _, _ = hazard_directions_supervised(S["dir"], basis=basis)
    else:
        v, _, _ = hazard_directions(S["dir"], basis=basis)
    peak, _ = select_peak_layer(S["sel"], v, basis=basis)

    proj = project(truth, v, peak, basis=basis)
    b = np.array([e["b"] for e in truth])
    d = np.array([e["d_ego"] for e in truth])
    pos = np.array([e["is_positive"] for e in truth])
    is_a = np.array([e["type"] == "A" for e in truth])

    u1 = stats.mannwhitneyu(proj[pos], proj[~pos])
    r2 = stats.spearmanr(proj, b)
    u3 = stats.mannwhitneyu(b[is_a], b[~pos])
    r4 = stats.pearsonr(d, b)
    return {
        "anchor": cfg["model"].get("prompt_anchor", "clean"),
        "peak": peak, "n": len(truth), "n_pos": int(pos.sum()),
        "auc_rep": float(u1.statistic / (pos.sum() * (~pos).sum())), "p_rep": float(u1.pvalue),
        "rho_pb": float(r2.statistic), "p_pb": float(r2.pvalue),
        "auc_bA": float(u3.statistic / (is_a.sum() * (~pos).sum())), "p_bA": float(u3.pvalue),
        "corr_dego": float(r4.statistic), "p_dego": float(r4.pvalue),
        "b_sd": float(b.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--pool-mode", default="vision_mean")
    args = ap.parse_args()

    rows = [analyse(c, args.pool_mode) for c in args.configs]
    print(f"\nQ4 prompt_anchor 敏感性（长间隔臂，truth holdout，{rows[0]['n']} 事件 / {rows[0]['n_pos']} 正）")
    print(f"{'anchor':>8} {'L*':>3} | {'表征AUC(正vsD)':>16} | {'ρ(投影,行为)':>16} | "
          f"{'行为AUC(A vs D)':>17} | {'corr(Δego,b)':>14} | {'sd(b)':>6}")
    print("-" * 104)
    for r in rows:
        print(f"{r['anchor']:>8} {r['peak']:>3} | {r['auc_rep']:8.3f} (p={r['p_rep']:.2g})".ljust(46)
              + f"| {r['rho_pb']:+8.3f} (p={r['p_pb']:.3f}) ".ljust(28)
              + f"| {r['auc_bA']:8.3f} (p={r['p_bA']:.4f}) ".ljust(30)
              + f"| {r['corr_dego']:+7.3f} ".ljust(16) + f"| {r['b_sd']:6.3f}")
    a = np.array([r["auc_rep"] for r in rows])
    p = np.array([r["rho_pb"] for r in rows])
    print(f"\n跨锚点极差: 表征AUC {a.max()-a.min():.3f}   ρ(投影,行为) {p.max()-p.min():.3f}")
    print(f"符号一致性: 表征AUC 全部 >0.5 = {bool((a > 0.5).all())}；ρ 全部同号 = "
          f"{bool((p > 0).all() or (p < 0).all())}")
    Path(args.configs[0]).parent.parent.joinpath("results", "q4_anchor_sensitivity.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
