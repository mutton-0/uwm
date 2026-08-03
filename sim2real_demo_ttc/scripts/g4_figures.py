"""G4 出图（手册 §9.4）：层剖面、投影-行为散点、估计 vs 真值分层对比、行为量分布。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文标签需要 CJK 字体，否则 matplotlib 画成方框
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import hazard_directions, load_cache, project, select_peak_layer, subset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--pool-mode", default="vision_mean")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    res = work / "results"
    fig_dir = res / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    m = json.loads((res / f"metrics_estimate_{args.pool_mode}.json").read_text())
    items = load_cache(work, args.pool_mode)
    splits = json.loads((work / "mining" / "splits.json").read_text())
    est = subset(items, (work / "mining" / "split_estimate.txt").read_text().split())
    truth = subset(items, (work / "mining" / "split_truth.txt").read_text().split())
    S = {k: [e for e in est if e["scene"] in splits["probe_split_scenes"][k]] for k in ("dir", "sel", "test")}
    v_haz, evr, _ = hazard_directions(S["dir"])
    peak, rho_layer = select_peak_layer(S["sel"], v_haz)

    # ---- 图1：层剖面（探针精度 & EVR1；D_L 因 Tier-S 砍掉域方向而缺席）----
    fig, ax = plt.subplots(figsize=(8, 4))
    L = len(rho_layer)
    ax.plot(range(L), rho_layer, "-o", ms=4, label=r"$\rho_{TTC}$ on $S_{sel}$")
    ax.axvline(peak, color="r", ls="--", label=f"peak layer L*={peak}")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("decoder layer"); ax.set_ylabel(r"Spearman $\rho$(proj, TTC)")
    ax2 = ax.twinx(); ax2.plot(range(L), evr, "-s", ms=3, color="tab:green", alpha=0.6, label="PCA EVR1")
    ax2.set_ylabel("EVR1"); ax2.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    ax.set_title(f"层剖面 ({args.pool_mode})  |  D_L 曲线: Tier-S 未做域方向 (V5=N/A)", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / f"layer_profile_{args.pool_mode}.png", dpi=110); plt.close(fig)

    # ---- 图2：投影-行为散点（S_test / truth 两块 held-out）----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, evs, tag in ((axes[0], S["test"], "S_test (主读数)"), (axes[1], truth, "truth (完全 held-out)")):
        p = project(evs, v_haz, peak)
        b = np.array([e["b"] for e in evs])
        for t, col in (("A", "tab:red"), ("B", "tab:orange"), ("C", "tab:purple"), ("D", "tab:blue")):
            k = [i for i, e in enumerate(evs) if e["type"] == t]
            if k:
                ax.scatter(p[k], b[k], c=col, label=f"{t} (n={len(k)})", s=34, alpha=0.85)
        ax.axhline(cfg["metrics"]["b_min_primary"], color="gray", ls=":", label=f"b_min={cfg['metrics']['b_min_primary']}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(f"projection on $v_{{hazard}}$ (L*={peak})"); ax.set_ylabel("b = v_plan(clean) - v_plan(ghost) [m/s]")
        ax.set_title(tag, fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle(f"投影-行为相关 ({args.pool_mode})", fontsize=11)
    fig.tight_layout(); fig.savefig(fig_dir / f"projection_behavior_{args.pool_mode}.png", dpi=110); plt.close(fig)

    # ---- 图3：估计 vs 真值 分层达标率 ----
    prim = m["behavior_scores"][str(m["b_min_primary"])]
    common = sorted(set(prim["estimate_strata"]) & set(prim["truth_strata"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.arange(len(common))
    ax.bar(x - 0.2, [prim["estimate_strata"][k]["rate"] for k in common], 0.4, label="估计集")
    ax.bar(x + 0.2, [prim["truth_strata"][k]["rate"] for k in common], 0.4, label="真值集")
    ax.set_xticks(x); ax.set_xticklabels(common, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(f"达标率 (b > {m['b_min_primary']} m/s)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"V2 分层对比 (共同层 n={len(common)}, Spearman={m['verdicts']['V2_spearman']:.2f})", fontsize=10)

    ax = axes[1]
    grid = [str(b) for b in cfg["metrics"]["b_min_mps"]]
    er = [m["behavior_scores"][g]["estimate_rate"] for g in grid]
    lo = [m["behavior_scores"][g]["estimate_ci95"][0] for g in grid]
    hi = [m["behavior_scores"][g]["estimate_ci95"][1] for g in grid]
    tr = [m["behavior_scores"][g]["truth_rate"] for g in grid]
    xx = np.arange(len(grid))
    ax.errorbar(xx, er, yerr=[np.array(er) - np.array(lo), np.array(hi) - np.array(er)],
                fmt="o-", capsize=4, label="估计集 ±95% CI (scene bootstrap)")
    ax.plot(xx, tr, "s--", color="tab:red", label="真值集")
    ax.set_xticks(xx); ax.set_xticklabels([f"b_min={g}" for g in grid])
    ax.set_ylabel("达标率"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("V1 总分一致性 + b_min 敏感性", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / f"consistency_{args.pool_mode}.png", dpi=110); plt.close(fig)

    # ---- 图4：行为量 b 与 TTC 分布 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    data, labels = [], []
    for t in "ABCD":
        d = [e["b"] for e in items.values() if e["type"] == t]
        if d:
            data.append(d); labels.append(f"{t}\n(n={len(d)})")
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0, color="k", lw=0.6); ax.axhline(cfg["metrics"]["b_min_primary"], color="r", ls=":")
    ax.set_ylabel("b = 减速响应 [m/s]"); ax.set_title("各类事件的行为响应", fontsize=10); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    stats_ = json.loads((work / "mining" / "mining_stats.json").read_text())
    edges, counts = stats_["ttc_hist_edges"], stats_["ttc_hist_counts"]
    ax.bar(range(len(counts)), counts)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))], rotation=30, fontsize=8)
    ax.set_xlabel("min TTC in 1s window [s]"); ax.set_ylabel("事件数"); ax.set_title("挖掘 TTC 分布", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / f"distributions_{args.pool_mode}.png", dpi=110); plt.close(fig)

    print(f"[G4-fig] wrote 4 figures -> {fig_dir}")


if __name__ == "__main__":
    main()
