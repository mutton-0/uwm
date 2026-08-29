"""T-C 辅助|失效 vs 正常的 t-SNE 可视化(**相关性证据,不能替代 patching 的因果结论**)。

工单 §5 步骤 2 明文要求:"必须明确标注为相关性证据"。
数据:T-I 已抽好的 DiffusionDrive 域配对激活(acts_dd.npz)。
着色:sim(CARLA 渲染,参考/正常) vs real(世界模型真实感,退化) —— 与 patching 的
参考/退化设定完全一致(FINDINGS 附2)。另用记号区分 12 个进入因果修补的场景。
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain/acts_dd.npz")
    ap.add_argument("--recovery", default="/data/ruolin/uwm/outputs/ghosthead_infer/patching/recovery.csv")
    ap.add_argument("--pool", default="vision_mean")
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 5, 6, 7])
    ap.add_argument("--out-png", default="/data/ruolin/uwm/sim2real_demo_ttc/results/figures/c_axis_tsne.png")
    ap.add_argument("--out-json", default="/data/ruolin/uwm/sim2real_demo_ttc/results/c_axis_tsne.json")
    args = ap.parse_args()

    import csv
    patched = {r["scene"] for r in csv.DictReader(open(args.recovery))}
    d = np.load(args.acts, allow_pickle=True)
    rows = {}
    for k in d.files:
        if k == "meta" or f"|{args.pool}|" not in k:
            continue
        sc, src, t, _, L = k.split("|")
        rows.setdefault(int(L[1:]), []).append((sc, src, int(t[1:]), d[k]))

    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(args.layers), figsize=(4.2 * len(args.layers), 4.2))
    stats_out = {}
    for ax, L in zip(np.atleast_1d(axes), args.layers):
        recs = sorted(rows[L], key=lambda r: (r[0], r[1], r[2]))
        X = np.stack([r[3] for r in recs]).astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        y = np.array([1 if r[1] == "real" else 0 for r in recs])
        mark = np.array([r[0] in patched for r in recs])
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        sil = float(silhouette_score(X, y))
        stats_out[f"L{L}"] = {"silhouette_sim_vs_real_fullspace": sil, "n": int(len(recs))}
        for lab, c, nm in ((0, "#3b7dd8", "sim = CARLA render (reference)"),
                           (1, "#d1495b", "real = world-model render (degraded)")):
            m = y == lab
            ax.scatter(Z[m & ~mark, 0], Z[m & ~mark, 1], s=12, c=c, alpha=.55, label=nm, linewidths=0)
            ax.scatter(Z[m & mark, 0], Z[m & mark, 1], s=34, facecolors="none", edgecolors=c, linewidths=1.1)
        ax.set_title(f"encoder L{L}   silhouette={sil:+.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle("DiffusionDrive: t-SNE of domain-paired activations "
                 "(open circles = the 12 scenes used in causal patching)\n"
                 "Correlational evidence only; does NOT replace the causal conclusion from activation patching",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    stats_out["note"] = ("t-SNE 与 silhouette 均为相关性/可分性证据；"
                         "散度大的层不等于因果责任层（本项目招牌结论：JS/CKA 峰在 L7，因果峰在 L6）。")
    stats_out["pool"] = args.pool
    Path(args.out_json).write_text(json.dumps(stats_out, indent=2, ensure_ascii=False))
    print(json.dumps(stats_out, indent=2, ensure_ascii=False))
    print(f"[T-C/t-SNE] wrote {args.out_png}")


if __name__ == "__main__":
    main()
