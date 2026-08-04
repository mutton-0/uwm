"""N1 交叉验证读数：scene 级 K-fold，把全部事件都用作 held-out，收紧功效边界。

单次 50/25/25 划分只有 ~60 正例进 S_test，只能排除 AUC>0.63。
K-fold 让每个事件都当过一次 held-out：方向与选层在其余折上拟合（嵌套，不看当前折），
汇总所有折的 held-out 投影后再算一次 AUC —— 有效样本量 = 全集。

D2c（无害性只由速度定义、单帧不可见）同样走这套流程，作为**噪声地板标定**：
若 D2a 与 D2c 的 CV-AUC 无法区分，则 D2a 的数值不构成类别特异性证据。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402
from n1_readout import auc, proj, supervised_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="vision_mean")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--direction", default="supervised", choices=["supervised", "random"],
                    help="random = 随机单位方向走同一套 CV+选层流程，量化选层带来的乐观偏差")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    items = load_cache(work, args.pool_mode)
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    TYPES = ("D", "D2a", "D2b", "D2bV", "D2c", "D2cV")
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in TYPES if (work / "mining" / f"matched_{t}.txt").exists()}
    VRU = ("human.", "vehicle.bicycle", "vehicle.motorcycle")

    by_type = defaultdict(list)
    for eid, e in items.items():
        t = evmap[eid]["event_type"]
        # D2cV/D2bV 是 D2c/D2b 的 VRU 子集，按各自的匹配清单单独归组
        for vt, src in (("D2cV", "D2c"), ("D2bV", "D2b")):
            if t == src and vt in matched and eid in matched[vt] \
                    and evmap[eid]["object_class"].startswith(VRU):
                by_type[vt].append(e)
        if t in matched and eid not in matched[t]:
            continue
        by_type[t].append(e)

    scenes = sorted({e["scene"] for v in by_type.values() for e in v})
    rng = np.random.default_rng(args.seed)
    fold_of = {s: int(i % args.folds) for i, s in enumerate(rng.permutation(scenes))}

    held = {t: [] for t in ("A", "D2a", "D2b", "D2bV", "D2c", "D2cV", "D")}
    peaks = []
    for f in range(args.folds):
        tr = lambda t: [e for e in by_type[t] if fold_of[e["scene"]] != f]          # noqa: E731
        te = lambda t: [e for e in by_type[t] if fold_of[e["scene"]] == f]          # noqa: E731
        pos_tr, neg_tr = tr("A"), tr("D2a")
        if len(pos_tr) < 20 or len(neg_tr) < 20:
            continue
        # 折内再切一小块选层（嵌套），不碰当前 held-out 折
        inner = sorted({e["scene"] for e in pos_tr})
        n_in = max(1, len(inner) // 4)
        sel_sc = set(inner[:n_in])
        fit_p = [e for e in pos_tr if e["scene"] not in sel_sc]
        fit_n = [e for e in neg_tr if e["scene"] not in sel_sc]
        sel_p = [e for e in pos_tr if e["scene"] in sel_sc]
        sel_n = [e for e in neg_tr if e["scene"] in sel_sc]
        if len(fit_p) < 15 or len(sel_p) < 5 or len(sel_n) < 5:
            fit_p, fit_n, sel_p, sel_n = pos_tr, neg_tr, pos_tr, neg_tr
        if args.direction == "random":
            rs = np.random.default_rng(1000 * args.seed + f)
            L, C = fit_p[0]["h_ghost"].shape
            v = rs.normal(size=(L, C)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        else:
            v = supervised_direction(fit_p, fit_n, seed=args.seed)
        a_by_l = np.array([auc(proj(sel_p, v, l), proj(sel_n, v, l))[0] for l in range(v.shape[0])])
        peak = int(np.nanargmax(a_by_l))
        peaks.append(peak)
        for t in held:
            held[t].append(proj(te(t), v, peak))

    def cat(t):
        return np.concatenate([x for x in held[t] if len(x)]) if held[t] else np.zeros(0)

    P = cat("A")
    print(f"[N1-CV] {args.folds} 折，pool={args.pool_mode}，各折峰层={peaks}")
    print(f"[N1-CV] held-out 正例 n={len(P)}\n")
    print(f"{'负类':>6} {'n_neg':>6} {'CV-AUC':>8} {'95% CI':>16} {'p':>9}   语义")
    print("-" * 78)
    lab = {"D2a": "类别对照(静态物) — 主读数 R①",
           "D2cV": "**纯证伪控制**(同类别VRU+同几何，只差速度)",
           "D2c": "证伪控制(混 55% 车辆，类别差可见 => 地板偏高)",
           "D2b": "上下文对照(走廊外，混类别)",
           "D2bV": "上下文对照(VRU only)",
           "D": "旧口径负例"}
    out = {}

    def hv(A, n1, n2):
        Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
        return (A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n2 - 1) * (Q2 - A * A)) / (n1 * n2)

    for t in ("D2a", "D2cV", "D2c", "D2b", "D2bV", "D"):
        N = cat(t)
        if len(N) < 20:
            continue
        a, p = auc(P, N)
        se = np.sqrt(hv(a, len(P), len(N)))
        out[t] = {"n_pos": len(P), "n_neg": len(N), "auc": a, "p": p,
                  "ci95": [a - 1.96 * se, a + 1.96 * se]}
        print(f"{t:>6} {len(N):>6} {a:8.3f} {f'[{a-1.96*se:.3f},{a+1.96*se:.3f}]':>16} {p:9.3g}   {lab[t]}")

    floor_key = "D2cV" if "D2cV" in out else "D2c"
    if "D2a" in out and floor_key in out:
        d = out["D2a"]["auc"] - out[floor_key]["auc"]
        print(f"\n[N1-CV] 主读数 − 证伪地板({floor_key}) = {d:+.3f}"
              f"  -> {'D2a 高于噪声地板' if d > 0.05 else '**D2a 与噪声地板无法区分**'}")
    (work / "results" / f"n1_cv_{args.pool_mode}_{args.direction}.json").write_text(json.dumps(
        {"folds": args.folds, "peaks": peaks, "pool_mode": args.pool_mode, "readouts": out},
        indent=2, ensure_ascii=False))
    print(f"[N1-CV] wrote results/n1_cv_{args.pool_mode}.json")


if __name__ == "__main__":
    main()
