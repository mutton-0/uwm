"""T1-L Step 3｜核心读取实验:视觉危险事件到底激不激活"危险概念轴"（guide 附录 A）。

全程**中性 prompt** —— 危险措辞只存在于 Step 1 的离线刺激集，此后绝不出现
（"测量不污染"红线）。因此本脚本直接吃 G2 缓存（缓存本来就是中性 prompt、
clean/ghost 两条件逐字一致的 prompt_anchor 口径），零额外前向。

主读数（预注册）：query_mean 的 δ=(ghost−clean) 在 $v_{danger}^{lang}$ 上的**投影差**，
层 = Step 1 用 held-out 一致性选出的 L*；scene 级 bootstrap。
辅助：①投影差 A vs D2a 的 AUC（几何稳健性）；②逐事件投影 vs 行为量 b 的相关（行为挂钩）。

判定接 guide 附录 A 的四格表：Step1✅ + Step2✅ + 本步显著 = 危险轴找到；
Step1✅ + Step2✅ + 本步无差 = **视觉→概念断**（OOD 退化的直接机制证据）。
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
from n1_readout import auc  # noqa: E402


def boot_ci(vals, scenes, n_boot=2000, seed=0):
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals, scenes = vals[ok], np.asarray(scenes)[ok]
    if len(vals) < 5:
        return float("nan"), (float("nan"), float("nan"))
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(vals.mean()), (float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--axis", required=True, help="Step 1 存下的 v_danger_lang_*.npy")
    ap.add_argument("--axis-json", default="", help="Step 1 的 json（读 L*）")
    ap.add_argument("--layer", type=int, default=-1, help="不给则从 axis-json 读 peak_layer")
    ap.add_argument("--pool-mode", default="query_mean")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    V = np.load(args.axis).astype(np.float32)
    L = args.layer
    if L < 0:
        L = int(json.loads(Path(args.axis_json).read_text())["peak_layer"])
    print(f"[Step3] 轴 {Path(args.axis).name}  层 L*={L}  池化={args.pool_mode}")

    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))
    matched = {}
    for t in ("D2a", "D2b", "D2c"):
        f = work / "mining" / f"matched_{t}.txt"
        if f.exists():
            matched[t] = set(f.read_text().split())

    by_type = defaultdict(list)
    for eid, e in items.items():
        t = evmap[eid]["event_type"]
        if t in matched and eid not in matched[t]:
            continue
        by_type[t].append(e)

    v = V[L] / (np.linalg.norm(V[L]) + 1e-8)

    def projdiff(evs):
        """δ=(ghost−clean) 在轴上的投影 —— 主读数量。"""
        return np.array([float((e["h_ghost"][L] - e["h_clean"][L]) @ v) for e in evs])

    A = by_type["A"]
    pa = projdiff(A)
    sc = [e["scene"] for e in A]
    m, ci = boot_ci(pa, sc)
    t, p = stats.ttest_1samp(pa, 0.0)
    print(f"\n[主读数] A 类 投影差(ghost−clean) 均值 = {m:+.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  n={len(pa)}  单样本 t p={p:.3g}")
    hit = "✅ 视觉危险**激活**了语言定义的危险概念轴" if ci[0] > 0 else \
          ("⚠️ 显著为负（方向相反）" if ci[1] < 0 else "❌ 无差 —— 视觉→概念断")
    print(f"          {hit}")

    print(f"\n[辅助①] 投影差的类别判别力（A vs 各负类）")
    rows = {}
    for nt in ("D2a", "D2b", "D2c"):
        N = by_type.get(nt, [])
        if len(N) < 20:
            continue
        a_, p_ = auc(pa, projdiff(N))
        rows[nt] = {"n": len(N), "auc": a_, "p": p_}
        print(f"          {nt:>5}  n={len(N):>4}  AUC={a_:.3f}  p={p_:.3g}")

    b = np.array([e["b"] for e in A])
    r_s, p_s = stats.spearmanr(pa, b)
    print(f"\n[辅助②] 投影差 vs 行为量 b：Spearman ρ={r_s:+.3f}  p={p_s:.3g}  "
          + ("✅ 行为挂钩" if p_s < 0.05 else "❌ 无关联"))

    out = {"axis": str(args.axis), "layer": L, "pool_mode": args.pool_mode,
           "n_pos": len(pa), "projdiff_mean": m, "projdiff_ci95": list(ci),
           "ttest_p": float(p), "auc_vs_neg": rows,
           "spearman_b": {"rho": float(r_s), "p": float(p_s)},
           "verdict": ("激活" if ci[0] > 0 else ("反向" if ci[1] < 0 else "无差=视觉→概念断"))}
    (work / "results" / f"t1l_readout_{args.pool_mode}{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[Step3] wrote results/t1l_readout_{args.pool_mode}{args.tag}.json")


if __name__ == "__main__":
    main()
