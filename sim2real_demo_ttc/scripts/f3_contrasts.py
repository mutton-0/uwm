"""GFIC 主假设的正式检验：场景间（ghost vs lead）与语料间（benchmark vs deployment）。

两者都是**独立样本**（不同事件、不同 scene），故对两侧各自做 scene-level bootstrap
再取差；不做配对。所有检验一次性列出并给 Bonferroni 校正后的判定门槛。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results"); RD = "arc_full"
MODELS = ["simlingo", "dd", "ltf", "ddv2", "alpa", "autovla"]
CELL = {("ghost", "nuscenes"): "f3_gt_axis_full_{m}.json",
        ("ghost", "navsim"): "f3_gt_dep_ghost_{m}.json",
        ("lead", "nuscenes"): "gt_lead_{m}.json",
        ("lead", "navsim"): "f3_gt_dep_lead_{m}.json"}


def have(cell, m):
    return (RES / CELL[cell].format(m=m)).exists()


def load(cell, m):
    ev = json.load(open(RES / CELL[cell].format(m=m)))["per_event"]
    return (np.array([e[f"b_model__{RD}"] for e in ev]),
            np.array([e[f"b_gt__{RD}"] for e in ev]),
            np.array([e["scene"] for e in ev]))


def boot(bm, bg, sc, n, rng):
    uq = np.unique(sc); ix = {s: np.where(sc == s)[0] for s in uq}
    out = np.empty(n)
    for i in range(n):
        sel = np.concatenate([ix[s] for s in rng.choice(uq, len(uq), replace=True)])
        d = bg[sel].mean()
        out[i] = bm[sel].mean() / d if abs(d) > 1e-9 else np.nan
    return out


def contrast(cA, cB, m, n=5000):
    rng = np.random.default_rng(0)
    a = boot(*load(cA, m), n, rng); b = boot(*load(cB, m), n, rng)
    d = a - b; d = d[np.isfinite(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (np.nanmean(a) - np.nanmean(b), np.percentile(d, [2.5, 97.5]), p)


def main():
    tests = [("场景对比 ghost−lead", "nuscenes", ("ghost", "nuscenes"), ("lead", "nuscenes")),
             ("场景对比 ghost−lead", "navsim", ("ghost", "navsim"), ("lead", "navsim")),
             ("语料对比 nusc−navsim", "ghost", ("ghost", "nuscenes"), ("ghost", "navsim")),
             ("语料对比 nusc−navsim", "lead", ("lead", "nuscenes"), ("lead", "navsim"))]
    k = sum(1 for _, _, cA, cB in tests for m in MODELS
            if have(cA, m) and have(cB, m))
    print(f"\n共 {k} 次检验，Bonferroni α = {0.05/k:.4f}\n")
    print(f"{'检验':<22}{'条件':<10}{'候选':<10}{'Δ响应比':>10}{'  95%CI':<22}{'p':>8}  判定")
    print("-" * 92)
    out = {}
    for name, cond, cA, cB in tests:
        for m in MODELS:
            # 两个 VLA 只有 deployment 侧数据；缺哪一侧就跳过该格，不编造
            if not (have(cA, m) and have(cB, m)):
                continue
            d, ci, p = contrast(cA, cB, m)
            sig = "**显著**" if p < 0.05 / k else ("(未校正显著)" if p < 0.05 else "")
            out[f"{name}|{cond}|{m}"] = dict(delta=float(d), ci=list(ci), p=float(p))
            print(f"{name:<22}{cond:<10}{m:<10}{d:>+10.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}]"
                  f"{'':<4}{p:>7.4f}  {sig}")
        print()
    (RES / "f3_contrasts.json").write_text(
        json.dumps({"readout": RD, "n_tests": k, "bonferroni_alpha": 0.05 / k,
                    "tests": out}, indent=2, ensure_ascii=False))
    print(f"[CONTRAST] wrote {RES/'f3_contrasts.json'}")


if __name__ == "__main__":
    main()
