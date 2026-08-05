"""M1 in-domain 读数：与 nuScenes N1 **同一条管线、同一套判据**。

裁决问题（guide §12.2-M1）：
  in-domain 探针强 + 行为强  => nuScenes 的空结果 = OOD 表征退化（sim2real 主结论）
  in-domain 也空              => 读数方法有病，修方法，Tier-L 缓行

对照设计（三组，从弱到强）：
  ① Hcar vs Ncar   同类别(车辆)、几何匹配、专家标注的"危险 vs 非危险" —— **主读数，有功效**
  ② Aexp/A vs D2a  行人 vs 静物（§12.5 的 R①）—— 样本少、几何难匹配，附带报告
  ③ Aexp/A vs D2cV 危险行人 vs 无害行人 —— 证伪地板（同类别，只差速度）

纪律：方向在 S_dir 上有监督提取，峰层在 S_sel 上按 AUC 选（ρ_TTC 对单帧无效），
scene(=route) 级 K 折，全部事件都当过 held-out；并列报随机方向地板。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402
from n1_readout import auc, proj, supervised_direction  # noqa: E402


def hv(a, n1, n2):
    Q1 = a / (2 - a); Q2 = 2 * a * a / (1 + a)
    return (a * (1 - a) + (n1 - 1) * (Q1 - a * a) + (n2 - 1) * (Q2 - a * a)) / (n1 * n2)


def cv_readout(by_type, pos_types, neg_type, folds=4, seed=0, random_dir=False):
    POS = [e for t in pos_types for e in by_type.get(t, [])]
    NEG = by_type.get(neg_type, [])
    if len(POS) < 20 or len(NEG) < 20:
        return None
    scenes = sorted({e["scene"] for e in POS + NEG})
    rng = np.random.default_rng(seed)
    fold = {s: i % folds for i, s in enumerate(rng.permutation(scenes))}
    hp, hn, peaks = [], [], []
    for f in range(folds):
        trP = [e for e in POS if fold[e["scene"]] != f]
        trN = [e for e in NEG if fold[e["scene"]] != f]
        teP = [e for e in POS if fold[e["scene"]] == f]
        teN = [e for e in NEG if fold[e["scene"]] == f]
        if len(trP) < 15 or len(trN) < 15 or len(teP) < 5 or len(teN) < 5:
            continue
        inner = sorted({e["scene"] for e in trP})
        sel = set(inner[: max(1, len(inner) // 4)])
        fp = [e for e in trP if e["scene"] not in sel]; fn = [e for e in trN if e["scene"] not in sel]
        sp = [e for e in trP if e["scene"] in sel];    sn = [e for e in trN if e["scene"] in sel]
        if len(fp) < 12 or len(sp) < 5 or len(sn) < 5:
            fp, fn, sp, sn = trP, trN, trP, trN
        if random_dir:
            rs = np.random.default_rng(1000 * seed + f)
            L, C = fp[0]["h_ghost"].shape
            v = rs.normal(size=(L, C)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        else:
            v = supervised_direction(fp, fn, seed=seed)
        a = np.array([auc(proj(sp, v, l), proj(sn, v, l))[0] for l in range(v.shape[0])])
        pk = int(np.nanargmax(a)) if np.isfinite(a).any() else v.shape[0] // 2
        peaks.append(pk)
        hp.append(proj(teP, v, pk)); hn.append(proj(teN, v, pk))
    if not hp:
        return None
    P = np.concatenate(hp); N = np.concatenate(hn)
    a, p = auc(P, N)
    se = np.sqrt(hv(a, len(P), len(N)))
    return {"auc": a, "p": p, "ci95": [a - 1.96 * se, a + 1.96 * se],
            "n_pos": len(P), "n_neg": len(N), "peaks": peaks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-modes", nargs="+", default=["region_mean", "vision_mean", "last_token"])
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a", "D2aP", "D2cV", "Ncar")
               if (work / "mining" / f"matched_{t}.txt").exists()}

    CONTRASTS = [(["Hcar"], "Ncar", "① 同类别车辆 危险vs非危险（几何匹配，主读数）"),
                 (["Aexp", "A"], "D2a", "② 行人 vs 静物（§12.5 R①）"),
                 (["Aexp", "A"], "D2cV", "③ 危险行人 vs 无害行人（证伪地板）")]
    out = {}
    for pool in args.pool_modes:
        items = load_cache(work, pool)
        by = defaultdict(list)
        for eid, e in items.items():
            t = evmap[eid]["event_type"]
            if t in matched and eid not in matched[t]:
                continue
            by[t].append(e)
        print(f"\n===== pool = {pool} =====")
        print(f"{'对照':>46} {'n(正/负)':>13} {'AUC':>7} {'95% CI':>16} {'p':>10} {'随机方向地板':>12}")
        print("-" * 112)
        for pos, neg, label in CONTRASTS:
            r = cv_readout(by, pos, neg)
            if r is None:
                print(f"{label:>46} {'样本不足':>13}")
                continue
            rr = cv_readout(by, pos, neg, random_dir=True)
            floor = f"{rr['auc']:.3f}" if rr else "n/a"
            npn = f"{r['n_pos']}/{r['n_neg']}"
            ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
            print(f"{label:>46} {npn:>13} {r['auc']:7.3f} {ci:>16} {r['p']:10.3g} {floor:>12}")
            out[f"{pool}|{'+'.join(pos)}_vs_{neg}"] = {**r, "random_floor": rr["auc"] if rr else None,
                                                      "label": label}
    (work / "results").mkdir(exist_ok=True)
    (work / "results" / "m1_readout.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[M1] wrote results/m1_readout.json")


if __name__ == "__main__":
    main()
