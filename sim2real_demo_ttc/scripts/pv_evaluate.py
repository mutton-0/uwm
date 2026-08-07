"""预测-验证评估：按 `results/amendments.md` §PV 冻结的口径执行，**不得调参**。

三条预测（PV.2）：
  P1  A 类语言行人提及率 < 30%              评估集 V-A（主）
  P2  P(b<0) ∈ [45%, 55%]                   评估集 V-BC（完全盲测）
  P3  提及×刹车「结构复现 = 仍无显著跳升」   评估集 V-A + V-BC

几何控制（PV.2b）：提及项在控 log10(area_px) + ecc 后是否仍显著
  —— 仍显著 ⇒「感知门控行为」；消失 ⇒「几何共因」。V-A 无几何字段，只报未控版本。

功效纪律（PV.2c）：提及子集 CI 半宽 > 0.10 时报「不可估」，不得读作"仍无跳升"。
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
from t25_language import FAST_RE, SLOW_RE, VRU_RE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def boot(vals, scenes, n_boot=4000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(float(v))
    keys = list(by)
    if not keys:
        return float("nan"), [float("nan")] * 2
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(np.mean([float(v) for v in vals])), \
        [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def load_set(cfg_name, cot_path, types):
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / cfg_name), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    b_min = float(cfg["metrics"]["b_min_primary"])
    raw = json.loads(Path(cot_path).read_text())
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, "vision_mean", keep=set(evmap), verbose=False)
    rows = []
    for eid, rec in raw.items():
        e = evmap.get(eid)
        if not e or e["event_type"] not in types or eid not in items:
            continue
        t = " ".join(rec["ghost"])
        rows.append({"id": eid, "type": e["event_type"], "scene": e["scene_name"],
                     "b": items[eid]["b"], "night": bool(e.get("is_night")),
                     "area": e.get("area_px"), "ecc": e.get("ecc"),
                     "d": e.get("d_long_at_emergence"),
                     "vru": bool(VRU_RE.search(t)), "slow": bool(SLOW_RE.search(t)),
                     "fast": bool(FAST_RE.search(t))})
    return rows, b_min, work.name


def mention_x_brake(rows, b_min, label, geo=True):
    """P3 主检验 + PV.2b 几何控制。"""
    men = np.array([r["vru"] for r in rows])
    did = np.array([r["b"] >= b_min for r in rows])
    n_m = int(men.sum())
    out = {"n": len(rows), "n_mentioned": n_m}
    if n_m < 5 or (len(rows) - n_m) < 5:
        out["verdict"] = "不可估（子集 n<5）"
        print(f"  [{label}] 提及子集 n={n_m} —— 不可估")
        return out
    r_m, ci_m = boot(did[men], [r["scene"] for r, k in zip(rows, men) if k])
    r_n, ci_n = boot(did[~men], [r["scene"] for r, k in zip(rows, men) if not k])
    p = stats.fisher_exact([[int((men & did).sum()), int((men & ~did).sum())],
                            [int((~men & did).sum()), int((~men & ~did).sum())]])[1]
    half = (ci_m[1] - ci_m[0]) / 2
    out.update({"rate_mentioned": r_m, "ci_mentioned": ci_m, "rate_not": r_n, "ci_not": ci_n,
                "delta": r_m - r_n, "fisher_p": float(p), "ci_halfwidth": half})
    print(f"  [{label}] 提及 n={n_m:4d} 达标率 {r_m:.3f} [{ci_m[0]:.3f},{ci_m[1]:.3f}]　"
          f"未提及 n={len(rows)-n_m:4d} {r_n:.3f} [{ci_n[0]:.3f},{ci_n[1]:.3f}]　"
          f"Δ={r_m-r_n:+.3f}  Fisher p={p:.3g}")
    if p >= 0.05 and half > 0.10:
        out["verdict"] = f"不可估（CI 半宽 ±{half:.3f} > 0.10，功效不足）"
    elif p >= 0.05:
        out["verdict"] = "仍无显著跳升（功效充足）⇒「看见也不刹」，双病灶"
    else:
        out["verdict"] = "出现显著跳升 —— 需看几何控制"
    print(f"        -> {out['verdict']}")

    if geo:
        sub = [r for r in rows if r["area"] and r["ecc"] is not None and r["area"] > 0]
        if len(sub) >= 60 and 5 <= sum(r["vru"] for r in sub):
            from sklearn.linear_model import LogisticRegression
            X = np.column_stack([[float(r["vru"]) for r in sub],
                                 [np.log10(r["area"]) for r in sub],
                                 [r["ecc"] for r in sub]])
            y = np.array([r["b"] >= b_min for r in sub], int)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y)
            coef = clf.coef_[0]
            # 系数显著性：scene 级 bootstrap
            rng = np.random.default_rng(0)
            by = defaultdict(list)
            for i, r in enumerate(sub):
                by[r["scene"]].append(i)
            keys = list(by)
            bs = []
            for _ in range(600):
                idx = [i for j in rng.choice(len(keys), len(keys), True) for i in by[keys[j]]]
                Xi, yi = X[idx], y[idx]
                if len(set(yi)) < 2 or Xi[:, 0].std() == 0:
                    continue
                m2, s2 = Xi.mean(0), Xi.std(0) + 1e-9
                bs.append(LogisticRegression(max_iter=1000).fit((Xi - m2) / s2, yi).coef_[0][0])
            lo, hi = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if bs else (np.nan,) * 2
            still = bool(lo > 0 or hi < 0)
            out["geo_control"] = {"n": len(sub), "coef_mention": float(coef[0]),
                                 "ci95": [lo, hi], "coef_logarea": float(coef[1]),
                                 "coef_ecc": float(coef[2]), "still_significant": still,
                                 "reading": "感知门控行为" if still else "几何共因（控完消失）"}
            print(f"        控几何 logistic（n={len(sub)}）：提及项系数 {coef[0]:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]　log面积 {coef[1]:+.3f}　离心率 {coef[2]:+.3f}")
            print(f"        -> 读作「{out['geo_control']['reading']}」")
        else:
            out["geo_control"] = {"verdict": "不可估（有几何字段的样本或提及数不足）"}
            print(f"        控几何：样本不足，不可估")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    R = ROOT / "results"
    res = {"registered_in": "results/amendments.md §PV", "predictions": {}}

    VA, b_min_a, _ = load_set("tier_m.yaml", ROOT / "variants/tier_m/results/pv_cot_A.json", ("A",))
    VBC, b_min_bc, _ = load_set("n1_d2.yaml", ROOT / "variants/n1_d2/results/pv_cot_BC.json", ("B", "C"))
    print(f"[PV] V-A n={len(VA)}（tier_m A 类）　V-BC n={len(VBC)}（n1_d2 B/C 类）\n")
    res["n"] = {"V_A": len(VA), "V_BC": len(VBC)}

    # ---------- P1 ----------
    print("=== P1：A 类语言行人提及率 < 30% ===")
    m, ci = boot([r["vru"] for r in VA], [r["scene"] for r in VA])
    hit1 = bool(m < 0.30)
    print(f"  V-A 提及率 {m:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]  n={len(VA)}  ->  "
          f"{'✅ 命中' if hit1 else '❌ 未中'}（阈值 <0.30，小集 0.205）")
    mb, cib = boot([r["vru"] for r in VBC], [r["scene"] for r in VBC])
    print(f"  V-BC（跨类型，不参与判定）提及率 {mb:.4f} [{cib[0]:.4f}, {cib[1]:.4f}]")
    res["predictions"]["P1"] = {"rate": m, "ci": ci, "threshold": 0.30, "hit": hit1,
                                "cross_type_VBC": {"rate": mb, "ci": cib}}

    # ---------- P2 ----------
    print("\n=== P2：P(b<0) ∈ [45%, 55%]（V-BC 完全盲测）===")
    neg = [r["b"] < 0 for r in VBC]
    m2, ci2 = boot(neg, [r["scene"] for r in VBC])
    hit2 = bool(0.45 <= m2 <= 0.55)
    print(f"  P(b<0) = {m2:.4f}  [{ci2[0]:.4f}, {ci2[1]:.4f}]  n={len(VBC)}  ->  "
          f"{'✅ 命中' if hit2 else '❌ 未中'}")
    bb = np.array([r["b"] for r in VBC])
    print(f"  参考：b 均值 {bb.mean():+.4f}  中位 {np.median(bb):+.4f}  "
          f"达标率 {np.mean(bb >= b_min_bc):.3f}")
    res["predictions"]["P2"] = {"p_neg": m2, "ci": ci2, "interval": [0.45, 0.55], "hit": hit2,
                                "b_mean": float(bb.mean()), "b_median": float(np.median(bb)),
                                "pass_rate": float(np.mean(bb >= b_min_bc))}

    # ---------- P3 ----------
    print("\n=== P3：提及×刹车 结构复现（仍无显著跳升）===")
    r_a = mention_x_brake(VA, b_min_a, "V-A", geo=False)
    r_bc = mention_x_brake(VBC, b_min_bc, "V-BC", geo=True)
    hits = [x for x in (r_a.get("fisher_p"), r_bc.get("fisher_p")) if x is not None]
    hit3 = bool(hits and all(p >= 0.05 for p in hits))
    est = all("不可估" not in (x.get("verdict") or "") for x in (r_a, r_bc))
    print(f"  -> P3 {'✅ 命中（两集均无显著跳升）' if (hit3 and est) else ('⚠️ 不可估' if not est else '❌ 未中')}")
    res["predictions"]["P3"] = {"V_A": r_a, "V_BC": r_bc, "hit": bool(hit3 and est),
                                "estimable": est}

    # ---------- PV.3 探索性分层 ----------
    print("\n=== PV.3 分层（探索性，不参与判定）===")
    strat = {}
    for name, groups in (("日/夜", [("日", lambda r: not r["night"]), ("夜", lambda r: r["night"])]),):
        rows_all = VA + VBC
        strat[name] = []
        for lab, f_ in groups:
            s = [r for r in rows_all if f_(r)]
            if len(s) < 20:
                continue
            mv, civ = boot([r["vru"] for r in s], [r["scene"] for r in s])
            mp, cip = boot([r["b"] >= 0.5 for r in s], [r["scene"] for r in s])
            strat[name].append({"group": lab, "n": len(s), "mention": mv, "mention_ci": civ,
                                "pass": mp, "pass_ci": cip})
            print(f"  {name} {lab}: n={len(s):4d}  提及率 {mv:.3f} [{civ[0]:.3f},{civ[1]:.3f}]  "
                  f"达标率 {mp:.3f}")
    # 面积三分位（仅 V-BC 有 area_px）
    ar = [r for r in VBC if r["area"]]
    if len(ar) >= 60:
        q = np.quantile([r["area"] for r in ar], [0, 1 / 3, 2 / 3, 1.0])
        strat["bbox 面积三分位（仅 V-BC）"] = []
        for i in range(3):
            s = [r for r in ar if q[i] <= r["area"] <= q[i + 1]] if i == 2 else \
                [r for r in ar if q[i] <= r["area"] < q[i + 1]]
            if len(s) < 15:
                continue
            mv, civ = boot([r["vru"] for r in s], [r["scene"] for r in s])
            mp, _ = boot([r["b"] >= b_min_bc for r in s], [r["scene"] for r in s])
            strat["bbox 面积三分位（仅 V-BC）"].append(
                {"group": f"Q{i+1}", "lo": float(q[i]), "hi": float(q[i + 1]), "n": len(s),
                 "mention": mv, "mention_ci": civ, "pass": mp})
            print(f"  面积 Q{i+1} ({q[i]:.0f}–{q[i+1]:.0f}px): n={len(s):4d}  "
                  f"提及率 {mv:.3f}  达标率 {mp:.3f}")
    res["stratified_exploratory"] = strat

    # ---------- 总判 ----------
    allhit = hit1 and hit2 and res["predictions"]["P3"]["hit"]
    res["verdict"] = {
        "P1": hit1, "P2": hit2, "P3": res["predictions"]["P3"]["hit"], "all_hit": bool(allhit),
        "reading": ("三条全中 ⇒ 小集诊断成功预测大集失效模式，demo 的外推主张获得端到端确认"
                    if allhit else "有预测未中 ⇒ 记录失配形态并标注诊断适用边界"),
    }
    print(f"\n[总判] P1={hit1}  P2={hit2}  P3={res['predictions']['P3']['hit']}  "
          f"=> {'三条全中' if allhit else '未全中'}")
    print(f"       {res['verdict']['reading']}")
    (R / f"pv_result{args.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[PV] wrote results/pv_result{args.tag}.json")


if __name__ == "__main__":
    main()
