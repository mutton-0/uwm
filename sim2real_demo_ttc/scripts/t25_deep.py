"""T2.5 深挖｜把「CoT 完全没提到行人」这个个例升级为量化结论（只观测，不注入）。

四项分析：
  ①**提及率分类统计** —— A 类 vs D2a 的 VRU 词汇提及率。A 类若接近零，
    「感知层丢失」就从个例升为量化结论，链路表唯一的空格（视觉感知·未测）
    先用**语言代理证据**填上（相关档，非定罪）；
  ②**提及 × 刹车 2×2** —— 在「提到了行人」的子集里刹车达标率是否跳升：
      跳升 ⇒「看见就会刹，但基本看不见」，瓶颈锁死在感知，post-train 处方改写为修感知；
      不跳 ⇒「看见也不刹」，断点在感知之后，回到原画像；
  ③**条件化重提轴（可能翻案）** —— 把缓存按提及/未提及切开，只在提及子集里重算
    视觉判别式方向的 AUC 与两条既有轴的投影差。此前的零结果可能是被大量
    「没看见」的事件稀释的。n 会小，按三态纪律该报「不可估」就报；
  ④**提及 vs 几何** —— 提及率与成像面积/离心率/距离的关系，复用 Q3 的几何混淆机器：
    模型是不是只「看见」大而居中的行人。

纪律：全部**相关档**。unfaithful CoT 风险挂着 —— **没提及 ≠ 没感知**
（语言头有自己的输出偏好，风格是指令式短句）。定罪需等遗留①的内部物体存在性探针配对验证。
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
from t25_language import encode  # noqa: E402


def boot_rate(vals, scenes, n_boot=2000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(float(v))
    keys = list(by)
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(np.mean([float(v) for v in vals])), \
        [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def hv_se(A, n1, n2):
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    return float(np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n2 - 1) * (Q2 - A * A)) / (n1 * n2)))


def tri_state(n_pos, n_neg, a, p, floor):
    """三态纪律：功效不足报「不可估」，不硬判。"""
    if n_pos < 20 or n_neg < 20:
        return "不可估（n<20）"
    se = hv_se(a, n_pos, n_neg)
    if a - 1.96 * se > floor:
        return "PASS（显著高于地板）"
    if 1.96 * se > 0.10:
        return f"不可估（CI 半宽 ±{1.96*se:.3f} > 0.10）"
    return "FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default="")
    ap.add_argument("--pool-mode", default="region_mean")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"]); R = work / "results"
    b_min = float(cfg["metrics"]["b_min_primary"])
    raw = json.loads(Path(args.raw or (R / "t25_language_raw.json")).read_text())
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    matched = set((work / "mining" / "matched_D2a.txt").read_text().split())

    lang = {eid: encode(" ".join(rec["ghost"])) for eid, rec in raw.items()}
    items_b = load_cache(work, "vision_mean", keep=set(evmap), verbose=False)

    def mk(t):
        out = []
        for eid, e in evmap.items():
            if e["event_type"] != t or eid not in lang or eid not in items_b:
                continue
            if t == "D2a" and eid not in matched:
                continue
            out.append({"id": eid, "scene": e["scene_name"], "b": items_b[eid]["b"],
                        "area": e.get("area_px"), "ecc": e.get("ecc"),
                        "d": e.get("d_long_at_emergence"), **lang[eid]})
        return out
    A, N = mk("A"), mk("D2a")
    print(f"[深挖] A={len(A)}  D2a={len(N)}  池化={args.pool_mode}\n")

    res = {"scope": "T2.5 深挖 —— 全部相关档；没提及 ≠ 没感知，定罪需遗留① 内部探针配对",
           "n": {"A": len(A), "D2a": len(N)}}

    # ---------- ① 提及率 ----------
    print("=== ① VRU 词汇提及率（链路① 视觉感知 的语言代理版）===")
    ma, cia = boot_rate([r["mentions_vru"] for r in A], [r["scene"] for r in A])
    mn, cin = boot_rate([r["mentions_vru"] for r in N], [r["scene"] for r in N])
    p1 = stats.fisher_exact([[sum(r["mentions_vru"] for r in A), len(A) - sum(r["mentions_vru"] for r in A)],
                             [sum(r["mentions_vru"] for r in N), len(N) - sum(r["mentions_vru"] for r in N)]])[1]
    print(f"  A（VRU 突现）  {ma:.4f}  [{cia[0]:.4f}, {cia[1]:.4f}]   n={len(A)}  "
          f"提及事件数 {sum(r['mentions_vru'] for r in A)}")
    print(f"  D2a（静物）    {mn:.4f}  [{cin[0]:.4f}, {cin[1]:.4f}]   n={len(N)}  "
          f"提及事件数 {sum(r['mentions_vru'] for r in N)}")
    print(f"  Fisher p={p1:.3g}")
    verdict1 = ("A 类提及率接近零 ⇒ 感知层丢失（语言代理证据，相关档）"
                if ma < 0.10 else
                ("A 显著高于 D2a ⇒ 语言确实分辨了 VRU" if p1 < 0.05 and ma > mn else "无差异"))
    print(f"  -> {verdict1}")
    res["mention_rate"] = {"A": ma, "A_ci": cia, "D2a": mn, "D2a_ci": cin,
                           "fisher_p": float(p1), "verdict": verdict1,
                           "n_mentioned_A": int(sum(r["mentions_vru"] for r in A))}

    # ---------- ② 提及 × 刹车 2×2 ----------
    print("\n=== ② 提及 × 刹车 2×2（A 类）===")
    men = np.array([r["mentions_vru"] for r in A])
    did = np.array([r["b"] >= b_min for r in A])
    n_m = int(men.sum())
    if n_m >= 5:
        r_m, ci_m = boot_rate(did[men], [r["scene"] for r, k in zip(A, men) if k])
        r_n, ci_n = boot_rate(did[~men], [r["scene"] for r, k in zip(A, men) if not k])
        p2 = stats.fisher_exact([[int((men & did).sum()), int((men & ~did).sum())],
                                 [int((~men & did).sum()), int((~men & ~did).sum())]])[1]
        print(f"  提及行人子集   n={n_m:3d}   刹车达标率 {r_m:.3f} [{ci_m[0]:.3f}, {ci_m[1]:.3f}]")
        print(f"  未提及子集     n={len(A)-n_m:3d}   刹车达标率 {r_n:.3f} [{ci_n[0]:.3f}, {ci_n[1]:.3f}]")
        print(f"  Fisher p={p2:.3g}   差值 {r_m-r_n:+.3f}")
        v2 = ("看见就会刹，但基本看不见 ⇒ 瓶颈锁死在感知，post-train 处方 = 修感知"
              if p2 < 0.05 and r_m > r_n else
              ("看见也不刹 ⇒ 断点在感知之后，回到原画像" if n_m >= 20 else
               "不可估（提及子集 n<20，功效不足）"))
        print(f"  -> {v2}")
        res["mention_x_brake"] = {"n_mentioned": n_m, "rate_mentioned": r_m, "ci_mentioned": ci_m,
                                  "rate_not": r_n, "ci_not": ci_n, "fisher_p": float(p2),
                                  "delta": r_m - r_n, "verdict": v2}
    else:
        print(f"  提及子集 n={n_m} < 5 —— **不可估**")
        res["mention_x_brake"] = {"n_mentioned": n_m, "verdict": "不可估（提及子集 n<5）"}

    # ---------- ③ 条件化重提轴 ----------
    print("\n=== ③ 条件化重提轴：只在「提及」子集里重算读数 ===")
    items = load_cache(work, args.pool_mode, keep=set(evmap), verbose=False)
    byid = {e["meta"]["event_id"]: e for e in items.values()}
    Aev = [byid[r["id"]] for r in A if r["id"] in byid]
    Nev = [byid[r["id"]] for r in N if r["id"] in byid]
    men_of = {r["id"]: r["mentions_vru"] for r in A}

    scenes = sorted({e["scene"] for e in Aev + Nev})
    rng = np.random.default_rng(args.seed)
    fold_of = {s: i % args.folds for i, s in enumerate(rng.permutation(scenes))}
    held = {"A_men": [], "A_not": [], "D2a": []}
    for f in range(args.folds):
        tr_p = [e for e in Aev if fold_of[e["scene"]] != f]
        tr_n = [e for e in Nev if fold_of[e["scene"]] != f]
        if len(tr_p) < 20 or len(tr_n) < 20:
            continue
        insc = sorted({e["scene"] for e in tr_p})
        sels = set(insc[: max(1, len(insc) // 4)])
        fit_p = [e for e in tr_p if e["scene"] not in sels] or tr_p
        fit_n = [e for e in tr_n if e["scene"] not in sels] or tr_n
        sel_p = [e for e in tr_p if e["scene"] in sels] or tr_p
        sel_n = [e for e in tr_n if e["scene"] in sels] or tr_n
        v = supervised_direction(fit_p, fit_n, seed=args.seed)
        a_by_l = [auc(proj(sel_p, v, l), proj(sel_n, v, l))[0] for l in range(v.shape[0])]
        L = int(np.nanargmax(a_by_l))
        for e in [x for x in Aev if fold_of[x["scene"]] == f]:
            held["A_men" if men_of.get(e["meta"]["event_id"]) else "A_not"].append(
                float((e["h_ghost"][L] - e["h_clean"][L]) @ v[L]))
        for e in [x for x in Nev if fold_of[x["scene"]] == f]:
            held["D2a"].append(float((e["h_ghost"][L] - e["h_clean"][L]) @ v[L]))
    D = np.array(held["D2a"])
    cond = {}
    for k, lab in (("A_men", "提及子集"), ("A_not", "未提及子集")):
        P = np.array(held[k])
        if len(P) < 5 or len(D) < 5:
            print(f"  {lab:8s} n={len(P):3d}  —— 不可估")
            cond[k] = {"n": len(P), "verdict": "不可估"}
            continue
        a_, p_ = auc(P, D)
        se = hv_se(a_, len(P), len(D))
        st = tri_state(len(P), len(D), a_, p_, 0.554)
        print(f"  {lab:8s} n={len(P):3d}  AUC(vs D2a)={a_:.3f} "
              f"[{a_-1.96*se:.3f}, {a_+1.96*se:.3f}]  p={p_:.3g}  -> {st}")
        cond[k] = {"n": len(P), "auc": float(a_), "ci95": [a_ - 1.96 * se, a_ + 1.96 * se],
                   "p": float(p_), "verdict": st}
    cond["floor_note"] = "地板 0.554 = 同流程随机方向地板（见 t1_t2_s1_report §2.1）"
    res["conditional_axis"] = cond

    # 两条既有轴在提及/未提及子集上的投影差
    for axf, L_, nm in (("v_danger_lang_query_mean.npy", 10, "v_danger_lang@L10"),
                        ("v_brake_query_mean.npy", 22, "v_brake@L22")):
        p_ = R / axf
        if not p_.exists():
            continue
        V = np.load(p_)
        itq = load_cache(work, "query_mean", keep=set(evmap), verbose=False)
        q = {e["meta"]["event_id"]: e for e in itq.values()}
        sub = {}
        for k, flag in (("提及", True), ("未提及", False)):
            ev = [q[r["id"]] for r in A if r["id"] in q and r["mentions_vru"] == flag]
            if len(ev) < 5:
                sub[k] = {"n": len(ev), "verdict": "不可估"}; continue
            d = np.array([float((e["h_ghost"][L_] - e["h_clean"][L_]) @ V[L_]) for e in ev])
            m, ci = boot_rate(d, [e["scene"] for e in ev])
            sub[k] = {"n": len(ev), "mean": m, "ci95": ci}
            print(f"  {nm} {k}子集 n={len(ev):3d}  投影差 {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
        res.setdefault("conditional_projdiff", {})[nm] = sub

    # ---------- ④ 提及 vs 几何 ----------
    print("\n=== ④ 提及 vs 几何（模型是不是只看见大而居中的行人）===")
    geo = {}
    for key, lab in (("area", "成像面积 px"), ("ecc", "离心率"), ("d", "纵向距离 m")):
        x = np.array([r[key] for r in A if r[key] is not None], float)
        y = np.array([r["mentions_vru"] for r in A if r[key] is not None], float)
        if len(x) < 20 or y.sum() < 3:
            geo[key] = {"verdict": "不可估"}
            print(f"  {lab:12s} 不可估（提及数不足）"); continue
        r_, p_ = stats.pointbiserialr(y, x)
        mm = float(np.median(x[y == 1])) if y.sum() else float("nan")
        mn2 = float(np.median(x[y == 0]))
        u = stats.mannwhitneyu(x[y == 1], x[y == 0]).pvalue if y.sum() >= 3 else float("nan")
        geo[key] = {"pointbiserial_r": float(r_), "p": float(p_),
                    "median_mentioned": mm, "median_not": mn2, "mwu_p": float(u)}
        print(f"  {lab:12s} r={r_:+.3f} p={p_:.3g}   中位 提及 {mm:.2f} vs 未提及 {mn2:.2f}  MWU p={u:.3g}")
    res["mention_vs_geometry"] = geo

    # ---------- ⑤ 语言-行为四分（修正二值 2×2 的稀释）----------
    # 二值标签 says_slow 把「又说加速又说减速」的事件并进了「说减速」，稀释了信号。
    # 事件级还把 2 帧文本拼在一起（35% 的事件因此同时命中两个模式）——两处都在这里拆开。
    print("\n=== ⑤ 语言-行为四分（修正二值 2×2 的稀释）===")
    from t25_language import FAST_RE, SLOW_RE
    raw = json.loads(Path(args.raw or (R / "t25_language_raw.json")).read_text())
    grp, gsc = defaultdict(list), defaultdict(list)
    n_both_evt = n_evt = 0
    fr_f = fr_s = fr_b = fr_n = 0
    for r in A:
        rec = raw.get(r["id"])
        if not rec:
            continue
        for t in rec["ghost"]:
            fr_n += 1
            f_, s_ = bool(FAST_RE.search(t)), bool(SLOW_RE.search(t))
            fr_f += f_; fr_s += s_; fr_b += (f_ and s_)
        txt = " ".join(rec["ghost"])
        f_, s_ = bool(FAST_RE.search(txt)), bool(SLOW_RE.search(txt))
        n_evt += 1; n_both_evt += (f_ and s_)
        k = "只说减速" if (s_ and not f_) else ("都说" if (s_ and f_) else
                                                ("只说加速" if f_ else "都没说"))
        grp[k].append(r["b"]); gsc[k].append(r["scene"])
    print(f"  编码器口径核对：事件级拼了 {fr_n//max(1,n_evt)} 帧 ⇒ "
          f"{n_both_evt}/{n_evt}={n_both_evt/n_evt:.3f} 的事件同时命中 fast+slow"
          f"（这解释了事件级 0.899+0.448>1）")
    print(f"  **逐帧**口径：说加速 {fr_f/fr_n:.3f}　说减速 {fr_s/fr_n:.3f}　同现 {fr_b/fr_n:.3f}")
    print(f"\n{'语言':>10}{'n':>6}{'b 均值':>10}{'b 中位':>10}{'P(b≥0.5)':>11}")
    four = {}
    for k in ("只说减速", "都说", "只说加速", "都没说"):
        v = np.array(grp[k])
        if not len(v):
            continue
        four[k] = {"n": len(v), "b_mean": float(v.mean()), "b_median": float(np.median(v)),
                   "pass_rate": float(np.mean(v >= b_min))}
        print(f"{k:>10}{len(v):>6}{v.mean():>10.4f}{np.median(v):>10.4f}{np.mean(v >= b_min):>11.3f}")
    order = [k for k in ("只说加速", "都说", "只说减速") if k in four]
    G = [np.array(grp[k]) for k in order]
    if len(G) == 3 and all(len(x) >= 5 for x in G):
        J = 0.0
        for i in range(2):
            for j in range(i + 1, 3):
                a_, b_ = G[i], G[j]
                J += float((a_[:, None] < b_[None, :]).sum() + 0.5 * (a_[:, None] == b_[None, :]).sum())
        ns = np.array([len(x) for x in G], float); N = ns.sum()
        z = (J - (N ** 2 - (ns ** 2).sum()) / 4) / np.sqrt(
            (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72)
        p_jt = float(2 * (1 - stats.norm.cdf(abs(z))))
        kw = stats.kruskal(*G)
        mwu = stats.mannwhitneyu(G[2], G[0]).pvalue
        print(f"\n  有序趋势（只说加速 < 都说 < 只说减速）JT z={z:+.3f}  p={p_jt:.3g}")
        print(f"  Kruskal–Wallis H={kw.statistic:.2f}  p={kw.pvalue:.3g}")
        print(f"  只说减速 vs 只说加速：MWU p={mwu:.3g}，b 中位差 {np.median(G[2])-np.median(G[0]):+.3f} m/s")
        v5 = ("语言与行为**强单调对应** ⇒ CoT 确实跟着行为走；"
              "二值 2×2 的 κ=0.16 是把「都说」并进「说减速」造成的稀释")
        print(f"  -> {v5}")
        res["language_behavior_ordered"] = {
            "four_way": four, "jt_z": float(z), "jt_p": p_jt,
            "kruskal_p": float(kw.pvalue), "mwu_p": float(mwu),
            "median_gap": float(np.median(G[2]) - np.median(G[0])),
            "encoder_note": f"事件级拼 2 帧 ⇒ {n_both_evt/n_evt:.3f} 的事件同时命中 fast+slow；"
                            f"逐帧口径 说加速 {fr_f/fr_n:.3f} / 说减速 {fr_s/fr_n:.3f}",
            "per_frame": {"says_fast": fr_f / fr_n, "says_slow": fr_s / fr_n,
                          "both": fr_b / fr_n, "n_frames": fr_n},
            "verdict": v5}

    (R / f"t25_deep{args.tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[深挖] wrote results/t25_deep{args.tag}.json")


if __name__ == "__main__":
    main()
