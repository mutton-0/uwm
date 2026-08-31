"""T-G|G 轴正向校准:DiffusionDrive 在与 SimLingo **同一刺激集**上的语义奠基性读数。

工单依据:docs/g_axis_positive_calibration_diffusiondrive.md(全文按其执行)。

刺激集完全一致:nuScenes G1 语料 + N1 的 D2a/D2b/D2c/D2cV 负例(经 dd_g1_adapter 转换),
**不使用 DiffusionDrive 自己的 NAVSIM/navhard 评测集**。

流程与 SimLingo 侧逐条对齐:
  方向 = 折内 S_dir 上 A vs D2a 的逐层线性判别(δ = ghost - clean);
  峰层 = 折内 S_sel 上 AUC argmax(单帧模型不能用 ρ_TTC,q_audit §Q6);
  报数 = scene 级 4 折 CV,全部事件都当过 held-out;
  并列 = D2cV 证伪地板 / 随机方向地板 / 标签置换零分布。
唯一的结构性差异:DiffusionDrive 的可读层是 8 个 encoder SelfAttention,各层通道数不同
(64/64/128/128/256/256/512/512),故方向按层各自归一化,跨层不共享维度。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

N_LAYERS = 8          # 由 load_dd_cache 按缓存实际层数改写（navsim 系 8，Alpamayo 36）


def load_dd_cache(cache_dir, pool, types=None):
    """兼容两种缓存 schema：
      navsim 系   `{cond}/{pool}/L{l}` -> [n_frames, C]，逐帧平均
      Alpamayo    `{cond}/{pool}`      -> [n_layers, C]（前向阶段已池化）
    行为量同理：commanded_speed_{cond} 或 {cond}/v_plan。
    """
    global N_LAYERS
    items = {}
    for p in sorted(Path(cache_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        if types and meta["event_type"] not in types:
            continue
        try:
            if f"clean/{pool}" in d.files:                      # Alpamayo schema
                A, B = d[f"clean/{pool}"], d[f"ghost/{pool}"]
                N_LAYERS = int(A.shape[0])
                hc = [A[l] for l in range(N_LAYERS)]
                hg = [B[l] for l in range(N_LAYERS)]
            else:                                               # navsim schema
                hc = [d[f"clean/{pool}/L{l}"].mean(0) for l in range(N_LAYERS)]
                hg = [d[f"ghost/{pool}/L{l}"].mean(0) for l in range(N_LAYERS)]
        except KeyError:
            continue
        items[meta["event_id"]] = {
            "meta": meta, "scene": meta["scene_name"], "etype": meta["event_type"],
            "delta": [g - c for c, g in zip(hc, hg)],
            "b": (float(d["commanded_speed_clean"].mean() - d["commanded_speed_ghost"].mean())
                  if "commanded_speed_clean" in d.files
                  else float(d["clean/v_plan"][0] - d["ghost/v_plan"][0])),
            "area_px": meta.get("area_px"), "ecc": meta.get("ecc"),
            "object_class": meta.get("object_class", ""),
        }
    return items


def sup_dir(pos, neg, seed=0):
    """逐层线性判别方向(各层维度不同 -> 逐层独立拟合并归一化)。"""
    out = []
    y = np.array([1] * len(pos) + [0] * len(neg))
    for l in range(N_LAYERS):
        Z = np.stack([e["delta"][l] for e in pos + neg])
        mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
        w = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit((Z - mu) / sd, y).coef_[0] / sd[0]
        out.append((w / (np.linalg.norm(w) + 1e-8)).astype(np.float32))
    return out


def rand_dir(pos, seed=0):
    rs = np.random.default_rng(seed)
    out = []
    for l in range(N_LAYERS):
        w = rs.normal(size=pos[0]["delta"][l].shape)
        out.append((w / np.linalg.norm(w)).astype(np.float32))
    return out


def perm_dir(pos, neg, seed=0):
    rng = np.random.default_rng(seed)
    a = pos + neg; idx = rng.permutation(len(a)); k = len(pos)
    return sup_dir([a[i] for i in idx[:k]], [a[i] for i in idx[k:]], seed=seed)


def proj(evs, v, l):
    return np.array([e["delta"][l] @ v[l] for e in evs]) if evs else np.zeros(0)


def auc(a, b):
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def hv_se(a, n1, n2):
    Q1 = a / (2 - a); Q2 = 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (Q1 - a * a) + (n2 - 1) * (Q2 - a * a)) / (n1 * n2)))


def cv(by, pos_t, neg_t, folds=4, seed=0, kind="supervised", report_negs=None):
    """口径与 SimLingo 侧 n1_cv.py / b1_hazard_clean.py 逐条对齐（修正案 DV/A16）：
    方向与峰层**只**在 A vs D2a 上拟合/选择，其余负类用**同一条方向、同一个峰层**投影后报数；
    折映射覆盖缓存里全部事件类型的场景。"""
    report_negs = report_negs or []
    POS = [e for t in pos_t for e in by.get(t, [])]; NEG = by.get(neg_t, [])
    if len(POS) < 20 or len(NEG) < 20:
        return None
    scenes = sorted({e["scene"] for v in by.values() for e in v})
    rng = np.random.default_rng(seed)
    fold = {s: i % folds for i, s in enumerate(rng.permutation(scenes))}
    recs, peaks, profs = [], [], []
    rec_neg = {t_: [] for t_ in report_negs}
    for f in range(folds):
        trP = [e for e in POS if fold[e["scene"]] != f]; trN = [e for e in NEG if fold[e["scene"]] != f]
        teP = [e for e in POS if fold[e["scene"]] == f]; teN = [e for e in NEG if fold[e["scene"]] == f]
        if min(len(trP), len(trN)) < 15 or min(len(teP), len(teN)) < 5:
            continue
        inner = sorted({e["scene"] for e in trP}); sel = set(inner[: max(1, len(inner) // 4)])
        fp = [e for e in trP if e["scene"] not in sel]; fn = [e for e in trN if e["scene"] not in sel]
        sp = [e for e in trP if e["scene"] in sel];     sn = [e for e in trN if e["scene"] in sel]
        if len(fp) < 12 or len(sp) < 5 or len(sn) < 5:
            fp, fn, sp, sn = trP, trN, trP, trN
        v = {"supervised": lambda: sup_dir(fp, fn, seed),
             "random": lambda: rand_dir(fp, 10007 * seed + f),
             "permuted": lambda: perm_dir(fp, fn, 10007 * seed + f)}[kind]()
        a = np.array([auc(proj(sp, v, l), proj(sn, v, l))[0] for l in range(N_LAYERS)])
        pk = int(np.nanargmax(a)) if np.isfinite(a).any() else 0
        peaks.append(pk); profs.append(a)
        for evs, y in ((teP, 1), (teN, 0)):
            for e, x in zip(evs, proj(evs, v, pk)):
                recs.append({"scene": e["scene"], "y": y, "proj": float(x),
                             "area": e["area_px"], "ecc": e["ecc"], "b": e["b"]})
        for t_ in report_negs:
            evs = [e for e in by.get(t_, []) if fold.get(e["scene"]) == f]
            for e, x in zip(evs, proj(evs, v, pk)):
                rec_neg[t_].append({"scene": e["scene"], "y": 0, "proj": float(x)})
    if not recs:
        return None
    P = np.array([r["proj"] for r in recs if r["y"] == 1]); N = np.array([r["proj"] for r in recs if r["y"] == 0])
    a, p = auc(P, N); se = hv_se(a, len(P), len(N))
    prof = np.nanmean(np.stack(profs), 0)
    other = {}
    posrec = [{"scene": r["scene"], "y": 1, "proj": r["proj"]} for r in recs if r["y"] == 1]
    for t_, rs in rec_neg.items():
        if len(rs) < 20:
            continue
        Nn = np.array([r["proj"] for r in rs])
        at, pt = auc(P, Nn); se_t = hv_se(at, len(P), len(Nn))
        other[t_] = {"auc": at, "p": pt, "ci95": [at - 1.96 * se_t, at + 1.96 * se_t],
                     "n_pos": int(len(P)), "n_neg": int(len(Nn)), "records": rs + posrec}
    return {"auc": a, "p": p, "other_negatives": other, "ci95": [a - 1.96 * se, a + 1.96 * se], "n_pos": int(len(P)), "n_neg": int(len(N)),
            "peaks": peaks, "sel_auc_profile_mean": prof.tolist(),
            "peak_layer_profile_argmax": int(np.nanargmax(prof)), "records": recs}


def boot_diff(m, f, n=2000, seed=0):
    def ix(rs):
        d = defaultdict(lambda: ([], []))
        for r in rs:
            d[r["scene"]][r["y"]].append(r["proj"])
        return d
    dm, df = ix(m["records"]), ix(f["records"])
    sc = sorted(set(dm) | set(df)); rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        pm, nm, pf, nf = [], [], [], []
        for i in rng.integers(0, len(sc), len(sc)):
            s = sc[i]
            if s in dm: nm += dm[s][0]; pm += dm[s][1]
            if s in df: nf += df[s][0]; pf += df[s][1]
        if min(len(pm), len(nm), len(pf), len(nf)) < 5:
            continue
        o.append(auc(np.array(pm), np.array(nm))[0] - auc(np.array(pf), np.array(nf))[0])
    o = np.array(o)
    return {"mean": float(o.mean()), "ci95": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
            "p_le_0": float((o <= 0).mean()), "n_boot": int(len(o))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2/dd_cache")
    ap.add_argument("--work", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
    ap.add_argument("--pools", nargs="+", default=["region_mean", "vision_mean"])
    ap.add_argument("--primary-pool", default="vision_mean")
    ap.add_argument("--n-null-seeds", type=int, default=5)
    ap.add_argument("--label", default="DiffusionDrive (diffusiondrive_sim_navhard.ckpt)")
    ap.add_argument("--dir-tag", default="v_hazard_dd")
    ap.add_argument("--native-domain", default="NAVSIM (real)")
    ap.add_argument("--readable-layers", default="8 x TransFuser encoder SelfAttention (320 token = 256 image + 64 BEV latent)")
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/results/g_positive_calibration_diffusiondrive.json")
    args = ap.parse_args()

    work = Path(args.work)
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2bV", "D2cV") if (work / "mining" / f"matched_{t}.txt").exists()}
    VRU = ("human.", "vehicle.bicycle", "vehicle.motorcycle", "walker.")
    OUT = {"model": args.label,
           "native_domain": args.native_domain, "stimuli": "nuScenes G1 语料 + N1 D2a/D2b/D2c/D2cV 负例（与 SimLingo 同一份）",
           "readable_layers": args.readable_layers,
           "arms": {}}

    for pool in args.pools:
        items = load_dd_cache(args.cache, pool)
        by = defaultdict(list)
        for eid, e in items.items():
            t = e["etype"]
            for vt, src in (("D2cV", "D2c"), ("D2bV", "D2b")):
                if t == src and vt in matched and eid in matched[vt] and str(e["object_class"]).startswith(VRU):
                    by[vt].append(e)
            by[t].append(e)
        print(f"\n===== {args.label} pool={pool} =====")
        print("组规模: " + "  ".join(f"{t}={len(by[t])}" for t in sorted(by)))
        arm = {"group_sizes": {t: len(by[t]) for t in sorted(by)}}
        REPORT = [t_ for t_ in ("D2cV", "D2c", "D2b", "D2bV") if len(by.get(t_, [])) >= 20]
        main_r = cv(by, ["A"], "D2a", report_negs=REPORT)
        if main_r is None:
            print("样本不足"); continue
        arm["main_D2a"] = {k: v for k, v in main_r.items() if k not in ("records", "other_negatives")}
        print(f"  主读数 CV-AUC(A vs D2a) = {main_r['auc']:.3f} "
              f"[{main_r['ci95'][0]:.3f},{main_r['ci95'][1]:.3f}] p={main_r['p']:.3g} 峰层={main_r['peaks']}")
        for neg, key in (("D2cV", "floor_D2cV"), ("D2c", "floor_D2c"), ("D2b", "context_D2b"), ("D2bV", "context_D2bV")):
            r = main_r.get("other_negatives", {}).get(neg)
            if r:
                arm[key] = {k: v for k, v in r.items() if k != "records"}
                print(f"  vs {neg:5s} CV-AUC = {r['auc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] p={r['p']:.3g}")
                if neg == "D2cV":
                    arm["main_minus_floor"] = main_r["auc"] - r["auc"]
                    arm["main_minus_floor_bootstrap"] = boot_diff(main_r, r)
                    print(f"  ** 主读数 − D2cV 证伪地板 = {arm['main_minus_floor']:+.3f}, "
                          f"scene 级 bootstrap 95% CI {np.round(arm['main_minus_floor_bootstrap']['ci95'],3).tolist()}")
        # 折分配稳定性
        ms, fs, ds = [], [], []
        for sd in range(10):
            r = cv(by, ["A"], "D2a", seed=sd, report_negs=["D2cV"])
            if r and "D2cV" in r.get("other_negatives", {}):
                ms.append(r["auc"]); fs.append(r["other_negatives"]["D2cV"]["auc"])
                ds.append(ms[-1] - fs[-1])
        if ds:
            arm["stability_across_cv_seeds"] = {
                "n_seeds": len(ds), "main_mean": float(np.mean(ms)), "main_sd": float(np.std(ms)),
                "floor_mean": float(np.mean(fs)), "floor_sd": float(np.std(fs)),
                "diff_mean": float(np.mean(ds)), "diff_sd": float(np.std(ds)),
                "diff_range": [float(np.min(ds)), float(np.max(ds))]}
            print(f"  折分配稳定性（{len(ds)} seed）：主读数 {np.mean(ms):.3f}±{np.std(ms):.3f} "
                  f"地板 {np.mean(fs):.3f}±{np.std(fs):.3f} 差 {np.mean(ds):+.3f}±{np.std(ds):.3f} "
                  f"范围 [{min(ds):+.3f},{max(ds):+.3f}]")

        for kind in ("random", "permuted"):
            a = [cv(by, ["A"], "D2a", seed=s, kind=kind) for s in range(args.n_null_seeds)]
            a = [r["auc"] for r in a if r]
            if a:
                arm[f"{kind}_floor"] = {"aucs": a, "mean": float(np.mean(a)), "sd": float(np.std(a))}
                print(f"  {kind:9s} 地板 = {np.mean(a):.3f} ± {np.std(a):.3f}")
        # 几何稳健性(与 SimLingo 侧同一诊断)
        R = main_r["records"]; pj = np.array([r["proj"] for r in R]); yy = np.array([r["y"] for r in R])
        geo = {}
        for lab, x in (("log_area", np.log(np.array([r["area"] or np.nan for r in R], float))),
                       ("ecc", np.array([r["ecc"] or np.nan for r in R], float))):
            m = np.isfinite(x)
            if m.sum() > 20:
                rho, p = stats.spearmanr(x[m], pj[m])
                geo[lab] = {"all": {"rho": float(rho), "p": float(p), "n": int(m.sum())}}
                for g, lb in ((1, "pos"), (0, "neg")):
                    mm = m & (yy == g)
                    if mm.sum() > 20:
                        r2, p2 = stats.spearmanr(x[mm], pj[mm])
                        geo[lab][lb] = {"rho": float(r2), "p": float(p2), "n": int(mm.sum())}
        arm["geometry_robustness"] = geo
        for k, v in geo.items():
            print(f"  几何稳健性 ρ(投影,{k}) 合并={v['all']['rho']:+.3f} (p={v['all']['p']:.3g})")
        # 冻结全量方向,供 T-I 干涉角使用
        ALLP = by["A"]; ALLN = by["D2a"]
        v_full = sup_dir(ALLP, ALLN, seed=0)
        pk = int(main_r["peak_layer_profile_argmax"])
        np.savez(Path(args.out).parent / f"{args.dir_tag}_{pool}.npz",
                 **{f"L{l}": v_full[l] for l in range(N_LAYERS)}, peak_layer=np.array([pk]))
        arm["frozen_direction"] = {"file": f"{args.dir_tag}_{pool}.npz", "peak_layer_prereg": pk,
                                   "n_pos_fit": len(ALLP), "n_neg_fit": len(ALLN)}
        print(f"  冻结 {args.dir_tag}_{pool}.npz, 预注册峰层 L*={pk}")
        OUT["arms"][pool] = arm

    # H1 判定(g_axis 文档 §2 决判规则)
    # 主口径 = vision_mean —— g_axis 文档 §3 表格明文:"至少跑 vision_mean(与 SimLingo 主读数
    # 口径一致,便于直接对比),其余口径按预算酌情补齐"。SimLingo 侧 n1_report 的主读数
    # (D2a 0.568 / D2cV 0.574) 也正是 vision_mean。region_mean 作为敏感性分析并列报告。
    PRIMARY_POOL = args.primary_pool
    OUT["primary_pool"] = PRIMARY_POOL
    OUT["primary_pool_rationale"] = ("g_axis_positive_calibration_diffusiondrive.md §3 指定 vision_mean "
                                     "为与 SimLingo 主读数可直接对比的口径；region_mean 为敏感性分析。")
    pri = OUT["arms"].get(PRIMARY_POOL) or next(iter(OUT["arms"].values()))
    d = pri.get("main_minus_floor"); ci = pri.get("main_minus_floor_bootstrap", {}).get("ci95", [None, None])
    if d is not None and ci[0] is not None:
        if ci[0] > 0:
            v = (f"H1 成立：G 轴读出口径有效 —— {args.label} 的 D2a 显著高于自身 D2cV 证伪地板，"
                 f"说明该口径能读出「有」；SimLingo 的 FAIL 因此可归为标本属性")
        elif ci[1] < 0:
            v = f"异常：{args.label} 的 D2a 显著低于自身证伪地板，需查管线"
        else:
            v = "不可估：CI 跨 0，本样本量无法区分 H1 与 H-artifact（不得据此宣称方法有效或无效）"
        OUT["verdict_H1"] = v
        sens = OUT["arms"].get("region_mean", {})
        OUT["sensitivity_region_mean"] = {"main_minus_floor": sens.get("main_minus_floor"),
                                          "ci95": sens.get("main_minus_floor_bootstrap", {}).get("ci95")}
        print(f"\n[T-G] H1 判定（主口径 {PRIMARY_POOL}）：{v}")
        print(f"[T-G] 敏感性（region_mean）：主读数−地板 = {sens.get('main_minus_floor')}, "
              f"CI {sens.get('main_minus_floor_bootstrap', {}).get('ci95')} —— 与主口径不一致，须在报告中如实并列")
    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
    print(f"[T-G] wrote {args.out}")


if __name__ == "__main__":
    main()
