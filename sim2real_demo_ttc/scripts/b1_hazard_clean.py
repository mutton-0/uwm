"""Stage B1|清白版感知端方向 v_hazard_clean(G 轴实例)。

计划依据:direction_vector_discovery_validation_plan.md Stage B1 + §2 H1。

"清白"的三个条件(此前从未同时满足过):
  ① 纯 vision-token **区域池化**(region_mean,只看目标框内的图像 token);
  ② **中性 prompt**(缓存的 prompt_anchor=clean,clean/ghost 逐字节相同,不涉及语言生成);
  ③ N1 已修复的**几何平衡负例** D2a(与正例在 log 面积/离心率上 caliper 匹配)。

本脚本做四件事:
  1. scene 级 4 折 CV 读数 —— 主读数 = CV-AUC(A vs D2a) − 证伪地板(D2cV),
     并列随机方向地板与标签置换零分布(各多 seed);
  2. 几何稳健性 —— 投影值与 bbox 面积/离心率的相关(方向是否又在读"大而居中");
  3. 全量数据拟合并**冻结**方向 v_hazard_clean.npy + 预注册峰层 L*;
  4. H1 —— v_hazard_clean vs v_brake 的逐层余弦 + 事件级投影相关,
     与 v_danger_lang vs v_brake 对照。

--arm carla 走同一套流程,但对照换成 M1 的 Hcar vs Ncar(CARLA in-domain,专家减速归因金标准),
产出 v_hazard_carla.npy —— 站内唯一一条已知有信号的 G 轴方向,供 Stage C/D/E 当锚。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache                                    # noqa: E402
from n1_readout import (auc, proj, supervised_direction,             # noqa: E402
                        permuted_direction, pca_direction)

VRU = ("human.", "vehicle.bicycle", "vehicle.motorcycle", "walker.")


def hv_se(a, n1, n2):
    Q1 = a / (2 - a); Q2 = 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (Q1 - a * a) + (n2 - 1) * (Q2 - a * a)) / (n1 * n2)))


def load_groups(cfg_path, pool, arm):
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, pool, keep=set(evmap))
    TY = ("D", "D2a", "D2aP", "D2b", "D2bV", "D2c", "D2cV", "Ncar")
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in TY if (work / "mining" / f"matched_{t}.txt").exists()}
    by = defaultdict(list)
    for eid, e in items.items():
        m = evmap[eid]
        e["etype"] = m["event_type"]
        e["area_px"] = m.get("area_px"); e["ecc"] = m.get("ecc")
        t = m["event_type"]
        if arm == "nuscenes":
            for vt, src in (("D2cV", "D2c"), ("D2bV", "D2b")):
                if t == src and vt in matched and eid in matched[vt] \
                        and str(m.get("object_class", "")).startswith(VRU):
                    by[vt].append(e)
        if t in matched and eid not in matched[t]:
            continue
        by[t].append(e)
    return cfg, work, by


def cv_records(by, pos_types, neg_type, folds=4, seed=0, direction="supervised", report_negs=None):
    """scene 级 K 折 + 折内嵌套选层。返回逐事件 held-out 投影记录(带 scene 标签,供 bootstrap)。

    **口径(与 n1_cv.py 一致,修正案 DV/A16)**:方向与峰层**只**在 A vs D2a 上拟合/选择,
    其余负类(D2cV 证伪地板、D2b 上下文…)用**同一条方向、同一个峰层**投影后报数。
    这才是"证伪地板"的正确含义 —— 量的是"这条 A-vs-D2a 方向有多少只是'A vs 任意 VRU'";
    若对每个负类各自重新拟合一条方向,量到的是另一件事(能不能找到一条分开它们的方向),
    数值系统性偏高,且与 n1_report 已发表的数字不可比。
    """
    report_negs = report_negs or []
    POS = [e for t in pos_types for e in by.get(t, [])]
    NEG = by.get(neg_type, [])
    if len(POS) < 20 or len(NEG) < 20:
        return None
    # fold 映射覆盖**缓存里全部事件类型**的场景 —— 与 n1_cv.py 逐字一致，
    # 这样本脚本的 CV-AUC 与 n1_report 已发表的数字可直接对照。
    # （首版只覆盖 POS+NEG，折分配不同 ⇒ 同一读数会漂 ±0.08，见修正案 DV/A16。）
    scenes = sorted({e["scene"] for v in by.values() for e in v})
    rng = np.random.default_rng(seed)
    fold = {s: i % folds for i, s in enumerate(rng.permutation(scenes))}
    recs, peaks, profiles = [], [], []
    rec_neg = {t: [] for t in report_negs}
    for f in range(folds):
        trP = [e for e in POS if fold[e["scene"]] != f]; trN = [e for e in NEG if fold[e["scene"]] != f]
        teP = [e for e in POS if fold[e["scene"]] == f]; teN = [e for e in NEG if fold[e["scene"]] == f]
        if len(trP) < 15 or len(trN) < 15 or len(teP) < 5 or len(teN) < 5:
            continue
        inner = sorted({e["scene"] for e in trP})
        sel = set(inner[: max(1, len(inner) // 4)])
        fp = [e for e in trP if e["scene"] not in sel]; fn = [e for e in trN if e["scene"] not in sel]
        sp = [e for e in trP if e["scene"] in sel];     sn = [e for e in trN if e["scene"] in sel]
        if len(fp) < 12 or len(sp) < 5 or len(sn) < 5:
            fp, fn, sp, sn = trP, trN, trP, trN
        if direction == "random":
            rs = np.random.default_rng(10007 * seed + f)
            L, C = fp[0]["h_ghost"].shape
            v = rs.normal(size=(L, C)).astype(np.float32)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
        elif direction == "permuted":
            v = permuted_direction(fp, fn, seed=10007 * seed + f)
        elif direction == "pca":
            v = pca_direction(fp, fn, seed=seed)
        else:
            v = supervised_direction(fp, fn, seed=seed)
        a = np.array([auc(proj(sp, v, l), proj(sn, v, l))[0] for l in range(v.shape[0])])
        pk = int(np.nanargmax(a)) if np.isfinite(a).any() else v.shape[0] // 2
        peaks.append(pk); profiles.append(a)
        for grp, evs in (("pos", teP), ("neg", teN)):
            pr = proj(evs, v, pk)
            for e, x in zip(evs, pr):
                recs.append({"scene": e["scene"], "y": 1 if grp == "pos" else 0, "proj": float(x),
                             "area": e.get("area_px"), "ecc": e.get("ecc"), "b": e.get("b")})
        for t in report_negs:                       # 同一条方向、同一个峰层投影其余负类
            evs = [e for e in by.get(t, []) if fold.get(e["scene"]) == f]
            for e, x in zip(evs, proj(evs, v, pk)):
                rec_neg[t].append({"scene": e["scene"], "y": 0, "proj": float(x),
                                   "area": e.get("area_px"), "ecc": e.get("ecc"), "b": e.get("b")})
    if not recs:
        return None
    P = np.array([r["proj"] for r in recs if r["y"] == 1])
    N = np.array([r["proj"] for r in recs if r["y"] == 0])
    a, p = auc(P, N)
    se = hv_se(a, len(P), len(N))
    prof = np.nanmean(np.stack(profiles), 0)
    other = {}
    for t, rs in rec_neg.items():
        if len(rs) < 20:
            continue
        Nn = np.array([r["proj"] for r in rs])
        at, pt = auc(P, Nn); se_t = hv_se(at, len(P), len(Nn))
        other[t] = {"auc": at, "p": pt, "ci95": [at - 1.96 * se_t, at + 1.96 * se_t],
                    "n_pos": int(len(P)), "n_neg": int(len(Nn)),
                    "records": [{"scene": r["scene"], "y": 0, "proj": r["proj"]} for r in rs] +
                               [{"scene": r["scene"], "y": 1, "proj": r["proj"]} for r in recs if r["y"] == 1]}
    return {"auc": a, "p": p, "other_negatives": other, "ci95": [a - 1.96 * se, a + 1.96 * se],
            "n_pos": int(len(P)), "n_neg": int(len(N)), "peaks": peaks,
            "sel_auc_profile_mean": prof.tolist(),
            "peak_layer_profile_argmax": int(np.nanargmax(prof)), "records": recs}


def boot_auc_diff(rec_main, rec_floor, n=2000, seed=0):
    """scene 级 bootstrap:主读数 AUC − 证伪地板 AUC 的 95% CI。

    两条读数共享同一批正例场景,故按 scene 重采样时**两边同步取同一批 scene**,
    保留配对结构(否则 CI 会被两条独立抽样的方差之和撑宽)。
    """
    def index(rs):
        d = defaultdict(lambda: ([], []))
        for r in rs:
            d[r["scene"]][r["y"]].append(r["proj"])
        return d
    dm, df = index(rec_main["records"]), index(rec_floor["records"])
    scenes = sorted(set(dm) | set(df))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.choice(len(scenes), size=len(scenes), replace=True)
        pm, nm, pf, nf = [], [], [], []
        for i in pick:
            s = scenes[i]
            if s in dm:
                nm += dm[s][0]; pm += dm[s][1]
            if s in df:
                nf += df[s][0]; pf += df[s][1]
        if min(len(pm), len(nm), len(pf), len(nf)) < 5:
            continue
        out.append(auc(np.array(pm), np.array(nm))[0] - auc(np.array(pf), np.array(nf))[0])
    o = np.array(out)
    return {"mean": float(o.mean()), "ci95": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
            "n_boot": int(len(o)), "p_ge_0": float((o <= 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="nuscenes", choices=["nuscenes", "carla"])
    ap.add_argument("--pool", default="region_mean")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--n-null-seeds", type=int, default=5)
    ap.add_argument("--stability-seeds", type=int, default=10,
                    help="主读数−地板 在多少个 CV 折分配 seed 上重复（量化折分配带来的漂移）")
    ap.add_argument("--out-dir", default="/data/ruolin/uwm/sim2real_demo_ttc/results")
    ap.add_argument("--tag", default="", help="非空时输出文件加后缀（敏感性口径，不覆盖主口径产出物）")
    ap.add_argument("--pos-types", nargs="+", default=None,
                    help="覆盖正例类别（§2.5 第二场景类型复现：换触发判据，其余口径一律不动）")
    args = ap.parse_args()

    if args.arm == "nuscenes":
        cfgp = "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"
        POS, NEG, FLOOR, name = ["A"], "D2a", "D2cV", "v_hazard_clean" + args.tag
    else:
        cfgp = "/data/ruolin/uwm/sim2real_demo_ttc/configs/m1_carla.yaml"
        POS, NEG, FLOOR, name = ["Hcar"], "Ncar", None, "v_hazard_carla" + args.tag

    if args.pos_types:
        POS = list(args.pos_types)
        name = f"v_hazard_{'_'.join(POS).lower()}" + args.tag
    cfg, work, by = load_groups(cfgp, args.pool, args.arm)
    print(f"[B1|{args.arm}] pool={args.pool}  组规模: " +
          "  ".join(f"{t}={len(by[t])}" for t in sorted(by) if len(by[t])))

    res = {"arm": args.arm, "pool": args.pool, "folds": args.folds,
           "pos_types": POS, "neg_type": NEG, "floor_type": FLOOR}

    REPORT_NEGS = [t for t in ([FLOOR, "D2c", "D2b", "D2bV", "D"] if FLOOR else []) if t and by.get(t)]
    main_r = cv_records(by, POS, NEG, args.folds, 0, "supervised", report_negs=REPORT_NEGS)
    assert main_r, "主读数样本不足"
    res["main"] = {k: v for k, v in main_r.items() if k not in ("records", "other_negatives")}
    res["other_negatives"] = {t: {k: v for k, v in r.items() if k != "records"}
                              for t, r in main_r.get("other_negatives", {}).items()}
    for t, r in main_r.get("other_negatives", {}).items():
        print(f"[B1]   同一方向投影 vs {t:5s}: AUC={r['auc']:.3f} "
              f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] p={r['p']:.3g} (n_neg={r['n_neg']})")
    print(f"[B1] 主读数 CV-AUC({'+'.join(POS)} vs {NEG}) = {main_r['auc']:.3f} "
          f"[{main_r['ci95'][0]:.3f},{main_r['ci95'][1]:.3f}]  p={main_r['p']:.3g}  峰层={main_r['peaks']}")

    floor_r = None
    if FLOOR:
        floor_r = main_r.get("other_negatives", {}).get(FLOOR)
        if floor_r:
            res["falsification_floor"] = {k: v for k, v in floor_r.items() if k != "records"}
            res["floor_convention"] = "同一条 A-vs-D2a 方向、同一峰层投影 D2cV（与 n1_cv 一致，修正案 DV/A16）"
            res["main_minus_floor"] = main_r["auc"] - floor_r["auc"]
            res["main_minus_floor_bootstrap"] = boot_auc_diff(main_r, floor_r)
            print(f"[B1] 证伪地板 CV-AUC(vs {FLOOR}) = {floor_r['auc']:.3f}; "
                  f"主读数−地板 = {res['main_minus_floor']:+.3f}  "
                  f"bootstrap 95% CI {res['main_minus_floor_bootstrap']['ci95']}")

    # ---- 折分配稳定性：同一读数在多个 CV seed 上重复 ----
    if args.stability_seeds > 1 and FLOOR:
        ms, fs, ds = [], [], []
        for sd in range(args.stability_seeds):
            r = cv_records(by, POS, NEG, args.folds, sd, "supervised", report_negs=[FLOOR])
            if not r or FLOOR not in r.get("other_negatives", {}):
                continue
            ms.append(r["auc"]); fs.append(r["other_negatives"][FLOOR]["auc"])
            ds.append(r["auc"] - r["other_negatives"][FLOOR]["auc"])
        if ds:
            res["stability_across_cv_seeds"] = {
                "n_seeds": len(ds), "main_mean": float(np.mean(ms)), "main_sd": float(np.std(ms)),
                "floor_mean": float(np.mean(fs)), "floor_sd": float(np.std(fs)),
                "diff_mean": float(np.mean(ds)), "diff_sd": float(np.std(ds)),
                "diff_range": [float(np.min(ds)), float(np.max(ds))],
                "diff_ci95_across_seeds": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
            print(f"[B1] 折分配稳定性（{len(ds)} 个 seed）: 主读数 {np.mean(ms):.3f}±{np.std(ms):.3f}  "
                  f"地板 {np.mean(fs):.3f}±{np.std(fs):.3f}  差 {np.mean(ds):+.3f}±{np.std(ds):.3f} "
                  f"范围 [{min(ds):+.3f},{max(ds):+.3f}]")

    for dname in ("random", "permuted"):
        arr = []
        for s in range(args.n_null_seeds):
            r = cv_records(by, POS, NEG, args.folds, s, dname)
            if r:
                arr.append(r["auc"])
        if arr:
            res[f"{dname}_floor"] = {"aucs": arr, "mean": float(np.mean(arr)), "sd": float(np.std(arr)),
                                     "q95": float(np.percentile(arr, 95))}
            print(f"[B1] {dname:9s} 地板 = {np.mean(arr):.3f} ± {np.std(arr):.3f}  (n_seed={len(arr)})")
    rp = cv_records(by, POS, NEG, args.folds, 0, "pca")
    if rp:
        res["pca_arm"] = {k: v for k, v in rp.items() if k != "records"}
        print(f"[B1] PCA 对照臂 = {rp['auc']:.3f}（敏感性分析,不参与判定）")

    # ---- 几何稳健性:方向是否又在读"大而居中" ----
    R = main_r["records"]
    ar = np.array([r["area"] if r["area"] else np.nan for r in R], float)
    ec = np.array([r["ecc"] if r["ecc"] else np.nan for r in R], float)
    pj = np.array([r["proj"] for r in R], float)
    yy = np.array([r["y"] for r in R])
    geo = {}
    for lab, x in (("log_area", np.log(ar)), ("ecc", ec)):
        m = np.isfinite(x) & np.isfinite(pj)
        if m.sum() > 20:
            rho, p = stats.spearmanr(x[m], pj[m])
            gg = {"all": {"rho": float(rho), "p": float(p), "n": int(m.sum())}}
            for g, lb in ((1, "pos"), (0, "neg")):     # §0.4 相关系数纪律:先分组再合并
                mm = m & (yy == g)
                if mm.sum() > 20:
                    r2, p2 = stats.spearmanr(x[mm], pj[mm])
                    gg[lb] = {"rho": float(r2), "p": float(p2), "n": int(mm.sum())}
            geo[lab] = gg
    res["geometry_robustness"] = geo
    for k, v in geo.items():
        print(f"[B1] 几何稳健性 ρ(投影, {k}) 合并={v['all']['rho']:+.3f} (p={v['all']['p']:.3g}) | " +
              " ".join(f"{g}={v[g]['rho']:+.3f}" for g in ("pos", "neg") if g in v))

    # ---- 冻结方向 + 预注册峰层 ----
    ALLP = [e for t in POS for e in by[t]]; ALLN = by[NEG]
    v_full = supervised_direction(ALLP, ALLN, seed=0)
    # 峰层规则(修正案 DV/A12,见 amendments.md):用**逐折 S_sel 层剖面的均值**取 argmax,
    # 而不是"逐折 argmax 的众数" —— 后者在 2-2 平局时会退化成"取层号最小者",
    # CARLA 臂上恰好命中该退化(峰层 [8,8,0,0])。均值剖面无平局退化问题。
    pk = int(main_r["peak_layer_profile_argmax"])
    pk_mode = int(stats.mode(main_r["peaks"], keepdims=False).mode) if main_r["peaks"] else 0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", v_full.astype(np.float32))
    res["frozen_direction"] = {"file": f"{name}.npy", "shape": list(v_full.shape),
                               "peak_layer_prereg": pk,
                               "peak_layer_rule": "4 折 S_sel 层剖面均值的 argmax(修正案 DV/A12)",
                               "peak_layer_mode_sensitivity": pk_mode,
                               "n_pos_fit": len(ALLP), "n_neg_fit": len(ALLN)}
    print(f"[B1] 冻结 {name}.npy {v_full.shape}, 预注册峰层 L*={pk}")

    # ---- H1:与行为轴 v_brake 的关系 ----
    ref = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2/results")
    h1 = {}
    for other, f in (("v_brake", "v_brake_query_mean.npy"),
                     ("v_danger_lang", "v_danger_lang_query_mean.npy")):
        if (ref / f).exists():
            w = np.load(ref / f)
            if w.shape == v_full.shape:
                cs = [float(np.dot(v_full[l], w[l]) /
                            (np.linalg.norm(v_full[l]) * np.linalg.norm(w[l]) + 1e-9))
                      for l in range(v_full.shape[0])]
                h1[f"cos({name},{other})"] = {"by_layer": cs, "at_peak": cs[pk],
                                              "max_abs": float(np.max(np.abs(cs))),
                                              "argmax_abs": int(np.argmax(np.abs(cs)))}
    # 事件级:投影 vs 行为量 b(分组报告后再合并,§0.4)
    bb = np.array([r["b"] if r["b"] is not None else np.nan for r in R], float)
    m = np.isfinite(bb) & np.isfinite(pj)
    if m.sum() > 20:
        rho, p = stats.spearmanr(pj[m], bb[m])
        h1["rho(proj, behavior_b)"] = {"all": {"rho": float(rho), "p": float(p), "n": int(m.sum())}}
        for g, lb in ((1, "pos"), (0, "neg")):
            mm = m & (yy == g)
            if mm.sum() > 20:
                r2, p2 = stats.spearmanr(pj[mm], bb[mm])
                h1["rho(proj, behavior_b)"][lb] = {"rho": float(r2), "p": float(p2), "n": int(mm.sum())}
    res["H1"] = h1
    for k, v in h1.items():
        if "by_layer" in v:
            print(f"[B1|H1] {k}: 峰层={v['at_peak']:+.3f}  max|cos|={v['max_abs']:.3f} @L{v['argmax_abs']}")
        else:
            print(f"[B1|H1] {k}: 合并 ρ={v['all']['rho']:+.3f} (p={v['all']['p']:.3g})")

    (out_dir / f"b1_{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[B1] wrote results/b1_{name}.json + {name}.npy")


if __name__ == "__main__":
    main()
