"""Stage E|效度验收:两条红线核验 + E1–E4 判定表。

计划依据:direction_vector_discovery_validation_plan.md Stage E。

**红线 1｜跨分布泛化**:每个方向必须在**从未参与筛选/提取**的 held-out 上复测。
  1a 未参与的场景类别:n1_d2 的 B(近距 cut-in)与 C(通用 TTC 骤降)——方向只在 A vs D2a 上拟合、
     峰层只在 A vs D2a 上选,B/C 从头到尾没进过任何拟合或选择;
  1b 跨域:v_hazard_carla(CARLA 提)→ nuScenes 复测,v_hazard_clean(nuScenes 提)→ CARLA 复测;
  1c 独立挖掘轮:tier_m(不同 clean/ghost 窗口与判据,不同事件集)。
  **一律使用冻结方向 + 冻结层,不重新拟合、不重新选层。**

**红线 2｜强 ground truth**:效度必须挂真实行为指标,不能只挂方向自身读数自洽。
  ① nuScenes:人类驾驶员真实纵向减速 a_brake(ego_pose 重算,`f_axis_abrake.json`)——外部真值;
  ② CARLA:专家逐帧减速归因(`speed_reduced_by_obj_*`,M1 的 Hcar/Ncar 金标准划分)——金标准;
  ③ 模型行为量 b = v_plan(clean) − v_plan(ghost) —— **代理量**,证据等级低于 ①②,单列标注。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache          # noqa: E402
from n1_readout import auc                 # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def hv_se(a, n1, n2):
    Q1 = a / (2 - a); Q2 = 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (Q1 - a * a) + (n2 - 1) * (Q2 - a * a)) / (n1 * n2)))


def load_by_type(cfg_path, pool):
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, pool, keep=set(evmap), verbose=False)
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a", "D2b", "D2c", "D2cV", "Ncar")
               if (work / "mining" / f"matched_{t}.txt").exists()}
    by = defaultdict(list)
    for eid, e in items.items():
        t = evmap[eid]["event_type"]
        if t in matched and eid not in matched[t]:
            continue
        e["etype"] = t
        by[t].append(e)
    return by, items, evmap


def projd(evs, v, L):
    return np.array([float((e["h_ghost"][L] - e["h_clean"][L]) @ v[L]) for e in evs]) if evs else np.zeros(0)


def degenerate(P, N):
    """池化在该缓存里结构性缺失时（如 tier_m 无 bbox ⇒ region_mean 恒为 0），
    投影全为常数，AUC 恒等于 0.500 —— 必须报'不可估'，不得当成读数。"""
    x = np.concatenate([P, N])
    return bool(len(x) == 0 or np.allclose(x, x[0], atol=1e-12))


def auc_ci(P, N):
    if len(P) < 5 or len(N) < 5:
        return None
    a, p = auc(P, N); se = hv_se(a, len(P), len(N))
    return {"auc": a, "p": p, "ci95": [a - 1.96 * se, a + 1.96 * se], "n_pos": len(P), "n_neg": len(N)}


def rand_floor(pos, neg, L, C, n=20, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        w = rng.normal(size=C); w /= np.linalg.norm(w)
        vv = np.zeros((L + 1, C), np.float32); vv[L] = w
        r = auc_ci(projd(pos, vv, L), projd(neg, vv, L))
        if r:
            out.append(r["auc"])
    return {"mean": float(np.mean(out)), "sd": float(np.std(out)), "n": len(out),
            "q95": float(np.percentile(out, 95))} if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "e1_validity.json"))
    args = ap.parse_args()
    OUT = {"red_line_1_cross_distribution": {}, "red_line_2_strong_ground_truth": {}}

    DIRS = {}
    for nm, f, L, pool in (
            ("v_hazard_clean", RES / "v_hazard_clean.npy", None, "region_mean"),
            ("v_hazard_carla", RES / "v_hazard_carla.npy", None, "region_mean"),
            ("v_hazard_clean_vision", RES / "v_hazard_clean_vision.npy", None, "vision_mean"),
            ("v_danger_lang", RES.parent / "variants/n1_d2/results/v_danger_lang_query_mean.npy", 10, "query_mean"),
            ("v_brake", RES.parent / "variants/n1_d2/results/v_brake_query_mean.npy", 22, "query_mean")):
        if not Path(f).exists():
            continue
        v = np.load(f)
        if L is None:
            j = json.load(open(RES / f"b1_{nm}.json"))
            L = j["frozen_direction"]["peak_layer_prereg"]
        DIRS[nm] = {"v": v, "layer": int(L), "pool": pool, "file": str(Path(f).name)}
    print("[E] 冻结方向:", {k: (v["layer"], v["pool"]) for k, v in DIRS.items()})

    NUS = "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"
    CAR = "/data/ruolin/uwm/sim2real_demo_ttc/configs/m1_carla.yaml"
    TM3 = "/data/ruolin/uwm/sim2real_demo_ttc/configs/tier_m_v3.yaml"
    cache = {}
    def get(cfg, pool):
        if (cfg, pool) not in cache:
            cache[(cfg, pool)] = load_by_type(cfg, pool)
        return cache[(cfg, pool)]

    # ---------- 红线 1a：从未参与拟合/选层的**正例类别与负例类别** ----------
    # 关键纪律（修正案 DV/A17）：held-out 必须**正负两侧都没进过拟合**。
    # v_hazard_clean / v_hazard_carla 分别在 (A, D2a) / (Hcar, Ncar) 上做过全量拟合，
    # 因此 "B vs D2a" 并不是干净的 held-out —— 负例侧 D2a 被拟合推低过，AUC 会虚高。
    # 干净组合：正例 ∈ {B, C}（从未参与）× 负例 ∈ {D, D2b}（从未参与拟合）。
    FIT_SETS = {"v_hazard_clean": {"A", "D2a"}, "v_hazard_clean_vision": {"A", "D2a"}, "v_hazard_carla": set(),
                "v_danger_lang": set(), "v_brake": set()}   # 后两者在语言/行为标签上提，未用视觉类别
    r1a = {}
    for nm, D in DIRS.items():
        by, _, _ = get(NUS, D["pool"])
        v, L, C = D["v"], D["layer"], D["v"].shape[1]
        fit = FIT_SETS.get(nm, set())
        ent = {}
        for pos_t in ("B", "C", "A"):
            for neg_t in ("D", "D2b", "D2a"):
                P, N = by.get(pos_t, []), by.get(neg_t, [])
                r = auc_ci(projd(P, v, L), projd(N, v, L))
                if not r:
                    continue
                clean = (pos_t not in fit) and (neg_t not in fit)
                rf = rand_floor(P, N, L, C)
                ent[f"{pos_t}_vs_{neg_t}"] = {**r, "random_floor": rf,
                                              "held_out_clean": clean,
                                              "excess_over_random": r["auc"] - (rf["mean"] if rf else np.nan)}
        r1a[nm] = ent
        for k, r in sorted(ent.items(), key=lambda kv: -kv[1]["held_out_clean"]):
            flag = "✅干净held-out" if r["held_out_clean"] else "⚠️含拟合集，仅作 sanity（非效度读数）"
            print(f"[E/红线1a] {nm:16s} {k:10s} AUC={r['auc']:.3f} [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] "
                  f"随机地板={r['random_floor']['mean']:.3f}±{r['random_floor']['sd']:.3f} "
                  f"超出={r['excess_over_random']:+.3f} {flag}")
    OUT["red_line_1_cross_distribution"]["1a_unseen_categories"] = r1a
    OUT["red_line_1_cross_distribution"]["1a_note"] = (
        "held_out_clean=true 才是效度读数；false 的行含方向拟合时用过的类别，"
        "按构造必然虚高（v_hazard_clean 在 A vs D2a 上得 AUC=1.000 即为此），仅作管线 sanity。")

    # ---------- 红线 1b：跨域（CARLA ↔ nuScenes），冻结方向直接复测 ----------
    r1b = {}
    byc, _, _ = get(CAR, "region_mean")
    for nm, D in DIRS.items():
        if D["pool"] != "region_mean":
            continue
        v, L, C = D["v"], D["layer"], D["v"].shape[1]
        if D["pool"] != "region_mean":
            continue
        # nuScenes 提的方向 → CARLA 复测（对 v_hazard_carla 而言是**同域同集**，标注为非 held-out）
        r = auc_ci(projd(byc.get("Hcar", []), v, L), projd(byc.get("Ncar", []), v, L))
        if r:
            rf = rand_floor(byc.get("Hcar", []), byc.get("Ncar", []), L, C)
            r1b[f"{nm}_on_CARLA_Hcar_vs_Ncar"] = {**r, "random_floor": rf,
                                                  "excess_over_random": r["auc"] - rf["mean"],
                                                  "held_out_clean": nm != "v_hazard_carla"}
            print(f"[E/红线1b] {nm:16s} → CARLA(Hcar vs Ncar) AUC={r['auc']:.3f} "
                  f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] 随机地板={rf['mean']:.3f} "
                  f"超出={r['auc']-rf['mean']:+.3f} "
                  f"{'✅干净held-out' if nm != 'v_hazard_carla' else '⚠️同集（拟合用），非效度读数'}")
    # CARLA 提的方向 → nuScenes 复测（对 v_hazard_carla 而言是干净的跨域 held-out）
    byn, _, _ = get(NUS, "region_mean")
    for nm in ("v_hazard_carla",):
        if nm not in DIRS:
            continue
        D = DIRS[nm]; v, L, C = D["v"], D["layer"], D["v"].shape[1]
        for pos_t, neg_t in (("A", "D2a"), ("A", "D2cV")):
            r = auc_ci(projd(byn.get(pos_t, []), v, L), projd(byn.get(neg_t, []), v, L))
            if r:
                rf = rand_floor(byn.get(pos_t, []), byn.get(neg_t, []), L, C)
                r1b[f"{nm}_on_nuScenes_{pos_t}_vs_{neg_t}"] = {
                    **r, "random_floor": rf, "excess_over_random": r["auc"] - rf["mean"],
                    "held_out_clean": True}
                print(f"[E/红线1b] {nm:16s} → nuScenes({pos_t} vs {neg_t}) AUC={r['auc']:.3f} "
                      f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] 随机地板={rf['mean']:.3f} "
                      f"超出={r['auc']-rf['mean']:+.3f} ✅干净held-out")
    OUT["red_line_1_cross_distribution"]["1b_cross_domain"] = r1b

    # ---------- 红线 1c：独立挖掘轮 tier_m ----------
    r1c = {}
    tm_cache = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/tier_m_v3/cache")
    if Path(TM3).exists() and len(list(tm_cache.glob("*.h5"))) > 100:
        for nm, D in DIRS.items():
            byt, _, _ = get(TM3, D["pool"])
            v, L, C = D["v"], D["layer"], D["v"].shape[1]
            P, N = projd(byt.get("A", []), v, L), projd(byt.get("D", []), v, L)
            if degenerate(P, N):
                r1c[f"{nm}_on_tier_m_A_vs_D"] = {
                    "status": "不可估（该口径在 tier_m 缓存里结构性缺失）",
                    "reason": "tier_m 挖掘轮早于 bbox 字段，region_* 池化恒为 0，投影无方差",
                    "pool": D["pool"], "held_out_clean": True}
                print(f"[E/红线1c] {nm:16s} → tier_m: **不可估**（{D['pool']} 在该轮缓存里恒为 0，无 bbox）")
                continue
            r = auc_ci(P, N)
            if r:
                rf = rand_floor(byt.get("A", []), byt.get("D", []), L, C)
                r1c[f"{nm}_on_tier_m_A_vs_D"] = {**r, "random_floor": rf,
                                                 "excess_over_random": r["auc"] - rf["mean"],
                                                 "held_out_clean": True}
                print(f"[E/红线1c] {nm:16s} → tier_m(A vs D) AUC={r['auc']:.3f} "
                      f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}] 随机地板={rf['mean']:.3f} 超出={r['auc']-rf['mean']:+.3f}")
    else:
        r1c["status"] = f"tier_m_v3 缓存未就绪（{len(list(tm_cache.glob('*.h5')))} 个 h5）"
        print(f"[E/红线1c] {r1c['status']}，跳过")
    OUT["red_line_1_cross_distribution"]["1c_independent_mining_round"] = r1c

    # ---------- 红线 2：强 ground truth ----------
    ab = json.load(open(RES / "f_axis_abrake.json"))["events"]
    r2 = {}
    for nm, D in DIRS.items():
        by, items, _ = get(NUS, D["pool"])
        v, L = D["v"], D["layer"]
        ev = [e for t in ("A", "B", "C") for e in by.get(t, [])]
        pj = projd(ev, v, L)
        ids = [e["meta"]["event_id"] for e in ev]
        sc = [e["scene"] for e in ev]
        a = np.array([ab[i]["a_brake"] if i in ab else np.nan for i in ids])
        v0 = np.array([ab[i]["v_at_emergence"] if i in ab else np.nan for i in ids])
        b = np.array([e["b"] for e in ev])
        m = np.isfinite(a) & np.isfinite(v0) & np.isfinite(pj)
        ent = {}
        if m.sum() > 30:
            A_ = np.vstack([v0[m], np.ones(m.sum())]).T
            co, *_ = np.linalg.lstsq(A_, -a[m], rcond=None)
            y = -a[m] - A_ @ co
            rho, p = stats.spearmanr(pj[m], y)
            byS = defaultdict(list)
            for i, s in enumerate(np.array(sc)[m]):
                byS[s].append(i)
            ks = sorted(byS); rng = np.random.default_rng(0); bs = []
            xx, yy = pj[m], y
            for _ in range(2000):
                idx = np.concatenate([byS[ks[i]] for i in rng.integers(0, len(ks), len(ks))])
                bs.append(stats.spearmanr(xx[idx], yy[idx]).statistic)
            ent["gt1_real_human_abrake"] = {
                "rho": float(rho), "p": float(p), "n": int(m.sum()),
                "ci95_scene_bootstrap": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "evidence_grade": "强（外部真值：人类驾驶员真实纵向减速）"}
        mb = np.isfinite(b) & np.isfinite(pj)
        if mb.sum() > 30:
            rho, p = stats.spearmanr(pj[mb], b[mb])
            ent["gt3_model_behavior_b_PROXY"] = {"rho": float(rho), "p": float(p), "n": int(mb.sum()),
                                                 "evidence_grade": "弱（代理量：模型自身输出差，与方向读数耦合）"}
        # CARLA 金标准（专家减速归因划分的 Hcar/Ncar）
        if D["pool"] == "region_mean":
            byc, _, _ = get(CAR, "region_mean")
            r = auc_ci(projd(byc.get("Hcar", []), v, L), projd(byc.get("Ncar", []), v, L))
            if r:
                ent["gt2_carla_expert_attribution"] = {
                    **r, "held_out_clean": nm != "v_hazard_carla",
                    "evidence_grade": ("金标准（CARLA 专家逐帧减速归因 speed_reduced_by_obj）"
                                       + ("" if nm != "v_hazard_carla"
                                          else " —— ⚠️ 该方向即在此对照上拟合，本行非效度读数"))}
        r2[nm] = ent
        for k, e in ent.items():
            if "rho" in e:
                print(f"[E/红线2] {nm:16s} {k:32s} ρ={e['rho']:+.3f} p={e['p']:.3g} n={e['n']}  [{e['evidence_grade']}]")
            else:
                print(f"[E/红线2] {nm:16s} {k:32s} AUC={e['auc']:.3f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]  [{e['evidence_grade']}]")
    OUT["red_line_2_strong_ground_truth"] = r2

    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False, default=float))
    print(f"[E] wrote {args.out}")


if __name__ == "__main__":
    main()
