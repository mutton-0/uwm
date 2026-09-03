"""F-3 弧长读数|action 换成**轨迹自身的伸展量**，不引入任何外部参照点。

工单：本轮（2026-09-03 第二份）。**纯重新分析**已落盘的 `f3_occlusion_*_dist.json`，
**不跑任何新的前向推理**。旧结果不改不删，新结果一律 `_arclen` 后缀。

## 为什么换这个量

上一轮（§FD/A63）的"轨迹终点到危险实体的距离"在跨帧比较下被几何严重污染：
$b^{(d)}_{ghost}$ 的 81%–108% 来自**实体自身走近**（纵距中位 30.5 → 24.6 m），
外加两帧间**ego 自身位移中位 7.2–8.2 m**，两项都无法从跨帧比较里消除。
根因是**距离依赖危险实体的绝对位置，不是"起点不变量"**。

本轮换成只用轨迹自己的 waypoint 序列算的量：

$$L^{arc}_{plan} = \\lVert w_0\\rVert + \\sum_{i} \\lVert w_{i+1}-w_i\\rVert,
\\qquad L^{end}_{plan} = \\lVert w_{-1}\\rVert$$

（$w$ 是 ego 系 waypoint，ego 原点即该帧自车位置。工单要求两种定义都算并对比。）

**起点不变性是构造性的**：两式都只由 waypoint 之间以及 waypoint 到 ego 原点的相对位移构成，
**不含实体位置、也不含 ego 的世界位姿** ⇒ 结构上不可能出现上一轮那种几何混淆。
本模块另做一次**实测核验**（§start_invariance）：把逐事件的 $b^{(L)}_{ghost}$ 与
上一轮存下的两帧间 ego 位移做相关，并与 $b^{(d)}_{ghost}$ 的同一相关并列对比。

## 两种对比方式（工单要求都做，并说明是否一致）

* **(a) 同帧配对**：$\\delta^{(L)}_{occ} = L(occ)-L(ghost)$ —— 同一事件、同一帧，
  唯一变量是那块灰斑；配 ctrl 对照臂。与上一轮 $\\delta_{occ}$ 的分析结构一致。
* **(b) 分组均值**：把所有事件的 occ 臂当一组、ghost 臂当另一组，各算组均值再相减。

**数学上**：同一批事件、且两臂都无缺失时，
$\\overline{L(occ)-L(ghost)} = \\overline{L(occ)}-\\overline{L(ghost)}$（均值的线性性）
⇒ **(a) 与 (b) 的点估计必然逐位相同**。差别只可能出在 **CI**：
  (b1) 两组用**同一次** scene 重抽 ⇒ 与 (a) 完全等价（配对结构被保留）；
  (b2) 两组**各自独立**重抽 ⇒ 事件间的共同方差不再抵消，CI 会显著变宽。
本模块把 (a) / (b1) / (b2) 三者都算出来，逐条核验上述预期。

判定与统计纪律照旧：scene 级 bootstrap、三态判定、必需的 ctrl 对照臂。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import boot_scene                          # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
         "simlingo": "SimLingo", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}


def sig(s):
    return bool(s and (s["ci95"][0] > 0 or s["ci95"][1] < 0))


def fmt(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def _by_scene(vals, scenes):
    d = defaultdict(list)
    for v, s in zip(vals, scenes):
        if np.isfinite(v):
            d[s].append(v)
    return d


def boot_two_arms(a, b, scenes, n=5000, seed=0, shared=True):
    """两臂的组均值及其差，scene 级 bootstrap。

    shared=True  两组用**同一次** scene 重抽（保留配对结构）;
    shared=False 两组**各自独立**重抽（丢掉配对结构）。
    """
    da, db = _by_scene(a, scenes), _by_scene(b, scenes)
    keys = sorted(set(da) | set(db))
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    ma, mb, md = [], [], []
    for _ in range(n):
        ia = rng.integers(0, len(keys), len(keys))
        ib = ia if shared else rng.integers(0, len(keys), len(keys))
        va = [x for i in ia for x in da.get(keys[i], [])]
        vb = [x for i in ib for x in db.get(keys[i], [])]
        if not va or not vb:
            continue
        ma.append(np.mean(va)); mb.append(np.mean(vb)); md.append(np.mean(va) - np.mean(vb))
    q = lambda z: [float(np.percentile(z, 2.5)), float(np.percentile(z, 97.5))]   # noqa: E731
    allv = lambda d: [x for k in sorted(d) for x in d[k]]                         # noqa: E731
    return {"group_a_mean": float(np.mean(allv(da))), "group_a_ci95": q(ma),
            "group_b_mean": float(np.mean(allv(db))), "group_b_ci95": q(mb),
            "mean": float(np.mean(allv(da)) - np.mean(allv(db))), "ci95": q(md),
            "shared_resample": bool(shared), "n_scenes": len(keys)}


def L_of(traj):
    """返回 (L_arc, L_end, L_arc_nw)；坐标取前两维（x 前、y 左）。"""
    a = np.asarray(traj, float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] < 2:
        return None
    w = a[:, :2]
    if not np.all(np.isfinite(w)):
        return None
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
    return (float(np.linalg.norm(w[0]) + seg), float(np.linalg.norm(w[-1])), float(seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(LABEL))
    ap.add_argument("--min-b", type=float, default=0.5,
                    help="分母守卫：|b^(L)_ghost| 门槛（米）。与距离版同量纲，沿用 0.5 m")
    ap.add_argument("--out", default=str(RES / "f3_arclength_readout.json"))
    args = ap.parse_args()

    # 上一轮存下的逐事件 ego 位移与距离读数，用于起点不变性的实测核验
    prev = {}
    p_prev = RES / "f3_distance_readout.json"
    if p_prev.exists():
        for m in json.load(open(p_prev))["models"]:
            prev[m["model"]] = {r["eid"]: r for r in m["per_event"]}

    rows = []
    for m in args.models:
        p = RES / f"f3_occlusion_{m}_dist.json"
        if not p.exists():
            print(f"[ARC] 缺 {p.name}，跳过"); continue
        d = json.load(open(p))
        if not d.get("save_traj"):
            print(f"[ARC] {p.name} 无轨迹字段，跳过"); continue

        recs, skip = [], defaultdict(int)
        for r in d["per_event"]:
            Ls = {}
            for arm in ("clean", "ghost", "occ", "ctrl"):
                t = L_of(r.get(f"traj_{arm}"))
                if t is None:
                    break
                Ls[arm] = t
            if len(Ls) < 4:
                skip["bad_traj"] += 1; continue
            rec = {"eid": r["eid"], "scene": r["scene"],
                   # 速度版读数，用于相关性对照（同帧擦除效应）
                   "d_occ_speed": r["v_ghost"] - r["v_occ"]}
            for i, tag in enumerate(("arc", "end", "arc_nw")):
                for arm in ("clean", "ghost", "occ", "ctrl"):
                    rec[f"L_{tag}_{arm}"] = Ls[arm][i]
                rec[f"delta_occ_{tag}"] = Ls["occ"][i] - Ls["ghost"][i]
                rec[f"delta_ctrl_{tag}"] = Ls["ctrl"][i] - Ls["ghost"][i]
                rec[f"b_ghost_{tag}"] = Ls["ghost"][i] - Ls["clean"][i]
                rec[f"b_occ_{tag}"] = Ls["occ"][i] - Ls["clean"][i]
                rec[f"b_ctrl_{tag}"] = Ls["ctrl"][i] - Ls["clean"][i]
            pv = prev.get(LABEL[m], {}).get(r["eid"])
            rec["ego_disp"] = (pv or {}).get("ego_disp_clean_to_ghost")
            rec["b_ghost_d"] = (pv or {}).get("b_ghost_d")
            recs.append(rec)

        sc = [r["scene"] for r in recs]
        res = {"model": LABEL[m], "src": p.name, "n_events": len(recs), "skipped": dict(skip),
               "traj_len": int(np.asarray(d["per_event"][0]["traj_ghost"]).shape[0]),
               "definitions": {
                   "L_arc": "‖w₀‖ + Σ‖w_{i+1}−w_i‖（含 ego 原点到首点那一段）",
                   "L_end": "‖w_{−1}‖（终点到 ego 原点的直线距离）",
                   "L_arc_nw": "Σ‖w_{i+1}−w_i‖（不含原点段，稳健性用）"},
               "sign_convention": ("$\\delta^{(L)}_{occ}=L(occ)-L(ghost)$：擦掉危险 ⇒ 模型不再收敛 "
                                   "⇒ 预期**为正**（规划伸展得更远）；"
                                   "$b^{(L)}_{ghost}=L(ghost)-L(clean)$ 预期**为负**（看见危险 ⇒ 收敛）"),
               "min_b_gate_m": args.min_b, "by_definition": {}}

        for tag in ("arc", "end", "arc_nw"):
            e = {}
            e["L_ghost_mean"] = float(np.mean([r[f"L_{tag}_ghost"] for r in recs]))
            for k in ("delta_occ", "delta_ctrl", "b_ghost", "b_occ", "b_ctrl"):
                e[k] = boot_scene([r[f"{k}_{tag}"] for r in recs], sc)
            e["abs_delta_occ_minus_ctrl"] = boot_scene(
                [abs(r[f"delta_occ_{tag}"]) - abs(r[f"delta_ctrl_{tag}"]) for r in recs], sc)
            # ---- (b) 分组均值：共享重抽 / 独立重抽 ----
            occ = [r[f"L_{tag}_occ"] for r in recs]; gho = [r[f"L_{tag}_ghost"] for r in recs]
            e["group_shared"] = boot_two_arms(occ, gho, sc, shared=True)
            e["group_independent"] = boot_two_arms(occ, gho, sc, shared=False)
            # ---- (a) vs (b) 点估计一致性核验 ----
            pa = e["delta_occ"]["mean"] if e["delta_occ"] else None
            pb = e["group_shared"]["mean"] if e["group_shared"] else None
            e["paired_vs_group_pointest"] = {
                "paired_mean": pa, "group_mean_diff": pb,
                "abs_gap": (abs(pa - pb) if (pa is not None and pb is not None) else None),
                "identical_to_1e9": (bool(abs(pa - pb) < 1e-9)
                                     if (pa is not None and pb is not None) else None)}
            e["ci_halfwidth"] = {
                "paired": (e["delta_occ"]["ci95"][1] - e["delta_occ"]["ci95"][0]) / 2 if e["delta_occ"] else None,
                "group_shared": ((e["group_shared"]["ci95"][1] - e["group_shared"]["ci95"][0]) / 2
                                 if e["group_shared"] else None),
                "group_independent": ((e["group_independent"]["ci95"][1] - e["group_independent"]["ci95"][0]) / 2
                                      if e["group_independent"] else None)}
            # ---- 三态判定（门 = b^(L)_ghost 显著） ----
            use = [r for r in recs if abs(r[f"b_ghost_{tag}"]) >= args.min_b]
            e["n_used_for_R"] = len(use)
            if len(use) >= 20:
                e["necessity_ratio_L"] = boot_scene(
                    [1.0 - r[f"b_occ_{tag}"] / r[f"b_ghost_{tag}"] for r in use],
                    [r["scene"] for r in use])
                e["necessity_ratio_L_control"] = boot_scene(
                    [1.0 - r[f"b_ctrl_{tag}"] / r[f"b_ghost_{tag}"] for r in use],
                    [r["scene"] for r in use])
            # ---- R 的两种估计量：逐事件比值的均值 vs 汇总均值之比 ----
            # 既有实现（速度版沿用至今）是**逐事件先算比值再求均值**。分母 b_ghost 逐事件可以很小，
            # 比值重尾 ⇒ 均值会被少数事件主导，且 E[X/Y] ≠ E[X]/E[Y]。
            # 这里并列给出"汇总均值之比" R = −mean(δ_occ)/mean(b_ghost)（scene 级 bootstrap），
            # 两者判定不一致时**不采纳任何一个**，判"不可估：估计量依赖"。
            dd_ = _by_scene([r[f"delta_occ_{tag}"] for r in recs], sc)
            bg_ = _by_scene([r[f"b_ghost_{tag}"] for r in recs], sc)
            kk = sorted(set(dd_) & set(bg_))
            if len(kk) >= 5:
                rng = np.random.default_rng(0); vals = []
                for _ in range(5000):
                    idx = rng.integers(0, len(kk), len(kk))
                    a_ = [x for i in idx for x in dd_[kk[i]]]
                    b_ = [x for i in idx for x in bg_[kk[i]]]
                    if not a_ or not b_ or abs(np.mean(b_)) < 1e-9:
                        continue
                    vals.append(-np.mean(a_) / np.mean(b_))
                allA = [x for k in kk for x in dd_[k]]; allB = [x for k in kk for x in bg_[k]]
                e["necessity_ratio_L_from_means"] = {
                    "mean": float(-np.mean(allA) / np.mean(allB)),
                    "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
                    "n_scenes": len(kk),
                    "note": "R = −mean(δ_occ)/mean(b^(L)_ghost)，汇总均值之比（非逐事件比值的均值）"}
            bs = sig(e["b_ghost"]); e["baseline_response_significant"] = bs
            if not bs:
                e["verdict"] = ("不可估：**基线响应本身与 0 不可区分**（b^(L)_ghost 的 CI 跨 0）⇒ "
                                "没有可供必要性检验的响应。")
            elif "necessity_ratio_L" not in e:
                e["verdict"] = f"不可估：过门槛事件仅 {len(use)}"
            else:
                def _v(rr):
                    if rr is None:
                        return None
                    c = rr["ci95"]
                    return ("PASS" if c[0] > 0.5 else ("FAIL" if c[1] < 0.5 else "INDET"))
                v1 = _v(e["necessity_ratio_L"]); v2 = _v(e.get("necessity_ratio_L_from_means"))
                e["verdict_by_estimator"] = {"per_event_ratio_mean": v1, "ratio_of_means": v2}
                if v2 is not None and v1 != v2:
                    e["verdict"] = ("不可估：**两种 R 估计量给出不同判定** —— 逐事件比值均值判 "
                                    f"{v1}、汇总均值之比判 {v2}。分母 b^(L)_ghost 逐事件偏小使比值重尾，"
                                    "E[X/Y] ≠ E[X]/E[Y]；不采纳任何一个。")
                else:
                    e["verdict"] = ("PASS：遮住关键实体后轨迹伸展量退回基线"
                                    if v1 == "PASS" else
                                    ("FAIL：遮住关键实体后轨迹伸展量基本不变" if v1 == "FAIL"
                                     else "不可估：必要性比的 scene 级 CI 跨 0.5"))
            e["delta_occ_significant"] = sig(e["delta_occ"])
            e["delta_occ_specific"] = bool(sig(e["delta_occ"]) and not sig(e["delta_ctrl"])
                                           and sig(e["abs_delta_occ_minus_ctrl"])
                                           and e["abs_delta_occ_minus_ctrl"]["mean"] > 0)
            res["by_definition"][tag] = e

        # ---- 起点不变性的实测核验 ----
        eg = np.array([r["ego_disp"] if r["ego_disp"] is not None else np.nan for r in recs])
        bl = np.array([r["b_ghost_arc"] for r in recs])
        bd = np.array([r["b_ghost_d"] if r["b_ghost_d"] is not None else np.nan for r in recs])
        ok = np.isfinite(eg) & np.isfinite(bl)
        okd = np.isfinite(eg) & np.isfinite(bd)
        res["start_invariance_check"] = {
            "n": int(ok.sum()),
            "spearman_b_arc_vs_ego_disp": (float(stats.spearmanr(bl[ok], eg[ok]).statistic)
                                           if ok.sum() > 20 else None),
            "spearman_b_dist_vs_ego_disp": (float(stats.spearmanr(bd[okd], eg[okd]).statistic)
                                            if okd.sum() > 20 else None),
            "note": "弧长按构造不含 ego 世界位姿与实体位置；此处与上一轮距离读数并列做实测对照"}
        # ---- 与速度版同帧擦除效应的相关 ----
        sp = np.array([r["d_occ_speed"] for r in recs])
        do = np.array([r["delta_occ_arc"] for r in recs])
        m2 = np.isfinite(sp) & np.isfinite(do)
        res["corr_with_speed_readout"] = {
            "n": int(m2.sum()),
            "spearman_delta_occ_arc_vs_speed_d_occ": (float(stats.spearmanr(do[m2], sp[m2]).statistic)
                                                      if m2.sum() > 20 else None),
            "note": "速度版 d_occ = v(ghost) − v(occ)；弧长若只是'更长时窗的速度'，两者应强相关"}
        res["per_event"] = recs
        rows.append(res)

        a = res["by_definition"]["arc"]; b = res["by_definition"]["end"]
        print(f"\n=== {LABEL[m]}  n={len(recs)}  轨迹 {res['traj_len']} 点 ===")
        for tag, e in (("L_arc", a), ("L_end", b)):
            print(f"  [{tag}] L(ghost) 均值 {e['L_ghost_mean']:.2f} m")
            print(f"    b^(L)_ghost {fmt(e['b_ghost']):32s} {'显著' if e['baseline_response_significant'] else '跨0'}")
            print(f"    δ_occ       {fmt(e['delta_occ']):32s} {'显著' if e['delta_occ_significant'] else '跨0'}"
                  f"   δ_ctrl {fmt(e['delta_ctrl'])}")
            print(f"    |δo|−|δc|   {fmt(e['abs_delta_occ_minus_ctrl']):32s} ⇒ 特异={e['delta_occ_specific']}")
            pv = e["paired_vs_group_pointest"]; hw = e["ci_halfwidth"]
            print(f"    (a)配对 {pv['paired_mean']:+.6f} vs (b)组均值差 {pv['group_mean_diff']:+.6f}"
                  f"  |差| {pv['abs_gap']:.2e}  逐位相同={pv['identical_to_1e9']}")
            print(f"    CI 半宽：配对 {hw['paired']:.4f} / 共享重抽 {hw['group_shared']:.4f}"
                  f" / 独立重抽 {hw['group_independent']:.4f}"
                  f"（独立/配对 = {hw['group_independent']/hw['paired']:.1f}×）")
            rm = e.get("necessity_ratio_L_from_means")
            if e.get("necessity_ratio_L"):
                print(f"    R^(L) 逐事件比值均值 {fmt(e['necessity_ratio_L'])}"
                      f"  |  汇总均值之比 {fmt(rm)}")
            print(f"    判定：{e['verdict'][:76]}")
        si = res["start_invariance_check"]; cw = res["corr_with_speed_readout"]
        print(f"  起点不变性核验：ρ(b^(L)_ghost, ego 位移) = {si['spearman_b_arc_vs_ego_disp']:+.3f}"
              f"  vs 上一轮 ρ(b^(d)_ghost, ego 位移) = {si['spearman_b_dist_vs_ego_disp']:+.3f}")
        print(f"  与速度版相关：ρ(δ_occ^arc, 速度版 d_occ) = "
              f"{cw['spearman_delta_occ_arc_vs_speed_d_occ']:+.3f}")

    payload = {"design": "F-3 弧长读数：action = 轨迹自身伸展量（不引入外部参照点），纯重新分析",
               "no_new_inference": True, "n_models": len(rows),
               "n_delta_occ_specific_arc": sum(r["by_definition"]["arc"]["delta_occ_specific"] for r in rows),
               "n_delta_occ_specific_end": sum(r["by_definition"]["end"]["delta_occ_specific"] for r in rows),
               "n_baseline_significant_arc": sum(r["by_definition"]["arc"]["baseline_response_significant"] for r in rows),
               "models": rows}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[ARC] {len(rows)} 个候选；L_arc 口径下 b^(L)_ghost 显著 "
          f"**{payload['n_baseline_significant_arc']}** 个；δ_occ 显著且特异 "
          f"**{payload['n_delta_occ_specific_arc']}** 个（L_end 口径 "
          f"**{payload['n_delta_occ_specific_end']}** 个）")
    print(f"[ARC] wrote {args.out}")


if __name__ == "__main__":
    main()
