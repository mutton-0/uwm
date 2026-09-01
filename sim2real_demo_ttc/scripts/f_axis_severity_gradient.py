"""F 轴严重度梯度|A 类事件**内部**，行动层响应 b 是否随危险严重程度分级。

工单：docs/severity_gradient_and_second_scenario_workorder.md 任务一。
**零新推理**：b 全部来自 F① 已有的缓存（`f_axis_action_counterfactual.py` 的同一批 loader），
剂量坐标全部来自 `mining/events_all.jsonl` 的几何/运动学元数据，本脚本不跑任何前向。

预注册假设（工单 §任务一）：
  「反应但不特异」的候选（SimLingo、DiffusionDriveV2）预期 b **不随接近速度显著分级**；
  「反应特异」的 LTF 预期显示更强的分级关系。

--------------------------------------------------------------------------
**一处必须先声明的设计张力（自我更正记录 §SG/A41）**

工单指定**主读数用接近速度**作剂量坐标。但本轮三个候选（SimLingo / LTF / DiffusionDriveV2）
**全是单帧模型**，而"相对速度对单帧模型结构性不可观测"正是 D2cV 证伪地板赖以成立的前提
（§CE/A34）。因此"b 不随接近速度分级"对单帧模型是**按构造预期**的，
它不能单独用来支持"该候选缺乏分级机制"。

处理办法（不改工单主读数，而是补一个必需的对照）：
  * **主读数**：b 对 **v_close**（接近速度）的斜率 —— 按工单执行；
  * **必需对照（本脚本新增）**：b 对 **d_long**（纵向距离，单帧成像可见）的斜率。
    早期 S1 已登记同一条纪律："单帧模型的剂量代理取 d_long"。
  * 只有把这两条并读，才能区分**"看不见剂量"**与**"看得见但不分级"**：
      - v_close 无分级 + d_long 有分级 ⇒ 有分级机制，只是盲于速度维（仪器/标本边界清楚）；
      - 两者都无分级       ⇒ 分级机制本身缺失（这才是工单假设想说的）；
      - v_close 有分级     ⇒ 反直觉，需追查是否经由与 v_close 相关的可见量泄漏。
  * TTC 按工单只作敏感性分析（混杂距离与速度）。
--------------------------------------------------------------------------

统计口径与既有各轴一致：scene 级 bootstrap（同场景多事件不独立）；
三态判定（斜率 CI 不含 0 且方向正确 → 显示分级；CI 含 0 → 不可估；方向错误 → 如实报告）；
JT 趋势检验与 isotonic R² 为敏感性分析（复用早期 S1 方法论）。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 剂量坐标定义。v_close 由挖掘管线自己的量导出：管线里 ttc = d_long / v_c
# （`g1_mine_events.compute_scene_geometry`），故 v_close = d_long / ttc **精确**还原 v_c，
# 不是一个新的近似。
DOSES = {
    "v_close": {"label": "接近速度 v_close [m/s]", "role": "主读数（工单指定）",
                "expect": "+", "observable_single_frame": False},
    "d_long": {"label": "纵向距离 d_long [m]", "role": "必需对照（单帧可观测剂量）",
               "expect": "-", "observable_single_frame": True},
    "ttc": {"label": "TTC [s]", "role": "敏感性分析（混杂距离与速度）",
            "expect": "-", "observable_single_frame": False},
}


def dose_of(ev, name):
    d, t = ev["d_long_at_emergence"], ev["ttc_at_emergence"]
    if name == "d_long":
        return float(d)
    if name == "ttc":
        return float(t) if np.isfinite(t) else np.nan
    return float(d / t) if (np.isfinite(t) and t > 1e-6) else np.nan


def scene_boot_slope(x, y, scenes, n=5000, seed=0):
    """scene 级重采样的 OLS 斜率 CI（同场景多事件不独立）。"""
    by = defaultdict(list)
    for xi, yi, s in zip(x, y, scenes):
        by[s].append((xi, yi))
    keys = list(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        pts = [p for i in pick for p in by[keys[i]]]
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        if len(xs) < 3 or np.ptp(xs) < 1e-9:
            continue
        out.append(np.polyfit(xs, ys, 1)[0])
    out = np.array(out)
    return {"ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n_boot": int(len(out)), "n_scenes": len(keys)}


def isotonic_r2(x, y, increasing):
    from sklearn.isotonic import IsotonicRegression
    o = np.argsort(x)
    xs, ys = np.asarray(x)[o], np.asarray(y)[o]
    fit = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(xs, ys).predict(xs)
    ss_res = float(np.sum((ys - fit) ** 2)); ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def jt_trend(x, y, k=4):
    """Jonckheere–Terpstra：按剂量分 k 个等频箱，检验 b 的有序趋势（早期 S1 的主检验）。"""
    q = np.quantile(x, np.linspace(0, 1, k + 1))
    q[-1] += 1e-9
    groups = [y[(x >= q[i]) & (x < q[i + 1])] for i in range(k)]
    groups = [g for g in groups if len(g) >= 3]
    if len(groups) < 3:
        return {"verdict": "不可估：有效箱 < 3", "n_bins": len(groups)}
    U = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            U += float(np.sum(np.sign(np.subtract.outer(b, a)) > 0)) + \
                0.5 * float(np.sum(np.subtract.outer(b, a) == 0))
    ns = np.array([len(g) for g in groups], float); N = ns.sum()
    mu = (N ** 2 - np.sum(ns ** 2)) / 4.0
    sd = np.sqrt((N ** 2 * (2 * N + 3) - np.sum(ns ** 2 * (2 * ns + 3))) / 72.0)
    z = (U - mu) / sd if sd > 0 else 0.0
    return {"JT_z": float(z), "p_two_sided": float(2 * (1 - stats.norm.cdf(abs(z)))),
            "n_bins": len(groups), "bin_sizes": ns.astype(int).tolist(),
            "bin_means": [float(np.mean(g)) for g in groups]}


def analyse(name, items, evmap, dose, winsor_q):
    rows = []
    for it in items:
        ev = evmap.get(it["eid"])
        if ev is None:
            continue
        x = dose_of(ev, dose)
        if not (np.isfinite(x) and np.isfinite(it["b"])):
            continue
        rows.append({"x": x, "b": it["b"], "scene": it["scene"], "eid": it["eid"]})
    if len(rows) < 20:
        return {"verdict": f"不可估：可用事件仅 {len(rows)}"}
    x = np.array([r["x"] for r in rows]); y = np.array([r["b"] for r in rows])
    sc = [r["scene"] for r in rows]
    # 预注册的极端值处理：剂量按 winsor_q 双侧缩尾（v_close 的上尾有 2 个 >50 m/s 的
    # 轨迹估计伪影；不删样本，只压剂量坐标，避免单点主导 OLS）
    lo, hi = np.quantile(x, [1 - winsor_q, winsor_q]) if winsor_q < 1 else (x.min(), x.max())
    xw = np.clip(x, lo, hi)
    n_wins = int((x != xw).sum())
    slope, intercept = np.polyfit(xw, y, 1)
    boot = scene_boot_slope(xw, y, sc)
    rho, prho = stats.spearmanr(x, y)
    exp = DOSES[dose]["expect"]
    ci = boot["ci95"]
    sig = (ci[0] > 0) or (ci[1] < 0)
    dir_ok = (slope > 0) if exp == "+" else (slope < 0)
    if not sig:
        verdict = "不可估：斜率的 scene 级 bootstrap CI 含 0（功效不足，不得据此宣称无分级）"
    elif dir_ok:
        verdict = "显示分级：斜率 CI 不含 0 且方向与预期一致"
    else:
        verdict = "**方向与预期相反**（CI 不含 0）——如实报告，不作有利解读"
    return {"n_events": len(rows), "n_scenes": boot["n_scenes"],
            "dose": dose, "dose_label": DOSES[dose]["label"], "role": DOSES[dose]["role"],
            "expected_sign": exp, "observable_to_single_frame": DOSES[dose]["observable_single_frame"],
            "winsor_quantile": winsor_q, "n_winsorized": n_wins,
            "dose_quantiles": np.quantile(x, [0, .25, .5, .75, 1]).round(3).tolist(),
            "b_mean": float(y.mean()),
            "slope": float(slope), "slope_ci95_scene_boot": ci,
            "spearman_rho": float(rho), "spearman_p": float(prho),
            "isotonic_r2": isotonic_r2(xw, y, increasing=(exp == "+")),
            "jt_trend": jt_trend(xw, y),
            "verdict": verdict}


def multivariate(items, evmap, winsor_q, full=False):
    """b ~ v_close + d_long + ego_speed（+ 可见几何/类别）的多元回归，scene 级 bootstrap 各系数。

    **为什么这一步是必需的**：v_close 与 d_long 正相关（本语料 r = 0.25），
    而 b 随 d_long 递减。因此"b 随 v_close 递减"完全可能只是距离效应经由相关性泄漏，
    而不是模型对速度维有任何（哪怕反向的）反应 —— 对**结构性盲于速度**的单帧模型，
    后者本来就是先验上不可能的。控制住 d_long 与 ego_speed 之后 v_close 的偏系数，
    才是"模型是否对速度维有响应"的读数。
    """
    rows = []
    for it in items:
        ev = evmap.get(it["eid"])
        if ev is None:
            continue
        v, d, e = dose_of(ev, "v_close"), dose_of(ev, "d_long"), ev.get("ego_speed_mps")
        ar = ev.get("area_px"); ec = ev.get("ecc"); la = ev.get("lat_at_emergence")
        if not all(np.isfinite(z) for z in (v, d, it["b"])) or e is None:
            continue
        if full and not (ar and ec is not None and la is not None):
            continue
        rows.append({"v": v, "d": d, "e": float(e), "b": it["b"], "scene": it["scene"],
                     "la": float(np.log(float(ar))) if ar else 0.0,
                     "lat": abs(float(la)) if la is not None else 0.0,
                     "ecc": float(ec) if ec is not None else 0.0,
                     "tw": float("motorcycle" in ev.get("object_class", "")
                                 or "bicycle" in ev.get("object_class", ""))})
    if len(rows) < 30:
        return {"verdict": f"不可估：可用事件仅 {len(rows)}"}
    V = np.array([r["v"] for r in rows]); D = np.array([r["d"] for r in rows])
    E = np.array([r["e"] for r in rows]); Y = np.array([r["b"] for r in rows])
    sc = [r["scene"] for r in rows]
    lo, hi = np.quantile(V, [1 - winsor_q, winsor_q])
    V = np.clip(V, lo, hi)
    cols = [V, D, E]
    if full:
        cols += [np.array([r["la"] for r in rows]), np.array([r["lat"] for r in rows]),
                 np.array([r["ecc"] for r in rows]), np.array([r["tw"] for r in rows])]
    X = np.column_stack(cols + [np.ones(len(V))])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    by = defaultdict(list)
    for i, s_ in enumerate(sc):
        by[s_].append(i)
    keys = list(by); rng = np.random.default_rng(0); B = []
    for _ in range(5000):
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in by[keys[k]]]
        Xi, Yi = X[idx], Y[idx]
        if np.linalg.matrix_rank(Xi) < Xi.shape[1]:
            continue
        B.append(np.linalg.lstsq(Xi, Yi, rcond=None)[0])
    B = np.array(B)
    names = (["v_close", "d_long", "ego_speed"]
             + (["log_area", "abs_lat", "ecc", "two_wheeler"] if full else []) + ["intercept"])
    out = {"spec": ("b ~ v_close + d_long + ego_speed + log_area + |lat| + ecc + two_wheeler"
                    if full else "b ~ v_close + d_long + ego_speed"),
           "n_events": len(rows), "n_scenes": len(keys),
           "corr_v_close_d_long": float(np.corrcoef(V, D)[0, 1]),
           "corr_v_close_ego_speed": float(np.corrcoef(V, E)[0, 1]), "coef": {}}
    for j, nm in enumerate(names):
        ci = [float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))]
        out["coef"][nm] = {"beta": float(beta[j]), "ci95_scene_boot": ci,
                           "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    cv = out["coef"]["v_close"]
    ctrl = "距离/自车速度" + ("/可见几何与类别" if full else "")
    out["verdict"] = (f"控制{ctrl}后，v_close 的偏系数 CI 仍不含 0 ⇒ "
                      "该关系不能由所控变量解释（**不等于模型感知了速度**，见报告讨论）"
                      if cv["excludes_zero"] else
                      f"控制{ctrl}后，v_close 的偏系数 CI 含 0 ⇒ "
                      "单变量上看到的 v_close 关系不能与「所控变量经相关性泄漏」区分")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winsor-q", type=float, default=0.99,
                    help="剂量坐标双侧缩尾分位（预注册；1.0 = 不缩尾）")
    ap.add_argument("--out", default=str(RES / "f_axis_severity_gradient.json"))
    args = ap.parse_args()

    import f_axis_action_counterfactual as FA
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}

    MODELS = [("SimLingo", FA.load_simlingo, "主对比：b(A) 显著非零、反应不特异"),
              ("LTF", FA.load_ltf, "主对比：唯一「反应特异」的单帧候选"),
              ("DiffusionDriveV2", FA.load_ddv2, "主对比：b(A) 显著非零、反应不特异"),
              ("DiffusionDrive", FA.load_dd, "平线基线：b(A) 本身不显著，不纳入主对比")]

    OUT = {"design": "A 类事件**内部**的剂量-响应：b 是否随危险严重程度分级",
           "b_definition": "b = v_plan(clean) − v_plan(ghost)，正 = 目标出现后减速",
           "reuses": "b 来自 f_axis_action_counterfactual 的同一批缓存；本脚本零前向",
           "preregistered_hypothesis":
               "「反应但不特异」的候选预期 b 不随接近速度显著分级；LTF 预期分级更强",
           "design_tension_SG_A41":
               "工单指定主读数为接近速度，但本轮三个主对比候选全是单帧模型，"
               "而相对速度对单帧模型结构性不可观测（D2cV 地板赖以成立的同一前提，§CE/A34）。"
               "故必须并读 d_long（单帧可观测剂量）对照，才能区分「看不见剂量」与「看得见但不分级」。",
           "models": {}}
    for name, loader, role in MODELS:
        try:
            items, _ = loader()
        except Exception as exc:                                    # noqa: BLE001
            OUT["models"][name] = {"skipped": f"{type(exc).__name__}: {exc}"}
            print(f"[SG] 跳过 {name}：{exc}")
            continue
        A = items.get("A", [])
        M = {"role": role, "n_A_cached": len(A), "doses": {}}
        for dose in DOSES:
            M["doses"][dose] = analyse(name, A, evmap, dose, args.winsor_q)
        M["multivariate"] = multivariate(A, evmap, args.winsor_q)
        # 全控模型：把与 v_close 相关的**单帧可见**量（成像面积、离心率、横向偏移、二轮车类别）
        # 一并控住。v_close 与它们的相关分别为 −0.178 / −0.154 / −0.248 / +0.033，
        # 因此不控住就无法排除"v_close 关系其实是这些可见量在起作用"。
        M["multivariate_full"] = multivariate(A, evmap, args.winsor_q, full=True)
        OUT["models"][name] = M
        print(f"\n===== {name} =====  A={len(A)}  ({role})")
        for dose, r in M["doses"].items():
            if "slope" not in r:
                print(f"  {dose:8s} {r['verdict']}"); continue
            print(f"  {dose:8s} [{DOSES[dose]['role']}] n={r['n_events']}/{r['n_scenes']}scene  "
                  f"slope={r['slope']:+.5f} {np.round(r['slope_ci95_scene_boot'],5).tolist()}  "
                  f"ρ={r['spearman_rho']:+.3f}(p={r['spearman_p']:.3f})  "
                  f"iso R²={r['isotonic_r2']:.3f}  JT z={r['jt_trend'].get('JT_z', float('nan')):+.2f}")
            print(f"           判定：{r['verdict']}")
        for tag, mv in (("多元-基本", M["multivariate"]), ("多元-全控", M["multivariate_full"])):
            if "coef" not in mv:
                continue
            c = mv["coef"]
            print(f"  {tag}  " + "  ".join(
                f"{k}={c[k]['beta']:+.5f}{'*' if c[k]['excludes_zero'] else ' '}"
                for k in c if k != "intercept"))
            print(f"           v_close 偏系数 CI {np.round(c['v_close']['ci95_scene_boot'],5).tolist()}"
                  f" ⇒ {mv['verdict']}")
    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
    print(f"\n[SG] wrote {args.out}")


if __name__ == "__main__":
    main()
