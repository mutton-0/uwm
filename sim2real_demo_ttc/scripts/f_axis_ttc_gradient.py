"""T-F|F 轴(Faithfulness):TTC/a_brake 梯度拟合的**观测法**刹车方向,与注入法 v_brake 做共线性检验。

工单依据:four_axis_proof_experiment_workorder.md §3。

假设:若 F 轴(概念→行为通路)真实存在,那么从"不同 TTC 下真实 a_brake 严重程度"这个
**连续观测量**里回归出的隐空间方向 v_brake^obs,应与已过 W1 因果校准的注入方向 v_brake 共线。
两条方法完全独立:
  v_brake     —— 标签来自**模型自身**的刹车残差(自证风险);
  v_brake^obs —— 标签来自**人类驾驶员的真实减速度**(nuScenes ego_pose 重算,外部真值)。
共线 ⇒ 两种独立方法互相印证同一条通路;正交 ⇒ 模型内部的"刹车轴"与真实刹车严重程度无关。

方法纪律:
  * 特征 = 与 v_brake **同一位置同一池化**(driving query 段均值 query_mean)的逐条件绝对激活,
    取 ghost 条件(与 a_brake 测量窗口 [t_e, t_e+3s] 时间对齐);clean 条件作敏感性;
  * 目标 = −a_brake(取负使"数值大 = 刹得狠",与 v_brake 标签 1='刹得多' 同号),
    并**先回归掉 v_at_emergence**(不扣掉的话拟合到的是"车速轴":快车刹得狠是平凡事实,
    且 prompt 里明写 Current speed,车速信息本就在激活里);
  * 回归器 = 岭回归,α 由 scene 级 GroupKFold 交叉验证在**训练折内**选,不看主读数;
  * 主读数 = cos(v_brake^obs, v_brake) @ L22(v_brake 自己的校准层),对 ≥20 个随机方向的
    余弦零分布报 p 与 CI;其余层/其余回归器/以 TTC 为目标 一律敏感性分析。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache   # noqa: E402


def fit_direction(X, y, groups, alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4), seed=0):
    """逐层岭回归 -> 单位方向。α 用 scene 级 GroupKFold 在训练集内选(不看主读数)。"""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    n, L, C = X.shape
    V = np.zeros((L, C), dtype=np.float32); chosen = []
    for l in range(L):
        Z = X[:, l, :]
        mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
        Zs = (Z - mu) / sd
        best, best_a = -np.inf, alphas[0]
        gkf = GroupKFold(n_splits=4)
        for a in alphas:
            sc = []
            for tr, te in gkf.split(Zs, y, groups):
                m = Ridge(alpha=a, random_state=seed).fit(Zs[tr], y[tr])
                p = m.predict(Zs[te])
                sc.append(np.corrcoef(p, y[te])[0, 1] if np.std(p) > 1e-9 else 0.0)
            s = float(np.nanmean(sc))
            if s > best:
                best, best_a = s, a
        w = Ridge(alpha=best_a, random_state=seed).fit(Zs, y).coef_ / sd[0]
        V[l] = w / (np.linalg.norm(w) + 1e-8)
        chosen.append({"layer": l, "alpha": best_a, "cv_r": best})
    return V, chosen


def cos_layer(a, b, l):
    return float(np.dot(a[l], b[l]) / (np.linalg.norm(a[l]) * np.linalg.norm(b[l]) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--pool", default="query_mean")
    ap.add_argument("--cond", default="ghost", choices=["ghost", "clean", "both"])
    ap.add_argument("--layer", type=int, default=22, help="v_brake 的校准层(t1q_axes: brake_layer=22)")
    ap.add_argument("--target", default="a_brake", choices=["a_brake", "ttc"])
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.out:
        args.out = f"/data/ruolin/uwm/sim2real_demo_ttc/results/f_axis_ttc_gradient{args.tag}.json"
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    ab = json.load(open("/data/ruolin/uwm/sim2real_demo_ttc/results/f_axis_abrake.json"))["events"]
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool, keep=set(evmap))

    X, y, groups, meta = [], [], [], []
    for eid, e in items.items():
        a = ab.get(eid)
        if not a or not np.isfinite(a["a_brake"]) or not np.isfinite(a["v_at_emergence"]):
            continue
        tgt = -a["a_brake"] if args.target == "a_brake" else -(e["ttc"])
        conds = ("ghost", "clean") if args.cond == "both" else (args.cond,)
        for c in conds:
            X.append(e[f"h_{c}"]); y.append(tgt); groups.append(e["scene"])
            meta.append({"event_id": eid, "cond": c, "v0": a["v_at_emergence"],
                         "etype": e["type"], "a_brake": a["a_brake"], "ttc": e["ttc"]})
    X = np.stack(X); y = np.asarray(y, float); groups = np.asarray(groups)
    v0 = np.array([m["v0"] for m in meta], float)
    # 扣掉车速主效应(见 docstring)
    A = np.vstack([v0, np.ones_like(v0)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_res = y - A @ coef
    print(f"[T-F] 样本 n={len(y)}（事件 {len(set(m['event_id'] for m in meta))}，scene {len(set(groups))}）"
          f"  目标={args.target}  条件={args.cond}  池化={args.pool}")
    print(f"[T-F] 车速主效应:β={coef[0]:+.4f}/(m/s)，R²={1-np.var(y_res)/np.var(y):.3f} -> 用残差做回归目标")

    V, chosen = fit_direction(X, y_res, groups)
    vb = np.load(work / "results" / "v_brake_query_mean.npy")
    assert vb.shape == V.shape, f"{vb.shape} vs {V.shape}"
    L = args.layer
    c_main = cos_layer(V, vb, L)

    rng = np.random.default_rng(0)
    C = V.shape[1]
    R = rng.normal(size=(args.n_random, C)); R /= np.linalg.norm(R, axis=1, keepdims=True)
    null = R @ (vb[L] / np.linalg.norm(vb[L]))
    p_two = float((np.abs(null) >= abs(c_main)).mean())
    z = float((c_main - null.mean()) / (null.std() + 1e-12))

    # scene 级 bootstrap:每次重采样 scene 后重拟合方向，得到 cos 的 CI
    sc = sorted(set(groups)); idx_by = {s: np.where(groups == s)[0] for s in sc}
    boots = []
    for b in range(args.n_boot):
        pick = np.concatenate([idx_by[sc[i]] for i in rng.integers(0, len(sc), len(sc))])
        Zl = X[pick, L, :]; yl = y_res[pick]
        mu, sd = Zl.mean(0, keepdims=True), Zl.std(0, keepdims=True) + 1e-6
        from sklearn.linear_model import Ridge
        w = Ridge(alpha=chosen[L]["alpha"]).fit((Zl - mu) / sd, yl).coef_ / sd[0]
        w = w / (np.linalg.norm(w) + 1e-8)
        boots.append(float(np.dot(w, vb[L]) / (np.linalg.norm(vb[L]) + 1e-12)))
    boots = np.array(boots)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    print(f"[T-F] **主读数** cos(v_brake^obs, v_brake) @L{L} = {c_main:+.4f}")
    print(f"      随机方向零分布(n={args.n_random}): mean={null.mean():+.4f} sd={null.std():.4f} "
          f"-> z={z:+.2f}, 双侧 p={p_two:.3g}")
    print(f"      scene 级 bootstrap 95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}] (n_boot={len(boots)})")

    prof = [cos_layer(V, vb, l) for l in range(V.shape[0])]
    print("[T-F] 逐层 cos 剖面(敏感性): " + " ".join(f"L{l}:{c:+.3f}" for l, c in enumerate(prof)))

    # ---- 仪器侧效度自检:观测法方向本身到底能不能预测真实 a_brake ----
    # 若 CV r 与 0 不可区分,则"cos 不显著"就不是标本侧结论(方向不存在),
    # 而是仪器侧结论(观测法方向根本没拟合出来),两者必须分开报。
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    Zl = X[:, L, :]; mu, sd = Zl.mean(0, keepdims=True), Zl.std(0, keepdims=True) + 1e-6
    Zs = (Zl - mu) / sd
    def cvr(yy, sd_seed=0):
        out = []
        for tr, te in GroupKFold(n_splits=4).split(Zs, yy, groups):
            m = Ridge(alpha=chosen[L]["alpha"]).fit(Zs[tr], yy[tr])
            pr = m.predict(Zs[te])
            out.append(np.corrcoef(pr, yy[te])[0, 1] if np.std(pr) > 1e-9 else 0.0)
        return float(np.nanmean(out))
    r_obs = cvr(y_res)
    perm_r = []
    uniq = np.array(sorted(set(groups)))
    for b in range(100):                       # 置换以 scene 为单位打乱目标(保持组内相关结构)
        mp = {s: t for s, t in zip(uniq, rng.permutation(uniq))}
        order = np.concatenate([idx_by[mp[g]][: len(idx_by[g])] if len(idx_by[mp[g]]) >= len(idx_by[g])
                                else np.resize(idx_by[mp[g]], len(idx_by[g])) for g in uniq])
        src = np.concatenate([idx_by[g] for g in uniq])
        yp = np.empty_like(y_res); yp[src] = y_res[order]
        perm_r.append(cvr(yp))
    perm_r = np.array(perm_r)
    p_r = float((perm_r >= r_obs).mean())
    print(f"[T-F|仪器侧] 观测法方向的 held-out 预测力 CV r = {r_obs:+.4f} "
          f"(scene 级置换零分布 mean={perm_r.mean():+.4f} sd={perm_r.std():.4f}, p={p_r:.3g})")

    # ---- 支持性读数(非主读数):v_brake 自身的投影能否预测真实人类 a_brake ----
    pj = Zl @ (vb[L] / np.linalg.norm(vb[L]))
    rho_d, p_d = stats.spearmanr(pj, y_res)
    bd = []
    for b in range(args.n_boot):
        pick = np.concatenate([idx_by[sc[i]] for i in rng.integers(0, len(sc), len(sc))])
        bd.append(stats.spearmanr(pj[pick], y_res[pick]).statistic)
    bd = np.array(bd); ci_d = [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))]
    print(f"[T-F|支持性] ρ(proj@v_brake, −a_brake 残差) = {rho_d:+.4f} (p={p_d:.3g}), "
          f"scene bootstrap 95% CI [{ci_d[0]:+.4f},{ci_d[1]:+.4f}]  —— 标注为支持性读数，不替代主读数")

    if ci[0] > 0 and p_two < 0.05:
        verdict = "PASS：观测法与注入法两条独立路径给出显著共线的刹车方向，F 轴获得双方法互证"
    elif (c_main > 0) and not (ci[0] > 0 and p_two < 0.05):
        verdict = "不可估：方向同号但未显著超出随机零分布（功效不足，不等于方向不存在）"
    elif c_main < 0 and p_two < 0.05:
        verdict = "FAIL：观测法方向与注入法方向显著反号"
    else:
        verdict = "FAIL：观测法方向落在随机零分布内，与注入法无共线证据"
    print(f"[T-F] 判定：{verdict}")

    out = {"instrument_check": {"cv_r_at_layer": r_obs, "perm_null_mean": float(perm_r.mean()),
                                "perm_null_sd": float(perm_r.std()), "p_perm": p_r,
                                "note": "CV r 显著>0 ⇒ 观测法方向确实拟合出来了，cos 不显著属标本侧结论；"
                                        "CV r 不显著 ⇒ cos 不显著属仪器侧结论"},
           "supporting_readout_proj_vs_abrake": {"spearman_rho": float(rho_d), "p": float(p_d),
                                                 "ci95_scene_bootstrap": ci_d,
                                                 "note": "支持性读数，非预注册主读数"},
           "pool": args.pool, "condition": args.cond, "target": args.target, "layer": L,
           "n_samples": int(len(y)), "n_events": len(set(m["event_id"] for m in meta)),
           "n_scenes": int(len(sc)),
           "speed_regression": {"beta": float(coef[0]), "intercept": float(coef[1]),
                                "r2_removed": float(1 - np.var(y_res) / np.var(y))},
           "ridge_alpha_by_layer": chosen,
           "cos_main": c_main, "cos_ci95_scene_bootstrap": ci, "n_boot": int(len(boots)),
           "random_null": {"n": args.n_random, "mean": float(null.mean()), "sd": float(null.std()),
                           "q95_abs": float(np.percentile(np.abs(null), 95))},
           "z_vs_null": z, "p_two_sided": p_two,
           "cos_profile_by_layer": prof, "verdict": verdict,
           "a_brake_source": "nuScenes ego_pose 重算，[t_emergence, +3s] 窗口内纵向加速度最小值（真实人类行为，强 ground truth）"}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[T-F] wrote {args.out}")


if __name__ == "__main__":
    main()
