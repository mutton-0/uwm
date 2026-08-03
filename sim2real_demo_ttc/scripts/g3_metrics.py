"""G3 指标计算 + G4 一致性验收（手册 §7、§8）。

7.1 危险方向 v_hazard 与探针读数 R —— S_dir(提方向)/S_sel(选层)/S_test(报数) 三分，选层与报数分离
7.2 投影-行为相关 —— scene 级 bootstrap CI
7.3 域暴露 D_L / 干涉角 —— Tier-S 按手册砍掉（config.metrics.domain_direction=false → V5 记 N/A）
7.4 行为分数 —— 估计集 vs 真值集同口径达标率
G4  V1–V5 判定

小样本纪律：Tier-S 事件量为几十级，所有读数标注"冒烟读数，不作结论"。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from scipy import stats

TTC_CAP = 30.0
POSITIVE_TYPES = ("A", "B", "C")


# --------------------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------------------
def load_cache(work: Path, pool_mode: str):
    """读取全部事件缓存 -> 每事件一条记录（帧维度已平均）。"""
    items = {}
    for p in sorted((work / "cache").glob("*.h5")):
        with h5py.File(p, "r") as f:
            meta = json.loads(f.attrs["meta"])
            c, g = f["clean"], f["ghost"]
            h_clean = c[pool_mode][:].astype(np.float32)      # [n_frames, L, C]
            h_ghost = g[pool_mode][:].astype(np.float32)
            ttc = meta.get("min_ttc_1s")
            items[meta["event_id"]] = {
                "meta": meta,
                "h_clean": h_clean.mean(0),                   # [L, C]
                "h_ghost": h_ghost.mean(0),
                "h_clean_frames": h_clean,
                "h_ghost_frames": h_ghost,
                "v_clean": float(c["pred_speed"][:].mean()),
                "v_ghost": float(g["pred_speed"][:].mean()),
                "b": float(c["pred_speed"][:].mean() - g["pred_speed"][:].mean()),
                "ttc": TTC_CAP if ttc is None else float(min(ttc, TTC_CAP)),
                "type": meta["event_type"],
                "scene": meta["scene_name"],
                "is_night": bool(meta["is_night"]),
                "is_positive": meta["event_type"] in POSITIVE_TYPES,
            }
    return items


def subset(items, ids):
    return [items[i] for i in ids if i in items]


# --------------------------------------------------------------------------------------
# 7.1 危险方向与探针
# --------------------------------------------------------------------------------------
def time_baseline_basis(events, k=2):
    """从 S_dir 的 **D 类负例** δ 里提每层"时间基线"子空间。

    clean/ghost 之间隔了 ~1s，δ 里必然混着自车位移带来的全局视角变化。
    D 类（无害出现）的 δ 里**只有**这份时间漂移，没有危险信号，
    因此它张成的子空间就是要从正例 δ 里回归掉的混淆方向（手册 §7 建议 a）。

    返回 [L, k, C]（未取到 k 个成分时行数会少），或 None（D 类样本不足）。
    """
    neg = [e for e in events if not e["is_positive"]]
    if len(neg) < 2:
        return None, len(neg)
    D = np.stack([e["h_ghost"] - e["h_clean"] for e in neg])       # [N, L, C]
    n, L, C = D.shape
    kk = min(k, n)
    U = np.zeros((L, kk, C), dtype=np.float32)
    for l in range(L):
        # 不去均值：全局漂移的**均值方向**本身就是最主要的混淆成分
        _, _, Vt = np.linalg.svd(D[:, l, :], full_matrices=False)
        U[l] = Vt[:kk]
    return U, len(neg)


def residualize(delta_l, U_l):
    """从 δ（[..., C]）里投影掉基线子空间 U_l（[k, C]，行已正交）。"""
    if U_l is None:
        return delta_l
    coef = delta_l @ U_l.T                    # [..., k]
    return delta_l - coef @ U_l


def hazard_directions(events, basis=None):
    """S_dir 上逐层 PCA 提 v_hazard(L) 与 EVR1。只用正例的 δ = h_ghost - h_clean。

    basis 非空时先把时间基线子空间从 δ 里回归掉，v_hazard 因而与基线正交。
    """
    pos = [e for e in events if e["is_positive"]]
    assert len(pos) >= 2, f"S_dir 正例不足（{len(pos)}）无法提方向"
    delta = np.stack([e["h_ghost"] - e["h_clean"] for e in pos])   # [N, L, C]
    n, L, C = delta.shape
    v = np.zeros((L, C), dtype=np.float32)
    evr = np.zeros(L, dtype=np.float32)
    for l in range(L):
        X = residualize(delta[:, l, :], None if basis is None else basis[l])
        Xc = X - X.mean(0, keepdims=True)
        # 经济 SVD：N << C
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        v[l] = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-8)
        evr[l] = float(S[0] ** 2 / max(1e-8, (S ** 2).sum()))
        # 方向定向：使多数正例的 δ 投影为正
        if np.mean(X @ v[l]) < 0:
            v[l] = -v[l]
    return v, evr, len(pos)


def hazard_directions_supervised(events, basis=None, C_reg=1.0, seed=0):
    """有监督版 v_hazard：逐层训练线性判别器区分「正例 δ」与「D 类 δ」。

    PCA 版的结构性弱点：第一主成分抓的是正例 δ 里方差最大的方向，
    而 clean/ghost 之间的自车位移恰恰就是方差最大的成分——所以 PCA 天然容易选中混淆。
    判别式方向直接以「正例 vs 无害负例」为目标，混淆成分在两类里都有、对判别无贡献，会被自动压低权重。

    只用 S_dir，S_sel/S_test/truth 一律不参与训练。
    """
    from sklearn.linear_model import LogisticRegression

    pos = [e for e in events if e["is_positive"]]
    neg = [e for e in events if not e["is_positive"]]
    assert len(pos) >= 2 and len(neg) >= 2, f"S_dir 正/负例不足（{len(pos)}/{len(neg)}）"
    X_all = np.stack([e["h_ghost"] - e["h_clean"] for e in pos + neg])     # [N, L, C]
    y = np.array([1] * len(pos) + [0] * len(neg))
    n, L, C = X_all.shape
    v = np.zeros((L, C), dtype=np.float32)
    auc_train = np.zeros(L, dtype=np.float32)
    for l in range(L):
        X = residualize(X_all[:, l, :], None if basis is None else basis[l])
        mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
        Xs = (X - mu) / sd
        clf = LogisticRegression(max_iter=2000, C=C_reg, random_state=seed).fit(Xs, y)
        w = (clf.coef_[0] / sd[0])                    # 折回原始尺度
        v[l] = w / (np.linalg.norm(w) + 1e-8)
        s = X @ v[l]
        auc_train[l] = float(stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / (len(pos) * len(neg)))
    return v, auc_train, len(pos)


def project(events, v, layer, mode="delta", basis=None):
    """把事件投影到 v_hazard(layer)；basis 非空时先回归掉时间基线。"""
    b = None if basis is None else basis[layer]
    if mode == "delta":
        X = np.stack([e["h_ghost"][layer] - e["h_clean"][layer] for e in events]) if events else np.zeros((0, v.shape[1]))
    elif mode == "ghost":
        X = np.stack([e["h_ghost"][layer] for e in events]) if events else np.zeros((0, v.shape[1]))
    elif mode == "clean":
        X = np.stack([e["h_clean"][layer] for e in events]) if events else np.zeros((0, v.shape[1]))
    else:
        raise ValueError(mode)
    if mode == "delta":
        X = residualize(X, b)
    return X @ v[layer]


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def select_peak_layer(events, v, basis=None):
    """S_sel：逐层算 ρ_TTC（投影 vs GT TTC，危险越近 TTC 越小 → 期望负相关），取 |ρ| 最大层。"""
    ttc = np.array([e["ttc"] for e in events])
    rhos = []
    for l in range(v.shape[0]):
        p = project(events, v, l, basis=basis)
        r, _ = spearman(p, ttc)
        rhos.append(r)
    rhos = np.array(rhos)
    valid = np.isfinite(rhos)
    assert valid.any(), "S_sel 上所有层 ρ_TTC 均无效"
    peak = int(np.nanargmax(np.where(valid, -rhos, -np.inf)))   # 期望负相关 → 取最负
    return peak, rhos


def auc(pos_scores, neg_scores):
    """Mann-Whitney AUC（等价于阈值无关的可分性）。"""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")
    return float(u.statistic / (len(pos_scores) * len(neg_scores))), float(u.pvalue)


def perm_test_corr(x, y, n_perm, seed=0):
    """标签置换对照：返回 (观测 ρ, 置换均值 ρ_ctrl, selectivity, p)。"""
    rng = np.random.default_rng(seed)
    obs, _ = spearman(x, y)
    if not np.isfinite(obs):
        return obs, float("nan"), float("nan"), float("nan")
    null = np.array([spearman(x, rng.permutation(y))[0] for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    p = float((np.abs(null) >= abs(obs)).mean()) if len(null) else float("nan")
    return obs, float(np.mean(np.abs(null))), float(abs(obs) - np.mean(np.abs(null))), p


# --------------------------------------------------------------------------------------
# 7.4 行为分数
# --------------------------------------------------------------------------------------
def pass_rate(events, b_min):
    if not events:
        return float("nan")
    return float(np.mean([e["b"] > b_min for e in events]))


def scene_bootstrap(events, b_min, n_boot, seed=0):
    """以 scene 为重采样单位（同 scene 多事件强相关，手册 §10.6）。"""
    by_scene = defaultdict(list)
    for e in events:
        by_scene[e["scene"]].append(e)
    scenes = list(by_scene)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(scenes), size=len(scenes), replace=True)
        evs = [e for i in pick for e in by_scene[scenes[i]]]
        out.append(pass_rate(evs, b_min))
    out = np.array([x for x in out if np.isfinite(x)])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


def ttc_bin(ttc):
    return "deep" if ttc < 5 else ("mid" if ttc < 10 else "far")


def stratified_rates(events, b_min, min_n=2):
    strata = defaultdict(list)
    for e in events:
        strata[(e["type"], "night" if e["is_night"] else "day", ttc_bin(e["ttc"]))].append(e)
    return {"|".join(k): {"n": len(v), "rate": pass_rate(v, b_min)}
            for k, v in sorted(strata.items()) if len(v) >= min_n}


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--pool-mode", default="vision_mean")
    ap.add_argument("--deconfound", default=None, choices=[None, "none", "d_baseline"],
                    help="覆盖 config.metrics.deconfound；d_baseline = 用 D 类 δ 回归掉时间基线")
    ap.add_argument("--tag", default="", help="输出文件后缀，便于消融对比")
    ap.add_argument("--direction", default=None, choices=[None, "pca", "supervised"],
                    help="v_hazard 的提取方式：pca=正例 δ 的第一主成分（手册原法）；"
                         "supervised=正例 δ vs D 类 δ 的线性判别方向")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    mcfg = cfg["metrics"]
    work = Path(cfg["paths"]["work_dir"])
    res_dir = work / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    items = load_cache(work, args.pool_mode)
    splits = json.loads((work / "mining" / "splits.json").read_text())
    est_ids = (work / "mining" / "split_estimate.txt").read_text().split()
    truth_ids = (work / "mining" / "split_truth.txt").read_text().split()
    est, truth = subset(items, est_ids), subset(items, truth_ids)

    probe_scenes = splits["probe_split_scenes"]
    S = {k: [e for e in est if e["scene"] in probe_scenes[k]] for k in ("dir", "sel", "test")}
    print(f"[G3] pool_mode={args.pool_mode}  estimate={len(est)} truth={len(truth)}  "
          f"S_dir={len(S['dir'])} S_sel={len(S['sel'])} S_test={len(S['test'])}")

    # ---------------- 7.1 ----------------
    deconf = args.deconfound or mcfg.get("deconfound", "none")
    basis, n_dir_neg = (None, 0)
    if deconf == "d_baseline":
        basis, n_dir_neg = time_baseline_basis(S["dir"], k=int(mcfg.get("deconfound_k", 2)))
        assert basis is not None, "S_dir 的 D 类负例不足，无法建时间基线"
    direction = args.direction or mcfg.get("direction_method", "pca")
    if direction == "supervised":
        v_haz, evr, n_dir_pos = hazard_directions_supervised(S["dir"], basis=basis)
    else:
        v_haz, evr, n_dir_pos = hazard_directions(S["dir"], basis=basis)
    peak, rho_by_layer = select_peak_layer(S["sel"], v_haz, basis=basis)
    print(f"[G3] deconfound={deconf}" + (f"（时间基线由 S_dir 的 {n_dir_neg} 个 D 类负例建，"
          f"k={basis.shape[1]}）" if basis is not None else ""))
    print(f"[G3] direction={direction}；v_hazard 由 S_dir 的 {n_dir_pos} 个正例"
          f"{'（+D 类负例做判别目标）' if direction == 'supervised' else ''}提出；峰层 L*={peak} "
          f"(S_sel ρ_TTC={rho_by_layer[peak]:+.3f}, "
          f"{'训练集 AUC' if direction == 'supervised' else 'EVR1'}={evr[peak]:.3f})")

    b_peak = None if basis is None else basis[peak]

    def probe_readout(events, tag):
        """在给定 held-out 集合上出 §7.1 + §7.2 的全部读数。"""
        proj = project(events, v_haz, peak, basis=basis)
        ttc = np.array([e["ttc"] for e in events])
        rho, rho_c, sel, p_perm_ = perm_test_corr(proj, ttc, mcfg["permutation_n"])

        g_sc = np.concatenate([residualize(e["h_ghost_frames"][:, peak, :], b_peak) @ v_haz[peak] for e in events])
        c_sc = np.concatenate([residualize(e["h_clean_frames"][:, peak, :], b_peak) @ v_haz[peak] for e in events])
        a_gc, p_gc_ = auc(g_sc, c_sc)

        pos = [e for e in events if e["is_positive"]]
        neg = [e for e in events if not e["is_positive"]]
        a_gd, p_gd_ = auc(project(pos, v_haz, peak, basis=basis), project(neg, v_haz, peak, basis=basis)) \
            if pos and neg else (float("nan"), float("nan"))

        b = np.array([e["b"] for e in events])
        rho_b_, p_b_ = spearman(proj, b)
        # scene 级 bootstrap（手册 §10.6）；集合内只剩 1 个 scene 时退化，降级为事件级并标记
        by_sc = defaultdict(list)
        for i, e in enumerate(events):
            by_sc[e["scene"]].append(i)
        units = list(by_sc.values()) if len(by_sc) >= 2 else [[i] for i in range(len(events))]
        degraded = len(by_sc) < 2
        rng_ = np.random.default_rng(0)
        bt = []
        for _ in range(mcfg["bootstrap_n"]):
            pick = rng_.choice(len(units), size=len(units), replace=True)
            idx = [i for k in pick for i in units[k]]
            r, _ = spearman(proj[idx], b[idx])
            if np.isfinite(r):
                bt.append(r)
        ci = (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))) if bt else (np.nan, np.nan)

        return {
            "set": tag, "n": len(events), "n_positive": len(pos), "n_negative": len(neg),
            "n_scenes": len(by_sc),
            "rho_ttc": rho, "rho_ctrl_permuted": rho_c, "selectivity": sel, "p_permutation": p_perm_,
            "auc_ghost_vs_clean": a_gc, "p_ghost_vs_clean": p_gc_,
            "auc_positive_vs_D": a_gd, "p_positive_vs_D": p_gd_,
            "rho_projection_behavior": rho_b_, "p_projection_behavior": p_b_,
            "ci95_scene_bootstrap": list(ci), "bootstrap_unit": "event(降级)" if degraded else "scene",
            "n_bootstrap": len(bt),
        }

    test = S["test"]
    ro_test = probe_readout(test, "S_test(手册主读数)")
    ro_truth = probe_readout(truth, "truth(次级：完全 held-out，未参与提方向/选层)")
    ro_pool = probe_readout(test + truth, "S_test ∪ truth(合并，提高功效)")

    rho_obs, rho_ctrl = ro_test["rho_ttc"], ro_test["rho_ctrl_permuted"]
    selectivity, p_perm = ro_test["selectivity"], ro_test["p_permutation"]
    auc_gc, p_gc = ro_test["auc_ghost_vs_clean"], ro_test["p_ghost_vs_clean"]
    auc_gd, p_gd = ro_test["auc_positive_vs_D"], ro_test["p_positive_vs_D"]
    pos_test = [e for e in test if e["is_positive"]]
    neg_test = [e for e in test if not e["is_positive"]]
    rho_b, p_b = ro_test["rho_projection_behavior"], ro_test["p_projection_behavior"]
    rho_b_ci, boot = ro_test["ci95_scene_bootstrap"], range(ro_test["n_bootstrap"])
    proj_test = project(test, v_haz, peak, basis=basis)

    # ---------------- 7.4 行为分数 ----------------
    b_primary = mcfg["b_min_primary"]
    behav = {}
    for b_min in mcfg["b_min_mps"]:
        lo, hi, _ = scene_bootstrap(est, b_min, mcfg["bootstrap_n"])
        behav[str(b_min)] = {
            "estimate_rate": pass_rate(est, b_min), "estimate_ci95": [lo, hi],
            "truth_rate": pass_rate(truth, b_min),
            "estimate_strata": stratified_rates(est, b_min),
            "truth_strata": stratified_rates(truth, b_min),
        }

    # ---------------- V4 加强版：估计集拟合 logistic，真值集测 AUC ----------------
    proj_est = project(est, v_haz, peak, basis=basis)
    y_est = np.array([e["b"] > b_primary for e in est]).astype(int)
    proj_truth = project(truth, v_haz, peak, basis=basis)
    y_truth = np.array([e["b"] > b_primary for e in truth]).astype(int)
    logit_auc, logit_note = float("nan"), ""
    if len(set(y_est)) == 2 and len(set(y_truth)) == 2:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X = ((proj_est - proj_est.mean()) / (proj_est.std() + 1e-8)).reshape(-1, 1)
        Xt = ((proj_truth - proj_est.mean()) / (proj_est.std() + 1e-8)).reshape(-1, 1)
        clf = LogisticRegression(max_iter=1000).fit(X, y_est)
        logit_auc = float(roc_auc_score(y_truth, clf.predict_proba(Xt)[:, 1]))
    else:
        logit_note = "估计集或真值集的达标标签单一，无法拟合/评估"

    # ---------------- G4 判定 ----------------
    prim = behav[str(b_primary)]
    v1 = bool(prim["estimate_ci95"][0] <= prim["truth_rate"] <= prim["estimate_ci95"][1])
    common = sorted(set(prim["estimate_strata"]) & set(prim["truth_strata"]))
    if len(common) >= 3:
        r_v2, p_v2 = spearman(np.array([prim["estimate_strata"][k]["rate"] for k in common]),
                              np.array([prim["truth_strata"][k]["rate"] for k in common]))
    else:
        r_v2, p_v2 = float("nan"), float("nan")
    v2 = bool(np.isfinite(r_v2) and r_v2 >= 0.8)
    v3 = bool(np.isfinite(p_perm) and p_perm < 0.05 and np.isfinite(p_gd) and p_gd < 0.05 and auc_gd > 0.5)
    v4 = bool(np.isfinite(p_b) and p_b < 0.05)
    v4plus = bool(np.isfinite(logit_auc) and logit_auc >= 0.65)

    out = {
        "tier": cfg["tier"], "pool_mode": args.pool_mode,
        "direction_method": direction, "deconfound": deconf, "deconfound_k": int(mcfg.get("deconfound_k", 2)) if basis is not None else 0,
        "n_dir_negatives_for_baseline": n_dir_neg,
        "clean_window_s": cfg["mining"]["clean_window_s"], "ghost_window_s": cfg["mining"]["ghost_window_s"],
        "disclaimer": ("Tier-S 冒烟读数，事件量为几十级，统计功效极低，不作结论。"
                       if str(cfg["tier"]).startswith("S") else
                       f"Tier-{cfg['tier']}：估计集 {len(est)} / 真值集 {len(truth)} 事件，"
                       "统计功效已可支撑趋势判断，但仍受限于 nuScenes 的日夜与场景构成。"),
        "n": {"estimate": len(est), "truth": len(truth),
              "S_dir": len(S["dir"]), "S_sel": len(S["sel"]), "S_test": len(S["test"]),
              "S_test_positive": len(pos_test), "S_test_negative": len(neg_test)},
        "probe": {
            "peak_layer": peak, "n_layers": int(v_haz.shape[0]), "hidden_dim": int(v_haz.shape[1]),
            "evr1_or_trainauc_by_layer": evr.tolist(), "rho_ttc_by_layer_S_sel": rho_by_layer.tolist(),
            "rho_ttc_S_test": rho_obs, "rho_ctrl_permuted": rho_ctrl,
            "selectivity": selectivity, "p_permutation": p_perm,
            "auc_ghost_vs_clean": auc_gc, "p_ghost_vs_clean": p_gc,
            "auc_positive_vs_D": auc_gd, "p_positive_vs_D": p_gd,
        },
        "readouts": {"S_test": ro_test, "truth_holdout": ro_truth, "pooled": ro_pool},
        "projection_behavior": {"rho": rho_b, "p": p_b, "ci95_scene_bootstrap": list(rho_b_ci),
                                "n_bootstrap": ro_test["n_bootstrap"]},
        "behavior_scores": behav, "b_min_primary": b_primary,
        "v4_strong": {"logistic_auc_on_truth": logit_auc, "note": logit_note},
        "domain": {"enabled": bool(mcfg["domain_direction"]),
                   "note": "Tier-S 按手册 §0.5 砍掉 CARLA 参考帧 → D_L / 干涉角 / V5 记 N/A"},
        "verdicts": {
            "V1_500est_vs_10k_truth": v1,
            "V2_stratified_trend": v2, "V2_spearman": r_v2, "V2_n_strata": len(common),
            "V3_probe_validity": v3,
            "V4_readout_predicts_behavior": v4,
            "V4_strong_logistic_auc": v4plus,
            "V5_domain_signal": None,
        },
        "verdicts_secondary_on_truth_holdout": {
            "note": "S_test 只剩 1 个 scene / 1 个正例，统计上退化；truth 集从未参与提方向与选层，"
                    "在其上重算 V3/V4 作为功效更高的次级判定（Tier-S 偏离记录）。",
            "V3_probe_validity": bool(np.isfinite(ro_truth["p_permutation"]) and ro_truth["p_permutation"] < 0.05
                                      and np.isfinite(ro_truth["p_positive_vs_D"])
                                      and ro_truth["p_positive_vs_D"] < 0.05
                                      and ro_truth["auc_positive_vs_D"] > 0.5),
            "V4_readout_predicts_behavior": bool(np.isfinite(ro_truth["p_projection_behavior"])
                                                 and ro_truth["p_projection_behavior"] < 0.05),
        },
    }
    (res_dir / f"metrics_estimate_{args.pool_mode}{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    np.save(res_dir / f"v_hazard_{args.pool_mode}{args.tag}.npy", v_haz)

    truth_out = {
        "n": len(truth), "b_min_grid": mcfg["b_min_mps"],
        "truth_rates": {str(b): behav[str(b)]["truth_rate"] for b in mcfg["b_min_mps"]},
        "truth_strata_primary": behav[str(b_primary)]["truth_strata"],
        "events": [{"event_id": e["meta"]["event_id"], "type": e["type"], "scene": e["scene"],
                    "is_night": e["is_night"], "ttc": e["ttc"], "b": e["b"],
                    "v_clean": e["v_clean"], "v_ghost": e["v_ghost"]} for e in truth],
    }
    (res_dir / "truth.json").write_text(json.dumps(truth_out, indent=2, ensure_ascii=False))

    for ro in (ro_test, ro_truth, ro_pool):
        print(f"[G3] {ro['set']}: n={ro['n']}({ro['n_positive']}+/{ro['n_negative']}-, {ro['n_scenes']} scene)  "
              f"ρ_TTC={ro['rho_ttc']:+.3f}(sel {ro['selectivity']:+.3f}, p={ro['p_permutation']:.3f})  "
              f"AUC(g/c)={ro['auc_ghost_vs_clean']:.3f}  AUC(pos/D)={ro['auc_positive_vs_D']:.3f}"
              f"(p={ro['p_positive_vs_D']:.3g})  ρ_proj-b={ro['rho_projection_behavior']:+.3f}"
              f"(p={ro['p_projection_behavior']:.3f})")
    print(f"[G3] 达标率 b>{b_primary}: 估计={prim['estimate_rate']:.3f} CI95={prim['estimate_ci95']} "
          f"真值={prim['truth_rate']:.3f}")
    print(f"[G4] V1={v1} V2={v2}(ρ={r_v2:.2f}, {len(common)} 层) V3={v3} V4={v4} V4+={v4plus} V5=N/A")
    print(f"[G3] wrote {res_dir}/metrics_estimate_{args.pool_mode}{args.tag}.json, truth.json")


if __name__ == "__main__":
    main()
