"""Pilot post-train|用 F 轴诊断设计的**耦合辅助损失**去修 SimLingo，并检验耦合是否真被改变。

诊断（本工作线已确证，见 f_axis_ttc_gradient_report_* 与 four_axis_evidence_summary_*）：
  模型内部**存在**一条能线性读出真实危险严重程度的方向，也**存在**一条能推动刹车的方向，
  但两者近乎正交（|cos| ≤ 0.064，95% CI 上界）。即"信息在里面，但驱动动作的不是它"。

据此设计的修法（**只训 speed_wps_head**，其余全部冻结）：
  这正是选型协议 §1 核心假设的直接检验 —— "内部已编码危险的候选，post-train 只需把读出接到动作上 ⇒ 便宜"。
  头之前一切冻结 ⇒ 输入特征是常量 ⇒ 一遍前向缓存后离线训练（p1_cache_query_states.py）。

损失：
  L = MSE(v_cmd, v_human)                     任务项：跟上人类在 [t+0.5, t+1.0]s 的真实速率
    + β · MSE(wp, wp0)                        蒸馏正则：不许把规划器整体推翻（wp0 = 原模型航点）
    + λ · mean( ReLU(Δ + m) )                 **耦合项**
  其中 Δ(x) = v_cmd(f + 1·σ·v̂_hazard) − v_cmd(f)
  —— 这与 steering 实验测的 α=+1 剂量响应**是同一个量**，不是另造的代理量；
  要求它 ≤ −m（注入 1σ 危险必须让下发速度至少降 m）。

三臂（预注册）：
  A0 baseline  原始头，不训练
  A1 task-only λ=0        ← **归因对照**：行为分变好是不是仅由任务微调带来
  A2 task+couple λ>0
预注册主读数两条，缺一不可：
  ① 行为分 b-AUC(A vs D2a)（held-out scene），b = v_cmd(clean) − v_cmd(ghost)
  ② **耦合量本身** Δ@1σ 与 cos(g, v̂_hazard)（g = ∂v_cmd/∂f，autograd 精确）
  判据：只有当 ② 确实被改变、且 ① 的改善在 A2 上显著大于 A1 时，才算"诊断被验证"。
"""
from __future__ import annotations

import argparse, copy, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import stats

W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
N_SPEED_WPS = 10


def auc(a, b):
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def boot_auc(pa, sa, pb, sb, n=2000, seed=0):
    ia, ib = defaultdict(list), defaultdict(list)
    for v, s in zip(pa, sa):
        ia[s].append(v)
    for v, s in zip(pb, sb):
        ib[s].append(v)
    ks = sorted(set(ia) | set(ib)); rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        A, B = [], []
        for i in rng.integers(0, len(ks), len(ks)):
            k = ks[i]; A += ia.get(k, []); B += ib.get(k, [])
        if min(len(A), len(B)) < 5:
            continue
        o.append(auc(np.array(A), np.array(B))[0])
    o = np.array(o)
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(W / "pilot_query_states.npz"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0, help="蒸馏正则权重（三臂相同）")
    ap.add_argument("--lam", type=float, default=1.0, help="耦合项权重（A2 用；A1 恒为 0）")
    ap.add_argument("--margin", type=float, default=0.05, help="注入 1σ 危险要求的最小减速（m/s）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RES / "p2_pilot_posttrain.json"))
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)

    d = np.load(args.cache, allow_pickle=True)
    H = torch.from_numpy(d["h_last"].astype(np.float32))        # [N, n_drv, 896]
    WP0 = torch.from_numpy(d["wp0"])                            # [N, 10, 2]
    VH = torch.from_numpy(d["v_human"])
    meta = json.loads(str(d["meta"]))
    n_drv = H.shape[1]; n_route = n_drv - N_SPEED_WPS
    assert n_route >= 0, f"n_drv={n_drv} 小于 speed_wps 段长度"
    ab = json.load(open(RES / "f_axis_abrake.json"))["events"]
    print(f"[P2] 样本 {len(H)}，n_drv={n_drv}（route {n_route} + speed_wps {N_SPEED_WPS}），"
          f"人类目标速度可用 {int(torch.isfinite(VH).sum())}")

    # ---------- 冻结件：final RMSNorm + 原始 speed_wps_head ----------
    cfg = OmegaConf.to_container(OmegaConf.load(
        "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"), resolve=True)
    cfg["model"]["device"] = args.device
    sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=False)
    core = runner.model.language_model.model
    while not hasattr(core, "norm"):
        core = core.model if hasattr(core, "model") else core.base_model
    final_norm = copy.deepcopy(core.norm).float().to(dev).eval()
    for p in final_norm.parameters():
        p.requires_grad_(False)
    head0 = copy.deepcopy(runner.model.adaptors.driving.heads["speed_wps"]).float().to(dev).eval()
    del runner
    torch.cuda.empty_cache()

    def v_cmd_of(head, h, delta=None):
        """h: [B, n_drv, 896] -> commanded_speed [B]；delta 非 None 时先在所有 query 位置加扰动。"""
        if delta is not None:
            h = h + delta
        f = final_norm(h)[:, n_route:n_route + N_SPEED_WPS, :]
        wp = head(f).cumsum(1)
        return torch.linalg.vector_norm(wp[:, 0] - wp[:, 2], dim=-1) * 2.0, wp

    # ---------- scene 级二分：train / test ----------
    scenes = sorted({m["scene"] for m in meta})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(scenes))
    tr_sc = {scenes[k] for i, k in enumerate(perm) if i < len(scenes) // 2}
    idx_tr = np.array([i for i, m in enumerate(meta) if m["scene"] in tr_sc])
    idx_te = np.array([i for i, m in enumerate(meta) if m["scene"] not in tr_sc])
    print(f"[P2] scene 二分：train {len(tr_sc)} scene / {len(idx_tr)} 样本；"
          f"test {len(scenes)-len(tr_sc)} scene / {len(idx_te)} 样本")

    # ---------- 耦合方向 v_hazard@L23：**只在 train 上拟合**，用真实人类 a_brake ----------
    with torch.no_grad():
        Fn = final_norm(H.to(dev)).mean(1).cpu().numpy()        # [N, 896] query 段均值
    y = np.array([-(ab[m["event_id"]]["a_brake"]) if m["event_id"] in ab else np.nan for m in meta])
    v0h = np.array([ab[m["event_id"]]["v_at_emergence"] if m["event_id"] in ab else np.nan for m in meta])
    # 只用 **ghost 条件** 拟合（危险只出现在 ghost 帧；与 T-F 预注册的条件一致）
    is_ghost = np.array([m["cond"] == "ghost" for m in meta])
    ok = np.isfinite(y) & np.isfinite(v0h) & is_ghost
    tr_ok = np.intersect1d(idx_tr, np.where(ok)[0])
    te_ok = np.intersect1d(idx_te, np.where(ok)[0])
    A_ = np.vstack([v0h[tr_ok], np.ones(len(tr_ok))]).T          # 扣掉车速主效应（只在 train 上拟合）
    co, *_ = np.linalg.lstsq(A_, y[tr_ok], rcond=None)
    y_res = y - (v0h * co[0] + co[1])
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    Zs = Fn[tr_ok]; mu, sd = Zs.mean(0, keepdims=True), Zs.std(0, keepdims=True) + 1e-6
    Zn = (Zs - mu) / sd; ytr = y_res[tr_ok]
    grp = np.array([meta[i]["scene"] for i in tr_ok])
    best, best_a = -np.inf, 1e3                                   # α 只用 train 内 scene 级 GroupKFold 选
    for a_ in (1e1, 1e2, 1e3, 1e4, 1e5):
        sc_ = []
        for tr2, te2 in GroupKFold(n_splits=4).split(Zn, ytr, grp):
            pr2 = Ridge(alpha=a_).fit(Zn[tr2], ytr[tr2]).predict(Zn[te2])
            sc_.append(np.corrcoef(pr2, ytr[te2])[0, 1] if np.std(pr2) > 1e-12 else 0.0)
        m_ = float(np.nanmean(sc_))
        if m_ > best:
            best, best_a = m_, a_
    w = Ridge(alpha=best_a).fit(Zn, ytr).coef_ / sd[0]
    v_haz = torch.from_numpy((w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)).to(dev)
    pr_te = Fn[te_ok] @ v_haz.cpu().numpy()
    r_te = stats.spearmanr(pr_te, y_res[te_ok])
    print(f"[P2] 耦合方向 v_hazard@L23：ridge α={best_a:g}（train 内 scene 级 GroupKFold 选，CV r={best:+.4f}）")
    print(f"[P2] 该方向在 **held-out（ghost 条件，n={len(te_ok)}）** 上对真实 a_brake 残差的 "
          f"ρ = {r_te.statistic:+.4f} (p={r_te.pvalue:.3g})  ← 仪器侧前提：方向本身必须先读得出东西")

    sigma = H.std().item()
    eps = (sigma * v_haz).view(1, 1, -1)                          # 1σ 沿 v̂_hazard 注入所有 query 位置
    Hd, WPd, VHd = H.to(dev), WP0.to(dev), VH.to(dev)
    fin = torch.isfinite(VHd)

    def evaluate(head, tag):
        head.eval()
        with torch.no_grad():
            v, wp = v_cmd_of(head, Hd)
            v_inj, _ = v_cmd_of(head, Hd, delta=eps)
        delta = (v_inj - v).cpu().numpy()
        vv = v.cpu().numpy()
        # 逐事件 b = v_cmd(clean) − v_cmd(ghost)
        acc = defaultdict(dict)
        for i, m in enumerate(meta):
            acc[m["event_id"]].setdefault(m["cond"], []).append(vv[i])
            acc[m["event_id"]]["scene"] = m["scene"]; acc[m["event_id"]]["type"] = m["type"]
        rows = [{"eid": k, "scene": r["scene"], "type": r["type"],
                 "b": float(np.mean(r["clean"]) - np.mean(r["ghost"]))}
                for k, r in acc.items() if "clean" in r and "ghost" in r]
        te = [r for r in rows if r["scene"] not in tr_sc]
        pa = [r["b"] for r in te if r["type"] == "A"]; sa = [r["scene"] for r in te if r["type"] == "A"]
        pb = [r["b"] for r in te if r["type"] == "D2a"]; sb = [r["scene"] for r in te if r["type"] == "D2a"]
        a, p = auc(np.array(pa), np.array(pb))
        ci, dist = boot_auc(pa, sa, pb, sb)
        # 耦合量 ②：Δ@1σ（held-out）与 autograd 精确的 cos(g, v̂)
        hte = Hd[idx_te].clone().requires_grad_(True)
        vte, _ = v_cmd_of(head, hte)
        g, = torch.autograd.grad(vte.sum(), hte)
        gq = g.mean(1)                                            # [n_te, 896]
        cosg = F.cosine_similarity(gq, v_haz.view(1, -1), dim=-1).detach().cpu().numpy()
        mae = float(torch.abs(v[fin] - VHd[fin]).mean())
        # 部署相关的第二个行为读数：模型的减速量是否跟着**人类真实减速**走（held-out，仅 A 类）
        eb = {r["eid"]: r["b"] for r in te if r["type"] == "A"}
        xs = [eb[k] for k in eb if k in ab]
        ys = [-(ab[k]["a_brake"]) for k in eb if k in ab]
        rho_h = stats.spearmanr(xs, ys) if len(xs) > 20 else None
        out = {"arm": tag, "b_auc_A_vs_D2a_heldout": a, "b_auc_p": p, "b_auc_ci95": ci,
               "n_pos": len(pa), "n_neg": len(pb),
               "delta_at_1sigma_heldout_mean": float(delta[idx_te].mean()),
               "delta_at_1sigma_heldout_ci95": [float(np.percentile(delta[idx_te], 2.5)),
                                                float(np.percentile(delta[idx_te], 97.5))],
               "cos_g_vhazard_heldout_mean": float(cosg.mean()),
               "cos_g_vhazard_heldout_sd": float(cosg.std()),
               "v_cmd_mae_vs_human_all": mae,
               "rho_b_vs_human_abrake_heldout_A": (float(rho_h.statistic) if rho_h else None),
               "rho_b_vs_human_abrake_p": (float(rho_h.pvalue) if rho_h else None),
               "n_A_events_heldout": len(xs)}
        print(f"  [{tag}] b-AUC(held-out) = {a:.4f} {np.round(ci,4).tolist()}  |  "
              f"Δ@1σ = {out['delta_at_1sigma_heldout_mean']:+.4f} m/s  |  "
              f"cos(g, v̂_hazard) = {out['cos_g_vhazard_heldout_mean']:+.4f} ± {cosg.std():.4f}  |  "
              f"MAE(v_cmd, human) = {mae:.3f}  |  "
              f"ρ(b, 人类 a_brake) = {rho_h.statistic:+.4f} (p={rho_h.pvalue:.3g})" if rho_h else "")
        return out, dist

    def train(lam, tag):
        head = copy.deepcopy(head0).train()
        opt = torch.optim.Adam(head.parameters(), lr=args.lr)
        htr, wtr, vtr = Hd[idx_tr], WPd[idx_tr], VHd[idx_tr]
        ftr = torch.isfinite(vtr)
        for ep in range(args.epochs):
            opt.zero_grad()
            v, wp = v_cmd_of(head, htr)
            loss = F.mse_loss(v[ftr], vtr[ftr]) + args.beta * F.mse_loss(wp, wtr)
            if lam > 0:
                v_inj, _ = v_cmd_of(head, htr, delta=eps)
                loss = loss + lam * F.relu((v_inj - v) + args.margin).mean()
            loss.backward(); opt.step()
        return head

    print("\n=== 三臂评估（预注册主读数：① b-AUC 行为分  ② Δ@1σ / cos(g,v̂) 耦合量）===")
    OUT = {"config": vars(args), "n_samples": int(len(H)), "n_scenes": len(scenes),
           "n_train_scenes": len(tr_sc), "sigma": sigma,
           "coupling_direction_heldout_rho_vs_abrake": float(r_te.statistic),
           "coupling_direction_heldout_p": float(r_te.pvalue),
           "coupling_direction_ridge_alpha": float(best_a),
           "coupling_direction_train_cv_r": float(best),
           "coupling_direction_fit_condition": "ghost only（危险只出现在 ghost 帧，与 T-F 预注册条件一致）",
           "coupling_direction_instrument_ok": bool(r_te.pvalue < 0.05 and r_te.statistic > 0),
           "arms": {}}
    dists = {}
    for tag, lam in (("A0_baseline", None), ("A1_task_only", 0.0), ("A2_task_plus_coupling", args.lam)):
        head = head0 if lam is None else train(lam, tag)
        OUT["arms"][tag], dists[tag] = evaluate(head, tag)
        if lam is not None:
            torch.save(head.state_dict(), RES / f"p2_head_{tag}.pt")

    a0, a1, a2 = OUT["arms"]["A0_baseline"], OUT["arms"]["A1_task_only"], OUT["arms"]["A2_task_plus_coupling"]
    d20 = dists["A2_task_plus_coupling"] - dists["A0_baseline"]
    d21 = dists["A2_task_plus_coupling"] - dists["A1_task_only"]
    d10 = dists["A1_task_only"] - dists["A0_baseline"]
    OUT["contrasts"] = {
        "b_auc_A2_minus_A0": {"mean": float(d20.mean()),
                              "ci95": [float(np.percentile(d20, 2.5)), float(np.percentile(d20, 97.5))]},
        "b_auc_A2_minus_A1": {"mean": float(d21.mean()),
                              "ci95": [float(np.percentile(d21, 2.5)), float(np.percentile(d21, 97.5))]},
        "b_auc_A1_minus_A0": {"mean": float(d10.mean()),
                              "ci95": [float(np.percentile(d10, 2.5)), float(np.percentile(d10, 97.5))]},
        "coupling_delta_A2_minus_A0": a2["delta_at_1sigma_heldout_mean"] - a0["delta_at_1sigma_heldout_mean"],
        "coupling_cos_A2_minus_A0": a2["cos_g_vhazard_heldout_mean"] - a0["cos_g_vhazard_heldout_mean"],
        "coupling_cos_A1_minus_A0": a1["cos_g_vhazard_heldout_mean"] - a0["cos_g_vhazard_heldout_mean"]}

    coupling_changed = abs(OUT["contrasts"]["coupling_cos_A2_minus_A0"]) > 0.05 or \
        abs(OUT["contrasts"]["coupling_delta_A2_minus_A0"]) > 0.02
    beh_ci = OUT["contrasts"]["b_auc_A2_minus_A1"]["ci95"]
    beh_better_than_control = beh_ci[0] > 0
    if coupling_changed and beh_better_than_control:
        v = "PASS：耦合量确实被改变，且行为分的改善显著大于纯任务微调对照 ⇒ 诊断被验证"
    elif coupling_changed and not beh_better_than_control:
        v = ("部分成立：**耦合量确实被改变**，但行为分相对纯任务微调对照的增量不显著 ⇒ "
             "修好了通路却没换来可测的行为收益（不得据行为分反推诊断成立或不成立）")
    elif (not coupling_changed) and beh_better_than_control:
        v = ("FAIL（归因失败）：行为分变好但耦合量没被改变 ⇒ 改善不可归因于耦合损失，"
             "**正是本轮预先要防的那种误判**")
    else:
        v = "FAIL：耦合量未被改变，行为分也无显著增量"
    OUT["verdict"] = v
    OUT["verdict_rule"] = ("① 耦合量改变判据：|Δcos| > 0.05 或 |ΔΔ@1σ| > 0.02 m/s；"
                           "② 行为分归因判据：A2 − A1 的 scene 级 bootstrap 95% CI 下界 > 0")
    print(f"\n[P2] b-AUC 对比：A2−A0 {d20.mean():+.4f} {np.round(OUT['contrasts']['b_auc_A2_minus_A0']['ci95'],4).tolist()}  |  "
          f"A2−A1 {d21.mean():+.4f} {np.round(beh_ci,4).tolist()}  |  A1−A0 {d10.mean():+.4f}")
    print(f"[P2] 耦合量对比：Δcos(A2−A0) = {OUT['contrasts']['coupling_cos_A2_minus_A0']:+.4f}；"
          f"ΔΔ@1σ(A2−A0) = {OUT['contrasts']['coupling_delta_A2_minus_A0']:+.4f} m/s；"
          f"Δcos(A1−A0) = {OUT['contrasts']['coupling_cos_A1_minus_A0']:+.4f}")
    print(f"[P2] 判定：{v}")
    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
    print(f"[P2] wrote {args.out}")


if __name__ == "__main__":
    main()
