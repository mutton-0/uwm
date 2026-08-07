"""S1 | 观察性剂量-响应曲线(guide v3 §4.5 S 线第一步,纯离线、零算力)。

产品评估的第二种候选形态:不报一个达标率,而报一条 **b–剂量曲线**。
本脚本只重分析现有缓存,不做任何前向。

剂量代理的选择(Q6 教训直接落进设计):
  SimLingo 严格单帧、实测盲于速度 => **TTC 不可测**,不能当剂量轴。
  单帧模型的剂量代理取 **目标纵向距离 d_long**(成像可见、单帧可判)。
  多帧模型日后可用 TTC —— `--dose ttc` 留着,但对单帧模型报数默认拒绝。

三个读数(对应 §4.5 S1):
  ① 单调性  —— isotonic 拟合 + 趋势检验(Jonckheere–Terpstra 为主,Page's L 见下)
  ② 阈值与斜率 —— 达标率穿过 50% 的剂量位置,及响应区斜率
  ③ 大小集一致 —— 500 集曲线是否落在 2k 集的 scene 级 bootstrap 置信带内
                  (V1"500 量级可估行为分"的机制升到曲线级)

**趋势检验的一处规范偏离(必须记录)**:§4.5 写的是 Page 趋势检验,但 Page's L 要求
完全区组(每个 scene 在每个剂量箱都有样本)。实测完全区组的 scene 数极少,直接用会
把 2k 事件缩到个位数 scene。因此**主检验改用 Jonckheere–Terpstra**(独立样本的有序
替代检验,不要求区组完整),Page's L 仍在完全区组子集上并列报告;若其 n_block < 5
则记"不可估"而非硬算。

悬崖纪律:相邻箱最大跃变显式定位并给 bootstrap CI,不允许被平滑掩盖。
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 校验过的两色分类板（light surface，六项检查全过）
C_MAIN, C_ALT = "#4269d0", "#c1571a"
C_INK, C_MUTED, C_GRID = "#22252a", "#6b7280", "#e5e7eb"


# --------------------------------------------------------------------------------------
def load_behavior(work: Path, ledger: set | None):
    """只读行为量与 meta —— 不碰 hidden，因此不受 region 池化 schema 轮次影响。"""
    out = []
    skipped = 0
    for p in sorted((work / "cache").glob("*.h5")):
        if ledger is not None and p.stem not in ledger:
            skipped += 1
            continue
        with h5py.File(p, "r") as f:
            meta = json.loads(f.attrs["meta"])
            c, g = f["clean"], f["ghost"]
            out.append({
                "event_id": meta["event_id"],
                "scene": meta["scene_name"],
                "type": meta["event_type"],
                "b_raw": float(c["pred_speed"][:].mean() - g["pred_speed"][:].mean()),
                "d_ego": float(c["ego_speed"][:].mean() - g["ego_speed"][:].mean()),
                "d_long": meta.get("d_long_at_emergence"),
                "ttc": meta.get("min_ttc_1s"),
            })
    return out, skipped


def ego_adjust(rows):
    """把 b 里跟着 Δego_speed 走的部分回归掉（与 g3_metrics.adjust_behavior_for_ego 同式）。"""
    d = np.array([r["d_ego"] for r in rows])
    b = np.array([r["b_raw"] for r in rows])
    A = np.vstack([d, np.ones_like(d)]).T
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    for r in rows:
        r["b_adj"] = float(r["b_raw"] - (coef[0] * r["d_ego"] + coef[1]))
    return {"slope": float(coef[0]), "intercept": float(coef[1])}


def curve(rows, edges, bkey, b_min):
    """每箱的 (n, 平均 b, 达标率)。空箱给 nan。"""
    n, mb, pr = [], [], []
    for i in range(len(edges) - 1):
        sel = [r for r in rows if edges[i] <= r["dose"] < edges[i + 1]]
        n.append(len(sel))
        if not sel:
            mb.append(np.nan); pr.append(np.nan); continue
        b = np.array([r[bkey] for r in sel])
        mb.append(float(b.mean()))
        pr.append(float(np.mean(b >= b_min)))
    return np.array(n), np.array(mb), np.array(pr)


def boot_band(rows, edges, bkey, b_min, n_boot=1000, seed=0):
    """scene 级 bootstrap 置信带（同场景事件不独立，必须整场景重采样）。"""
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(seed)
    MB, PR = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rs = [r for i in pick for r in by[keys[i]]]
        _, mb, pr = curve(rs, edges, bkey, b_min)
        MB.append(mb); PR.append(pr)
    f = lambda M, q: np.nanpercentile(np.array(M), q, axis=0)      # noqa: E731
    return (f(MB, 2.5), f(MB, 97.5)), (f(PR, 2.5), f(PR, 97.5))


def jonckheere(groups):
    """Jonckheere–Terpstra 有序趋势检验（正态近似）。groups 按剂量箱升序。"""
    groups = [np.asarray(g) for g in groups if len(g)]
    k = len(groups)
    if k < 3:
        return float("nan"), float("nan")
    J = 0.0
    for i in range(k - 1):
        for j in range(i + 1, k):
            a, b = groups[i], groups[j]
            J += float((a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum())
    ns = np.array([len(g) for g in groups], dtype=float)
    N = ns.sum()
    mu = (N ** 2 - (ns ** 2).sum()) / 4
    var = (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72
    z = (J - mu) / np.sqrt(var)
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def page_l(rows, edges, bkey):
    """Page's L —— 只在完全区组（每箱都有样本的 scene）上算；不足 5 个区组返回"不可估"。"""
    k = len(edges) - 1
    by = defaultdict(lambda: [[] for _ in range(k)])
    for r in rows:
        for i in range(k):
            if edges[i] <= r["dose"] < edges[i + 1]:
                by[r["scene"]][i].append(r[bkey]); break
    blocks = [[np.mean(c) for c in cols] for cols in by.values() if all(len(c) for c in cols)]
    if len(blocks) < 5:
        return {"verdict": "不可估", "n_blocks": len(blocks),
                "why": "完全区组不足 5 个 —— Page's L 不适用，见模块 docstring"}
    R = np.zeros(k)
    for blk in blocks:
        R += stats.rankdata(blk)
    n = len(blocks)
    L = float(sum((i + 1) * R[i] for i in range(k)))
    mu = n * k * (k + 1) ** 2 / 4
    var = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144
    z = (L - mu) / np.sqrt(var)
    return {"verdict": "可估", "n_blocks": n, "L": L, "z": float(z),
            "p": float(2 * (1 - stats.norm.cdf(abs(z))))}


def isotonic_fit(rows, bkey, increasing):
    from sklearn.isotonic import IsotonicRegression
    x = np.array([r["dose"] for r in rows]); y = np.array([r[bkey] for r in rows])
    yh = IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(x, y).predict(x)
    ss = float(((y - y.mean()) ** 2).sum())
    return {"r2": float(1 - ((y - yh) ** 2).sum() / ss) if ss > 0 else float("nan"),
            "increasing": increasing}


def cross_50(centers, pr):
    """达标率首次穿过 50% 的剂量位置（线性内插）；不穿过返回 None。"""
    for i in range(len(pr) - 1):
        a, b = pr[i], pr[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if (a - 0.5) * (b - 0.5) <= 0 and a != b:
            return float(centers[i] + (0.5 - a) / (b - a) * (centers[i + 1] - centers[i]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dose", default="d_long", choices=["d_long", "ttc"])
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--b-key", default="b_raw", choices=["b_raw", "b_adj"])
    ap.add_argument("--small-n", type=int, default=500, help="小集规模（V1 量级）")
    ap.add_argument("--small-reps", type=int, default=200)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--ledger-only", action="store_true",
                    help="只用当前 events_all.jsonl 里的事件（默认用全部缓存 h5，meta 自带）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    b_min = float(cfg["metrics"]["b_min_primary"])

    ledger = None
    if args.ledger_only:
        ledger = {json.loads(l)["event_id"] for l in open(work / "mining" / "events_all.jsonl")}
    rows, skipped = load_behavior(work, ledger)
    coef = ego_adjust(rows)

    for r in rows:
        r["dose"] = r["d_long"] if args.dose == "d_long" else r["ttc"]
    rows = [r for r in rows if r["dose"] is not None and np.isfinite(r["dose"])]
    if args.dose == "ttc":
        print("[S1] ⚠️ TTC 剂量轴对单帧模型无效（Q6）——本次只作多帧候选的预演，不作结论")

    doses = np.array([r["dose"] for r in rows])
    edges = np.unique(np.quantile(doses, np.linspace(0, 1, args.bins + 1)))
    edges[-1] += 1e-6                                    # 让最大值落进最后一箱
    centers = np.array([np.median(doses[(doses >= edges[i]) & (doses < edges[i + 1])])
                        for i in range(len(edges) - 1)])

    print(f"[S1] 事件 n={len(rows)}（跳过不在账本 {skipped}）  scene={len({r['scene'] for r in rows})}  "
          f"剂量轴={args.dose}  箱数={len(edges)-1}  b 口径={args.b_key}  达标阈值={b_min} m/s")
    print(f"[S1] ego 校正系数 slope={coef['slope']:+.4f}（--b-key b_adj 时生效）")

    n, mb, pr = curve(rows, edges, args.b_key, b_min)
    (mb_lo, mb_hi), (pr_lo, pr_hi) = boot_band(rows, edges, args.b_key, b_min, args.boot)

    print(f"\n{'剂量箱[m]':>16} {'n':>5} {'平均 b[m/s]':>14} {'95% 带':>20} {'达标率':>8} {'95% 带':>18}")
    print("-" * 92)
    for i in range(len(centers)):
        print(f"{f'{edges[i]:.1f}–{edges[i+1]:.1f}':>16} {n[i]:>5} {mb[i]:>14.4f} "
              f"{f'[{mb_lo[i]:+.3f},{mb_hi[i]:+.3f}]':>20} {pr[i]:>8.3f} "
              f"{f'[{pr_lo[i]:.3f},{pr_hi[i]:.3f}]':>18}")

    # ① 单调性
    groups = [[r[args.b_key] for r in rows if edges[i] <= r["dose"] < edges[i + 1]]
              for i in range(len(edges) - 1)]
    z_jt, p_jt = jonckheere(groups)
    pg = page_l(rows, edges, args.b_key)
    iso_dec = isotonic_fit(rows, args.b_key, increasing=False)
    iso_inc = isotonic_fit(rows, args.b_key, increasing=True)
    print(f"\n[① 单调性] Jonckheere–Terpstra z={z_jt:+.3f}  p={p_jt:.3g}"
          f"   (z<0 = 剂量越大/越远则响应越小，即「越近越刹」)")
    print(f"           Page's L: {pg['verdict']}"
          + (f"  n_block={pg['n_blocks']}  z={pg['z']:+.3f}  p={pg['p']:.3g}"
             if pg["verdict"] == "可估" else f"  ({pg['why']})"))
    print(f"           isotonic R²: 递减 {iso_dec['r2']:.4f} / 递增 {iso_inc['r2']:.4f}")

    # ② 阈值与斜率
    thr = cross_50(centers, pr)
    ok = np.isfinite(mb)
    slope = float(np.polyfit(centers[ok], mb[ok], 1)[0]) if ok.sum() >= 3 else float("nan")
    print(f"\n[② 阈值/斜率] 达标率穿 50% 的剂量 = "
          + (f"{thr:.2f} m" if thr is not None else "**未穿过 —— 不可估**")
          + f"   曲线斜率 db/d(dose) = {slope:+.5f} (m/s)/m")

    # ③ 悬崖
    d = np.abs(np.diff(mb))
    ci = int(np.nanargmax(d)) if np.isfinite(d).any() else -1
    cliff = None
    if ci >= 0:
        cliff = {"between": [float(centers[ci]), float(centers[ci + 1])],
                 "jump": float(mb[ci + 1] - mb[ci]),
                 "band_disjoint": bool(mb_hi[ci] < mb_lo[ci + 1] or mb_hi[ci + 1] < mb_lo[ci])}
        print(f"\n[③ 悬崖] 最大跃变在 {centers[ci]:.1f}→{centers[ci+1]:.1f} m，"
              f"Δb={cliff['jump']:+.4f}，两箱置信带{'不相交（真跃变）' if cliff['band_disjoint'] else '相交（不构成悬崖）'}")

    # ④ 500 集 vs 2k 集
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(r)
    keys = list(by)
    rng = np.random.default_rng(7)
    inside, compat, subs = [], [], []
    for _ in range(args.small_reps):
        perm = rng.permutation(len(keys))
        sub, cnt = [], 0
        for i in perm:
            sub += by[keys[i]]; cnt = len(sub)
            if cnt >= args.small_n:
                break
        _, mb_s, _ = curve(sub, edges, args.b_key, b_min)
        subs.append(mb_s)
        okk = np.isfinite(mb_s) & np.isfinite(mb_lo) & np.isfinite(mb_hi)
        inside.append(float(np.mean((mb_s[okk] >= mb_lo[okk]) & (mb_s[okk] <= mb_hi[okk]))))
    frac = float(np.mean(inside))
    # 字面判据（"落在 2k 带内"）**有机械偏差**:500 集自身的抽样误差约是 2k 带的 sqrt(2281/500)≈2.1 倍，
    # 即使两条曲线完全一致，落入比例也到不了 0.95 —— 这个数不能单独当判据。
    # 主判据改为**统计相容性**:小集与大集的差是否落在**小集自己的**误差内。
    se_s = np.nanstd(np.array(subs), axis=0)
    for mb_s in subs:
        okk = np.isfinite(mb_s) & np.isfinite(mb) & (se_s > 0)
        compat.append(float(np.mean(np.abs(mb_s[okk] - mb[okk]) <= 1.96 * se_s[okk])))
    frac_compat = float(np.mean(compat))
    print(f"\n[④ 大小集一致] {args.small_reps} 次 {args.small_n} 事件子采样")
    print(f"           字面判据(落在 2k 置信带内) = {frac:.3f}"
          f"   ← 含机械偏差，小集误差约为 2k 带的 {np.sqrt(len(rows)/args.small_n):.1f} 倍，不单独采信")
    print(f"           **主判据**(小集与大集之差在小集自身 95% 误差内) = {frac_compat:.3f}")

    # E-S1 判定
    mono = np.isfinite(p_jt) and p_jt < 0.05
    cliff_ok = bool(cliff and cliff["band_disjoint"])
    es1 = "PASS" if ((mono or cliff_ok) and frac_compat >= 0.9) else \
          ("不可估" if len(rows) < 200 else "FAIL")
    print(f"\n[E-S1 判定] 单调显著={mono}  悬崖被定位={cliff_ok}  "
          f"小集相容={frac_compat:.3f}  => **{es1}**")
    if es1 == "PASS" and max(iso_dec["r2"], iso_inc["r2"]) < 0.02:
        print("           ⚠️ 但 isotonic R² < 0.02 —— 趋势方向对，曲线本身几乎不解释方差，"
              "作为产品坐标的可用性存疑，须由 E-S3 增量价值裁决")

    out = {"dose": args.dose, "b_key": args.b_key, "b_min": b_min, "n_events": len(rows),
           "n_scenes": len({r["scene"] for r in rows}), "ego_coef": coef,
           "edges": edges.tolist(), "centers": centers.tolist(), "n_per_bin": n.tolist(),
           "mean_b": mb.tolist(), "mean_b_ci": [mb_lo.tolist(), mb_hi.tolist()],
           "pass_rate": pr.tolist(), "pass_rate_ci": [pr_lo.tolist(), pr_hi.tolist()],
           "monotonic": {"jt_z": z_jt, "jt_p": p_jt, "page": pg,
                         "isotonic_r2_decreasing": iso_dec["r2"],
                         "isotonic_r2_increasing": iso_inc["r2"]},
           "threshold_dose_m": thr, "slope": slope, "cliff": cliff,
           "small_set_consistency": {"n": args.small_n, "reps": args.small_reps,
                                     "frac_inside_2k_band": frac,
                                     "frac_compatible": frac_compat,
                                     "note": "frac_inside_2k_band 含机械偏差（小集误差远大于大集带宽），"
                                             "主判据用 frac_compatible"},
           "E_S1": es1}
    (work / "results" / f"s1_curve_{args.dose}{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    plot(out, work / "results" / f"s1_curve_{args.dose}{args.tag}.png", args)
    print(f"[S1] wrote results/s1_curve_{args.dose}{args.tag}.json + .png")


def plot(o, path, args):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    c = np.array(o["centers"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=9)
        ax.set_xlabel(f"剂量代理:目标纵向距离 d_long [m]" if o["dose"] == "d_long" else "TTC [s]",
                      color=C_MUTED, fontsize=9)

    # 左:平均 b —— 单序列，不需要图例，标题即标名
    ax = axes[0]
    lo, hi = np.array(o["mean_b_ci"][0]), np.array(o["mean_b_ci"][1])
    ax.fill_between(c, lo, hi, color=C_MAIN, alpha=0.16, lw=0)
    ax.plot(c, o["mean_b"], color=C_MAIN, lw=2, marker="o", ms=5.5,
            mec="white", mew=1.2, zorder=3)
    ax.axhline(0, color=C_MUTED, lw=1, ls=":")
    if o["cliff"] and o["cliff"]["band_disjoint"]:
        x = float(np.mean(o["cliff"]["between"]))
        ax.axvline(x, color=C_ALT, lw=1.6, ls="--")
        ax.annotate(f"悬崖 Δb={o['cliff']['jump']:+.2f}", (x, ax.get_ylim()[1]),
                    color=C_ALT, fontsize=9, ha="center", va="top")
    ax.set_ylabel("行为响应 b = v_plan(clean) − v_plan(ghost) [m/s]", color=C_MUTED, fontsize=9)
    ax.set_title("剂量-响应曲线(scene 级 bootstrap 95% 带)", color=C_INK, fontsize=11, loc="left")

    # 右:达标率
    ax = axes[1]
    lo, hi = np.array(o["pass_rate_ci"][0]), np.array(o["pass_rate_ci"][1])
    ax.fill_between(c, lo, hi, color=C_ALT, alpha=0.16, lw=0)
    ax.plot(c, o["pass_rate"], color=C_ALT, lw=2, marker="o", ms=5.5,
            mec="white", mew=1.2, zorder=3)
    ax.axhline(0.5, color=C_MUTED, lw=1, ls=":")
    if o["threshold_dose_m"] is not None:
        ax.axvline(o["threshold_dose_m"], color=C_MAIN, lw=1.6, ls="--")
        ax.annotate(f"阈值 {o['threshold_dose_m']:.1f} m", (o["threshold_dose_m"], 0.5),
                    color=C_MAIN, fontsize=9, ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points")
    ax.set_ylabel(f"达标率 P(b ≥ {o['b_min']} m/s)", color=C_MUTED, fontsize=9)
    ax.set_title("达标率曲线与响应阈值", color=C_INK, fontsize=11, loc="left")

    fig.suptitle(f"S1 观察性剂量-响应曲线 | n={o['n_events']} 事件 / {o['n_scenes']} scene "
                 f"| E-S1 = {o['E_S1']}", color=C_INK, fontsize=12, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
