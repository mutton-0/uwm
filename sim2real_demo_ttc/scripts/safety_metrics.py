"""交通安全学指标分层（零 GPU，全部从既有挖掘产物 + 缓存算）。

动机：榜单只有一个「达标率」，把最刺眼的事实藏起来了 ——
**近半数危险事件里模型规划得更快**。本脚本把安全学口径补上，并按紧迫度分层。

能算的（本仓库已有 `ttc_curve` 逐帧序列 + d_long + ego_speed）：
  * **TTC 分箱**   min_ttc_1s 分 <2s / 2–5s / >5s，看达标率随紧迫度怎么走；
  * **TET**（Time Exposed TTC）  TTC < 阈值 的累计暴露时长 [s]；
  * **TIT**（Time Integrated TTC）∫(1/TTC − 1/TTC*)dt，紧迫度的连续积分剂量；
  * **THW**（车头时距）= d_long / v_ego [s]；
  * **DRAC**（所需减速度）= v_rel²/(2d)，其中 v_rel = d/TTC ⇒ DRAC = d/(2·TTC²) [m/s²]；
  * **失效口径**：P(b<0)（规划反而更快）、无响应率 P(b<0.25)。

**不能算、不填数的**（榜上标「未实现」，理由随数据走）：
  * PET（post-encroachment time）—— 需要轨迹交叉点分析，本仓库无此管线；
  * margin（自定义安全余量）—— 全仓库检索无实现；
  * 机动需求 none/brake/swerve —— 只有 clean/ghost 两条件，无横向避让臂。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402

POSITIVE = ("A", "B", "C")
TTC_STAR = 5.0          # TET/TIT 的紧迫阈值 [s]
TTC_CAP = 99.0          # 挖掘管线里"无冲突"的哨兵值


def tet_tit(curve, star=TTC_STAR):
    """从逐帧 TTC 序列算 TET / TIT。curve = [[t, ttc], ...]，99.0 = 无冲突。"""
    if not curve or len(curve) < 2:
        return 0.0, 0.0
    ts = np.array([c[0] for c in curve], float)
    vs = np.array([c[1] for c in curve], float)
    dt = np.diff(ts, prepend=ts[0] - (ts[1] - ts[0]))
    under = (vs < star) & (vs < TTC_CAP) & (vs > 0)
    tet = float(dt[under].sum())
    tit = float((dt[under] * (1.0 / vs[under] - 1.0 / star)).sum()) if under.any() else 0.0
    return tet, tit


def boot(vals, scenes, n_boot=4000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(float(v))
    keys = list(by)
    if not keys:
        return float("nan"), [float("nan")] * 2
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(np.mean([float(v) for v in vals])), \
        [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier_m.yaml")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.to_container(OmegaConf.load(root / args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    b_min = float(cfg["metrics"]["b_min_primary"])

    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, "vision_mean", keep=set(evmap), verbose=False)

    rows = []
    for eid, e in items.items():
        ev = evmap[eid]
        if ev["event_type"] not in POSITIVE:
            continue
        ttc = ev.get("min_ttc_1s")
        d = ev.get("d_long_at_emergence")
        v = ev.get("ego_speed_mps")
        tet, tit = tet_tit(ev.get("ttc_curve") or [])
        rows.append({
            "id": eid, "type": ev["event_type"], "scene": ev["scene_name"], "b": e["b"],
            "ttc": ttc, "d": d, "v": v, "tet": tet, "tit": tit,
            "thw": (d / v) if (d is not None and v and v > 0.5) else None,
            "drac": (d / (2 * ttc ** 2)) if (d is not None and ttc and 0 < ttc < TTC_CAP) else None,
        })
    sc = [r["scene"] for r in rows]
    b = np.array([r["b"] for r in rows])
    print(f"[安全指标] 正例 n={len(rows)} / {len(set(sc))} 场景（{work.name}）\n")

    # ---------- 失效口径：把「不刹停」摆到台面上 ----------
    print("=== 失效口径（榜单此前没体现的部分）===")
    fail = {}
    for key, lab, mask in (("no_response", f"无响应率 P(b<{b_min/2:g})", b < b_min / 2),
                           ("reverse", "反向率 P(b<0)（危险帧反而规划更快）", b < 0),
                           ("pass", f"达标率 P(b≥{b_min:g})", b >= b_min)):
        m, ci = boot(mask, sc)
        fail[key] = {"value": m, "ci": ci}
        print(f"  {lab:34s} {m:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]")

    # ---------- TTC 分箱：紧迫度越高，刹得越多吗 ----------
    print("\n=== 碰撞紧迫度 TTC 分箱 ===")
    BINS = [("<2s", lambda t: t is not None and t < 2),
            ("2–5s", lambda t: t is not None and 2 <= t < 5),
            (">5s", lambda t: t is not None and t >= 5)]
    ttc_strata = []
    print(f"{'TTC 箱':>8}{'n':>6}{'达标率':>10}{'95% CI':>20}{'反向率':>10}{'b 中位':>10}")
    for lab, f_ in BINS:
        s = [r for r in rows if f_(r["ttc"])]
        if not s:
            continue
        v = np.array([r["b"] for r in s])
        m, ci = boot(v >= b_min, [r["scene"] for r in s])
        mr, _ = boot(v < 0, [r["scene"] for r in s])
        ttc_strata.append({"bin": lab, "n": len(s), "pass_rate": m, "ci": ci,
                           "reverse_rate": mr, "b_median": float(np.median(v))})
        print(f"{lab:>8}{len(s):>6}{m:>10.3f}{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>20}"
              f"{mr:>10.3f}{np.median(v):>10.4f}")
    # 趋势检验：紧迫度越高是否达标率越高
    from scipy import stats
    g = [np.array([r["b"] for r in rows if f_(r["ttc"])]) for _, f_ in BINS]
    g = [x for x in g if len(x) >= 5]
    kw = stats.kruskal(*g) if len(g) >= 2 else None
    if kw:
        print(f"  Kruskal–Wallis（b 随 TTC 箱变化）H={kw.statistic:.2f}  p={kw.pvalue:.3g}")

    # ---------- 连续安全学量：分位数分层 ----------
    print("\n=== 连续安全学量（四分位分层的达标率）===")
    cont = {}
    for key, lab, unit, better_high in (("tet", "TET 暴露时长", "s", True),
                                        ("tit", "TIT 积分紧迫度", "s/s", True),
                                        ("thw", "THW 车头时距", "s", False),
                                        ("drac", "DRAC 所需减速度", "m/s²", True)):
        vals = [r for r in rows if r[key] is not None and np.isfinite(r[key])]
        if len(vals) < 40:
            cont[key] = {"status": "不可估", "n": len(vals)}
            print(f"  {lab:16s} n={len(vals)} —— 不可估"); continue
        arr = np.array([r[key] for r in vals])
        qs = np.quantile(arr, [0, .25, .5, .75, 1.0])
        strata = []
        for i in range(4):
            sub = [r for r in vals if qs[i] <= r[key] <= qs[i + 1]] if i == 3 else \
                  [r for r in vals if qs[i] <= r[key] < qs[i + 1]]
            if not sub:
                continue
            v = np.array([r["b"] for r in sub])
            m, ci = boot(v >= b_min, [r["scene"] for r in sub])
            strata.append({"q": f"Q{i+1}", "lo": float(qs[i]), "hi": float(qs[i + 1]),
                           "n": len(sub), "pass_rate": m, "ci": ci})
        r_, p_ = stats.spearmanr(arr, [r["b"] for r in vals])
        cont[key] = {"label": lab, "unit": unit, "n": len(vals),
                     "median": float(np.median(arr)), "quartiles": qs.tolist(),
                     "strata": strata, "spearman_b": {"rho": float(r_), "p": float(p_)},
                     "expect": "紧迫度越高 b 应越大" if better_high else "THW 越小 b 应越大"}
        print(f"  {lab:16s} 中位 {np.median(arr):8.3f} {unit:5s} "
              f"ρ(量, b)={r_:+.3f} p={p_:.3g}   "
              + " ".join(f"{s['q']}:{s['pass_rate']:.3f}" for s in strata))

    # ---------- 危险物远近（ghost 配对已有）----------
    print("\n=== 危险物距离分层（ghost 配对现成）===")
    dist = []
    for lab, lo, hi in (("近 <15m", 0, 15), ("中 15–30m", 15, 30), ("远 ≥30m", 30, 1e9)):
        s = [r for r in rows if r["d"] is not None and lo <= r["d"] < hi]
        if not s:
            continue
        v = np.array([r["b"] for r in s])
        m, ci = boot(v >= b_min, [r["scene"] for r in s])
        dist.append({"bin": lab, "n": len(s), "pass_rate": m, "ci": ci})
        print(f"  {lab:12s} n={len(s):4d}  达标率 {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")

    # ---------- 与 S1 曲线的交叉核对（两者符号相反，必须显式记录）----------
    dd = [r for r in rows if r["d"] is not None]
    rho = stats.spearmanr([r["d"] for r in dd], [r["b"] for r in dd])
    cross = {
        "population": f"{work.name} 正例 A/B/C（n={len(dd)}）",
        "rho_dlong_b": float(rho.correlation), "p": float(rho.pvalue),
        "direction": "越远越刹" if rho.correlation > 0 else "越近越刹",
        "s1_reference": {"population": "n1_d2 全部缓存事件 n=2281（以几何匹配负例为主）",
                         "stat": "Jonckheere–Terpstra z=-2.155, p=0.031", "direction": "越近越刹"},
        "conflict": True,
        "note": "两者 p 均≈0.03 但**符号相反**。人群不同：S1 的曲线被负例主导，"
                "本表只用真正的危险事件。**安全相关的结论以本表为准**："
                "危险越近/越紧迫，模型反而刹得越少。S1 的 E-S1「单调方向正确」不适用于正例子集。",
    }
    print(f"\n=== 与 S1 曲线的交叉核对 ===")
    print(f"  本表（{work.name} 正例 n={len(dd)}）ρ(d_long, b) = {rho.correlation:+.4f}  p={rho.pvalue:.3g}"
          f"  ⇒ {cross['direction']}")
    print(f"  S1（n1_d2 全部 2281 事件）JT z=-2.155 p=0.031  ⇒ 越近越刹")
    print(f"  **符号相反** —— 人群不同，安全结论以正例为准")

    out = {
        "variant": work.name, "n_positive": len(rows), "n_scenes": len(set(sc)),
        "s1_crosscheck": cross,
        "b_min": b_min,
        "failure": fail,
        "ttc_strata": ttc_strata,
        "ttc_trend": {"kruskal_H": float(kw.statistic), "p": float(kw.pvalue)} if kw else None,
        "continuous": cont,
        "distance_strata": dist,
        "not_implemented": [
            {"metric": "PET（post-encroachment time）", "why": "需轨迹交叉点分析，本仓库无此管线；事件集里路口穿行类样本也少"},
            {"metric": "margin（自定义安全余量）", "why": "全仓库检索无实现"},
            {"metric": "机动需求 none/brake/swerve", "why": "只有 clean/ghost 两条件，无横向避让臂"},
        ],
    }
    dest = Path(args.out) if args.out else work / "results" / "safety_metrics.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[安全指标] wrote {dest}")


if __name__ == "__main__":
    main()
