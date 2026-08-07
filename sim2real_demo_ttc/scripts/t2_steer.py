"""T2 | Manipulation：把 v_hazard 注回模型，看行为端动不动（guide v3 §1-T2）。

这是 demo 的灵魂：前面所有工作只做到 correlation（读数），RepE 的效度真正来自
"操纵表征 → 行为按预期改变"。本脚本一次跑完 guide T2 的四件事：

  1. 剂量注入   clean 帧在 L* 注入 Z' = Z + α·σ_L·v̂,  α ∈ {0.5, 1, 2, 4}
                主读数 = Δv_plan 对 α 的斜率（**预注册**，其余全是敏感性）
  2. 随机方向   同模长随机单位方向，≥5 个 seed —— 应无效应
  3. 负向注入   α<0 —— 应反向或至少无减速
  4. termination  ghost 帧上 Z' = Z − (Zᵀv̂)v̂ —— 对危险的减速响应应减弱；
                recovery 把方向加回应复原

方向与选层沿用 T1 的纪律（S_dir 提方向 / S_sel 选层 / S_test 报数，三分不重叠），
所有注入只发生在 S_test 的场景上。

行为量 v_plan = commanded_speed(waypoints)，与 G2/G3 完全同口径（模型实际下发的目标速度）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2_cache import commanded_speed  # noqa: E402
from g3_metrics import load_cache  # noqa: E402
from n1_readout import auc, proj, supervised_direction  # noqa: E402


def lateral_offset(wp: np.ndarray) -> float:
    """1s 处的横向位置 [m]（特异性对照：注入危险方向不应把横向轨迹搞坏）。"""
    return float(wp[2, 1]) if len(wp) > 2 else float("nan")


def comfort(wp: np.ndarray) -> float:
    """舒适度代理：航点二阶差分模长均值（越大越颠）。"""
    return float(np.mean(np.linalg.norm(np.diff(wp, n=2, axis=0), axis=1))) if len(wp) > 2 else float("nan")


def run_frames(runner, root: Path, frames, anchor: float):
    """在当前 steering 设置下跑一组帧，返回帧平均的行为量。"""
    vp, lat, cf = [], [], []
    for fr in frames:
        img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
        r = runner.infer(img, fr["ego_speed_mps"], pool_modes=("vision_mean",), prompt_speed=anchor)
        vp.append(commanded_speed(r.waypoints))
        lat.append(lateral_offset(r.waypoints))
        cf.append(comfort(r.waypoints))
    return float(np.mean(vp)), float(np.mean(lat)), float(np.mean(cf))


def boot_ci(vals, scenes, n_boot=2000, seed=0):
    """scene 级 bootstrap CI（同场景内事件不独立，必须按 scene 重采样）。"""
    vals = np.asarray(vals, dtype=float)
    ok = np.isfinite(vals)
    vals, scenes = vals[ok], np.asarray(scenes)[ok]
    if len(vals) < 5:
        return float("nan"), (float("nan"), float("nan"))
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    stat = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        stat.append(np.mean([v for i in pick for v in by[keys[i]]]))
    return float(np.mean(vals)), (float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="region_mean")
    ap.add_argument("--seed", type=int, default=0, help="scene 三分的 seed，与 T1 一致")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--random-seeds", type=int, default=5)
    ap.add_argument("--random-seed-offset", type=int, default=0)
    ap.add_argument("--random-only", action="store_true",
                    help="只跑随机方向臂（扩充零分布用），跳过危险轴与 termination")
    ap.add_argument("--random-alpha", type=float, default=2.0)
    ap.add_argument("--tokens", default="vision", choices=["vision", "all", "query"])
    ap.add_argument("--vec-npy", default="",
                    help="外部轴 [L,C].npy（T1-L Step 2：注入 v_danger^lang）。"
                         "给了就不从缓存提方向，--layer 必须显式指定")
    ap.add_argument("--max-events", type=int, default=0, help=">0 时截断 S_test 事件数（控成本）")
    ap.add_argument("--layer", type=int, default=-1,
                    help="覆盖注入层（默认 -1 = 用 S_sel 选出的 L*）。"
                         "非默认值一律记为**敏感性**，不得当主读数（预注册纪律）")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.device:
        cfg["model"]["device"] = args.device
    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])
    alphas = [float(a) for a in args.alphas.split(",")]

    # ---------- 1. 方向与峰层：与 T1 同一套三分 ----------
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))
    matched = {}
    for t in ("D2a",):
        f = work / "mining" / f"matched_{t}.txt"
        if f.exists():
            matched[t] = set(f.read_text().split())

    by_type = defaultdict(list)
    for eid, e in items.items():
        by_type[evmap[eid]["event_type"]].append(e)

    scenes = sorted({e["scene"] for v in by_type.values() for e in v})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(scenes))
    n_dir, n_sel = int(len(scenes) * 0.5), int(len(scenes) * 0.25)
    part = {"dir": set(), "sel": set(), "test": set()}
    for i, k in enumerate(perm):
        part["dir" if i < n_dir else ("sel" if i < n_dir + n_sel else "test")].add(scenes[k])

    def grp(t, p):
        return [e for e in by_type[t] if e["scene"] in part[p]
                and (t not in matched or e["meta"]["event_id"] in matched[t])]

    if args.vec_npy:
        # T1-L Step 2：轴来自语言配对（离线刺激集），与本处的视觉 S_dir 无关，
        # 因此不做选层——层由 Step 1 的 held-out 一致性定，必须 --layer 显式传入。
        v = np.load(args.vec_npy).astype(np.float32)
        assert args.layer >= 0, "--vec-npy 必须配 --layer（用 Step 1 选出的 L*）"
        sel_a = np.full(v.shape[0], np.nan)
        peak = peak_sel = args.layer
        print(f"[T2] 外部轴 {args.vec_npy}  shape={v.shape}  注入层 L={peak}")
    else:
        v = supervised_direction(grp("A", "dir"), grp("D2a", "dir"), seed=args.seed)
        sel_a = np.array([auc(proj(grp("A", "sel"), v, l), proj(grp("D2a", "sel"), v, l))[0]
                          for l in range(v.shape[0])])
        peak = int(np.nanargmax(sel_a))
        peak_sel = peak
    if args.layer >= 0 and not args.vec_npy:
        peak = args.layer
        print(f"[T2] ⚠️ 注入层被手动覆盖为 L={peak}（S_sel 选出的是 L*={peak_sel}）"
              f" —— 本次结果记为敏感性，不是主读数")
    test_a = grp("A", "test")
    if args.max_events:
        test_a = test_a[: args.max_events]
    print(f"[T2] pool={args.pool_mode}  L*={peak}（S_sel AUC={sel_a[peak]:.3f}）  "
          f"S_test A 事件 n={len(test_a)}  注入 token={args.tokens}")

    # ---------- 2. 起模型 ----------
    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=True)
    vhat = v[peak]

    rand_dirs = []
    for s in range(args.random_seed_offset, args.random_seed_offset + args.random_seeds):
        rs = np.random.default_rng(10_000 + s)
        r = rs.normal(size=vhat.shape).astype(np.float32)
        rand_dirs.append(r / np.linalg.norm(r))

    # ---------- 3. 逐事件跑全部条件 ----------
    rec = []
    for i, e in enumerate(test_a):
        ev = evmap[e["meta"]["event_id"]]
        clean, ghost = ev["x_clean_frames"], ev["x_ghost_frames"]
        # prompt 锚定与 G2 一致：两个条件都写 clean 帧均速，排除语言侧污染
        anchor = float(np.mean([f["ego_speed_mps"] for f in clean]))
        row = {"event_id": ev["event_id"], "scene": ev["scene_name"]}

        runner.set_steering(None)
        row["base_clean"] = run_frames(runner, root, clean, anchor)
        if not args.random_only:
            row["base_ghost"] = run_frames(runner, root, ghost, anchor)

            for a in alphas + [-a for a in alphas]:
                runner.set_steering(peak, vhat, alpha=a, mode="add", tokens=args.tokens)
                row[f"a{a:+g}"] = run_frames(runner, root, clean, anchor)

        # 随机方向对照走**整条 ± 阶梯**，与主读数同口径比斜率。
        # 只跑单个 +α 是不够的：注入任何方向都可能造成"无符号"的扰动性减速，
        # 单点对照分不开"无符号扰动"与"有符号的方向效应"；斜率把前者约掉。
        for s, rd in enumerate(rand_dirs):
            for a in alphas + [-a for a in alphas]:
                runner.set_steering(peak, rd, alpha=a, mode="add", tokens=args.tokens)
                row[f"rand{s}_a{a:+g}"] = run_frames(runner, root, clean, anchor)
            row[f"rand{s}"] = row[f"rand{s}_a{args.random_alpha:+g}"]

        if not args.random_only:
            runner.set_steering(peak, vhat, mode="project_out", tokens=args.tokens)
            row["term_ghost"] = run_frames(runner, root, ghost, anchor)
            runner.set_steering(peak, vhat, mode="recover", tokens=args.tokens)
            row["recov_ghost"] = run_frames(runner, root, ghost, anchor)
        runner.set_steering(None)

        rec.append(row)
        if (i + 1) % 5 == 0 or i + 1 == len(test_a):
            print(f"[T2] {i+1}/{len(test_a)}")

    # ---------- 4. 统计 ----------
    sc = [r["scene"] for r in rec]
    out = {"pool_mode": args.pool_mode, "peak_layer": peak, "peak_layer_from_sel": peak_sel,
           "is_sensitivity": args.layer >= 0, "tokens": args.tokens,
           "n_events": len(rec), "alphas": alphas, "sel_auc_peak": float(sel_a[peak]),
           "n_scenes": {k: len(v_) for k, v_ in part.items()}}

    if args.random_only:
        # 只输出随机臂的全阶梯斜率，供主跑合并零分布
        A_full = np.array(sorted(alphas + [-a for a in alphas] + [0.0]))
        res = []
        for s_i, s in enumerate(range(args.random_seed_offset,
                                      args.random_seed_offset + args.random_seeds)):
            sl = []
            for r in rec:
                y = [r["base_clean"][0] if a == 0 else r[f"rand{s_i}_a{a:+g}"][0] for a in A_full]
                sl.append(float(np.polyfit(A_full, np.array(y) - r["base_clean"][0], 1)[0]))
            m_, ci_ = boot_ci(sl, sc)
            res.append({"seed": s, "full_slope": m_, "ci95": ci_})
            print(f"  rand seed {s}: 全阶梯斜率 = {m_:+.5f}  [{ci_[0]:+.5f}, {ci_[1]:+.5f}]")
        (work / "results" / f"t2_random_null{args.tag}.json").write_text(
            json.dumps({"n_events": len(rec), "alphas": alphas, "layer": peak,
                        "tokens": args.tokens, "slopes": res}, indent=2, ensure_ascii=False))
        print(f"[T2] wrote results/t2_random_null{args.tag}.json")
        return

    print("\n=== 剂量响应（主读数：Δv_plan 对 α 的斜率）===")
    print(f"{'α':>6} {'Δv_plan[m/s]':>14} {'95% CI':>20} {'Δ横向[m]':>12} {'Δ舒适度':>12}")
    print("-" * 70)
    dose = {}
    for a in alphas + [-a for a in alphas]:
        d = [r[f"a{a:+g}"][0] - r["base_clean"][0] for r in rec]
        dl = [r[f"a{a:+g}"][1] - r["base_clean"][1] for r in rec]
        dc = [r[f"a{a:+g}"][2] - r["base_clean"][2] for r in rec]
        m, ci = boot_ci(d, sc)
        ml, _ = boot_ci(dl, sc)
        mc, _ = boot_ci(dc, sc)
        dose[f"{a:+g}"] = {"dv": m, "ci95": ci, "d_lateral": ml, "d_comfort": mc}
        print(f"{a:>6.1f} {m:>14.4f} {f'[{ci[0]:+.4f},{ci[1]:+.4f}]':>20} {ml:>12.4f} {mc:>12.4f}")
    out["dose"] = dose

    # 逐事件 OLS 斜率（只用正向 α = 预注册口径），再 scene 级 bootstrap
    A_ = np.array([0.0] + alphas)
    slopes = []
    for r in rec:
        y = np.array([0.0] + [r[f"a{a:+g}"][0] - r["base_clean"][0] for a in alphas])
        slopes.append(float(np.polyfit(A_, y, 1)[0]))
    ms, cis = boot_ci(slopes, sc)
    out["dose_slope"] = {"mean": ms, "ci95": cis}
    print(f"\n[主读数] Δv_plan/Δα = {ms:+.4f} m/s per σ   95% CI [{cis[0]:+.4f}, {cis[1]:+.4f}]")

    # 全阶梯（±α）斜率：无符号的扰动效应在这里被约掉，只剩有符号的方向效应
    A_full = np.array(sorted(alphas + [-a for a in alphas] + [0.0]))
    def full_slope(r, pre=""):
        y = [r["base_clean"][0] if a == 0 else r[f"{pre}a{a:+g}"][0] for a in A_full]
        return float(np.polyfit(A_full, np.array(y) - r["base_clean"][0], 1)[0])
    fs = [full_slope(r) for r in rec]
    mf, cif = boot_ci(fs, sc)
    out["dose_slope_full"] = {"mean": mf, "ci95": cif}
    print(f"[全阶梯] Δv_plan/Δα(±α) = {mf:+.4f}   95% CI [{cif[0]:+.4f}, {cif[1]:+.4f}]")

    print("\n=== 对照 ① 随机方向（同模长，整条 ± 阶梯）===")
    rnd, rnd_slopes = [], []
    for s in range(args.random_seeds):
        d = [r[f"rand{s}"][0] - r["base_clean"][0] for r in rec]
        m, ci = boot_ci(d, sc)
        rs_ = [full_slope(r, pre=f"rand{s}_") for r in rec]
        msr, _ = boot_ci(rs_, sc)
        rnd_slopes.append(msr)
        rnd.append({"seed": s, "dv_at_alpha": m, "ci95": ci, "full_slope": msr})
        print(f"  seed {s}: Δv@α={args.random_alpha:g} = {m:+.4f}   全阶梯斜率 = {msr:+.5f}")
    out["random"] = rnd
    rs_arr = np.array(rnd_slopes)
    z = (mf - rs_arr.mean()) / (rs_arr.std(ddof=1) + 1e-12)
    n_ge = int((np.abs(rs_arr) >= abs(mf)).sum())
    spec = bool(n_ge == 0 and abs(z) > 2)
    print(f"  随机方向全阶梯斜率: 均值 {rs_arr.mean():+.5f}  sd {rs_arr.std(ddof=1):.5f}  "
          f"范围 [{rs_arr.min():+.5f}, {rs_arr.max():+.5f}]")
    print(f"  危险轴 {mf:+.5f}  ->  z={z:+.2f}，随机中 |斜率|≥危险轴的有 {n_ge}/{len(rs_arr)}  "
          f"-> {'✅ 方向特异' if spec else '❌ 与随机方向不可区分'}")
    out["random_slope_null"] = {"mean": float(rs_arr.mean()), "sd": float(rs_arr.std(ddof=1)),
                                "z": float(z), "n_ge": n_ge}
    hz = dose[f"{args.random_alpha:+g}"]["dv"]

    print("\n=== 对照 ② termination / recovery（ghost 帧）===")
    dt = [r["term_ghost"][0] - r["base_ghost"][0] for r in rec]
    dr = [r["recov_ghost"][0] - r["base_ghost"][0] for r in rec]
    mt, cit = boot_ci(dt, sc)
    mr, cir = boot_ci(dr, sc)
    out["termination"] = {"project_out": {"dv": mt, "ci95": cit}, "recover": {"dv": mr, "ci95": cir}}
    print(f"  剔除方向 Δv_plan = {mt:+.4f}  [{cit[0]:+.4f}, {cit[1]:+.4f}]   （预期 >0：减速响应减弱）")
    print(f"  加回方向 Δv_plan = {mr:+.4f}  [{cir[0]:+.4f}, {cir[1]:+.4f}]   （预期回到 0 附近）")

    # ---------- 5. E2 判定 ----------
    mono = ms < 0 and cis[1] < 0                       # 剂量单调且方向正确（注入危险 => 减速）
    term = mt > 0 and cit[0] > 0
    e2 = "PASS" if (mono and spec and term) else ("FAIL" if len(rec) >= 30 else "不可估")
    out["E2"] = {"verdict": e2, "dose_monotonic": bool(mono),
                 "random_null": bool(spec), "termination_ok": bool(term)}
    print(f"\n[E2 判定] 剂量单调={mono}  随机零效应={spec}  termination={term}  => **{e2}**")

    (work / "results" / f"t2_steer_{args.pool_mode}{args.tag}.json").write_text(
        json.dumps({**out, "per_event": rec}, indent=2, ensure_ascii=False))
    print(f"[T2] wrote results/t2_steer_{args.pool_mode}{args.tag}.json")


if __name__ == "__main__":
    main()
