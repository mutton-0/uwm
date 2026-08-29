"""F 轴补测|DiffusionDrive 的 steering 四件套（补齐轴矩阵缺口）。

依据：选型协议 §3④（F = 通路利用率）+ 方向发现计划 Stage D 四件套。
架构声明：DiffusionDrive 的动作头是**扩散头**，按计划 Stage D「对无法做解析投影的架构
（扩散/flow-matching 动作头）**只用经验 steering，不尝试解析捷径**——这是架构决定的，
不是退而求其次」。故本脚本只跑经验臂，不跑 Jacobian 解析臂。

设定与 SimLingo 侧逐条同构：
  注入 Z' = Z + α·σ_L·v̂ 于第 L* 个 encoder SelfAttention 的**图像 token 段**（方向从同一批 token 提出）；
  α 阶梯 ±{0.5,1,2,4}；主读数 = 整条 ±α 阶梯的最小二乘斜率（修正案 A3）；
  对照 ① 同层 ≥20 seed 随机方向零分布（整条阶梯比斜率）；
       ② 特异性：注入后横向偏移/舒适度不应随之劣化；
       ③ termination（project_out）/ recovery。
  刺激集 = 与 SimLingo 侧同一批 S_test A 类事件（同 seed、同三分）。
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
ALPHAS = [0.5, 1, 2, 4]


def boot_ci(vals, scenes, n=2000, seed=0):
    vals = np.asarray(vals, float); ok = np.isfinite(vals)
    vals, scenes = vals[ok], np.asarray(scenes)[ok]
    if len(vals) < 5:
        return float("nan"), (float("nan"), float("nan"))
    d = defaultdict(list)
    for v, s in zip(vals, scenes):
        d[s].append(v)
    ks = sorted(d); rng = np.random.default_rng(seed)
    o = np.array([np.mean([x for i in rng.integers(0, len(ks), len(ks)) for x in d[ks[i]]]) for _ in range(n)])
    return float(vals.mean()), (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=-1, help="默认取 T-G 冻结的概念峰层")
    ap.add_argument("--tokens", default="image")
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="scene 三分 seed，与 SimLingo 侧一致")
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vec-npz", default=str(RES / "v_hazard_dd_vision_mean.npz"),
                    help="外部方向 npz（键 L0..L7 + peak_layer）")
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    global ALPHAS
    ALPHAS = [float(a) for a in args.alphas.split(",")]
    if not args.out:
        args.out = str(RES / f"f_axis_dd_steer{args.tag}.json")
    z = np.load(args.vec_npz)
    L = args.layer if args.layer >= 0 else int(z["peak_layer"][0])
    vhat = z[f"L{L}"].astype(np.float32)
    print(f"[F-dd] 注入层 L*={L}（T-G 冻结），方向维度 {vhat.shape}，token 集={args.tokens}")

    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}
    # 与 SimLingo 侧完全相同的 scene 三分（同 seed、同全类型场景表）
    scenes_all = sorted({e["scene_name"] for e in evmap.values()})
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(scenes_all))
    n_dir, n_sel = int(len(scenes_all) * .5), int(len(scenes_all) * .25)
    test = {scenes_all[k] for i, k in enumerate(perm) if i >= n_dir + n_sel}
    evs = [e for e in evmap.values() if e["event_type"] == "A" and e["scene_name"] in test]
    evs = sorted(evs, key=lambda e: e["event_id"])[: args.max_events]
    print(f"[F-dd] S_test A 类事件 n={len(evs)}（{len(set(e['scene_name'] for e in evs))} scene）")

    sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
    from dd_adapter import DDRunner
    runner = DDRunner(device=args.device)

    rs = [np.random.default_rng(10_000 + s).normal(size=vhat.shape).astype(np.float32) for s in range(args.random_seeds)]
    rs = [r / np.linalg.norm(r) for r in rs]

    def read(img, spd):
        r = runner.run(img, spd)
        return (r["commanded_speed"], DDRunner.lateral_offset(r["trajectory"]),
                DDRunner.comfort(r["trajectory"]))

    rec, t0 = [], time.time()
    for i, ev in enumerate(evs):
        frames = ev["x_clean_frames"]
        anchor = float(np.mean([f["ego_speed_mps"] for f in frames]))
        imgs = [cv2.cvtColor(cv2.imread(str(Path(args.nuscenes_root) / f["filename"])), cv2.COLOR_BGR2RGB)
                for f in frames]
        gh = [cv2.cvtColor(cv2.imread(str(Path(args.nuscenes_root) / f["filename"])), cv2.COLOR_BGR2RGB)
              for f in ev["x_ghost_frames"]]
        row = {"event_id": ev["event_id"], "scene": ev["scene_name"]}

        def avg(images, spd):
            o = np.array([read(im, spd) for im in images], float)
            return o.mean(0).tolist()

        runner.set_steering(None)
        row["base_clean"] = avg(imgs, anchor)
        row["base_ghost"] = avg(gh, anchor)
        for a in ALPHAS + [-a for a in ALPHAS]:
            runner.set_steering(L, vhat, alpha=a, mode="add", tokens=args.tokens)
            row[f"a{a:+g}"] = avg(imgs, anchor)
        for si, r in enumerate(rs):
            for a in ALPHAS + [-a for a in ALPHAS]:
                runner.set_steering(L, r, alpha=a, mode="add", tokens=args.tokens)
                row[f"rand{si}_a{a:+g}"] = avg(imgs, anchor)
        runner.set_steering(L, vhat, mode="project_out", tokens=args.tokens)
        row["term_ghost"] = avg(gh, anchor)
        runner.set_steering(L, vhat, mode="recover", tokens=args.tokens)
        row["recov_ghost"] = avg(gh, anchor)
        runner.set_steering(None)
        rec.append(row)
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"[F-dd] {i+1}/{len(evs)}  {el:.0f}s ({el/(i+1):.1f}s/event)")

    sc = [r["scene"] for r in rec]
    A_full = np.array(sorted(ALPHAS + [-a for a in ALPHAS] + [0.0]))

    def full_slope(r, pre=""):
        y = [r["base_clean"][0] if a == 0 else r[f"{pre}a{a:+g}"][0] for a in A_full]
        return float(np.polyfit(A_full, np.array(y) - r["base_clean"][0], 1)[0])

    out = {"model": "DiffusionDrive", "direction_file": Path(args.vec_npz).name,
           "layer": L, "tokens": args.tokens, "n_events": len(rec),
           "n_scenes": len(set(sc)), "alphas": ALPHAS,
           "architecture_note": "扩散动作头 ⇒ 按计划 Stage D 只跑经验 steering，不做解析投影",
           "dose": {}}
    print("\n=== 剂量响应 ===")
    for a in ALPHAS + [-a for a in ALPHAS]:
        dv, ci = boot_ci([r[f"a{a:+g}"][0] - r["base_clean"][0] for r in rec], sc)
        dl, _ = boot_ci([r[f"a{a:+g}"][1] - r["base_clean"][1] for r in rec], sc)
        dc, _ = boot_ci([r[f"a{a:+g}"][2] - r["base_clean"][2] for r in rec], sc)
        out["dose"][f"{a:+g}"] = {"dv": dv, "ci95": list(ci), "d_lateral": dl, "d_comfort": dc}
        print(f"  α={a:+.1f}  Δv={dv:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  Δ横向={dl:+.4f}  Δ舒适={dc:+.4f}")

    mf, cif = boot_ci([full_slope(r) for r in rec], sc)
    out["dose_slope_full"] = {"mean": mf, "ci95": list(cif)}
    print(f"[主读数] ±α 全阶梯斜率 = {mf:+.5f}  95% CI [{cif[0]:+.5f}, {cif[1]:+.5f}]")

    nulls = []
    for si in range(len(rs)):
        m_, _ = boot_ci([full_slope(r, pre=f"rand{si}_") for r in rec], sc)
        nulls.append(m_)
    nl = np.array(nulls)
    zsc = (mf - nl.mean()) / (nl.std(ddof=1) + 1e-12)
    n_ge = int((np.abs(nl) >= abs(mf)).sum())
    out["random_null"] = {"n_seeds": len(nl), "mean": float(nl.mean()), "sd": float(nl.std(ddof=1)),
                          "z": float(zsc), "n_ge": n_ge, "p_empirical": float((n_ge + 1) / (len(nl) + 1)),
                          "exceeds": bool(n_ge == 0 and abs(zsc) > 2), "slopes": nl.tolist()}
    print(f"[对照①] 同层随机零分布 {nl.mean():+.5f} ± {nl.std(ddof=1):.5f} (n={len(nl)})  "
          f"z={zsc:+.2f}  经验 p={(n_ge+1)/(len(nl)+1):.3f}  "
          f"{'✅超出' if out['random_null']['exceeds'] else '❌未超出'}")

    d2 = out["dose"][f"{max(ALPHAS):+g}"]
    out["specificity"] = {"dv": d2["dv"], "d_lateral": d2["d_lateral"], "d_comfort": d2["d_comfort"],
                          "lateral_over_dv": abs(d2["d_lateral"]) / (abs(d2["dv"]) + 1e-12),
                          "comfort_over_dv": abs(d2["d_comfort"]) / (abs(d2["dv"]) + 1e-12)}
    print(f"[对照②] 特异性@α={max(ALPHAS):+g}  |Δ横向|/|Δv|={out['specificity']['lateral_over_dv']:.2f}  "
          f"|Δ舒适|/|Δv|={out['specificity']['comfort_over_dv']:.2f}")

    mt, cit = boot_ci([r["term_ghost"][0] - r["base_ghost"][0] for r in rec], sc)
    mr, cir = boot_ci([r["recov_ghost"][0] - r["base_ghost"][0] for r in rec], sc)
    out["termination"] = {"project_out": {"dv": mt, "ci95": list(cit)},
                          "recover": {"dv": mr, "ci95": list(cir)}}
    print(f"[对照③] project_out Δv={mt:+.4f} [{cit[0]:+.4f},{cit[1]:+.4f}]   "
          f"recover Δv={mr:+.4f} [{cir[0]:+.4f},{cir[1]:+.4f}]")

    # F_m = 通路利用率（协议 §3④）：steering 诱发的减速峰值 / 真实危险诱发的减速（同批场景）
    real_b, real_ci = boot_ci([r["base_clean"][0] - r["base_ghost"][0] for r in rec], sc)
    peak = min(out["dose"][f"{a:+g}"]["dv"] for a in ALPHAS)
    out["pathway_utilization_F"] = {
        "steering_peak_deceleration": peak, "real_hazard_deceleration": real_b,
        "real_hazard_ci95": list(real_ci),
        "F_m": (peak / real_b) if abs(real_b) > 1e-6 else None,
        "note": ("分母（真实危险诱发的减速）与 0 不可区分时 F_m 无定义，不得报比值"
                 if not (real_ci[0] > 0 or real_ci[1] < 0) else "分母显著≠0，F_m 可读")}
    print(f"[F_m] 真实危险诱发减速 = {real_b:+.4f} [{real_ci[0]:+.4f},{real_ci[1]:+.4f}]；"
          f"steering 峰值 = {peak:+.4f}  -> {out['pathway_utilization_F']['note']}")

    out["per_event"] = rec
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[F-dd] wrote {args.out}")


if __name__ == "__main__":
    main()
