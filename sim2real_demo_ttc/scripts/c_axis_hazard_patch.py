"""C 轴（G1 口径）|危险信号的失效定位：clean↔ghost 配对输入互换的逐层 activation patching。

**为什么是这个配对**（工单 §0 的直接落实，见 amendments.md §CE/A27）：
工单要求新候选一律用现有 G1 语料 + N1 负例体系测 G/F/C，只有 I 轴才需要真实↔仿真域配对。
G1 语料本身就自带一组**配对真实输入**：同一事件的 clean 帧（危险未出现）与 ghost 帧（危险已出现）。
跑 ghost 前向、把第 L 层的图像 token 换成同一事件 clean 侧的激活，测行为回到 clean 的比例：

    recovery(L) = ( v_cmd(patch) − v_cmd(ghost) ) / ( v_cmd(clean) − v_cmd(ghost) )

它回答的是「**危险引起的行为变化从哪一层进入网络**」，与既有 C 轴（sim↔real 域配对）
问的「域迁移引起的退化从哪一层进入」是**同一套 patching 方法、不同的配对来源**。
两者在报告与矩阵中分列为 **C-hazard** 与 **C-domain**，不混为一谈。

口径与既有 C 轴逐条一致：corruption = 配对真实输入互换（禁用噪声破坏）；
指标 = 连续量（禁用二值化）；退化样本按 |v_clean − v_ghost| 取前 K；
C_m = top-2 层 recovery 占比（recovery 先 clip 至 ≥0），逐场景先算再汇总，scene 级 bootstrap；
并附恢复剖面形状诊断（修正案 HL/A24：剖面单调递减时该公式前提不成立）。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import cv2
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def boot(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dd", "ltf", "ddv2"])
    ap.add_argument("--k", type=int, default=12, help="按 |v_clean − v_ghost| 取前 K 个事件")
    ap.add_argument("--min-gap", type=float, default=0.02)
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    label = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2"}[args.model]
    cache = W / {"dd": "dd_cache", "ltf": "ltf_cache", "ddv2": "ddv2_cache"}[args.model]
    if not args.out:
        args.out = str(RES / f"c_axis_hazard_{args.model}.json")

    # 零 GPU 选样：从既有缓存里按 |v_clean − v_ghost| 取退化最狠的 A 类事件
    rows = []
    for p in sorted(cache.glob("*.npz")):
        d = np.load(p, allow_pickle=True); m = json.loads(str(d["meta"]))
        if m["event_type"] != "A":
            continue
        vc = float(d["commanded_speed_clean"].mean()); vg = float(d["commanded_speed_ghost"].mean())
        rows.append({"eid": m["event_id"], "scene": m["scene_name"], "gap": vc - vg,
                     "absgap": abs(vc - vg)})
    rows.sort(key=lambda r: -r["absgap"])
    sel = [r for r in rows if r["absgap"] >= args.min_gap][: args.k]
    print(f"[C-haz/{label}] A 类 {len(rows)} 个，选中 {len(sel)}（|gap| ≥ {args.min_gap}，"
          f"范围 {sel[-1]['absgap']:.3f} ~ {sel[0]['absgap']:.3f}）")

    sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
    sys.path.insert(0, str(RES / "ltf_g1_adapter"))
    sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
    if args.model == "dd":
        from dd_adapter import DDRunner as Runner, bbox_to_tokens
    elif args.model == "ltf":
        from ltf_adapter import LTFRunner as Runner, bbox_to_tokens
    else:
        from ddv2_adapter import DDV2Runner as Runner, bbox_to_tokens
    runner = Runner(device=args.device)
    lidar = None
    if args.model == "ddv2":
        from ddv2_adapter import NuScenesLidar
        lidar = NuScenesLidar(args.nuscenes_root)

    def _run(img, spd, tok):
        return runner.run(img, spd, **({} if lidar is None else {"lidar_xyz": lidar.ego_points(tok)}))

    nL = len(runner.sas)
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}

    def read(fn):
        img = cv2.imread(str(Path(args.nuscenes_root) / fn))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recs, dropped = [], []
    for i, r in enumerate(sel):
        ev = evmap[r["eid"]]
        anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        ic = read(ev["x_clean_frames"][0]["filename"])
        ig = read(ev["x_ghost_frames"][0]["filename"])
        runner.set_patch(None); runner.set_steering(None)
        tc = ev["x_clean_frames"][0].get("sd_token"); tg = ev["x_ghost_frames"][0].get("sd_token")
        oc = _run(ic, anchor, tc); vc = oc["commanded_speed"]
        # 参考侧（clean）逐层图像 token 激活
        # patch **全部 320 个融合 token**（256 图像 + 64 BEV latent），与既有 C-domain 实现一致。
        # 只 patch 图像段时 patch-ALL 充分割集自检不通过（DD +0.52 / LTF −0.03），见 amendments §CE/A32。
        V = [runner._buf[l].cpu().numpy() for l in range(nL)]
        og = _run(ig, anchor, tg); vg = og["commanded_speed"]
        den = vc - vg
        # **运行时分母守卫**：选样用的是缓存里 2 帧平均的 gap，而 patching 只跑 frame[0]，
        # 两者可能相差很多。分母接近 0 时 recovery 会爆炸（实测出现过 −439 的单点，
        # 把 patch-ALL 均值从 +0.50 拖到 −36）。故按**本次运行实测的 den** 再筛一次。
        if abs(den) < args.min_gap:
            dropped.append({"eid": r["eid"], "run_gap": den})
            print(f"[C-haz/{label}] skip {r['eid']}：运行时 gap={den:+.4f} < {args.min_gap}", flush=True)
            continue
        rec = {"eid": r["eid"], "scene": r["scene"], "v_clean": vc, "v_ghost": vg, "gap": den}
        for l in range(nL):
            runner.set_patch({l: V[l]}, tokens="all")
            vp = _run(ig, anchor, tg)["commanded_speed"]
            rec[f"L{l}"] = (vp - vg) / den if abs(den) > 1e-9 else float("nan")
        runner.set_patch({l: V[l] for l in range(nL)}, tokens="all")
        rec["ALL"] = (_run(ig, anchor, tg)["commanded_speed"] - vg) / den if abs(den) > 1e-9 else float("nan")
        runner.set_patch(None)
        recs.append(rec)
        print(f"[C-haz/{label}] {i+1}/{len(sel)} {r['eid']} gap={den:+.3f} ALL={rec['ALL']:+.3f}", flush=True)

    R = np.array([[x[f"L{l}"] for l in range(nL)] for x in recs], float)
    Xc = np.clip(R, 0, None)
    top2 = np.array([np.sort(x)[-2:].sum() / x.sum() if x.sum() > 1e-9 else np.nan for x in Xc])
    prof = Xc.mean(0)
    rho, pv = stats.spearmanr(np.arange(nL), prof)
    arg = [int(np.argmax(Xc[i])) if Xc[i].sum() > 1e-9 else -1 for i in range(len(R))]
    cnt = {l: int(sum(a == l for a in arg)) for l in range(nL)}
    pdist = np.array([cnt[l] for l in range(nL)], float); pdist /= max(pdist.sum(), 1)
    ent = float(-(pdist[pdist > 0] * np.log(pdist[pdist > 0])).sum() / np.log(nL))
    allv = np.array([x["ALL"] for x in recs], float)
    base = 2.0 / nL
    applicable = bool(rho > -0.7)
    cm = boot(top2)
    out = {"model": label, "pairing": "G1 clean↔ghost（配对真实输入互换）",
           "construct": "C-hazard：危险引起的行为变化从哪一层进入",
           "metric": "recovery(L) = (v_patch − v_ghost)/(v_clean − v_ghost)，v = commanded_speed",
           "n_events": len(recs), "n_dropped_small_gap": len(dropped), "dropped": dropped,
           "n_layers": nL, "diffuse_baseline_top2_share": base,
           "patch_all_recovery": {"mean": float(np.nanmean(allv)), "median": float(np.nanmedian(allv)),
                                  "sufficient_cut_set": bool(0.7 <= float(np.nanmedian(allv)) <= 1.3),
                                  "note": "中位数远离 1 ⇒ 被 patch 的 token 不是该配对下的充分割集，"
                                          "逐层占比不可解释"},
           "C_m_top2_share": cm, "recovery_profile_mean": prof.tolist(),
           "spearman_layer_vs_recovery": float(rho), "spearman_p": float(pv),
           "argmax_layer_hist": cnt, "argmax_layer_mode": int(max(cnt, key=cnt.get)),
           "argmax_normalized_entropy": ent,
           "profile_shape": ("级联（cascade）" if rho < -0.7 else
                             ("内部峰（localized）" if rho > -0.3 else "混合/不明确")),
           "top2_share_formula_applicable": applicable, "per_event": recs}
    if not out["patch_all_recovery"]["sufficient_cut_set"]:
        out["verdict"] = (f"不可估（**充分割集自检不通过**：patch-ALL 中位数 "
                          f"{np.nanmedian(allv):+.3f}，远离 1）")
    elif not applicable:
        out["verdict"] = "不可估（公式前提不成立：剖面单调递减）"
    else:
        ci = cm["ci95"]
        out["verdict"] = ("PASS：C 显著高于弥散基线" if ci[0] > base else
                          ("FAIL：显著低于弥散基线" if ci[1] < base else "不可估：CI 跨弥散基线"))
    print(f"[C-haz/{label}] patch-ALL = {np.nanmean(allv):+.3f}；C_m = {cm['mean']:.3f} "
          f"{np.round(cm['ci95'],3).tolist()}（基线 {base:.3f}）")
    print(f"[C-haz/{label}] 剖面 {prof[0]:.3f}→{prof[-1]:.3f}，Spearman={rho:+.3f}，"
          f"{out['profile_shape']}；责任层众数 L{out['argmax_layer_mode']}，熵 {ent:.3f}")
    print(f"[C-haz/{label}] 判定：{out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[C-haz/{label}] wrote {args.out}")


if __name__ == "__main__":
    main()
