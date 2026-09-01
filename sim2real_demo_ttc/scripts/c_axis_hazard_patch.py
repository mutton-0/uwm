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

import argparse, json, importlib.util, sys
from pathlib import Path

# 两个 VLA 候选（Alpamayo / AutoVLA）跑在没有 numpy 的 py312 解释器里，
# 依赖来自 Alpamayo 的 venv；**必须在 import numpy 之前**挂上（append 而非 insert，
# 故 simscale 环境自带的 numpy 仍然优先，DD 系候选的行为一字不变）。
if importlib.util.find_spec("numpy") is None:
    for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
               "/data/Zhengyang/alpamayo/src"):
        if _p not in sys.path:
            sys.path.append(_p)

import numpy as np
from scipy import stats

# cv2 只在 DD 系候选的图像读取路径里用；VLA 候选自己走 nuScenes devkit 取帧，
# 且它们所在的解释器没有 opencv，故延迟到用时再 import。
cv2 = None

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def boot(v, n=5000, seed=0):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n": int(len(v))}


def main():
    global W
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dd", "ltf", "ddv2", "alpa", "autovla", "simlingo"])
    ap.add_argument("--k", type=int, default=12, help="按 |v_clean − v_ghost| 取前 K 个事件")
    # 第二场景类型（前车急刹）：换工作目录与正例类名，patching 方法与判定一律不动。
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--pos", default="A", help="正例类名（G1 用 A，前车急刹用 LB）")
    ap.add_argument("--sl-config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--min-gap", type=float, default=0.02)
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", default="all", choices=["all", "image"],
                    help="patch 范围。DD 系用 all（320 融合 token，§CE/A32）；"
                         "纯 transformer 栈上 all 会结构性饱和，见 §CE/A39")
    ap.add_argument("--seed-floor", type=int, default=0,
                    help="Alpamayo 专用：同输入换 N 个 seed 跑 clean/ghost，量化采样噪声地板")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    W = Path(args.work)
    label = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
             "alpa": "Alpamayo-R1", "autovla": "AutoVLA", "simlingo": "SimLingo"}[args.model]
    cache = W / {"dd": "dd_cache", "ltf": "ltf_cache", "ddv2": "ddv2_cache",
                 "alpa": "alpa_cache", "autovla": "autovla_cache", "simlingo": "cache"}[args.model]
    IS_VLA = args.model in ("alpa", "autovla")
    IS_SL = args.model == "simlingo"
    if not args.out:
        args.out = str(RES / f"c_axis_hazard_{args.model}.json")

    # 零 GPU 选样：从既有缓存里按 |v_clean − v_ghost| 取退化最狠的 A 类事件
    rows = []
    if IS_SL:
        import h5py
        _ev = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}
        for p in sorted(cache.glob("*.h5")):
            m = _ev.get(p.stem)
            if m is None or m["event_type"] != args.pos:
                continue
            try:
                with h5py.File(p, "r") as f:
                    if "pred_speed" not in f["clean"]:
                        continue
                    vc = float(f["clean"]["pred_speed"][:].mean())
                    vg = float(f["ghost"]["pred_speed"][:].mean())
            except Exception:                                   # noqa: BLE001
                continue
            if not (np.isfinite(vc) and np.isfinite(vg)):
                continue
            rows.append({"eid": m["event_id"], "scene": m["scene_name"], "gap": vc - vg,
                         "absgap": abs(vc - vg)})
    for p in ([] if IS_SL else sorted(cache.glob("*.npz"))):
        d = np.load(p, allow_pickle=True); m = json.loads(str(d["meta"]))
        if m["event_type"] != args.pos:
            continue
        if "commanded_speed_clean" in d.files:
            vc = float(d["commanded_speed_clean"].mean()); vg = float(d["commanded_speed_ghost"].mean())
        else:                                   # VLA 缓存口径：{cond}/v_plan
            vc = float(d["clean/v_plan"][0]); vg = float(d["ghost/v_plan"][0])
        if not (np.isfinite(vc) and np.isfinite(vg)):
            continue
        rows.append({"eid": m["event_id"], "scene": m["scene_name"], "gap": vc - vg,
                     "absgap": abs(vc - vg)})
    rows.sort(key=lambda r: -r["absgap"])
    sel = [r for r in rows if r["absgap"] >= args.min_gap][: args.k]
    print(f"[C-haz/{label}] {args.pos} 类 {len(rows)} 个，选中 {len(sel)}（|gap| ≥ {args.min_gap}，"
          f"范围 {sel[-1]['absgap']:.3f} ~ {sel[0]['absgap']:.3f}）")

    if IS_SL:
        # SimLingo 侧：复用它在 C-domain 上已有的 patching 实现（`set_patch` 换 vision token 段）。
        # **patch 范围沿用 SimLingo 自己的 C-domain 约定（仅 vision token 段）**，
        # 这样同一个模型的 C-hazard 与 C-domain 是逐条可比的；而按 §CE/A39 的 AutoVLA 对照，
        # 换成整条 prompt 也不改变剖面形态，故这个选择不影响本次预测检验的结论。
        from omegaconf import OmegaConf
        sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
        cfg = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        from g2_cache import commanded_speed
        runner = SimLingoRunner(cfg, capture_hidden=True)
        nL = runner.n_layers
        import torch as _torch

        def sl_run(img, spd):
            r = runner.infer(img, spd, pool_modes=("vision_mean",))
            return float(commanded_speed(r.waypoints))

        def sl_vis_states():
            hs = runner._layer_outputs[-runner.n_layers:]
            ids = runner._adaptor_dict["language__ids"][0]
            vis = _torch.nonzero(ids == runner.img_context_token_id).flatten()
            return [h[0, vis, :].float().cpu().numpy() for h in hs]
    elif IS_VLA:
        # VLA 侧：读出对象是语言塔 decoder layer 的 prefill 隐状态；
        # 参考侧激活由 clean 那一次前向**当场抓全序列**（不能用缓存，缓存只存了池化结果）。
        if args.model == "alpa":
            sys.path.insert(0, str(RES / "alpamayo_g1_adapter"))
            from alpa_patch import AlpaPatchRunner
            runner = AlpaPatchRunner(device=args.device)
        else:
            sys.path.insert(0, str(RES / "autovla_g1_adapter"))
            from autovla_adapter import AutoVLARunner
            runner = AutoVLARunner(device=args.device, nuscenes_root=args.nuscenes_root)
        nL = runner.cap.n_layers
    else:
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
        nL = len(runner.sas)

    def _run(img, spd, tok):
        return runner.run(img, spd, **({} if lidar is None else {"lidar_xyz": lidar.ego_points(tok)}))

    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}

    def read(fn):
        global cv2
        if cv2 is None:
            import cv2 as _cv2; cv2 = _cv2
        img = cv2.imread(str(Path(args.nuscenes_root) / fn))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def vla_run(ev, cond):
        """VLA 侧的一次前向：图像取该条件的帧，ego/运动史一律锚到 clean（与缓存口径同构）。"""
        if args.model == "alpa":
            return runner.run(ev["scene_name"], ev[f"x_{cond}_frames"][0]["t"],
                              ego_anchor_t=ev["x_clean_frames"][0]["t"])
        anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        return runner.run(ev[f"x_{cond}_frames"][0]["sd_token"], anchor)

    seed_floor = None
    if IS_VLA and args.seed_floor:
        # **采样噪声地板**（Alpamayo 的 rollout 是 top_p/temperature 采样，非贪心）：
        # 同一输入换 N 个 seed，量化 v_plan 的离散度。若它与 clean−ghost 的 gap 同量级，
        # 则 recovery 的分母本身淹没在采样噪声里，C-hazard 一律判不可估。
        vals = {}
        for cond in ("clean", "ghost"):
            ev0 = evmap[sel[0]["eid"]]
            vs = []
            for sd in range(args.seed_floor):
                o = (runner.run(ev0["scene_name"], ev0[f"x_{cond}_frames"][0]["t"],
                                ego_anchor_t=ev0["x_clean_frames"][0]["t"], seed=1000 + sd)
                     if args.model == "alpa" else vla_run(ev0, cond))
                vs.append(o["commanded_speed"])
            vals[cond] = vs
            print(f"[C-haz/{label}] seed-floor {cond}: {np.round(vs,4).tolist()}", flush=True)
        sp = float(np.std(vals["clean"] + vals["ghost"]))
        seed_floor = {"event": sel[0]["eid"], "n_seeds": args.seed_floor,
                      "v_clean_by_seed": vals["clean"], "v_ghost_by_seed": vals["ghost"],
                      "sd_pooled": sp,
                      "note": "同输入换 seed 的 v_plan 标准差；与 |v_clean − v_ghost| 同量级即不可估"}

    recs, dropped = [], []
    for i, r in enumerate(sel):
        ev = evmap[r["eid"]]
        runner.set_patch(None)
        if IS_SL:
            # ego 速度锚到 clean（与 SimLingo 的 prompt_anchor 纪律一致：两条件唯一差异是图像）
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            ic = read(ev["x_clean_frames"][0]["filename"])
            ig = read(ev["x_ghost_frames"][0]["filename"])
            runner.set_steering(None)
            vc = sl_run(ic, anchor); V = sl_vis_states()
            vg = sl_run(ig, anchor)
        elif IS_VLA:
            runner.set_capture_full(True)
            oc = vla_run(ev, "clean"); vc = oc["commanded_speed"]
            V = [np.asarray(x, np.float32) for x in oc["full"]]
            Sc = V[0].shape[0]
            runner.set_capture_full(False)
            og = vla_run(ev, "ghost"); vg = og["commanded_speed"]
            if og["seq_len"] != Sc:
                dropped.append({"eid": r["eid"], "reason": "seq_len mismatch",
                                "clean_seq_len": Sc, "ghost_seq_len": og["seq_len"]})
                print(f"[C-haz/{label}] skip {r['eid']}：seq_len {Sc} vs {og['seq_len']}", flush=True)
                continue
        else:
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            ic = read(ev["x_clean_frames"][0]["filename"])
            ig = read(ev["x_ghost_frames"][0]["filename"])
            runner.set_steering(None)
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
        if IS_SL:
            ghost_again = lambda: sl_run(ig, anchor)                      # noqa: E731
        elif IS_VLA:
            ghost_again = lambda: vla_run(ev, "ghost")["commanded_speed"]  # noqa: E731
        else:
            ghost_again = lambda: _run(ig, anchor, tg)["commanded_speed"]  # noqa: E731
        setp = (lambda d: runner.set_patch(d)) if IS_SL else \
            (lambda d: runner.set_patch(d, tokens=args.tokens))            # noqa: E731
        for l in range(nL):
            setp({l: V[l]})
            rec[f"L{l}"] = (ghost_again() - vg) / den if abs(den) > 1e-9 else float("nan")
        setp({l: V[l] for l in range(nL)})
        rec["ALL"] = (ghost_again() - vg) / den if abs(den) > 1e-9 else float("nan")
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
    # **承诺层（commitment layer）**：剖面为阶跃/级联时 top-2 占比无意义，但有一个
    # 良定义且跨模型可比的替代读数 —— 「patch 到第几层为止行为仍能完全回到 clean」。
    # 定义：满足 mean_recovery(L) ≥ 0.9 的最大 L（要求 L 及之前所有层都满足，避免抖动误判）。
    # 它回答「决策在多深处被定死」，对阶跃剖面是**唯一**有信息量的形状量。
    ok = np.where(prof >= 0.9)[0]
    Lc = int(ok[-1]) if len(ok) else -1
    pre = -1
    for l in range(nL):
        if prof[l] >= 0.9:
            pre = l
        else:
            break
    commitment = {"layer": Lc, "depth_fraction": float((Lc + 1) / nL),
                  "prefix_layer": pre,
                  "criterion": "mean recovery ≥ 0.9 的**最深**层（前缀版并列报告："
                               "要求 L 及之前每一层都满足）",
                  "reading": "patch 到该层为止行为仍完全回到 clean ⇒ 危险信号在此之后才被定死",
                  "why_not_top2": "剖面为阶跃/级联时 top-2 占比只量到 1/L 的倍数（§4.4.4），"
                                  "承诺层是此形态下唯一有信息量且跨模型可比的形状量"}
    allv = np.array([x["ALL"] for x in recs], float)
    base = 2.0 / nL
    applicable = bool(rho > -0.7)
    cm = boot(top2)
    out = {"model": label, "pairing": "G1 clean↔ghost（配对真实输入互换）",
           "patch_tokens": ("vision（沿用 SimLingo C-domain 约定）" if IS_SL else args.tokens),
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
           "top2_share_formula_applicable": applicable, "commitment_layer": commitment,
           "per_event": recs}
    if seed_floor is not None:
        out["sampling_noise_floor"] = seed_floor
        gaps = [abs(x["gap"]) for x in recs]
        ratio = seed_floor["sd_pooled"] / max(float(np.median(gaps)), 1e-9) if gaps else float("inf")
        out["sampling_noise_floor"]["sd_over_median_gap"] = float(ratio)
        out["sampling_noise_floor"]["dominates"] = bool(ratio >= 0.5)
    if seed_floor is not None and out["sampling_noise_floor"]["dominates"]:
        out["verdict"] = (f"不可估（**采样噪声地板不通过**：同输入换 seed 的 v_plan sd "
                          f"{seed_floor['sd_pooled']:.4f}，达到 clean−ghost 中位 gap 的 "
                          f"{out['sampling_noise_floor']['sd_over_median_gap']:.2f} 倍，"
                          f"recovery 的分母淹没在采样噪声里）")
    elif not out["patch_all_recovery"]["sufficient_cut_set"]:
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
    print(f"[C-haz/{label}] 承诺层 L{commitment['layer']}（深度 {commitment['depth_fraction']:.2f}）")
    print(f"[C-haz/{label}] 剖面 {prof[0]:.3f}→{prof[-1]:.3f}，Spearman={rho:+.3f}，"
          f"{out['profile_shape']}；责任层众数 L{out['argmax_layer_mode']}，熵 {ent:.3f}")
    print(f"[C-haz/{label}] 判定：{out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[C-haz/{label}] wrote {args.out}")


if __name__ == "__main__":
    main()
