"""C 轴补测|SimLingo 的逐层 activation patching（补齐轴矩阵缺口）。

依据：选型协议 §3⑤（C = 失效集中度）。口径与 DiffusionDrive 侧**逐条同构**：
  * 域配对：sim = CARLA 引擎渲染 ↔ real = 世界模型真实感重绘（同场景同几何，do(appearance)）；
  * corruption **一律为配对真实输入互换**（禁用噪声破坏）：跑 real 前向，把第 L 层的激活换成
    同一场景 sim 侧的对应激活，测行为回到 sim 的比例 recovery(L)；
  * 指标**一律为连续量**（禁用二值化）：recovery(L) = (v_patch − v_real) / (v_sim − v_real)，
    v = 模型下发的目标速度 commanded_speed（与 DD 侧的轨迹 recovery 同为"后果连续量"）；
  * 退化样本选择：按 |v_sim − v_real| 取前 K 个 scene-variant（与 DD 侧"取 real 退化最狠的 top-12"同规则）。

**与 DD 侧的一处结构性差异（须并列声明）**：DD patch 的是 encoder 的全部 320 个融合 token；
SimLingo 只 patch **vision token 段**——语言段在两条件之间长度与内容都会变，对应关系无定义
（与 g2_cache schema v2 的同一条纪律）。故 SimLingo 的"充分割集"检验（patch-ALL）
问的是"全部 24 层的 vision token 是否是差异的充分割集"，而非全序列。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from omegaconf import OmegaConf

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
IDOM = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain")
DATA_ROOT = "/data/Zhengyang/Auto_Eval/ghosthead_v1"
FPS = 10.0


def ego_speed(scene, fi):
    sc = json.load(open(Path(f"{DATA_ROOT}/renders/{scene}") / "scene.json"))
    M = np.array([np.array(f["ego_to_world"], float) for f in sc["frames"]])
    i = min(max(fi, 1), len(M) - 1)
    return float(np.linalg.norm((M[i][:3, 3] - M[i - 1][:3, 3])[:2]) * FPS)


def select_degraded(k, sec):
    """零 GPU：从 T-I 已抽好的 acts_simlingo.npz meta 里按 |v_sim − v_real| 选退化最狠的 K 个。"""
    d = np.load(IDOM / "acts_simlingo.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    by = {}
    for m in meta:
        by[(m["scene"], m["source"], m["sec"])] = m.get("commanded_speed")
    rows = []
    for (sc, src, t), v in by.items():
        if src != "real" or v is None or t != sec:
            continue
        s = by.get((sc, "sim", t))
        if s is None:
            continue
        rows.append({"scene": sc, "sec": t, "v_sim": s, "v_real": v, "gap": abs(s - v)})
    rows.sort(key=lambda r: -r["gap"])
    return rows[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12, help="取行为退化最狠的前 K 个 scene-variant")
    ap.add_argument("--sec", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=0.05, help="v_sim−v_real 太小则 recovery 分母不稳，剔除")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(RES / "c_axis_simlingo.json"))
    args = ap.parse_args()

    sel = [r for r in select_degraded(args.k * 3, args.sec) if r["gap"] >= args.min_gap][: args.k]
    print(f"[C-sl] 选中 {len(sel)} 个退化场景（|v_sim−v_real| ≥ {args.min_gap}，"
          f"gap 范围 {sel[-1]['gap']:.3f} ~ {sel[0]['gap']:.3f}）")

    cfg = OmegaConf.to_container(OmegaConf.load(
        "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"), resolve=True)
    cfg["model"]["device"] = args.device
    sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
    from simlingo_runner import SimLingoRunner
    from g2_cache import commanded_speed
    runner = SimLingoRunner(cfg, capture_hidden=True)

    def run(img, spd):
        r = runner.infer(img, spd, pool_modes=("vision_mean",))
        return float(commanded_speed(r.waypoints)), r

    def vis_states():
        """最后一次全序列前向里，各层 vision token 的激活 [L][n_vis, C]。"""
        hs = runner._layer_outputs[-runner.n_layers:]
        ids = runner._adaptor_dict["language__ids"][0]
        import torch
        vis = torch.nonzero(ids == runner.img_context_token_id).flatten()
        return [h[0, vis, :].float().cpu().numpy() for h in hs]

    rows = []
    for i, r in enumerate(sel):
        fi = int(round(r["sec"] * FPS))
        spd = ego_speed(r["scene"], fi)
        im = {s: cv2.cvtColor(cv2.imread(str(IDOM / "frames" / f"{r['scene']}__{s}_t{r['sec']}.png")),
                              cv2.COLOR_BGR2RGB) for s in ("sim", "real")}
        runner.set_patch(None); runner.set_steering(None)
        v_sim, _ = run(im["sim"], spd); V = vis_states()
        v_real, _ = run(im["real"], spd)
        den = v_sim - v_real
        rec = {"scene": r["scene"], "sec": r["sec"], "v_sim": v_sim, "v_real": v_real, "gap": den}
        for l in range(runner.n_layers):
            runner.set_patch({l: V[l]})
            vp, _ = run(im["real"], spd)
            rec[f"L{l}"] = (vp - v_real) / den if abs(den) > 1e-9 else float("nan")
        runner.set_patch({l: V[l] for l in range(runner.n_layers)})
        va, _ = run(im["real"], spd)
        rec["ALL"] = (va - v_real) / den if abs(den) > 1e-9 else float("nan")
        runner.set_patch(None)
        rows.append(rec)
        print(f"[C-sl] {i+1}/{len(sel)} {r['scene']} gap={den:+.3f}  ALL={rec['ALL']:+.3f}  "
              f"argmax=L{int(np.nanargmax([max(rec[f'L{l}'],0) for l in range(runner.n_layers)]))}")

    nL = runner.n_layers
    R = np.array([[rw[f"L{l}"] for l in range(nL)] for rw in rows], float)
    out = {"model": "SimLingo", "n_scenes": len(rows), "n_layers": nL,
           "domain_pair": "sim = CARLA 引擎渲染 ↔ real = 世界模型真实感重绘",
           "patched_positions": "仅 vision token 段（语言段长度/内容在两条件间会变，对应关系无定义）",
           "metric": "recovery(L) = (v_patch − v_real) / (v_sim − v_real)，v = commanded_speed",
           "diffuse_baseline_top2_share": 2.0 / nL, "per_scene": rows}

    allv = np.array([rw["ALL"] for rw in rows], float)
    out["patch_all_recovery"] = {"mean": float(np.nanmean(allv)), "median": float(np.nanmedian(allv)),
                                 "frac_ge_0.9": float(np.mean(allv >= 0.9)),
                                 "note": "充分割集自检：接近 1 表示 24 层 vision token 足以解释 sim/real 差异"}
    print(f"\n[C-sl] patch-ALL recovery 均值 = {np.nanmean(allv):+.3f}（中位 {np.nanmedian(allv):+.3f}）")

    def conc(X, k=2):
        Xc = np.clip(X, 0, None); s = Xc.sum(1)
        ok = np.abs(s) > 1e-9
        c = np.full(len(Xc), np.nan)
        c[ok] = np.sort(Xc, 1)[:, -k:].sum(1)[ok] / s[ok]
        return c

    def boot(v, n=5000, seed=0):
        v = v[np.isfinite(v)]; rng = np.random.default_rng(seed)
        m = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n_scene": int(len(v))}

    out["primary_clipped"] = boot(conc(R))
    out["sensitivity_unclipped"] = boot(np.array([
        (np.sort(r)[-2:].sum() / r.sum()) if abs(r.sum()) > 1e-9 else np.nan for r in R]))
    out["topk_share_curve"] = {str(k): boot(conc(R, k)) for k in (1, 2, 3, 4)}
    Xc = np.clip(R, 0, None)
    arg = [int(np.argmax(Xc[i])) if Xc[i].sum() > 1e-9 else -1 for i in range(len(R))]
    cnt = {l: int(sum(a == l for a in arg)) for l in range(nL)}
    p = np.array([cnt[l] for l in range(nL)], float); p /= max(p.sum(), 1)
    out["argmax_layer_hist"] = cnt
    out["argmax_layer_mode"] = int(max(cnt, key=cnt.get))
    out["argmax_normalized_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(nL))
    out["recovery_profile_mean"] = Xc.mean(0).tolist()

    ci = out["primary_clipped"]["ci95"]; base = 2.0 / nL
    out["verdict"] = ("PASS：C 显著高于弥散基线，失效集中在少数层" if ci[0] > base else
                      ("FAIL：C 显著低于弥散基线，失效弥散" if ci[1] < base else
                       "不可估：CI 跨过弥散基线"))
    print(f"[C-sl] C_m = {out['primary_clipped']['mean']:.3f} "
          f"{np.round(ci,3).tolist()}（弥散基线 {base:.3f}，24 层）")
    print(f"[C-sl] 责任层众数 L{out['argmax_layer_mode']}，归一化熵 {out['argmax_normalized_entropy']:.3f}")
    print(f"[C-sl] 判定：{out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[C-sl] wrote {args.out}")


if __name__ == "__main__":
    main()
