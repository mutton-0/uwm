"""T-C|C 轴(Concentration,失效集中度)读数 —— 复用既有 DiffusionDrive 因果修补数据。

数据源(不重新跑因果实验,工单 §5 明文要求"整理现有数据"):
  outputs/ghosthead_infer/patching/recovery.csv       逐场景 × 8 层 activation-patching recovery
  outputs/ghosthead_infer/cka_recovery/cka_recovery.csv 逐层 JS / 1-CKA / recovery 均值

公式(选型协议 §3⑤):
  C_m = top-2 层 recovery 占比 / Σ_L recovery(L)

口径纪律(预注册):
  * recovery<0 表示"换上参考侧激活反而更差",不是定位证据 ⇒ **主读数先 clip 到 0** 再算占比;
    不 clip 的版本作为敏感性分析并列报告(分母可能过零 ⇒ 比值不稳定,这正是要 clip 的理由);
  * 逐场景先算 C 再汇总(不是先汇总 recovery 再算一个 C),bootstrap 以 **scene 为重采样单位**;
  * 弥散基线 = 2/8 = 0.250(8 层均匀分布时 top-2 应占的份额),是 C 的"无集中"参照点。
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np


def load_recovery(path):
    rows = list(csv.DictReader(open(path)))
    layers = [k for k in rows[0] if k.startswith("L") and k[1:].isdigit()]
    layers.sort(key=lambda s: int(s[1:]))
    R, names, gaps = [], [], []
    for r in rows:
        R.append([float(r[l]) for l in layers]); names.append(r["scene"]); gaps.append(float(r["gap"]))
    return np.array(R), names, np.array(gaps), layers


def conc(R, clip=True, k=2):
    X = np.clip(R, 0, None) if clip else R.copy()
    s = X.sum(1)
    ok = np.abs(s) > 1e-9
    top = np.sort(X, 1)[:, -k:].sum(1)
    C = np.full(len(X), np.nan)
    C[ok] = top[ok] / s[ok]
    return C


def boot(C, n=5000, seed=0):
    C = C[np.isfinite(C)]
    rng = np.random.default_rng(seed)
    m = np.array([C[rng.integers(0, len(C), len(C))].mean() for _ in range(n)])
    return {"mean": float(C.mean()), "median": float(np.median(C)),
            "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], "n_scene": int(len(C))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery", default="/data/ruolin/uwm/outputs/ghosthead_infer/patching/recovery.csv")
    ap.add_argument("--cka", default="/data/ruolin/uwm/outputs/ghosthead_infer/cka_recovery/cka_recovery.csv")
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/results/c_axis_concentration.json")
    args = ap.parse_args()

    R, names, gaps, layers = load_recovery(args.recovery)
    n_layers = R.shape[1]
    out = {"model": "DiffusionDrive (TransFuser encoder, 8x SelfAttention)",
           "source": args.recovery, "n_scenes": len(names), "n_layers": n_layers,
           "layers": layers, "scenes": names,
           "diffuse_baseline_top2_share": 2.0 / n_layers}

    for clip, key in ((True, "primary_clipped"), (False, "sensitivity_unclipped")):
        C = conc(R, clip=clip)
        out[key] = {**boot(C), "per_scene": {n: (None if not np.isfinite(c) else float(c))
                                             for n, c in zip(names, C)}}
        print(f"[T-C] C_m ({'clip>=0 主读数' if clip else '不 clip 敏感性'}) = "
              f"{out[key]['mean']:.3f}  95% CI {np.round(out[key]['ci95'],3).tolist()}  "
              f"n={out[key]['n_scene']}  (弥散基线 {2/n_layers:.3f})")

    # top-k 曲线(k=1..4):集中度的形状,而不是只报一个点
    out["topk_share_curve"] = {str(k): boot(conc(R, True, k)) for k in (1, 2, 3, 4)}

    # 责任层的一致性:argmax 分布 + 归一化熵。这才是"可定位"的直接证据
    Xc = np.clip(R, 0, None)
    arg = [int(np.argmax(Xc[i])) if Xc[i].sum() > 1e-9 else -1 for i in range(len(R))]
    cnt = {l: int(sum(a == l for a in arg)) for l in range(n_layers)}
    p = np.array([cnt[l] for l in range(n_layers)], float); p = p / max(p.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(n_layers))
    out["argmax_layer_hist"] = cnt
    out["argmax_layer_mode"] = int(max(cnt, key=cnt.get))
    out["argmax_normalized_entropy"] = ent
    out["deep_layer_share_L4_6"] = float(np.mean([a in (4, 5, 6) for a in arg]))
    print(f"[T-C] 责任层 argmax 直方图 {cnt} -> 众数 L{out['argmax_layer_mode']}, "
          f"归一化熵={ent:.3f} (0=完全集中,1=完全弥散), L4-6 占比={out['deep_layer_share_L4_6']:.3f}")

    # 逐层 recovery 均值剖面 + 与散度指标的对照(症状≠病因,复用既有 cka 表)
    out["recovery_profile_mean"] = np.clip(R, 0, None).mean(0).tolist()
    if Path(args.cka).exists():
        ck = list(csv.DictReader(open(args.cka)))
        out["divergence_vs_causal"] = {
            "layer": [int(r["layer"]) for r in ck],
            "JS": [float(r["JS"]) for r in ck],
            "one_minus_CKA": [float(r["1-CKA"]) for r in ck],
            "recovery": [float(r["recovery"]) for r in ck]}
        js = np.array(out["divergence_vs_causal"]["JS"]); rc = np.array(out["divergence_vs_causal"]["recovery"])
        cka = np.array(out["divergence_vs_causal"]["one_minus_CKA"])
        out["divergence_vs_causal"]["argmax"] = {"JS": int(js.argmax()), "one_minus_CKA": int(cka.argmax()),
                                                 "recovery": int(rc.argmax())}
        print(f"[T-C] 散度 vs 因果 argmax: JS=L{js.argmax()}  1-CKA=L{cka.argmax()}  recovery=L{rc.argmax()}")

    # 三态判定
    ci = out["primary_clipped"]["ci95"]; base = 2.0 / n_layers
    if ci[0] > base:
        verdict = "PASS：C 显著高于弥散基线，失效集中在少数层，支持低成本定向修复(LoRA)"
    elif ci[1] < base:
        verdict = "FAIL：C 显著低于弥散基线，失效弥散"
    else:
        verdict = "不可估：CI 跨过弥散基线，现有场景数不足以判定集中与否"
    out["verdict"] = verdict
    out["verdict_rule"] = "primary_clipped 的 scene 级 bootstrap 95% CI 与弥散基线 2/L 的位置关系"
    print(f"[T-C] 判定：{verdict}")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[T-C] wrote {args.out}")


if __name__ == "__main__":
    main()
