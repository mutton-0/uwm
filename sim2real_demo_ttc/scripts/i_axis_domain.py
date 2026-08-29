"""T-I|I 轴(Invariance,域不变性):域配对激活散度 D_L + 干涉角 θ_L + I_m。

工单依据:four_axis_proof_experiment_workorder.md §4;公式见选型协议 §3③:
    D_L = E_i ||Z_L(x_real) - Z_L(x_sim)|| / E_i ||Z_L(x)||
    θ_L = cos( v_domain(L), v_hazard(L) )
    I_m = 1 - D_{L*}   (扣分项 |θ_{L*}|, L* = 该模型自己的概念峰层)

域配对(见 amendments.md FA.1 偏离 1):sim = CARLA 引擎渲染,real = 世界模型真实感重绘,
同场景同构图同 actor,唯一变量是渲染风格 = do(appearance)。

纪律:
  * D_L 是**模型内归一化**的比值(协议 §4 规则 1),这是它可以跨模型比较的唯一理由;
  * v_domain(L) = 配对差 δ 的 PC1(与 G 轴 v_hazard 的提取法同源),符号按"real 投影更大"校准;
  * bootstrap 以 **scene 为重采样单位**(同一场景的 4 个时刻不是独立样本);
  * SimLingo 侧 v_domain 落盘为 `v_domain.npy` —— 这是方向发现计划 Stage B6 的缺口产出物。
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = {
    "simlingo":  {"pools": ["vision_mean", "query_mean", "last_token"], "n_layers": 24, "primary": "vision_mean"},
    "dd":        {"pools": ["vision_mean", "all_mean"],                 "n_layers": 8,  "primary": "vision_mean"},
}


def load(npz_path, pool, n_layers):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    idx = {(m["scene"], m["source"], m["sec"]): m for m in meta}
    keys = defaultdict(dict)
    for k in d.files:
        if k == "meta" or f"|{pool}|" not in k:
            continue
        sc, src, t, pl, L = k.split("|")
        keys[(sc, src, int(t[1:]))][int(L[1:])] = d[k]
    pairs = []
    for (sc, src, t), layers in keys.items():
        if src != "real":
            continue
        s = keys.get((sc, "sim", t))
        if s is None or len(layers) < n_layers or len(s) < n_layers:
            continue
        pairs.append({"scene": sc, "sec": t,
                      "real": [layers[l] for l in range(n_layers)],
                      "sim": [s[l] for l in range(n_layers)],
                      "meta_real": idx.get((sc, "real", t), {}), "meta_sim": idx.get((sc, "sim", t), {})})
    return pairs, meta


def pc1(D):
    """δ 的 PC1(不去均值 —— δ 的均值方向本身就是域方向,见 amendments A5)。"""
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    w = Vt[0]
    if (D @ w).mean() < 0:
        w = -w
    evr = float((D @ w).var() / (D.var(0).sum() + 1e-12))
    return w / (np.linalg.norm(w) + 1e-12), evr


def boot_scene(vals, scenes, n=2000, seed=0):
    vals = np.asarray(vals, float); scenes = np.asarray(scenes)
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(v)
    ks = sorted(by); rng = np.random.default_rng(seed); o = []
    for _ in range(n):
        x = [v for i in rng.integers(0, len(ks), len(ks)) for v in by[ks[i]]]
        o.append(np.mean(x))
    return {"mean": float(np.mean(vals)), "ci95": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
            "n_pairs": int(len(vals)), "n_scenes": int(len(ks))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain")
    ap.add_argument("--res", default="/data/ruolin/uwm/sim2real_demo_ttc/results")
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/results/i_axis_domain.json")
    args = ap.parse_args()
    RES = Path(args.res)
    OUT = {"domain_pair": "sim = CARLA 引擎渲染 ↔ real = 世界模型真实感重绘（同场景同几何，do(appearance)）",
           "note": "非 3DGS 重建；替代方案与代价见 amendments.md FA.1 偏离 1", "models": {}}

    for mkey, spec in MODELS.items():
        f = Path(args.dir) / f"acts_{mkey}.npz"
        if not f.exists():
            print(f"[T-I] 跳过 {mkey}（{f} 不存在）"); continue
        M = {"pools": {}}
        for pool in spec["pools"]:
            pairs, _ = load(f, pool, spec["n_layers"])
            if not pairs:
                continue
            scenes = [p["scene"] for p in pairs]
            DL, dl_boot, vdom, evr = [], [], [], []
            for l in range(spec["n_layers"]):
                dn = np.array([np.linalg.norm(p["real"][l] - p["sim"][l]) for p in pairs])
                zn = np.array([np.linalg.norm(p["real"][l]) for p in pairs] +
                              [np.linalg.norm(p["sim"][l]) for p in pairs])
                DL.append(float(dn.mean() / (zn.mean() + 1e-12)))
                dl_boot.append(boot_scene(dn / (zn.mean() + 1e-12), scenes, n=1000))
                D = np.stack([p["real"][l] - p["sim"][l] for p in pairs])
                w, e = pc1(D); vdom.append(w); evr.append(e)
            M["pools"][pool] = {"D_L": DL, "D_L_bootstrap": dl_boot, "EVR1_by_layer": evr,
                                "n_pairs": len(pairs), "n_scenes": len(set(scenes))}
            print(f"[T-I/{mkey}/{pool}] n_pair={len(pairs)} ({len(set(scenes))} scene)  "
                  f"D_L = " + " ".join(f"{x:.3f}" for x in DL))

            # 干涉角 θ_L = cos(v_domain, v_hazard)
            ang = {}
            if mkey == "simlingo":
                for nm, fn in (("v_hazard_clean", "v_hazard_clean.npy"),
                               ("v_hazard_carla", "v_hazard_carla.npy")):
                    if (RES / fn).exists():
                        vh = np.load(RES / fn)
                        if vh.shape[0] == spec["n_layers"] and vh.shape[1] == vdom[0].shape[0]:
                            ang[nm] = [float(abs(np.dot(vdom[l], vh[l]))) for l in range(spec["n_layers"])]
                vb = RES.parent / "variants/n1_d2/results/v_brake_query_mean.npy"
                if vb.exists():
                    w = np.load(vb)
                    if w.shape[1] == vdom[0].shape[0]:
                        ang["v_brake"] = [float(abs(np.dot(vdom[l], w[l]))) for l in range(spec["n_layers"])]
                if pool == spec["primary"]:
                    np.save(RES / "v_domain.npy", np.stack(vdom).astype(np.float32))
                    M["v_domain_file"] = "v_domain.npy（Stage B6 产出物，I 轴计算的直接输入）"
            else:
                p = RES / "v_hazard_dd_vision_mean.npz"
                if p.exists():
                    z = np.load(p)
                    ang["v_hazard_dd"] = [float(abs(np.dot(vdom[l], z[f"L{l}"]))) for l in range(spec["n_layers"])]
                    if pool == spec["primary"]:
                        np.savez(RES / "v_domain_dd.npz", **{f"L{l}": vdom[l] for l in range(spec["n_layers"])})
            M["pools"][pool]["interference_abs_cos"] = ang
            for nm, a in ang.items():
                print(f"    |θ_L| vs {nm}: " + " ".join(f"{x:.3f}" for x in a))

        # 行为端域敏感度(模型内无量纲,协议 §4 规则 1/2)
        d = np.load(f, allow_pickle=True); meta = json.loads(str(d["meta"]))
        cs = {(m["scene"], m["source"], m["sec"]): m.get("commanded_speed") for m in meta}
        rel, scn = [], []
        for (sc, src, t), v in cs.items():
            if src != "real" or v is None:
                continue
            s = cs.get((sc, "sim", t))
            if s is None:
                continue
            denom = 0.5 * (abs(v) + abs(s)) + 1e-6
            rel.append(abs(v - s) / denom); scn.append(sc)
        if rel:
            M["behavioral_domain_sensitivity"] = boot_scene(rel, scn)
            print(f"[T-I/{mkey}] 行为端域敏感度 |Δv_cmd|/mean = "
                  f"{M['behavioral_domain_sensitivity']['mean']:.4f} "
                  f"{np.round(M['behavioral_domain_sensitivity']['ci95'],4).tolist()}")
        OUT["models"][mkey] = M

    # I_m 汇总(在各自概念峰层上)
    peaks = {}
    for nm, fn, key in (("simlingo", "b1_v_hazard_clean.json", "frozen_direction"),
                        ("dd", "g_positive_calibration_diffusiondrive.json", None)):
        p = RES / fn
        if not p.exists():
            continue
        j = json.load(open(p))
        peaks[nm] = (j["frozen_direction"]["peak_layer_prereg"] if key else
                     j["arms"][j.get("primary_pool", "vision_mean")]["frozen_direction"]["peak_layer_prereg"])
    summary = {}
    for mkey, M in OUT["models"].items():
        pool = MODELS[mkey]["primary"]
        if pool not in M["pools"]:
            continue
        Ls = peaks.get(mkey)
        if Ls is None:
            continue
        D = M["pools"][pool]["D_L"][Ls]
        ang = M["pools"][pool]["interference_abs_cos"]
        th = {k: v[Ls] for k, v in ang.items()}
        summary[mkey] = {"peak_layer": Ls, "D_at_peak": D, "I_m": 1 - D,
                         "D_ci95": M["pools"][pool]["D_L_bootstrap"][Ls]["ci95"],
                         "interference_abs_cos_at_peak": th,
                         "behavioral_domain_sensitivity": M.get("behavioral_domain_sensitivity", {}).get("mean")}
        print(f"[T-I] {mkey}: L*={Ls}  D_L*={D:.3f}  I_m={1-D:.3f}  |θ|={th}")
    OUT["summary_I_m"] = summary
    if len(summary) == 2:
        a, b = sorted(summary, key=lambda k: -summary[k]["I_m"])
        ci_a = summary[a]["D_ci95"]; ci_b = summary[b]["D_ci95"]
        overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
        OUT["ranking"] = {"order_by_I_m": [a, b], "D_ci_overlap": overlap,
                          "verdict": ("不可估：两模型 D_L* 的 scene 级 bootstrap CI 重叠，排序不稳健"
                                      if overlap else
                                      f"PASS：{a} 的域不变性显著优于 {b}（D_L* CI 不重叠）")}
        print(f"[T-I] 排序 {a} > {b}；CI 重叠={overlap} -> {OUT['ranking']['verdict']}")
    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
    print(f"[T-I] wrote {args.out}")


if __name__ == "__main__":
    main()
