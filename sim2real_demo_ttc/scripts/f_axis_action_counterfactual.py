"""F 轴主验证|行动层反事实测试（Action-Level Counterfactual Test），两个模型同一套刺激。

依据：选型协议 §3.5「F 的验证：行动层反事实测试 —— 取代 Bench2Drive 作为主验证」。
    与 G 的测试完全对称，但读出对象从"内部表征"换成"最终动作"：
      ① 操纵**因果特征**（危险目标有无）= A 类事件的 ghost 帧 vs clean 帧；
      ② 操纵**几何混淆特征**（大小/居中度，控制不变）= D2a 几何平衡负例的 ghost vs clean。
    判据：① 应显著改变动作，② 不应（或应远小于 ①）。
    若动作对 ② 也有显著响应 ⇒ 动作层面存在 causal confusion，即便 G 读数干净，F 也不能打高分。

零 GPU：两个模型的逐条件规划速度都已在缓存里
  SimLingo    variants/n1_d2/cache/*.h5      pred_speed（= control_pid 的 desired_speed）
  DiffusionDrive variants/n1_d2/dd_cache/*.npz commanded_speed（= ‖traj[0]‖ / PRED_DT）

主读数（预注册）：**b-AUC = AUC( b(A) vs b(D2a) )**，b = v_plan(clean) − v_plan(ghost)（正 = 见到目标后减速）。
    b-AUC > 0.5 表示模型对真危险的减速强于对几何匹配的无害物。scene 级 bootstrap。
并列：② 的绝对效应量（D2a 的 b 是否显著 ≠ 0）——这是 causal confusion 的直接症状。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
VRU = ("human.", "vehicle.bicycle", "vehicle.motorcycle", "walker.")


def auc(a, b):
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def boot_auc(pa, sa, pb, sb, n=2000, seed=0):
    """scene 级 bootstrap：两组共用同一批重采样 scene，保留配对结构。"""
    ia, ib = defaultdict(list), defaultdict(list)
    for v, s in zip(pa, sa):
        ia[s].append(v)
    for v, s in zip(pb, sb):
        ib[s].append(v)
    ks = sorted(set(ia) | set(ib)); rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        A, B = [], []
        for i in rng.integers(0, len(ks), len(ks)):
            k = ks[i]; A += ia.get(k, []); B += ib.get(k, [])
        if min(len(A), len(B)) < 5:
            continue
        out.append(auc(np.array(A), np.array(B))[0])
    o = np.array(out)
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))], float((o <= 0.5).mean())


def boot_mean(v, s, n=2000, seed=0):
    d = defaultdict(list)
    for x, k in zip(v, s):
        d[k].append(x)
    ks = sorted(d); rng = np.random.default_rng(seed)
    o = np.array([np.mean([x for i in rng.integers(0, len(ks), len(ks)) for x in d[ks[i]]]) for _ in range(n)])
    return float(np.mean(v)), [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def load_simlingo():
    import h5py
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}
    out = defaultdict(list)
    for p in sorted((W / "cache").glob("*.h5")):
        if p.stem not in evmap:
            continue
        try:
            with h5py.File(p, "r") as f:
                if "pred_speed" not in f["clean"]:
                    continue
                b = float(f["clean"]["pred_speed"][:].mean() - f["ghost"]["pred_speed"][:].mean())
        except Exception:
            continue
        m = evmap[p.stem]
        out[m["event_type"]].append({"b": b, "scene": m["scene_name"], "eid": p.stem,
                                     "cls": m.get("object_class", "")})
    return out, evmap


def load_npz(cache_dir, kind="navsim"):
    """通用 npz 缓存读取。

    kind=navsim   : commanded_speed_{clean,ghost}（DiffusionDrive / LTF）
    kind=alpamayo : {clean,ghost}/v_plan（Alpamayo-R1，同口径的规划目标速度）
    """
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(W / "mining" / "events_all.jsonl")}
    out = defaultdict(list)
    for p in sorted(Path(cache_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        try:
            if kind == "navsim":
                b = float(d["commanded_speed_clean"].mean() - d["commanded_speed_ghost"].mean())
            else:
                b = float(d["clean/v_plan"][0] - d["ghost/v_plan"][0])
        except KeyError:
            continue
        if not np.isfinite(b):
            continue
        out[m["event_type"]].append({"b": b, "scene": m["scene_name"], "eid": m["event_id"],
                                     "cls": m.get("object_class", "")})
    return out, evmap


def load_dd():
    return load_npz(W / "dd_cache", "navsim")


def load_ltf():
    return load_npz(W / "ltf_cache", "navsim")


def load_ddv2():
    return load_npz(W / "ddv2_cache", "navsim")


def load_alpa():
    return load_npz(W / "alpa_cache", "alpamayo")


def load_autovla():
    return load_npz(W / "autovla_cache", "alpamayo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "f_axis_action_counterfactual.json"))
    args = ap.parse_args()

    matched = {t: set((W / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a", "D2b", "D2c", "D2cV")
               if (W / "mining" / f"matched_{t}.txt").exists()}
    OUT = {"design": "行动层反事实测试（协议 §3.5）：① 危险有无 = A 的 ghost−clean；"
                     "② 几何混淆 = D2a 的 ghost−clean。读出对象为最终规划动作，不是表征。",
           "b_definition": "b = v_plan(clean) − v_plan(ghost)，正 = 目标出现后减速",
           "models": {}}

    MODELS = [("SimLingo", load_simlingo), ("DiffusionDrive", load_dd),
              ("LTF", load_ltf), ("DiffusionDriveV2", load_ddv2), ("Alpamayo-R1", load_alpa),
              ("AutoVLA", load_autovla)]
    for name, loader in MODELS:
        try:
            _probe = loader()
        except Exception as _e:                                        # noqa: BLE001
            print(f"\n===== {name} =====  缓存不可用，跳过（{_e}）"); continue
        if not any(len(v) >= 20 for v in _probe[0].values()):
            print(f"\n===== {name} =====  缓存为空或样本不足，跳过"); continue
        by, evmap = _probe
        # D2cV 子集（同类别 VRU、同几何，只差速度）——最硬的对照
        d2cv = [e for e in by.get("D2c", []) if e["eid"] in matched.get("D2cV", set())
                and str(e["cls"]).startswith(VRU)]
        grp = {"A": by.get("A", []),
               "D2a": [e for e in by.get("D2a", []) if e["eid"] in matched.get("D2a", set())] or by.get("D2a", []),
               "D2cV": d2cv}
        M = {"group_sizes": {k: len(v) for k, v in grp.items()}}
        print(f"\n===== {name} =====  " + "  ".join(f"{k}={len(v)}" for k, v in grp.items()))

        for k, v in grp.items():
            if len(v) < 20:
                continue
            m, ci = boot_mean([e["b"] for e in v], [e["scene"] for e in v])
            M[f"b_mean_{k}"] = {"mean": m, "ci95": ci, "n": len(v),
                                "significant": bool(ci[0] > 0 or ci[1] < 0)}
            print(f"  b({k:4s}) = {m:+.4f} m/s  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  n={len(v)}  "
                  f"{'显著≠0' if (ci[0] > 0 or ci[1] < 0) else '与 0 不可区分'}")

        for neg in ("D2a", "D2cV"):
            if len(grp[neg]) < 20 or len(grp["A"]) < 20:
                continue
            pa = [e["b"] for e in grp["A"]]; sa = [e["scene"] for e in grp["A"]]
            pb = [e["b"] for e in grp[neg]]; sb = [e["scene"] for e in grp[neg]]
            a, p = auc(np.array(pa), np.array(pb))
            ci, p_le = boot_auc(pa, sa, pb, sb)
            M[f"b_auc_A_vs_{neg}"] = {"auc": a, "p_mwu": p, "ci95_scene_bootstrap": ci,
                                      "p_boot_le_0.5": p_le, "n_pos": len(pa), "n_neg": len(pb)}
            print(f"  **b-AUC(A vs {neg:4s}) = {a:.4f}**  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"MWU p={p:.3g}  bootstrap P(AUC≤0.5)={p_le:.3f}")

        r = M.get("b_auc_A_vs_D2a")
        conf = M.get("b_mean_D2a", {})
        if r:
            if r["ci95_scene_bootstrap"][0] > 0.5:
                v = "PASS：动作对真危险的响应显著强于对几何匹配无害物的响应"
            elif r["ci95_scene_bootstrap"][1] < 0.5:
                v = "FAIL：动作对几何匹配无害物的响应反而更强"
            else:
                v = "不可估：b-AUC 的 scene 级 CI 跨 0.5"
            M["verdict_action_counterfactual"] = v
            M["confound_response_significant"] = bool(conf.get("significant"))
            print(f"  判定：{v}")
            print(f"  混淆特征（②）本身是否显著驱动动作：{'是 ⇒ 动作层存在 causal confusion 症状' if conf.get('significant') else '否'}")
        OUT["models"][name] = M

    Path(args.out).write_text(json.dumps(OUT, indent=2, ensure_ascii=False))
    print(f"\n[F-act] wrote {args.out}")


if __name__ == "__main__":
    main()
