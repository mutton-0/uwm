"""F 轴补测前置|构造 DiffusionDrive 的**站内上界方向** v_brake_dd（与 SimLingo 的 v_brake 同构）。

为什么必须有这一条：T1-Q 的纪律——先造一条"按构造必然存在"的行为定义轴当**站内上界标定**。
若连它都推不动行为，那是**仪器/注入机制**没有分辨力，而不是"危险表征不驱动动作"。
这正是把 F 轴的零结果分流到仪器侧还是标本侧的唯一办法。

构造（逐条对应 t1q_axes.brake_direction）：
  特征 = 逐条件的图像 token 均值激活（dd_cache 的 vision_mean，每层各自维度）；
  标签 = 该条件下 commanded_speed 对 ego 速度回归后的残差是否低于中位数（低残差 = 相对该车速刹得多）；
  **必须扣掉车速主效应**，否则提到的是"车速轴"而不是"刹车轴"。
选层：scene 级 4 折 CV 的 held-out AUC argmax。
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
N_LAYERS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="vision_mean")
    ap.add_argument("--out", default=str(RES / "v_brake_dd.npz"))
    args = ap.parse_args()

    X = {l: [] for l in range(N_LAYERS)}
    v_cmd, ego, scenes = [], [], []
    for p in sorted((W / "dd_cache").glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        m = json.loads(str(d["meta"]))
        for cond in ("clean", "ghost"):
            try:
                for l in range(N_LAYERS):
                    X[l].append(d[f"{cond}/{args.pool}/L{l}"].mean(0))
            except KeyError:
                for l in range(N_LAYERS):
                    if len(X[l]) > len(v_cmd):
                        X[l].pop()
                continue
            v_cmd.append(float(d[f"commanded_speed_{cond}"].mean()))
            frames = m[f"x_{cond}_frames"]
            ego.append(float(np.mean([f["ego_speed_mps"] for f in frames])))
            scenes.append(m["scene_name"])
    v_cmd = np.array(v_cmd); ego = np.array(ego); scenes = np.array(scenes)
    A = np.vstack([ego, np.ones_like(ego)]).T
    co, *_ = np.linalg.lstsq(A, v_cmd, rcond=None)
    resid = v_cmd - A @ co
    y = (resid < np.median(resid)).astype(int)      # 1 = 相对该车速刹得多
    print(f"[F-dd/brake] n={len(y)} 条件样本（{len(set(scenes))} scene）；"
          f"车速主效应 β={co[0]:+.4f}，解释 {1 - resid.var()/v_cmd.var():.3f} 方差（已扣除）")

    ks = sorted(set(scenes)); rng = np.random.default_rng(0)
    fold = {s: i % 4 for i, s in enumerate(rng.permutation(ks))}
    fa = np.array([fold[s] for s in scenes])
    auc_by_l = []
    for l in range(N_LAYERS):
        Z = np.stack(X[l]); pr = np.zeros(len(y))
        for f in range(4):
            tr, te = fa != f, fa == f
            mu, sd = Z[tr].mean(0, keepdims=True), Z[tr].std(0, keepdims=True) + 1e-6
            clf = LogisticRegression(max_iter=2000).fit((Z[tr] - mu) / sd, y[tr])
            pr[te] = clf.decision_function((Z[te] - mu) / sd)
        u = stats.mannwhitneyu(pr[y == 1], pr[y == 0], alternative="two-sided")
        auc_by_l.append(float(u.statistic / ((y == 1).sum() * (y == 0).sum())))
    L = int(np.argmax(auc_by_l))
    print("[F-dd/brake] 逐层 held-out AUC: " + " ".join(f"L{l}:{a:.3f}" for l, a in enumerate(auc_by_l)))
    print(f"[F-dd/brake] 选层 L*={L}（AUC {auc_by_l[L]:.3f}）")

    store = {}
    for l in range(N_LAYERS):
        Z = np.stack(X[l])
        mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
        w = LogisticRegression(max_iter=2000).fit((Z - mu) / sd, y).coef_[0] / sd[0]
        store[f"L{l}"] = (w / (np.linalg.norm(w) + 1e-8)).astype(np.float32)
    store["peak_layer"] = np.array([L])
    np.savez(args.out, **store)
    (RES / "v_brake_dd.json").write_text(json.dumps(
        {"pool": args.pool, "n_samples": int(len(y)), "n_scenes": len(ks),
         "ego_speed_beta": float(co[0]), "cv_auc_by_layer": auc_by_l, "peak_layer": L,
         "construction": "标签 = 模型自身 commanded_speed 对 ego 速度回归后的残差二分；"
                         "与 SimLingo 的 v_brake 同构，用作站内上界标定"},
        indent=2, ensure_ascii=False))
    print(f"[F-dd/brake] wrote {args.out}")


if __name__ == "__main__":
    main()
