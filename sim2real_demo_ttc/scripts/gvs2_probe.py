"""G-VS 第二步|从各候选的 token 级表征训线性探针，预测 SAM 伪 GT 的物体性掩码。

工单：docs/g_vs_f3_simplified_axes_workorder.md §1。

**主读数**：held-out mIoU（二类：background / object），scene 级 4 折 CV + scene 级 bootstrap。
**必需对照（Hewitt & Liang 2019 的 selectivity 纪律）**：
用**同架构、随机初始化**模型的 token 特征训同一个探针。若随机初始化也能拿到相近 mIoU，
说明是探针自己在"脑补"（比如仅凭 token 的空间位置就能猜出物体大致在哪），
而不是模型表征里真有这个信息。
**selectivity = mIoU(训练好的模型) − mIoU(随机初始化)**，这才是 G-VS 的净读数。

**为什么还要 position-only 地板**：token 网格本身带空间先验（物体多在画面下半部），
一个只看 (row, col) 的探针就能拿到不低的 mIoU。故除随机初始化外，
再报一条**只用 token 坐标**的探针地板。三条线一起才能说明"表征里有信息"。

探针：单层线性（无隐层），特征先做 per-dim 标准化。
**刻意不用 MLP**：探针容量越大越容易自己完成任务，selectivity 越不可信（Hewitt & Liang 的核心论点）。
"""
from __future__ import annotations

import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def miou(pred, gt):
    """二类 mIoU（background / object 的 IoU 取平均）。"""
    out = []
    for c in (0, 1):
        inter = float(((pred == c) & (gt == c)).sum())
        union = float(((pred == c) | (gt == c)).sum())
        out.append(inter / union if union > 0 else np.nan)
    return float(np.nanmean(out)), out


def fit_logreg(X, y, l2=1.0):
    """单层线性探针（逻辑回归，L2 正则，lbfgs 收敛）。

    **刻意保持单层线性、不加隐层**：探针容量越大越容易自己完成任务，
    selectivity 就越不可信 —— 这是 Hewitt & Liang 2019 的核心论点。
    类别不平衡用 `class_weight="balanced"`（object token 占比随图像变化，10%~40% 不等）。

    首版用手写全批 GD（200 步 lr 0.5），实测**未收敛**：mIoU 塌到 0.297，
    恰好等于"全预测背景"的解析值，selectivity ≈ 0 是优化失败而不是表征没信息（§GF/A49）。
    """
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0 / max(l2, 1e-6), max_iter=1000,
                             class_weight="balanced", solver="lbfgs")
    clf.fit(np.asarray(X, np.float64), np.asarray(y, np.int64))
    return clf


def predict(X, clf):
    return clf.predict(np.asarray(X, np.float64)).astype(np.int64)


def cv_probe(F, Y, scenes, folds=4, seed=0, l2=1.0):
    """scene 级 K 折：折外训、折内测，返回逐 scene 的 mIoU（供 bootstrap）。"""
    uniq = sorted(set(scenes))
    rng = np.random.default_rng(seed)
    assign = {s: int(i % folds) for i, s in enumerate(rng.permutation(uniq))}
    fold = np.array([assign[s] for s in scenes])
    per_scene = defaultdict(lambda: [0, 0, 0, 0])      # tp0,u0,tp1,u1 累加
    for k in range(folds):
        tr, te = fold != k, fold == k
        if tr.sum() < 100 or te.sum() < 50:
            continue
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
        clf = fit_logreg((F[tr] - mu) / sd, Y[tr], l2=l2)
        p = predict((F[te] - mu) / sd, clf)
        gt = Y[te].astype(np.int64)
        sc = np.array(scenes)[te]
        for s in set(sc.tolist()):
            m = sc == s
            acc = per_scene[s]
            for ci, off in ((0, 0), (1, 2)):
                acc[off] += int(((p[m] == ci) & (gt[m] == ci)).sum())
                acc[off + 1] += int(((p[m] == ci) | (gt[m] == ci)).sum())
    out = {}
    for s, (t0, u0, t1, u1) in per_scene.items():
        i0 = t0 / u0 if u0 else np.nan
        i1 = t1 / u1 if u1 else np.nan
        out[s] = float(np.nanmean([i0, i1]))
    return out


def boot_scene_dict(d, n=5000, seed=0):
    ks = sorted(d)
    if len(ks) < 5:
        return None
    v = np.array([d[k] for k in ks], float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    o = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return {"mean": float(v.mean()),
            "ci95": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
            "n_scenes": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="gvs3_extract 落盘的 npz")
    ap.add_argument("--label", required=True)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.feats, allow_pickle=True)
    F, Fr, Y, POS = d["feat"], d["feat_rand"], d["gt"], d["pos"]
    scenes = [str(s) for s in d["scene"]]
    meta = json.loads(str(d["meta"]))
    print(f"[GVS/{args.label}] token {F.shape[0]}，特征维 {F.shape[1]}，"
          f"scene {len(set(scenes))}，object 占比 {Y.mean():.3f}")

    arms = {}
    for nm, X in (("trained", F), ("random_init", Fr), ("position_only", POS.astype(np.float64))):
        if X is None or (hasattr(X, "size") and X.size == 0):
            continue
        per = cv_probe(np.asarray(X, np.float64), Y, scenes, l2=args.l2)
        arms[nm] = {"per_scene_miou": per, "bootstrap": boot_scene_dict(per)}
        b = arms[nm]["bootstrap"]
        print(f"[GVS/{args.label}] {nm:14s} mIoU = {b['mean']:.4f}  "
              f"CI {np.round(b['ci95'],4).tolist()}  ({b['n_scenes']} scene)")

    out = {"model": args.label, "task": "G-VS 二类物体性分割探针（SAM 伪 GT）",
           "primary": "held-out mIoU（scene 级 4 折 CV + scene 级 bootstrap）",
           "selectivity": "mIoU(trained) − mIoU(random_init)，Hewitt & Liang 2019 的 control task 纪律",
           "meta": meta, "n_tokens": int(F.shape[0]), "feat_dim": int(F.shape[1]),
           "object_token_frac": float(Y.mean()), "arms": arms}
    if "trained" in arms and "random_init" in arms:
        # selectivity 的 CI：逐 scene 配对差（同一 scene 在两臂上都有读数）
        a, b = arms["trained"]["per_scene_miou"], arms["random_init"]["per_scene_miou"]
        common = sorted(set(a) & set(b))
        diff = {s: a[s] - b[s] for s in common}
        out["selectivity"] = boot_scene_dict(diff)
        s = out["selectivity"]
        ci = s["ci95"]
        out["verdict"] = ("PASS：表征里确有物体性信息（selectivity 的 scene 级 CI 下界 > 0）"
                          if ci[0] > 0 else
                          ("FAIL：训练好的模型不优于随机初始化（探针在自行完成任务）"
                           if ci[1] < 0 else "不可估：selectivity 的 scene 级 CI 跨 0"))
        print(f"[GVS/{args.label}] **selectivity = {s['mean']:+.4f}**  CI {np.round(ci,4).tolist()}")
        print(f"[GVS/{args.label}] 判定：{out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[GVS/{args.label}] wrote {args.out}")


if __name__ == "__main__":
    main()
