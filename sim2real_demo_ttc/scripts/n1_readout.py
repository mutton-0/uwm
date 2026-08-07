"""N1 读数：几何匹配后的 R① 类别特异性 + D2c 证伪检验。

设计（guide §12.4 + q_audit §Q3/§Q6）：
  - 方向 v_hazard 由 S_dir 上的 **A（VRU 突现）vs D2a（静态无害物）** 有监督判别提出；
    两类已在 (log 面积, 离心率) 上匹配 => 方向不能靠"大而居中"取胜。
  - 峰层 L* 在 S_sel 上用 **AUC** 选（ρ_TTC 对单帧模型无效，见 §Q6）。
  - S_test 上对四个负类分别报 AUC：
      D2a  类别对照   —— 主读数（R① 类别特异性）
      D2b  上下文对照 —— R② 上下文调制
      D2c  证伪控制   —— 无害性只由速度定义，单帧不可见 => **预测 ≈0.5**
      D    旧口径     —— 对照，看几何匹配后还剩多少

定位：本轮在 trainval 上做，trainval 已被此前分析消费，因此 N1 结果是
**口径验证（含证伪控制是否守住 0.5）**，不是确证。确证在 AV2（N3）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache, subset  # noqa: E402


def supervised_direction(pos, neg, seed=0):
    """逐层训练线性判别器区分两组 δ；返回 [L,C] 单位方向与各层训练 AUC。"""
    from sklearn.linear_model import LogisticRegression
    X = np.stack([e["h_ghost"] - e["h_clean"] for e in pos + neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    n, L, C = X.shape
    v = np.zeros((L, C), dtype=np.float32)
    for l in range(L):
        Z = X[:, l, :]
        mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit((Z - mu) / sd, y)
        w = clf.coef_[0] / sd[0]
        v[l] = w / (np.linalg.norm(w) + 1e-8)
    return v


def pca_direction(pos, neg, seed=0):
    """主成分式方向(RepE LAT 的 PCA 变体):逐层对 δ 取 PC1，符号按"正例投影更大"定。

    与判别式方向共用同一配对设计——若配对足够干净，两法应收敛；
    两法分歧则说明信号被非主成分方向携带（或根本没有信号）。
    """
    X = np.stack([e["h_ghost"] - e["h_clean"] for e in pos + neg])
    npos = len(pos)
    n, L, C = X.shape
    v = np.zeros((L, C), dtype=np.float32)
    for l in range(L):
        Z = X[:, l, :]
        Z = Z - Z.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        w = Vt[0]
        if (Z[:npos] @ w).mean() < (Z[npos:] @ w).mean():
            w = -w
        v[l] = w / (np.linalg.norm(w) + 1e-8)
    return v


def permuted_direction(pos, neg, seed=0):
    """标签置换方向:把 pos/neg 标签打乱后走同一套判别式提取。

    用来回答"主读数那个数是不是纯噪声"——置换零分布的上分位就是这条管线
    (含逐层拟合 + 后续选层)在无真实信号时能造出的 AUC 上限。
    """
    rng = np.random.default_rng(seed)
    allev = pos + neg
    idx = rng.permutation(len(allev))
    k = len(pos)
    return supervised_direction([allev[i] for i in idx[:k]], [allev[i] for i in idx[k:]], seed=seed)


def proj(events, v, layer):
    if not events:
        return np.zeros(0)
    return np.stack([e["h_ghost"][layer] - e["h_clean"][layer] for e in events]) @ v[layer]


def auc(a, b):
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u.statistic / (len(a) * len(b))), float(u.pvalue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="vision_mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))

    # 只保留匹配子集里的事件，并按类型归组
    matched = {}
    for t in ("D", "D2a", "D2b", "D2c"):
        f = work / "mining" / f"matched_{t}.txt"
        if f.exists():
            matched[t] = set(f.read_text().split())

    by_type = defaultdict(list)
    for eid, e in items.items():
        et = evmap[eid]["event_type"]
        e["etype"] = et
        by_type[et].append(e)

    # scene 级三分：S_dir 提方向 / S_sel 选层 / S_test 报数
    scenes = sorted({e["scene"] for v in by_type.values() for e in v})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(scenes))
    n_dir = int(len(scenes) * 0.5); n_sel = int(len(scenes) * 0.25)
    part = {"dir": set(), "sel": set(), "test": set()}
    for i, k in enumerate(perm):
        part["dir" if i < n_dir else ("sel" if i < n_dir + n_sel else "test")].add(scenes[k])

    def grp(t, p):
        return [e for e in by_type[t] if e["scene"] in part[p]
                and (t not in matched or e["meta"]["event_id"] in matched[t])]

    A = {p: grp("A", p) for p in part}
    print(f"[N1] scene 三分: dir {len(part['dir'])} / sel {len(part['sel'])} / test {len(part['test'])}")
    print(f"[N1] A 类: dir {len(A['dir'])} / sel {len(A['sel'])} / test {len(A['test'])}")
    for t in ("D2a", "D2b", "D2c", "D"):
        print(f"[N1] {t:4s}: dir {len(grp(t,'dir')):4d} / sel {len(grp(t,'sel')):4d} / test {len(grp(t,'test')):4d}")

    # 方向：S_dir 上 A vs D2a（几何已匹配）
    v = supervised_direction(A["dir"], grp("D2a", "dir"), seed=args.seed)

    # 峰层：S_sel 上 AUC(A vs D2a) 最大（ρ_TTC 对单帧无效）
    sel_p, sel_n = A["sel"], grp("D2a", "sel")
    aucs = [auc(proj(sel_p, v, l), proj(sel_n, v, l))[0] for l in range(v.shape[0])]
    aucs = np.array(aucs)
    peak = int(np.nanargmax(aucs))
    print(f"[N1] 峰层 L*={peak}（S_sel 上 AUC={aucs[peak]:.3f}；选层准则=AUC，非 ρ_TTC）")

    # S_test 报数
    test_p = A["test"]
    pp = proj(test_p, v, peak)
    print(f"\n{'负类':>6} {'语义':>22} {'n(正/负)':>12} {'AUC':>8} {'p':>10}   预期")
    print("-" * 82)
    rows = {}
    labels = {"D2a": ("类别对照(静态物)", "主读数 R①，>0.5 才算类别特异"),
              "D2b": ("上下文对照(走廊外)", "R② 上下文调制"),
              "D2c": ("证伪控制(仅速度无害)", "**预测 ≈0.5**"),
              "D":   ("旧口径负例", "对照")}
    for t in ("D2a", "D2b", "D2c", "D"):
        nn = grp(t, "test")
        a, p = auc(pp, proj(nn, v, peak))
        rows[t] = {"n_pos": len(test_p), "n_neg": len(nn), "auc": a, "p": p}
        lab, exp = labels[t]
        print(f"{t:>6} {lab:>22} {f'{len(test_p)}/{len(nn)}':>12} {a:8.3f} {p:10.3g}   {exp}")

    d2c = rows.get("D2c", {})
    verdict = "未知"
    if np.isfinite(d2c.get("auc", np.nan)):
        if d2c["p"] < 0.05 and d2c["auc"] > 0.5:
            verdict = "❌ 泄漏警报：D2c 显著 >0.5，但其无害性单帧不可见 —— 管线或匹配有泄漏"
        elif d2c["p"] >= 0.05:
            verdict = "✅ 证伪控制守住：D2c ≈0.5，符合单帧物理预期"
        else:
            verdict = "⚠️ D2c 显著 <0.5，方向被负类主导，需查"
    print(f"\n[N1] D2c 证伪检验：{verdict}")

    out = {"pool_mode": args.pool_mode, "peak_layer": peak, "seed": args.seed,
           "n_scenes": {k: len(v_) for k, v_ in part.items()},
           "sel_auc_by_layer": aucs.tolist(), "readouts": rows,
           "d2c_verdict": verdict,
           "scope": "trainval 已被消费 => 本结果是口径验证（含证伪控制），不是确证；确证在 AV2(N3)"}
    (work / "results" / f"n1_readout_{args.pool_mode}{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[N1] wrote results/n1_readout_{args.pool_mode}{args.tag}.json")


if __name__ == "__main__":
    main()
