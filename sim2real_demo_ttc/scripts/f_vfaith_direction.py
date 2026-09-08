"""F 轴第二部分：从 clean/ghost 特征差里提出 v_faith 方向，比较两域的夹角。

## 域怎么分（P-1）
**左舵(LHD) = benchmark，右舵(RHD) = deployment。** 不再按 nuScenes / NAVSIM 分。
两个语料先合并，再按舵位切 —— 所以 benchmark 侧同时含 NAVSIM(维加斯/匹兹堡)
与 nuScenes(波士顿)，deployment 侧同时含 nuScenes(新加坡) 与 NAVSIM 的少量右舵 log。
这意味着**语料与域不再共线**，跨域夹角量的是舵位差异，不是渲染/采集差异。

## 方向怎么提
逐事件差 δ_i(L) = h_ghost,i(L) − h_clean,i(L)（符号：**恢复危险物**的方向）。

稀疏化**不用 softmax** —— softmax 把每个分量映射成正概率，没有任何一个会变成 0，
且温度是个可事后调的自由度。改用有统计含义的判据：逐坐标做 scene 级配对
bootstrap，保留 95% CI 不含 0 的坐标，再用 Benjamini–Hochberg 控制 FDR（q=0.05）。
"稀疏"于是是数据说的，保留几个坐标本身也是个可报的读数。

    v(L) = normalize( mean_i δ_i(L) 限制在保留坐标上，其余置 0 )

## 夹角为什么必须配零假设
d 维空间里两个随机方向的夹角以极高概率就是 90°，方差极小（d=64 时 sd≈7°，
d=512 时≈2.5°）。所以裸报"两语料夹角 62°"无法解释。故同时给两个参照：

  θ_half   同一语料内部随机劈两半各算方向的夹角 —— 同分布、同数据量下
           能达到的最好一致性，即**噪声地板**
  θ_rand   两个随机方向的夹角分布 —— 完全无关时的样子

主读数是对齐度  A(L) = (θ_rand − θ_cross) / (θ_rand − θ_half) ∈ (-inf, 1]
  A=1  跨语料和同语料内部一样一致（方向完全迁移）
  A=0  跨语料和随机方向没区别（方向完全不迁移）
分母是"这批数据能达到的上限"，所以 A 已经把数据量的影响除掉了。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
_MAP = json.load(open(RES / "driveside_map.json"))


def side_of(scene, corpus):
    if corpus == "nuscenes":
        return _MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):
        s = _MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s
    return None
NAME = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
        "simlingo": "SimLingo"}
Q_FDR = 0.05
B_BOOT = 2000
B_HALF = 200
B_NULL = 4000        # 零假设抽样数：BH 需要细的 p 分辨率（1/B），200 太粗


def scene_boot_ci(D, scenes, B, rng):
    """逐坐标的 scene 级 bootstrap 95% CI。D: [n, d] 的逐事件差。"""
    uq = np.unique(scenes)
    ix = [np.where(scenes == s)[0] for s in uq]
    out = np.empty((B, D.shape[1]), np.float32)
    for b in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])
        out[b] = D[sel].mean(0)
    return np.percentile(out, 2.5, axis=0), np.percentile(out, 97.5, axis=0), out


def bh_support(lo, hi, boots, q=Q_FDR):
    """BH-FDR。逐坐标的双侧 bootstrap p 值 = 2*min(P[b<=0], P[b>=0])。"""
    p = 2 * np.minimum((boots <= 0).mean(0), (boots >= 0).mean(0))
    p = np.clip(p, 1.0 / boots.shape[0], 1.0)
    m = len(p); order = np.argsort(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    keep = np.zeros(m, bool)
    keep[order[:k]] = True
    return keep, p


def direction(D, scenes, rng, sparse=True):
    """返回 (单位方向, 保留坐标数, 总维数)。"""
    if len(D) < 3:
        return None, 0, D.shape[1]
    mu = D.mean(0)
    keep = np.ones(D.shape[1], bool)
    if sparse:
        lo, hi, boots = scene_boot_ci(D, scenes, B_BOOT, rng)
        keep, _ = bh_support(lo, hi, boots)
        if keep.sum() < 2:                      # 一个坐标都留不下 -> 退回稠密
            return None, int(keep.sum()), D.shape[1]
    v = np.where(keep, mu, 0.0)
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else None), int(keep.sum()), D.shape[1]


def ang(a, b):
    c = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return float(np.degrees(np.arccos(c)))


def split_half_angle(D, scenes, rng, B=B_HALF):
    """同语料内部劈两半的夹角中位数 —— 噪声地板。按 scene 劈，不按 event。"""
    uq = np.unique(scenes)
    if len(uq) < 6:
        return None
    out = []
    for _ in range(B):
        perm = rng.permutation(uq)
        h = len(uq) // 2
        A = np.isin(scenes, perm[:h]); Bm = np.isin(scenes, perm[h:])
        # 半样本上不做稀疏化：样本减半会让 FDR 支撑集塌掉，
        # 地板与跨语料量必须用同一口径，故两边都用稠密均值方向。
        va, vb = D[A].mean(0), D[Bm].mean(0)
        if np.linalg.norm(va) > 1e-12 and np.linalg.norm(vb) > 1e-12:
            out.append(ang(va, vb))
    return float(np.median(out)) if out else None


def matched_within_angle(D, scenes, n_small, rng, B=B_NULL):
    """**同域内、样本量配平的夹角** —— 本文件最重要的对照。

    从 benchmark 里抽 n_small 个 scene 当"小样本侧"，剩下的当"大样本侧"，
    两边各算稠密均值方向再取夹角，中位数即为：
    「**同一个域**里，一侧只有 n_small 个场景时，两个方向天然会差多少」。

    为什么必须有它：deployment 侧只有 8 个场景，它自己的 split-half 地板
    （4 vs 4）会大到 20–70°，把 θ_half 的均值拖上去，于是对齐度
    A=(θ_rand−θ_cross)/(θ_rand−θ_half) 被抬到 ≈1 —— 那不是"方向迁移得好"，
    只是"跨域夹角和 8 个样本的噪声一样大"。θ_cross ≲ 本量 ⇒ 没有任何
    证据表明两域方向不同；θ_cross ≫ 本量才是真的域差异。
    """
    uq = np.unique(scenes)
    if len(uq) < n_small + 3:
        return None
    out = []
    for _ in range(B):
        perm = rng.permutation(uq)
        A_ = np.isin(scenes, perm[:n_small]); B_ = np.isin(scenes, perm[n_small:])
        va, vb = D[A_].mean(0), D[B_].mean(0)
        if np.linalg.norm(va) > 1e-12 and np.linalg.norm(vb) > 1e-12:
            out.append(ang(va, vb))
    return (float(np.median(out)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)),
            np.asarray(out)) if out else None


def part_ratio(D):
    """有效维度 = (Σλ)²/Σλ²。PR≈1 ⇒ 该层所有扰动挤在同一根轴上，夹角只能是
    0° 或 180°，**不携带信息**。实测 TransFuser 系 L0/L1 的 PR 是 1.0–1.9。"""
    X = np.asarray(D, float); X = X - X.mean(0)
    if len(X) < 3:
        return float("nan")
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    t = lam.sum()
    return float(t * t / max((lam ** 2).sum(), 1e-30))


def pick_layers(Ds, scenes, rng, k, pr_min=3.0, floor_max=30.0):
    """选层：**三重门 + SNR**，全部只用 benchmark 侧，不看跨域比较。

      1. 有效维度 PR ≥ pr_min      —— 角度必须携带信息
      2. split-half 噪声地板 ≤ floor_max —— 方向必须估得出来
      3. 存活层里按 SNR 取 top-k

    裸 top-K 会选中最前几层（SNR 高但 PR≈1，角度无意义）和最后一层
    （地板 33–39°，纯噪声），故必须先过前两道门。
    """
    nL = len(Ds)
    snr = [layer_snr(Ds[l], scenes, rng) for l in range(nL)]
    pr = [part_ratio(Ds[l]) for l in range(nL)]
    fl = [split_half_angle(Ds[l], scenes, rng) for l in range(nL)]
    ok = [l for l in range(nL)
          if np.isfinite(pr[l]) and pr[l] >= pr_min
          and (fl[l] is None or fl[l] <= floor_max)]
    if len(ok) < k:                       # 门太严 -> 放宽到只要 PR
        ok = [l for l in range(nL) if np.isfinite(pr[l]) and pr[l] >= pr_min] or list(range(nL))
    ok = sorted(ok, key=lambda l: -snr[l])[:k]
    return sorted(ok), snr, pr, fl


def layer_snr(D, scenes, rng, B=400):
    """层的信噪比 = ‖均值方向‖ / 均值的 scene 级 bootstrap 标准误（各分量取模）。

    **只用 benchmark 侧算**，不看 deployment、不看跨域夹角 —— 选层必须与结果无关，
    否则就是 C 轴那批作废文件犯的同一个错（按 |gap| 预选样本）。
    """
    uq = np.unique(scenes); ix = [np.where(scenes == s)[0] for s in uq]
    mu = D.mean(0)
    bs = np.empty((B, D.shape[1]), np.float32)
    for b in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])
        bs[b] = D[sel].mean(0)
    se = bs.std(0)
    return float(np.linalg.norm(mu) / max(np.linalg.norm(se), 1e-12))


def pooled_dir(Ds, idx):
    """把选中的层压成**一个**向量：每层的均值方向各自单位化后拼接。

    单位化是必须的 —— TransFuser 各层维度 64/128/256/512、尺度差一个量级，
    不归一化就是最后两层说了算，等于没有池化。等权拼接 ⇒ 每层贡献相同。
    """
    vs = []
    for l in idx:
        v = Ds[l].mean(0)
        n = np.linalg.norm(v)
        vs.append(v / n if n > 1e-12 else v)
    return np.concatenate(vs) if vs else None


def pooled_matched_null(Ds, scenes, idx, n_small, rng, B=B_NULL):
    """池化向量的样本量配平零假设：同域内劈成 n_small vs 其余，各自池化后取夹角。"""
    uq = np.unique(scenes)
    if len(uq) < n_small + 3:
        return None
    out = []
    for _ in range(B):
        perm = rng.permutation(uq); n_small_ = n_small
        A_ = np.isin(scenes, perm[:n_small_]); B_ = np.isin(scenes, perm[n_small_:])
        va = pooled_dir([D[A_] for D in Ds], idx)
        vb = pooled_dir([D[B_] for D in Ds], idx)
        a = ang(va, vb) if va is not None and vb is not None else None
        if a is not None:
            out.append(a)
    o = np.asarray(out)
    return (float(np.median(o)), float(np.percentile(o, 2.5)),
            float(np.percentile(o, 97.5)), o) if len(o) else None


def rand_angle(d, rng, B=2000):
    a = rng.normal(size=(B, d)); b = rng.normal(size=(B, d))
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())


def load_by_side(scen, m):
    """两语料合并后按舵位切（P-1）。返回 {"LHD": {...}, "RHD": {...}}。"""
    parts = []
    for corpus in ("nuscenes", "navsim"):
        p = RES / f"vfaith_acts_{corpus}_{scen}_{m}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        nL = sum(1 for k in z.files if k.startswith("h_clean__L"))
        sc = np.array([str(x) for x in z["scene"]])
        sd = np.array([side_of(x, corpus) or "?" for x in sc])
        parts.append({"corpus": corpus, "scene": sc, "side": sd, "nL": nL,
                      "d": [z[f"h_ghost__L{l}"] - z[f"h_clean__L{l}"] for l in range(nL)]})
    if not parts:
        return None
    nL = min(x["nL"] for x in parts)
    out = {}
    for side in ("LHD", "RHD"):
        msk = [x["side"] == side for x in parts]
        if not any(mm.any() for mm in msk):
            out[side] = None
            continue
        # scene 键要跨语料唯一，否则 scene 级 bootstrap 会把两个语料的同名场景并成一个
        out[side] = {
            "scene": np.concatenate([np.array([f"{x['corpus']}|{v}" for v in x["scene"][mm]],
                                              dtype=object)
                                     for x, mm in zip(parts, msk)]),
            "corpus": np.concatenate([np.repeat(x["corpus"], mm.sum())
                                      for x, mm in zip(parts, msk)]),
            "nL": nL,
            "d": [np.concatenate([x["d"][l][mm] for x, mm in zip(parts, msk)])
                  for l in range(nL)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="lead", choices=["ghost", "lead"])
    ap.add_argument("--models", default="dd,ltf,ddv2,simlingo")
    ap.add_argument("--corpus-control", action="store_true",
                    help="**混杂检验**：在 benchmark(LHD) 内部比 nuScenes 与 NAVSIM 的方向。"
                         "P-1 切完后舵位与语料仍近共线（NAVSIM 94%% 左舵、右舵主体是新加坡），"
                         "所以'有域差'可能是语料差。若同为左舵的两语料也差，就是语料效应。")
    ap.add_argument("--pooled", action="store_true",
                    help="RepE 式：按 benchmark 侧信噪比选 top-K 层，各层单位化后拼成"
                         "**一个**向量，于是每个模型只有一个 θ")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--pr-min", type=float, default=3.0,
                    help="有效维度门：低于此的层角度不携带信息")
    ap.add_argument("--floor-max", type=float, default=30.0,
                    help="噪声地板门：split-half 夹角高于此的层方向估不准")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    if A.pooled and A.corpus_control:
        # 同一套池化 + 同一个零假设，但比较的是**同为左舵**的 nuScenes vs NAVSIM
        print(f"\n场景 = {A.scen}   **池化混杂检验**：同为左舵，nuScenes vs NAVSIM")
        print(f"{'候选':<18}{'选中层':>18}{'n_nusc':>8}{'n_navsim':>9}"
              f"{'θ(跨语料)':>12}{'同域配平零假设[95%CI]':>26}{'p':>8}")
        print("-" * 100)
        cc = []
        for m in A.models.split(","):
            bs = load_by_side(A.scen, m)
            if bs is None or bs["LHD"] is None:
                continue
            b = bs["LHD"]
            nu = b["corpus"] == "nuscenes"; nv = b["corpus"] == "navsim"
            if nu.sum() < 3 or nv.sum() < 6:
                print(f"{NAME.get(m,m):<18}  左舵侧 nuScenes 仅 {int(nu.sum())} 事件 —— 无法检验")
                continue
            rng = np.random.default_rng(7)
            nL = b["nL"]
            # **选层标准与跨舵位完全一致**：仍用左舵全体的 SNR，不看语料标签
            snr = [layer_snr(b["d"][l], b["scene"], rng) for l in range(nL)]
            idx = sorted(np.argsort(snr)[::-1][:A.topk].tolist())
            vu = pooled_dir([D[nu] for D in b["d"]], idx)
            vv = pooled_dir([D[nv] for D in b["d"]], idx)
            th = ang(vu, vv)
            nsc = int(len(np.unique(b["scene"][nu])))
            mw = pooled_matched_null([D[nv] for D in b["d"]], b["scene"][nv], idx, nsc, rng)
            pv = float((mw[3] >= th).mean()) if mw else None
            cc.append({"model": m, "layers": idx, "theta": th, "p": pv,
                       "n_nusc": int(nu.sum()), "n_navsim": int(nv.sum()),
                       "null_med": (mw[0] if mw else None),
                       "null_ci": (list(mw[1:3]) if mw else None)})
            ci = f"{mw[0]:5.1f} [{mw[1]:.0f},{mw[2]:.0f}]".rjust(26) if mw else "--".rjust(26)
            print(f"{NAME.get(m,m):<18}{','.join('L%d'%l for l in idx):>18}"
                  f"{int(nu.sum()):>8}{int(nv.sum()):>9}{th:>11.1f}°{ci}"
                  f"{(f'{pv:.3f}' if pv is not None else '--'):>8}")
        (Path(A.out) if A.out else RES / f"vfaith_pooled_corpus_{A.scen}.json").write_text(
            json.dumps(cc, ensure_ascii=False, indent=1))
        return

    if A.pooled:
        res = []
        print(f"\n场景 = {A.scen}   **池化**：按 benchmark 侧 SNR 选 top-{A.topk} 层，"
              f"各层单位化后拼接 ⇒ 每个模型一个 θ")
        print(f"{'候选':<18}{'总层':>5}{'选中层(SNR)':>26}{'θ(右舵vs左舵)':>15}"
              f"{'同域配平零假设[95%CI]':>26}{'p':>8}{'判定':>10}")
        print("-" * 110)
        for m in A.models.split(","):
            bs = load_by_side(A.scen, m)
            if bs is None or bs["LHD"] is None or bs["RHD"] is None:
                continue
            b, d = bs["LHD"], bs["RHD"]
            nL = min(b["nL"], d["nL"])
            rng = np.random.default_rng(7)
            idx, snr, pr, fl = pick_layers(b["d"][:nL], b["scene"], rng, A.topk,
                                           pr_min=A.pr_min, floor_max=A.floor_max)
            vb = pooled_dir(b["d"], idx); vd = pooled_dir(d["d"], idx)
            th = ang(vb, vd)
            nsc = int(len(np.unique(d["scene"])))
            mw = pooled_matched_null(b["d"], b["scene"], idx, nsc, rng)
            pv = float((mw[3] >= th).mean()) if mw else None
            res.append({"model": m, "n_layers": nL, "topk": A.topk,
                        "layers": idx, "snr": [round(x, 2) for x in snr],
                        "part_ratio": [round(x, 2) for x in pr],
                        "theta_half_per_layer": [None if x is None else round(x, 1) for x in fl],
                        "gate": {"pr_min": A.pr_min, "floor_max": A.floor_max},
                        "dim_pooled": int(len(vb)), "theta": th,
                        "n_bench": int(len(b["scene"])), "n_dep": int(len(d["scene"])),
                        "null_med": (mw[0] if mw else None),
                        "null_ci": (list(mw[1:3]) if mw else None), "p": pv})
            ls = ",".join(f"L{l}" for l in idx)
            ci = f"{mw[0]:5.1f} [{mw[1]:.0f},{mw[2]:.0f}]".rjust(26) if mw else "--".rjust(26)
            print(f"{NAME.get(m,m):<18}{nL:>5}{ls:>26}{th:>14.1f}°{ci}"
                  f"{(f'{pv:.3f}' if pv is not None else '--'):>8}", end="")
            print(f"{'':>10}")
        ps = [(i, r["p"]) for i, r in enumerate(res) if r["p"] is not None]
        ps.sort(key=lambda t: t[1]); mt = max(len(ps), 1); kk = 0
        for j, (_, pv) in enumerate(ps, 1):
            if pv <= Q_FDR * j / mt:
                kk = j
        sig = set(i for i, _ in ps[:kk])
        print()
        for i, r in enumerate(res):
            r["domain_diff_bh"] = bool(i in sig)
            print(f"  {NAME.get(r['model'],r['model']):<18} θ={r['theta']:5.1f}°  "
                  f"零假设中位 {r['null_med']:.1f}°  p={r['p']:.3f}  "
                  f"{'**有舵位差**' if r['domain_diff_bh'] else '不可区分（θ 落在同域抽样噪声内）'}")
        out = Path(A.out) if A.out else RES / f"vfaith_pooled_{A.scen}.json"
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"\n-> {out}")
        return

    rows = []
    if A.corpus_control:
        print(f"\n混杂检验（场景={A.scen}）：**同为左舵**的 nuScenes vs NAVSIM")
        print(f"{'候选':<18}{'层':>3}{'维':>5}{'n_nusc':>8}{'n_navsim':>10}"
              f"{'θ跨语料':>9}{'θ同语料配平[95%CI]':>26}{'判定':>10}")
        print("-" * 90)
        cc = []
        for m in A.models.split(","):
            bs = load_by_side(A.scen, m)
            if bs is None or bs["LHD"] is None:
                continue
            b = bs["LHD"]
            nu = b["corpus"] == "nuscenes"; nv = b["corpus"] == "navsim"
            if nu.sum() < 3 or nv.sum() < 6:
                print(f"{NAME.get(m,m):<18}  左舵侧 nuScenes 只有 {int(nu.sum())} 个事件，"
                      f"NAVSIM {int(nv.sum())} 个 —— 样本不足，此混杂**无法检验**")
                continue
            for l in range(b["nL"]):
                rng = np.random.default_rng(100 + l)
                D = b["d"][l]
                cross = ang(D[nu].mean(0), D[nv].mean(0))
                mw = matched_within_angle(D[nv], b["scene"][nv],
                                          int(len(np.unique(b["scene"][nu]))), rng)
                cc.append({"model": m, "layer": l, "dim": int(D.shape[1]),
                           "n_nusc": int(nu.sum()), "n_navsim": int(nv.sum()),
                           "theta_cross": cross,
                           "null_med": (mw[0] if mw else None),
                           "null_ci": (list(mw[1:3]) if mw else None),
                           # **与跨舵位检验完全同一口径**：单边 p = 零分布中 ≥ 实测的比例
                           "p": (float((mw[3] >= cross).mean()) if mw else None)})
        # 同一套 BH-FDR(q=0.05)，否则和跨舵位的 1/48 不可比
        ps = [(i, r["p"]) for i, r in enumerate(cc) if r["p"] is not None]
        ps.sort(key=lambda t: t[1]); mt = len(ps); kk = 0
        for j, (_, pv) in enumerate(ps, 1):
            if pv <= Q_FDR * j / mt:
                kk = j
        sig = set(i for i, _ in ps[:kk])
        for i, r in enumerate(cc):
            r["corpus_diff_bh"] = bool(i in sig)
            ci = (f"{r['null_med']:6.1f} [{r['null_ci'][0]:.0f},{r['null_ci'][1]:.0f}]".rjust(26)
                  if r["null_ci"] else "--".rjust(26))
            print(f"{NAME.get(r['model'],r['model']):<18}{r['layer']:>3}{r['dim']:>5}"
                  f"{r['n_nusc']:>8}{r['n_navsim']:>10}{r['theta_cross']:>9.1f}{ci}"
                  f"{('**有语料差**' if r['corpus_diff_bh'] else '不可区分'):>10}")
        nsig = sum(1 for r in cc if r["corpus_diff_bh"])
        raw = sum(1 for r in cc if r["null_ci"] and r["theta_cross"] > r["null_ci"][1])
        print(f"\n{len(cc)} 个 layer-cell：BH-FDR(q=0.05) 后 **{nsig}** 个显著"
              f"（未校正、只看落在 95% 区间外：{raw} 个）")
        json.dump(cc, open(RES / f"vfaith_corpus_control_{A.scen}.json", "w"),
                  ensure_ascii=False, indent=1)
        return
    for m in A.models.split(","):
        byside = load_by_side(A.scen, m)
        if byside is None:
            print(f"[{NAME.get(m, m)}] 缺数据，跳过"); continue
        bench, dep = byside["LHD"], byside["RHD"]
        if bench is None or dep is None:
            print(f"[{NAME.get(m, m)}] 有一侧为空，跳过"); continue
        nL = min(bench["nL"], dep["nL"])
        for l in range(nL):
            rng = np.random.default_rng(l)
            Db, Dd = bench["d"][l], dep["d"][l]
            if Db.shape[1] != Dd.shape[1]:
                continue
            vb, kb, dim = direction(Db, bench["scene"], rng)
            vd, kd, _ = direction(Dd, dep["scene"], rng)
            hb = split_half_angle(Db, bench["scene"], rng)
            hd = split_half_angle(Dd, dep["scene"], rng)
            th_r = rand_angle(dim, rng)
            # 跨语料夹角用**稠密**均值方向，与噪声地板同口径
            cross = ang(Db.mean(0), Dd.mean(0))
            floor = np.mean([x for x in (hb, hd) if x is not None]) if (hb or hd) else None
            align = ((th_r - cross) / (th_r - floor)) if floor and th_r > floor else None
            # 样本量配平的同域对照（见 matched_within_angle 的说明）
            mw = matched_within_angle(Db, bench["scene"],
                                      len(np.unique(dep["scene"])), rng)
            # 判据：θ_cross 落在同域配平夹角的 95% 区间内 ⇒ 两域方向**无法区分**
            indist = bool(mw and mw[1] <= cross <= mw[2])
            # 单边 p：同域配平夹角里有多少比例 ≥ 实测跨域夹角
            p_dom = (float((mw[3] >= cross).mean()) if mw else None)
            from collections import Counter
            rows.append({"model": m, "layer": l, "dim": dim,
                         "n_bench": len(Db), "n_dep": len(Dd),
                         "corpus_mix_bench": dict(Counter(bench["corpus"].tolist())),
                         "corpus_mix_dep": dict(Counter(dep["corpus"].tolist())),
                         "n_scene_bench": int(len(np.unique(bench["scene"]))),
                         "n_scene_dep": int(len(np.unique(dep["scene"]))),
                         "keep_bench": kb, "keep_dep": kd,
                         "theta_cross": cross, "theta_half_bench": hb,
                         "theta_half_dep": hd, "theta_half": floor,
                         "theta_rand": th_r, "align": align,
                         "theta_within_matched": (mw[0] if mw else None),
                         "theta_within_matched_ci": (list(mw[1:3]) if mw else None),
                         "p_domain": p_dom,
                         "indistinguishable_from_within_domain": indist})
    # 24 个 layer-cell 一起做 BH-FDR（q=0.05），否则"有域差"里混着多重比较的假阳
    ps = [(i, r["p_domain"]) for i, r in enumerate(rows) if r["p_domain"] is not None]
    if ps:
        ps.sort(key=lambda t: t[1]); mtot = len(ps)
        kk = 0
        for j, (_, pv) in enumerate(ps, 1):
            if pv <= Q_FDR * j / mtot:
                kk = j
        sig = set(i for i, _ in ps[:kk])
        for i, r in enumerate(rows):
            r["domain_diff_bh"] = bool(i in sig)
    out = Path(A.out) if A.out else RES / f"vfaith_angles_{A.scen}.json"
    out.write_text(json.dumps(rows, indent=1))
    if rows:
        r0 = rows[0]
        print(f"\n场景 = {A.scen}    P-1：LHD=benchmark({r0['n_bench']} 事件 "
              f"/{r0['n_scene_bench']} scene，语料 {r0['corpus_mix_bench']}）  "
              f"RHD=deployment({r0['n_dep']} 事件/{r0['n_scene_dep']} scene，"
              f"语料 {r0['corpus_mix_dep']}）")
        if r0["n_scene_dep"] < 6:
            print("  ! deployment 侧 scene 数 < 6：split-half 噪声地板无法估，"
                  "对齐度 A 不可算；θ_cross 仍报，但无参照不可解读")
    print(f"{'候选':<18}{'层':>3}{'维':>5}{'θ跨域':>8}{'θ同域配平[95%CI]':>24}"
          f"{'地板b':>7}{'地板d':>7}{'判定':>10}")
    print("-" * 84)
    for r in rows:
        f_ = lambda v, w=8, p=1: (f"{v:{w}.{p}f}" if v is not None else f"{'--':>{w}}")
        ci = r["theta_within_matched_ci"]
        mw = (f"{r['theta_within_matched']:6.1f} [{ci[0]:.0f},{ci[1]:.0f}]".rjust(24)
              if ci else "--".rjust(24))
        vd = "**有域差**" if r.get("domain_diff_bh") else "不可区分"
        print(f"{NAME.get(r['model'], r['model']):<18}{r['layer']:>3}{r['dim']:>5}"
              f"{f_(r['theta_cross'],8)}{mw}"
              f"{f_(r['theta_half_bench'],7)}{f_(r['theta_half_dep'],7)}{vd:>10}")
    nsig = sum(1 for r in rows if r.get("domain_diff_bh"))
    raw_ = sum(1 for r in rows if r.get("theta_within_matched_ci")
               and r["theta_cross"] > r["theta_within_matched_ci"][1])
    print(f"（未校正、只看落在 95% 区间外：{raw_} 个）")
    print(f"\n{len(rows)} 个 layer-cell，BH-FDR(q=0.05) 后 {nsig} 个跨域夹角显著大于"
          f"同域配平夹角。其余 {len(rows)-nsig} 个：**θ_cross 落在"
          f"「同一个域里、一侧只有 {rows[0]['n_scene_dep'] if rows else '?'} 个场景时"
          f"天然会有的夹角」范围内 ⇒ 没有证据表明两域方向不同**。")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
