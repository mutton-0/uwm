"""F 轴：**左舵定轴，右舵当探针**。

设计（用户 2026-09-07）：
  右舵只有 8–10 个事件，用它估一个 512 维方向本来就不该做 —— 估出来的东西
  95% 是抽样噪声，两个方向的夹角于是量的是噪声不是域差。
  改成：**方向只在左舵上估**（325 事件，估得准），右舵那几个事件不估方向，
  只作为**探针**投影到这根轴上，逐个看落在哪。

  参照轴   v̂ = 左舵 top-K 层（按左舵 SNR 选，与结果无关）各层单位化后拼接
  逐事件   cos θ_i = <δ_i, v̂> / ‖δ_i‖          δ_i = h_ghost,i − h_clean,i
  零假设   **左舵自己的逐事件 cos 分布**（留一法：算 v̂ 时排除该事件本身，
           否则事件参与了自己的参照，cos 会被系统性抬高）

读法：右舵事件若落在左舵分布内 ⇒ 同一个机制在右舵照常工作；
落在左尾/负值 ⇒ 该场景下模型的危险响应与左舵学到的方向不一致。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
NAME = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
        "simlingo": "SimLingo"}
import sys                                                            # noqa: E402
sys.path.insert(0, str(Path(__file__).parent))
from f_vfaith_direction import load_by_side, layer_snr                # noqa: E402


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def pooled_event(Ds, idx, i):
    """单个事件的池化向量：选中层各自单位化后拼接（与参照轴同构造）。"""
    return np.concatenate([unit(Ds[l][i]) for l in idx])


def pooled_mean(Ds, idx, mask=None):
    out = []
    for l in idx:
        D = Ds[l] if mask is None else Ds[l][mask]
        out.append(unit(D.mean(0)))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="lead", choices=["ghost", "lead"])
    ap.add_argument("--models", default="dd,ltf,ddv2,simlingo")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    res = []
    for m in A.models.split(","):
        bs = load_by_side(A.scen, m)
        if bs is None or bs["LHD"] is None or bs["RHD"] is None:
            continue
        b, d = bs["LHD"], bs["RHD"]
        nL = min(b["nL"], d["nL"])
        rng = np.random.default_rng(7)
        snr = [layer_snr(b["d"][l], b["scene"], rng) for l in range(nL)]
        idx = sorted(np.argsort(snr)[::-1][:A.topk].tolist())

        nb = len(b["scene"])
        # 左舵逐事件 cos，**留一**（排除该事件自己）
        cos_b = []
        for i in range(nb):
            msk = np.ones(nb, bool); msk[i] = False
            v = pooled_mean(b["d"], idx, msk)
            cos_b.append(float(unit(pooled_event(b["d"], idx, i)) @ unit(v)))
        cos_b = np.array(cos_b)

        vref = pooled_mean(b["d"], idx)          # 右舵探针用全部左舵事件定的轴
        cos_d, det = [], []
        for i in range(len(d["scene"])):
            c = float(unit(pooled_event(d["d"], idx, i)) @ unit(vref))
            cos_d.append(c)
            det.append({"scene": str(d["scene"][i]), "corpus": str(d["corpus"][i]),
                        "cos": c, "theta_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1)))),
                        "pctile_in_LHD": float((cos_b < c).mean() * 100)})
        cos_d = np.array(cos_d)

        # 右舵这几个点是不是从左舵分布里抽出来的？置换检验（Mann–Whitney 的置换版）
        allc = np.concatenate([cos_b, cos_d]); nd = len(cos_d)
        obs = cos_d.mean() - cos_b.mean()
        rr = np.random.default_rng(0)
        null = np.array([(lambda p: allc[p[:nd]].mean() - allc[p[nd:]].mean())
                         (rr.permutation(len(allc))) for _ in range(20000)])
        p2 = float((np.abs(null) >= abs(obs)).mean())

        res.append({"model": m, "scen": A.scen, "layers": idx, "topk": A.topk,
                    "n_lhd": int(nb), "n_rhd": int(len(cos_d)),
                    "cos_lhd_mean": float(cos_b.mean()),
                    "cos_lhd_q": [float(np.percentile(cos_b, q)) for q in (5, 25, 50, 75, 95)],
                    "cos_rhd_mean": float(cos_d.mean()),
                    "diff": float(obs), "perm_p": p2, "rhd_events": det})

        print(f"\n===== {NAME.get(m,m)}  |  {A.scen}  |  轴 = 左舵 {nb} 事件 "
              f"top-{A.topk} 层 {idx} =====")
        q = res[-1]["cos_lhd_q"]
        print(f"  左舵逐事件 cos（留一）: 均值 {cos_b.mean():+.3f}   "
              f"分位 5%={q[0]:+.3f} 25%={q[1]:+.3f} 50%={q[2]:+.3f} 75%={q[3]:+.3f} 95%={q[4]:+.3f}")
        print(f"  右舵 {len(cos_d)} 个探针: 均值 {cos_d.mean():+.3f}   "
              f"差 {obs:+.3f}   置换检验 p={p2:.4f}")
        print(f"  {'右舵事件':<26}{'语料':>9}{'cos':>9}{'θ':>8}{'在左舵分布的分位':>18}")
        for x in sorted(det, key=lambda z: z["cos"]):
            print(f"  {x['scene']:<26}{x['corpus']:>9}{x['cos']:>+9.3f}"
                  f"{x['theta_deg']:>7.1f}°{x['pctile_in_LHD']:>17.1f}%")

    out = Path(A.out) if A.out else RES / f"vfaith_probe_{A.scen}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
