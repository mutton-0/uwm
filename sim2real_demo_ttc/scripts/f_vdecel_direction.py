"""按 (左右舵, 速度段) 分组算减速方向，并与配对法的 v_faith 比夹角。

两条路线：
  v_decel(side, band) = mean(h | brake) − mean(h | cruise)      非配对、群体差、无遮挡
  v_faith(L)          = mean_i( h_ghost,i − h_clean,i )          配对、同帧、靠遮挡

按速度段分组是构造上消掉速度混杂的手段（自车速度是模型显式输入）。

## 三个必须一起看的量
  θ(v_decel, v_faith)   两条路线是否在说同一件事
  θ_half                各自的 split-half 噪声地板（同分布同数据量能达到的上限）
  θ_rand ≈ 90°          随机方向的样子；不给这个，任何角度都无法解释
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
NAME = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2"}
BANDS = ["0.5-4", "4-8", "8-12", "12-30"]


def ang(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    if n < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(float(a @ b) / n, -1, 1))))


def rand_angle(d, rng, B=2000):
    a, b = rng.normal(size=(B, d)), rng.normal(size=(B, d))
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())


def split_half(A, B_, rng, B=200):
    """两组各自随机劈半 -> 两个组差方向 -> 夹角中位。同数据量下的噪声地板。"""
    if len(A) < 6 or len(B_) < 6:
        return None
    out = []
    for _ in range(B):
        pa, pb = rng.permutation(len(A)), rng.permutation(len(B_))
        ha, hb = len(A) // 2, len(B_) // 2
        v1 = B_[pb[:hb]].mean(0) - A[pa[:ha]].mean(0)
        v2 = B_[pb[hb:]].mean(0) - A[pa[ha:]].mean(0)
        a = ang(v1, v2)
        if a is not None:
            out.append(a)
    return float(np.median(out)) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True, help="vdecel_acts_*.npz")
    ap.add_argument("--model", required=True, choices=list(NAME))
    ap.add_argument("--vfaith", default="", help="vfaith_acts_navsim_lead_{m}.npz，用于比夹角")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    z = np.load(A.acts, allow_pickle=True)
    nL = sum(1 for k in z.files if k.startswith("h__L"))
    kind, band, sideb = z["kind"], z["band"].astype(int), z["side"]

    # 配对法方向（若给了）
    VF = None
    if A.vfaith and Path(A.vfaith).exists():
        f = np.load(A.vfaith, allow_pickle=True)
        VF = [(f[f"h_ghost__L{l}"] - f[f"h_clean__L{l}"]).mean(0) for l in range(nL)]

    rows = []
    for sd in ("LHD", "RHD"):
        for b in range(len(BANDS)):
            mC = (kind == "cruise") & (sideb == sd) & (band == b)
            mB = (kind == "brake") & (sideb == sd) & (band == b)
            if mC.sum() < 5 or mB.sum() < 5:
                rows.append({"side": sd, "band": BANDS[b], "n_cruise": int(mC.sum()),
                             "n_brake": int(mB.sum()), "layers": None,
                             "note": "样本不足（任一组 <5），不可估"})
                continue
            per_layer = []
            for l in range(nL):
                Hc, Hb = z[f"h__L{l}"][mC], z[f"h__L{l}"][mB]
                rng = np.random.default_rng(100 * b + l)
                v = Hb.mean(0) - Hc.mean(0)
                fl = split_half(Hc, Hb, rng)
                th_r = rand_angle(v.shape[0], rng)
                per_layer.append({
                    "layer": l, "dim": int(v.shape[0]),
                    "theta_half": fl, "theta_rand": th_r,
                    "theta_vs_vfaith": ang(v, VF[l]) if VF is not None else None,
                    "norm_ratio": float(np.linalg.norm(v) /
                                        max(np.linalg.norm(Hc.mean(0)), 1e-9))})
            rows.append({"side": sd, "band": BANDS[b], "n_cruise": int(mC.sum()),
                         "n_brake": int(mB.sum()), "layers": per_layer})

    out = Path(A.out) if A.out else RES / f"vdecel_dirs_{A.model}.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\n{NAME[A.model]}  —— v_decel = mean(brake) − mean(cruise)，按 (舵向, 速度段) 分组")
    print(f"{'舵':<5}{'速度段':<9}{'n巡航':>6}{'n刹车':>6}"
          f"{'θ地板(中位层)':>14}{'θ随机':>8}{'θ vs v_faith':>14}")
    print("-" * 64)
    for r in rows:
        if r["layers"] is None:
            print(f"{r['side']:<5}{r['band']:<9}{r['n_cruise']:>6}{r['n_brake']:>6}"
                  f"{'  ' + r['note']:>36}")
            continue
        fl = [x["theta_half"] for x in r["layers"] if x["theta_half"] is not None]
        vf = [x["theta_vs_vfaith"] for x in r["layers"] if x["theta_vs_vfaith"] is not None]
        tr = [x["theta_rand"] for x in r["layers"]]
        f_ = lambda v: f"{np.median(v):.1f}" if v else "--"
        print(f"{r['side']:<5}{r['band']:<9}{r['n_cruise']:>6}{r['n_brake']:>6}"
              f"{f_(fl):>14}{f_(tr):>8}{f_(vf):>14}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
