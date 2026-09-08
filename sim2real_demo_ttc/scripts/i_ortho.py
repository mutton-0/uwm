"""I 轴 ⊥ F 轴：亮度敏感方向与两个「动作侧」方向的正交程度。

## 三个方向
  v_bright(L) = mean_i( h_变夜,i − h_原图,i )      同帧、只改外观
  v_faith(L)  = mean_i( h_ghost,i − h_clean,i )    同帧、只删危险物   ← F 轴方向
  v_decel(L)  = mean(h|brake) − mean(h|cruise)     非配对、群体差     ← 动作方向

## 「正交」必须反过来问
d 维空间里两个随机方向的夹角以极高概率就是 90°，所以 θ≈90° **不是**
「亮度不影响动作」的证据 —— 它就是零假设本身。可证伪的问法是
**cos θ 是否显著偏离 0**：

  cos θ 的 scene 级 bootstrap 95% CI 含 0  ⇒ 与正交不可区分（好）
  CI 不含 0                                ⇒ 亮度方向确实压在动作方向上（坏）

同时报三个参照量，缺一不可读：
  θ_half(v_bright)  亮度方向自身的 split-half 噪声地板 —— 它自己估不准就别谈夹角
  ‖v_bright‖/‖v_*‖  相对幅度 —— 正交但巨大 ≠ 无害
  proj = ‖v_bright‖·cos θ  沿动作轴的有符号投影，单位与动作轴一致
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
NAME = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
        "simlingo": "SimLingo"}
MAP = json.load(open(RES / "driveside_map.json"))
BANDS = ["0.5-4", "4-8", "8-12", "12-30"]


def side_of(s):
    for sp in ("test", "trainval"):
        x = MAP["navsim_side"].get(f"{sp}|{s}")
        if x:
            return x


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ang(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return None if n < 1e-12 else float(np.degrees(np.arccos(np.clip(float(a @ b) / n, -1, 1))))


def part_ratio(D):
    """有效维度（participation ratio）= (Σλ)² / Σλ²，λ 为逐事件差的协方差特征值。

    PR ≈ 1 ⇒ 该层的所有扰动都挤在**同一根轴**上，此时任意两个方向的夹角
    要么 0° 要么 180°，**角度不携带信息**。实测 DD 的 L0/L1 就是这种情况
    （cos(v_bright, v_decel) = −0.99，而 split-half 地板只有 0.1–1.3°）。
    故设门：PR < 3 的层不参与正交性判定。
    """
    X = D - D.mean(0)
    if len(X) < 3:
        return float("nan")
    lam = np.linalg.svd(X, compute_uv=False) ** 2
    s = lam.sum()
    return float(s * s / max((lam ** 2).sum(), 1e-30))


def split_half(D, sc, rng, B=200):
    uq = np.unique(sc)
    if len(uq) < 6:
        return None
    o = []
    for _ in range(B):
        p = rng.permutation(uq); h = len(uq) // 2
        a = ang(D[np.isin(sc, p[:h])].mean(0), D[np.isin(sc, p[h:])].mean(0))
        if a is not None:
            o.append(a)
    return float(np.median(o)) if o else None


def cos_ci(D, sc, vref, rng, B=2000):
    """cos θ 的 scene 级 bootstrap：每次重抽 scene 重算 v_bright，对固定参照取 cos。"""
    uq = np.unique(sc); ix = [np.where(sc == s)[0] for s in uq]
    r = unit(vref); o = []
    for _ in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])
        v = D[sel].mean(0)
        if np.linalg.norm(v) > 1e-12:
            o.append(float(unit(v) @ r))
    o = np.array(o)
    return float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def vdecel(m, l, band_ix, side="LHD"):
    f = RES / f"vdecel_acts_navsim_test_{m}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    sel = (z["side"] == side) & (z["band"] == band_ix)
    br = sel & (z["kind"] == "brake"); cr = sel & (z["kind"] == "cruise")
    if br.sum() < 5 or cr.sum() < 5:
        return None
    H = z[f"h__L{l}"]
    return H[br].mean(0) - H[cr].mean(0)


def vfaith(m, l):
    f = RES / f"vfaith_acts_navsim_lead_{m}.npz"
    if not f.exists():
        return None, None, None
    z = np.load(f, allow_pickle=True)
    sc = np.array([str(x) for x in z["scene"]])
    k = np.array([side_of(s) == "LHD" for s in sc])
    D = (z[f"h_ghost__L{l}"] - z[f"h_clean__L{l}"])[k]
    return D.mean(0), D, sc[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dd,ltf,ddv2,simlingo")
    ap.add_argument("--band", default="4-8")
    ap.add_argument("--out", default=str(RES / "i_ortho.json"))
    A = ap.parse_args()
    rows = []
    for m in A.models.split(","):
        for scope in ("sky", "global"):
            for nn in (True, False):
                f = RES / (f"vbright_acts_navsim_lead_{m}_night_{scope}"
                           f"{'_nonoise' if nn else ''}.npz")
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                nL = sum(1 for k in z.files if k.startswith("h_orig__L"))
                sc = np.array([str(x) for x in z["scene"]])
                keep = np.array([side_of(s) == "LHD" for s in sc])
                db = (z["v_alt"] - z["v_orig"])[keep]
                for l in range(nL):
                    D = (z[f"h_alt__L{l}"] - z[f"h_orig__L{l}"])[keep]
                    rng = np.random.default_rng(l)
                    vb = D.mean(0)
                    r = {"model": m, "scope": scope, "no_noise": nn, "layer": l,
                         "dim": int(D.shape[1]), "n": int(keep.sum()),
                         "db_mean": float(db.mean()), "db_absmean": float(np.abs(db).mean()),
                         "norm_bright": float(np.linalg.norm(vb)),
                         "theta_half_bright": split_half(D, sc[keep], rng),
                         "part_ratio": part_ratio(D)}
                    vf, _, _ = vfaith(m, l)
                    if vf is not None and len(vf) == len(vb):
                        c = cos_ci(D, sc[keep], vf, rng)
                        r.update(theta_vs_vfaith=ang(vb, vf), cos_vfaith=c[0],
                                 cos_vfaith_ci=[c[1], c[2]],
                                 norm_ratio_vfaith=float(np.linalg.norm(vb)
                                                         / max(np.linalg.norm(vf), 1e-12)),
                                 proj_on_vfaith=float(vb @ unit(vf)))
                    vd = vdecel(m, l, BANDS.index(A.band))
                    if vd is not None and len(vd) == len(vb):
                        c = cos_ci(D, sc[keep], vd, rng)
                        r.update(theta_vs_vdecel=ang(vb, vd), cos_vdecel=c[0],
                                 cos_vdecel_ci=[c[1], c[2]],
                                 proj_on_vdecel=float(vb @ unit(vd)))
                    rows.append(r)
    Path(A.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    print(f"{'候选':<17}{'范围':<7}{'噪声':<5}{'层':>3}{'Δb(>0=变快)':>13}"
          f"{'θ vs F轴':>10}{'cos [95%CI]':>24}{'判':>6}"
          f"{'θ vs 减速':>10}{'cos [95%CI]':>24}{'判':>6}{'地板':>7}")
    print("-" * 133)
    for r in rows:
        def blk(key):
            c = r.get(f"cos_{key}"); ci = r.get(f"cos_{key}_ci"); t = r.get(f"theta_vs_{key}")
            if c is None:
                return f"{'--':>10}{'--':>24}{'--':>6}"
            orth = "正交" if (ci[0] <= 0 <= ci[1]) else "**压**"
            return (f"{t:>9.1f}°" + f"{c:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]".rjust(24)
                    + f"{orth:>6}")
        hb = r["theta_half_bright"]
        print(f"{NAME.get(r['model'],r['model']):<17}{r['scope']:<7}"
              f"{('无' if r['no_noise'] else '有'):<5}{r['layer']:>3}{r['db_mean']:>+13.4f}"
              + blk("vfaith") + blk("vdecel")
              + (f"{hb:>7.1f}" if hb else f"{'--':>7}"))
    print(f"\n-> {A.out}")


if __name__ == "__main__":
    main()
