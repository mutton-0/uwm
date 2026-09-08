"""I 轴：亮度敏感方向 v_bright，及它与减速方向 v_decel 的正交程度。

## 读数（按用户 2026-09-07 的口径）
主读数是**行为改动量** Δb = arc_full(变外观) − arc_full(原图)，不量 D_L。

**符号**：arc_full = 弧长/时域 = 规划速度，故 **Δb > 0 = 变外观之后开得更快**。
与 F 轴同序（b_model = arc_full(危险物已移除) − arc_full(原图)，>0 = 移除危险后变快）。
所以 Δb > 0 的读法是：模型把"天黑了"当成了和"危险没了"同方向的信号 —— **不安全方向**。
天变暗理应让它更谨慎或不动，变快是反的。
机制读数是 v_bright 对 v_decel 的关系。

## 「正交」必须反过来问
d 维空间里两个随机方向的夹角以极高概率就是 90°。所以 θ≈90° **不是**
"亮度不影响动作"的证据 —— 它就是零假设本身，任何一对无关向量都长这样。
可证伪的问法是反过来的：**cos θ 是否显著偏离 0**。

  投影 proj = <v_bright, v̂_decel> = ‖v_bright‖·cos θ    （有符号，单位 = 减速轴）
  cos θ 的 scene 级 bootstrap CI 是否含 0
  ‖v_bright‖ 本身多大（正交但巨大 ≠ 无害）

同时报两个噪声地板（各自 split-half）：若 v_bright 自己都估不准，
它和任何东西的夹角都不可解读。
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
NAME = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
        "simlingo": "SimLingo"}
MAP = json.load(open(RES / "driveside_map.json"))
BANDS = ["0.5-4", "4-8", "8-12", "12-30"]


def side_of(scene, corpus="navsim"):
    if corpus == "nuscenes":
        return MAP["nuscenes_side"].get(scene)
    for sp in ("test", "trainval"):
        s = MAP["navsim_side"].get(f"{sp}|{scene}")
        if s:
            return s


def ang(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return None if n < 1e-12 else float(np.degrees(np.arccos(np.clip(float(a @ b) / n, -1, 1))))


def split_half(D, scenes, rng, B=200):
    uq = np.unique(scenes)
    if len(uq) < 6:
        return None
    out = []
    for _ in range(B):
        perm = rng.permutation(uq); h = len(uq) // 2
        va = D[np.isin(scenes, perm[:h])].mean(0)
        vb = D[np.isin(scenes, perm[h:])].mean(0)
        a = ang(va, vb)
        if a is not None:
            out.append(a)
    return float(np.median(out)) if out else None


def cos_boot(D, scenes, vd, rng, B=2000):
    """cos θ 的 scene 级 bootstrap：每次重抽 scene 重算 v_bright，再对固定的 v_decel 取 cos。"""
    uq = np.unique(scenes); ix = [np.where(scenes == s)[0] for s in uq]
    vdn = vd / max(np.linalg.norm(vd), 1e-12)
    out = []
    for _ in range(B):
        sel = np.concatenate([ix[i] for i in rng.integers(0, len(ix), len(ix))])
        v = D[sel].mean(0); n = np.linalg.norm(v)
        if n > 1e-12:
            out.append(float(v @ vdn) / n)
    o = np.array(out)
    return float(o.mean()), float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dd,ltf,ddv2")
    ap.add_argument("--kind", default="night")
    ap.add_argument("--band", default="4-8", help="v_decel 用哪个速度段（样本最多的）")
    ap.add_argument("--out", default=str(RES / "i_bright_orthogonality.json"))
    A = ap.parse_args()

    def vdecel(m, l, side="LHD", band_ix=1):
        """从原始激活直接重算 v_decel(side, band, L) = mean(brake) − mean(cruise)。
        vdecel_dirs_*.json 只存了角度没存向量，故在此重算（同一批数据、同一口径）。"""
        f = RES / f"vdecel_acts_navsim_test_{m}.npz"
        if not f.exists():
            return None, 0, 0
        z = np.load(f, allow_pickle=True)
        sel = (z["side"] == side) & (z["band"] == band_ix)
        br = sel & (z["kind"] == "brake"); cr = sel & (z["kind"] == "cruise")
        if br.sum() < 5 or cr.sum() < 5:
            return None, int(cr.sum()), int(br.sum())
        H = z[f"h__L{l}"]
        return H[br].mean(0) - H[cr].mean(0), int(cr.sum()), int(br.sum())

    rows = []
    print(f"{'候选':<17}{'范围':<8}{'噪声':<6}{'层':>3}{'维':>5}"
          f"{'Δb均值(>0=变快)':>16}{'|Δb|':>8}{'‖v_b‖':>9}{'θ vs v_decel':>13}"
          f"{'cosθ [95%CI]':>22}{'地板b':>7}")
    print("-" * 116)
    for m in A.models.split(","):
        for scope in ("sky", "global"):
            for nn, ntag in ((False, "有"), (True, "无")):
                f = (RES / f"vbright_acts_navsim_lead_{m}_{A.kind}_{scope}"
                     f"{'_nonoise' if nn else ''}.npz")
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                nL = sum(1 for k in z.files if k.startswith("h_orig__L"))
                sc = np.array([str(x) for x in z["scene"]])
                keep = np.array([side_of(s) == "LHD" for s in sc])   # benchmark 侧
                db = (z["v_alt"] - z["v_orig"])[keep]
                for l in range(nL):
                    D = (z[f"h_alt__L{l}"] - z[f"h_orig__L{l}"])[keep]
                    rng = np.random.default_rng(l)
                    vb = D.mean(0)
                    nb = float(np.linalg.norm(vb))
                    hb = split_half(D, sc[keep], rng)
                    vdec, ncr, nbr = vdecel(m, l, "LHD", BANDS.index(A.band))
                    th = cb = None
                    if vdec is not None and len(vdec) == len(vb):
                        th = ang(vb, vdec)
                        cb = cos_boot(D, sc[keep], vdec, rng)
                    rows.append({"model": m, "scope": scope, "no_noise": nn, "layer": l,
                                 "dim": int(D.shape[1]), "n": int(keep.sum()),
                                 "db_mean": float(db.mean()), "db_absmean": float(np.abs(db).mean()),
                                 "norm_v_bright": nb, "theta_half_bright": hb,
                                 "theta_vs_vdecel": th,
                                 "cos_mean": (cb[0] if cb else None),
                                 "cos_ci95": (list(cb[1:]) if cb else None),
                                 "vdecel_band": A.band, "n_cruise": ncr, "n_brake": nbr})
                    cs = (f"{cb[0]:+.3f} [{cb[1]:+.3f},{cb[2]:+.3f}]" if cb else "--").rjust(22)
                    print(f"{NAME.get(m,m):<17}{scope:<8}{ntag:<6}{l:>3}{D.shape[1]:>5}"
                          f"{db.mean():>+16.4f}{np.abs(db).mean():>8.4f}{nb:>9.3f}"
                          f"{(f'{th:.1f}' if th else '--'):>13}{cs}"
                          f"{(f'{hb:.1f}' if hb else '--'):>7}")
    json.dump(rows, open(A.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {A.out}")


if __name__ == "__main__":
    main()
