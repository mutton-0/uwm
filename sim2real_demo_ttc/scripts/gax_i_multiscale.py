"""Separate 'texture changed' (= appearance, legitimate) from 'layout changed' (= content,
which would break the do(appearance) claim), by measuring structural distance at several
spatial scales.

At fine scales, local-normalised correlation is dominated by surface texture, which a
photoreal repaint is *supposed* to change. At coarse scales texture is averaged away and
what remains is scene layout: horizon line, road geometry, where the actors are. If the
cross-domain excess over the within-domain control survives to coarse scales, the pair
differs in layout, not merely appearance.
"""
import collections, glob, json, os, re
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gax_i_appearance_check import local_norm

FR = "/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain/frames"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"
SCALES = [(800, 450), (400, 225), (200, 112), (100, 56), (50, 28), (25, 14)]


def gray(p, wh):
    return np.asarray(Image.open(p).convert("L").resize(wh, Image.BILINEAR), np.float32) / 255.0


def dist(pa, pb, wh, k):
    a, b = local_norm(gray(pa, wh), k), local_norm(gray(pb, wh), k)
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return None if d < 1e-8 else 1.0 - float((a * b).sum() / d)


def main():
    key = collections.defaultdict(dict)
    for p in sorted(glob.glob(f"{FR}/*.png")):
        m = re.match(r"(.+)__(sim|real)_t(\d+)$", os.path.basename(p)[:-4])
        if m:
            key[m.group(1)][(m.group(2), int(m.group(3)))] = p

    print(f"{'scale':<12}{'k':>3}{'cross':>9}{'within':>9}{'excess':>9}{'ratio':>8}{'p':>10}")
    print("-" * 62)
    out = []
    rng = np.random.default_rng(0)
    for wh in SCALES:
        k = max(3, (min(wh) // 15) | 1)          # window ~1/15 of frame height, odd
        cross, within = [], []
        for tag, d in key.items():
            ts = sorted({t for (_, t) in d})
            for t in ts:
                ps, pr = d.get(("sim", t)), d.get(("real", t))
                if ps and pr:
                    v = dist(ps, pr, wh, k)
                    if v is not None:
                        cross.append(v)
            for a, b in zip(ts, ts[1:]):
                for src in ("sim", "real"):
                    pa, pb = d.get((src, a)), d.get((src, b))
                    if pa and pb:
                        v = dist(pa, pb, wh, k)
                        if v is not None:
                            within.append(v)
        A, B = np.array(cross), np.array(within)
        obs = A.mean() - B.mean()
        pool = np.concatenate([A, B]); null = np.empty(10000)
        for i in range(10000):
            rng.shuffle(pool)
            null[i] = pool[:len(A)].mean() - pool[len(A):].mean()
        p = float((np.abs(null) >= abs(obs)).mean())
        print(f"{str(wh):<12}{k:>3}{A.mean():>9.4f}{B.mean():>9.4f}{obs:>+9.4f}"
              f"{A.mean()/max(B.mean(),1e-9):>8.2f}{p:>10.4g}")
        out.append(dict(scale=list(wh), k=k, cross=float(A.mean()), within=float(B.mean()),
                        excess=float(obs), ratio=float(A.mean() / max(B.mean(), 1e-9)), p=p,
                        n_cross=len(A), n_within=len(B)))
    with open(f"{RES}/gax_i_multiscale.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", f"{RES}/gax_i_multiscale.json")
    coarse = [r for r in out if r["scale"][0] <= 100]
    sig = [r for r in coarse if r["p"] < 0.05 and r["excess"] > 0]
    print(f"\ncoarse scales (<=100px wide) with significant cross-domain excess: "
          f"{len(sig)}/{len(coarse)}")
    print("=> " + ("layout differs, not just appearance" if len(sig) >= 2
                   else "excess is confined to fine scales = texture/appearance only"))


if __name__ == "__main__":
    main()
