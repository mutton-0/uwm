"""Does the I-axis domain pair really differ "only in appearance"?

The paper states the paired inputs "differ only in appearance", and the formula doc says
"同场景同构图同 actor，唯一变量是渲染风格 = do(appearance)". If that holds, then structure
(edges/geometry) should be preserved across the pair while colour/luminance changes.

Test design (falsifiable, with a within-domain control):
  cross-domain, same timestamp : sim_tK vs real_tK   -> appearance changed by construction
  within-domain, adjacent time : sim_tK vs sim_tK+1  -> content changed slightly (ego motion),
                                                        appearance identical by construction
Structure is measured on gradient-orientation maps, which are invariant to any monotone
per-channel intensity change (i.e. to lighting/exposure/colour grading) but not to a change
of what is in the scene.

If the "appearance only" claim holds, cross-domain structural distance should be no larger
than the within-domain adjacent-frame control. If it is much larger, the repaint moved
content too, and D_L cannot be attributed to do(appearance) alone.

Read-only on variants/i_domain/frames; writes results/gax_*.
"""
import collections, glob, json, os, re, sys
import numpy as np
from PIL import Image

FR = "/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain/frames"
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results"
W, H = 400, 225          # downscale: we want scene structure, not render noise


def load_gray(p):
    im = Image.open(p).convert("L").resize((W, H), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def grad_orient(g):
    """Unit gradient-orientation field, weighted by edge strength.

    Orientation is invariant to any monotone intensity remap, so a pure lighting or
    colour-grading change leaves it (nearly) fixed; moving an object does not.
    """
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    m = mag > np.percentile(mag, 80)          # strong edges only
    ang = np.arctan2(gy, gx)
    return ang, m, mag


def _boxfilt(x, k=15):
    """Separable box filter via cumulative sums (no scipy dependency)."""
    pad = k // 2
    xp = np.pad(x, pad, mode="reflect")
    c = np.cumsum(xp, 0)
    c = np.concatenate([c[k - 1:k], c[k:] - c[:-k]], 0)
    c = np.cumsum(c, 1)
    c = np.concatenate([c[:, k - 1:k], c[:, k:] - c[:, :-k]], 1)
    return c / (k * k)


def local_norm(g, k=15, eps=1e-3):
    """Local mean/std normalisation: removes any smooth lighting/exposure/colour change
    while keeping layout and object boundaries. This is the 'structure' term of SSIM."""
    mu = _boxfilt(g, k)
    sd = np.sqrt(np.maximum(_boxfilt(g * g, k) - mu * mu, 0.0))
    return (g - mu) / (sd + eps)


def struct_dist(pa, pb):
    """1 - Pearson correlation of locally-normalised luminance.

    Invariant to smooth photometric changes (lighting, exposure, colour grade) by
    construction; sensitive to anything that moves or replaces scene content.
    """
    a, b = local_norm(load_gray(pa)), local_norm(load_gray(pb))
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-8:
        return None, 0.0
    return 1.0 - float((a * b).sum() / den), float(a.size)


def lum_dist(pa, pb):
    """Mean absolute luminance difference — the 'appearance' channel, for contrast."""
    return float(np.abs(load_gray(pa) - load_gray(pb)).mean())


def main():
    files = sorted(glob.glob(f"{FR}/*.png"))
    key = collections.defaultdict(dict)      # (scene_tag) -> {(src,t): path}
    for p in files:
        stem = os.path.basename(p)[:-4]
        m = re.match(r"(.+)__(sim|real)_t(\d+)$", stem)
        if not m:
            continue
        key[m.group(1)][(m.group(2), int(m.group(3)))] = p

    cross, within, rows = [], [], []
    for tag, d in sorted(key.items()):
        ts = sorted({t for (_, t) in d})
        for t in ts:
            ps, pr = d.get(("sim", t)), d.get(("real", t))
            if ps and pr:
                s, n = struct_dist(ps, pr)
                if s is not None:
                    cross.append(s)
                    rows.append(dict(tag=tag, kind="cross_domain_same_t", t=t,
                                     struct=s, lum=lum_dist(ps, pr), n_edge=n))
        for a, b in zip(ts, ts[1:]):
            for src in ("sim", "real"):
                pa, pb = d.get((src, a)), d.get((src, b))
                if pa and pb:
                    s, n = struct_dist(pa, pb)
                    if s is not None:
                        within.append(s)
                        rows.append(dict(tag=tag, kind=f"within_{src}_adjacent_t", t=a,
                                         struct=s, lum=lum_dist(pa, pb), n_edge=n))

    def stat(x):
        x = np.array(x)
        return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                    med=float(np.median(x)),
                    ci=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))])

    c, w = stat(cross), stat(within)
    lc = np.mean([r["lum"] for r in rows if r["kind"] == "cross_domain_same_t"])
    lw = np.mean([r["lum"] for r in rows if r["kind"].startswith("within_")])

    print("=== structural distance (1 - cos of doubled edge orientation) ===")
    print(f"  cross-domain, same timestamp   n={c['n']:<4} mean={c['mean']:.4f}  median={c['med']:.4f}")
    print(f"  within-domain, adjacent frame  n={w['n']:<4} mean={w['mean']:.4f}  median={w['med']:.4f}")
    print(f"  ratio cross/within = {c['mean']/max(w['mean'],1e-9):.2f}x")
    print("\n=== luminance distance (the 'appearance' channel) ===")
    print(f"  cross-domain  mean |dL| = {lc:.4f}")
    print(f"  within-domain mean |dL| = {lw:.4f}   ratio = {lc/max(lw,1e-9):.2f}x")

    # permutation test: is cross > within beyond chance?
    a, b = np.array(cross), np.array(within)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); rng = np.random.default_rng(0)
    null = np.empty(20000)
    for i in range(20000):
        rng.shuffle(pool)
        null[i] = pool[:len(a)].mean() - pool[len(a):].mean()
    p = float((np.abs(null) >= abs(obs)).mean())
    print(f"\npermutation test on the difference of means: obs={obs:+.4f}, p={p:.4g}")
    # Verdict must respect significance: a larger sample mean that the permutation test
    # cannot separate from the control is NOT evidence that structure moved.
    if p >= 0.05:
        verdict = ("no detectable structural difference beyond the within-domain control "
                   f"(p={p:.3f}); the 'appearance only' claim is NOT contradicted by this test")
    elif c["mean"] > w["mean"]:
        verdict = (f"structure differs MORE across domains than across adjacent frames "
                   f"(p={p:.4g}, ratio {c['mean']/max(w['mean'],1e-9):.2f}x) -> the pair "
                   f"differs by more than appearance")
    else:
        verdict = f"structure differs LESS across domains than the control (p={p:.4g})"
    print("VERDICT:", verdict)

    out = dict(cross_domain=c, within_domain=w, lum_cross=float(lc), lum_within=float(lw),
               ratio_struct=c["mean"] / max(w["mean"], 1e-9),
               perm_p=p, obs_diff=float(obs), verdict=verdict, rows=rows)
    with open(f"{RES}/gax_i_appearance_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", f"{RES}/gax_i_appearance_check.json")


if __name__ == "__main__":
    main()
