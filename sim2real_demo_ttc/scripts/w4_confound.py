"""W4｜为"语言轴投影差与 b 负相关"排两个混淆（guide 附录 B W4，零 GPU）。

已测事实：ρ(投影$_{lang}$, b) = −0.161（p=0.006）——危险轴亮得越强，模型实际刹得越少。
与 steering 的因果方向（+α ⇒ 减速）**反号**。两个候选解释：

  ① **Simpson**：相关只是事件类型/场景构成造成的假象
     -> 仅在 A 类内部算（本来就是），再拆 scene 内 / scene 间；
  ② **"可谈论度"而非"危险度"**：词汇轴测的可能是"这一帧有多少可说的东西"
     （目标多、画面满 => 语言侧激活强），而画面满的场景模型本来就开得保守、
     b 的下降空间小 -> 控住**场景复杂度**（目标数、bbox 总面积）后重算。

复杂度取自 nuScenes 原生标注（ghost 帧所属 sample 在 CAM_FRONT 里的可见目标）。
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
from g3_metrics import load_cache  # noqa: E402


def partial_spearman(x, y, zs):
    """秩空间偏相关：x、y 各自对 zs（秩）回归后取残差再算相关。"""
    R = stats.rankdata
    Z = np.column_stack([R(z) for z in zs] + [np.ones(len(x))])
    rx = R(x) - Z @ np.linalg.lstsq(Z, R(x), rcond=None)[0]
    ry = R(y) - Z @ np.linalg.lstsq(Z, R(y), rcond=None)[0]
    return stats.spearmanr(rx, ry)


def scene_boot(x, y, scenes, n_boot=2000, seed=0):
    by = defaultdict(list)
    for i, s in enumerate(scenes):
        by[s].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    st = []
    for _ in range(n_boot):
        idx = [i for j in rng.choice(len(keys), len(keys), replace=True) for i in by[keys[j]]]
        r_ = stats.spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(r_):
            st.append(r_)
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--axis", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--pool-mode", default="query_mean")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    V = np.load(args.axis).astype(np.float32)
    L = args.layer
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))
    A = [(eid, e) for eid, e in items.items() if evmap[eid]["event_type"] == "A"]
    print(f"[W4] A 类事件 n={len(A)}  轴={Path(args.axis).name}@L{L}")

    d = np.array([float((e["h_ghost"][L] - e["h_clean"][L]) @ V[L]) for _, e in A])
    b = np.array([e["b"] for _, e in A])
    dego = np.array([e["d_ego"] for _, e in A])
    nrm = np.array([np.linalg.norm(e["h_ghost"][L] - e["h_clean"][L]) for _, e in A])
    sc = np.array([e["scene"] for _, e in A])

    # ---------- W4-① Simpson ----------
    print("\n=== W4-① 类内 + Simpson 分解 ===")
    r0, p0 = stats.spearmanr(d, b)
    lo, hi = scene_boot(d, b, sc)
    print(f"  A 类内 ρ(投影, b) = {r0:+.3f}  p={p0:.4g}  scene 级 bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")

    by = defaultdict(list)
    for i, s in enumerate(sc):
        by[s].append(i)
    wi_d, wi_b = [], []
    multi = [v for v in by.values() if len(v) >= 3]
    for idx in multi:
        wi_d += list(d[idx] - d[idx].mean()); wi_b += list(b[idx] - b[idx].mean())
    bt_d = np.array([d[i].mean() for i in by.values()])
    bt_b = np.array([b[i].mean() for i in by.values()])
    rw, pw = stats.spearmanr(np.array(wi_d), np.array(wi_b))
    rb, pb = stats.spearmanr(bt_d, bt_b)
    print(f"  scene 内（{len(multi)} 场景, n={len(wi_d)}）ρ={rw:+.3f}  p={pw:.4g}")
    print(f"  scene 间（{len(bt_d)} 场景均值）      ρ={rb:+.3f}  p={pb:.4g}")
    simpson = "❌ 非 Simpson：两层级同号同量级" if (rw < 0 and rb < 0) else "⚠️ 两层级不一致，需查"
    print(f"  -> {simpson}")

    # ---------- W4-② 复杂度偏相关 ----------
    print("\n=== W4-② 场景复杂度偏相关 ===")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"],
                    dataroot=cfg["paths"]["nuscenes_root"], verbose=False)
    n_obj, area_sum = [], []
    for eid, e in A:
        ev = evmap[eid]
        sd_tok = ev["x_ghost_frames"][0]["sd_token"]
        try:
            sd = nusc.get("sample_data", sd_tok)
            _, boxes, cam = nusc.get_sample_data(sd["token"] if sd["is_key_frame"]
                                                 else nusc.get("sample", sd["sample_token"])["data"]["CAM_FRONT"])
            n_obj.append(len(boxes))
            tot = 0.0
            for bx in boxes:
                w, h, l = bx.wlh
                dist = float(np.linalg.norm(bx.center))
                tot += cam[0, 0] * cam[1, 1] * (w * h) / max(dist, 1.0) ** 2
            area_sum.append(tot)
        except Exception:
            n_obj.append(np.nan); area_sum.append(np.nan)
    n_obj = np.array(n_obj, float); area_sum = np.array(area_sum, float)
    ok = np.isfinite(n_obj) & np.isfinite(area_sum)
    print(f"  取到复杂度的事件 {ok.sum()}/{len(A)}   目标数中位 {np.nanmedian(n_obj):.0f}  "
          f"bbox 总面积中位 {np.nanmedian(area_sum):.0f} px")

    for lab, z in (("目标数", n_obj), ("bbox 总面积", area_sum)):
        r_, p_ = stats.spearmanr(d[ok], z[ok])
        r2, p2 = stats.spearmanr(b[ok], z[ok])
        print(f"  ρ(投影, {lab}) = {r_:+.3f} p={p_:.3g}    ρ(b, {lab}) = {r2:+.3f} p={p2:.3g}")

    rows = {}
    for lab, zs in (("控 目标数", [n_obj[ok]]),
                    ("控 bbox 总面积", [area_sum[ok]]),
                    ("控 目标数+总面积", [n_obj[ok], area_sum[ok]]),
                    ("控 复杂度+‖δ‖+Δego", [n_obj[ok], area_sum[ok], nrm[ok], dego[ok]])):
        r_, p_ = partial_spearman(d[ok], b[ok], zs)
        rows[lab] = {"rho": float(r_), "p": float(p_)}
        print(f"  {lab:24s} ρ={r_:+.3f}  p={p_:.4g}")

    survives = all(v["rho"] < 0 and v["p"] < 0.05 for v in rows.values())
    verdict = ("负相关**扛住全部复杂度控制** ⇒ 不是'可谈论度'伪相关，解耦结论保留"
               if survives else
               "负相关在某项控制下消失 ⇒ 可能是场景复杂度的伪相关，需降级表述")
    print(f"\n[W4 判定] {verdict}")

    out = {"axis": str(args.axis), "layer": L, "n": len(A),
           "raw": {"rho": float(r0), "p": float(p0), "ci95_scene_boot": [lo, hi]},
           "simpson": {"within_scene": {"rho": float(rw), "p": float(pw), "n": len(wi_d)},
                       "between_scene": {"rho": float(rb), "p": float(pb), "n": len(bt_d)},
                       "verdict": simpson},
           "complexity_partial": rows, "verdict": verdict}
    (work / "results" / f"w4_confound{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[W4] wrote results/w4_confound{args.tag}.json")


if __name__ == "__main__":
    main()
