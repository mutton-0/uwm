"""Q3 判别性检验：v_hazard 抓的是"危险"还是"画面中央出现大东西"？

AUC(正例 vs D) 已经控掉了"时间流逝/自车运动"（D 类同间隔、同自车运动统计），
但没控掉目标的**图像几何**：A 类目标入走廊中央、成像更大更居中，D 类偏外围。

两个检验：
  ① 投影值 与 目标 2D bbox 面积 / 离心率 的相关
  ② 按 bbox 面积分层后，AUC(正例 vs D) 是否仍 > 0.5
若 ② 在每一层内仍 >0.5，则"画面中央大目标"解释不了该 AUC。
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
from g3_metrics import (hazard_directions, hazard_directions_supervised, load_cache,  # noqa: E402
                        project, select_peak_layer, subset, time_baseline_basis)


def event_bbox_geometry(nusc, ev):
    """取 ghost 首帧上目标的 2D bbox 面积(px^2) 与中心离心率(归一化)。"""
    from nuscenes.utils.geometry_utils import view_points
    fr = ev["x_ghost_frames"][0]
    sd = nusc.get("sample_data", fr["sd_token"])
    W, H = sd["width"], sd["height"]
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(calib["camera_intrinsic"])
    _, boxes, _ = nusc.get_sample_data(fr["sd_token"])
    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        if ann["instance_token"] != ev["object_token"]:
            continue
        corners = view_points(box.corners(), K, normalize=True)[:2]
        w = float(corners[0].max() - corners[0].min())
        h = float(corners[1].max() - corners[1].min())
        cx = float(corners[0].mean()); cy = float(corners[1].mean())
        ecc = float(np.hypot((cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2)))
        return {"area": w * h, "ecc": ecc, "in_frame": 0 <= cx <= W and 0 <= cy <= H}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="vision_mean")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    mcfg = cfg["metrics"]
    work = Path(cfg["paths"]["work_dir"])
    items = load_cache(work, args.pool_mode)
    splits = json.loads((work / "mining" / "splits.json").read_text())
    est = subset(items, (work / "mining" / "split_estimate.txt").read_text().split())
    truth = subset(items, (work / "mining" / "split_truth.txt").read_text().split())
    S = {k: [e for e in est if e["scene"] in splits["probe_split_scenes"][k]] for k in ("dir", "sel", "test")}

    basis = None
    if mcfg.get("deconfound", "none") == "d_baseline":
        basis, _ = time_baseline_basis(S["dir"], k=int(mcfg.get("deconfound_k", 2)))
    if mcfg.get("direction_method", "pca") == "supervised":
        v_haz, _, _ = hazard_directions_supervised(S["dir"], basis=basis)
    else:
        v_haz, _, _ = hazard_directions(S["dir"], basis=basis)
    peak, _ = select_peak_layer(S["sel"], v_haz, basis=basis)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"], dataroot=cfg["paths"]["nuscenes_root"], verbose=False)
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}

    rows = []
    for e in truth:
        g = event_bbox_geometry(nusc, evmap[e["meta"]["event_id"]])
        if g is None or not np.isfinite(g["area"]):
            continue
        rows.append({"e": e, **g})
    proj = project([r["e"] for r in rows], v_haz, peak, basis=basis)
    area = np.array([r["area"] for r in rows])
    ecc = np.array([r["ecc"] for r in rows])
    pos = np.array([r["e"]["is_positive"] for r in rows])

    print(f"[Q3] L*={peak}  truth 可取 bbox 的事件 {len(rows)}/{len(truth)}"
          f"  ({pos.sum()} 正 / {(~pos).sum()} 负)")
    print(f"[Q3] bbox 面积中位: 正例 {np.median(area[pos]):.0f} px²  D 类 {np.median(area[~pos]):.0f} px²"
          f"   离心率中位: 正例 {np.median(ecc[pos]):.2f}  D 类 {np.median(ecc[~pos]):.2f}")
    ua = stats.mannwhitneyu(area[pos], area[~pos]); ue = stats.mannwhitneyu(ecc[pos], ecc[~pos])
    print(f"[Q3] 几何本身就能分类：AUC(面积)={ua.statistic/(pos.sum()*(~pos).sum()):.3f} (p={ua.pvalue:.2g})，"
          f"AUC(离心率)={ue.statistic/(pos.sum()*(~pos).sum()):.3f} (p={ue.pvalue:.2g})")

    print("\n① 投影值 vs 图像几何的相关")
    for nm, x in (("bbox 面积", area), ("log 面积", np.log10(area + 1)), ("离心率", ecc)):
        r = stats.spearmanr(proj, x)
        print(f"   Spearman(投影, {nm}) = {r.statistic:+.3f}  (p={r.pvalue:.3g})")

    print("\n② 按 bbox 面积分层后的 AUC(正例 vs D)")
    q = np.quantile(area, [0, .25, .5, .75, 1.0])
    overall = stats.mannwhitneyu(proj[pos], proj[~pos])
    print(f"   全体: AUC={overall.statistic/(pos.sum()*(~pos).sum()):.3f} (p={overall.pvalue:.3g})")
    aucs = []
    for i in range(4):
        m = (area >= q[i]) & (area <= q[i + 1] if i == 3 else area < q[i + 1])
        p_, n_ = proj[m & pos], proj[m & ~pos]
        if len(p_) < 5 or len(n_) < 5:
            print(f"   Q{i+1} [{q[i]:.0f},{q[i+1]:.0f}] px²: 样本不足 ({len(p_)}/{len(n_)})")
            continue
        u = stats.mannwhitneyu(p_, n_); a = u.statistic / (len(p_) * len(n_))
        aucs.append(a)
        print(f"   Q{i+1} [{q[i]:6.0f},{q[i+1]:7.0f}] px²: n={len(p_):3d}正/{len(n_):4d}负  "
              f"AUC={a:.3f} (p={u.pvalue:.3g})")
    if aucs:
        print(f"   -> 4 层 AUC 范围 [{min(aucs):.3f}, {max(aucs):.3f}]，"
              f"{'全部 >0.5，几何解释不了' if min(aucs) > 0.5 else '存在 ≤0.5 的层，几何混淆无法排除'}")

    # 面积匹配的配对检验：每个正例配一个面积最接近的 D
    order = np.argsort(area)
    used = set(); pairs = []
    for i in np.where(pos)[0]:
        cands = [j for j in np.where(~pos)[0] if j not in used]
        if not cands:
            break
        j = min(cands, key=lambda j: abs(np.log10(area[j] + 1) - np.log10(area[i] + 1)))
        used.add(j); pairs.append((i, j))
    if len(pairs) > 20:
        dp = np.array([proj[i] - proj[j] for i, j in pairs])
        da = np.array([abs(np.log10(area[i] + 1) - np.log10(area[j] + 1)) for i, j in pairs])
        w = stats.wilcoxon(dp)
        print(f"\n③ 面积匹配配对（{len(pairs)} 对，log10 面积差中位 {np.median(da):.3f} dex）")
        print(f"   配对投影差中位 = {np.median(dp):+.4f}，Wilcoxon p={w.pvalue:.3g}  "
              f"-> {'匹配后仍显著，几何解释不了' if w.pvalue < 0.05 else '匹配后不显著'}")


if __name__ == "__main__":
    main()
