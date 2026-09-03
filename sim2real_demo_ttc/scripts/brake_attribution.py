"""刹车归因|先抠出所有刹车 case，再判定"这次刹车里行人占多大比例"。

工单：2026-09-03（用户提出的框架，替代此前的排除链）。

## 为什么这个框架更好

此前是一串**排除条件**（在途 / 可归因 / 距离上限 / 速度下限），每条都对，但彼此独立、
阈值各拍各的。改成**归因**后：

* **在途判据被自动包含**：行人不在自车真实路径的走廊里 ⇒ 它的刹车需求 $a=0$ ⇒ 占比 0 ⇒ 自动排除。
* **"有更近/更大障碍"被自动包含**：护栏、前车的 $a$ 更大 ⇒ 行人占比被摊薄到阈值以下。
* **距离与自车速度被自动包含**：$a \propto v^2/d$，太远或自车太慢时行人的 $a$ 本就趋于 0。

## 刹车需求怎么算

对 ghost 帧上每个候选目标 $i$（在自车**真实未来路径**的走廊内、位于前方）：

$$a_i = \frac{\max(v^{close}_i, 0)^2}{2\,\max(d_i - d_{safe},\ \epsilon)}$$

$v^{close}_i$ = 自车与该目标沿路径方向的接近速度；$d_i$ = 沿路径的纵向距离；
$d_{safe}$ = 2 m 停车余量。这是"要在它前面停住所需的匀减速度"，
是交通安全学里最标准的紧迫度量，同时编码了距离、相对速度与自车速度。

**行人占比** $= a_{VRU} / \sum_j a_j$（主口径），另给 $a_{VRU}/\max_j a_j$ 作敏感性。

## 已知局限（必须写明）

nuScenes **不标注交通信号灯**。因此"为红灯刹车、路上恰好没有其他标注目标"这种情形，
行人会被算成 100% 占比。本模块用 `stopped_long_s`（自车停住并持续的时长）作为
红灯/排队的**代理指标**单独报出，不并入占比公式 —— 因为它不是几何量，混进去会污染定义。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from lane_path_filter import point_to_polyline                         # noqa: E402

D_SAFE = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["A", "D", "D2b", "D2c"])
    ap.add_argument("--classes", default="pedestrian,cyclist,bicycle,motorcycle")
    ap.add_argument("--corridor", type=float, default=2.0)
    ap.add_argument("--min-rel-decel", type=float, default=0.30)
    ap.add_argument("--min-abs-decel", type=float, default=0.5)
    ap.add_argument("--extend-m", type=float, default=20.0)
    ap.add_argument("--max-window-s", type=float, default=10.0)
    ap.add_argument("--out", default=str(RES / "brake_attribution.json"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
    sc = {s["name"]: s for s in nusc.scene}
    pats = [c.strip() for c in args.classes.split(",") if c.strip()]

    evs = [json.loads(l) for l in open(ROOT / "variants/n1_d2/mining/events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] in args.pools
           and any(c in str(e.get("object_class", "")) for c in pats)
           and e.get("x_ghost_frames") and e["x_ghost_frames"][0].get("bbox_xyxy")]
    print(f"[ATTR] 候选事件 {len(evs)}（{'+'.join(args.pools)} 池的行人/骑行类）")

    cache, out, n_brake = {}, [], 0
    for i, e in enumerate(evs):
        sn = e["scene_name"]
        if sn not in cache:
            if len(cache) > 3:
                cache.pop(next(iter(cache)))
            cache[sn] = G1.compute_scene_geometry(nusc, sc[sn], cfg)
        geo = cache[sn]; gt = geo["grid_t"]
        j = int(np.argmin(np.abs(gt - e["x_ghost_frames"][0]["t"])))
        tgt = geo["per_obj"].get(e["object_token"])
        if tgt is None or not bool(tgt["valid"][j]):
            continue

        # ---- 第一步：这是不是一次刹车 ----
        exyz = geo["ego_xyz"]; R = geo["R_we"][j]
        d_v = float(tgt["d_long"][j])
        need = abs(d_v) + 5.0
        k, arc = j, 0.0
        while k + 1 < len(gt) and (gt[k+1] - gt[j]) <= args.max_window_s and arc < need:
            arc += float(np.linalg.norm(exyz[k+1, :2] - exyz[k, :2])); k += 1
        es = geo["ego_speed"][j:k+1]
        if len(es) < 2:
            continue
        v0 = float(geo["ego_speed"][j]); vmin = float(es.min())
        dv = vmin - v0
        # 观测到的实际减速度：从 ghost 帧到达到最低速所用的时间
        i_min = int(np.argmin(es))
        t_to_min = float(gt[j + i_min] - gt[j]) if i_min > 0 else 0.0
        a_obs = (abs(dv) / t_to_min) if t_to_min > 0.2 else float("nan")
        rel = abs(dv) / max(v0, 0.1)
        is_brake = (dv < -args.min_abs_decel) and (rel > args.min_rel_decel)
        # 红灯/排队代理：停住(<0.5m/s)并持续的时长
        stopped = float(np.sum(es < 0.5) * np.median(np.diff(gt[j:k+1]))) if len(es) > 1 else 0.0
        if not is_brake:
            continue
        n_brake += 1

        # ---- 第二步：路径与走廊 ----
        poly = (exyz[j:k+1, :2] - exyz[j, :2]) @ R[:2, :2]
        if len(poly) >= 1 and args.extend_m > 0:
            fwd = geo["R_we"][k][:2, 0] @ R[:2, :2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            n_seg = max(2, int(args.extend_m))
            poly = np.vstack([poly, poly[-1] + np.outer(
                np.linspace(0, args.extend_m, n_seg)[1:], fwd)])
        if len(poly) < 2:
            continue

        # ---- 第三步：逐目标算刹车需求 ----
        def demand(o, idx):
            p = np.asarray(o["p_ego"][idx], float)[:2]
            dmin, seg, tang = point_to_polyline(p, poly)
            if dmin >= args.corridor:
                return 0.0, dmin, 0.0
            # 沿路径的弧长位置 = 该目标最近点之前的折线长度
            s = float(np.linalg.norm(np.diff(poly[:seg+1], axis=0), axis=1).sum()) if seg > 0 else 0.0
            v_obj_along = float(np.asarray(o["v_obj_ego"][idx], float)[:2] @ tang)
            v_close = v0 - v_obj_along
            a = max(v_close, 0.0) ** 2 / (2.0 * max(s - D_SAFE, 0.5))
            return a, dmin, s

        a_v, lat_v, s_v = demand(tgt, j)
        others = []
        for tok, o in geo["per_obj"].items():
            if tok == e["object_token"] or not bool(o["valid"][j]):
                continue
            a, dmin, s = demand(o, j)
            if a > 0:
                others.append((a, o["cat"], round(s, 1), round(dmin, 2)))
        others.sort(reverse=True)
        # F-3 遮挡组：走廊内**全部 VRU 类**实例（含目标本身）——横向位置相近的一群人必须一起遮，
        # 否则遮掉一个、剩下的还在，必然测出假 FAIL（用户 2026-09-03 指出）。
        mask_group = [{"token": e["object_token"], "cat": tgt["cat"],
                       "s_m": round(s_v, 1), "lat_m": round(lat_v, 2),
                       "a_req": round(a_v, 3), "is_target": True}]
        for tok, o in geo["per_obj"].items():
            if tok == e["object_token"] or not bool(o["valid"][j]):
                continue
            if not o["cat"].startswith(("human.", "vehicle.bicycle", "vehicle.motorcycle")):
                continue
            a, dmin, s_o = demand(o, j)
            if a > 0:                                  # a>0 <=> 在走廊内且在前方
                mask_group.append({"token": tok, "cat": o["cat"], "s_m": round(s_o, 1),
                                   "lat_m": round(dmin, 2), "a_req": round(a, 3),
                                   "is_target": False})
        mask_group.sort(key=lambda z: z["s_m"])
        tot = a_v + sum(x[0] for x in others)
        share = (a_v / tot) if tot > 1e-9 else 0.0
        amax = max([a_v] + [x[0] for x in others])
        out.append({"eid": e["event_id"], "scene": sn, "pool": e["event_type"],
                    "object_class": e.get("object_class"),
                    "d_long_ghost_m": d_v, "lat_to_real_path_m": lat_v,
                    "ego_v0_mps": v0, "ego_vmin_mps": vmin, "ego_delta_v_mps": dv,
                    "rel_decel": rel, "stopped_long_s": stopped,
                    "a_obs_mps2": a_obs, "t_to_min_s": t_to_min,
                    "a_req_vru": a_v, "a_req_total": tot, "a_req_max": amax,
                    "vru_share": share, "vru_share_of_max": (a_v / amax) if amax > 1e-9 else 0.0,
                    "explained_ratio": (a_v / a_obs) if (a_obs and np.isfinite(a_obs) and a_obs > 1e-6) else 0.0,
                    "vru_class_share": ((a_v + sum(x[0] for x in others if x[1].startswith(("human.", "vehicle.bicycle", "vehicle.motorcycle")))) / tot) if tot > 1e-9 else 0.0,
                    "n_competitors": len(others),
                    "f3_mask_group": mask_group, "n_mask_group": len(mask_group),
                    "top_competitors": [{"a_req": round(x[0], 3), "cat": x[1],
                                         "s_m": x[2], "lat_m": x[3]} for x in others[:4]]})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(evs)}  刹车 case {n_brake}")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sh = np.array([o["vru_share"] for o in out])
    print(f"\n[ATTR] 刹车 case {len(out)} / {len(evs)}")
    print(f"  行人占比 分位 p50/p75/p90/p95: {np.round(np.percentile(sh,[50,75,90,95]),3)}")
    for t in (0.4, 0.5, 0.6, 0.8):
        g = [o for o in out if o["vru_share"] > t]
        print(f"  单实例占比 > {t}: {len(g):4d} 事件 / {len({o['scene'] for o in g}):3d} scene")
    for t in (0.6,):
        g = [o for o in out if o["vru_class_share"] > t]
        print(f"  VRU **类别**合计占比 > {t}: {len(g):4d} 事件 / {len({o['scene'] for o in g}):3d} scene")
    print("  再要求行人的刹车需求能解释观测减速（explained_ratio）：")
    for er in (0.2, 0.3, 0.5):
        g = [o for o in out if o["vru_share"] > 0.6 and o["explained_ratio"] > er]
        gc = [o for o in out if o["vru_class_share"] > 0.6 and o["explained_ratio"] > er]
        print(f"    >{er}: 单实例 {len(g):3d}/{len({o['scene'] for o in g}):2d}sc   "
              f"类别合计 {len(gc):3d}/{len({o['scene'] for o in gc}):2d}sc")
    print(f"[ATTR] wrote {args.out}")


if __name__ == "__main__":
    main()
