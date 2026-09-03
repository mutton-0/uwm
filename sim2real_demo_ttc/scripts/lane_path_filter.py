"""语料在途判据修正|用自车**真实未来轨迹**判断行人是否在路径上，替换瞬时朝向判据。

工单：2026-09-03（用户已决定推翻 LTF「盲动」结论，根因见 `ltf_sanity_check_report_zh.md`）。

## 旧判据错在哪

挖矿期用的是 `g1_mine_events.compute_scene_geometry` 里的

    in_corridor = (|y| < corridor_half_width_m) & (0 < x < 50)      # y = p_ego[:,1]

`p_ego` 是**该帧自车瞬时朝向**下的坐标 ⇒ `y` 是"相对当前车头方向"的横向偏移。
**它假设自车会沿当前朝向直行。** 弯道上该假设失效：`scene-0433_000_A` 的行人
|y| = 1.56 m、纵向 44.8 m，看着"几乎正前方"，但道路向右急弯，
自车实际路径根本不经过他（人还站在挡墙后的人行道上）。
于是大量**路侧行人**被算成"在途危险"。这与 §FE/A69 是同一类坑：
**用瞬时朝向做纵/横分解，在远距离 + 弯道下会漂移。**

## 新判据

以自车**实际走过的未来轨迹**为参考轴：

1. 取 ghost 帧时刻 $t_g$ 之后自车真实位置序列（世界系，挖矿期算 TTC 时已在追踪，直接复用 `ego_xyz`）；
2. **窗口**：一直取到自车累计弧长 ≥ 行人纵向距离 + 5 m 为止，上限 10 s。
   理由：判据要回答"自车后来有没有从行人身边经过"，所以路径**必须至少延伸到行人所在的纵深**，
   否则最短距离会被路径长度本身截断（远处行人会因为路径没走到而假性"不在途"）。
   固定 3–5 s 在慢速/远距离事件上做不到这一点，故改用弧长条件 + 时间上限。
3. 把该路径与行人在 $t_g$ 的位置一起转到 $t_g$ 的 ego 系（纯刚体变换，不改变任何距离）；
4. **横向距离** = 行人点到这条折线（逐段线段）的最短欧氏距离；
5. **走廊**：半宽 2.0 m 为主口径（半车宽≈1 m + 1 m 缓冲，总宽 4 m），1.5 m 作敏感性；
6. **同向约束**：把目标自身速度投影到路径最近点处的切向。若投影 ≤ −2 m/s
   （即以车辆量级的速度**迎着**自车行进方向运动）判为对向，剔除。
   行人横穿速度小、且以横向为主，不会被该阈值误伤。

**保留** = 最短横向距离 < 阈值 **且** 非对向 **且** 路径足够长。
`path_insufficient`（自车停住 / scene 结束，路径没走到行人纵深）单列，不计入"保留"。

本模块**不重新挖矿、不修改任何事件字段**，只对现有事件重新打标。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                              # noqa: E402


def point_to_polyline(p, poly):
    """点到折线的最短距离，及最近点处的段索引与该段单位切向。p/poly 为 2D。"""
    best = (np.inf, 0, np.array([1.0, 0.0]))
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best[0]:
            tang = ab / (np.linalg.norm(ab) + 1e-12)
            best = (d, i, tang)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "variants" / "n1_d2"))
    ap.add_argument("--pos", default="A")
    ap.add_argument("--cfg", default=str(ROOT / "configs" / "n1_d2.yaml"))
    ap.add_argument("--corridor", type=float, default=2.0)
    ap.add_argument("--corridor-tight", type=float, default=1.5)
    ap.add_argument("--oncoming-mps", type=float, default=-2.0)
    ap.add_argument("--pad-m", type=float, default=5.0)
    ap.add_argument("--max-window-s", type=float, default=10.0)
    ap.add_argument("--extend-m", type=float, default=20.0,
                    help="沿窗口最后一帧自车**真实航向**把路径向前直线延伸的长度")
    ap.add_argument("--classes", default="",
                    help="逗号分隔的 object_class 子串过滤（如 pedestrian,cyclist）；空=不过滤")
    ap.add_argument("--restrict-to-f3", action="store_true", default=False,
                    help="只复判 F-3 LTF 那轮用过的事件（A 类默认行为）")
    ap.add_argument("--out", default=str(RES / "lane_path_filter.json"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    cfg = OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
    scene_by_name = {s["name"]: s for s in nusc.scene}

    evs = [json.loads(l) for l in open(Path(args.work) / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == args.pos]
    if args.restrict_to_f3:
        keep_ids = {r["eid"] for r in json.load(open(RES / "f3_occlusion_ltf.json"))["per_event"]}
        evs = [e for e in evs if e["event_id"] in keep_ids]
    if args.classes:
        pats = [c.strip() for c in args.classes.split(",") if c.strip()]
        evs = [e for e in evs if any(c in str(e.get("object_class", "")) for c in pats)]
    # 复判需要 ghost 帧有投影框（与 F-3 的可用性条件一致）
    evs = [e for e in evs if e.get("x_ghost_frames") and e["x_ghost_frames"][0].get("bbox_xyxy")]
    print(f"[LPF] 复判事件 {len(evs)}  (pos={args.pos}, classes={args.classes or 'all'})")

    geo_cache = {}
    rows = []
    for i, ev in enumerate(evs):
        sn = ev["scene_name"]
        if sn not in geo_cache:
            if len(geo_cache) > 3:
                geo_cache.pop(next(iter(geo_cache)))
            geo_cache[sn] = G1.compute_scene_geometry(nusc, scene_by_name[sn], cfg)
        geo = geo_cache[sn]
        o = geo["per_obj"].get(ev["object_token"])
        gt = geo["grid_t"]
        t_g = ev["x_ghost_frames"][0]["t"]
        j = int(np.argmin(np.abs(gt - t_g)))
        if o is None or not bool(o["valid"][j]):
            rows.append({"eid": ev["event_id"], "scene": sn, "status": "no_track"}); continue

        p_ped_ego = np.asarray(o["p_ego"][j], float)[:2]          # 行人在 t_g 的 ego 系位置
        d_long = float(o["d_long"][j]); lat_inst = float(o["lat"][j])

        # ---- 未来自车路径（世界系 -> t_g 的 ego 系）----
        need = abs(d_long) + args.pad_m
        exyz = geo["ego_xyz"]; R = geo["R_we"][j]
        k, arc = j, 0.0
        while k + 1 < len(gt) and (gt[k + 1] - t_g) <= args.max_window_s and arc < need:
            arc += float(np.linalg.norm(exyz[k + 1, :2] - exyz[k, :2])); k += 1
        poly_w = exyz[j:k + 1, :2]
        rel = poly_w - exyz[j, :2]
        poly = (rel @ R[:2, :2])                                   # world -> ego(t_g)
        n_real, dur = len(poly), float(gt[min(k, len(gt) - 1)] - t_g)

        # ---- 沿最后一帧自车真实航向做直线延伸（§2.3）----
        # 用 R_we[k] 的车头方向而不是末端几点拟合：后者在自车近乎停住时方向会抖。
        if args.extend_m > 0 and n_real >= 1:
            fwd_w = geo["R_we"][k][:2, 0]                           # 世界系下 k 帧车头方向
            fwd = fwd_w @ R[:2, :2]                                 # -> ego(t_g)
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            n_seg = max(2, int(args.extend_m / 1.0))
            ext = poly[-1] + np.outer(np.linspace(0, args.extend_m, n_seg)[1:], fwd)
            poly = np.vstack([poly, ext])
        n_pts = len(poly)
        reach = arc + (args.extend_m if n_real >= 1 else 0.0)

        if n_pts < 2:
            rows.append({"eid": ev["event_id"], "scene": sn, "status": "path_insufficient",
                         "arc_travelled_m": arc, "arc_needed_m": need, "window_s": dur,
                         "n_path_pts": n_pts, "d_long_ghost_m": d_long,
                         "lat_instantaneous_m": lat_inst,
                         "ego_speed_ghost_mps": float(geo["ego_speed"][j])}); continue

        # 自车在该窗口内的航向变化（弯道诊断）：路径首尾切向夹角
        def _tang(a, b):
            v = poly[min(b, len(poly) - 1)] - poly[a]
            return v / (np.linalg.norm(v) + 1e-12)
        yaw_chg = float(np.degrees(np.arctan2(
            np.cross(_tang(0, 3), _tang(max(0, len(poly) - 4), len(poly) - 1)),
            np.dot(_tang(0, 3), _tang(max(0, len(poly) - 4), len(poly) - 1)))))

        # ---- 时空最近接近（判据 C）：同一时刻自车与行人的真实最小距离 ----
        # o["p_ego"][k] 就是 k 时刻行人相对自车原点的位置 ⇒ 其模长即该时刻真实间距。
        # 这一条**同时考虑了行人自己的运动**，故能捕捉"行人随后走进车辆路径"的情形，
        # 而判据 A（t_g 位置 vs 未来路径）按定义捕捉不到。
        seg_v = o["valid"][j:k + 1]
        seg_p = np.asarray(o["p_ego"][j:k + 1], float)[:, :2]
        if seg_v.any():
            dd_t = np.linalg.norm(seg_p[seg_v], axis=1)
            min_approach = float(dd_t.min())
            i_min = int(np.argmin(dd_t))
            lat_at_min = float(abs(seg_p[seg_v][i_min][1]))
        else:
            min_approach, lat_at_min = float("nan"), float("nan")

        dmin, seg, tang = point_to_polyline(p_ped_ego, poly)
        # 路径没走满时仍可**确定性保留**：既然已走的这段就进了走廊，再走下去也改变不了"在途"这一事实。
        # 只有"已走完的部分没进走廊、但路径还没延伸到行人纵深"才是真的判不了。
        # 最近点是否落在**延伸段**上（延伸是外推假设，需单独标记）
        on_ext = bool(seg >= max(0, n_real - 1)) if n_real >= 1 else True
        if reach < need and dmin >= args.corridor:
            rows.append({"eid": ev["event_id"], "scene": sn, "status": "path_insufficient",
                         "arc_travelled_m": arc, "arc_needed_m": need, "window_s": dur,
                         "n_path_pts": n_pts, "d_long_ghost_m": d_long,
                         "lat_to_real_path_partial_m": dmin,
                         "ego_heading_change_deg": yaw_chg,
                         "reach_m": reach,
                         "lat_instantaneous_m": lat_inst,
                         "ego_speed_ghost_mps": float(geo["ego_speed"][j])}); continue
        v_obj = np.asarray(o["v_obj_ego"][j], float)[:2]           # ego(t_g) 系下目标速度
        v_along = float(v_obj @ tang)
        oncoming = bool(v_along <= args.oncoming_mps)

        # ---- 人类司机在该窗口内是否减速（交叉验证用）----
        es = geo["ego_speed"][j:k + 1]
        v0 = float(geo["ego_speed"][j])
        rows.append({"eid": ev["event_id"], "scene": sn, "status": "ok",
                     "lat_to_real_path_m": dmin, "lat_instantaneous_m": lat_inst,
                     "d_long_ghost_m": d_long, "oncoming": oncoming, "v_along_path_mps": v_along,
                     "keep_2m": bool(dmin < args.corridor and not oncoming),
                     "keep_1p5m": bool(dmin < args.corridor_tight and not oncoming),
                     "window_s": dur, "arc_travelled_m": arc, "n_path_pts": n_pts, "n_real_pts": n_real,
                     "path_complete": bool(reach >= need),
                     "arc_realized_m": arc, "reach_m": reach,
                     "closest_on_extension": on_ext,
                     "min_approach_m": min_approach, "lat_at_min_approach_m": lat_at_min,
                     "keep_approach_2m": bool(min_approach < args.corridor and not oncoming),
                     "keep_approach_1p5m": bool(min_approach < args.corridor_tight and not oncoming),
                     "ego_heading_change_deg": yaw_chg,
                     "ego_speed_ghost_mps": v0, "ego_speed_min_future_mps": float(es.min()),
                     "ego_delta_v_mps": float(es.min() - v0),
                     "object_class": ev.get("object_class")})
        if (i + 1) % 60 == 0:
            print(f"  {i+1}/{len(evs)}")

    ok = [r for r in rows if r["status"] == "ok"]
    ins = [r for r in rows if r["status"] == "path_insufficient"]
    nt = [r for r in rows if r["status"] == "no_track"]
    kept = [r for r in ok if r["keep_2m"]]
    drop = [r for r in ok if not r["keep_2m"]]
    n = len(rows)
    print(f"\n=== 复判结果（共 {n}）===")
    print(f"  可判定 ok            {len(ok):4d} ({len(ok)/n*100:5.1f}%)")
    print(f"  路径不足 (含自车停住) {len(ins):4d} ({len(ins)/n*100:5.1f}%)")
    print(f"  无轨迹                {len(nt):4d}")
    print(f"  --- 主口径 2.0 m ---")
    print(f"  保留                 {len(kept):4d} ({len(kept)/n*100:5.1f}% of {n})")
    print(f"  筛掉                 {len(drop):4d} ({len(drop)/n*100:5.1f}% of {n})")
    print(f"  敏感性 1.5 m 保留     {sum(r['keep_1p5m'] for r in ok):4d}")
    print(f"  其中因对向被剔        {sum(r['oncoming'] for r in ok):4d}")
    kx = [r for r in ok if r["keep_2m"] and r.get("closest_on_extension")]
    print(f"  其中最近点落在延伸段  {len(kx):4d}（外推所致，需单独看）")
    ka = [r for r in ok if r.get("keep_approach_2m")]
    print(f"  保留组 scene 数       {len({r['scene'] for r in kept})}")
    print(f"  --- 判据 C（时空最近接近 < 2.0 m，含行人自身运动）---")
    print(f"  保留                 {len(ka):4d} ({len(ka)/n*100:5.1f}% of {n})，scene 数 {len({r['scene'] for r in ka})}")
    print(f"  敏感性 1.5 m          {sum(r.get('keep_approach_1p5m', False) for r in ok):4d}")

    out = {"rule": {"corridor_half_width_m": args.corridor,
                    "corridor_tight_m": args.corridor_tight,
                    "oncoming_along_path_mps": args.oncoming_mps,
                    "window": f"弧长 >= d_long + {args.pad_m} m，上限 {args.max_window_s} s",
                    "extend_m": args.extend_m,
                    "extend_dir": "窗口最后一帧自车真实航向（R_we[k] 车头方向），直线外推",
                    "reference_axis": "自车真实未来轨迹（world ego_xyz），非瞬时朝向"},
           "counts": {"total": n, "ok": len(ok), "path_insufficient": len(ins),
                      "no_track": len(nt), "keep_2m": len(kept), "drop_2m": len(drop),
                      "keep_1p5m": int(sum(r["keep_1p5m"] for r in ok)),
                      "keep_approach_2m": int(sum(r.get("keep_approach_2m", False) for r in ok)),
                      "keep_approach_1p5m": int(sum(r.get("keep_approach_1p5m", False) for r in ok)),
                      "keep_approach_scenes": len({r["scene"] for r in ok if r.get("keep_approach_2m")}),
                      "oncoming": int(sum(r["oncoming"] for r in ok)),
                      "keep_scenes": len({r["scene"] for r in kept}),
                      "keep_via_extension": int(sum(1 for r in kept if r.get("closest_on_extension")))},
           "per_event": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[LPF] wrote {args.out}")


if __name__ == "__main__":
    main()
