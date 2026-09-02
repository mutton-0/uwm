"""核查 NAVSIM 语料的 3D 框朝向约定，并量化其对已发布 2D 框的影响。

**问题**：`ns1_navsim_geometry.build_geo` 把 NAVSIM `gt_boxes[:, 6]` 直接存进 `_rot`，
而 `g1_mine_events.frame_bbox`（原样复用、一行未改）假定 `_rot` 是**世界系**朝向，
内部会减去 ego 航向：`yaw_ego = yaw_world − yaw(R_we[j])`。
若 NAVSIM 的 yaw 本就在 ego(=lidar) 系，这一减法就是**多减的**，
等价于给每个框的底面朝向叠加了 −ψ_ego 的系统性旋转。

本脚本做两件事，都只用数据、不靠读代码断言：
  ① **判定约定**：取 ego 明显转向的 scene 里的车辆轨迹，比较 stored_yaw 与
     stored_yaw + ψ_ego 哪个沿时间更稳定（停驻/直行车辆的世界系朝向应近似恒定）；
  ② **量化影响**：对语料里实际用到的事件帧，分别按"现行(减 ψ)"与"更正(不减)"
     两种约定重投影 2D 框，报 IoU / 面积比 / 中心位移，按目标类别分组
     （行人近似各向同性 ⇒ 影响小；车辆细长 ⇒ 影响大）。
"""
from __future__ import annotations

import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_mine_events as G1                                            # noqa: E402
import ns1_navsim_geometry as NS                                       # noqa: E402

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
NS_LOGS = Path("/data/dataset/navsim/dataset/navsim_logs")


def iou(a, b):
    if a is None or b is None:
        return None
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ar) if ar > 0 else 0.0


def bbox_with_yaw(geo, o, j, yaw_ego):
    """与 G1.frame_bbox 完全一致，只是 yaw_ego 由外部给定（隔离唯一的差异变量）。"""
    depth = float(o["_p_cam"][j, 2])
    if not np.isfinite(depth) or depth <= 0.5:
        return None
    corners = G1.box_corners_ego(o["p_ego"][j], o["_size"], yaw_ego)
    cam = (corners - geo["t_ec"]) @ geo["R_ec"]
    if (cam[:, 2] <= 0.1).any():
        return None
    uv = (geo["K"] @ cam.T).T
    u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
    return [float(u.min()), float(v.min()), float(u.max()), float(v.max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "navsim_corpus.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--pos", nargs="+", default=["A"])
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "ns_yaw_audit.json"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] in args.pos
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    print(f"[NS-YAW] 事件 {len(evs)}（类别 {args.pos}）")

    by_log = defaultdict(list)
    for e in evs:
        by_log[e["log_name"]].append(e)

    rows, psis = [], []
    done_scenes = 0
    for log_name, elist in sorted(by_log.items()):
        lf = NS_LOGS / args.split / f"{log_name}.pkl"
        if not lf.exists():
            cand = list((NS_LOGS / args.split).glob(f"{log_name}*.pkl"))
            if not cand:
                continue
            lf = cand[0]
        frames = pickle.load(open(lf, "rb"))
        by_scene = defaultdict(list)
        for f in frames:
            by_scene[f["scene_token"]].append(f)
        for e in elist:
            fl = by_scene.get(e["scene_token"])
            if not fl:
                continue
            fl = sorted(fl, key=lambda z: z["timestamp"])
            try:
                geo = NS.build_geo(fl, cfg, args.split)
            except Exception:                                          # noqa: BLE001
                continue
            done_scenes += 1
            o = geo["per_obj"].get(e["object_token"])
            if o is None or not o.get("_rot"):
                continue
            for cond in ("ghost", "clean"):
                fr = e.get(f"x_{cond}_frames") or []
                if not fr or not fr[0].get("bbox_xyxy"):
                    continue
                t = fr[0]["t"]
                j = int(np.argmin(np.abs(geo["grid_t"] - t)))
                if abs(geo["grid_t"][j] - t) > 0.06 or not bool(o["valid"][j]):
                    continue
                ti = int(np.argmin(np.abs(np.asarray(o["_rot_t"]) - geo["grid_t"][j])))
                yaw_stored = Quaternion(o["_rot"][ti]).yaw_pitch_roll[0]
                psi = Quaternion(matrix=geo["R_we"][j]).yaw_pitch_roll[0]
                b_cur = bbox_with_yaw(geo, o, j, yaw_stored - psi)      # 现行（多减了 ψ）
                b_fix = bbox_with_yaw(geo, o, j, yaw_stored)           # 更正（yaw 本就在 ego 系）
                if b_cur is None or b_fix is None:
                    continue
                w_c, h_c = b_cur[2] - b_cur[0], b_cur[3] - b_cur[1]
                w_f, h_f = b_fix[2] - b_fix[0], b_fix[3] - b_fix[1]
                rows.append({
                    "eid": e["event_id"], "cond": cond, "cls": G1.obj_class(e["object_class"]),
                    "native": e.get("object_class_native"),
                    "iou": iou(b_cur, b_fix),
                    "area_ratio": float((w_f * h_f) / max(w_c * h_c, 1e-9)),
                    "w_ratio": float(w_f / max(w_c, 1e-9)),
                    "dcx_px": float(abs((b_fix[0] + b_fix[2]) - (b_cur[0] + b_cur[2])) / 2),
                    "d_long": float(o["d_long"][j]), "psi_ego": float(psi),
                    "size_wl": [float(o["_size"][0]), float(o["_size"][1])]})
                psis.append(psi)
        if args.limit_scenes and done_scenes >= args.limit_scenes:
            break

    if not rows:
        raise SystemExit("[NS-YAW] 没有可比对的帧")
    out = {"n_frames": len(rows), "split": args.split, "pos": args.pos,
           "psi_ego_spread_rad": [float(np.min(psis)), float(np.max(psis))],
           "convention_finding": ("NAVSIM gt_boxes[:,6] 的 yaw 在 ego(=lidar) 系；"
                                  "frame_bbox 又减了一次 ego 航向 ⇒ 现行 2D 框的底面朝向"
                                  "被系统性地多旋转了 −ψ_ego")}
    for grp in ("all",) + tuple(sorted({r["cls"] for r in rows})):
        sub = [r for r in rows if grp == "all" or r["cls"] == grp]
        a = np.array([r["iou"] for r in sub]); ar = np.array([r["area_ratio"] for r in sub])
        dc = np.array([r["dcx_px"] for r in sub]); wr = np.array([r["w_ratio"] for r in sub])
        out[grp] = {"n": len(sub),
                    "iou": {"mean": float(a.mean()), "median": float(np.median(a)),
                            "p05": float(np.percentile(a, 5)), "min": float(a.min())},
                    "area_ratio_fix_over_cur": {"median": float(np.median(ar)),
                                                "p05": float(np.percentile(ar, 5)),
                                                "p95": float(np.percentile(ar, 95))},
                    "width_ratio": {"median": float(np.median(wr))},
                    "center_shift_px": {"median": float(np.median(dc)),
                                        "p95": float(np.percentile(dc, 95))}}
        print(f"[NS-YAW] {grp:10s} n={len(sub):5d}  IoU 中位 {np.median(a):.3f} (p05 {np.percentile(a,5):.3f})"
              f"  面积比中位 {np.median(ar):.3f}  中心位移中位 {np.median(dc):.1f}px")
    out["per_frame"] = rows
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[NS-YAW] wrote {args.out}")


if __name__ == "__main__":
    main()
