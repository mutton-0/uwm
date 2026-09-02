"""F-3 复核|clean 臂（"危险实体本来就不在场"）里，实体到底可不可见。

`f3_occlusion_necessity.py` 把 clean 臂描述为"危险实体本来就不在场"。
但 clean 帧取自 t_emergence 之前的固定窗口，而 `t_emergence` 标记的是
"进走廊 / TTC 越阈"，**不是"变得可见"**（§FM/A57 已把这条升为纪律）。
本脚本逐事件用挖掘期同一套投影核实 clean / ghost 两帧上实体的可见性与成像面积，
产出 F-3 主判据 b_ghost 究竟在对比什么的直接证据（§FC/A61）。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "variants" / "n1_d2"))
    ap.add_argument("--pos", default="A")
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = str(ROOT / "results" / f"f3_clean_arm_audit{args.tag}.json")

    from nuscenes.nuscenes import NuScenes
    from f3_window_boxes import WindowBoxes
    import g1_mine_events as G1

    W = Path(args.work)
    evs = [json.loads(l) for l in open(W / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == args.pos
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False)
    WB = WindowBoxes(nusc)

    rows = []
    for e in evs:
        try:
            geo = WB.geo(e)
        except Exception:                                              # noqa: BLE001
            continue
        o = geo["per_obj"].get(e["object_token"])
        if o is None:
            continue
        gt = geo["grid_t"]
        jc = int(np.argmin(np.abs(gt - e["x_clean_frames"][0]["t"])))
        jg = int(np.argmin(np.abs(gt - e["x_ghost_frames"][0]["t"])))
        area = lambda b: None if b is None else float((b[2]-b[0]) * (b[3]-b[1]))   # noqa: E731
        bc = G1.frame_bbox(geo, o, jc) if bool(o["visible"][jc]) else None
        bg = G1.frame_bbox(geo, o, jg) if bool(o["visible"][jg]) else None
        rows.append({"eid": e["event_id"], "scene": e["scene_name"],
                     "vis_clean": bool(o["visible"][jc]), "vis_ghost": bool(o["visible"][jg]),
                     "area_clean": area(bc), "area_ghost": area(bg),
                     "d_long_clean": float(o["d_long"][jc]), "d_long_ghost": float(o["d_long"][jg])})

    n = len(rows)
    vc = sum(r["vis_clean"] for r in rows)
    both = [r for r in rows if r["area_clean"] and r["area_ghost"]]
    ra = np.array([r["area_ghost"] / r["area_clean"] for r in both]) if both else np.zeros(0)
    out = {"work": args.work, "pos": args.pos, "n_events": n,
           "n_entity_visible_in_clean_frame": int(vc),
           "frac_entity_visible_in_clean_frame": float(vc / n) if n else None,
           "n_entity_visible_in_ghost_frame": int(sum(r["vis_ghost"] for r in rows)),
           "both_visible": {
               "n": len(both),
               "area_ratio_ghost_over_clean": {
                   "median": float(np.median(ra)) if len(ra) else None,
                   "p25": float(np.percentile(ra, 25)) if len(ra) else None,
                   "p75": float(np.percentile(ra, 75)) if len(ra) else None},
               "d_long_median_clean": float(np.median([r["d_long_clean"] for r in both])) if both else None,
               "d_long_median_ghost": float(np.median([r["d_long_ghost"] for r in both])) if both else None},
           "reading": ("clean 帧上实体已可见的比例越高，b_ghost 就越不是'危险出现引起的响应'，"
                       "而是'危险走近/状态改变引起的响应'；F-3 三态判定的门恰是 b_ghost 的显著性。"),
           "per_event": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[CA] {args.pos} 类 {n} 事件：clean 帧上实体已可见 {vc}/{n} = {vc/n:.1%}；"
          f"两帧都可见 {len(both)} 例，成像面积比 ghost/clean 中位 "
          f"{np.median(ra):.2f}，纵距中位 {out['both_visible']['d_long_median_clean']:.1f} m -> "
          f"{out['both_visible']['d_long_median_ghost']:.1f} m")
    print(f"[CA] wrote {args.out}")


if __name__ == "__main__":
    main()
