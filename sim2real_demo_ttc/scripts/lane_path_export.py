"""在途判据复判|导出 ghost 帧图片供人工确认规则筛得对不对。

标注：行人到**自车真实未来路径**的最短横向距离（判据 A，工单指定的主口径）、
旧的瞬时朝向横向偏移、自车在窗口内的航向变化、时空最近接近（判据 C）。
不做四臂、不重跑推理 —— 这一轮只是给人看规则。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2, numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
OUT = RES / "figures" / "lane_path_filter_check"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"


def annotate(img, title, lines, bbox, color):
    im = img.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    cv2.rectangle(im, (x0, y0), (x1, y1), color, 3)
    bar = np.zeros((34 * (len(lines) + 1) + 10, im.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (12, 30 + 34 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.78, (180, 230, 255), 2)
    return np.vstack([bar, im])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lpf = {r["eid"]: r for r in json.load(open(RES / "lane_path_filter.json"))["per_event"]}
    evs = {}
    for l in open(ROOT / "variants" / "n1_d2" / "mining" / "events_all.jsonl"):
        e = json.loads(l)
        if e["event_id"] in lpf:
            evs[e["event_id"]] = e

    ok = [r for r in lpf.values() if r["status"] == "ok"]
    kept_all = sorted([r for r in ok if r["keep_2m"]], key=lambda z: z["d_long_ghost_m"])
    kept = []                                    # 按距离分层抽 5 个，覆盖不同距离
    for lo, hi in ((0, 15), (15, 22), (22, 30), (30, 40), (40, 200)):
        g = [r for r in kept_all if lo <= r["d_long_ghost_m"] < hi]
        if g:
            kept.append(min(g, key=lambda z: z["lat_to_real_path_m"]))
    drop = [r for r in ok if not r["keep_2m"]]
    # 筛掉组：按距离分层各取一个，尽量覆盖不同距离与不同转角
    drop_sel = []
    for lo, hi in ((0, 15), (15, 25), (25, 35), (35, 200)):
        g = sorted([r for r in drop if lo <= r["d_long_ghost_m"] < hi],
                   key=lambda z: -abs(z["ego_heading_change_deg"]))
        if g:
            drop_sel.append(g[0])
    g0 = sorted([r for r in drop if abs(r["ego_heading_change_deg"]) < 5],
                key=lambda z: -z["lat_to_real_path_m"])
    if g0:
        drop_sel.append(g0[0])                       # 直行但人在远侧的例子

    meta = {"rule": json.load(open(RES / "lane_path_filter.json"))["rule"],
            "primary_criterion": "A：行人在 ghost 帧的位置 → 自车真实未来路径的最短横向距离 < 2.0 m",
            "note": "只导 ghost 帧，未跑推理；本轮仅供人工确认筛选规则",
            "groups": {}}

    for grp, sel, col in (("kept", kept, (0, 235, 0)), ("dropped", drop_sel, (60, 60, 255))):
        meta["groups"][grp] = []
        for r in sel:
            ev = evs[r["eid"]]
            fg = ev["x_ghost_frames"][0]
            bb = fg["bbox_xyxy"]
            img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / fg["filename"])), cv2.COLOR_BGR2RGB)
            verdict = "KEEP (on ego's real path)" if grp == "kept" else "DROP (not on real path)"
            im = annotate(img, f"{r['eid']}   {verdict}", [
                f"lateral dist to REAL future path = {r['lat_to_real_path_m']:.2f} m"
                f"   (corridor 2.00 m)",
                f"OLD criterion (instantaneous heading) = {abs(r['lat_instantaneous_m']):.2f} m"
                f"  <- this is what mining used",
                f"VRU longitudinal dist = {r['d_long_ghost_m']:.1f} m"
                f" | ego heading change over window = {r['ego_heading_change_deg']:+.1f} deg",
                f"closest ego-VRU approach in space-time = {r['min_approach_m']:.2f} m"
                f" | ego v0 = {r['ego_speed_ghost_mps']:.1f} m/s"
                f" | human driver dv = {r['ego_delta_v_mps']:+.2f} m/s",
                f"path: realized {r['arc_realized_m']:.1f} m + straight extension 20 m"
                f"   closest point on {'EXTENSION (extrapolated)' if r['closest_on_extension'] else 'REALIZED path'}",
            ], bb, col)
            p = OUT / f"{grp}__{r['eid']}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            meta["groups"][grp].append({
                "eid": r["eid"], "scene": r["scene"], "file": p.name,
                "lat_to_real_path_m": r["lat_to_real_path_m"],
                "lat_instantaneous_m_OLD": r["lat_instantaneous_m"],
                "d_long_ghost_m": r["d_long_ghost_m"],
                "ego_heading_change_deg": r["ego_heading_change_deg"],
                "min_approach_m": r["min_approach_m"],
                "ego_speed_ghost_mps": r["ego_speed_ghost_mps"],
                "ego_delta_v_mps": r["ego_delta_v_mps"],
                "oncoming": r["oncoming"], "arc_realized_m": r["arc_realized_m"],
                "closest_on_extension": r["closest_on_extension"],
                "ghost_file": fg["filename"]})
            print(f"[{grp}] {r['eid']:20s} {'ext' if r['closest_on_extension'] else 'real'} "
                  f"到真实路径 {r['lat_to_real_path_m']:5.2f}m "
                  f"旧 {abs(r['lat_instantaneous_m']):4.2f}m  d {r['d_long_ghost_m']:5.1f}m "
                  f"转角 {r['ego_heading_change_deg']:+6.1f}°")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[LPF-EXPORT] wrote {OUT} 共 {len(list(OUT.glob('*.jpg')))} 张")


if __name__ == "__main__":
    main()
