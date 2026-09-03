"""事件类型图库|每个挖矿事件类型抽若干个，导出 ghost 帧标注图供人工查看。

类型定义取自 `configs/n1_d2.yaml` 的 `mining.event_*` 段，见各子目录的 README。
纯读图 + 画框，不跑推理、不改任何事件字段。
"""
from __future__ import annotations

import json, sys
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
OUT = RES / "figures" / "event_types"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"

DEFS = {
    "A":   ("VRU emergence", "POSITIVE",
            "VRU enters corridor, TTC < 5s (1.0s window)"),
    "B":   ("close cut-in", "POSITIVE",
            "d_long <= 25m, TTC < 6s, lateral velocity toward ego lane -- the lead-brake corpus"),
    "C":   ("generic TTC drop", "POSITIVE",
            "TTC drops >= 1.5s within 1s and ends < 5s (catch-all)"),
    "D":   ("harmless appearance", "NEGATIVE",
            "never enters corridor AND own TTC > 6s throughout (max 2 per scene)"),
    "D2a": ("parked / stationary vehicle", "GEOMETRY-BALANCED NEG",
            "large and central target, but harmless by state (parked/stopped)"),
    "D2b": ("distant large target", "GEOMETRY-BALANCED NEG",
            "top 3 by apparent area -- deliberately geometry-matched to positives"),
    "D2c": ("in corridor but not urgent", "FALSIFICATION CONTROL",
            "enters corridor but TTC > 8s throughout"),
    "D2aP":("D2a placebo", "CONTROL",
            "window shifted 2.5s earlier so both frames precede corridor entry"),
    "D2ctxP":("D2c context placebo", "CONTROL", "window shifted 2.5s earlier"),
}


def annotate(img, title, lines, bbox, color):
    im = img.copy()
    if bbox:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), color, 3)
    bar = np.zeros((32 * (len(lines) + 1) + 8, im.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (12, 28 + 32 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (180, 230, 255), 2)
    return np.vstack([bar, im])


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    evs = [json.loads(l) for l in open(ROOT / "variants" / "n1_d2" / "mining" / "events_all.jsonl")]
    by = defaultdict(list)
    for e in evs:
        if e.get("x_ghost_frames") and e["x_ghost_frames"][0].get("bbox_xyxy"):
            by[e["event_type"]].append(e)

    OUT.mkdir(parents=True, exist_ok=True)
    idx = ["# 挖矿事件类型图库\n",
           "每个子目录 = 一个事件类型，取自 `configs/n1_d2.yaml` 的 `mining.event_*`。\n",
           "图为该事件的 **ghost 帧**（危险帧），绿框=正例类 / 红框=负例·对照类，框内是该事件的目标。\n",
           "\n| 类型 | 含义 | 角色 | 判据 | 库内总数 | 本次抽样 |\n| --- | --- | --- | --- | --- | --- |\n"]
    meta_all = {}
    for t, evl in sorted(by.items(), key=lambda kv: -len(kv[1])):
        name, role, rule = DEFS.get(t, (t, "?", "?"))
        d = OUT / t; d.mkdir(exist_ok=True)
        # 按纵向距离分层抽样，覆盖近/中/远
        evl2 = sorted(evl, key=lambda e: (e.get("d_long_at_emergence") or 1e9))
        pick = [evl2[int(i * (len(evl2) - 1) / max(n_per - 1, 1))] for i in range(min(n_per, len(evl2)))]
        col = (0, 235, 0) if role == "POSITIVE" else (60, 60, 255)
        recs = []
        for e in pick:
            fg = e["x_ghost_frames"][0]; bb = fg["bbox_xyxy"]
            im = cv2.imread(str(Path(NUSC) / fg["filename"]))
            if im is None:
                continue
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ttc = e.get("ttc_at_emergence"); mt = e.get("min_ttc_1s")
            out = annotate(img, f"[{t}] {name} ({role})   {e['event_id']}", [
                f"rule: {rule}",
                f"class = {e.get('object_class','?')}",
                f"d_long = {e.get('d_long_at_emergence', float('nan')):.1f} m"
                f" | lat (OLD instantaneous) = {e.get('lat_at_emergence', float('nan')):.2f} m"
                f" | ego v = {e.get('ego_speed_mps', float('nan')):.1f} m/s",
                f"TTC@emergence = {ttc if ttc is None else round(float(ttc),1)} s"
                f" | min_ttc_1s = {mt if mt is None else round(float(mt),1)} s"
                f" | night={e.get('is_night')} rain={e.get('is_rain')}",
            ], bb, col)
            p = d / f"{e['event_id']}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
            recs.append({"event_id": e["event_id"], "scene": e["scene_name"], "file": p.name,
                         "object_class": e.get("object_class"),
                         "d_long_at_emergence": e.get("d_long_at_emergence"),
                         "lat_at_emergence_OLD": e.get("lat_at_emergence"),
                         "ttc_at_emergence": ttc, "min_ttc_1s": mt,
                         "ego_speed_mps": e.get("ego_speed_mps")})
        (d / "meta.json").write_text(json.dumps(
            {"type": t, "name": name, "role": role, "rule": rule,
             "n_total_in_corpus": len(evl), "n_sampled": len(recs), "events": recs},
            indent=2, ensure_ascii=False))
        meta_all[t] = {"name": name, "role": role, "n": len(evl)}
        idx.append(f"| **{t}** | {name} | {role} | {rule} | {len(evl)} | {len(recs)} |\n")
        print(f"[{t:7s}] {name:16s} {role:12s} 库内 {len(evl):5d}  导出 {len(recs)}")
    idx.append("\n> 注：`lat (OLD instantaneous)` 是**旧的瞬时朝向横向偏移**，"
               "已证实在弯道下失真（见 `results/lane_path_filter_report_zh.md`）。\n")
    (OUT / "README.md").write_text("".join(idx))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
