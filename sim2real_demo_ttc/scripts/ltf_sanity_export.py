"""LTF 盲动结论的实例核查|导出具体案例的 clean/ghost/occ/ctrl 四帧图。

工单：2026-09-03（LTF sanity check）。用户质疑「LTF 是验证过的 checkpoint，
不该完全看不见危险实体，怀疑是管线问题」。本模块**不重跑推理**，
只把已落盘 `f3_occlusion_ltf.json` 里挑出的事件，用**与主实验逐字相同**的
`occlude` / `control_box` / `read` 重建四臂图像并标注数值，供目视检查。

关键：`control_box(img, bbox, rng)` 内部并不消费 `rng`（纯几何镜像），
故对照框可精确复现，不需要重放随机数序列。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import control_box, occlude                # noqa: E402

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
OUT = RES / "figures" / "ltf_sanity_check"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"
WORK = ROOT / "variants" / "n1_d2"


def read(fn):
    return cv2.cvtColor(cv2.imread(str(Path(NUSC) / fn)), cv2.COLOR_BGR2RGB)


def annotate(img, title, lines, bbox=None, cbox=None, color=(0, 255, 0)):
    im = img.copy()
    if bbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), color, 3)
    if cbox is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in cbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 160, 0), 3)
    bar = np.zeros((34 * (len(lines) + 1) + 10, im.shape[1], 3), np.uint8)
    cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    for k, t in enumerate(lines):
        cv2.putText(bar, t, (12, 30 + 34 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (180, 230, 255), 2)
    return np.vstack([bar, im])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {r["eid"]: r for r in json.load(open(RES / "f3_occlusion_ltf.json"))["per_event"]}
    au = {r["eid"]: r for r in json.load(open(RES / "f3_clean_arm_audit_g1.json"))["per_event"]}
    evs = {}
    for l in open(WORK / "mining" / "events_all.jsonl"):
        e = json.loads(l)
        if e["event_id"] in rec:
            evs[e["event_id"]] = e

    order = sorted(rec.values(), key=lambda r: r["b_ghost"])
    react = order[:3]                                      # 减速最明显
    flat = sorted(rec.values(), key=lambda r: abs(r["b_ghost"]))[:3]   # 几乎没反应
    meta = {"note": "纯图像重建，未重跑推理；数值取自 f3_occlusion_ltf.json",
            "b_ghost = v_ghost - v_clean": "负 = 看到危险后规划速度下调",
            "groups": {}}

    for grp, sel in (("reacted", react), ("flat", flat)):
        meta["groups"][grp] = []
        for r in sel:
            eid = r["eid"]; ev = evs[eid]
            fc, fg = ev["x_clean_frames"][0], ev["x_ghost_frames"][0]
            bb = fg["bbox_xyxy"]
            ic, ig = read(fc["filename"]), read(fg["filename"])
            cb = control_box(ig, bb, np.random.default_rng(0))
            a = au.get(eid, {})
            imgs = {
                "clean": annotate(ic, f"{eid}  [clean] baseline frame",
                                  [f"v_clean = {r['v_clean']:.3f} m/s",
                                   ("NOTE: entity ALREADY VISIBLE in clean frame (contaminated)" if a.get("vis_clean")
                                    else "entity genuinely not visible in clean frame")]),
                "ghost": annotate(ig, f"{eid}  [ghost] hazard frame",
                                  [f"v_ghost = {r['v_ghost']:.3f} m/s",
                                   f"b_ghost = {r['b_ghost']:+.4f} m/s",
                                   f"VRU dist {a.get('d_long_ghost', float('nan')):.1f} m,"
                                   f" bbox {r['bbox_area_frac']*100:.3f}% of frame"], bbox=bb),
                "occ": annotate(occlude(ig, bb), f"{eid}  [occ] VRU covered by grey patch",
                                [f"v_occ = {r['v_occ']:.3f} m/s",
                                 f"d_occ = v_occ - v_ghost = {r['v_occ']-r['v_ghost']:+.4f} m/s"],
                                bbox=bb, color=(255, 60, 60)),
                "ctrl": annotate(occlude(ig, cb), f"{eid}  [ctrl] equal-area patch elsewhere",
                                 [f"v_ctrl = {r['v_ctrl']:.3f} m/s",
                                  f"d_ctrl = {r['v_ctrl']-r['v_ghost']:+.4f} m/s"],
                                 bbox=bb, cbox=cb),
            }
            files = {}
            for k, im in imgs.items():
                p = OUT / f"{grp}__{eid}__{k}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 88])
                files[k] = p.name
            # 行人框区域放大图（目视确认灰斑是否真盖住了人）
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
            pad = max(60, int(0.6 * max(x1 - x0, y1 - y0)))
            X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
            X1, Y1 = min(ig.shape[1], x1 + pad), min(ig.shape[0], y1 + pad)
            crop = np.hstack([cv2.resize(ig[Y0:Y1, X0:X1], (320, 320)),
                              cv2.resize(occlude(ig, bb)[Y0:Y1, X0:X1], (320, 320))])
            p = OUT / f"{grp}__{eid}__zoom_ghost_vs_occ.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            files["zoom"] = p.name
            meta["groups"][grp].append({
                "eid": eid, "scene": r["scene"],
                "v_clean": r["v_clean"], "v_ghost": r["v_ghost"],
                "v_occ": r["v_occ"], "v_ctrl": r["v_ctrl"],
                "b_ghost": r["b_ghost"], "d_occ": r["v_occ"] - r["v_ghost"],
                "d_ctrl": r["v_ctrl"] - r["v_ghost"],
                "bbox_xyxy": bb, "control_bbox_xyxy": cb,
                "bbox_area_frac": r["bbox_area_frac"],
                "bbox_px": int((bb[2]-bb[0])*(bb[3]-bb[1])),
                "d_long_ghost_m": a.get("d_long_ghost"),
                "entity_visible_in_clean_frame": a.get("vis_clean"),
                "clean_file": fc["filename"], "ghost_file": fg["filename"],
                "files": files})
            print(f"[{grp}] {eid}  b_ghost {r['b_ghost']:+.4f}  "
                  f"d {a.get('d_long_ghost', float('nan')):.1f}m  框 {int((bb[2]-bb[0])*(bb[3]-bb[1]))}px")
    (OUT / "meta_export_v2.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n[LTF-SANITY] wrote {OUT}  共 {len(list(OUT.glob('*.jpg')))} 张图")


if __name__ == "__main__":
    main()
