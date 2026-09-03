"""遮挡组可视化|把 f3_mask_group 里的**每一个** VRU 都画出来，并叠加实际的灰斑效果。

此前 brake_attr_export 只画了事件的 target 一个框，容易被误读成"另一个行人没标注"。
实际上 nuScenes 两个都标了、遮挡组里也都在 —— 是画少了，不是数据缺。
本模块左图画全部遮挡框（目标绿、同组黄），右图是**实际会喂给模型的遮挡后图像**。
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2, numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
OUT = RES / "figures" / "mask_groups"; NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from f3_occlusion_necessity import occlude                             # noqa: E402


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "T2_main"
    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
    sc = {s["name"]: s for s in nusc.scene}
    man = json.load(open(RES / "f3_candidate_pool.json"))
    evs = {}
    for l in open(ROOT / "variants/n1_d2/mining/events_all.jsonl"):
        e = json.loads(l); evs[e["event_id"]] = e

    d = OUT / tier; d.mkdir(parents=True, exist_ok=True)
    cache, meta = {}, []
    for o in man["tiers"][tier]["events"]:
        e = evs[o["eid"]]; sn = e["scene"] if "scene" in e else e["scene_name"]
        if sn not in cache:
            if len(cache) > 3:
                cache.pop(next(iter(cache)))
            cache[sn] = G1.compute_scene_geometry(nusc, sc[sn], cfg)
        geo = cache[sn]; gt = geo["grid_t"]
        fg = e["x_ghost_frames"][0]
        j = int(np.argmin(np.abs(gt - fg["t"])))
        img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / fg["filename"])), cv2.COLOR_BGR2RGB)

        boxes, miss = [], []
        for g in o["f3_mask_group"]:
            ob = geo["per_obj"].get(g["token"])
            bb = (fg["bbox_xyxy"] if g["is_target"]
                  else (G1.frame_bbox(geo, ob, j) if ob is not None else None))
            if bb is None:
                miss.append(g["token"][:8]); continue
            boxes.append((list(bb), g["is_target"], g))

        occ = img.copy()
        for bb, _, _ in boxes:
            occ = occlude(occ, bb)
        left = img.copy()
        for bb, is_t, g in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
            cv2.rectangle(left, (x0, y0), (x1, y1),
                          (0, 235, 0) if is_t else (255, 210, 0), 3)
            cv2.putText(left, "TARGET" if is_t else "same group",
                        (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 235, 0) if is_t else (255, 210, 0), 2)
        pair = np.hstack([left, occ])
        bar = np.zeros((150, pair.shape[1], 3), np.uint8)
        for i, t in enumerate([
            f"{o['eid']}  [{tier}]   mask group = {o['n_mask_group']} VRU(s)"
            f"{'   MISSING PROJECTION: ' + ','.join(miss) if miss else ''}",
            f"LEFT: all mask boxes drawn (green=event target, yellow=same group)"
            f"   RIGHT: image actually fed to the model (all masked)",
            f"d={o['d_long']}m TTC={o['ttc_s']}s a_req={o['a_req_vru']}"
            f"  vs non-VRU {o['a_non_vru_max']}  bbox={o['bbox_frac_pct']}%",
            f"drawn {len(boxes)}/{o['n_mask_group']} boxes"]):
            cv2.putText(bar, t, (12, 30 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                        (255, 255, 255) if i == 0 else (180, 230, 255), 2)
        outimg = np.vstack([bar, pair])
        p = d / f"{o['eid']}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 84])
        meta.append({"eid": o["eid"], "file": p.name, "n_mask_group": o["n_mask_group"],
                     "n_boxes_drawn": len(boxes), "missing_projection": miss,
                     "boxes": [{"bbox_xyxy": b, "is_target": t, **{k: g[k] for k in
                                ("token", "cat", "s_m", "lat_m", "a_req")}}
                               for b, t, g in boxes]})
        print(f"  {o['eid']:20s} 组 {o['n_mask_group']}  画出 {len(boxes)}"
              f"{'  缺投影: ' + ','.join(miss) if miss else ''}")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {d}  ({len(meta)} 张)")


if __name__ == "__main__":
    main()
