"""F-3 三臂 for Alpamayo-R1 / AutoVLA on NAVSIM。

与 `f3_brakefirst.py` 同一套设计（同帧 origin / clean / ctrl、遮挡归因组全体、
等面积对照补丁），只在两处不同：
  1. 跑在 py312 环境（两个 VLA 的依赖在那儿），该环境没有 cv2 —— 图像 IO 用 PIL；
  2. 这两个候选吃**多相机多帧**输入 —— 遮挡必须落在**每一张该目标可见的图**上。

**§FM/A56 补齐（2026-09-07）**：旧版只遮「前视当前帧」，理由写作"与其余四家
干预范围一致"。这个理由是错的：其余四家只吃一张图，遮掉那张 = 目标从输入中消失；
而 VLA 吃 3–4 相机 × 4 帧，只遮 1 张 = 目标在其余 11–15 张里仍然在。
实测（ghost 池 49 事件）：**前视 3 个历史帧里 47 个事件目标仍可见**，侧相机 3–6%。
后果是 VLA 挨的扰动比 TransFuser 系弱一个数量级（干预像素占比 0.04% vs 0.43%），
其 v_faith 因此估不出来（AutoVLA SNR 1.5 vs DDv2 9.0）。
§FM/A56 当年在 nuScenes 路径（f3_occlusion_vla.py）修过，**NAVSIM 路径是后写的，
把同一个洞原样复制了一遍**，而 NAVSIM 恰是左舵池的主体（94%）。
现改为逐 (相机, 帧) 独立投影、可见才遮；**ctrl 臂逐格涂等面积对照框**，
保证遮挡面积在两臂之间可比。

**必须随结果报告的口径限制**：NAVSIM 与 nuScenes 的相机内参、FOV、装车位置都不同
（fx 1545 vs 1266）。这两个候选在 deployment 上的分数含相机几何差异 ——
按用户 2026-09-05 的口径，这属于 benchmark→deployment 域偏移本身，不作校正。
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

# 两个 VLA 的依赖（含 numpy）在这个 venv 里，必须在任何第三方 import 之前挂上
for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
           "/data/Zhengyang/alpamayo/src"):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"
WP_DT = {"alpa": 0.1, "autovla": 0.5}


def occlude(img, bbox):
    """与 f3_occlusion_necessity.occlude 逐行同义（纯 numpy，不依赖 cv2）。"""
    if bbox is None:
        return img
    H, W = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, min(x0, W - 1)); x1 = max(x0 + 1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    out = img.copy()
    out[y0:y1, x0:x1] = img.reshape(-1, 3).mean(0).astype(img.dtype)
    return out


def _ov(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def control_boxes(img, boxes):
    """等面积对照补丁：与所有遮挡框、已放置对照框都不重叠。"""
    H, W = img.shape[:2]
    out = []
    for bb in boxes:
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        cands = []
        for dx in (2 * w, -2 * w, 3 * w, -3 * w, 4 * w, -4 * w, 6 * w, -6 * w):
            cands.append((x0 + dx, y0, x1 + dx, y1))
        for dy in (-2 * h, 2 * h, -3 * h, 3 * h):
            for dx in (0, 2 * w, -2 * w):
                cands.append((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
        for c in cands:
            if c[0] < 0 or c[1] < 0 or c[2] > W or c[3] > H:
                continue
            if any(_ov(c, b) for b in boxes) or any(_ov(c, o) for o in out):
                continue
            out.append(list(c)); break
    return out


def arc_full(wp, dt):
    w = np.asarray(wp, float)[:, :2]
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1).sum() if len(w) > 1 else 0.0
    return float((np.linalg.norm(w[0]) + seg) / (len(w) * dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["alpa", "autovla"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(RES / "autovla_g1_adapter"))
    from PIL import Image
    from vla_navsim_input import NavsimVLAInput

    pool = json.load(open(args.pool))["candidates"]
    vin = NavsimVLAInput()

    # 遮挡框由主环境预先算好（geo 依赖 nuscenes-devkit，py312 没有）
    boxf = RES / f"vla_boxes_{Path(args.pool).stem}.json"
    if not boxf.exists():
        print(f"缺 {boxf}：请先在主环境跑 scripts/vla_export_boxes.py"); return
    BOX = {f"{b['split']}|{b['scene']}|{b['frame_idx']}": b for b in json.load(open(boxf))}

    if args.model == "autovla":
        from autovla_adapter import AutoVLARunner
        runner = AutoVLARunner(device=args.device)

        CAM2KEY = {"CAM_F0": "front_camera", "CAM_L0": "front_left_camera",
                   "CAM_R0": "front_right_camera"}

        def infer(split, scene, j, patched):
            """patched: None 或 {"CAM|k": 该格遮挡图的绝对路径}（全窗口全相机）。"""
            im = vin.images(split, scene, j)
            if im is None:
                return None
            if patched:
                im = {k: list(v) for k, v in im.items()}
                for slot, pth in patched.items():
                    cam, k = slot.split("|")
                    key = CAM2KEY.get(cam)
                    if key in im and int(k) < len(im[key]):
                        im[key][int(k)] = pth
            spd, acc = vin.ego(split, scene, j)
            o = runner.run(None, spd, acc, images=im)
            return np.asarray(o["trajectory"], float)
    else:
        from alpa_navsim_loader import load_navsim
        from alpamayo_runner import AlpamayoRunner
        runner = AlpamayoRunner(device=args.device)

        # camera_indices 排序后 = [0,1,2,6]：槽0=前左(CAM_L0)、槽1=前(CAM_F0)、
        # 槽2=前右(CAM_R0)、槽3(索引6)=前窄，同样复用 CAM_F0 —— **两路前视都要遮**。
        CAM2SLOTS = {"CAM_L0": [0], "CAM_F0": [1, 3], "CAM_R0": [2]}

        def infer(split, scene, j, patched):
            d = load_navsim(vin, split, scene, j)
            if d is None:
                return None
            if patched:
                import torch
                for slot, pth in patched.items():
                    cam, k = slot.split("|"); k = int(k)
                    arr = np.array(Image.open(pth).convert("RGB"))
                    for sl in CAM2SLOTS.get(cam, []):
                        if sl < d["image_frames"].shape[0] and k < d["image_frames"].shape[1]:
                            d["image_frames"][sl, k] = torch.from_numpy(arr).permute(2, 0, 1)
            return np.asarray(runner.infer(scene, 0.0, data=d).traj, float)

    tmp = Path("/tmp/claude-1001/-data-ruolin-uwm-sim2real-demo-ttc/007773d2-b773-4b39-be8d-122fceddf6fe/scratchpad/vla_arms")
    tmp.mkdir(parents=True, exist_ok=True)
    recs, skip = [], defaultdict(int)
    for i, c in enumerate(pool):
        sp = c.get("split", "test")
        key = f"{sp}|{c['scene']}|{c['frame_idx']}"
        if key not in BOX or not BOX[key]["boxes"]:
            skip["no_box"] += 1; continue
        boxes = BOX[key]["boxes"]
        win = BOX[key].get("window")
        if not win:
            skip["no_window"] += 1; continue
        # 逐 (相机, 帧) 渲染 clean / ctrl：该格有框才渲，没框就用原图（不外推）
        pc_map, pt_map, n_slot, n_box = {}, {}, 0, 0
        for slot, w in sorted(win.items()):
            if not w["boxes"]:
                continue
            src = Path(NS_BLOBS) / sp / w["path"]
            if not src.exists():
                skip["slot_no_image"] += 1; continue
            im0 = np.array(Image.open(src).convert("RGB"))
            cb_s = control_boxes(im0, w["boxes"])
            if not cb_s:
                skip["slot_no_ctrl"] += 1; continue
            cl, ct = im0.copy(), im0.copy()
            for b in w["boxes"]:
                cl = occlude(cl, b)
            for b in cb_s:                      # ctrl 臂逐格涂等面积对照框
                ct = occlude(ct, b)
            tag = slot.replace("|", "_")
            pcs = tmp / f"{args.model}_{tag}_c.jpg"
            pts = tmp / f"{args.model}_{tag}_t.jpg"
            Image.fromarray(cl).save(pcs, quality=95)
            Image.fromarray(ct).save(pts, quality=95)
            pc_map[slot] = str(pcs); pt_map[slot] = str(pts)
            n_slot += 1; n_box += len(w["boxes"])
        if not pc_map:
            skip["no_box"] += 1; continue
        cb = boxes                              # 仅用于记录 n_ctrl，与旧版同义

        to = infer(sp, c["scene"], c["frame_idx"], None)
        tc = infer(sp, c["scene"], c["frame_idx"], pc_map)
        tt = infer(sp, c["scene"], c["frame_idx"], pt_map)
        if to is None or tc is None or tt is None:
            skip["infer"] += 1; continue
        dt = WP_DT[args.model]
        vo, vc, vt = arc_full(to, dt), arc_full(tc, dt), arc_full(tt, dt)
        recs.append({"scene": c["scene"], "eid": key, "split": sp,
                     "frame_idx": c["frame_idx"], "city": c.get("city"),
                     "n_mask": len(boxes), "n_ctrl": len(cb),
                     "n_slots_masked": n_slot, "n_boxes_masked": n_box,
                     "ego_v0": c["ego_v0"], "a_req": c["a_vru_max"],
                     "v_origin": vo, "v_clean": vc, "v_ctrl": vt,
                     "b": vc - vo, "b_ctrl": vt - vo,
                     "traj_origin": to.tolist(), "traj_clean": tc.tolist(),
                     "traj_ctrl": tt.tolist()})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(pool)}] 成功 {len(recs)}", flush=True)

    b = np.array([r["b"] for r in recs]) if recs else np.array([])
    print(f"\n[F3VLA/{args.model}] n={len(recs)}  跳过 {dict(skip)}")
    if len(b):
        print(f"  b 均值 {b.mean():+.4f}   方向一致(b>0) {(b>0).sum()}/{len(b)}")
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "pool": args.pool, "corpus": "navsim",
         "primary_readout": "arc_full", "wp_dt": WP_DT[args.model],
         "note": "遮挡仅作用于前视当前帧，与其余四家干预范围一致；"
                 "NAVSIM 相机内参/FOV 与 nuScenes 不同，属域偏移本身，不作校正",
         "n_events": len(recs), "skipped": dict(skip), "per_event": recs},
        indent=2, ensure_ascii=False))
    print(f"[F3VLA] wrote {args.out}")


if __name__ == "__main__":
    main()
