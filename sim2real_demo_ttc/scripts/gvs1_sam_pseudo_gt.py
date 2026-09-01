"""G-VS 第一步|用 SAM 生成分割伪 GT（六候选共用的客观 ground truth）。

工单：docs/g_vs_f3_simplified_axes_workorder.md §1。

**任务定义（预注册，必须先讲清楚，否则 mIoU 不可解释）**：
SAM 的 automatic mask generator 产出的是**无类别的实例掩码**，没有语义标签。
要把它变成一个良定义的密集预测任务，必须给出一个**客观、可复现**的标签规则。
本轮取**二类「物体性（object-ness）」分割**：

    class 1 (object)     ：被 SAM 掩码覆盖，且该掩码面积 < 画幅的 `--max-area-frac`
    class 0 (background) ：其余（无掩码，或属于路面/天空/建筑这类超大区域）

**为什么是这个规则而不是别的**：
  * 它**只用 SAM**，不引入任何人工语义判据 —— 这正是本工单要摆脱"危险"这个构念的动机；
  * 面积阈值把"路面/天空/墙面"这类 stuff 区域与"车/人/杆/牌"这类 thing 区域分开，
    是 SAM 自动掩码的标准用法，阈值本身随结果落盘、可复现；
  * 二类任务的 mIoU 良定义、跨候选可比，且随机初始化对照（selectivity）有明确含义。

**它不是什么**：这不是语义分割（没有类别），也不是实例分割（不区分个体）。
G-VS 问的是"模型的内部表征里，**哪里有物体**这件事能不能被线性读出"，
不是"模型认得出这是行人还是锥桶"。后者留给 Future Work 的 G-VL。

伪 GT 落盘为**原图分辨率的 uint8 掩码**，各候选按自己的 token 网格再做下采样，
故同一份伪 GT 被六个候选共用，网格差异不进入伪 GT 本身。
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import cv2

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
SAM_CKPT = "/data/ruolin/uwm/external/ckpts/sam_vit_b_01ec64.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--types", nargs="+", default=["A"])
    ap.add_argument("--frame", default="ghost", choices=["ghost", "clean"])
    ap.add_argument("--max-area-frac", type=float, default=0.05,
                    help="掩码面积占画幅比例上限；超过者判为 stuff（路面/天空/建筑），不算 object")
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    work = Path(args.work)
    out_dir = Path(args.out) if args.out else work / "sam_gt"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(args.device).eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side,
                                    pred_iou_thresh=0.86, stability_score_thresh=0.9,
                                    min_mask_region_area=200)

    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] in args.types and e[f"x_{args.frame}_frames"]]
    if args.limit:
        evs = evs[: args.limit]
    print(f"[GVS-SAM] {len(evs)} 帧待处理 -> {out_dir}")

    stats, done = [], 0
    for i, ev in enumerate(evs):
        p = out_dir / f"{ev['event_id']}.png"
        if p.exists():
            done += 1; continue
        fn = ev[f"x_{args.frame}_frames"][0]["filename"]
        img = cv2.imread(str(Path(args.nuscenes_root) / fn))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, Wd = img.shape[:2]
        with torch.no_grad():
            masks = gen.generate(img)
        lim = args.max_area_frac * H * Wd
        obj = np.zeros((H, Wd), np.uint8)
        n_obj = 0
        for m in masks:
            if m["area"] < lim:
                obj[m["segmentation"]] = 1
                n_obj += 1
        cv2.imwrite(str(p), obj)
        stats.append({"eid": ev["event_id"], "n_masks": len(masks), "n_object_masks": n_obj,
                      "object_frac": float(obj.mean())})
        done += 1
        if (i + 1) % 50 == 0:
            print(f"[GVS-SAM] {i+1}/{len(evs)} done={done}", flush=True)

    if stats:
        fr = np.array([s["object_frac"] for s in stats])
        rep = {"n_frames": done, "frame": args.frame, "types": args.types,
               "max_area_frac": args.max_area_frac, "points_per_side": args.points_per_side,
               "sam_ckpt": "sam_vit_b_01ec64",
               "object_pixel_frac": {"mean": float(fr.mean()),
                                     "p10_p50_p90": np.percentile(fr, [10, 50, 90]).round(4).tolist()},
               "n_masks_per_image_median": float(np.median([s["n_masks"] for s in stats])),
               "per_frame": stats}
        (work / "results" / "sam_pseudo_gt_report.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"[GVS-SAM] object 像素占比 均值 {fr.mean():.3f}  "
              f"分位 {np.percentile(fr,[10,50,90]).round(3).tolist()}")
        print(f"[GVS-SAM] 每图掩码数中位 {np.median([s['n_masks'] for s in stats]):.0f}")
    print(f"[GVS-SAM] 完成 {done}")


if __name__ == "__main__":
    main()
