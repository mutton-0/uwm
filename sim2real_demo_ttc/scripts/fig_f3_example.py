"""F 轴示例图：同帧三臂 origin / clean / ctrl + 读数，取自新方法的真实事件。

论文里的 F 轴插图必须是新方法的案例（旧图用的是跨帧 clean，已废）。
三张并排 + 一行读数，直接说明干预是什么、控制臂控的是什么。
"""
from __future__ import annotations
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import cv2, numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                             # noqa: E402
from f3_occlusion_necessity import occlude                              # noqa: E402
from f3_brakefirst import control_boxes_for_group                       # noqa: E402

NUSC = "/data/dataset/nuscenes/v1.0-trainval"
NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", required=True, choices=["nuscenes", "navsim"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--f3", nargs="*", default=[],
                    help="可选：一个或多个 f3 轨迹文件，逐个在脚注里列出其三臂读数。"
                         "给多个时可直观展示同一刺激下不同候选的相反响应。")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    G1.set_include_animal(True)
    cands = json.load(open(RES / args.pool))["candidates"]
    c = next(x for x in cands if x["scene"] == args.scene)
    if args.corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
        geo = G1.compute_scene_geometry(nusc, {s["name"]: s for s in nusc.scene}[args.scene], cfg)
        root = Path(NUSC)
    else:
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"),
                                     resolve=True)
        sp = c.get("split", "test"); cache = defaultdict(list)
        for lf in sorted((NS.NS_ROOT / "navsim_logs" / sp).glob("*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                cache[f["scene_name"]].append(f)
        fl = sorted(cache[args.scene], key=lambda z: z["timestamp"])
        geo = NS.build_geo(fl, cfg, sp); root = Path(NS_BLOBS)

    j = c["frame_idx"]; fr = geo["frames"][j]
    img = cv2.cvtColor(cv2.imread(str(root / fr["filename"])), cv2.COLOR_BGR2RGB)
    boxes = []
    for g in c["f3_mask_group"]:
        o = geo["per_obj"].get(g["token"])
        bb = G1.frame_bbox(geo, o, j) if o is not None else None
        if bb is not None:
            boxes.append(list(bb))
    rng = np.random.default_rng(0)
    cboxes = control_boxes_for_group(img, boxes, rng)

    clean = img.copy()
    for bb in boxes:
        clean = occlude(clean, bb)
    ctrl = img.copy()
    for bb in cboxes:
        ctrl = occlude(ctrl, bb)
    ann = img.copy()
    for bb in boxes:
        x0, y0, x1, y1 = [int(round(v)) for v in bb]
        cv2.rectangle(ann, (x0, y0), (x1, y1), (0, 235, 0), 3)

    reads = []
    for f in args.f3:
        if not (RES / f).exists():
            continue
        d0 = json.load(open(RES / f))
        for e in d0["per_event"]:
            if e["scene"] == args.scene:
                reads.append((d0.get("model", f), e)); break

    H = img.shape[0]
    lab = ["(a) origin  (hazard present)", "(b) clean  (attributed group erased)",
           "(c) ctrl  (equal-area patch elsewhere)"]
    panels = []
    for im, t in zip([ann, clean, ctrl], lab):
        p = im.copy()
        bar = np.zeros((46, p.shape[1], 3), np.uint8)
        cv2.putText(bar, t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        panels.append(np.vstack([bar, p]))
    grid = np.hstack(panels)
    foot = np.zeros((38 + 32 * max(1, len(reads)), grid.shape[1], 3), np.uint8)
    s1 = (f"{args.scene} f{j}   human {c['ego_v0']:.2f} -> {c['ego_vmin']:.2f} m/s"
          f"   a_req {c['a_vru_max']:.2f}   mask group = {len(boxes)}")
    lines2 = [s1]
    for nm, e in reads:
        lines2.append(f"{nm:>10s}   arc_full origin {e['v_origin']:.3f} | clean {e['v_clean']:.3f}"
                      f" | ctrl {e['v_ctrl']:.3f}    b = {e['b']:+.3f}   b_ctrl = {e['b_ctrl']:+.3f}")
    for i, t in enumerate(lines2):
        cv2.putText(foot, t, (10, 28 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255) if i == 0 else (170, 225, 255), 2)
    out = np.vstack([grid, foot])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[FIG] {args.scene}  遮挡 {len(boxes)} / 对照 {len(cboxes)} -> {args.out}")


if __name__ == "__main__":
    main()
