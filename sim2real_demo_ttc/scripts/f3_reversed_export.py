"""反向案例出图|b_model 与 b_GT 反号的事件，四模型数值一并标注。

反向 = $b_{model} < 0$（本池 $b^{GT}$ 13/13 全正）⇒ "遮掉危险后模型反而减速"。
图像与遮挡组对所有模型相同（同一帧、同一遮挡组），故一个 scene 一张图，
标注里并排列出四个模型各自的 b 与共同的 b_GT。
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
OUT = RES / "figures" / "f3_reversed"; NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from f3_occlusion_necessity import occlude                             # noqa: E402
from f3_brakefirst import control_boxes_for_group                      # noqa: E402

K = "arc_full"
MODELS = [("SimLingo", "simlingo"), ("LTF", "ltf"), ("DD", "dd"), ("DDv2", "ddv2")]


def main():
    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    G1.set_include_animal(True)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
    scmap = {s["name"]: s for s in nusc.scene}
    pool = {c["scene"]: c for c in json.load(open(RES / "brake_first_pool_final.json"))["candidates"]}
    D = {}
    for name, m in MODELS:
        d = json.load(open(RES / f"f3_gt_axis_full_{m}.json"))
        D[name] = {r["scene"]: r for r in d["per_event"]}
    scenes = sorted(D["SimLingo"])
    rev = {s: [n for n, _ in MODELS if D[n][s][f"b_model__{K}"] < 0] for s in scenes}
    todo = [s for s in scenes if rev[s]]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"至少一个模型反向的 scene: {len(todo)}/{len(scenes)}")

    rng = np.random.default_rng(0); meta = []
    for s in todo:
        c = pool[s]; geo = G1.compute_scene_geometry(nusc, scmap[s], cfg); j = c["frame_idx"]
        img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / geo["frames"][j]["filename"])),
                           cv2.COLOR_BGR2RGB)
        boxes = []
        for g in c["f3_mask_group"]:
            o = geo["per_obj"].get(g["token"])
            bb = G1.frame_bbox(geo, o, j) if o is not None else None
            if bb is not None:
                boxes.append(list(bb))
        cb = control_boxes_for_group(img, boxes, rng)
        clean = img.copy(); ctrl = img.copy()
        for b in boxes:
            clean = occlude(clean, b)
        for b in cb:
            ctrl = occlude(ctrl, b)
        left = img.copy()
        for b in boxes:
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            cv2.rectangle(left, (x0, y0), (x1, y1), (0, 235, 0), 3)
        b_gt = D["SimLingo"][s][f"b_gt__{K}"]
        pair = np.hstack([left, clean, ctrl])
        bar = np.zeros((190, pair.shape[1], 3), np.uint8)
        lines = [f"{s}   frame {j}   b_GT(arc_full) = {b_gt:+.4f}   "
                 f"reversed for: {', '.join(rev[s])}",
                 f"LEFT origin (mask group outlined, n={len(boxes)})   "
                 f"MID clean (hazard masked)   RIGHT ctrl (equal-area patch elsewhere)",
                 "   ".join(f"{n}: b={D[n][s][f'b_model__{K}']:+.4f}"
                            f"{'  <REV>' if D[n][s][f'b_model__{K}']<0 else ''}"
                            for n, _ in MODELS),
                 f"human {c['ego_v0']:.2f} -> {c['ego_vmin']:.2f} m/s   a_obs {c['a_obs']}   "
                 f"a_req {c['a_vru_max']}   VRU {c['lead_vru']['cat'].split('.')[-1]} "
                 f"@{c['lead_vru']['s_m']}m   night={pool[s].get('is_night','?')}"]
        for i, t in enumerate(lines):
            cv2.putText(bar, t, (12, 32 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.82,
                        (255, 255, 255) if i == 0 else (180, 230, 255), 2)
        p = OUT / f"{len(rev[s])}rev__{s}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(np.vstack([bar, pair]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        meta.append({"scene": s, "file": p.name, "n_reversed": len(rev[s]),
                     "reversed_models": rev[s], "b_gt_arc_full": b_gt,
                     "b_model": {n: D[n][s][f"b_model__{K}"] for n, _ in MODELS},
                     "ego_v0": c["ego_v0"], "ego_vmin": c["ego_vmin"], "a_obs": c["a_obs"],
                     "a_req": c["a_vru_max"], "n_mask": len(boxes),
                     "lead_vru": c["lead_vru"]})
        print(f"  {s:14s} 反向 {len(rev[s])} 个模型: {rev[s]}")
    (OUT / "meta.json").write_text(json.dumps(
        {"readout": K, "definition": "反向 = b_model < 0（b_GT 13/13 全正）",
         "events": sorted(meta, key=lambda z: -z["n_reversed"])}, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}  ({len(meta)} 张)")


if __name__ == "__main__":
    main()
