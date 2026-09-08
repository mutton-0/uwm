"""cruise / brake 两组帧的隐藏层激活导出（**不做任何遮挡**）。

与 f_vfaith_extract 的区别：那边是同帧 clean/ghost 配对差，这边是两个**群体**
的均值差。输入就是原图，只跑一次前向。

产出 results/vdecel_acts_{corpus}_{split}_{model}.npz：
    h__L{l} [n, d_l]，以及 kind(cruise/brake) / band / side / v0 / scene
方向在 f_vdecel_direction.py 里按 (side, band) 分组算，不在这里算。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from c_axis_hazard_patch import arc_full, WP_DT                       # noqa: E402

LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LABEL))
    ap.add_argument("--pool", required=True)
    ap.add_argument("--corpus", default="navsim", choices=["nuscenes", "navsim"])
    ap.add_argument("--pool-tag", default="")
    ap.add_argument("--pool-cap", type=int, default=0, help="每 (kind,side,band) 上限，0=不限")
    ap.add_argument("--crop-center-row", type=int, default=560)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    from PIL import Image
    BASE = ("/data/dataset/navsim/dataset/sensor_blobs" if A.corpus == "navsim"
            else "/data/dataset/nuscenes/v1.0-trainval")

    for d in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
        sys.path.insert(0, str(RES / d))
    if A.model == "dd":
        from dd_adapter import DDRunner as Runner
    elif A.model == "ltf":
        from ltf_adapter import LTFRunner as Runner
    else:
        from ddv2_adapter import DDV2Runner as Runner
    if A.crop_center_row:
        import dd_adapter as _DD
        _DD.set_crop_center_row(A.crop_center_row)
    runner = Runner(device=A.device)
    nL = len(runner.sas)

    D = json.load(open(A.pool))
    items = []
    seen = {}
    for kind in ("cruise", "brake"):
        for r in D.get(kind, []):
            if not r.get("filename"):
                continue
            k = (kind, r["side"], r["band"])
            if A.pool_cap and seen.get(k, 0) >= A.pool_cap:
                continue
            seen[k] = seen.get(k, 0) + 1
            items.append({**r, "kind": kind})
    print(f"[vdecel/{LABEL[A.model]}] {len(items)} 帧，{nL} 层", flush=True)

    H = [[] for _ in range(nL)]
    meta = {k: [] for k in ("kind", "band", "side", "v0", "scene", "split")}
    for i, r in enumerate(items):
        try:
            p = r["filename"]
            img = np.asarray(Image.open(p if Path(p).is_absolute()
                                        else str(Path(BASE) / p)).convert("RGB"))
            runner.set_steering(None)
            o = runner.run(img, float(r["v0"]))
        except Exception as e:                                   # noqa: BLE001
            print(f"[vdecel] skip {r['scene']}_{r['frame_idx']}: {e}", flush=True)
            continue
        pooled = o["pooled"]["vision_mean"]
        for l in range(nL):
            H[l].append(np.asarray(pooled[l], np.float32))
        for k in meta:
            meta[k].append(r.get(k))
        if (i + 1) % 100 == 0:
            print(f"[vdecel] {i+1}/{len(items)}", flush=True)

    out = Path(A.out) if A.out else RES / f"vdecel_acts_{A.pool_tag or 'x'}_{A.model}.npz"
    Z = {k: np.array(v) for k, v in meta.items()}
    for l in range(nL):
        Z[f"h__L{l}"] = np.stack(H[l]).astype(np.float32)
    np.savez_compressed(out, **Z)
    import collections
    c = collections.Counter(zip(meta["kind"], meta["side"], meta["band"]))
    print(f"[vdecel] -> {out}  n={len(meta['kind'])}")
    for k, v in sorted(c.items()):
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
