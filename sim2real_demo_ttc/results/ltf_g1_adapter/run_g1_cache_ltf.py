"""把 G1 语料喂给 LTF，缓存 8 层可读表征（与 DiffusionDrive 侧同一份刺激集、同一套口径）。

用法：
  PY=/home/mut0/.conda/envs/simscale/bin/python
  $PY results/ltf_g1_adapter/run_g1_cache_ltf.py --types A D2a D2b D2c --device cuda:1
产物：variants/n1_d2/ltf_cache/<event_id>.npz（schema 与 dd_cache 逐字段一致）
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import cv2

R = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
sys.path.insert(0, str(R / "ltf_g1_adapter"))
sys.path.insert(0, str(R / "diffusiondrive_g1_adapter"))
from ltf_adapter import LTFRunner, bbox_to_tokens          # noqa: E402


def read_rgb(root, fn):
    img = cv2.imread(str(Path(root) / fn))
    if img is None:
        raise FileNotFoundError(f"{root}/{fn}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--types", nargs="+", default=["A", "D2a", "D2b", "D2c"])
    ap.add_argument("--events", default="", help="事件 id 清单；给定时忽略 --types")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--crop-center-row", type=int, default=0,
                    help="4:1 裁剪的竖直中心 = 相机主点行；0 = 沿用默认(nuScenes 450)。NAVSIM 用 560（§NS/A46）")
    args = ap.parse_args()
    if getattr(args, "crop_center_row", 0):
        import dd_adapter as _DD; _DD.set_crop_center_row(args.crop_center_row)
        print(f"[crop] CROP_CENTER_ROW -> {args.crop_center_row}")

    work = Path(args.work); out_dir = work / "ltf_cache"; out_dir.mkdir(exist_ok=True)
    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D", "D2a", "D2b", "D2c", "D2bV", "D2cV")
               if (work / "mining" / f"matched_{t}.txt").exists()}
    if args.events:
        want = set(Path(args.events).read_text().split())
        keep = [e for e in events if e["event_id"] in want]
    else:
        keep = [e for e in events if e["event_type"] in args.types
                and not (e["event_type"] in matched and e["event_id"] not in matched[e["event_type"]])]
    if args.limit:
        keep = keep[: args.limit]
    print(f"[LTF-G1] {len(keep)} 事件待缓存 -> {out_dir}")

    runner = LTFRunner(device=args.device)
    t0, done, fail = time.time(), 0, []
    for i, ev in enumerate(keep):
        p = out_dir / f"{ev['event_id']}.npz"
        if p.exists():
            continue
        try:
            reg = None
            for f_ in ev["x_ghost_frames"]:
                if f_.get("bbox_xyxy"):
                    reg = bbox_to_tokens(f_["bbox_xyxy"], f_["im_wh"]); break
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            store, extra = {}, {}
            for cond in ("clean", "ghost"):
                per_pool, spds, trajs = {}, [], []
                for f_ in ev[f"x_{cond}_frames"]:
                    r = runner.run(read_rgb(args.nuscenes_root, f_["filename"]), anchor, reg)
                    spds.append(r["commanded_speed"]); trajs.append(r["trajectory"])
                    for pool, v in r["pooled"].items():
                        per_pool.setdefault(pool, []).append(v)
                for pool, lst in per_pool.items():
                    for l in range(len(lst[0])):
                        store[f"{cond}/{pool}/L{l}"] = np.stack([x[l] for x in lst]).astype(np.float32)
                extra[f"commanded_speed_{cond}"] = np.array(spds, np.float32)
                extra[f"traj_{cond}"] = np.stack(trajs).astype(np.float32)
            meta = {k: v for k, v in ev.items() if k != "ttc_curve"}
            meta["_n_region_tokens"] = len(reg or []); meta["_prompt_anchor_speed"] = anchor
            np.savez_compressed(p, meta=json.dumps(meta, ensure_ascii=False), **store, **extra)
            done += 1
        except Exception as exc:                                        # noqa: BLE001
            fail.append({"event_id": ev["event_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[LTF-G1] FAIL {ev['event_id']}: {exc}")
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"[LTF-G1] {i+1}/{len(keep)} done={done} fail={len(fail)} "
                  f"{el:.0f}s ({el/max(1,done):.2f}s/event)", flush=True)
    (work / "results" / "ltf_g1_cache_report.json").write_text(json.dumps(
        {"n_events": len(keep), "cached": done, "failed": fail, "types": args.types,
         "elapsed_s": time.time() - t0}, indent=2, ensure_ascii=False))
    print(f"[LTF-G1] 完成 {done}/{len(keep)}，失败 {len(fail)}")


if __name__ == "__main__":
    main()
