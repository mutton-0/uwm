"""合并 NAVSIM test + trainval 成最终 deployment 池，并按传感器可得性过滤。

## 两套并行维护
  vp   仅 us-nv-las-vegas-strip + us-pa-pittsburgh-hazelwood
       —— **注意：这条理由已随 P-1 作废。** 现在 benchmark/deployment 按左右舵划分，
       本脚本产出的池实测 100% 左舵，按 P-1 属 **benchmark** 侧；文件名里的 deploy 是遗留。
       原始理由（留档）：nuScenes 只采了波士顿+新加坡，故城市集合与本池零重叠，
          跨语料差异才是真域偏移，不是"同城不同语料"的混合物。
  all  不限城市（对照，用来量化限城这件事本身的代价）

## 传感器可得性
挖矿只用元数据（位姿+3D框），全部 1310 个 trainval log 都能扫；
但 F/G/C 三轴要把像素喂进模型，只有下载了 sensor blobs 的 log 才可用。
本脚本按**磁盘上实际存在的 log 目录**过滤（不靠 split 名单推断），并记录缺口。
"""
from __future__ import annotations
import argparse, json, pickle, glob
from collections import Counter
from pathlib import Path

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
BLOBS = Path("/data/dataset/navsim/dataset/sensor_blobs")


def frame_index():
    """(split, scene, frame_idx) -> CAM_F0 相对路径。用于**逐帧**校验图像是否真的在盘上
    —— 只看 log 目录存在是不够的：rsync 中途/部分缺失都会让目录在而图不在，
    实测导致 F-3 在第 5 个事件处 cv2 崩溃。"""
    out = {}
    for sp in ("test", "trainval"):
        for lf in sorted(glob.glob(f"/data/dataset/navsim/dataset/navsim_logs/{sp}/*.pkl")):
            by = {}
            for f in pickle.load(open(lf, "rb")):
                by.setdefault(f["scene_name"], []).append(f)
            for sn, fl in by.items():
                fl.sort(key=lambda z: z["timestamp"])
                for i, f in enumerate(fl):
                    out[(sp, sn, i)] = f["cams"]["CAM_F0"]["data_path"]
    return out


def scene_meta():
    """(split, scene) -> (city, log)。

    **必须用复合键**：scene_name 跨 split 不唯一 —— test 的 2026 个 scene 名里
    1334 个（65.8%）在 trainval 里也存在，但指向完全不同的 log 与数据
    （log-0001-scene-0001 在 test 是 2021.05.25 那条，在 trainval 是 2021.05.12 那条）。
    用裸 scene_name 做键会张冠李戴，实测让限城池里混进波士顿事件。
    """
    out = {}
    for sp in ("test", "trainval"):
        for lf in sorted(glob.glob(f"/data/dataset/navsim/dataset/navsim_logs/{sp}/*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                out.setdefault((sp, f["scene_name"]), (f["map_location"], f["log_name"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True,
                    help="要合并的池文件名。每项写成 `文件名:split`（如 a.json:test）")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--tag", required=True, help="vp 或 all")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = scene_meta()
    fidx = frame_index()

    seen, keep, drop = set(), [], Counter()
    for spec in args.pools:
        pf, _, sp = spec.partition(":")
        assert sp in ("test", "trainval"), f"必须指定 split：{spec}"
        d = json.load(open(RES / pf))
        for c in d["candidates"]:
            if c["a_vru_max"] < 0.4:
                continue
            key = (sp, c["scene"], c["frame_idx"])
            if key in seen:
                drop["重复"] += 1
                continue
            seen.add(key)
            city, log = meta.get((sp, c["scene"]), ("?", "?"))
            c = dict(c, city=city, log_name=log, split=sp,
                     uid=f"{sp}:{c['scene']}:{c['frame_idx']}")
            rel = fidx.get((sp, c["scene"], c["frame_idx"]))
            if rel is None or not (BLOBS / sp / rel).exists():
                drop["图像不在盘上"] += 1
                continue
            keep.append(c)

    cc = Counter(c["city"] for c in keep); sp = Counter(c["split"] for c in keep)
    out = {"scenario": args.scenario, "tag": args.tag, "corpus": "navsim",
           "sources": args.pools, "n_events": len(keep),
           "n_scenes": len({(c["split"], c["scene"]) for c in keep}),
           "n_logs": len({c["log_name"] for c in keep}),
           "city_counts": dict(cc), "split_counts": dict(sp),
           "dropped": dict(drop), "candidates": keep}
    p = Path(args.out or RES / f"deploy_pool_{args.scenario}_{args.tag}.json")
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[POOL] {args.scenario}/{args.tag}: {len(keep)} 事件 / "
          f"{out['n_scenes']} scene / {out['n_logs']} log")
    print(f"        城市 {dict(cc)}")
    print(f"        来源 {dict(sp)}   丢弃 {dict(drop)}")
    print(f"        -> {p}")


if __name__ == "__main__":
    main()
