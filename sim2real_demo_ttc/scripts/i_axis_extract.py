"""T-I|在域配对帧上抽取两个模型的逐层表征(v_domain / D_L / t-SNE 的共同原料)。

--model dd        DiffusionDrive: 8 层 encoder SelfAttention,池化 vision_mean / all_mean
                  (需 simscale 环境: /home/mut0/.conda/envs/simscale/bin/python)
--model simlingo  SimLingo:      24 层 decoder,池化 vision_mean / last_token / query_mean
                  (需 simlingo 环境: /data/ruolin/envs/simlingo/bin/python)

两个模型吃**同一批 1600x900 原始帧**,各自走自己的原生预处理(协议 §4 规则 4:
读取位置对齐管线,架构差异大的候选单列,不强行同表排序)。
sim/real 两侧的 ego 速度完全相同(同场景同帧号),故域配对的唯一变量仍是渲染风格。
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

DATA_ROOT = "/data/Zhengyang/Auto_Eval/ghosthead_v1"
FPS = 10.0


def ego_speed(scene_dir, fi):
    """从 renders/<scene>/scene.json 的 ego_to_world 有限差分求速率(m/s)。"""
    sc = json.load(open(Path(scene_dir) / "scene.json"))
    M = np.array([np.array(f["ego_to_world"], float) for f in sc["frames"]])
    n = len(M); i = min(max(fi, 1), n - 1)
    d = M[i][:3, 3] - M[i - 1][:3, 3]
    return float(np.linalg.norm(d[:2]) * FPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dd", "simlingo"])
    ap.add_argument("--frames", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain/frames")
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import cv2
    frames = sorted(Path(args.frames).glob("*.png"))
    recs = []
    for p in frames:
        stem = p.stem                                   # <scene>__<src>_t<sec>
        scene, rest = stem.split("__", 1)[0], stem.split("__")[-1]
        # 注意 scene 名本身含 "__"(如 gh_001000__brake) -> 用最后一个 "__" 切分
        scene = stem.rsplit("__", 1)[0]; rest = stem.rsplit("__", 1)[1]
        src, sec = rest.split("_t")
        recs.append({"path": str(p), "scene": scene, "source": src, "sec": int(sec)})
    print(f"[T-I/{args.model}] {len(recs)} 帧, {len(set(r['scene'] for r in recs))} 场景")

    store, meta = {}, []
    if args.model == "dd":
        sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/results/diffusiondrive_g1_adapter")
        from dd_adapter import DDRunner
        runner = DDRunner(device=args.device)
        for i, r in enumerate(recs):
            img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
            v = ego_speed(f"{DATA_ROOT}/renders/{r['scene']}", int(round(r["sec"] * FPS)))
            out = runner.run(img, v, None)
            key = f"{r['scene']}|{r['source']}|t{r['sec']}"
            for pool in ("vision_mean", "all_mean"):
                for l, x in enumerate(out["pooled"][pool]):
                    store[f"{key}|{pool}|L{l}"] = np.asarray(x, np.float32)
            meta.append({**r, "ego_speed": v, "commanded_speed": out["commanded_speed"],
                         "trajectory": out["trajectory"].tolist()})
            if (i + 1) % 100 == 0:
                print(f"[T-I/dd] {i+1}/{len(recs)}")
    else:
        from omegaconf import OmegaConf
        sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
        cfg = OmegaConf.to_container(
            OmegaConf.load("/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        runner = SimLingoRunner(cfg, capture_hidden=True)
        POOLS = ("vision_mean", "last_token", "query_mean", "seq_mean")
        for i, r in enumerate(recs):
            img = cv2.cvtColor(cv2.imread(r["path"]), cv2.COLOR_BGR2RGB)
            v = ego_speed(f"{DATA_ROOT}/renders/{r['scene']}", int(round(r["sec"] * FPS)))
            res = runner.infer(img, v, pool_modes=POOLS)
            key = f"{r['scene']}|{r['source']}|t{r['sec']}"
            for pool in POOLS:
                h = res.hidden.get(pool)
                if h is None:
                    continue
                for l in range(h.shape[0]):
                    store[f"{key}|{pool}|L{l}"] = h[l].astype(np.float32)
            wp = res.waypoints
            meta.append({**r, "ego_speed": v,
                         "commanded_speed": float(np.linalg.norm(wp[0] - wp[2]) * 2) if wp is not None else None,
                         "language": res.language})
            if (i + 1) % 50 == 0:
                print(f"[T-I/simlingo] {i+1}/{len(recs)}")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"acts_{args.model}.npz", meta=json.dumps(meta, ensure_ascii=False), **store)
    print(f"[T-I/{args.model}] wrote {out_dir}/acts_{args.model}.npz  ({len(store)} 数组)")


if __name__ == "__main__":
    main()
