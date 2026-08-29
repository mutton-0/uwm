"""T-I 前置|从 ghosthead_v1 抽取域配对原始帧(1600x900)。

域配对(P1 的等价物,见 amendments.md FA.1 偏离 1):
  sim  = renders/<scene>/frames.mp4                          CARLA 引擎渲染
  real = ghosthead_result/<scene>/seg1p0/<scene>_seg1p0.mp4  世界模型真实感重绘
同一场景、同一构图、同一 actor 与 ego-GT,唯一变量是渲染风格 = do(appearance)。
原始帧恰为 1600x900(与 nuScenes 同分辨率),因此两个模型都能走各自的原生预处理。
"""
from __future__ import annotations
import argparse, os, subprocess
from pathlib import Path

DATA_ROOT = "/data/Zhengyang/Auto_Eval/ghosthead_v1"
FPS, SEC_LIST = 10.0, [1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/ruolin/uwm/sim2real_demo_ttc/variants/i_domain/frames")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(os.listdir(f"{DATA_ROOT}/renders"))
    n = 0
    for sc in scenes:
        srcs = {"sim": f"{DATA_ROOT}/renders/{sc}/frames.mp4",
                "real": f"{DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"}
        if not all(os.path.exists(p) for p in srcs.values()):
            print(f"[I/frames] skip {sc} (缺件)"); continue
        for src, mp4 in srcs.items():
            for s in SEC_LIST:
                fi = int(round(s * FPS))
                png = out / f"{sc}__{src}_t{s}.png"
                if png.exists():
                    continue
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
                                "-vf", f"select='eq(n\\,{fi})'", "-frames:v", "1", str(png)],
                               check=False)
                n += png.exists()
    print(f"[I/frames] 新抽 {n} 帧 -> {out}  (共 {len(list(out.glob('*.png')))} 帧)")


if __name__ == "__main__":
    main()
