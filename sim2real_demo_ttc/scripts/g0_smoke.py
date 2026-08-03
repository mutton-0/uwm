"""G0 冒烟测试（手册 §4）。

通过标准：
  1) 输出合理：waypoints 非 NaN、数值量级正常；
  2) 全层 hidden 可抓取（打印层数/维度）；
  3) 固定 seed 跑两次，逐位一致。
未过 G0 不得进入后续任何步骤。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlingo_runner import SimLingoRunner, load_cfg, preprocess_hash  # noqa: E402


def pick_frame(cfg) -> Path:
    root = Path(cfg["paths"]["nuscenes_root"])
    cam = cfg["mining"]["camera"]
    files = sorted((root / "samples" / cam).glob("*.jpg"))
    assert files, f"没找到 {cam} 图像"
    return files[len(files) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--image", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_dir = Path(cfg["paths"]["work_dir"]) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = Path(args.image) if args.image else pick_frame(cfg)
    img = np.array(Image.open(img_path).convert("RGB"))
    print(f"[G0] frame: {img_path}  shape={img.shape}")

    runner = SimLingoRunner(cfg, capture_hidden=True)
    print(f"[G0] ckpt sha1(head16)={runner.ckpt_sha1}  preprocess_hash={preprocess_hash(cfg['model']['preprocess'])}")
    print(f"[G0] decoder layers found: {len(runner.layer_names)}  (e.g. {runner.layer_names[0]})")

    r1 = runner.infer(img, args.speed)
    r2 = runner.infer(img, args.speed)

    checks = {}
    wp = r1.waypoints
    print(f"[G0] prompt: {r1.prompt}")
    print(f"[G0] waypoints shape={None if wp is None else wp.shape}")
    print(f"[G0] waypoints[:4]=\n{None if wp is None else np.round(wp[:4], 3)}")
    print(f"[G0] route shape={None if r1.route is None else r1.route.shape}")
    print(f"[G0] hidden: n_layers={r1.n_layers} dim={r1.hidden_dim} seq_len={r1.seq_len} "
          f"modes={list(r1.hidden.keys())} per-mode shape={r1.hidden['vision_mean'].shape}")
    print(f"[G0] generated language: {r1.language!r}")

    checks["waypoints_not_nan"] = bool(wp is not None and np.isfinite(wp).all())
    checks["waypoints_magnitude_ok"] = bool(wp is not None and np.abs(wp).max() < 200.0)
    checks["hidden_all_layers"] = bool(r1.n_layers >= 8 and r1.hidden_dim >= 256)
    checks["hidden_finite"] = bool(all(np.isfinite(v).all() for v in r1.hidden.values()))
    checks["deterministic_waypoints"] = bool(np.array_equal(r1.waypoints, r2.waypoints))
    checks["deterministic_hidden"] = bool(all(np.array_equal(r1.hidden[k], r2.hidden[k]) for k in r1.hidden))

    passed = all(checks.values())
    print("\n[G0] checks:")
    for k, v in checks.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    print(f"[G0] === {'PASSED' if passed else 'FAILED'} ===")

    report = {
        "frame": str(img_path),
        "ego_speed_mps": args.speed,
        "ckpt_sha1_head16": runner.ckpt_sha1,
        "preprocess_hash": preprocess_hash(cfg["model"]["preprocess"]),
        "preprocess": cfg["model"]["preprocess"],
        "prompt": r1.prompt,
        "n_layers": r1.n_layers,
        "hidden_dim": r1.hidden_dim,
        "seq_len": r1.seq_len,
        "generated_language": r1.language,
        "waypoints": None if wp is None else wp.tolist(),
        "route": None if r1.route is None else r1.route.tolist(),
        "checks": checks,
        "passed": passed,
    }
    (out_dir / "g0_smoke.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[G0] wrote {out_dir/'g0_smoke.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
