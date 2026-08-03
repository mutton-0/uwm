"""G1 人工抽检可视化（手册 §5.3）：随机 N 个事件，画 clean/ghost 帧 + 3D 框 + TTC 曲线。"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image


def draw_event(nusc, ev, root: Path, out_path: Path):
    from nuscenes.utils.geometry_utils import view_points

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 0.65]})
    for ax, key, title in ((axes[0], "x_clean_frames", "clean"), (axes[1], "x_ghost_frames", "ghost")):
        fr = ev[key][-1]
        img = Image.open(root / fr["filename"])
        ax.imshow(img)
        ax.set_title(f"{title}  t={fr['t'] - ev['t_emergence']:+.2f}s  v_ego={fr['ego_speed_mps']:.1f} m/s", fontsize=10)
        ax.axis("off")

        sd = nusc.get("sample_data", fr["sd_token"])
        calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(calib["camera_intrinsic"])
        _, boxes, _ = nusc.get_sample_data(fr["sd_token"])
        for box in boxes:
            ann = nusc.get("sample_annotation", box.token)
            if ann["instance_token"] != ev["object_token"]:
                continue
            box.render(ax, view=K, normalize=True, colors=("red", "red", "red"), linewidth=2)

    ax = axes[2]
    curve = np.array(ev["ttc_curve"])
    if len(curve):
        ax.plot(curve[:, 0], np.clip(curve[:, 1], 0, 20), "-o", ms=3)
    ax.axvline(0, color="r", ls="--", label="t_emergence")
    ax.axhline(3, color="gray", ls=":", label="TTC=3s")
    ax.set_xlabel("t - t_emergence [s]"); ax.set_ylabel("frame TTC [s] (clip 20)")
    ax.set_ylim(0, 20); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f"[{ev['event_type']}] {ev['event_id']}  cls={ev['object_class']}  "
                 f"min_ttc_1s={ev['min_ttc_1s']}  d_long={ev['d_long_at_emergence']:.1f}m  "
                 f"lat={ev['lat_at_emergence']:.1f}m  {'NIGHT' if ev['is_night'] else 'day'}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=72, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    from nuscenes.nuscenes import NuScenes

    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])
    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    out_dir = work / "mining" / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"], dataroot=str(root), verbose=False)

    random.seed(args.seed)
    # 分层抽样：每类按比例抽，保证 A/B/C/D 都被检查到
    by_type = {}
    for e in events:
        by_type.setdefault(e["event_type"], []).append(e)
    picked = []
    for t, lst in sorted(by_type.items()):
        k = max(2, round(args.n * len(lst) / len(events)))
        picked += random.sample(lst, min(k, len(lst)))
    picked = picked[: args.n]

    for e in picked:
        draw_event(nusc, e, root, out_dir / f"{e['event_id']}.png")
    (out_dir / "picked.json").write_text(json.dumps([e["event_id"] for e in picked], indent=2))
    print(f"[G1-viz] wrote {len(picked)} figures -> {out_dir}")


if __name__ == "__main__":
    main()
