"""G2 | 批量前向与缓存（手册 §6）。

对每个事件的 x_clean / x_ghost 帧逐帧独立前向（固定 seed），缓存：
  cache/{event_id}.h5
    clean/vision_mean  [n_frames, L, C] float16   每层 vision-token 均值池化
    clean/last_token   [n_frames, L, C] float16   每层末 token
    clean/waypoints    [n_frames, 10, 2]          speed waypoints（dt=0.25s）
    clean/route        [n_frames, 20, 2]
    clean/pred_speed   [n_frames]                 = ||wp[0]-wp[2]||*2，与部署端 control_pid 同口径
    clean/ego_speed    [n_frames]
    ghost/...
    attrs: meta(json) = 事件字段 + 预处理配置哈希 + ckpt 哈希

断点续跑：event_id 对应 h5 已完整则跳过。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlingo_runner import SimLingoRunner, preprocess_hash  # noqa: E402

ONE_SECOND_IDX = 4      # carla_fps 20 // (wp_dilation 1 * data_save_freq 5) -> waypoint dt = 0.25s
HALF_SECOND_IDX = 2


def commanded_speed(wp: np.ndarray) -> float:
    """复刻 agent_simlingo.control_pid 的 desired_speed：模型实际下发的目标速度 [m/s]。"""
    return float(np.linalg.norm(wp[HALF_SECOND_IDX - 2] - wp[ONE_SECOND_IDX - 2]) * 2.0)


def mean_diff_speed(wp: np.ndarray, dt: float = 0.25) -> float:
    """备用行为量：waypoints 差分速度均值。"""
    return float(np.mean(np.linalg.norm(np.diff(wp, axis=0), axis=1)) / dt)


def run_condition(runner, root: Path, frames, pool_modes, prompt_speed=None):
    out = {m: [] for m in pool_modes}
    wps, routes, pspd, mspd, espd, lang = [], [], [], [], [], []
    for fr in frames:
        img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
        r = runner.infer(img, fr["ego_speed_mps"], pool_modes=pool_modes, prompt_speed=prompt_speed)
        for m in pool_modes:
            out[m].append(r.hidden[m])
        wps.append(r.waypoints)
        routes.append(r.route)
        pspd.append(commanded_speed(r.waypoints))
        mspd.append(mean_diff_speed(r.waypoints))
        espd.append(fr["ego_speed_mps"])
        lang.append(r.language)
    return {
        **{m: np.stack(out[m]).astype(np.float16) for m in pool_modes},
        "waypoints": np.stack(wps).astype(np.float32),
        "route": np.stack(routes).astype(np.float32),
        "pred_speed": np.asarray(pspd, dtype=np.float32),
        "mean_diff_speed": np.asarray(mspd, dtype=np.float32),
        "ego_speed": np.asarray(espd, dtype=np.float32),
        "prompt_speed": np.asarray([prompt_speed if prompt_speed is not None else e for e in espd],
                                   dtype=np.float32),
        "language": lang,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    ap.add_argument("--events", default=None, help="事件清单文件（默认全量 events_all.jsonl）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default=None, help="覆盖 config.model.device，便于两卡并行")
    ap.add_argument("--report-tag", default="", help="报告文件后缀，并行跑时避免互相覆盖")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.device:
        cfg["model"]["device"] = args.device
    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])
    cache_dir = work / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool_modes = list(cfg["cache"]["pool_modes"])

    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    if args.events:
        keep = set(Path(args.events).read_text().split())
        events = [e for e in events if e["event_id"] in keep]
    if args.limit:
        events = events[: args.limit]

    runner = SimLingoRunner(cfg, capture_hidden=True)
    ph = preprocess_hash(cfg["model"]["preprocess"])
    print(f"[G2] {len(events)} events, ckpt={runner.ckpt_sha1}, preprocess_hash={ph}")

    done, failed, skipped = 0, [], 0
    t_start = time.time()
    for i, ev in enumerate(events):
        path = cache_dir / f"{ev['event_id']}.h5"
        if path.exists() and not args.overwrite:
            try:
                with h5py.File(path, "r") as f:
                    if all(f"{c}/{m}" in f for c in ("clean", "ghost") for m in pool_modes):
                        skipped += 1
                        continue
            except Exception:
                pass
        try:
            # 手册 §10.4：clean/ghost 之间 prompt 必须完全一致。
            # prompt_anchor=clean -> 两个条件都用 clean 帧的平均 ego 速度写进 prompt，
            # 使条件间唯一的差异是图像本身。per_frame = 旧行为（各写各的，有污染）。
            anchor = None
            if cfg["model"].get("prompt_anchor", "clean") == "clean":
                anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            res = {c: run_condition(runner, root, ev[f"x_{c}_frames"], pool_modes, prompt_speed=anchor)
                   for c in ("clean", "ghost")}
            with h5py.File(path, "w") as f:
                for cond, d in res.items():
                    g = f.create_group(cond)
                    for k, v in d.items():
                        if k == "language":
                            g.attrs["language"] = json.dumps(v, ensure_ascii=False)
                        else:
                            g.create_dataset(k, data=v, compression="gzip", compression_opts=1)
                meta = dict(ev)
                meta.pop("ttc_curve", None)
                f.attrs["meta"] = json.dumps(meta, ensure_ascii=False)
                f.attrs["preprocess_hash"] = ph
                f.attrs["ckpt_sha1"] = runner.ckpt_sha1
                f.attrs["waypoint_dt_s"] = 0.25
                f.attrs["pool_modes"] = json.dumps(pool_modes)
                f.attrs["prompt_anchor"] = cfg["model"].get("prompt_anchor", "clean")
                f.attrs["prompt_speed"] = -1.0 if anchor is None else anchor
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed.append({"event_id": ev["event_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[G2] FAIL {ev['event_id']}: {exc}")
        if (i + 1) % 10 == 0 or i + 1 == len(events):
            el = time.time() - t_start
            print(f"[G2] {i+1}/{len(events)}  done={done} skip={skipped} fail={len(failed)}  "
                  f"{el:.0f}s ({el/max(1,done+len(failed)):.1f}s/event)")

    total = done + skipped
    report = {
        "n_events": len(events), "cached": done, "skipped_existing": skipped,
        "failed": failed, "completeness": total / max(1, len(events)),
        "ckpt_sha1": runner.ckpt_sha1, "preprocess_hash": ph,
        "pool_modes": pool_modes, "waypoint_dt_s": 0.25,
        "prompt_anchor": cfg["model"].get("prompt_anchor", "clean"),
        "elapsed_s": time.time() - t_start,
    }
    (work / "results" / f"g2_cache_report{args.report_tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[G2] completeness={report['completeness']:.3f}  "
          f"{'PASS' if report['completeness'] >= 0.99 else 'FAIL'} (门槛 0.99)")


if __name__ == "__main__":
    main()
