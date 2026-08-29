"""pilot post-train 前置|缓存 SimLingo 最终层 driving-query 隐状态 + 原始航点 + 人类目标速度。

为什么这样切：post-train 只训 `speed_wps_head`（Linear 896→256 → SiLU → Linear 256→2，229k 参数），
其余全部冻结。这**正是选型协议 §1 的核心假设的直接检验**——
"内部已编码危险的候选，post-train 只需把读出接到动作上 ⇒ 便宜"。
头之前的一切都冻结 ⇒ 每个样本的输入特征 h_23[query] 是常量 ⇒ **一次前向缓存，之后离线训练**，
整个 pilot 的 GPU 成本 = 一遍前向，不需要反复跑主干。

缓存内容（每事件每条件每帧一条样本）：
  h_last   [n_drv, 896]  最终 decoder 层输出（**未过 final RMSNorm**，训练时再过，norm 是冻结的）
  wp0      [10, 2]       原始模型的 speed 航点（cumsum 后），用作蒸馏正则的锚
  v_cmd0   标量          原始 commanded_speed
  v_human  标量          人类在该帧之后 [0.5, 1.0] s 的实际平均速率（post-train 的任务目标）
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def human_speed_series(nusc, scene_token, camera="CAM_FRONT"):
    scene = nusc.get("scene", scene_token)
    sd = nusc.get("sample_data", nusc.get("sample", scene["first_sample_token"])["data"][camera])
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])
    ts, xy = [], []
    while True:
        p = nusc.get("ego_pose", sd["ego_pose_token"])
        ts.append(sd["timestamp"] * 1e-6); xy.append(p["translation"][:2])
        if not sd["next"]:
            break
        sd = nusc.get("sample_data", sd["next"])
    t = np.array(ts); x = np.array(xy)
    o = np.argsort(t); t, x = t[o], x[o]
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])[:, None]
    d[0], d[-1] = d[1], d[-2]
    v = np.convolve(np.linalg.norm(d, axis=1), np.ones(3) / 3, mode="same")
    return t, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["A", "D2a"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--horizon", nargs=2, type=float, default=[0.5, 1.0],
                    help="人类目标速度的取值窗口（相对该帧）")
    ap.add_argument("--out", default=str(W / "pilot_query_states.npz"))
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(
        "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"), resolve=True)
    cfg["model"]["device"] = args.device
    root = Path(cfg["paths"]["nuscenes_root"])
    evs = [json.loads(l) for l in open(W / "mining" / "events_all.jsonl")]
    matched = {t: set((W / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a",) if (W / "mining" / f"matched_{t}.txt").exists()}
    keep = [e for e in evs if e["event_type"] in args.types
            and not (e["event_type"] in matched and e["event_id"] not in matched[e["event_type"]])]
    print(f"[P1] {len(keep)} 事件待缓存（{args.types}）")

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"],
                    dataroot=cfg["paths"]["nuscenes_root"], verbose=False)
    series = {}

    sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
    from simlingo_runner import SimLingoRunner
    from g2_cache import commanded_speed
    runner = SimLingoRunner(cfg, capture_hidden=True)
    import torch

    H, WP, V0, VH, META = [], [], [], [], []
    for i, ev in enumerate(keep):
        st = ev["scene_token"]
        if st not in series:
            try:
                series[st] = human_speed_series(nusc, st)
            except Exception:
                series[st] = None
        ser = series[st]
        anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        for cond in ("clean", "ghost"):
            for fr in ev[f"x_{cond}_frames"]:
                try:
                    img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
                    res = runner.infer(img, fr["ego_speed_mps"], pool_modes=("vision_mean",),
                                       prompt_speed=anchor)
                    hs = runner._layer_outputs[-runner.n_layers:]
                    n_drv = int(runner._len_driving)
                    h = hs[-1][0, -n_drv:, :].float().cpu().numpy()
                except Exception as exc:                                   # noqa: BLE001
                    print(f"[P1] skip {ev['event_id']}/{cond}: {exc}")
                    continue
                if ser is None:
                    vh = float("nan")
                else:
                    t, v = ser
                    m = (t >= fr["t"] + args.horizon[0]) & (t <= fr["t"] + args.horizon[1])
                    vh = float(v[m].mean()) if m.sum() else float("nan")
                H.append(h.astype(np.float16)); WP.append(res.waypoints.astype(np.float32))
                V0.append(commanded_speed(res.waypoints)); VH.append(vh)
                META.append({"event_id": ev["event_id"], "scene": ev["scene_name"],
                             "type": ev["event_type"], "cond": cond, "t": fr["t"],
                             "ego_speed": fr["ego_speed_mps"], "prompt_speed": anchor})
        if (i + 1) % 50 == 0:
            print(f"[P1] {i+1}/{len(keep)}  样本 {len(H)}")

    np.savez_compressed(args.out, h_last=np.stack(H), wp0=np.stack(WP),
                        v_cmd0=np.array(V0, np.float32), v_human=np.array(VH, np.float32),
                        meta=json.dumps(META, ensure_ascii=False), n_layers=runner.n_layers)
    ok = np.isfinite(np.array(VH))
    print(f"[P1] wrote {args.out}  样本 {len(H)}（人类目标速度可用 {ok.sum()}）  "
          f"h_last shape={np.stack(H).shape}")


if __name__ == "__main__":
    main()
