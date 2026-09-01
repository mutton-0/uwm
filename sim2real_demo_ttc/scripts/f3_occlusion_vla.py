"""F-3|遮挡必要性检验（VLA 候选：Alpamayo-R1 / AutoVLA）。

与 `f3_occlusion_necessity.py` **同一套设计与判定**，只是输入构造不同：
两个 VLA 吃多相机 × 多时刻，遮挡必须落在**前视相机、目标时刻**那一格上。

  Alpamayo：`load_nuscenes` 返回 image_frames [n_cam, T, C, H, W]（原分辨率）。
            `NUSCENES_CAMERA_MAPPING` 里 CAM_FRONT 出现两次（索引 1 与 6，
            分别对应 120° 广角与 30° 长焦位），按 camera_indices 排序后是第 1、3 行。
            **两行都要遮**，否则模型仍能从另一路前视看到该实体。
            时间维取最后一帧（`img_ts` 以 t0_us 结尾）。
  AutoVLA：`temporal_paths()` 返回 3 相机 × 4 时刻的**文件路径**，
            故把遮挡版写到临时文件再替换 `front_camera` 的最后一个路径。

统计与判定与非 VLA 版逐条一致（含必需的 ctrl 对照臂）。
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, tempfile
from collections import defaultdict
from pathlib import Path

# 两个 VLA 候选跑在没有 numpy 的 py312 解释器里，依赖来自 Alpamayo 的 venv；
# 必须在 import numpy 之前挂上（append 而非 insert，故其余环境自带的 numpy 仍然优先）。
if importlib.util.find_spec("numpy") is None:
    for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
               "/data/Zhengyang/alpamayo/src"):
        if _p not in sys.path:
            sys.path.append(_p)

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import boot_scene, occlude, control_box     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["alpa", "autovla"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--limit", type=int, default=120, help="慢模型控成本：默认只跑前 120 个事件")
    ap.add_argument("--min-b", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    LABEL = {"alpa": "Alpamayo-R1", "autovla": "AutoVLA"}[args.model]
    work = Path(args.work)
    if not args.out:
        args.out = str(RES / f"f3_occlusion_{args.model}.json")

    from PIL import Image      # VLA 环境（Alpamayo venv）没有 cv2，只有 PIL；功能等价
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == "A"
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    print(f"[F3/{LABEL}] A 类可用事件 {len(evs)}（含投影框）")

    tmp = Path(tempfile.mkdtemp(prefix="f3_occ_"))
    rng = np.random.default_rng(0)

    if args.model == "alpa":
        sys.path.insert(0, str(RES / "alpamayo_g1_adapter"))
        from alpa_patch import AlpaPatchRunner
        runner = AlpaPatchRunner(device=args.device)
        FRONT_ROWS = (1, 3)          # 排序后 camera_indices = [0,1,2,6]，1 与 6 都是 CAM_FRONT

        def run_with(ev, cond, box=None):
            """cond='clean'|'ghost'；box 非空时遮挡前视相机最后一帧的该区域。"""
            r = runner.r
            data = r.load(ev["scene_name"], ev[f"x_{cond}_frames"][0]["t"])
            a = r.load(ev["scene_name"], ev["x_clean_frames"][0]["t"])
            data["ego_history_xyz"] = a["ego_history_xyz"]; data["ego_history_rot"] = a["ego_history_rot"]
            if box is not None:
                imf = data["image_frames"]                      # [n_cam, T, C, H, W]
                x0, y0, x1, y1 = [int(round(v)) for v in box]
                H, Wd = imf.shape[-2], imf.shape[-1]
                x0 = max(0, min(x0, Wd - 1)); x1 = max(x0 + 1, min(x1, Wd))
                y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
                for row in FRONT_ROWS:
                    if row < imf.shape[0]:
                        fill = imf[row, -1].float().mean(dim=(1, 2))[:, None, None].to(imf.dtype)
                        imf[row, -1, :, y0:y1, x0:x1] = fill
            return runner.run_from_data(data)

        if not hasattr(runner, "run_from_data"):
            raise SystemExit("AlpaPatchRunner 需要 run_from_data（见 alpa_patch.py）")
    else:
        sys.path.insert(0, str(RES / "autovla_g1_adapter"))
        from autovla_adapter import AutoVLARunner
        runner = AutoVLARunner(device=args.device, nuscenes_root=args.nuscenes_root)

        def run_with(ev, cond, box=None):
            tok = ev[f"x_{cond}_frames"][0]["sd_token"]
            spd = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            if box is None:
                return runner.run(tok, spd)["commanded_speed"]
            paths = runner.temporal_paths(tok)
            im = np.array(Image.open(paths["front_camera"][-1]).convert("RGB"))
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            H, Wd = im.shape[:2]
            x0 = max(0, min(x0, Wd - 1)); x1 = max(x0 + 1, min(x1, Wd))
            y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
            im[y0:y1, x0:x1] = im.reshape(-1, 3).mean(0).astype(im.dtype)
            p = tmp / f"{ev['event_id']}_occ.jpg"
            Image.fromarray(im).save(str(p), quality=95)
            orig = runner.temporal_paths
            runner.temporal_paths = lambda t, _o=orig, _p=str(p): (
                lambda d: (d["front_camera"].__setitem__(-1, _p), d)[1])(_o(t))
            try:
                return runner.run(tok, spd)["commanded_speed"]
            finally:
                runner.temporal_paths = orig

    if args.model == "alpa":
        # 覆盖预筛：adapter 无条件构造 6.4 s 未来轨迹，贴近场景末尾的事件不可用。
        # 与 alpa_g1_cache.py 的预筛口径一致，不是本轮新加的条件。
        n0 = len(evs)
        evs = [e for e in evs
               if all(runner.covers(e["scene_name"], e[f"x_{c}_frames"][0]["t"])
                      for c in ("clean", "ghost"))]
        print(f"[F3/{LABEL}] 覆盖预筛后 {len(evs)}/{n0}")
    evs = evs[: args.limit] if args.limit else evs
    recs, skipped = [], defaultdict(int)
    for i, ev in enumerate(evs):
        bb = ev["x_ghost_frames"][0]["bbox_xyxy"]
        try:
            probe = np.array(Image.open(
                Path(args.nuscenes_root) / ev["x_ghost_frames"][0]["filename"]).convert("RGB"))
        except Exception:                                            # noqa: BLE001
            skipped["read_fail"] += 1; continue
        cb = control_box(probe, bb, rng)
        if cb is None:
            skipped["no_control_box"] += 1; continue
        try:
            v_clean = run_with(ev, "clean")
            v_ghost = run_with(ev, "ghost")
            v_occ = run_with(ev, "ghost", bb)
            v_ctrl = run_with(ev, "ghost", cb)
        except Exception as exc:                                     # noqa: BLE001
            skipped[f"fwd:{type(exc).__name__}"] += 1
            print(f"[F3/{LABEL}] FAIL {ev['event_id']}: {exc}"); continue
        recs.append({"eid": ev["event_id"], "scene": ev["scene_name"],
                     "v_clean": v_clean, "v_ghost": v_ghost, "v_occ": v_occ, "v_ctrl": v_ctrl,
                     "b_ghost": v_ghost - v_clean, "b_occ": v_occ - v_clean,
                     "b_ctrl": v_ctrl - v_clean})
        if (i + 1) % 20 == 0:
            print(f"[F3/{LABEL}] {i+1}/{len(evs)} done={len(recs)}", flush=True)

    out = {"model": LABEL, "design": "F-3 遮挡必要性检验（VLA：遮前视相机目标时刻）",
           "occlusion": "前视相机最后一帧的实体投影框涂为该帧均值色"
                        + ("；Alpamayo 两路前视（120° 与 30°）都遮" if args.model == "alpa" else ""),
           "control_arm": "同面积、同离心率带、不重叠的对照框",
           "n_events": len(recs), "n_limit": args.limit, "skipped": dict(skipped),
           "per_event": recs}
    sc = [r["scene"] for r in recs]
    for k in ("b_ghost", "b_occ", "b_ctrl"):
        out[k] = boot_scene([r[k] for r in recs], sc)
        if out[k]:
            s = out[k]
            print(f"[F3/{LABEL}] {k:8s} = {s['mean']:+.4f}  CI {np.round(s['ci95'],4).tolist()}  "
                  f"{'显著≠0' if (s['ci95'][0] > 0 or s['ci95'][1] < 0) else '与 0 不可区分'}")
    use = [r for r in recs if abs(r["b_ghost"]) >= args.min_b]
    out["n_used_for_R"] = len(use); out["min_b_gate"] = args.min_b
    base_sig = out["b_ghost"] and (out["b_ghost"]["ci95"][0] > 0 or out["b_ghost"]["ci95"][1] < 0)
    out["baseline_response_significant"] = bool(base_sig)
    if len(use) >= 20:
        out["necessity_ratio"] = boot_scene([1 - r["b_occ"] / r["b_ghost"] for r in use],
                                            [r["scene"] for r in use])
        out["necessity_ratio_control"] = boot_scene([1 - r["b_ctrl"] / r["b_ghost"] for r in use],
                                                    [r["scene"] for r in use])
    if not base_sig:
        out["verdict"] = ("不可估：**基线响应本身与 0 不可区分** ⇒ 没有可供必要性检验的响应。")
    elif "necessity_ratio" not in out:
        out["verdict"] = f"不可估：过门槛事件仅 {len(use)}"
    else:
        ci = out["necessity_ratio"]["ci95"]
        out["verdict"] = ("PASS：遮住关键实体后动作显著退回基线" if ci[0] > 0.5 else
                          ("FAIL：遮住关键实体后动作基本不变（盲目泛化签名）" if ci[1] < 0.5 else
                           "不可估：必要性比的 scene 级 CI 跨 0.5"))
        r = out["necessity_ratio"]
        print(f"[F3/{LABEL}] **必要性比 R = {r['mean']:+.3f}**  CI {np.round(r['ci95'],3).tolist()}")
    print(f"[F3/{LABEL}] 判定：{out['verdict']}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[F3/{LABEL}] wrote {args.out}")


if __name__ == "__main__":
    main()
