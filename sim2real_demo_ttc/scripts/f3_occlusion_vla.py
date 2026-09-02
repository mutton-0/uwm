"""F-3|遮挡必要性检验（VLA 候选：Alpamayo-R1 / AutoVLA）。

与 `f3_occlusion_necessity.py` **同一套设计与判定**，只是输入构造不同：
两个 VLA 吃多相机 × 多时刻，遮挡必须落在**前视相机、目标时刻**那一格上。

  Alpamayo：`load_nuscenes` 返回 image_frames [n_cam, T, C, H, W]（原分辨率）。
            `NUSCENES_CAMERA_MAPPING` 里 CAM_FRONT 出现两次（索引 1 与 6，
            分别对应 120° 广角与 30° 长焦位），按 camera_indices 排序后是第 1、3 行。
            **两行都要遮**，否则模型仍能从另一路前视看到该实体。
  AutoVLA：`temporal_paths()` 返回 3 相机 × 4 时刻的**文件路径**，
            故把遮挡版写到临时文件再替换 `front_camera` 的对应路径。

**§FM/A56 修复（本版）：窗口内每一帧独立投影 + 独立遮挡。**
旧版只遮窗口最后一帧（t0），隐含假设"更早的帧早于 emergence 因而看不到实体"。
实测该假设对 Alpamayo **完全不成立**（窗口 4×0.1 s = 0.3 s，而 t0 − t_emergence 的分位
是 [0.30, 0.35, 0.40] s ⇒ 288 个 A 类事件里 280 个的 4 帧**全部**落在 emergence 之后）。
现改为：对窗口内每个时间戳，用 `scripts/f3_window_boxes.WindowBoxes` 重新投影该实体
（复用挖掘期的 `compute_scene_geometry` + `frame_bbox`，一行未改），
可见则用**该帧自己的均值色**涂掉，不可见则该帧不动。
对照臂同理逐帧涂同一个对照框，保证"遮挡面积"在两臂之间可比。

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

# `f3_window_boxes` 会 import g1_mine_events -> pyquaternion / nuscenes devkit。
# 这两个包在本机只存在于 `external/autovla_deps`，而该目录**必须在 torch/torchvision
# 完成算子注册之后**才能上 sys.path（§CE/A35 的 transformers 4.49 顺序约束）。
# 故这里**延迟到 runner 构造完成之后**再 import，且用 append 而非 insert。
def _load_window_boxes():
    import importlib.util as _u
    if _u.find_spec("pyquaternion") is None or _u.find_spec("nuscenes") is None:
        d = "/data/ruolin/uwm/external/autovla_deps"
        if d not in sys.path:
            sys.path.append(d)
    from f3_window_boxes import WindowBoxes as _WB
    return _WB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["alpa", "autovla"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--pos", default="A", help="正例类名（G1 用 A，前车急刹用 LB）")
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
    evs = [e for e in evs if e["event_type"] == args.pos
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    print(f"[F3/{LABEL}] {args.pos} 类可用事件 {len(evs)}（含投影框）")

    tmp = Path(tempfile.mkdtemp(prefix="f3_occ_"))
    rng = np.random.default_rng(0)
    WB = None                       # 逐帧投影器，在拿到 runner.nusc / runner.r.ndi 后构造

    if args.model == "alpa":
        sys.path.insert(0, str(RES / "alpamayo_g1_adapter"))
        from alpa_patch import AlpaPatchRunner
        runner = AlpaPatchRunner(device=args.device)
        FRONT_ROWS = (1, 3)          # 排序后 camera_indices = [0,1,2,6]，1 与 6 都是 CAM_FRONT

        def _fill(imf, row, t, box):
            """把 image_frames[row, t] 的 box 区域涂成**该帧自己的**均值色。"""
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            H, Wd = imf.shape[-2], imf.shape[-1]
            x0 = max(0, min(x0, Wd - 1)); x1 = max(x0 + 1, min(x1, Wd))
            y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
            fill = imf[row, t].float().mean(dim=(1, 2))[:, None, None].to(imf.dtype)
            imf[row, t, :, y0:y1, x0:x1] = fill

        def run_with(ev, cond, mode=None, cbox=None):
            """mode=None 原始；'occ' 逐帧遮实体；'ctrl' 逐帧遮对照框（cbox）。"""
            r = runner.r
            data = r.load(ev["scene_name"], ev[f"x_{cond}_frames"][0]["t"])
            a = r.load(ev["scene_name"], ev["x_clean_frames"][0]["t"])
            data["ego_history_xyz"] = a["ego_history_xyz"]; data["ego_history_rot"] = a["ego_history_rot"]
            if mode is None:
                return runner.run_from_data(data), 0
            imf = data["image_frames"]                     # [n_cam, T, C, H, W]
            ts = np.asarray(data["absolute_timestamps"]) * 1e-6        # [n_cam, T]，µs -> s
            # 只遮两路前视；两路时间戳相同，取第一路作为窗口时间轴
            ref_row = next((r_ for r_ in FRONT_ROWS if r_ < imf.shape[0]), None)
            if ref_row is None:
                return runner.run_from_data(data), 0
            win_t = ts[ref_row].tolist()
            boxes = (WB.boxes_for(ev, win_t) if mode == "occ"
                     else [cbox] * len(win_t))             # 对照框逐帧同位置
            n_hit = 0
            for ti, bx in enumerate(boxes):
                if bx is None:
                    continue
                for row in FRONT_ROWS:
                    if row < imf.shape[0]:
                        _fill(imf, row, ti, bx)
                n_hit += 1
            return runner.run_from_data(data), n_hit

        if not hasattr(runner, "run_from_data"):
            raise SystemExit("AlpaPatchRunner 需要 run_from_data（见 alpa_patch.py）")
        _WB = _load_window_boxes()
        from nuscenes.nuscenes import NuScenes
        WB = _WB(NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False))
    else:
        sys.path.insert(0, str(RES / "autovla_g1_adapter"))
        from autovla_adapter import AutoVLARunner
        runner = AutoVLARunner(device=args.device, nuscenes_root=args.nuscenes_root)
        WB = _load_window_boxes()(runner.nusc)   # AutoVLA 适配器自带 NuScenes 实例，直接复用

        def _front_window(tok):
            """复刻 temporal_paths 的 prev 链，返回前视 4 帧的 (路径, 绝对时间秒)。"""
            nusc = runner.nusc
            sd = nusc.get("sample_data", tok)
            samples, t = [], sd["sample_token"]
            for _ in range(4):
                sm = nusc.get("sample", t)
                samples.append(sm)
                t = sm["prev"] if sm["prev"] else t
            samples = samples[::-1]
            out = []
            for sm in samples:
                s_ = nusc.get("sample_data", sm["data"]["CAM_FRONT"])
                out.append((str(Path(runner.root) / s_["filename"]), s_["timestamp"] * 1e-6))
            return out

        def run_with(ev, cond, mode=None, cbox=None):
            tok = ev[f"x_{cond}_frames"][0]["sd_token"]
            spd = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            if mode is None:
                return runner.run(tok, spd)["commanded_speed"], 0
            win = _front_window(tok)
            boxes = (WB.boxes_for(ev, [t for _p, t in win]) if mode == "occ"
                     else [cbox] * len(win))
            newp, n_hit = [], 0
            for k, ((pth, _t), bx) in enumerate(zip(win, boxes)):
                if bx is None:
                    newp.append(pth); continue
                im = np.array(Image.open(pth).convert("RGB"))
                x0, y0, x1, y1 = [int(round(v)) for v in bx]
                H, Wd = im.shape[:2]
                x0 = max(0, min(x0, Wd - 1)); x1 = max(x0 + 1, min(x1, Wd))
                y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
                im[y0:y1, x0:x1] = im.reshape(-1, 3).mean(0).astype(im.dtype)
                q = tmp / f"{ev['event_id']}_{mode}_{k}.jpg"
                Image.fromarray(im).save(str(q), quality=95)
                newp.append(str(q)); n_hit += 1
            orig = runner.temporal_paths

            def _patched(t, _o=orig, _np=list(newp)):
                d = _o(t)
                d["front_camera"] = _np
                return d
            runner.temporal_paths = _patched
            try:
                return runner.run(tok, spd)["commanded_speed"], n_hit
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
            v_clean, _ = run_with(ev, "clean")
            v_ghost, _ = run_with(ev, "ghost")
            v_occ, n_occ = run_with(ev, "ghost", "occ")
            v_ctrl, n_ctrl = run_with(ev, "ghost", "ctrl", cb)
        except Exception as exc:                                     # noqa: BLE001
            skipped[f"fwd:{type(exc).__name__}"] += 1
            print(f"[F3/{LABEL}] FAIL {ev['event_id']}: {exc}"); continue
        if n_occ == 0:
            # 窗口内一帧都没投影出实体 ⇒ 这个事件的 occ 臂等于什么都没做，不能进统计
            skipped["occ_no_frame_hit"] += 1; continue
        recs.append({"eid": ev["event_id"], "scene": ev["scene_name"],
                     "v_clean": v_clean, "v_ghost": v_ghost, "v_occ": v_occ, "v_ctrl": v_ctrl,
                     "b_ghost": v_ghost - v_clean, "b_occ": v_occ - v_clean,
                     "b_ctrl": v_ctrl - v_clean,
                     "n_frames_occluded": int(n_occ), "n_frames_ctrl": int(n_ctrl)})
        if (i + 1) % 20 == 0:
            print(f"[F3/{LABEL}] {i+1}/{len(evs)} done={len(recs)}", flush=True)

    out = {"model": LABEL, "design": "F-3 遮挡必要性检验（VLA：**窗口内每一帧**独立投影+遮挡，§FM/A56）",
           "occlusion": "窗口内每一帧独立投影该实体，可见则用**该帧自己的**均值色涂掉"
                        + ("；Alpamayo 两路前视（120° 与 30°）都遮" if args.model == "alpa" else ""),
           "window_projection_stats": (WB.stats if WB else None),
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
