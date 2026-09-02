"""F-3|遮挡必要性检验：拿掉危险实体，动作是否退回基线。

工单：docs/g_vs_f3_simplified_axes_workorder.md §2。

**为什么要这个检验**：F① 依赖"什么算危险"的人工判据去构造匹配负例，
三个语料上都不稳定（G1 上两个 PASS，NAVSIM 上全掉）。
F-3 换成**输入层的因果操作**：不构造负例，直接把真实存在的关键实体从画面里抹掉，
看动作是否跟着消失。这是必要性检验（necessity），不需要任何语义标注。

四个臂（每个 A 类事件都跑满）：
    clean      危险实体本来就不在场
    ghost      危险实体可见（原始）
    **occ**    ghost + 把危险实体的投影框涂掉
    **ctrl**   ghost + 在**别处**涂一个同样大小的框（必需对照，见下）

**为什么 ctrl 臂是必需的而不是可选的**：occ 臂同时做了两件事——
「移除了这个实体」和「在画面里加了一块灰斑」。若只有 occ，
一个对任意灰斑都会反应的模型会被误判为"通过必要性检验"。
ctrl 臂把「加灰斑」这件事单独隔离出来：它与 occ 同面积、同离心率带、**不覆盖危险实体**。
这条纪律与 N1 的 D2a 几何匹配同源：**任何操作都要有一个只差目标语义的对照。**

主读数（预注册）：
    b_ghost = v_plan(ghost) − v_plan(clean)      原始响应
    b_occ   = v_plan(occ)   − v_plan(clean)      遮挡后残余响应
    **必要性比 R = 1 − b_occ / b_ghost**         1 = 完全必要，0 = 完全不必要
判定：
    R 的 scene 级 bootstrap CI 下界 > 0.5  → PASS（动作确由该实体的可见性因果驱动）
    CI 上界 < 0.5                          → FAIL（动作不是被这个实体驱动的 ⇒ 盲目泛化签名）
    CI 跨 0.5                              → 不可估
**分母守卫**：|b_ghost| 太小时 R 会爆炸。按事件级 |b_ghost| 门槛筛选，
且若某候选的 b_ghost 本身与 0 不可区分，则整格判"不可估：基线响应本身不存在"——
这不是失败，是"没有可供必要性检验的响应"（多个候选实测如此）。
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from collections import defaultdict
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
               "/data/Zhengyang/alpamayo/src"):
        if _p not in sys.path:
            sys.path.append(_p)

import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
NUSC = "/data/dataset/nuscenes/v1.0-trainval"


def boot_scene(vals, scenes, n=5000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        if np.isfinite(v):
            by[s].append(v)
    keys = list(by)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    out = [np.mean([x for i in rng.integers(0, len(keys), len(keys)) for x in by[keys[i]]])
           for _ in range(n)]
    return {"mean": float(np.mean([x for k in keys for x in by[k]])),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "n": int(sum(len(v) for v in by.values())), "n_scenes": len(keys)}


def occlude(img, bbox, mode="mean"):
    """把 bbox 区域涂成图像均值色（中性灰斑，不引入新的高频结构）。"""
    if bbox is None:
        return None
    H, W_ = img.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, min(x0, W_ - 1)); x1 = max(x0 + 1, min(x1, W_))
    y0 = max(0, min(y0, H - 1)); y1 = max(y0 + 1, min(y1, H))
    out = img.copy()
    out[y0:y1, x0:x1] = img.reshape(-1, 3).mean(0).astype(img.dtype) if mode == "mean" else 0
    return out


def control_box(img, bbox, rng):
    """同面积、同离心率带、**不与原框重叠**的对照框（水平镜像优先，重叠则上下另找）。"""
    if bbox is None:
        return None
    H, W_ = img.shape[:2]
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    # 水平镜像：离画幅中心的水平距离相同 ⇒ 离心率带一致
    mx0 = W_ - x1
    cand = [(mx0, y0, mx0 + w, y0 + h)]
    # 镜像若与原框重叠（目标本就在画面中央），改在竖直方向另找一条同高度带
    for dy in (-2 * h, 2 * h, -3 * h, 3 * h):
        cand.append((x0, y0 + dy, x1, y1 + dy))
    for c in cand:
        cx0, cy0, cx1, cy1 = c
        if cx0 < 0 or cy0 < 0 or cx1 > W_ or cy1 > H:
            continue
        if not (cx1 <= x0 or cx0 >= x1 or cy1 <= y0 or cy0 >= y1):
            continue                          # 与原框重叠，弃用
        return list(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["dd", "ltf", "ddv2", "simlingo", "alpa", "autovla"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default=NUSC)
    ap.add_argument("--pos", default="A")
    ap.add_argument("--limit", type=int, default=0, help="事件数上限（慢模型控成本用）")
    ap.add_argument("--min-b", type=float, default=0.02,
                    help="|b_ghost| 门槛：低于此值的事件不进 R 的统计（分母守卫）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--crop-center-row", type=int, default=0)
    ap.add_argument("--corpus", default="nuscenes", choices=["nuscenes", "navsim"],
                    help="DDv2 的点云来源；navsim 直接读 MergedPointCloud/*.pcd（§NS/A47）")
    ap.add_argument("--lidar-margin", type=float, default=0.25,
                    help="3D 框各轴外扩（米）。**不是为了多删点而调的**：nuScenes 标注/标定误差约 "
                         "0.1~0.3 m，实测有点落在框外 5 cm 处；2D 侧的 bbox_to_tokens 也已外扩 1 token。"
                         "0.0 作为敏感性口径并列报告（§FM/A59）")
    ap.add_argument("--occlude-lidar", action="store_true",
                    help="DDv2 专用：同时删掉落在实体 3D 框内的点云点（§FM/A59）。"
                         "不开启时行为与旧版逐位一致（只涂 RGB）")
    ap.add_argument("--sl-config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
             "simlingo": "SimLingo", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}[args.model]
    work = Path(args.work)
    if not args.out:
        args.out = str(RES / f"f3_occlusion_{args.model}.json")

    import cv2
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == args.pos
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    if args.limit:
        evs = evs[: args.limit]
    print(f"[F3/{LABEL}] {args.pos} 类可用事件 {len(evs)}（要求 ghost 帧有投影框）")

    def read(fn):
        return cv2.cvtColor(cv2.imread(str(Path(args.nuscenes_root) / fn)), cv2.COLOR_BGR2RGB)

    # ---------------- 各候选的前向接口（统一成 img -> v_plan） ----------------
    IS_VLA = args.model in ("alpa", "autovla")
    if args.model in ("dd", "ltf", "ddv2"):
        sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
        sys.path.insert(0, str(RES / "ltf_g1_adapter")); sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
        import dd_adapter as DD
        if args.crop_center_row:
            DD.set_crop_center_row(args.crop_center_row)
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        lidar = None
        if args.model == "ddv2":
            if args.corpus == "navsim":
                from ddv2_adapter import NavsimLidar
                lidar = NavsimLidar()
            else:
                from ddv2_adapter import NuScenesLidar
                lidar = NuScenesLidar(args.nuscenes_root)

        WB3 = None
        if args.occlude_lidar:
            if args.model != "ddv2":
                raise SystemExit("--occlude-lidar 只对 ddv2 有意义（其余候选不吃点云）")
            from f3_window_boxes import WindowBoxes, points_in_box, mirror_box3d
            WB3 = WindowBoxes(lidar.nusc if hasattr(lidar, "nusc") else None)
            LID_STATS = {"events": 0, "n_pts_occ": [], "n_pts_ctrl": [], "no_box3d": 0}

        def _lidar_for(ev, fr, arm):
            """arm ∈ {None, 'occ', 'ctrl'}；返回该臂应当喂给模型的点云。

            **两臂对称处理**：occ 臂删实体 3D 框内的点，ctrl 臂删**镜像 3D 框**内的点
            （沿 ego 纵轴 y -> −y，体积严格相等、纵向距离相同），
            使"删除范围"这个变量在两臂之间可比 —— 与 RGB 侧的水平镜像对照框同一构造。
            """
            key = (fr["filename"].split("/", 1)[1] if args.corpus == "navsim" else fr.get("sd_token"))
            pts = None if lidar is None else lidar.ego_points(key)
            if pts is None or arm is None or WB3 is None:
                return pts
            b3 = WB3.box3d_for(ev, fr["t"])
            if b3 is None:
                LID_STATS["no_box3d"] += 1
                return pts
            c, sz, yaw = b3 if arm == "occ" else mirror_box3d(*b3)
            g = float(args.lidar_margin)
            sz = (sz[0] + 2 * g, sz[1] + 2 * g, sz[2] + 2 * g)
            m = points_in_box(pts, c, sz, yaw)
            LID_STATS["n_pts_" + arm].append(int(m.sum()))
            return pts[~m]

        def infer(img, ev, fr, arm=None):
            kw = {} if lidar is None else {"lidar_xyz": _lidar_for(ev, fr, arm)}
            return float(runner.run(img, spd(ev), **kw)["commanded_speed"])
    elif args.model == "simlingo":
        from omegaconf import OmegaConf
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        cfg = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        from g2_cache import commanded_speed
        runner = SimLingoRunner(cfg, capture_hidden=False)

        def infer(img, ev, fr, arm=None):
            return float(commanded_speed(runner.infer(img, spd(ev), pool_modes=()).waypoints))
    else:
        raise SystemExit("VLA 候选走 f3_occlusion_vla.py（多帧输入，接口不同）")

    def spd(ev):
        return float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))

    rng = np.random.default_rng(0)
    recs, skipped = [], defaultdict(int)
    for i, ev in enumerate(evs):
        fc, fg = ev["x_clean_frames"][0], ev["x_ghost_frames"][0]
        bb = fg["bbox_xyxy"]
        try:
            ic, ig = read(fc["filename"]), read(fg["filename"])
        except Exception:                                            # noqa: BLE001
            skipped["read_fail"] += 1; continue
        cb = control_box(ig, bb, rng)
        if cb is None:
            skipped["no_control_box"] += 1; continue
        v_clean = infer(ic, ev, fc)
        v_ghost = infer(ig, ev, fg)
        v_occ = infer(occlude(ig, bb), ev, fg, "occ")
        v_ctrl = infer(occlude(ig, cb), ev, fg, "ctrl")
        if args.occlude_lidar:
            LID_STATS["events"] += 1
        recs.append({"eid": ev["event_id"], "scene": ev["scene_name"],
                     "v_clean": v_clean, "v_ghost": v_ghost, "v_occ": v_occ, "v_ctrl": v_ctrl,
                     "b_ghost": v_ghost - v_clean, "b_occ": v_occ - v_clean,
                     "b_ctrl": v_ctrl - v_clean,
                     "bbox_area_frac": float((bb[2]-bb[0])*(bb[3]-bb[1]) / (ig.shape[0]*ig.shape[1]))})
        if (i + 1) % 50 == 0:
            print(f"[F3/{LABEL}] {i+1}/{len(evs)}", flush=True)

    out = {"model": LABEL, "design": "F-3 遮挡必要性检验：clean / ghost / occ / ctrl 四臂",
           "occlusion": ("危险实体投影框涂为图像均值色（中性灰斑）"
                         + ("；**并删掉落在该实体 3D 框内的点云点**（§FM/A59）"
                            if args.occlude_lidar else "")),
           "occlude_lidar": bool(args.occlude_lidar),
           "control_arm": "同面积、同离心率带、不与原框重叠的对照框（隔离「加灰斑」这一效应）",
           "primary": "必要性比 R = 1 − b_occ / b_ghost；PASS 门槛：R 的 scene 级 CI 下界 > 0.5",
           "n_events": len(recs), "skipped": dict(skipped), "per_event": recs}

    sc = [r["scene"] for r in recs]
    for k in ("b_ghost", "b_occ", "b_ctrl"):
        out[k] = boot_scene([r[k] for r in recs], sc)
        if out[k]:
            s = out[k]
            print(f"[F3/{LABEL}] {k:8s} = {s['mean']:+.4f}  CI {np.round(s['ci95'],4).tolist()}  "
                  f"n={s['n']}/{s['n_scenes']}scene  "
                  f"{'显著≠0' if (s['ci95'][0] > 0 or s['ci95'][1] < 0) else '与 0 不可区分'}")

    # 必要性比：只在 |b_ghost| 过门槛的事件上算（分母守卫）
    use = [r for r in recs if abs(r["b_ghost"]) >= args.min_b]
    out["n_used_for_R"] = len(use); out["min_b_gate"] = args.min_b
    base_sig = out["b_ghost"] and (out["b_ghost"]["ci95"][0] > 0 or out["b_ghost"]["ci95"][1] < 0)
    out["baseline_response_significant"] = bool(base_sig)
    if len(use) >= 20:
        Rv = [1.0 - r["b_occ"] / r["b_ghost"] for r in use]
        out["necessity_ratio"] = boot_scene(Rv, [r["scene"] for r in use])
        Rc = [1.0 - r["b_ctrl"] / r["b_ghost"] for r in use]
        out["necessity_ratio_control"] = boot_scene(Rc, [r["scene"] for r in use])
    if not base_sig:
        out["verdict"] = ("不可估：**基线响应本身与 0 不可区分**（b_ghost 的 CI 跨 0）⇒ "
                          "没有可供必要性检验的响应。这不是必要性检验失败，是无从检验。")
    elif "necessity_ratio" not in out:
        out["verdict"] = f"不可估：过门槛事件仅 {len(use)}"
    else:
        ci = out["necessity_ratio"]["ci95"]
        out["verdict"] = ("PASS：遮住关键实体后动作显著退回基线 ⇒ 动作确由该实体的可见性因果驱动"
                          if ci[0] > 0.5 else
                          ("FAIL：遮住关键实体后动作基本不变 ⇒ 驱动动作的不是这个实体（盲目泛化签名）"
                           if ci[1] < 0.5 else "不可估：必要性比的 scene 级 CI 跨 0.5"))
        r = out["necessity_ratio"]; c = out.get("necessity_ratio_control")
        print(f"[F3/{LABEL}] **必要性比 R = {r['mean']:+.3f}**  CI {np.round(r['ci95'],3).tolist()}  "
              f"（n={r['n']}，门槛 |b_ghost| ≥ {args.min_b}）")
        if c:
            print(f"[F3/{LABEL}] 对照臂 R_ctrl = {c['mean']:+.3f}  CI {np.round(c['ci95'],3).tolist()}"
                  f"  ← 应接近 0（涂别处不该让动作退回）")
    print(f"[F3/{LABEL}] 判定：{out['verdict']}")
    if args.occlude_lidar:
        po = np.array(LID_STATS["n_pts_occ"], float); pc = np.array(LID_STATS["n_pts_ctrl"], float)
        out["lidar_removal"] = {
            "events": LID_STATS["events"], "no_box3d": LID_STATS["no_box3d"],
            "control_volume": "实体 3D 框沿 ego 纵轴镜像（y -> −y, yaw -> −yaw），体积严格相等",
            "margin_m": float(args.lidar_margin),
            "n_points_removed_occ": {"mean": float(po.mean()) if len(po) else None,
                                     "median": float(np.median(po)) if len(po) else None,
                                     "n": int(len(po))},
            "n_points_removed_ctrl": {"mean": float(pc.mean()) if len(pc) else None,
                                      "median": float(np.median(pc)) if len(pc) else None,
                                      "n": int(len(pc))},
            "note": "两臂的删点数应同量级；若 ctrl 臂显著更少，说明镜像位置落在空旷处，"
                    "「删除范围」未完全匹配，须在报告中如实写出"}
        print(f"[F3/{LABEL}] lidar 删点：occ 均值 {po.mean():.0f} / 中位 {np.median(po):.0f}；"
              f"ctrl 均值 {pc.mean():.0f} / 中位 {np.median(pc):.0f}"
              f"（{LID_STATS['events']} 事件，3D 框缺失 {LID_STATS['no_box3d']}）")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[F3/{LABEL}] wrote {args.out}")


if __name__ == "__main__":
    main()
