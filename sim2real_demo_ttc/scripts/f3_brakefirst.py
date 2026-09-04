"""F-3 on brake-first 语料|四臂 + 群体遮挡 + 与归因一致的 clean 臂定义。

工单：2026-09-04（用户："基于新的数据和遮挡，完全重跑 SimLingo 的 F3"）。

## 与旧 F-3 的三处不同（都是前面几轮的结论倒逼的）

1. **事件来源**：`brake_first_pool_final.json`（13 事件/13 scene），
   查询帧是**人类减速起点**，不是 `t_emergence`。
2. **遮挡组**：遮走廊内**全部危险类**实例（含 animal），不是单个目标。
   只遮一个、剩下的还在 ⇒ 必测出假 FAIL。
3. **clean 臂重新定义**：旧口径的 clean 帧 **76% 已经能看见实体**（§FC/A61），
   $b_{ghost}$ 量的其实是"危险走近"而非"危险出现"。
   本模块把 clean 定义为**同 scene 内走廊前方没有任何危险类目标的最近帧** ——
   与归因判据同一套语义（"没有东西要求刹车"），并逐事件审计其真实洁净度。

## 四臂

| 臂 | 图像 | 说明 |
| clean | 无危险帧原图 | 走廊前方无危险类目标 |
| ghost | 减速起点原图 | 危险在场 |
| occ | ghost + 遮挡组**全部**框涂灰 | 移除危险 |
| ctrl | ghost + 等面积对照框涂灰 | 隔离"加灰斑"本身 |

**四臂共用同一个喂入自车速度**（锚在 ghost 帧）—— 配对相减时抵消，
见 speed_override_report §4.2：跨事件背景速度差异从未进入配对差。

## 读数

* 跨帧（有 clean 臂）：$b_x = v_x - v_{clean}$，$R = 1 - b_{occ}/b_{ghost}$
* **同帧（不依赖 clean 臂）**：$\\delta_{occ} = v_{occ} - v_{ghost}$（擦掉危险 ⇒ 预期提速 > 0）、
  $\\delta_{ctrl} = v_{ctrl} - v_{ghost}$（预期 ≈ 0）、配对 $|\\delta_{occ}|-|\\delta_{ctrl}|$
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from f3_occlusion_necessity import boot_scene, occlude, control_box    # noqa: E402

HAZARD_PREFIX = ("human.", "vehicle.bicycle", "vehicle.motorcycle", "animal")
LIDAR_MARGIN = 0.25          # 与 §FM/A56-59 的主口径一致


def box3d_from_geo(geo, token, j):
    """该目标在第 j 帧的 **ego 系 3D 框**：(center[3], (w,l,h), yaw)。

    与 `f3_window_boxes.WindowBoxes.box3d_for` 逐行同源，只是直接用帧下标而非时间戳查找 ——
    保证"删掉的 3D 框"与"涂掉的 2D 框"作用在同一个物理实体上。
    """
    from pyquaternion import Quaternion
    o = geo["per_obj"].get(token)
    if o is None or not o.get("_rot") or o["_size"] is None or not bool(o["valid"][j]):
        return None
    ti = int(np.argmin(np.abs(np.asarray(o["_rot_t"]) - geo["grid_t"][j])))
    yaw_s = Quaternion(o["_rot"][ti]).yaw_pitch_roll[0]
    yaw_e = yaw_s - Quaternion(matrix=geo["R_we"][j]).yaw_pitch_roll[0]
    w, l, h = (list(o["_size"]) + [0, 0, 0])[:3]
    c = np.asarray(o["p_ego"][j], float)
    if not np.all(np.isfinite(c)):
        return None
    return c, (float(w), float(l), float(h)), float(yaw_e)


def _overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def control_boxes_for_group(img, boxes, rng):
    """给遮挡组里**每个**框配一个等面积对照框。

    要求：与**任何**遮挡框不重叠、与已放置的对照框不重叠、在画幅内。
    先用 `control_box` 的水平镜像/垂直位移候选；不够再沿水平方向多试几个位移 ——
    首版只用 control_box 的候选，导致 scene-0187（2 个遮挡框挤在画幅中央）
    一个对照框都放不下、ctrl 臂退化成与 origin 逐位相同。
    """
    H, W = img.shape[:2]
    out = []
    for bb in boxes:
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        cands = []
        c0 = control_box(img, bb, rng)
        if c0 is not None:
            cands.append(c0)
        for dx in (2 * w, -2 * w, 3 * w, -3 * w, 4 * w, -4 * w, 6 * w, -6 * w):
            cands.append((x0 + dx, y0, x1 + dx, y1))
        for dy in (-2 * h, 2 * h, -3 * h, 3 * h):
            for dx in (0, 2 * w, -2 * w):
                cands.append((x0 + dx, y0 + dy, x1 + dx, y1 + dy))
        for c in cands:
            cx0, cy0, cx1, cy1 = c
            if cx0 < 0 or cy0 < 0 or cx1 > W or cy1 > H:
                continue
            if any(_overlap(c, b) for b in boxes) or any(_overlap(c, o) for o in out):
                continue
            out.append(list(c)); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simlingo", choices=["simlingo", "dd", "ltf", "ddv2"])
    ap.add_argument("--pool", default=str(RES / "brake_first_pool_final.json"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sl-config", default=str(ROOT / "configs" / "n1_d2.yaml"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.out = str(Path(args.out).resolve() if args.out
                   else RES / f"f3_brakefirst_{args.model}.json")

    import cv2
    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    G1.set_include_animal(True)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
    scmap = {s["name"]: s for s in nusc.scene}
    pool = json.load(open(args.pool))
    cands = pool["candidates"]
    print(f"[F3BF/{args.model}] {len(cands)} 事件 / {pool['n_scenes']} scene")

    # ---------------- runner ----------------
    if args.model == "simlingo":
        cfg_sl = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg_sl["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        from g2_cache import commanded_speed
        runner = SimLingoRunner(cfg_sl, capture_hidden=False)

        def infer(img, spd):
            r = runner.infer(img, spd, pool_modes=())
            wp = np.asarray(r.waypoints, float)
            return float(commanded_speed(wp)), wp.tolist()
    else:
        sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
        sys.path.insert(0, str(RES / "ltf_g1_adapter"))
        sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
        import dd_adapter as DD                                        # noqa: F401
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        lidar = None
        if args.model == "ddv2":
            # DDv2 吃点云 —— 只遮 RGB 不删点，等于"危险从图像消失但雷达还在"，
            # 与前几轮确立的口径（§FM/A56-59）不一致。故此处一并做 3D 框删点。
            from ddv2_adapter import NuScenesLidar
            lidar = NuScenesLidar(NUSC)

        def infer(img, spd, pts=None):
            kw = {} if lidar is None else {"lidar_xyz": pts}
            o = runner.run(img, spd, **kw)
            return float(o["commanded_speed"]), np.asarray(o["trajectory"], float).tolist()

    rng = np.random.default_rng(0)
    recs, skipped, audit = [], defaultdict(int), []
    for i, c in enumerate(cands):
        geo = G1.compute_scene_geometry(nusc, scmap[c["scene"]], cfg)
        j = c["frame_idx"]
        img_o = cv2.cvtColor(cv2.imread(str(Path(NUSC) / geo["frames"][j]["filename"])),
                             cv2.COLOR_BGR2RGB)
        boxes = []
        for g in c["f3_mask_group"]:
            o = geo["per_obj"].get(g["token"])
            bb = G1.frame_bbox(geo, o, j) if o is not None else None
            if bb is not None:
                boxes.append(list(bb))
        if not boxes:
            skipped["no_box"] += 1; continue
        cboxes = control_boxes_for_group(img_o, boxes, rng)
        if len(cboxes) < len(boxes):
            skipped["control_box_short"] += 1
        if not cboxes:
            skipped["no_control_box"] += 1; continue     # ctrl 臂会退化成与 origin 相同
        clean_img, ctrl_img = img_o.copy(), img_o.copy()
        for bb in boxes:
            clean_img = occlude(clean_img, bb)
        for cb in cboxes:
            ctrl_img = occlude(ctrl_img, cb)

        # ---- DDv2 的点云三臂：origin 原样，clean 删遮挡组 3D 框内的点，
        #      ctrl 删**镜像 3D 框**内的点（体积严格相等、纵向距离相同）----
        pts_o = pts_c = pts_t = None
        n_del_c = n_del_t = 0
        if lidar is not None:
            from f3_window_boxes import points_in_box, mirror_box3d
            # geo["frames"][j] 的 sample_data token 键名是 "token"（g1_mine_events.py:71），
            # 不是 "sd_token"（那是事件 x_*_frames 里的命名）
            pts_o = lidar.ego_points(geo["frames"][j]["token"])
            pts_c, pts_t = pts_o, pts_o
            if pts_o is not None:
                mc = np.zeros(len(pts_o), bool); mt = np.zeros(len(pts_o), bool)
                for g in c["f3_mask_group"]:
                    b3 = box3d_from_geo(geo, g["token"], j)
                    if b3 is None:
                        continue
                    cc, sz, yaw = b3
                    gsz = tuple(x + 2 * LIDAR_MARGIN for x in sz)
                    mc |= points_in_box(pts_o, cc, gsz, yaw)
                    cm, szm, yawm = mirror_box3d(cc, sz, yaw)
                    mt |= points_in_box(pts_o, cm,
                                        tuple(x + 2 * LIDAR_MARGIN for x in szm), yawm)
                pts_c = pts_o[~mc]; pts_t = pts_o[~mt]
                n_del_c, n_del_t = int(mc.sum()), int(mt.sum())

        spd = float(geo["ego_speed"][j])          # 三臂共用
        v_origin, tj_origin = infer(img_o, spd, pts_o)
        v_clean, tj_clean = infer(clean_img, spd, pts_c)
        v_ctrl, tj_ctrl = infer(ctrl_img, spd, pts_t)

        area = lambda bs: sum((b[2]-b[0])*(b[3]-b[1]) for b in bs)
        audit.append({"scene": c["scene"], "n_mask": len(boxes), "n_ctrl": len(cboxes),
                      "mask_area_px": area(boxes), "ctrl_area_px": area(cboxes),
                      "area_ratio_ctrl_over_mask": round(area(cboxes)/max(area(boxes),1), 3)})
        recs.append({"scene": c["scene"], "eid": c["scene"], "frame_idx": j,
                     "n_mask": len(boxes), "n_ctrl": len(cboxes),
                     "n_lidar_del_clean": n_del_c, "n_lidar_del_ctrl": n_del_t,
                     "ego_v0": spd, "a_req": c["a_vru_max"],
                     "v_origin": v_origin, "v_clean": v_clean, "v_ctrl": v_ctrl,
                     "b": v_clean - v_origin, "b_ctrl": v_ctrl - v_origin,
                     # **完整规划轨迹**：commanded_speed 只用了 0.25-0.75s 两个点，
                     # 而 SimLingo 轨迹是 10 点 x 0.25s = 2.5s。全轨迹读数需要它。
                     "traj_origin": tj_origin, "traj_clean": tj_clean, "traj_ctrl": tj_ctrl})
        print(f"  [{i+1}/{len(cands)}] {c['scene']:14s} 遮{len(boxes)}/对照{len(cboxes)}  "
              f"origin {v_origin:.3f} | clean {v_clean:.3f} | ctrl {v_ctrl:.3f}  "
              f"b={v_clean-v_origin:+.3f}")

    sc = [r["scene"] for r in recs]
    out = {"model": args.model, "pool": Path(args.pool).name,
           "n_events": len(recs), "n_scenes": len(set(sc)), "skipped": dict(skipped),
           "design": "三臂同帧：origin(原图) / clean(遮挡组全遮) / ctrl(等面积对照灰斑)",
           "primary_readout": "b = v_clean - v_origin，预期 > 0（移除危险 ⇒ 敢开快点）",
           "mask_audit": audit, "per_event": recs}
    for k in ("b", "b_ctrl"):
        out[k] = boot_scene([r[k] for r in recs], sc)
    out["paired_specificity"] = boot_scene(
        [abs(r["b"]) - abs(r["b_ctrl"]) for r in recs], sc)
    n_pos = sum(1 for r in recs if r["b"] > 0)
    out["sign_consistency"] = {"n_positive": n_pos, "n": len(recs),
                               "frac": round(n_pos / max(len(recs), 1), 3),
                               "note": "b>0 = 移除危险后提速 = 预期方向"}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    g = lambda s: "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    print(f"\n[F3BF/{args.model}] n={len(recs)}/{len(set(sc))}scene  skipped={dict(skipped)}")
    for k in ("b", "b_ctrl", "paired_specificity"):
        print(f"  {k:22s} {g(out[k])}")
    print(f"  方向一致率 (b>0)      {out['sign_consistency']['n_positive']}"
          f"/{out['sign_consistency']['n']} = {out['sign_consistency']['frac']*100:.1f}%")
    print(f"[F3BF] wrote {args.out}")


if __name__ == "__main__":
    main()
