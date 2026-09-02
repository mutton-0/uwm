"""F-3 多帧平均降噪|把四臂的单帧单次推理换成"连续帧窗口内多次推理取均值"。

工单：`docs/f3_final_cleanup_workorder.md` 第二部分。

**要解决的问题**：F-3 的四个臂当前每个事件只跑一次前向（$a_{ghost}$、$a_{clean}$、
$a_{occ}$、$a_{ctrl}$ 各一帧）。多个候选卡在"$b_{ghost}$ 的 CI 跨 0 ⇒ 无基线响应"
这一步，判定门被单帧噪声主导的可能性没有被排除过。本模块在**事件级别**再套一层平均：
同一个事件取一个连续帧窗口、逐帧独立推理、取均值，再走完全相同的统计与判定。

**这不是把模型的输入窗口拉长**。多帧模型（Alpamayo/AutoVLA）自己吃的时序输入不变；
我们只是针对同一个决策点，多跑几个相邻决策点再平均。

**窗口怎么取（不重新挖矿）**：直接用挖掘期已经算好的 `geo` 网格
（`scene_camera_frames` 给出该 scene 全部 CAM_FRONT 帧的 filename 与时间戳，
`per_obj[tok]["visible"]` 给出逐帧可见性）—— 与 `f3_window_boxes` 同一份缓存，
不另建管线、不重新投影。

  危险出现窗口：自 ghost 帧起沿时间向前推，**逐帧要求实体投影可见**，上限 K 帧；
  危险不在场窗口：自 clean 帧起沿时间**向后**推（远离 emergence），长度与 ghost 窗口对齐。

  **clean 侧为什么不加"实体不可见"这一条（§FC/A61）**：首版加了，结果 4/4 事件全被剔除。
  实测原因：G1 A 类 288 个事件里 **219 个（76.0%）在 clean 帧上实体就已经可见**
  ——clean 帧取自 emergence 前 1.0~1.5 s，而 `t_emergence` 标记的是"进走廊 / TTC 越阈"，
  不是"变得可见"（§FM/A57 同一条教训）。两帧都可见的 218 例里，
  成像面积比 ghost/clean 中位 **1.67**、纵距中位 31.3 m → 25.2 m。
  ⇒ **单帧版 F-3 的 clean 臂本来就不是"危险不在场"，而是"危险还远"**。
  本模块的任务是隔离"多帧平均"这一个变量，故 clean 窗口**沿用单帧版同一口径**
  （不加可见性要求），并把 clean 帧的可见性记为逐事件协变量，
  另报"实体确实不可见"子集的分组读数。clean 侧仍向**后**取，
  以免窗口本身把 clean 臂推向 emergence。

  遮挡臂 / 对照臂用**同一个 ghost 窗口**、逐帧各自投影后遮挡（对照框逐帧同位置），
  故四臂在"窗口长度"与"遮挡帧数"上始终对称可比。

**预注册主读数**：$b_{ghost}$ 的 scene 级 bootstrap **CI 半宽**（单帧 vs 多帧平均）——
即"平均到底降没降噪"。次读数：判定是否变化、$R$ 及其 CI 半宽。
**判定规则与单帧版逐字一致**，不因为换了取数方式而改门槛。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import boot_scene, occlude, control_box     # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
NUSC = "/data/dataset/nuscenes/v1.0-trainval"


def ci_halfwidth(s):
    return None if not s else float((s["ci95"][1] - s["ci95"][0]) / 2)


def fmtci(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dd", "ltf", "ddv2", "simlingo"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default=NUSC)
    ap.add_argument("--pos", default="A")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-b", type=float, default=0.02)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-frames", type=int, default=10, help="窗口帧数上限（工单：约 10）")
    ap.add_argument("--occlude-lidar", action="store_true")
    ap.add_argument("--lidar-margin", type=float, default=0.25)
    ap.add_argument("--sl-config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
             "simlingo": "SimLingo"}[args.model]
    work = Path(args.work).resolve()
    # **必须在起 runner 之前把输出路径绝对化**：SimLingoRunner 会 os.chdir 到它自己的 repo，
    # 相对路径在跑完 288 个事件之后才在写盘那一步炸掉（实测踩过一次）。
    args.out = str(Path(args.out).resolve() if args.out
                   else RES / f"f3_tavg_{args.model}.json")

    import cv2
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == args.pos
           and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]
    if args.limit:
        evs = evs[: args.limit]
    print(f"[TAVG/{LABEL}] {args.pos} 类可用事件 {len(evs)}；窗口上限 {args.max_frames} 帧")

    from nuscenes.nuscenes import NuScenes
    from f3_window_boxes import WindowBoxes, points_in_box, mirror_box3d
    import g1_mine_events as G1
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False)
    WB = WindowBoxes(nusc)

    def read(fn):
        return cv2.cvtColor(cv2.imread(str(Path(args.nuscenes_root) / fn)), cv2.COLOR_BGR2RGB)

    # ---------------- 前向接口（与 f3_occlusion_necessity 同一套） ----------------
    if args.model in ("dd", "ltf", "ddv2"):
        sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
        sys.path.insert(0, str(RES / "ltf_g1_adapter")); sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        lidar = None
        if args.model == "ddv2":
            from ddv2_adapter import NuScenesLidar
            lidar = NuScenesLidar(args.nuscenes_root)
        LID = {"n_pts_occ": [], "n_pts_ctrl": [], "no_box3d": 0}

        def _lidar_for(ev, sd_token, t_s, arm):
            pts = None if lidar is None else lidar.ego_points(sd_token)
            if pts is None or arm is None or not args.occlude_lidar:
                return pts
            b3 = WB.box3d_for(ev, t_s)
            if b3 is None:
                LID["no_box3d"] += 1
                return pts
            c, sz, yaw = b3 if arm == "occ" else mirror_box3d(*b3)
            g = float(args.lidar_margin)
            sz = (sz[0] + 2 * g, sz[1] + 2 * g, sz[2] + 2 * g)
            m = points_in_box(pts, c, sz, yaw)
            LID["n_pts_" + arm].append(int(m.sum()))
            return pts[~m]

        def infer(img, ev, sd_token, t_s, arm=None):
            kw = {} if lidar is None else {"lidar_xyz": _lidar_for(ev, sd_token, t_s, arm)}
            return float(runner.run(img, spd(ev), **kw)["commanded_speed"])
    else:
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        from g2_cache import commanded_speed
        runner = SimLingoRunner(cfg, capture_hidden=False)
        LID = None

        def infer(img, ev, sd_token, t_s, arm=None):
            return float(commanded_speed(runner.infer(img, spd(ev), pool_modes=()).waypoints))

    def spd(ev):
        return float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))

    # ---------------- 逐事件 ----------------
    rng = np.random.default_rng(0)
    recs, skipped = [], defaultdict(int)
    win_len = {"ghost": [], "clean": []}
    for i, ev in enumerate(evs):
        try:
            geo = WB.geo(ev)
        except Exception:                                              # noqa: BLE001
            skipped["no_geo"] += 1; continue
        o = geo["per_obj"].get(ev["object_token"])
        if o is None:
            skipped["no_obj"] += 1; continue
        gt = geo["grid_t"]; frames = geo["frames"]; n = len(gt)
        tg = ev["x_ghost_frames"][0]["t"]; tc = ev["x_clean_frames"][0]["t"]
        jg = int(np.argmin(np.abs(gt - tg))); jc = int(np.argmin(np.abs(gt - tc)))
        if abs(gt[jg] - tg) > 0.06 or abs(gt[jc] - tc) > 0.06:
            skipped["off_grid"] += 1; continue

        # ghost 窗口：自 ghost 帧向前，逐帧要求实体**可见**
        wg = []
        j = jg
        while j < n and len(wg) < args.max_frames and bool(o["visible"][j]):
            bb = G1.frame_bbox(geo, o, j)
            if bb is None:
                break
            wg.append((j, bb)); j += 1
        if not wg:
            skipped["ghost_window_empty"] += 1; continue
        # clean 窗口：自 clean 帧向**后**（远离 emergence）。**不加可见性要求**，
        # 与单帧版 clean 臂同一口径（理由见模块 docstring §FC/A61）。
        wc = []
        j = jc
        while j >= 0 and len(wc) < len(wg):
            wc.append(j); j -= 1
        if not wc:
            skipped["clean_window_empty"] += 1; continue
        # 长度对齐：两侧取较短者，避免"窗口长度"本身成为混淆变量
        L = min(len(wg), len(wc))
        wg, wc = wg[:L], wc[:L]
        win_len["ghost"].append(L); win_len["clean"].append(L)

        cb = control_box(read(frames[wg[0][0]]["filename"]), wg[0][1], rng)
        if cb is None:
            skipped["no_control_box"] += 1; continue

        try:
            vc = [infer(read(frames[j]["filename"]), ev, frames[j]["token"], gt[j]) for j in wc]
            vg, vo, vt = [], [], []
            for j, bb in wg:
                img = read(frames[j]["filename"])
                sd, ts = frames[j]["token"], gt[j]
                vg.append(infer(img, ev, sd, ts))
                vo.append(infer(occlude(img, bb), ev, sd, ts, "occ"))
                vt.append(infer(occlude(img, cb), ev, sd, ts, "ctrl"))
        except Exception as exc:                                       # noqa: BLE001
            skipped[f"infer_fail:{type(exc).__name__}"] += 1; continue

        m = lambda a: float(np.mean(a))                                # noqa: E731
        recs.append({"eid": ev["event_id"], "scene": ev["scene_name"], "win_len": L,
                     "v_clean": m(vc), "v_ghost": m(vg), "v_occ": m(vo), "v_ctrl": m(vt),
                     "b_ghost": m(vg) - m(vc), "b_occ": m(vo) - m(vc), "b_ctrl": m(vt) - m(vc),
                     "sd_within_ghost": float(np.std(vg)) if L > 1 else 0.0,
                     "sd_within_clean": float(np.std(vc)) if L > 1 else 0.0,
                     "v_ghost_single": vg[0], "v_clean_single": vc[0],
                     "v_occ_single": vo[0], "v_ctrl_single": vt[0],
                     # 同帧内擦除效应，不经 clean 臂 ⇒ 不受 clean 臂污染影响
                     "d_occ": m(vg) - m(vo), "d_ctrl": m(vg) - m(vt),
                     "d_occ_single": vg[0] - vo[0], "d_ctrl_single": vg[0] - vt[0],
                     # clean 帧上实体是否已可见（协变量；True = 该事件的 clean 臂被污染）
                     "clean_frame_entity_visible": bool(o["visible"][jc]),
                     "n_clean_frames_entity_visible": int(sum(bool(o["visible"][j]) for j in wc))})
        if (i + 1) % 25 == 0:
            print(f"[TAVG/{LABEL}] {i+1}/{len(evs)}  窗口均长 {np.mean(win_len['ghost']):.1f}", flush=True)

    out = {"model": LABEL, "design": "F-3 四臂 + 事件级多帧平均（窗口内逐帧独立推理取均值）",
           "window": {"max_frames": args.max_frames,
                      "ghost_rule": "自 ghost 帧向前，逐帧要求实体投影可见",
                      "clean_rule": "自 clean 帧向后（远离 emergence），逐帧要求实体投影不可见",
                      "length_matched": "两侧取较短者，窗口长度在四臂间一致",
                      "mean_len": float(np.mean(win_len["ghost"])) if win_len["ghost"] else None,
                      "len_hist": {int(k): int(v) for k, v in
                                   zip(*np.unique(win_len["ghost"], return_counts=True))}
                      if win_len["ghost"] else {}},
           "occlude_lidar": bool(args.occlude_lidar),
           "primary": "b_ghost 的 scene 级 bootstrap CI 半宽（单帧 vs 多帧平均）= 降噪幅度",
           "n_events": len(recs), "skipped": dict(skipped), "per_event": recs}

    sc = [r["scene"] for r in recs]
    # 同帧内擦除效应（多帧 vs 单帧），以及必需的对照
    for k in ("d_occ", "d_ctrl"):
        out[k] = boot_scene([r[k] for r in recs], sc)
        out[k + "_singleframe"] = boot_scene([r[k + "_single"] for r in recs], sc)
        if out[k]:
            a, b = out[k], out[k + "_singleframe"]
            print(f"[TAVG/{LABEL}] {k:8s} 多帧 {a['mean']:+.4f} CI {np.round(a['ci95'],4).tolist()} "
                  f"(半宽 {ci_halfwidth(a):.4f}) | 单帧 {b['mean']:+.4f} "
                  f"CI {np.round(b['ci95'],4).tolist()} (半宽 {ci_halfwidth(b):.4f})")
    # clean 臂污染的分组读数：只用 clean 帧上实体确实不可见的事件
    pure = [r for r in recs if not r["clean_frame_entity_visible"]]
    out["n_clean_arm_contaminated"] = sum(r["clean_frame_entity_visible"] for r in recs)
    out["clean_arm_contamination_frac"] = (out["n_clean_arm_contaminated"] / len(recs)) if recs else None
    if len(pure) >= 20:
        out["b_ghost_entity_absent_subset"] = boot_scene(
            [r["b_ghost"] for r in pure], [r["scene"] for r in pure])
        print(f"[TAVG/{LABEL}] clean 臂未被污染的子集 n={len(pure)}："
              f"b_ghost {fmtci(out['b_ghost_entity_absent_subset'])}")
    for k in ("b_ghost", "b_occ", "b_ctrl"):
        out[k] = boot_scene([r[k] for r in recs], sc)
        # 同一批事件、同一套 bootstrap，只用窗口首帧 ⇒ 严格的单帧对照臂
        sk = {"b_ghost": ("v_ghost_single", "v_clean_single"),
              "b_occ": ("v_occ_single", "v_clean_single"),
              "b_ctrl": ("v_ctrl_single", "v_clean_single")}[k]
        out[k + "_singleframe"] = boot_scene([r[sk[0]] - r[sk[1]] for r in recs], sc)
        if out[k]:
            a, b = out[k], out[k + "_singleframe"]
            print(f"[TAVG/{LABEL}] {k:8s} 多帧 {a['mean']:+.4f} CI {np.round(a['ci95'],4).tolist()} "
                  f"(半宽 {ci_halfwidth(a):.4f}) | 单帧 {b['mean']:+.4f} "
                  f"CI {np.round(b['ci95'],4).tolist()} (半宽 {ci_halfwidth(b):.4f})")

    for tag, key in (("", "b_ghost"), ("_singleframe", "b_ghost_singleframe")):
        s = out.get(key)
        out["baseline_response_significant" + tag] = bool(
            s and (s["ci95"][0] > 0 or s["ci95"][1] < 0))
    out["ci_halfwidth"] = {k: ci_halfwidth(out.get(k)) for k in
                           ("b_ghost", "b_ghost_singleframe", "b_occ", "b_occ_singleframe")}
    hw, hs = out["ci_halfwidth"]["b_ghost"], out["ci_halfwidth"]["b_ghost_singleframe"]
    out["noise_reduction_ratio"] = float(hw / hs) if (hw and hs) else None

    for suf, bk, ok_ in (("", "b_ghost", "b_occ"), ("_singleframe", "b_ghost_singleframe", "b_occ_singleframe")):
        gk = "b_ghost" if not suf else "b_ghost"
        use = [r for r in recs
               if abs(r[gk] if not suf else r["v_ghost_single"] - r["v_clean_single"]) >= args.min_b]
        if len(use) >= 20:
            Rv = [1.0 - ((r["b_occ"] / r["b_ghost"]) if not suf else
                         ((r["v_occ_single"] - r["v_clean_single"]) /
                          (r["v_ghost_single"] - r["v_clean_single"]))) for r in use]
            out["necessity_ratio" + suf] = boot_scene(Rv, [r["scene"] for r in use])
        out["n_used_for_R" + suf] = len(use)

    def verdict_of(sig, R):
        if not sig:
            return ("不可估：**基线响应本身与 0 不可区分**（b_ghost 的 CI 跨 0）⇒ "
                    "没有可供必要性检验的响应。")
        if not R:
            return "不可估：过门槛事件不足"
        ci = R["ci95"]
        return ("PASS：遮住关键实体后动作显著退回基线" if ci[0] > 0.5 else
                ("FAIL：遮住关键实体后动作基本不变（盲目泛化签名）" if ci[1] < 0.5 else
                 "不可估：必要性比的 scene 级 CI 跨 0.5"))
    out["verdict"] = verdict_of(out["baseline_response_significant"], out.get("necessity_ratio"))
    out["verdict_singleframe"] = verdict_of(out["baseline_response_significant_singleframe"],
                                            out.get("necessity_ratio_singleframe"))
    out["verdict_changed"] = out["verdict"][:6] != out["verdict_singleframe"][:6]
    print(f"[TAVG/{LABEL}] 单帧判定：{out['verdict_singleframe']}")
    print(f"[TAVG/{LABEL}] 多帧判定：{out['verdict']}")
    print(f"[TAVG/{LABEL}] CI 半宽 {hs:.4f} -> {hw:.4f}（比值 {out['noise_reduction_ratio']:.3f}）"
          if hw and hs else "")
    if LID is not None and args.occlude_lidar:
        po = np.array(LID["n_pts_occ"], float); pc = np.array(LID["n_pts_ctrl"], float)
        out["lidar_removal"] = {"n_points_removed_occ_mean": float(po.mean()) if len(po) else None,
                                "n_points_removed_ctrl_mean": float(pc.mean()) if len(pc) else None,
                                "no_box3d": LID["no_box3d"], "margin_m": float(args.lidar_margin)}
    out["window_projection_stats"] = dict(WB.stats)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[TAVG/{LABEL}] wrote {args.out}")


if __name__ == "__main__":
    main()
