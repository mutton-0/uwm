"""LB2|前车急刹场景：套阈值 + caliper 匹配负例 + 落盘 events_all.jsonl。

工单：docs/severity_gradient_and_second_scenario_workorder.md 任务二。
输入：`lb1_mine_lead_brake.py` 落盘的候选池 `_lb_pool.npy`。
输出：`variants/lead_brake/mining/events_all.jsonl`，**schema 与 G1 逐字段相同**，
      因此下游全部适配器（SimLingo / DiffusionDrive / LTF）与全部分析脚本零改动复用。

事件族：
  LB    正例：走廊内前车大幅减速
  LBn   几何/运动学匹配负例（对应 D2a）：同为走廊内前车，距离与车速 caliper 匹配，但没有减速
  LBv   证伪地板（对应 D2cV）：距离与车速匹配、**同样在减速但幅度小**
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_mine_events import compute_scene_geometry, pick_frames, frame_record   # noqa: E402

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")


def caliper_match(pos, neg, cal_d, cal_v, cal_e):
    """在 (log d_long, log lead_speed, log ego_speed) 上做 caliper 匹配（1:1，无放回）。

    与 N1 的 D2a 匹配同一口径：对数尺度 + caliper 容差 + 贪心最近邻。
    匹配失败的正例**不删**（正例集合独立于负例可得性），只记录匹配率。

    **ego 速度是本场景必须匹配的第三个变量**（首版只匹配了距离与车速，
    实测 LB vs LBn 的 ego 速度 SMD = +0.600，见 §LB/A42）：
    前车急刹与自车快本就相关，而自车速度**强驱动**各模型的下发速度，
    不匹配的话 G/F 上测到的差异会混入"这一组自车本来就开得快"。
    """
    def key(c):
        return np.array([np.log(max(c["d_long_at_onset"], 1.0)),
                         np.log(max(c["lead_speed_mps"], 0.5)),
                         np.log(max(c["ego_speed_mps"], 0.5))])
    # **最优 1:1 指派（匈牙利算法）而非贪心最近邻**（§LB/A42）：
    # 贪心的结果依赖遍历顺序（要靠随机种子），且在负例池小的时候容易把好负例先用掉，
    # 使后续正例只能配到边缘样本 —— 实测这正是 LBv 的 d_long SMD 达到 −0.33 的原因。
    # 最优指派最小化总配对距离，无顺序依赖、可复现，且在同一 caliper 下平衡显著更好。
    from scipy.optimize import linear_sum_assignment
    P = np.array([key(c) for c in pos]); N = np.array([key(c) for c in neg])
    if len(N) == 0:
        return [], []
    D = np.abs(P[:, None, :] - N[None, :, :])                      # [nP, nN, 3]
    feas = (D[:, :, 0] <= cal_d) & (D[:, :, 1] <= cal_v) & (D[:, :, 2] <= cal_e)
    cost = np.sqrt((D ** 2).sum(-1))
    BIG = 1e6
    cost = np.where(feas, cost, BIG)
    ri, ci = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(cost[i, j])) for i, j in zip(ri, ci) if cost[i, j] < BIG / 2]
    return pairs, sorted({j for _i, j, _d in pairs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "lead_brake.yaml"))
    # 阈值：由 lb1 --survey 的全局分布 + 物理合理性共同确定（见挖掘报告"阈值回调记录"）
    ap.add_argument("--a-brake-max", type=float, default=-3.0)
    ap.add_argument("--a-brake-floor", type=float, default=-10.0)
    ap.add_argument("--a-mild", type=float, nargs=2, default=[-1.5, -0.3])
    ap.add_argument("--a-steady-min", type=float, default=-0.2)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    lbc = cfg["lead_brake"]
    work = Path(cfg["paths"]["work_dir"]); out_dir = work / "mining"
    pool = list(np.load(out_dir / "_lb_pool.npy", allow_pickle=True))
    a = np.array([c["a_min_window"] for c in pool])

    LB = [c for c, x in zip(pool, a) if args.a_brake_floor <= x <= args.a_brake_max]
    LBv = [c for c, x in zip(pool, a) if args.a_mild[0] <= x <= args.a_mild[1]]
    LBn = [c for c, x in zip(pool, a) if x >= args.a_steady_min]
    dropped_implausible = int((a < args.a_brake_floor).sum())
    print(f"[LB2] 阈值后 LB={len(LB)}  LBv={len(LBv)}  LBn={len(LBn)}  "
          f"（物理不合理丢弃 {dropped_implausible}）")

    ce = lbc.get("caliper_ego_speed", 0.30)
    pn, un = caliper_match(LB, LBn, lbc["caliper_d_long"], lbc["caliper_speed"], ce)
    pv, uv = caliper_match(LB, LBv, lbc["caliper_d_long"], lbc["caliper_speed"], ce)
    LBn_m = [LBn[j] for j in un]; LBv_m = [LBv[j] for j in uv]
    print(f"[LB2] caliper 匹配后 LBn={len(LBn_m)}（匹配率 {len(pn)}/{len(LB)}）  "
          f"LBv={len(LBv_m)}（匹配率 {len(pv)}/{len(LB)}）")

    keep = {"LB": LB, "LBn": LBn_m, "LBv": LBv_m}
    by_scene = defaultdict(list)
    for kind, lst in keep.items():
        for c in lst:
            by_scene[c["scene_name"]].append((kind, c))

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=cfg["paths"]["nuscenes_version"],
                    dataroot=cfg["paths"]["nuscenes_root"], verbose=False)
    scene_by_name = {s["name"]: s for s in nusc.scene}

    events, skipped = [], defaultdict(int)
    for si, (sname, items) in enumerate(sorted(by_scene.items())):
        geo = compute_scene_geometry(nusc, scene_by_name[sname], cfg)
        for ei, (kind, c) in enumerate(items):
            o = geo["per_obj"].get(c["inst"])
            if o is None:
                skipped["no_obj"] += 1; continue
            clean, ghost = pick_frames(geo, c["t_onset"], cfg)
            if not clean or not ghost:
                skipped["no_frames"] += 1; continue
            k = c["idx"]
            rec = {
                "event_id": f"{sname}_{ei:03d}_{kind}",
                "scene_token": c["scene_token"], "scene_name": sname, "event_type": kind,
                # 与 G1 同名字段：t_emergence 在本场景里是**刹车起始时刻**
                "t_emergence": c["t_onset"],
                "object_token": c["inst"], "object_class": c["object_class"],
                # 本场景独有的严重度量
                "a_min_window": c["a_min_window"], "a_min_pre": c["a_min_pre"],
                "dv_window": c["dv_window"], "lead_speed_mps": c["lead_speed_mps"],
                # 与 G1 同名字段（下游脚本按名取用，故必须齐备）
                "min_ttc_1s": None,
                "ttc_at_emergence": float(min(geo["frame_ttc"][k], 99.0)),
                "d_long_at_emergence": float(o["d_long"][k]),
                "lat_at_emergence": float(o["lat"][k]),
                "area_px": (lambda z: None if not np.isfinite(z) else float(z))(
                    np.nanmedian(o["area_px"][ghost]) if len(ghost) else np.nan),
                "ecc": (lambda z: None if not np.isfinite(z) else float(z))(
                    np.nanmedian(o["ecc"][ghost]) if len(ghost) else np.nan),
                "is_night": bool(c["is_night"]), "is_rain": bool(c["is_rain"]),
                "ego_speed_mps": c["ego_speed_mps"],
                "x_clean_frames": [frame_record(geo, j, o) for j in clean],
                "x_ghost_frames": [frame_record(geo, j, o) for j in ghost],
            }
            events.append(rec)
        if (si + 1) % 40 == 0:
            print(f"[LB2] {si+1}/{len(by_scene)} scene，事件累计 {len(events)}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "events_all.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for e in events:
        cnt[e["event_type"]] += 1
    # 匹配质检：正例 vs 各负例在匹配变量上的分布差（SMD），与 N1 的质检口径一致
    def smd(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
        return float((x.mean() - y.mean()) / s) if s > 1e-9 else 0.0
    g = defaultdict(list)
    for e in events:
        g[e["event_type"]].append(e)
    qc = {}
    for neg in ("LBn", "LBv"):
        if not g[neg]:
            continue
        qc[neg] = {v: smd([e[v] for e in g["LB"]], [e[v] for e in g[neg]])
                   for v in ("d_long_at_emergence", "lead_speed_mps", "ego_speed_mps")}
        qc[neg]["area_px"] = smd([e["area_px"] for e in g["LB"] if e["area_px"]],
                                 [e["area_px"] for e in g[neg] if e["area_px"]])
        qc[neg]["a_min_window"] = smd([e["a_min_window"] for e in g["LB"]],
                                      [e["a_min_window"] for e in g[neg]])
    rep = {"thresholds": {"a_brake_max": args.a_brake_max, "a_brake_floor": args.a_brake_floor,
                          "a_mild": args.a_mild, "a_steady_min": args.a_steady_min,
                          "caliper_d_long": lbc["caliper_d_long"],
                          "caliper_speed": lbc["caliper_speed"],
                          "caliper_ego_speed": ce},
           "n_pool": len(pool), "dropped_implausible_a": dropped_implausible,
           "n_after_threshold": {k: len(v) for k, v in
                                 {"LB": LB, "LBv": LBv, "LBn": LBn}.items()},
           "match_rate": {"LBn": f"{len(pn)}/{len(LB)}", "LBv": f"{len(pv)}/{len(LB)}"},
           "n_events_written": dict(cnt), "skipped": dict(skipped),
           "n_scenes": len(set(e["scene_name"] for e in events)),
           "match_qc_smd": qc,
           "smd_note": "|SMD| < 0.1 视为匹配良好；a_min_window 的 SMD **应当大**（那正是唯一差异）"}
    (out_dir / "lead_brake_mining_stats.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n[LB2] 落盘 {len(events)} 事件：{dict(cnt)}，{rep['n_scenes']} scene")
    for neg, v in qc.items():
        print(f"[LB2] 匹配质检 LB vs {neg} SMD：" +
              "  ".join(f"{k}={x:+.3f}" for k, x in v.items()))
    print(f"[LB2] wrote {out_dir/'events_all.jsonl'}")


if __name__ == "__main__":
    main()
