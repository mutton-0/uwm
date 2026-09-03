"""F-3 距离读数|把 action 从"指令速度"换成"轨迹终点到危险实体的距离"。

工单：`docs/`（本轮工单，2026-09-03）。**不改动速度版的任何产物**，新结果一律 `_dist` 后缀。

## 为什么换读数

现有的 $v_{plan}$（commanded speed）只由轨迹**第一步**算出，是最短时窗的瞬时速度指令；
除第一步以外的规划信息全被丢掉。若模型看见危险后的反应体现在**改路径**
（往旁边挪、调整终点落在哪）而不是"立刻改下一步速度"，速度读数测不出来。
距离是**零阶空间量**，速度差是二阶量，前者信噪比应更好。

## 记号与符号约定（**与速度版不同，不要混用**）

对每个臂 $a \\in \\{clean, ghost, occ, ctrl\\}$，取该臂规划轨迹的**终点**（时域末端 waypoint，
不是第一步）$w^a_{-1}$，与危险实体在**该臂所查询的那一帧**的 ego 系位置 $p_{ent}$：

$$d_{plan}^{a} = \\lVert w^{a}_{-1}[:2] - p_{ent}[:2] \\rVert_2$$

* $b^{(d)}_{ghost} = d^{ghost}_{plan} - d^{clean}_{plan}$ —— **预期为正**（看见危险 ⇒ 终点离它更远）。
  **注意与速度版相反**：速度版 $b_{ghost}=v(ghost)-v(clean)$ 预期为**负**（看见危险 ⇒ 减速）。
* $b^{(d)}_{occ} = d^{occ}_{plan} - d^{clean}_{plan}$，$b^{(d)}_{ctrl}$ 同理。
* **必要性比**（与速度版**同一函数形式**，故 PASS 门槛 0.5 原样沿用）：

$$R^{(d)} = 1 - \\frac{b^{(d)}_{occ}}{b^{(d)}_{ghost}}
          = \\frac{b^{(d)}_{ghost}-b^{(d)}_{occ}}{b^{(d)}_{ghost}}
          = \\frac{d^{ghost}_{plan}-d^{occ}_{plan}}{b^{(d)}_{ghost}}$$

  $R^{(d)}$ 是两个同号量之比，**不受上面那处符号翻转影响**：
  遮挡完全移除响应 ⇒ $d^{occ}\\approx d^{clean}$ ⇒ $b^{(d)}_{occ}\\approx 0$ ⇒ $R^{(d)}\\approx 1$。

## 三层读数，控制程度依次变好（报告里必须分开写）

1. **$b^{(d)}_{ghost}$（工单指定的主读数）**：ghost 帧对 clean 帧。
   **带一处几何混淆**：两帧的实体 ego 位置本来就不同（G1 上纵距中位 31.3 m → 25.2 m）。
   若规划在 ego 系里**完全不变**，光凭实体走近这一项就会让 $d_{plan}$ 变小 ⇒ $b^{(d)}_{ghost}$ 偏负。
   **该偏置的方向与待测效应（正）相反 ⇒ 主读数是一个保守检验**：测出显著正值是强证据，
   测不出则部分可归因于该偏置。本模块把这一项**显式算出来**（见 2）。
2. **$b^{(d),adj}_{ghost} = d^{ghost}_{plan} - d^{null}$**，其中
   $d^{null} = \\lVert w^{clean}_{-1}[:2] - p_{ent}(t_{ghost})[:2]\\rVert_2$
   —— 拿 **clean 臂的轨迹**去量**ghost 帧的实体位置**。
   两者参照同一个点 ⇒ 只剩"两个规划是从不同 ego 起点做出的"这一残留（见报告限度）。
   $b^{(d),null}_{ghost} = d^{null} - d^{clean}_{plan}$ 即纯几何那一项的大小。
3. **$\\delta_{occ} = d^{occ}_{plan} - d^{ghost}_{plan}$（控制最好，预期为负）**：
   occ / ghost **同一帧、同一参照点、同一 ego 起点**，唯一差异是那块灰斑。
   $\\delta_{ctrl}$ 为必需对照。注意 $R^{(d)} = -\\delta_{occ}/b^{(d)}_{ghost}$ ——
   **分子天然无混淆，被混淆的只是分母**（与速度版 §FC/A61 的结构完全相同）。

判定与统计纪律照旧：scene 级 bootstrap、三态判定、分母守卫、ctrl 臂必需。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import boot_scene                          # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def sig(s):
    return bool(s and (s["ci95"][0] > 0 or s["ci95"][1] < 0))


def fmt(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def endpoint(traj):
    a = np.asarray(traj, float)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 2:
        return None
    p = a[-1, :2]
    return p if np.all(np.isfinite(p)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["dd", "ltf", "ddv2", "simlingo", "alpa", "autovla"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--min-b", type=float, default=0.5,
                    help="分母守卫：|b^(d)_ghost| 门槛（米）。距离是零阶量，量纲与速度版不同，"
                         "故门槛另设，不沿用速度版的 0.02 m/s")
    ap.add_argument("--out", default=str(RES / "f3_distance_readout.json"))
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from f3_window_boxes import WindowBoxes
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False)
    WB = WindowBoxes(nusc)
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(Path(args.work) / "mining" / "events_all.jsonl")}

    LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
             "simlingo": "SimLingo", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}
    out_rows, ent_cache = [], {}

    def ego_disp(ev, t0, t1):
        """两帧之间 ego 自身的位移（米）—— 跨帧比较无法消除的残留混淆，必须报出来。"""
        try:
            geo = WB.geo(ev)
        except Exception:                                              # noqa: BLE001
            return None
        gt = geo["grid_t"]
        j0, j1 = int(np.argmin(np.abs(gt - float(t0)))), int(np.argmin(np.abs(gt - float(t1))))
        if max(abs(gt[j0] - t0), abs(gt[j1] - t1)) > 0.06:
            return None
        a = np.asarray(geo["frames"][j0]["ego_t"], float)[:2]
        b = np.asarray(geo["frames"][j1]["ego_t"], float)[:2]
        return float(np.linalg.norm(b - a))

    def ent_xy(ev, t):
        k = (ev["event_id"], round(float(t), 3))
        if k not in ent_cache:
            b3 = WB.box3d_for(ev, float(t))
            ent_cache[k] = None if b3 is None else np.asarray(b3[0], float)[:2]
        return ent_cache[k]

    for m in args.models:
        p = RES / f"f3_occlusion_{m}_dist.json"
        if not p.exists():
            print(f"[DIST] 缺 {p.name}，跳过"); continue
        d = json.load(open(p))
        if not d.get("save_traj"):
            print(f"[DIST] {p.name} 没有轨迹字段，跳过"); continue

        recs, skip = [], defaultdict(int)
        for r in d["per_event"]:
            ev = evmap.get(r["eid"])
            if ev is None:
                skip["no_event"] += 1; continue
            wg, wc = endpoint(r.get("traj_ghost")), endpoint(r.get("traj_clean"))
            wo, wt = endpoint(r.get("traj_occ")), endpoint(r.get("traj_ctrl"))
            if wg is None or wo is None or wt is None:
                skip["bad_traj_ghost_side"] += 1; continue
            pg = ent_xy(ev, r["t_ghost"])
            if pg is None:
                skip["no_entity_pos_ghost"] += 1; continue
            rec = {"eid": r["eid"], "scene": r["scene"],
                   "d_ghost": float(np.linalg.norm(wg - pg)),
                   "d_occ": float(np.linalg.norm(wo - pg)),
                   "d_ctrl": float(np.linalg.norm(wt - pg))}
            # 控制最好的一层：同帧、同参照点、同 ego 起点
            rec["delta_occ"] = rec["d_occ"] - rec["d_ghost"]
            rec["delta_ctrl"] = rec["d_ctrl"] - rec["d_ghost"]
            pc = ent_xy(ev, r["t_clean"]) if wc is not None else None
            if wc is not None and pc is not None:
                rec["d_clean"] = float(np.linalg.norm(wc - pc))
                rec["b_ghost_d"] = rec["d_ghost"] - rec["d_clean"]
                rec["b_occ_d"] = rec["d_occ"] - rec["d_clean"]
                rec["b_ctrl_d"] = rec["d_ctrl"] - rec["d_clean"]
                # 纯几何那一项：clean 臂的轨迹 vs ghost 帧的实体位置
                rec["d_null"] = float(np.linalg.norm(wc - pg))
                rec["b_ghost_d_geom"] = rec["d_null"] - rec["d_clean"]
                rec["b_ghost_d_adj"] = rec["d_ghost"] - rec["d_null"]
                rec["ent_dlong_clean"] = float(pc[0]); rec["ent_dlong_ghost"] = float(pg[0])
                rec["ego_disp_clean_to_ghost"] = ego_disp(ev, r["t_clean"], r["t_ghost"])
            else:
                skip["no_clean_side"] += 1
            recs.append(rec)

        sc = [r["scene"] for r in recs]
        res = {"model": LABEL[m], "src": p.name, "n_events": len(recs), "skipped": dict(skip),
               "traj_len": int(np.asarray(d["per_event"][0]["traj_ghost"]).shape[0]),
               "readout": "d_plan = ||轨迹终点 − 危险实体 ego 位置||₂（米）",
               "sign_convention": ("b^(d)_ghost = d(ghost) − d(clean)，**预期为正**"
                                   "（与速度版相反：速度版预期为负）；"
                                   "δ_occ = d(occ) − d(ghost)，**预期为负**"),
               "min_b_gate_m": args.min_b}

        has_clean = [r for r in recs if "b_ghost_d" in r]
        res["n_with_clean_arm"] = len(has_clean)
        scc = [r["scene"] for r in has_clean]
        for k in ("b_ghost_d", "b_occ_d", "b_ctrl_d", "b_ghost_d_geom", "b_ghost_d_adj"):
            res[k] = boot_scene([r[k] for r in has_clean], scc) if has_clean else None
        for k in ("delta_occ", "delta_ctrl", "d_ghost"):
            res[k] = boot_scene([r[k] for r in recs], sc)
        eg = [r["ego_disp_clean_to_ghost"] for r in has_clean
              if r.get("ego_disp_clean_to_ghost") is not None]
        res["ego_displacement_clean_to_ghost_m"] = (
            {"median": float(np.median(eg)), "p25": float(np.percentile(eg, 25)),
             "p75": float(np.percentile(eg, 75)), "n": len(eg)} if eg else None)
        res["entity_dlong_m"] = ({"clean_median": float(np.median([r["ent_dlong_clean"] for r in has_clean])),
                                  "ghost_median": float(np.median([r["ent_dlong_ghost"] for r in has_clean]))}
                                 if has_clean else None)
        res["abs_delta_occ_minus_ctrl"] = boot_scene(
            [abs(r["delta_occ"]) - abs(r["delta_ctrl"]) for r in recs], sc)

        # 必要性比（工单指定口径），分母守卫用 |b^(d)_ghost|
        use = [r for r in has_clean if abs(r["b_ghost_d"]) >= args.min_b]
        res["n_used_for_R"] = len(use)
        if len(use) >= 20:
            res["necessity_ratio_d"] = boot_scene(
                [1.0 - r["b_occ_d"] / r["b_ghost_d"] for r in use], [r["scene"] for r in use])
            res["necessity_ratio_d_control"] = boot_scene(
                [1.0 - r["b_ctrl_d"] / r["b_ghost_d"] for r in use], [r["scene"] for r in use])

        base_sig = sig(res["b_ghost_d"])
        res["baseline_response_significant"] = base_sig
        if not base_sig:
            res["verdict"] = ("不可估：**基线响应本身与 0 不可区分**（b^(d)_ghost 的 CI 跨 0）⇒ "
                              "没有可供必要性检验的响应。")
        elif "necessity_ratio_d" not in res:
            res["verdict"] = f"不可估：过门槛事件仅 {len(use)}"
        else:
            ci = res["necessity_ratio_d"]["ci95"]
            res["verdict"] = ("PASS：遮住关键实体后轨迹终点退回基线 ⇒ 路径确由该实体驱动"
                              if ci[0] > 0.5 else
                              ("FAIL：遮住关键实体后轨迹终点基本不变 ⇒ 路径不是被这个实体驱动的"
                               if ci[1] < 0.5 else "不可估：必要性比的 scene 级 CI 跨 0.5"))
        # ---- 门的效度检查：b^(d)_ghost 的"显著"是不是几何混淆造成的 ----
        # 速度版的三态判定把"b_ghost 显著"当门。距离版里这道门可以被**实体自身走近**
        # 单独打开（与模型有没有反应无关）⇒ 机械套用会产出**虚假的 FAIL**。
        # 判据：纯几何项占主读数的比例；以及混淆调整后是否显著、其量级与
        # 无法控制的残留（两帧间 ego 位移）相比是否可忽略。
        bg, bgeo, badj = res["b_ghost_d"], res["b_ghost_d_geom"], res["b_ghost_d_adj"]
        egm = (res.get("ego_displacement_clean_to_ghost_m") or {}).get("median")
        geo_share = (abs(bgeo["mean"]) / abs(bg["mean"])) if (bg and bgeo and abs(bg["mean"]) > 1e-9) else None
        res["gate_diagnostics"] = {
            "geometry_share_of_b_ghost_d": geo_share,
            "b_ghost_d_adj_significant": sig(badj),
            "ego_displacement_median_m": egm,
            "adj_effect_vs_uncontrolled_residual": (abs(badj["mean"]) / egm) if (badj and egm) else None}
        res["gate_valid"] = bool(geo_share is not None and geo_share < 0.5)
        if not res["gate_valid"]:
            res["verdict_mechanical"] = res["verdict"]
            res["verdict"] = ("不可估：**主读数的门被几何混淆打开**——$b^{(d)}_{ghost}$ 的显著性中 "
                              f"{geo_share:.0%} 由实体自身走近贡献（规划不变时也会出现），"
                              "非模型响应。机械套用速度版判据会给出**虚假的 FAIL**，此处不采纳；"
                              "结论以受控的 δ_occ 为准。")
        res["delta_occ_significant"] = sig(res["delta_occ"])
        res["delta_occ_specific"] = bool(sig(res["delta_occ"]) and not sig(res["delta_ctrl"])
                                         and sig(res["abs_delta_occ_minus_ctrl"])
                                         and res["abs_delta_occ_minus_ctrl"]["mean"] > 0)
        res["delta_occ_direction"] = ("预期（擦掉危险 ⇒ 终点靠近实体）"
                                      if res["delta_occ"] and res["delta_occ"]["mean"] < 0
                                      else "反向")
        res["per_event"] = recs
        out_rows.append(res)

        print(f"\n=== {LABEL[m]}  n={len(recs)}（含 clean 臂 {len(has_clean)}）"
              f"  轨迹长度 {res['traj_len']} 点  d_ghost 均值 {res['d_ghost']['mean']:.2f} m ===")
        print(f"  b^(d)_ghost   {fmt(res['b_ghost_d']):36s} {'显著' if base_sig else '跨0'}"
              f"   （预期为正）")
        print(f"    其中纯几何项 {fmt(res['b_ghost_d_geom']):36s}"
              f"   混淆调整后 {fmt(res['b_ghost_d_adj'])}")
        eg = res.get("ego_displacement_clean_to_ghost_m"); en = res.get("entity_dlong_m")
        if eg and en:
            print(f"    残留：两帧间 ego 自身位移中位 {eg['median']:.1f} m"
                  f"（p25–p75 {eg['p25']:.1f}–{eg['p75']:.1f}）；"
                  f"实体纵距中位 {en['clean_median']:.1f} → {en['ghost_median']:.1f} m")
        print(f"  δ_occ         {fmt(res['delta_occ']):36s} "
              f"{'显著' if res['delta_occ_significant'] else '跨0'}   δ_ctrl {fmt(res['delta_ctrl'])}")
        print(f"    |δ_occ|−|δ_ctrl| {fmt(res['abs_delta_occ_minus_ctrl'])}"
              f"  ⇒ 特异={res['delta_occ_specific']}  方向={res['delta_occ_direction']}")
        if "necessity_ratio_d" in res:
            print(f"  R^(d) {fmt(res['necessity_ratio_d'])}（n={res['n_used_for_R']}）"
                  f"  R^(d)_ctrl {fmt(res['necessity_ratio_d_control'])}")
        gd = res["gate_diagnostics"]
        print(f"    门效度：几何占比 {gd['geometry_share_of_b_ghost_d']:.0%}"
              f"  调整后显著={gd['b_ghost_d_adj_significant']}"
              f"  调整后效应/未受控残留 = {gd['adj_effect_vs_uncontrolled_residual']:.3f}"
              f"  ⇒ 门有效={res['gate_valid']}")
        if not res["gate_valid"]:
            print(f"  机械判定（不采纳）：{res['verdict_mechanical'][:34]}")
        print(f"  最终判定：{res['verdict'][:80]}")

    payload = {"design": "F-3 距离读数：action = 轨迹终点到危险实体的欧氏距离",
               "n_models": len(out_rows),
               "n_baseline_significant_mechanical": sum(r["baseline_response_significant"] for r in out_rows),
               "n_gate_valid": sum(r["gate_valid"] for r in out_rows),
               "n_delta_occ_specific": sum(r["delta_occ_specific"] for r in out_rows),
               "models": out_rows}
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[DIST] {len(out_rows)} 个候选：b^(d)_ghost 机械上显著 "
          f"**{payload['n_baseline_significant_mechanical']}** 个，但门有效的只有 "
          f"**{payload['n_gate_valid']}** 个；受控读数 δ_occ 显著且特异的 "
          f"**{payload['n_delta_occ_specific']}** 个")
    print(f"[DIST] wrote {args.out}")


if __name__ == "__main__":
    main()
