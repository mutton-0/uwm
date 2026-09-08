"""刹车优先挖矿|绕过原事件挖矿，直接在 (刹车片段 × 走廊内 VRU) 上建候选池。

工单：2026-09-03（用户指出"能否不在 1805 而在更大的数据集上筛"）。

## 为什么必须绕过原挖矿

`events_all.jsonl` 的 1805 个行人事件是**已经被筛过一轮**的：原挖矿用了
`in_corridor = |y| < 2.0`（**瞬时朝向**走廊）+ TTC 门槛 + 每 scene 上限。
其中走廊判据正是本轮已证伪的那个（弯道下失真，见 lane_path_filter_report_zh.md）。
在它的输出上再筛，等于**继承了它的假阴性**——被它错杀的真危险永远回不来。

全空间对比：nuScenes 有 246640 个 VRU 标注、12995 个不同的 VRU 实例，
而原挖矿只产出 1805 个行人事件。

## 本模块的顺序（与原挖矿相反）

原挖矿：先按几何找"危险" → 再看人类怎么开。
本模块：**先找人类真的刹车的片段** → 再问这次刹车能不能归到某个 VRU 头上。

1. 逐 scene 扫自车速度曲线，找**减速片段**：局部极大 v0 → 后续极小 vmin，
   要求 dv = vmin - v0 < -0.5 m/s 且 |dv|/v0 > 0.3；查询帧取**减速起点**
   （此时模型还没开始刹，必须自己决定要不要刹）。
2. 在查询帧上，对**每个**走廊内 VRU 算刹车需求 a = max(v_close,0)²/(2·max(s-2,0.5))，
   走廊 = 到自车**真实未来轨迹**（已实现 + 沿末帧航向外推 20m）的横向距离 < 2m。
3. 判据与 brake_attribution 完全一致：VRU 类合计占比 > 0.6、
   支配性留 1.5x 余量、explained_ratio > 0.3、保留区间 (TTC<=5s) ∪ (d<=40m)。
4. 每个候选给出 f3_mask_group（走廊内全部 VRU）。

**去重**：同一 scene 的同一减速片段只取一个查询帧；同一 (片段, 遮挡组) 只留一条。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from lane_path_filter import point_to_polyline                         # noqa: E402

D_SAFE = 2.0
# **静态物专用的更紧走廊**（场景特定调整，理由见下）。
# 2.0 m 走廊 = 半车宽 1 m + 1 m 缓冲；那 1 m 缓冲是给**会动的**目标留的
# （行人可能走进来、车可能变道）。静态路障不会动 ⇒ 缓冲无依据。
# 实测后果：路边一整排护栏平行于行车路径、永远落在 2 m 走廊内，
# 且近距离使 a_req 爆炸（scene-0047：护栏 4.2 m / 横向 1.81 m ⇒ a_req 149.67，
# 是实测减速的 150 倍）。对静态物收紧到半车宽。
STATIC_CORRIDOR = 1.0
STATIC_PREFIX = ("movable_object.", "static_object.")
# **几何与人类行为的一致性上界**：a_req 远大于实测减速，说明几何判定与真实情况脱节
# （通常是走廊误纳了不该纳的东西）。下界 0.3 早已有，此处补上界。
# 注意这是**几何 vs 人类行为**的一致性检查，不是按模型测量结果筛样本。
MAX_EXPLAINED_RATIO = 10.0
# **危险类 = 归因口径 = 遮挡口径**（用户 2026-09-04）。
# 三者必须是同一个集合：归因认定"是这些东西触发了减速"，遮挡臂就必须遮掉这些东西，
# 否则遮完之后触发物还在，测出的必然是假 FAIL。
# `animal` 此前被 g1_mine_events.obj_class 映射为 other 并在 geo 构造时丢弃 ——
# 既不可遮、也不当竞争者（双向缺失）：狗冲上路、走廊里恰有行人时，
# 会把这次刹车错算到行人头上。现通过 G1.set_include_animal(True) 纳入。
# **场景类型 -> 危险类**。归因口径与遮挡口径必须是同一集合（见 f3_gt_axis_method_v2.md §1.1）。
SCENARIO_HAZARD = {
    # 鬼探头：VRU 突现。animal 此前被 obj_class 映射为 other 并在 geo 构造时丢弃
    # （既不可遮也不当竞争者），现经 set_include_animal(True) 纳入。
    "ghost": ("human.", "vehicle.bicycle", "vehicle.motorcycle", "animal"),
    # 前车 / 静态路障：机动车 + 可移动路障类。
    # **与本项目旧的"前车急刹(B 类)"语料无关** —— 那批用 cut-in 判据挖的，
    # 已知自车常停在路口、判据误伤严重；此处是用 brake-first 重新挖。
    "lead": ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
             "vehicle.construction", "vehicle.emergency", "movable_object."),
    # 无保护左转遇障碍：冲突对象可以是**任何**迫使减速的东西
    # （对向车、横穿 VRU、路口内的静态物），故取全集。
    "left_turn": ("human.", "vehicle.", "movable_object.", "static_object.", "animal"),
    # 十字路口遇障碍：同左转，冲突对象取全集
    "intersection": ("human.", "vehicle.", "movable_object.", "static_object.", "animal"),
}
# **场景专属的事件闸门**（在归因之前先筛事件几何）。
# 无保护左转没有信号灯/路口类型标注 ⇒ 只能用自车轨迹几何做代理判据：
#   自车在减速窗口内**向左**转过的航向角 >= 阈值。
# 角度取自**已实现**路径段的首尾切向（不含 20m 直线外推段 —— 那段按构造是直的，
# 计入会稀释转角）。左为正（ego 系 y 轴指左）。
LEFT_TURN_MIN_DEG = 25.0
LEFT_TURN_MAX_DEG = 135.0   # 上界排除掉头/环岛；左转路口按定义 <135°
LEFT_TURN_MIN_PATH_M = 5.0  # 位移闸门：自车没动就无从谈"转向"
# **十字路口的代理判据**（无路口类型/信号灯标注）：
#   查询帧 60m×40m 范围内，存在 >= N 辆**在动**(>1 m/s) 且航向与自车**近垂直**
#   (|cos| < 0.5) 的车辆 ⇒ 判为存在横向车流 ⇒ 处于路口。
# 局限：该判据识别的是"有横向车流"，不等于"标准十字路口"；
# 环岛、T 型口、大型停车场出入口都可能命中。已在报告中写明。
ISEC_MIN_CROSSFLOW = 2
ISEC_COS_MAX = 0.5
ISEC_RANGE_XY = (60.0, 40.0)
HAZARD_PREFIX = SCENARIO_HAZARD["ghost"]     # 由 --scenario 覆盖
VRU_PREFIX = HAZARD_PREFIX          # 向后兼容旧名


def brake_episodes(v, t, min_dv=0.5, min_rel=0.30, max_win_s=10.0):
    """在速度曲线上找减速片段，返回 [(i_start, i_min, v0, vmin, dv, rel, a_obs)]。"""
    out, n = [], len(v)
    i = 0
    while i < n - 1:
        if v[i] < 0.5:                       # 已经停住，不算刹车起点
            i += 1; continue
        # 局部极大：v[i] 不小于前一帧
        if i > 0 and v[i] < v[i - 1]:
            i += 1; continue
        j = i + 1; jmin = i
        while j < n and (t[j] - t[i]) <= max_win_s:
            if v[j] < v[jmin]:
                jmin = j
            if v[j] > v[i]:                  # 又加速回去了，片段结束
                break
            j += 1
        if jmin > i:
            v0, vmin = float(v[i]), float(v[jmin])
            dv = vmin - v0; rel = abs(dv) / max(v0, 0.1)
            dt = float(t[jmin] - t[i])
            if dv < -min_dv and rel > min_rel and dt > 0.2:
                out.append((i, jmin, v0, vmin, dv, rel, abs(dv) / dt))
                i = jmin                     # 跳到谷底，避免同一片段重复计
                continue
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="ghost", choices=sorted(SCENARIO_HAZARD),
                    help="决定危险类 H（=归因口径=遮挡口径）")
    ap.add_argument("--corpus", default="nuscenes", choices=["nuscenes", "navsim"],
                    help="navsim 走 ns1_navsim_geometry.build_geo —— 其 geo 与 "
                         "compute_scene_geometry 逐字段对齐，故挖矿逻辑一行未改")
    ap.add_argument("--split", default="test", help="仅 navsim")
    ap.add_argument("--min-frames", type=int, default=20, help="仅 navsim：scene 最少帧数")
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--corridor", type=float, default=2.0)
    ap.add_argument("--dump-rejects", default="",
                    help="把**过了场景闸门但被归因否掉**的事件连同拒绝原因写到该文件，"
                         "用于回答『这套原则到底能不能选出有效事件』")
    ap.add_argument("--signal", default="any", choices=["any", "signalized", "unsignalized"],
                    help="自车实际驶入的路口是否灯控（仅 navsim；靠 nuplan map + 每帧 traffic_lights）")
    ap.add_argument("--exclude-red", action="store_true",
                    help="排掉查询帧自车进路为红灯的事件：红灯下减速的原因是灯不是障碍物，"
                         "F-3 的前提（遮掉原因看响应变化）不成立")
    ap.add_argument("--require-oncoming", action="store_true",
                    help="要求存在对向来车（无保护左转的行为学代理：有专用左转绿箭头时"
                         "对向直行被红灯拦住，不会有对向车流需要让行）")
    ap.add_argument("--cities", default="",
                    help="逗号分隔的 map_location 白名单（仅 navsim）。"
                         "设计用途：nuScenes 只采了波士顿+新加坡，把 NAVSIM 限定成 "
                         "us-nv-las-vegas-strip,us-pa-pittsburgh-hazelwood 后，"
                         "benchmark 与 deployment 的城市集合**完全不相交**，"
                         "跨语料差异才是真域偏移，而不是同城不同语料的混合物。")
    ap.add_argument("--min-vmin", type=float, default=0.0,
                    help="减速谷底速度下限 m/s。路口场景用它排掉红灯/人群停车："
                         "自车停死的减速由信号灯或人群支配，归因公式看不见信号灯。"
                         "这是对**人类行为**的前置判据，与模型响应无关，不构成按结果筛样本。")
    ap.add_argument("--extend-m", type=float, default=20.0)
    ap.add_argument("--out", default=str(RES / "brake_first_pool.json"))
    args = ap.parse_args()

    from omegaconf import OmegaConf

    global HAZARD_PREFIX
    HAZARD_PREFIX = SCENARIO_HAZARD[args.scenario]
    if args.scenario == "ghost":
        G1.set_include_animal(True)     # 危险类含 animal —— 归因与遮挡口径必须一致
    print(f"[BFM] scenario={args.scenario}  危险类 H = {HAZARD_PREFIX}")

    # ---- 两个语料的 scene 迭代器；都产出 (scene 显示名, geo) ----
    if args.corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval",
                        verbose=False)
        scenes = nusc.scene[: args.limit_scenes] if args.limit_scenes else nusc.scene
        n_units = len(scenes)

        def iter_scenes():
            for sc in scenes:
                try:
                    yield sc["name"], G1.compute_scene_geometry(nusc, sc, cfg), None
                except Exception:                                      # noqa: BLE001
                    continue
    else:
        import pickle
        from collections import defaultdict as _dd
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"), resolve=True)
        logs = sorted((NS.NS_ROOT / "navsim_logs" / args.split).glob("*.pkl"))
        n_units = len(logs)

        def iter_scenes():
            n = 0
            for lf in logs:
                by = _dd(list)
                for f in pickle.load(open(lf, "rb")):
                    by[f["scene_token"]].append(f)
                for stok, fl in by.items():
                    fl = sorted(fl, key=lambda z: z["timestamp"])
                    if len(fl) < args.min_frames:
                        continue
                    if args.limit_scenes and n >= args.limit_scenes:
                        return
                    n += 1
                    try:
                        yield fl[0]["scene_name"], NS.build_geo(fl, cfg, args.split), fl
                    except Exception:                                  # noqa: BLE001
                        continue

    print(f"[BFM] corpus={args.corpus}  待扫 {n_units} 个"
          f"{'scene' if args.corpus == 'nuscenes' else ' log'}")

    cands, n_ep, n_sc = [], 0, 0
    rejects = []
    CITIES = {x.strip() for x in args.cities.split(",") if x.strip()}
    n_city_skip = 0
    for si, (scene_name, geo, raw) in enumerate(iter_scenes()):
        if CITIES and raw is not None and raw[0].get("map_location") not in CITIES:
            n_city_skip += 1
            continue
        n_sc += 1
        gt = geo["grid_t"]; es = geo["ego_speed"]; exyz = geo["ego_xyz"]
        eps = brake_episodes(es, gt)
        n_ep += len(eps)
        for (j, jmin, v0, vmin, dv, rel, a_obs) in eps:
            if vmin < args.min_vmin:        # 见 --min-vmin：排掉停死的（红灯/人群）
                continue
            R = geo["R_we"][j]
            # 未来路径：走到本片段谷底，再沿末帧航向外推
            k = min(jmin, len(gt) - 1)
            if k <= j:
                continue
            poly_real = (exyz[j:k + 1, :2] - exyz[j, :2]) @ R[:2, :2]
            # 有符号航向变化（左正右负），只用**已实现**段 —— 20m 外推段按构造是直的。
            # **必须用真实自车位姿偏航角**，不能用路径切线：自车停住时末段位置重合，
            # 切线方向是数值噪声，会伪造出 ~180° 的"转向"（2026-09-04 QA 查出 31/38 伪例）。
            yaw_deg = 0.0
            path_len = float(np.linalg.norm(np.diff(poly_real, axis=0), axis=1).sum())
            if path_len >= LEFT_TURN_MIN_PATH_M:
                yw = np.unwrap([Quaternion(matrix=geo["R_we"][i]).yaw_pitch_roll[0]
                                for i in range(j, k + 1)])
                yaw_deg = float(np.degrees(yw[-1] - yw[0]))
            if args.scenario == "left_turn" and not (
                    LEFT_TURN_MIN_DEG <= yaw_deg <= LEFT_TURN_MAX_DEG):
                continue
            # ---- 信号灯闸门（仅 navsim：nuplan map 解析自车实际驶入的路口）----
            sig_on, sig_col = None, None
            if raw is not None and (args.signal != "any" or args.exclude_red):
                import ns_traffic_light as TLQ
                ml = raw[j]["map_location"]
                # 搜索窗口取 j→scene 末尾：刹车往往发生在**进入路口之前**，
                # 只搜刹车窗口 j..k 会系统性漏检（实测 68 例里漏 4 例、灯控检出 20.6%→25.0%）。
                pw = [raw[i]["ego2global_translation"][:2] for i in range(j, len(raw))]
                sig_on, sig_col, _nlc = TLQ.ego_signal(ml, pw, raw[j]["traffic_lights"])
                if args.signal == "signalized" and not sig_on:
                    continue
                if args.signal == "unsignalized" and sig_on:
                    continue
                if args.exclude_red and sig_col == "RED":
                    continue
            # ---- 对向来车（无保护左转的行为学代理）----
            n_oncoming = 0
            if args.require_oncoming or args.scenario == "left_turn":
                for _t3, _o in geo["per_obj"].items():
                    if not bool(_o["valid"][j]) or not _o["cat"].startswith("vehicle."):
                        continue
                    if float(_o["d_long"][j]) <= 0 or abs(float(_o["lat"][j])) > 40.0:
                        continue
                    _v = np.asarray(_o["v_obj_ego"][j], float)[:2]
                    _sp = float(np.linalg.norm(_v))
                    if _sp >= 1.0 and _v[0] / _sp < -0.5:      # 朝我开来
                        n_oncoming += 1
                if args.require_oncoming and n_oncoming < 1:
                    continue
            if args.scenario == "intersection":
                ncross = 0
                for _t2, _o in geo["per_obj"].items():
                    if not bool(_o["valid"][j]) or not _o["cat"].startswith("vehicle."):
                        continue
                    _x = float(_o["d_long"][j]); _y = float(_o["lat"][j])
                    if not (abs(_x) < ISEC_RANGE_XY[0] and abs(_y) < ISEC_RANGE_XY[1]):
                        continue
                    _v = np.asarray(_o["v_obj_ego"][j], float)[:2]
                    _sp = float(np.linalg.norm(_v))
                    if _sp < 1.0:
                        continue
                    if abs(_v[0] / _sp) < ISEC_COS_MAX:      # ego 系 x=前向
                        ncross += 1
                if ncross < ISEC_MIN_CROSSFLOW:
                    continue
            poly = poly_real
            fwd = geo["R_we"][k][:2, 0] @ R[:2, :2]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            poly = np.vstack([poly, poly[-1] + np.outer(
                np.linspace(0, args.extend_m, max(2, int(args.extend_m)))[1:], fwd)])
            if len(poly) < 2:
                continue

            def demand(o):
                p = np.asarray(o["p_ego"][j], float)[:2]
                dmin, seg, tang = point_to_polyline(p, poly)
                lim = (STATIC_CORRIDOR if o["cat"].startswith(STATIC_PREFIX)
                       else args.corridor)
                if dmin >= lim:
                    return 0.0, dmin, 0.0
                s = (float(np.linalg.norm(np.diff(poly[:seg + 1], axis=0), axis=1).sum())
                     if seg > 0 else 0.0)
                v_along = float(np.asarray(o["v_obj_ego"][j], float)[:2] @ tang)
                return (max(v0 - v_along, 0.0) ** 2 / (2.0 * max(s - D_SAFE, 0.5)), dmin, s)

            vrus, others = [], []
            for tok, o in geo["per_obj"].items():
                if not bool(o["valid"][j]):
                    continue
                a, dmin, s = demand(o)
                if a <= 0:
                    continue
                rec = (a, tok, o["cat"], round(s, 1), round(dmin, 2),
                       bool(o["visible"][j]))
                (vrus if o["cat"].startswith(HAZARD_PREFIX) else others).append(rec)
            if not vrus:
                if args.dump_rejects:
                    rejects.append({"scene": scene_name, "frame_idx": j,
                        "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2),
                        "a_obs": round(a_obs, 3), "reason": "走廊内没有任何危险类目标(a>0)", "n_other_in_corridor": len(others)})
                continue
            vrus.sort(reverse=True); others.sort(reverse=True)
            a_vru_sum = sum(x[0] for x in vrus); a_oth_sum = sum(x[0] for x in others)
            tot = a_vru_sum + a_oth_sum
            if tot < 1e-9:
                continue
            share = a_vru_sum / tot
            a_vru_max = vrus[0][0]; a_oth_max = others[0][0] if others else 0.0
            n_animal = sum(1 for x in vrus if x[2].startswith("animal"))
            if share <= 0.6 or a_vru_max < 1.5 * max(a_oth_max, 1e-9):
                if args.dump_rejects:
                    rejects.append({"scene": scene_name, "frame_idx": j,
                        "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2),
                        "a_obs": round(a_obs, 3), "reason": "危险类不占主导(share<=0.6 或 无1.5倍优势)", "share": round(share, 3), "a_vru_max": round(a_vru_max, 3), "a_other_max": round(a_oth_max, 3), "lead_cat": (vrus[0][2] if vrus else None)})
                continue
            er = (a_vru_max / a_obs) if a_obs > 1e-6 else 0.0
            if er <= 0.3 or er > MAX_EXPLAINED_RATIO:
                if args.dump_rejects:
                    rejects.append({"scene": scene_name, "frame_idx": j,
                        "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2),
                        "a_obs": round(a_obs, 3), "reason": "解释度越界(explained_ratio 不在 (0.3,10])", "explained_ratio": round(er, 3), "a_vru_max": round(a_vru_max, 3), "lead_cat": vrus[0][2]})
                continue
            vis = [x for x in vrus if x[5]]                 # 必须能投影出来才谈遮挡
            if not vis:
                if args.dump_rejects:
                    rejects.append({"scene": scene_name, "frame_idx": j,
                        "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2),
                        "a_obs": round(a_obs, 3), "reason": "归因目标投影不到图像上(无法遮挡)", "a_vru_max": round(a_vru_max, 3), "lead_cat": vrus[0][2]})
                continue
            lead = vis[0]
            # 归因目标的运动方向 vs 自车前向：+1 同向(前车)，-1 对向，~0 横穿
            _lo = geo["per_obj"][lead[1]]
            _lv = np.asarray(_lo["v_obj_ego"][j], float)[:2]
            _lsp = float(np.linalg.norm(_lv))
            lead_cos = float(_lv[0] / _lsp) if _lsp >= 0.5 else float("nan")
            ttc = lead[3] / max(v0, 0.01)
            if not (ttc <= 5.0 or lead[3] <= 40.0):
                if args.dump_rejects:
                    rejects.append({"scene": scene_name, "frame_idx": j,
                        "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2),
                        "a_obs": round(a_obs, 3), "reason": "太远且不紧迫(TTC>5s 且 距离>40m)", "ttc_s": round(ttc, 1), "s_m": lead[3], "a_vru_max": round(a_vru_max, 3), "lead_cat": lead[2]})
                continue
            cands.append({
              "scene": scene_name, "frame_idx": j, "t": float(gt[j]),
              "ego_v0": round(v0, 2), "ego_vmin": round(vmin, 2), "dv": round(dv, 2),
              "rel_decel": round(rel, 3), "a_obs": round(a_obs, 3),
              "brake_dur_s": round(float(gt[jmin] - gt[j]), 2),
              "a_vru_sum": round(a_vru_sum, 3), "a_vru_max": round(a_vru_max, 3),
              "a_non_vru_max": round(a_oth_max, 3), "vru_class_share": round(share, 3),
              "explained_ratio": round(a_vru_max / a_obs, 3),
              "lead_vru": {"token": lead[1], "cat": lead[2], "s_m": lead[3],
                           "lat_m": lead[4]},
              "ttc_s": round(ttc, 1),
              "ego_yaw_change_deg": round(yaw_deg, 1),
              "signalized": sig_on, "signal_color": sig_col,
              "n_oncoming": n_oncoming,
              "lead_cos": round(lead_cos, 3),
              "n_mask_group": len(vrus), "n_animal_in_mask": n_animal,
              "hazard_classes": list(HAZARD_PREFIX),
              "f3_mask_group": [{"token": x[1], "cat": x[2], "s_m": x[3], "lat_m": x[4],
                                 "a_req": round(x[0], 3), "visible": x[5]} for x in vrus],
            })
        if (si + 1) % 100 == 0:
            print(f"  {si+1} scene 已扫  减速片段 {n_ep}  候选 {len(cands)}")

    # 去重：同一 scene 同一遮挡组只留 a_vru_max 最大的一条
    best = {}
    for c in cands:
        key = (c["scene"], frozenset(g["token"] for g in c["f3_mask_group"]))
        if key not in best or c["a_vru_max"] > best[key]["a_vru_max"]:
            best[key] = c
    ded = sorted(best.values(), key=lambda z: (z["scene"], z["frame_idx"]))
    T1 = [c for c in ded if c["a_vru_max"] >= 0.4]
    if args.dump_rejects:
        for c in ded:
            if c["a_vru_max"] < 0.4:
                rejects.append({"scene": c["scene"], "frame_idx": c["frame_idx"],
                                "ego_v0": c["ego_v0"], "ego_vmin": c["ego_vmin"],
                                "a_obs": c["a_obs"], "reason": "归因需求太弱(a_req<0.4)",
                                "a_vru_max": c["a_vru_max"], "lead_cat": c["lead_vru"]["cat"],
                                "s_m": c["lead_vru"]["s_m"], "f3_mask_group": c["f3_mask_group"]})
        from collections import Counter as _C
        json.dump({"scenario": args.scenario, "corpus": args.corpus,
                   "n_rejects": len(rejects),
                   "by_reason": dict(_C(r["reason"] for r in rejects).most_common()),
                   "rejects": rejects}, open(args.dump_rejects, "w"),
                  indent=2, ensure_ascii=False)
        print(f"[BFM] 拒绝原因分布 -> {args.dump_rejects}")
        for k_, v_ in _C(r["reason"] for r in rejects).most_common():
            print(f"        {v_:5d}  {k_}")
    out = {"design": "刹车优先挖矿：先找人类减速片段，再归因到走廊内 VRU",
           "corpus": args.corpus, "split": (args.split if args.corpus == "navsim" else None),
           "scenario": args.scenario, "hazard_classes": list(HAZARD_PREFIX),
           "static_corridor_m": STATIC_CORRIDOR,
           "cities": (sorted(CITIES) or None), "n_scenes_skipped_by_city": n_city_skip,
           "min_vmin_mps": args.min_vmin, "signal_gate": args.signal,
           "exclude_red": args.exclude_red, "require_oncoming": args.require_oncoming,
           "left_turn_gate": ({"min_deg": LEFT_TURN_MIN_DEG, "max_deg": LEFT_TURN_MAX_DEG,
                              "min_path_m": LEFT_TURN_MIN_PATH_M,
                              "yaw_source": "ego pose quaternion (unwrapped)"}
                             if args.scenario == "left_turn" else None),
           "intersection_proxy": ({"min_crossflow": ISEC_MIN_CROSSFLOW,
                                   "cos_max": ISEC_COS_MAX, "range_xy": ISEC_RANGE_XY}
                                  if args.scenario == "intersection" else None),
           "explained_ratio_bounds": [0.3, MAX_EXPLAINED_RATIO],
           "n_scenes_scanned": n_sc, "n_brake_episodes": n_ep,
           "n_candidates_raw": len(cands), "n_candidates_dedup": len(ded),
           "n_scenes_with_candidate": len({c["scene"] for c in ded}),
           "T1_a_req_ge_0.4": {"n": len(T1), "n_scenes": len({c["scene"] for c in T1})},
           "candidates": ded}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[BFM] scene {n_sc}  减速片段 {n_ep}")
    print(f"      候选 {len(cands)} -> 去重 {len(ded)} / {len({c['scene'] for c in ded})} scene")
    print(f"      其中 a_vru_max>=0.4: {len(T1)} / {len({c['scene'] for c in T1})} scene")
    print(f"[BFM] wrote {args.out}")


if __name__ == "__main__":
    main()
