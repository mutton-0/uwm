"""F-3 候选语料池|按刹车归因 + 支配性 + 群体遮挡组分档落盘。"""
import json
from pathlib import Path
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
d = json.load(open(RES / "brake_attribution.json"))

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
EV = {}
for _l in open(ROOT / "variants/n1_d2/mining/events_all.jsonl"):
    _e = json.loads(_l); EV[_e["event_id"]] = _e
FRAME = 1600 * 900


def bbox_px(o):
    bb = EV[o["eid"]]["x_ghost_frames"][0]["bbox_xyxy"]
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


# 支配性**留余量**：严格不等式不够 —— scene-0717 的 VRU/锥桶比值只有 1.11，
# 目视即可看出是路侧施工场景。要求 VRU 的刹车需求至少是最大非 VRU 的 1.5 倍。
margin = lambda o: o["a_vru_max"] >= 1.5 * max(o["a_non_vru_max"], 1e-9)
base = lambda o: (o["vru_class_share"] > 0.6 and o["explained_ratio"] > 0.3 and margin(o))
T3 = [o for o in d if base(o)]
# 保留区间 = (TTC <= 5s) ∪ (d <= 40m)，取**并集**（用户 2026-09-03）。
# 纯面积门槛或纯距离上限都会砍错：
#   scene-0666_001_A 46.6m 但 TTC 4.5s、a_req 1.01（全集最紧迫）⇒ 靠 TTC 留下；
#   scene-0710_000_D2c 21.3m 但 TTC 11.1s（自车仅 1.9m/s）⇒ 靠 40m 硬距离留下。
# 该规则之后剩余事件的最小成像面积已是 0.114%，故**不再另设面积门槛**（会是空转）。
def ttc_s(o):
    return o["d_long_ghost_m"] / max(o["ego_v0_mps"], 0.01)


T2 = [o for o in T3 if ttc_s(o) <= 5.0 or o["d_long_ghost_m"] <= 40.0]
T1 = [o for o in T2 if o["a_req_vru"] >= 0.4]

man = {
 "note": "F-3 候选语料池。遮挡臂必须遮 f3_mask_group 里的**全部** VRU 实例。",
 "attribution": "a_i = max(v_close,0)^2 / (2*max(s-2.0, 0.5))，s 为沿自车**真实未来路径**的弧长位置",
 "criteria": {
   "vru_class_share>0.6": "走廊内全部 VRU 的 a 之和占总和 >60%",
   "dominant_margin_1.5x": "走廊内单个刹车需求最大的必须是 VRU，且**至少是最大非 VRU 的 1.5 倍** —— "
                           "严格不等式不够：scene-0717 的比值仅 1.11（锥桶），目视即是施工路段",
   "T2 保留区间": "(TTC<=5s) ∪ (d<=40m) 取并集 —— 单用距离会砍掉 scene-0666_001_A"
                  "（46.6m 但 TTC 4.5s、a_req 1.01 全集最紧迫），"
                  "单用 TTC 会砍掉 scene-0710_000_D2c（TTC 11.1s 但仅 21.3m、自车 1.9m/s）",
   "不设面积门槛的理由": "T2 之后最小成像面积已是 0.114%，再设门槛是空转；"
                        "距离那件事保留区间已经做完了",
   "中心度不进筛选": "centrality = max(0, 1-|lat|/2) 仅作**诊断量**报出，不乘进面积门槛："
                     "(a) 横向位置已是 2m 走廊硬门且已进 a_req，再乘一次是重复计分；"
                     "(b) 面积是仪器侧量、横向是标本侧量，相乘后谁也解释不了；"
                     "(c) 会把近处大目标压到远处小目标之下 —— scene-0017_014_A "
                     "(19m, 0.474%) 加权后仅 0.026%，低于 69m 外的 0.022% 只一点点",
   "T1 a_req>=0.4": "紧迫度门槛（a ∝ v²/d，已隐含自车速度要求）",
   "_dominant_is_vru_note": "走廊内**单个**刹车需求最大的目标必须是 VRU —— "
                      "群体合计有漏洞：分散的几个行人加起来能盖过一辆更近的大车，"
                      "但司机是为最紧迫的那一个刹车的，不是为算术和",
   "explained_ratio>0.3": "VRU 的刹车需求要能解释观测到的减速幅度",
   "stopped_long_s": "红灯/排队代理（nuScenes 不标注信号灯），逐事件报出但**不并入**筛选",
 },
 "known_gaps": [
   "nuScenes 不标注交通信号灯 ⇒ 红灯刹车无法几何归因，仅有 stopped_long_s 代理",
   "全集观测减速度 a_obs 从未达到 1.0 m/s² —— 这批没有一次是急刹",
   "参考路径用**已实现**的未来轨迹 ⇒ 以司机行为为条件，成功避让的真危险会被低估",
 ],
 "tiers": {},
}
for k, g in (("T1_strict", T1), ("T2_main", T2), ("T3_wide", T3)):
    man["tiers"][k] = {
      "n_events": len(g), "n_scenes": len({o["scene"] for o in g}),
      "events": [{"eid": o["eid"], "scene": o["scene"], "pool": o["pool"],
                  "object_class": o["object_class"],
                  "vru_share": round(o["vru_share"], 3),
                  "vru_class_share": round(o["vru_class_share"], 3),
                  "a_req_vru": round(o["a_req_vru"], 3),
                  "a_non_vru_max": round(o["a_non_vru_max"], 3),
                  "a_obs": round(o["a_obs_mps2"], 3),
                  "explained_ratio": round(o["explained_ratio"], 3),
                  "ego_v0": round(o["ego_v0_mps"], 2),
                  "d_long": round(o["d_long_ghost_m"], 1),
                  "lat_to_real_path": round(o["lat_to_real_path_m"], 2),
                  "stopped_long_s": round(o["stopped_long_s"], 1),
                  "bbox_px": bbox_px(o),
                  "bbox_frac_pct": round(bbox_px(o) / FRAME * 100, 3),
                  "ttc_s": round(ttc_s(o), 1),
                  "centrality_DIAGNOSTIC_ONLY": round(
                      max(0.0, 1.0 - abs(o["lat_to_real_path_m"]) / 2.0), 2),
                  "n_mask_group": o["n_mask_group"],
                  "f3_mask_group": o["f3_mask_group"]}
                 for o in sorted(g, key=lambda z: z["d_long_ghost_m"])]}
    sz = sorted(o["n_mask_group"] for o in g)
    print(f"{k:10s} {len(g):3d} 事件 / {len({o['scene'] for o in g}):2d} scene   "
          f"遮挡组大小 中位 {sz[len(sz)//2] if sz else 0}  最大 {max(sz) if sz else 0}")
rej = [o for o in d if o["vru_class_share"] > 0.6 and o["explained_ratio"] > 0.3
       and not margin(o)]
man["rejected_by_dominance"] = [{"eid": o["eid"], "a_vru_max": round(o["a_vru_max"], 3),
                                 "top_non_vru": o["top_non_vru"]} for o in rej]
(RES / "f3_candidate_pool.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
print(f"支配性剔除 {len(rej)}（全部为 scene-0477）")
print(f"wrote {RES/'f3_candidate_pool.json'}")
