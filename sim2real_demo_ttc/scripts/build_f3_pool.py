"""F-3 候选语料池|按刹车归因 + 支配性 + 群体遮挡组分档落盘。"""
import json
from pathlib import Path
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
d = json.load(open(RES / "brake_attribution.json"))

base = lambda o: (o["vru_class_share"] > 0.6 and o["explained_ratio"] > 0.3
                  and o["dominant_is_vru"])
T3 = [o for o in d if base(o)]
T2 = [o for o in T3 if o["stopped_long_s"] < 2]
T1 = [o for o in T2 if o["a_req_vru"] >= 0.4 and o["d_long_ghost_m"] <= 40
      and o["ego_v0_mps"] >= 3]

man = {
 "note": "F-3 候选语料池。遮挡臂必须遮 f3_mask_group 里的**全部** VRU 实例。",
 "attribution": "a_i = max(v_close,0)^2 / (2*max(s-2.0, 0.5))，s 为沿自车**真实未来路径**的弧长位置",
 "criteria": {
   "vru_class_share>0.6": "走廊内全部 VRU 的 a 之和占总和 >60%",
   "dominant_is_vru": "走廊内**单个**刹车需求最大的目标必须是 VRU —— "
                      "群体合计有漏洞：分散的几个行人加起来能盖过一辆更近的大车，"
                      "但司机是为最紧迫的那一个刹车的，不是为算术和",
   "explained_ratio>0.3": "VRU 的刹车需求要能解释观测到的减速幅度",
   "T2 stopped_long_s<2": "红灯/排队代理（nuScenes 不标注信号灯，只能用停车时长近似）",
   "T1 a_req>=0.4 & d<=40m & v0>=3": "紧迫度、距离、自车速度余量三重收紧",
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
                  "n_mask_group": o["n_mask_group"],
                  "f3_mask_group": o["f3_mask_group"]}
                 for o in sorted(g, key=lambda z: z["d_long_ghost_m"])]}
    sz = sorted(o["n_mask_group"] for o in g)
    print(f"{k:10s} {len(g):3d} 事件 / {len({o['scene'] for o in g}):2d} scene   "
          f"遮挡组大小 中位 {sz[len(sz)//2] if sz else 0}  最大 {max(sz) if sz else 0}")
rej = [o for o in d if o["vru_class_share"] > 0.6 and o["explained_ratio"] > 0.3
       and not o["dominant_is_vru"]]
man["rejected_by_dominance"] = [{"eid": o["eid"], "a_vru_max": round(o["a_vru_max"], 3),
                                 "top_non_vru": o["top_non_vru"]} for o in rej]
(RES / "f3_candidate_pool.json").write_text(json.dumps(man, indent=2, ensure_ascii=False))
print(f"支配性剔除 {len(rej)}（全部为 scene-0477）")
print(f"wrote {RES/'f3_candidate_pool.json'}")
