"""探针配套核查：ghost 帧上除被遮实体外，画面里还剩多少其它 VRU / 车辆。

**为什么必须查这个**：occ 臂只抹掉挖掘阶段锁定的那一个目标实体。
若同一帧里还有别的行人，模型完全可以因为**别人**而继续输出
"Ignore instruction ... because of the pedestrian" —— 那么 ghost−occ 的零效应
就不能读作"模型没用到行人"，只能读作"没用到**这一个**行人"。
两者的结论强度差很多，必须先量出来。
"""
import json, sys
from collections import Counter
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility

nusc = NuScenes(version="v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
evs = [json.loads(l) for l in open("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2/mining/events_all.jsonl")]
ok = [e for e in evs if e["event_type"] == "A" and e["x_ghost_frames"] and e["x_ghost_frames"][0].get("bbox_xyxy")]

n_ped, n_veh, rows = [], [], []
for e in ok:
    sd = e["x_ghost_frames"][0].get("sd_token")
    if not sd:
        continue
    _, boxes, K = nusc.get_sample_data(sd, box_vis_level=BoxVisibility.ANY)
    tgt = e.get("object_token")   # 挖掘阶段锁定的目标实体 instance_token
    p = sum(1 for b in boxes if b.name.startswith("human.pedestrian")
            and nusc.get("sample_annotation", b.token)["instance_token"] != tgt)
    v = sum(1 for b in boxes if b.name.startswith("vehicle."))
    n_ped.append(p); n_veh.append(v)
    rows.append({"event_id": e["event_id"], "other_peds_in_view": p, "vehicles_in_view": v})

n_ped, n_veh = np.array(n_ped), np.array(n_veh)
out = {"n_events": len(n_ped),
       "other_pedestrians_in_view": {
           "mean": float(n_ped.mean()), "median": float(np.median(n_ped)),
           "frac_zero": float((n_ped == 0).mean()), "frac_ge1": float((n_ped >= 1).mean()),
           "quantiles_p10_p50_p90": [float(x) for x in np.percentile(n_ped, [10, 50, 90])]},
       "vehicles_in_view": {"mean": float(n_veh.mean()), "median": float(np.median(n_veh)),
                            "frac_zero": float((n_veh == 0).mean())},
       "per_event": rows}
p = "/data/ruolin/uwm/sim2real_demo_ttc/results/simlingo_instruction_probe_scene_census.json"
json.dump(out, open(p, "w"), indent=2)
print(f"事件数 {len(n_ped)}")
print(f"ghost 帧中**除目标外**仍可见的行人数：均值 {n_ped.mean():.2f}，中位 {np.median(n_ped):.0f}，"
      f"至少 1 个的比例 {(n_ped>=1).mean():.1%}，一个都没有的比例 {(n_ped==0).mean():.1%}")
print(f"可见车辆数：均值 {n_veh.mean():.2f}，中位 {np.median(n_veh):.0f}，为 0 的比例 {(n_veh==0).mean():.1%}")
print("wrote", p)
