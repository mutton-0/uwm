"""按城市统计 ghost / lead 两场景的事件数与 scene ID 清单（两语料合并列出）。

用途：判断能否把问题重述为**右舵↔左舵**的迁移。
  右舵（车辆右置、靠左行驶）：新加坡 —— 只在 nuScenes 里
  左舵（车辆左置、靠右行驶）：波士顿 / 拉斯维加斯 / 匹兹堡
"""
from __future__ import annotations
import json, pickle, glob
from collections import defaultdict
from pathlib import Path

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
DRIVE = {"singapore-onenorth": "右舵", "singapore-queenstown": "右舵",
         "singapore-hollandvillage": "右舵", "boston-seaport": "左舵",
         "us-nv-las-vegas-strip": "左舵", "us-ma-boston": "左舵",
         "us-pa-pittsburgh-hazelwood": "左舵", "sg-one-north": "右舵"}


def nusc_city():
    from nuscenes.nuscenes import NuScenes
    n = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval",
                 verbose=False)
    return {s["name"]: n.get("log", s["log_token"])["location"] for s in n.scene}


def navsim_city():
    out = {}
    for sp in ("test", "trainval"):
        for lf in sorted(glob.glob(
                f"/data/dataset/navsim/dataset/navsim_logs/{sp}/*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                out.setdefault((sp, f["scene_name"]), f["map_location"])
    return out


def main():
    NC, VC = nusc_city(), navsim_city()
    rows = []
    for scen, corp, pf in [("ghost", "nuScenes", "brake_first_pool_final.json"),
                           ("lead", "nuScenes", "brake_first_pool_lead_final.json"),
                           ("ghost", "NAVSIM", "deploy_pool_ghost_vp.json"),
                           ("lead", "NAVSIM", "deploy_pool_lead_vp.json")]:
        p = RES / pf
        if not p.exists():
            continue
        for c in json.load(open(p))["candidates"]:
            if c.get("a_vru_max", 1) < 0.4:
                continue
            if corp == "nuScenes":
                city = NC.get(c["scene"], "?")
                sid = c["scene"]
            else:
                sp = c.get("split", "test")
                city = VC.get((sp, c["scene"]), c.get("city", "?"))
                sid = f"{sp}:{c['scene']}"
            rows.append({"scenario": scen, "corpus": corp, "city": city,
                         "drive": DRIVE.get(city, "?"), "scene_id": sid,
                         "frame_idx": c["frame_idx"], "a_req": c["a_vru_max"],
                         "ego_v0": c["ego_v0"]})
    (RES / "city_breakdown.json").write_text(
        json.dumps({"note": "右舵=新加坡；左舵=美国三城", "events": rows},
                   indent=2, ensure_ascii=False))

    # ---- 汇总 ----
    agg = defaultdict(int)
    for r in rows:
        agg[(r["scenario"], r["drive"], r["city"], r["corpus"])] += 1
    print(f"{'场景':<7}{'驾驶侧':<7}{'城市':<28}{'语料':<10}{'事件数':>7}")
    print("-" * 62)
    for k in sorted(agg, key=lambda z: (z[0], z[1], -agg[z])):
        print(f"{k[0]:<7}{k[1]:<7}{k[2]:<28}{k[3]:<10}{agg[k]:>7}")
    print()
    for scen in ("ghost", "lead"):
        d = defaultdict(int)
        for r in rows:
            if r["scenario"] == scen:
                d[r["drive"]] += 1
        tot = sum(d.values())
        print(f"{scen}: 合计 {tot}   " +
              "  ".join(f"{k} {v} ({v/tot:.0%})" for k, v in sorted(d.items())))
    print(f"\n[CITY] wrote {RES/'city_breakdown.json'}")


if __name__ == "__main__":
    main()
