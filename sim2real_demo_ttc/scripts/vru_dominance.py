"""VRU 支配性判据|要求"影响刹车的最近走廊内障碍就是这个 VRU 本身"。

工单：2026-09-03（用户看图追加）。
反例 `scene-0345_017_A`：施工工人站在**一排混凝土护栏**后面，旁边还有挖掘机。
护栏体积与行人相当（0.95 vs 0.90 m³）但**成像面积 26369 px vs 行人 7563 px（3.5 倍）**，
且道路已被封闭 —— 刹车是为了封路，遮不遮那个工人都一样。

**判据**：在自车走廊内、纵向不比 VRU 远太多的范围内，
不得存在**成像面积 >= VRU** 的其他目标（成像面积而非体积：它同时编码了
物理尺寸与距离，正是"谁在视觉上压过谁"的量，与遮挡实验的作用面直接对应）。
另单独标记"施工/封路"特征：走廊内 barrier/trafficcone 计数。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[
        "lane_path_filter.json", "lane_path_filter_Dclass.json",
        "lane_path_filter_D2b.json", "lane_path_filter_D2c.json"])
    ap.add_argument("--corridor", type=float, default=2.5,
                    help="判定支配性时的走廊半宽（略宽于 2.0，因为压过视野的东西可以稍偏）")
    ap.add_argument("--depth-slack", type=float, default=3.0,
                    help="纵向容差：只看 d <= d_vru * 1.2 + slack 的目标")
    ap.add_argument("--out", default=str(RES / "vru_dominance.json"))
    args = ap.parse_args()

    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
    sc = {s["name"]: s for s in nusc.scene}
    ev = {e["event_id"]: e for e in
          (json.loads(l) for l in open(ROOT / "variants/n1_d2/mining/events_all.jsonl"))}

    todo = []
    for f in args.inputs:
        for r in json.load(open(RES / f))["per_event"]:
            if r["status"] == "ok" and r["keep_2m"]:
                r["pool"] = f.replace("lane_path_filter", "").strip("_.json") or "A"
                todo.append(r)
    print(f"[DOM] 对 {len(todo)} 个在途事件做支配性检查")

    cache, out = {}, []
    for i, r in enumerate(todo):
        e = ev[r["eid"]]; sn = e["scene_name"]
        if sn not in cache:
            if len(cache) > 3:
                cache.pop(next(iter(cache)))
            cache[sn] = G1.compute_scene_geometry(nusc, sc[sn], cfg)
        geo = cache[sn]; gt = geo["grid_t"]
        j = int(np.argmin(np.abs(gt - e["x_ghost_frames"][0]["t"])))
        tgt = geo["per_obj"].get(e["object_token"])
        if tgt is None or not bool(tgt["valid"][j]):
            continue
        d_v = float(tgt["d_long"][j])
        a_v = float(tgt["area_px"][j]) if np.isfinite(tgt["area_px"][j]) else 0.0
        lim = d_v * 1.2 + args.depth_slack
        bigger, n_barrier, n_closer = [], 0, 0
        for tok, o in geo["per_obj"].items():
            if tok == e["object_token"] or not bool(o["valid"][j]):
                continue
            x = float(o["d_long"][j]); y = float(o["lat"][j])
            if not (0 < x < lim and abs(y) < args.corridor):
                continue
            a = float(o["area_px"][j]) if np.isfinite(o["area_px"][j]) else 0.0
            if o["cat"].startswith(("movable_object.barrier", "movable_object.trafficcone")):
                n_barrier += 1
            if x < d_v:
                n_closer += 1
            if a >= a_v:
                bigger.append((round(x, 1), o["cat"], round(a)))
        bigger.sort()
        out.append({**{k: r[k] for k in ("eid", "scene", "pool", "d_long_ghost_m",
                                         "ego_delta_v_mps", "ego_speed_ghost_mps",
                                         "ego_speed_min_future_mps", "lat_to_real_path_m")},
                    "vru_area_px": a_v, "n_bigger_in_corridor": len(bigger),
                    "n_barrier_cone": n_barrier, "n_closer_in_corridor": n_closer,
                    "vru_is_dominant": len(bigger) == 0,
                    "bigger_examples": bigger[:4],
                    "object_class": e.get("object_class")})
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(todo)}")

    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    dom = [o for o in out if o["vru_is_dominant"]]
    print(f"\n[DOM] VRU 是走廊内成像最大目标: {len(dom)} / {len(out)}")
    print(f"      走廊内有 barrier/cone 的: {sum(1 for o in out if o['n_barrier_cone']>0)}")
    print(f"[DOM] wrote {args.out}")


if __name__ == "__main__":
    main()
