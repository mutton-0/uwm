"""N2-sim | 在 CARLA 里直接构造受控 VRU 横穿场景，产出**真反事实配对**。

为什么不用录制数据（M1 的教训）：
  录制数据里场景行人只存在于一个包，正例卡在 58；几何匹配后仅 20 对、离心率 SMD 仍 0.45。
  而且录制数据**没有配对**——只能在不同场景间做统计匹配，残余混淆无法排除（q_audit §Q3 的教训）。

本脚本用模拟器直接造，一次性解决四件事：
  1. **真反事实配对**：同一条 ego 轨迹跑多遍，只改行人是否存在 => 除目标外逐像素相同；
  2. **几何由构造控死**：行人放在指定的纵向距离与横向偏移，成像大小/位置可复现；
  3. **类别替换**：行人 → 同尺寸静物（§12.4-N2 要 3DGS 做的事，这里原生免费）；
  4. **时机剂量阶梯**：横穿触发时刻 Δt 可扫（§G5.3）。

危险标签**由构造给出**，不依赖 TTC 启发式，也不依赖专家归因。

相机与 SimLingo 训练完全一致：1024×512、fov 110、挂载 x=-1.5 / z=2.0、rpy=0。
输出格式与 m1_carla_mine 一致（rgb/ + boxes/ + measurements/），下游零改动。
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import queue
import random
from pathlib import Path

import carla
import numpy as np

CAM_W, CAM_H, CAM_FOV = 1024, 512, 110
CAM_X, CAM_Z = -1.5, 2.0          # 与 team_code/config.py 的 camera_pos 一致
FPS = 20                          # 模拟器步进
SAVE_EVERY = 5                    # 与 SimLingo 数据采集一致 => 落盘 4 Hz


def find_straight_stretch(world, min_len=120.0):
    """找一段足够长的直路：返回起点 waypoint。"""
    m = world.get_map()
    best = None
    for wp in m.generate_waypoints(2.0):
        if wp.is_junction:
            continue
        cur, total = wp, 0.0
        ok = True
        while total < min_len:
            nxt = cur.next(2.0)
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            if abs(nxt[0].transform.rotation.yaw - wp.transform.rotation.yaw) > 5:
                ok = False
                break
            cur = nxt[0]; total += 2.0
        if ok and (best is None or random.random() < 0.05):
            best = wp
            if random.random() < 0.3:
                break
    return best


class Recorder:
    def __init__(self, world, vehicle, out: Path):
        self.out = out
        (out / "rgb").mkdir(parents=True, exist_ok=True)
        (out / "boxes").mkdir(exist_ok=True)
        (out / "measurements").mkdir(exist_ok=True)
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CAM_W))
        bp.set_attribute("image_size_y", str(CAM_H))
        bp.set_attribute("fov", str(CAM_FOV))
        # 渲染可复现性：关掉带时间累积的后处理。
        # 零编辑对照实测：不关时，同一条件跑两遍的像素差在行程后段达 5–9%，
        # 比行人信号（~0.3%）还大 —— 配对直接作废。
        for k, v in (("motion_blur_intensity", "0.0"), ("lens_flare_intensity", "0.0"),
                     ("bloom_intensity", "0.0"), ("exposure_mode", "manual")):
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z), carla.Rotation())
        self.cam = world.spawn_actor(bp, tf, attach_to=vehicle)
        self.q = queue.Queue()
        self.cam.listen(self.q.put)

    def grab(self, timeout=5.0):
        return self.q.get(timeout=timeout)

    def destroy(self):
        self.cam.stop(); self.cam.destroy()


def ego_relative(ego_tf, actor_tf):
    """actor 在 ego 系下的 (x 前, y 右, z 上)。"""
    d = actor_tf.location - ego_tf.location
    yaw = math.radians(ego_tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return [d.x * c + d.y * s, -d.x * s + d.y * c, d.z]


def collect(world, cfg, out: Path, with_target: str, seed: int):
    """跑一遍场景。with_target ∈ {walker, prop, none}。"""
    random.seed(seed); np.random.seed(seed)
    bl = world.get_blueprint_library()
    start_wp = cfg["_start_wp"]

    veh_bp = bl.filter(cfg["ego_bp"])[0]
    ego_tf = start_wp.transform
    ego_tf.location.z += 0.5
    ego = world.try_spawn_actor(veh_bp, ego_tf)
    assert ego is not None, "ego 生成失败"

    # 横穿几何（关键）：由**遭遇距离**反推触发时刻，让行人恰好在 ego 逼近到
    # d_encounter 米时走到车道中心。首版把行人放在右侧 6m 后往左走，
    # ego 5.6s 就开过去了而行人要 4.3s 才到车道 => 从未真正横穿到车前（占 0.14% 像素）。
    #   t_enc  = (spawn_dist - d_encounter) / v_ego     ego 到达遭遇点的时刻
    #   t_walk = |lat0| / v_walk                        行人从起始横向位置走到车道中心
    #   cross_start = t_enc - t_walk
    # 这样 d_encounter 成为设计参数，直接支持 §G5.3 的时机剂量阶梯。
    fwd = start_wp.transform.get_forward_vector()
    right = start_wp.transform.get_right_vector()
    lat0 = cfg["lateral"]                       # 起始横向偏移（负=左侧）
    t_enc = (cfg["spawn_dist"] - cfg["d_encounter"]) / cfg["ego_speed"]
    t_walk = abs(lat0) / cfg["walk_speed"]
    cfg["_cross_start"] = max(0.0, t_enc - t_walk)
    cfg["_cross_dir"] = 1.0 if lat0 < 0 else -1.0        # 朝车道中心走
    tgt_loc = start_wp.transform.location + fwd * cfg["spawn_dist"] + right * lat0
    tgt_loc.z += 1.0
    target = None
    if with_target != "none":
        bp = (list(bl.filter("walker.pedestrian.*"))[cfg["walker_idx"] % 30]
              if with_target.startswith("walker") else bl.find(cfg["prop_bp"]))
        yaw = start_wp.transform.rotation.yaw - 90
        # try_spawn_actor 失败时返回 None —— 必须重试并断言，否则"有目标"条件会静默退化成"无目标"，
        # 两个条件的图像逐像素相同，得出的任何差异都是零（首轮就踩了这个坑）。
        for dz in (1.0, 1.6, 2.2, 0.6):
            for dlat in (0.0, -0.8, 0.8, -1.6, 1.6):
                loc = carla.Location(tgt_loc); loc.z = start_wp.transform.location.z + dz
                loc += right * dlat
                target = world.try_spawn_actor(bp, carla.Transform(loc, carla.Rotation(yaw=yaw)))
                if target is not None:
                    tgt_loc = loc
                    break
            if target is not None:
                break
        assert target is not None, f"{with_target} 目标在 {cfg['scene']} 生成失败（已试 20 个位置）"

    # 类别对照的**精确几何匹配**：静物与行人同高不可得（最接近的 gnome 0.88m vs 行人 1.86m）。
    # 解法：沿相机射线按高度比缩放距离 —— 角尺寸与图像位置**同时**严格匹配。
    #   d_prop = d_walker · h_prop/h_walker      => 角高度相同
    #   p_prop = cam + k·(p_walker − cam)        => 同一条射线 => 图像位置相同
    # 代价：静物物理上更近（单帧看不出，正是我们要测的类别信息）。
    cfg["_ray_scale"] = 1.0
    if with_target == "prop":
        h_prop = target.bounding_box.extent.z
        h_walk = cfg.get("_walker_half_h", 0.93)
        cfg["_ray_scale"] = float(h_prop / h_walk)
    elif with_target == "walker":
        cfg["_walker_half_h"] = float(target.bounding_box.extent.z)

    rec = Recorder(world, ego, out)
    ego.set_simulate_physics(False)          # 运动学控制：ego 精确匀速直行，保证多次重跑逐帧对齐
    if target is not None:
        target.set_simulate_physics(False)

    frames = []
    n_steps = cfg["n_frames"] * SAVE_EVERY
    v = cfg["ego_speed"]
    for step in range(n_steps):
        t = step / FPS
        # ego：沿起点朝向匀速前进
        loc = start_wp.transform.location + fwd * (v * t)
        loc.z = ego_tf.location.z
        ego.set_transform(carla.Transform(loc, start_wp.transform.rotation))
        # 行人：到达 cross_start 时刻后以 walk_speed 横穿（沿 -right 方向）
        if target is not None:
            dt = max(0.0, t - cfg["_cross_start"])
            # 走到车道中心后继续前行，不回头
            wloc = tgt_loc + right * (cfg["_cross_dir"] * cfg["walk_speed"] * dt)
            if with_target == "walker_side":
                # R② 上下文对照：同一行人资产、同纵向距离剖面，但横向停在路侧不入车道。
                # => 类别相同、角尺寸相同（同资产同距离），唯一差异是图像横向位置。
                # nuScenes 上这一组只配上 49 例就废了（走廊外行人天然偏心）；
                # 在模拟器里可以构造性地让它同距离同尺寸。
                wloc = tgt_loc
            if with_target == "prop" and cfg["_ray_scale"] != 1.0:
                # 沿相机射线缩放，使成像位置与角尺寸都与行人一致
                cam = loc + start_wp.transform.rotation.get_forward_vector() * CAM_X
                cam.z = ego_tf.location.z + CAM_Z
                k = cfg["_ray_scale"]
                wloc = carla.Location(cam.x + k * (wloc.x - cam.x),
                                      cam.y + k * (wloc.y - cam.y),
                                      cam.z + k * (wloc.z - cam.z))
            target.set_transform(carla.Transform(wloc, carla.Rotation(
                yaw=start_wp.transform.rotation.yaw - 90)))
        if step % SAVE_EVERY:
            world.tick(); rec.grab()
            continue
        # 落盘帧：定位后多 tick 几次，让 LOD/资产流式加载/TAA 收敛到稳态，
        # 使"同一位姿 -> 同一图像"可复现（零编辑对照的前提）
        for _ in range(cfg.get("settle_ticks", 8)):
            world.tick(); img = rec.grab()
        i = step // SAVE_EVERY
        img.save_to_disk(str(out / "rgb" / f"{i:04d}.jpg"))
        etf = ego.get_transform()
        boxes = [{"class": "ego_car", "extent": [2.45, 0.92, 0.75],
                  "position": [0.0, 0.0, 0.0], "yaw": 0.0, "speed": v, "id": 0}]
        if target is not None:
            bb = target.bounding_box
            p = ego_relative(etf, target.get_transform())
            boxes.append({
                "class": "walker" if with_target.startswith("walker") else "static",
                "type_id": target.type_id,
                "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                "position": p, "yaw": -math.pi / 2,
                "distance": float(math.hypot(p[0], p[1])),
                "speed": cfg["walk_speed"] if (with_target == "walker" and t >= cfg["_cross_start"]) else 0.0,
                "id": 1})
        with gzip.open(out / "boxes" / f"{i:04d}.json.gz", "wt") as f:
            json.dump(boxes, f)
        with gzip.open(out / "measurements" / f"{i:04d}.json.gz", "wt") as f:
            json.dump({"speed": v, "target_point": [50.0, 0.0], "command": 4,
                       "pos_global": [etf.location.x, etf.location.y],
                       "ego_matrix": [[1, 0, 0, etf.location.x], [0, 1, 0, etf.location.y],
                                      [0, 0, 1, 0], [0, 0, 0, 1]],
                       "speed_reduced_by_obj_id": 1 if (with_target == "walker" and
                                                        t >= cfg["_cross_start"]) else None,
                       "n2sim": {"condition": with_target, **{k: v_ for k, v_ in cfg.items()
                                                              if not k.startswith("_")}}}, f)
        frames.append(i)

    rec.destroy()
    for a in (target, ego):
        if a is not None:
            a.destroy()

    # 可见性校验：目标必须在足够多帧里投影进画幅，否则该场景作废
    if with_target != "none":
        vis = sum(1 for i in frames if _visible(out, i))
        assert vis >= cfg.get("min_visible_frames", 8), \
            f"{cfg['scene']}/{with_target}: 目标仅在 {vis} 帧可见（<{cfg.get('min_visible_frames', 8)}）"
        print(f"      目标可见 {vis}/{len(frames)} 帧")
    return frames


def _visible(out: Path, i: int) -> bool:
    """目标中心是否投影进画幅（与 m1_carla_mine 同一投影约定）。"""
    with gzip.open(out / "boxes" / f"{i:04d}.json.gz", "rt") as f:
        boxes = json.load(f)
    for b in boxes:
        if b["class"] == "ego_car":
            continue
        x, y, z = b["position"]
        zc = x - CAM_X
        if zc <= 0.3:
            return False
        focal = CAM_W / (2 * math.tan(math.radians(CAM_FOV / 2)))
        u = focal * y / zc + CAM_W / 2
        v = focal * (CAM_Z - z) / zc + CAM_H / 2
        return 0 < u < CAM_W and 0 < v < CAM_H
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3654)
    ap.add_argument("--out", default="/data/ruolin/n2sim")
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--d-encounter-ladder", default="",
                    help="逗号分隔的遭遇距离阶梯，如 8,14,20,28（§G5.3 时机剂量阶梯）")
    ap.add_argument("--d-encounter", type=float, default=15.0,
                    help="行人走到车道中心时 ego 与其的纵向距离（时机剂量阶梯的扫描参数）")
    args = ap.parse_args()

    client = carla.Client("localhost", args.port); client.set_timeout(60.0)
    world = client.load_world(args.town)
    st = world.get_settings()
    st.synchronous_mode = True; st.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(st)

    out_root = Path(args.out)
    made = []
    rungs = [float(x) for x in args.d_encounter_ladder.split(",")] if args.d_encounter_ladder \
        else [args.d_encounter]
    for si in range(args.n_scenes):
        random.seed(1000 + si)
        wp = find_straight_stretch(world)
        if wp is None:
            continue
        cfg = {"_start_wp": wp, "ego_bp": "vehicle.lincoln.mkz*", "ego_speed": 8.0,
               "n_frames": 40, "spawn_dist": 60.0, "lateral": -4.5,
               "d_encounter": args.d_encounter, "walk_speed": 1.4, "walker_idx": si,
               "prop_bp": "static.prop.trafficcone02", "scene": f"n2sim_{si:03d}",
               "min_visible_frames": 6}
        # none2 = **零编辑对照**（手册 §G5.2 强制先行）：与 none 完全相同的条件跑第二遍，
        # 量出"同样输入两次渲染"的像素差 —— 这是本配对方案的伪影地板，
        # 所有编辑档的读数都要先扣掉它。
        # none / none2 与 d_encounter 无关（ego 轨迹不变），每场景各跑一次
        for cond in ("none", "none2"):
            o = out_root / f"{cfg['scene']}__{cond}"
            if o.exists():
                continue
            fr = collect(world, cfg, o, "none", seed=si)
            made.append((cfg["scene"], cond, len(fr)))
            print(f"  {cfg['scene']} / {cond:6s}: {len(fr)} 帧")
        for d in rungs:
            cfg["d_encounter"] = d
            for cond in ("walker", "prop", "walker_side"):
                o = out_root / f"{cfg['scene']}__d{int(d):02d}__{cond}"
                if o.exists():
                    continue
                try:
                    fr = collect(world, cfg, o, cond, seed=si)
                except AssertionError as e:
                    print(f"  跳过 {o.name}: {e}"); continue
                made.append((cfg["scene"], f"d{int(d)}_{cond}", len(fr)))
                # 退化对照：主档的 walker 再跑一遍 —— 图像逐像素相同，读数**必须**是 0.5。
                # 这比 D2cV 更硬：D2cV 依赖"模型看不见速度"这个推论，本对照不依赖任何推论。
                if cond == "walker" and abs(d - rungs[0]) < 1e-6:
                    o2 = out_root / f"{cfg['scene']}__d{int(d):02d}__walker2"
                    if not o2.exists():
                        collect(world, cfg, o2, "walker", seed=si)
                        print(f"  {cfg['scene']} / d={d:.0f}m walker2 (退化对照)")
                print(f"  {cfg['scene']} / d={d:.0f}m {cond:6s}: {len(fr)} 帧")
    print(f"\n[N2-sim] 完成 {len(made)} 次采集")


if __name__ == "__main__":
    main()
