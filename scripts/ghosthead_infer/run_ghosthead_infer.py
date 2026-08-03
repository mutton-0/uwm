#!/usr/bin/env python
"""
Ghosthead -> DiffusionDrive standalone 推理 + PDM 式 BEV 评分。

对一个 ghosthead 场景 (默认 gh_001000__brake) 的两路视频：
  transfered = renders/<scene>/frames.mp4              (sim 渲染)
  origin     = ghosthead_result/<scene>/seg1p0/<scene>_seg1p0.mp4  (世界模型生成)
在 t=1/2/3/4s 各抽 1 帧 -> 居中裁 4:1 去天空 -> 喂 diffusiondrive ckpt 推理出 8 点轨迹，
再对着 renders/<scene>/scene.json 的真实 actor/ego GT 计算 PDM 式分数并画 BEV。

不碰 navmini / 不下 dataset / backbone 走进程内 monkeypatch (pretrained=False, ckpt 提供权重)。
"""
import os, sys, json, csv, argparse, subprocess, tempfile
import numpy as np

# ---------------------------------------------------------------- backbone monkeypatch
# ckpt 内含全部 backbone 权重，agent.initialize() 会 strict 覆盖加载；
# 这里强制 timm 用 pretrained=False 建模，避免联网/下载，对推理结果零影响。
import timm
_ORIG_CREATE = timm.create_model
def _patched_create(*a, **k):
    k['pretrained'] = False
    k.pop('pretrained_cfg_overlay', None)
    return _ORIG_CREATE(*a, **k)
timm.create_model = _patched_create

import cv2
import torch
from torchvision import transforms
from shapely.geometry import Polygon, MultiPoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

# ---------------------------------------------------------------- 常量
DATA_ROOT = "/data/Zhengyang/Auto_Eval/ghosthead_v1"
DEVKIT = os.environ.get("NAVSIM_DEVKIT_ROOT", "/data/ruolin/uwm")
CKPT = "/data/ruolin/uwm/ckpt/diffusiondrive_sim_navhard.ckpt"
ANCHOR = f"{DEVKIT}/traj_final/kmeans_navsim_traj_20.npy"

FPS = 10.0
DT = 1.0 / FPS
SEC_LIST = [1, 2, 3, 4]                    # 抽帧秒
FRAME_IDX = [int(round(s * FPS)) for s in SEC_LIST]  # -> 10,20,30,39(clamp)
CROP_CENTER_ROW = 450                       # 主点行，居中裁 4:1 的竖直中心
DRIVING_COMMAND = np.array([0, 1, 0, 0], dtype=np.float32)  # 直行 one-hot(假设)

# ego footprint (navsim 默认车尺寸)
EGO_L, EGO_W = 4.6, 1.9
REAR_AXLE_TO_CENTER = 1.461                 # 位姿在后轴，box 中心前移

# 舒适边界 (navsim)
COMFORT = dict(max_lon_acc=2.40, max_lat_acc=4.89, max_yaw_rate=0.95, max_lon_jerk=4.13)
TTC_HORIZON = 0.95
PRED_DT = 0.5                               # 预测点时间间隔
N_POSES = 8


# ---------------------------------------------------------------- 抽帧
def extract_frame(video, idx, out_png):
    """用 ffmpeg 抽出第 idx 帧 (0-based) 存 png。"""
    vf = f"select='eq(n\\,{idx})'"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
           "-vf", vf, "-frames:v", "1", out_png]
    subprocess.run(cmd, check=True)
    if not os.path.exists(out_png):
        raise RuntimeError(f"ffmpeg 未产出帧 {idx} from {video}")


def crop_4to1_no_sky(img_rgb):
    """居中裁 4:1 去天空: 宽 W 取高 W/4，以 CROP_CENTER_ROW 为竖直中心。返回裁剪图(未resize)。"""
    h, w = img_rgb.shape[:2]
    target_h = int(round(w / 4.0))
    half = target_h // 2
    c = CROP_CENTER_ROW
    top = max(0, min(c - half, h - target_h))
    return img_rgb[top:top + target_h, :, :]


def build_camera_feature(crop_rgb):
    """复用 feature builder 口径: resize(2048,512)+ToTensor -> [1,3,512,2048]。"""
    resized = cv2.resize(crop_rgb, (2048, 512))
    t = transforms.ToTensor()(resized)
    return t.unsqueeze(0)


# ---------------------------------------------------------------- scene.json / GT
def load_scene(scene_dir):
    with open(os.path.join(scene_dir, "scene.json")) as f:
        return json.load(f)


def ego_mats(scene):
    """返回每帧 ego_to_world 4x4 (N,4,4)。"""
    return np.array([np.array(fr["ego_to_world"], dtype=np.float64) for fr in scene["frames"]])


def ego_status_at(mats, idx):
    """从 ego_to_world 有限差分求该帧 ego 系速度/加速度(2D)。"""
    n = len(mats)
    pos = mats[:, :3, 3]
    R = mats[idx, :3, :3]

    def vel_world(k):
        k0, k1 = max(0, k - 1), min(n - 1, k + 1)
        return (pos[k1] - pos[k0]) / ((k1 - k0) * DT)

    vw = vel_world(idx)
    aw = (vel_world(min(n - 1, idx + 1)) - vel_world(max(0, idx - 1))) / (
        (min(n - 1, idx + 1) - max(0, idx - 1)) * DT + 1e-9)
    v_ego = R.T @ vw
    a_ego = R.T @ aw
    return v_ego[:2].astype(np.float32), a_ego[:2].astype(np.float32)


def actor_footprints_in_ego(scene, mats, input_idx):
    """
    返回 dict: actor_id -> list over pred step i(0..7) 的 shapely 多边形(输入帧 ego 系)。
    超出场景末尾用常速外推。附 ghost 标记。
    """
    frames = scene["frames"]
    n = len(frames)
    R_in = mats[input_idx, :3, :3]
    t_in = mats[input_idx, :3, 3]

    def world_to_ego(pts_world):  # (M,3)->(M,2)
        return ((pts_world - t_in) @ R_in)[:, :2]  # R_in.T @ x == x @ R_in

    # 收集每个 actor 每帧的世界角点
    actor_ids = [a["id"] for a in frames[input_idx]["actors"]]
    per_actor = {}
    for aid in actor_ids:
        polys = []
        for i in range(N_POSES):
            tau = PRED_DT * (i + 1)
            abs_f = input_idx + tau * FPS
            f0 = int(np.floor(abs_f))
            if f0 <= n - 1:
                corners = _get_corners(frames[f0], aid)
            else:
                # 常速外推: frame(n-1) + 速度*(超出秒数)
                corners_last = _get_corners(frames[n - 1], aid)
                corners_prev = _get_corners(frames[n - 2], aid)
                if corners_last is None:
                    corners = None
                else:
                    vel = (corners_last.mean(0) - corners_prev.mean(0)) / DT
                    extra = (abs_f - (n - 1)) * DT
                    corners = corners_last + vel * extra
            if corners is None:
                polys.append(None)
            else:
                xy = world_to_ego(corners)
                hull = MultiPoint(xy).convex_hull
                polys.append(hull if hull.geom_type == "Polygon" else None)
        per_actor[aid] = polys
    return per_actor, actor_ids


def _get_corners(frame, aid):
    for a in frame["actors"]:
        if a["id"] == aid:
            return np.array(a["corners_world"], dtype=np.float64)
    return None


# ---------------------------------------------------------------- ego box
def ego_box_poly(x, y, heading, forward_shift=REAR_AXLE_TO_CENTER, L=EGO_L, W=EGO_W):
    cx = x + np.cos(heading) * forward_shift
    cy = y + np.sin(heading) * forward_shift
    dx, dy = L / 2.0, W / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    world = corners @ R.T + np.array([cx, cy])
    return Polygon(world)


# ---------------------------------------------------------------- 评分
def score_trajectory(poses, per_actor, mats, input_idx):
    """poses:(8,3) ego 系。返回分项 dict。"""
    n = len(mats)
    # NoCollision + 记录碰撞点
    collision_pt = None
    no_collision = 1.0
    ego_polys = [ego_box_poly(*poses[i]) for i in range(N_POSES)]
    for i in range(N_POSES):
        for aid, polys in per_actor.items():
            ap = polys[i]
            if ap is not None and ego_polys[i].intersects(ap):
                no_collision = 0.0
                if collision_pt is None:
                    collision_pt = (poses[i, 0], poses[i, 1])
    # TTC: 在每步以当前速度前推 TTC_HORIZON，看是否会撞
    ttc = 1.0
    for i in range(N_POSES - 1):
        vx = (poses[i + 1, 0] - poses[i, 0]) / PRED_DT
        vy = (poses[i + 1, 1] - poses[i, 1]) / PRED_DT
        fx = poses[i, 0] + vx * TTC_HORIZON
        fy = poses[i, 1] + vy * TTC_HORIZON
        fbox = ego_box_poly(fx, fy, poses[i, 2])
        j = min(N_POSES - 1, i + int(round(TTC_HORIZON / PRED_DT)))
        for aid, polys in per_actor.items():
            ap = polys[j]
            if ap is not None and fbox.intersects(ap):
                ttc = 0.0
    # Comfort
    comfort = _comfort(poses)
    # Progress: 预测弧长 vs GT ego 行驶距离(同段)
    pred_len = float(np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)))
    end_idx = min(n - 1, input_idx + N_POSES * int(PRED_DT * FPS))
    gt_pos = mats[input_idx:end_idx + 1, :3, 3]
    gt_len = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))) if len(gt_pos) > 1 else 0.0
    if gt_len < 0.5:      # ego 基本静止(刹停) -> progress 不惩罚
        progress = 1.0
    else:
        progress = float(np.clip(pred_len / gt_len, 0.0, 1.0))
    total = no_collision * ttc * (5 * progress + 2 * comfort) / 7.0
    cov = max(0.0, (n - 1 - input_idx) / FPS)
    return dict(total=total, no_collision=no_collision, ttc=ttc, comfort=comfort,
                progress=progress, gt_coverage_s=cov, collision_pt=collision_pt,
                pred_len=pred_len, gt_len=gt_len)


def _comfort(poses):
    xy = poses[:, :2]
    head = poses[:, 2]
    v = np.diff(xy, axis=0) / PRED_DT
    speed = np.linalg.norm(v, axis=1)
    lon_acc = np.diff(speed) / PRED_DT
    yaw_rate = np.diff(head) / PRED_DT
    lat_acc = speed * yaw_rate                 # v*omega, 均为 (7,)
    lon_jerk = np.diff(lon_acc) / PRED_DT if len(lon_acc) > 1 else np.array([0.0])
    ok = (np.all(np.abs(lon_acc) <= COMFORT['max_lon_acc']) and
          np.all(np.abs(lat_acc) <= COMFORT['max_lat_acc']) and
          np.all(np.abs(yaw_rate) <= COMFORT['max_yaw_rate']) and
          np.all(np.abs(lon_jerk) <= COMFORT['max_lon_jerk']))
    return 1.0 if ok else 0.0


# ---------------------------------------------------------------- BEV 画图
def _ego2screen(x, y):
    # ego x 前向 -> 屏幕上; ego y 左向 -> 屏幕左
    return -y, x


def draw_bev(ax, poses, per_actor, actor_ids, ghost_id, agent_states, agent_labels, sc, title):
    lim = 32
    ax.set_xlim(-lim, lim); ax.set_ylim(-2, lim + 8)
    ax.set_aspect('equal'); ax.grid(True, ls=':', alpha=0.4)
    ax.set_xlabel('lateral y (m)   left <--  --> right'); ax.set_ylabel('forward x (m) -->')

    # ego 车框
    ego0 = ego_box_poly(0, 0, 0)
    ex, ey = zip(*[_ego2screen(px, py) for px, py in ego0.exterior.coords])
    ax.add_patch(MplPoly(np.c_[ex, ey], closed=True, fc='#3b7', ec='k', alpha=0.5, zorder=5))

    # actor GT box: 画输入时刻(i=0)实线 + 2s(i=3)淡色
    for aid in actor_ids:
        polys = per_actor[aid]
        is_ghost = (aid == ghost_id)
        for i, alpha, lw in [(0, 0.9, 1.6), (3, 0.35, 1.0)]:
            p = polys[i] if i < len(polys) else None
            if p is None:
                continue
            xs, ys = zip(*[_ego2screen(px, py) for px, py in p.exterior.coords])
            col = '#e33' if is_ghost else '#888'
            ax.add_patch(MplPoly(np.c_[xs, ys], closed=True, fill=False, ec=col, lw=lw, alpha=alpha, zorder=4))
        # 标签
        p0 = polys[0]
        if p0 is not None:
            cx, cy = p0.centroid.x, p0.centroid.y
            sx, sy = _ego2screen(cx, cy)
            ax.text(sx, sy, aid, fontsize=6, color='#e33' if is_ghost else '#555', ha='center', zorder=6)

    # 模型检测框 (青)
    if agent_states is not None:
        prob = 1 / (1 + np.exp(-agent_labels))
        for k in range(len(agent_states)):
            if prob[k] > 0.5:
                x, y, hd, l, w = agent_states[k][:5]
                bp = ego_box_poly(x, y, hd, forward_shift=0.0, L=max(l, 0.5), W=max(w, 0.5))
                xs, ys = zip(*[_ego2screen(px, py) for px, py in bp.exterior.coords])
                ax.add_patch(MplPoly(np.c_[xs, ys], closed=True, fill=False, ec='#0bf', lw=0.8, alpha=0.7, zorder=3))

    # 预测轨迹 (绿)
    tx, ty = zip(*[_ego2screen(px, py) for px, py in poses[:, :2]])
    tx = (0,) + tx; ty = (0,) + ty
    ax.plot(tx, ty, '-o', color='#1a1', ms=3, lw=2, zorder=7, label='pred traj')
    # 碰撞点
    if sc['collision_pt'] is not None:
        cx, cy = _ego2screen(*sc['collision_pt'])
        ax.plot([cx], [cy], 'x', color='red', ms=12, mew=3, zorder=8)

    ax.set_title(title, fontsize=8)


def score_title(source, sec, sc):
    return (f"{source} t={sec}s | total={sc['total']:.2f}  "
            f"NC={sc['no_collision']:.0f} TTC={sc['ttc']:.0f} "
            f"P={sc['progress']:.2f} C={sc['comfort']:.0f} | cov={sc['gt_coverage_s']:.1f}s")


# ---------------------------------------------------------------- 模型
def load_agent():
    from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
    from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
    cfg = TransfuserConfig(bkb_path='unused', plan_anchor_path=ANCHOR, latent=True)
    agent = TransfuserAgent(config=cfg, lr=6e-4, checkpoint_path=CKPT)
    agent.initialize()
    agent.eval()
    if torch.cuda.is_available():
        agent = agent.cuda()
    return agent


def infer(agent, cam_feat, status_feat):
    dev = next(agent.parameters()).device
    feats = {"camera_feature": cam_feat.to(dev), "status_feature": status_feat.to(dev)}
    with torch.no_grad():
        out = agent.forward(feats)
    poses = out["trajectory"][0].cpu().numpy()            # (8,3)
    ag_states = out.get("agent_states", None)
    ag_labels = out.get("agent_labels", None)
    if ag_states is not None:
        ag_states = ag_states[0].cpu().numpy()
        ag_labels = ag_labels[0].cpu().numpy().reshape(-1)
    return poses, ag_states, ag_labels


# ---------------------------------------------------------------- 单场景处理
def process_scene(agent, scene_name, out_root, tmp):
    """跑一个 scene-variant 的 8 次推理 + 评分 + 出图 + 导出。返回汇总行列表。"""
    # 复现: 每个场景前重置种子，保证与场景顺序无关、可复现
    torch.manual_seed(0); np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    trans_dir = os.path.join(DATA_ROOT, "renders", scene_name)
    trans_mp4 = os.path.join(trans_dir, "frames.mp4")
    origin_mp4 = os.path.join(DATA_ROOT, "ghosthead_result", scene_name, "seg1p0",
                              f"{scene_name}_seg1p0.mp4")
    for p in (trans_mp4, origin_mp4, os.path.join(trans_dir, "scene.json")):
        if not os.path.exists(p):
            print(f"   [skip] {scene_name}: 缺件 {p}")
            return []

    scene = load_scene(trans_dir)               # GT 来自 renders 的 scene.json
    mats = ego_mats(scene)
    n = len(mats)
    frame_idx = [min(fi, n - 1) for fi in FRAME_IDX]
    ghost_id = "ghost" if any(a["id"] == "ghost" for a in scene["frames"][0]["actors"]) else None

    out_dir = os.path.join(out_root, scene_name)
    in_dir = os.path.join(out_dir, "inputs"); bev_dir = os.path.join(out_dir, "bev")
    os.makedirs(in_dir, exist_ok=True); os.makedirs(bev_dir, exist_ok=True)

    sources = [("transfered", trans_mp4), ("origin", origin_mp4)]
    results = []      # (source, sec, poses, sc, agent_states, agent_labels, per_actor, actor_ids)

    for source, mp4 in sources:
        for sec, fi in zip(SEC_LIST, frame_idx):
            tag = f"{source}_t{sec}"
            raw_png = os.path.join(tmp, f"{scene_name}_{tag}_raw.png")
            extract_frame(mp4, fi, raw_png)
            img = cv2.cvtColor(cv2.imread(raw_png), cv2.COLOR_BGR2RGB)
            crop = crop_4to1_no_sky(img)
            cv2.imwrite(os.path.join(in_dir, f"{tag}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cam = build_camera_feature(crop)

            v_ego, a_ego = ego_status_at(mats, fi)
            status = torch.from_numpy(np.concatenate([DRIVING_COMMAND, v_ego, a_ego])).float().unsqueeze(0)

            poses, ag_states, ag_labels = infer(agent, cam, status)
            per_actor, actor_ids = actor_footprints_in_ego(scene, mats, fi)
            sc = score_trajectory(poses, per_actor, mats, fi)
            results.append((source, sec, poses, sc, ag_states, ag_labels, per_actor, actor_ids))

    # ---- 单张 BEV
    for source, sec, poses, sc, ag_s, ag_l, per_actor, actor_ids in results:
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_bev(ax, poses, per_actor, actor_ids, ghost_id, ag_s, ag_l, sc,
                 score_title(source, sec, sc))
        fig.tight_layout()
        fig.savefig(os.path.join(bev_dir, f"bev_{source}_t{sec}.png"), dpi=120)
        plt.close(fig)

    # ---- 2x4 对比总览
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    for row, source in enumerate(["transfered", "origin"]):
        for col, sec in enumerate(SEC_LIST):
            r = next(x for x in results if x[0] == source and x[1] == sec)
            _, _, poses, sc, ag_s, ag_l, per_actor, actor_ids = r
            draw_bev(axes[row][col], poses, per_actor, actor_ids, ghost_id, ag_s, ag_l, sc,
                     score_title(source, sec, sc))
    fig.suptitle(f"{scene_name}  |  DiffusionDrive pred-traj & PDM-style BEV score  "
                 f"(top=transfered / bottom=origin)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "bev_compare.png"), dpi=110)
    plt.close(fig)

    # ---- scores.csv + pred_traj.json
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "frame_sec", "total", "no_collision", "ttc", "comfort",
                    "progress", "gt_coverage_s", "pred_len_m", "gt_len_m"])
        for source, sec, poses, sc, *_ in results:
            w.writerow([source, sec, f"{sc['total']:.4f}", sc['no_collision'], sc['ttc'],
                        sc['comfort'], f"{sc['progress']:.4f}", f"{sc['gt_coverage_s']:.2f}",
                        f"{sc['pred_len']:.2f}", f"{sc['gt_len']:.2f}"])

    dump = {"scene": scene_name, "driving_command": DRIVING_COMMAND.tolist(),
            "crop_center_row": CROP_CENTER_ROW, "note": "PDM-style score (non-official), drivable-area omitted",
            "runs": []}
    for source, sec, poses, sc, *_ in results:
        d = {k: v for k, v in sc.items() if k != "collision_pt"}
        d.update(source=source, frame_sec=sec, poses=poses.tolist())
        dump["runs"].append(d)
    with open(os.path.join(out_dir, "pred_traj.json"), "w") as f:
        json.dump(dump, f, indent=2)

    variant = scene_name.split("__")[-1]
    rows = [(scene_name, variant, source, sec, sc['total'], sc['no_collision'], sc['ttc'],
             sc['comfort'], sc['progress'], sc['gt_coverage_s']) for source, sec, poses, sc, *_ in results]
    return rows


def discover_all():
    """返回两边齐备的 scene-variant 名列表(排序)。"""
    root = os.path.join(DATA_ROOT, "renders")
    names = []
    for s in sorted(os.listdir(root)):
        if not s.startswith("gh_") or "__" not in s:
            continue
        trans_mp4 = os.path.join(root, s, "frames.mp4")
        origin_mp4 = os.path.join(DATA_ROOT, "ghosthead_result", s, "seg1p0", f"{s}_seg1p0.mp4")
        if os.path.exists(trans_mp4) and os.path.exists(origin_mp4) and \
           os.path.exists(os.path.join(root, s, "scene.json")):
            names.append(s)
    return names


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="gh_001000__brake", help="单个 scene-variant 名")
    ap.add_argument("--all", action="store_true", help="跑全部齐备的 scene-variant")
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer")
    args = ap.parse_args()

    scenes = discover_all() if args.all else [args.scene]
    print(f">> 待跑 {len(scenes)} 个 scene-variant")

    print(">> 加载 agent ...")
    agent = load_agent()
    tmp = tempfile.mkdtemp()

    all_rows = []
    for i, sn in enumerate(scenes):
        try:
            rows = process_scene(agent, sn, args.out, tmp)
        except Exception as e:
            print(f"   [FAIL] {sn}: {e}")
            rows = []
        if rows:
            m_tr = np.mean([r[4] for r in rows if r[2] == "transfered"])
            m_or = np.mean([r[4] for r in rows if r[2] == "origin"])
            print(f"[{i+1}/{len(scenes)}] {sn}: mean total  transfered={m_tr:.2f}  origin={m_or:.2f}")
        all_rows.extend(rows)

    # ---- 全局汇总
    if args.all and all_rows:
        os.makedirs(args.out, exist_ok=True)
        summ = os.path.join(args.out, "_summary_all.csv")
        with open(summ, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scene", "variant", "source", "frame_sec", "total",
                        "no_collision", "ttc", "comfort", "progress", "gt_coverage_s"])
            for r in all_rows:
                w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.4f}", r[5], r[6], r[7],
                            f"{r[8]:.4f}", f"{r[9]:.2f}"])
        print(f"\n>> 汇总表: {summ}")
        # 按 variant × source 的均值
        print(">> variant × source 平均 total 分:")
        for v in ["none", "brake", "swerve"]:
            for src in ["transfered", "origin"]:
                vals = [r[4] for r in all_rows if r[1] == v and r[2] == src]
                if vals:
                    print(f"   {v:7s} {src:10s} mean_total={np.mean(vals):.3f}  n={len(vals)}")
    print(">> 完成")


if __name__ == "__main__":
    main()
