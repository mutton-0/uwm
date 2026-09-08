"""把 deployment 池的遮挡组 2D 框导出成纯 JSON，供 py312 环境里的 VLA F-3 使用。

py312 没有 nuscenes-devkit / omegaconf 依赖链，算不了 geo；
框的定义与 f3_brakefirst 完全一致（G1.frame_bbox 作用在同一 geo 与同一帧上）。

## 本版：**全窗口 × 全相机**（§FM/A56 向 NAVSIM 路径的补齐）
旧版只导出「前视 × 查询帧」一格。但两个 VLA 吃 3–4 相机 × 4 帧，
于是干预只覆盖 1/12（AutoVLA）或 1/16（Alpamayo）张图，
实测**前视的 3 个历史帧里 38/40 个事件目标仍然清晰可见**（侧相机 5–10%）。
跨模型比 b / θ 时这等于给 VLA 施加了弱一个数量级的扰动。

现改为：对 CAM_F0 / CAM_L0 / CAM_R0 × 窗口 4 帧各投影一次。
投影沿用 `G1.frame_bbox` 的同一套数学，只把写死的前视 (K, R_ec, t_ec)
换成该相机自己的标定（NAVSIM log 里 8 个相机全有）。
"""
from __future__ import annotations
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np                                                     # noqa: E402
from pyquaternion import Quaternion                                     # noqa: E402
import g1_mine_events as G1                                             # noqa: E402
import ns1_navsim_geometry as NS                                        # noqa: E402

CAMS = ["CAM_F0", "CAM_L0", "CAM_R0"]      # 两个 VLA 实际消费的相机
N_FRAMES = 4                                # 时序窗口长度，两家一致


def cam_bbox(geo, o, j, K, R_ec, t_ec, im_w, im_h):
    """任意相机、任意帧的 2D 框。与 G1.frame_bbox 同一套数学，只是标定可变。"""
    if o is None or not o.get("_rot") or o["_size"] is None:
        return None
    if j < 0 or j >= len(o["p_ego"]) or not bool(o["visible"][j]):
        return None
    ti = int(np.argmin(np.abs(np.asarray(o["_rot_t"]) - geo["grid_t"][j])))
    yaw = (Quaternion(o["_rot"][ti]).yaw_pitch_roll[0]
           - Quaternion(matrix=geo["R_we"][j]).yaw_pitch_roll[0])
    corners = G1.box_corners_ego(o["p_ego"][j], o["_size"], yaw)
    cam = (corners - t_ec) @ R_ec
    if (cam[:, 2] <= 0.1).any():                 # 有角点在相机后方 -> 不投
        return None
    uv = (K @ cam.T).T
    u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
    u0, v0, u1, v1 = float(u.min()), float(v.min()), float(u.max()), float(v.max())
    if u1 < 0 or v1 < 0 or u0 > im_w or v0 > im_h:      # 完全出画
        return None
    return [max(u0, 0.0), max(v0, 0.0), min(u1, im_w - 1.0), min(v1, im_h - 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    G1.set_include_animal(True)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"),
                                 resolve=True)
    cache = defaultdict(list)
    for sp in ("test", "trainval"):
        d0 = NS.NS_ROOT / "navsim_logs" / sp
        if not d0.exists():
            continue
        for lf in sorted(d0.glob("*.pkl")):
            for f in pickle.load(open(lf, "rb")):
                cache[(sp, f["scene_name"])].append(f)
    for k in cache:
        cache[k].sort(key=lambda z: z["timestamp"])

    out, geo_cache = [], {}
    for c in json.load(open(RES / args.pool))["candidates"]:
        sp = c.get("split", "test"); key = (sp, c["scene"])
        if key not in geo_cache:
            geo_cache[key] = NS.build_geo(cache[key], cfg, sp)
        geo = geo_cache[key]; j = c["frame_idx"]
        fl = cache[key]
        objs = [geo["per_obj"].get(g["token"]) for g in c["f3_mask_group"]]
        # 旧字段：前视 × 查询帧，保持逐位不变，向后兼容
        boxes = []
        for o in objs:
            bb = G1.frame_bbox(geo, o, j) if o is not None else None
            if bb is not None:
                boxes.append([float(x) for x in bb])
        # 新字段：全相机 × 全窗口。键 "CAM|k"，k=0..3，k=3 是查询帧（窗口最后一帧）
        win = {}
        for cam in CAMS:
            c0 = fl[j]["cams"][cam]
            K = np.asarray(c0["cam_intrinsic"], float)
            R_ec = np.asarray(c0["sensor2lidar_rotation"], float)
            t_ec = np.asarray(c0["sensor2lidar_translation"], float)
            for k in range(N_FRAMES):
                jj = max(j - (N_FRAMES - 1 - k), 0)
                bbs = []
                for o in objs:
                    bb = cam_bbox(geo, o, jj, K, R_ec, t_ec, 1920.0, 1080.0)
                    if bb is not None:
                        bbs.append(bb)
                win[f"{cam}|{k}"] = {"path": fl[jj]["cams"][cam]["data_path"],
                                     "boxes": bbs, "frame_idx": jj}
        out.append({"split": sp, "scene": c["scene"], "frame_idx": j,
                    "cam_f0": fl[j]["cams"]["CAM_F0"]["data_path"],
                    "boxes": boxes, "window": win})
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    nb = sum(1 for x in out if x["boxes"])
    from collections import Counter
    cov = Counter()
    for x in out:
        for k, v in x.get("window", {}).items():
            if v["boxes"]:
                cov[k] += 1
    print(f"[BOXES] {len(out)} 事件，其中 {nb} 个前视查询帧有框 -> {args.out}")
    print("  全窗口覆盖（有框的事件数）：")
    for cam in CAMS:
        r = [cov[f"{cam}|{k}"] for k in range(N_FRAMES)]
        print(f"    {cam}: 帧0={r[0]:4d} 帧1={r[1]:4d} 帧2={r[2]:4d} 帧3(查询)={r[3]:4d}")


if __name__ == "__main__":
    main()
