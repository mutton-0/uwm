"""Alpamayo-R1 的 NAVSIM 侧加载器：与 `load_nuscenes()` 返回同构的字典。

模型只吃 image_frames / camera_indices / ego_history_xyz / ego_history_rot，
`load_nuscenes` 的其余作用只是把 nuScenes 的表结构解析成这四样。这里换成 NAVSIM
的日志与图像，逐项对齐它的构造方式（不 resize、不归一化、同样转到 t0 局部系）。

相机位对应（沿用 NUSCENES_CAMERA_MAPPING 的槽位语义）：
    槽 0 前左  <- CAM_L0 ；槽 1 前  <- CAM_F0 ；槽 2 前右 <- CAM_R0 ；
    槽 6 前窄  <- CAM_F0（nuScenes 侧同样复用 CAM_FRONT，此处保持一致）
NAVSIM 内参 fx=1545 vs nuScenes 1266，FOV/装车位置亦不同 ——
**这属于 benchmark→deployment 的域偏移本身，不作校正**，但必须随结果报告。
"""
from __future__ import annotations
import numpy as np


def load_navsim(vin, split, scene, j, *, num_history_steps=16, time_step=0.1,
                num_frames=4):
    """vin: vla_navsim_input.NavsimVLAInput。j 为查询帧下标。"""
    import torch
    from PIL import Image
    from einops import rearrange
    from scipy.spatial.transform import Rotation, Slerp
    from scipy.interpolate import interp1d

    fl = vin.frames(split, scene)
    t = np.array([f["timestamp"] for f in fl], dtype=np.int64)
    xyz = np.array([f["ego2global_translation"] for f in fl], float)
    quat_wxyz = np.array([f["ego2global_rotation"] for f in fl], float)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]                    # -> scipy 约定

    t0 = int(t[j])
    hist_ts = t0 - np.arange(num_history_steps - 1, -1, -1) * int(time_step * 1e6)
    hist_ts = np.clip(hist_ts, t.min(), t.max())

    fx = interp1d(t, xyz, axis=0, bounds_error=False,
                  fill_value=(xyz[0], xyz[-1]))
    hist_xyz = fx(hist_ts)
    slerp = Slerp(t, Rotation.from_quat(quat_xyzw))
    hist_rot = slerp(hist_ts)
    hq = hist_rot.as_quat()

    t0_xyz = hist_xyz[-1].copy()
    t0_inv = Rotation.from_quat(hq[-1]).inv()
    hist_xyz_local = t0_inv.apply(hist_xyz - t0_xyz)
    hist_rot_local = (t0_inv * Rotation.from_quat(hq)).as_matrix()

    img_ts = np.array([t0 - (num_frames - 1 - i) * int(time_step * 1e6)
                       for i in range(num_frames)], dtype=np.int64)
    idx = [int(np.argmin(np.abs(t - ts))) for ts in img_ts]

    CAMS = [("CAM_L0", 0), ("CAM_F0", 1), ("CAM_R0", 2), ("CAM_F0", 6)]
    frames_list, idx_list, ts_list = [], [], []
    for cam, slot in CAMS:
        paths = []
        for i in idx:
            c = fl[i]["cams"].get(cam)
            if c is None:
                return None
            paths.append(f"/data/dataset/navsim/dataset/sensor_blobs/{split}/{c['data_path']}")
        try:
            imgs = np.stack([np.array(Image.open(p).convert("RGB")) for p in paths])
        except Exception:                                     # noqa: BLE001
            return None
        frames_list.append(rearrange(torch.from_numpy(imgs), "t h w c -> t c h w"))
        idx_list.append(slot)
        ts_list.append(torch.from_numpy(t[idx].copy()))

    image_frames = torch.stack(frames_list, 0)
    camera_indices = torch.tensor(idx_list, dtype=torch.int64)
    all_ts = torch.stack(ts_list, 0)
    order = torch.argsort(camera_indices)
    image_frames, camera_indices, all_ts = (image_frames[order], camera_indices[order],
                                            all_ts[order])
    rel = (all_ts - all_ts.min()).float() * 1e-6
    return {
        "image_frames": image_frames, "camera_indices": camera_indices,
        "ego_history_xyz": torch.from_numpy(hist_xyz_local).float()[None, None],
        "ego_history_rot": torch.from_numpy(hist_rot_local).float()[None, None],
        "relative_timestamps": rel, "t0_us": t0, "scene_name": scene,
    }
