"""为两个 VLA 候选构造 NAVSIM 侧输入 —— 它们的 nuScenes devkit 只是路径/位姿解析器。

## 为什么可以这么做
AutoVLA 的 `run(cam_sd_token, ...)` 里 devkit 唯一的作用是把 token 解析成
`feats["images"]`（3 相机 × 4 帧的**绝对文件路径**）；下游 `get_prompt`/`predict`
吃的就是普通路径。Alpamayo 的 `load_nuscenes()` 只返回 image_frames / ego_history_xyz /
ego_history_rot 三样。两者都**不依赖 nuScenes 本身**。

## 相机对应与它的代价
nuScenes CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT
  ->  NAVSIM  CAM_F0    / CAM_L0          / CAM_R0
内参不同（NAVSIM fx=1545 vs nuScenes fx=1266），装车位置与 FOV 也不同。
**这不是要消除的混淆，而是 benchmark→deployment 域偏移的一部分**（用户 2026-09-05 明确）。
报告时必须写明：这两个候选在 deployment 上的分数含相机几何差异。
"""
from __future__ import annotations
import pickle, glob
from collections import defaultdict
from pathlib import Path

NS_LOGS = "/data/dataset/navsim/dataset/navsim_logs"
NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"
CAM_MAP = {"front_camera": "CAM_F0", "front_left_camera": "CAM_L0",
           "front_right_camera": "CAM_R0"}
N_TEMPORAL = 4


class NavsimVLAInput:
    """按 (split, scene) 索引；scene_name 跨 split 不唯一，必须复合键。"""

    def __init__(self, splits=("test", "trainval")):
        self.fr = defaultdict(list)
        for sp in splits:
            d = Path(NS_LOGS) / sp
            if not d.exists():
                continue
            for lf in sorted(d.glob("*.pkl")):
                for f in pickle.load(open(lf, "rb")):
                    self.fr[(sp, f["scene_name"])].append(f)
        for k in self.fr:
            self.fr[k].sort(key=lambda z: z["timestamp"])

    def frames(self, split, scene):
        return self.fr[(split, scene)]

    def images(self, split, scene, j):
        """3 相机 × 4 帧绝对路径（窗口以第 j 帧结尾，前端不足则重复首帧）。"""
        fl = self.fr[(split, scene)]
        idx = [max(0, j - k) for k in range(N_TEMPORAL)][::-1]     # 旧 -> 新
        out = {}
        for key, cam in CAM_MAP.items():
            paths = []
            for i in idx:
                c = fl[i]["cams"].get(cam)
                if c is None:
                    return None
                p = Path(NS_BLOBS) / split / c["data_path"]
                if not p.exists():
                    return None
                paths.append(str(p))
            out[key] = paths
        return out

    def ego(self, split, scene, j):
        """自车速度与加速度（ego_dynamic_state = [vx, vy, ax, ay]，已在 ego 系）。"""
        st = self.fr[(split, scene)][j]["ego_dynamic_state"]
        return float(st[0]), float(st[2]) if len(st) > 2 else 0.0

    def ego_history(self, split, scene, j, n=N_TEMPORAL):
        """自车位姿史（全局系 xyz + 四元数），窗口以第 j 帧结尾。"""
        fl = self.fr[(split, scene)]
        idx = [max(0, j - k) for k in range(n)][::-1]
        xyz = [list(map(float, fl[i]["ego2global_translation"])) for i in idx]
        rot = [list(map(float, fl[i]["ego2global_rotation"])) for i in idx]
        return xyz, rot
