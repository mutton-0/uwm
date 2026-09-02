"""F-3 多帧遮挡修复|按**窗口内每一帧**独立把危险实体的 3D 框投影成 2D 框。

工单：docs/f3_multiframe_occlusion_fix_workorder.md。

**原实现的漏洞**：`f3_occlusion_vla.py` 只遮窗口的最后一帧（t0），
隐含假设"更早的帧早于 emergence，本来就看不到实体"。实测这个假设对 Alpamayo **完全不成立**：
它的窗口是 4 帧 × 0.1 s，总跨度 0.3 s，而 t0 − t_emergence 的分位是 [0.30, 0.35, 0.40] s，
⇒ **288 个 A 类事件里 280 个的 4 帧全部落在 emergence 之后**，另外 8 个也有 3 帧落在之后。
只遮最后一帧时，模型仍能从前 3 帧看见实体，测到的是"能不能从历史帧里看出来"，
而不是"实体从输入里消失后动作会不会退回基线"。

**本模块做什么**：复用挖掘阶段**已有**的投影逻辑
（`g1_mine_events.compute_scene_geometry` + `frame_bbox`，一行未改），
对窗口内每个时间戳找到对应的 nuScenes CAM_FRONT 帧，重新投影该实体，返回逐帧的 2D 框。
某一帧投影不出来（出画幅 / 深度为负 / 该帧无标注）就返回 None，该帧保持不动——
**不做近似、不外推**，并把这种情况计入统计。

**为什么可以用挖掘期的网格**：两个 VLA 取的都是**真实存在的 nuScenes CAM_FRONT 图像**
（Alpamayo 用 `find_nearest_camera_images` 找最近邻，AutoVLA 沿 sample 的 prev 链取关键帧），
而挖掘期的 `geo` 网格正是该 scene 的全部 CAM_FRONT 帧（含 sweeps）。
故"窗口时间戳 → 网格下标"是精确匹配而不是插值；超过 `--tol` 的匹配一律判失败并计数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g1_mine_events as G1                                            # noqa: E402

CFG = "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"


class WindowBoxes:
    """按 scene 缓存 geo，提供 (event, 时间戳列表) -> 逐帧 bbox 的查询。"""

    def __init__(self, nusc, cfg_path=CFG, tol_s=0.06):
        self.nusc = nusc
        self.cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        self.tol = float(tol_s)
        self._geo = {}
        self._scene = {s["name"]: s for s in nusc.scene}
        self.stats = {"frames_total": 0, "frames_visible": 0, "frames_no_box": 0,
                      "frames_off_grid": 0, "events_no_geo": 0}

    def geo(self, scene_name):
        if scene_name not in self._geo:
            if len(self._geo) > 3:                       # geo 很大，只留最近几个 scene
                self._geo.pop(next(iter(self._geo)))
            self._geo[scene_name] = G1.compute_scene_geometry(
                self.nusc, self._scene[scene_name], self.cfg)
        return self._geo[scene_name]

    def boxes_for(self, ev, timestamps_s):
        """timestamps_s: 窗口内每一帧的**绝对时间（秒）**，与 geo['grid_t'] 同一时基。

        返回与之等长的 list，每项是 [x0,y0,x1,y1] 或 None（该帧投影不出来/不可见）。
        """
        try:
            geo = self.geo(ev["scene_name"])
        except Exception:                                              # noqa: BLE001
            self.stats["events_no_geo"] += 1
            return [None] * len(timestamps_s)
        o = geo["per_obj"].get(ev["object_token"])
        if o is None:
            self.stats["events_no_geo"] += 1
            return [None] * len(timestamps_s)
        gt = geo["grid_t"]
        out = []
        for t in timestamps_s:
            self.stats["frames_total"] += 1
            j = int(np.argmin(np.abs(gt - t)))
            if abs(gt[j] - t) > self.tol:
                self.stats["frames_off_grid"] += 1
                out.append(None); continue
            if not bool(o["visible"][j]):
                self.stats["frames_no_box"] += 1
                out.append(None); continue
            bb = G1.frame_bbox(geo, o, j)                # ← 挖掘期同一套投影，未改
            if bb is None:
                self.stats["frames_no_box"] += 1
                out.append(None); continue
            self.stats["frames_visible"] += 1
            out.append(bb)
        return out
