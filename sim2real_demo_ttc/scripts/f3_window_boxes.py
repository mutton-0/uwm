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


NS_CFG = "/data/ruolin/uwm/sim2real_demo_ttc/configs/navsim_corpus.yaml"
NS_LOGS = Path("/data/dataset/navsim/dataset/navsim_logs")


class WindowBoxes:
    """按 scene 缓存 geo，提供 (event, 时间戳列表) -> 逐帧 bbox / 3D 框的查询。

    **两个语料的后端（§FC/A60）**：
      corpus="nuscenes"（默认，行为与本模块首版逐位一致）
          geo 来自 `g1_mine_events.compute_scene_geometry(nusc, scene, cfg)`；
          `_rot` 存的是**世界系**朝向 ⇒ 取 ego 系朝向要减去 ego 航向 ψ。
      corpus="navsim"
          geo 来自 `ns1_navsim_geometry.build_geo(log 帧, cfg)`（同一份结构，字段逐个对齐）；
          NAVSIM 的 `gt_boxes[:, 6]` **本就在 ego(=lidar) 系**（`ns1` 注释即写明位置在 lidar 系，
          且 lidar2ego 为恒等），故 `_rot` 里存的是 **ego 系**朝向 ⇒ **不能再减 ψ**。
          该约定不是读代码推断的，是实测判定的：取 ego 转向 > 0.15 rad 的 scene 里 276 条车辆轨迹，
          `std(yaw)` 0.308 → `std(yaw + ψ)` 0.106，77.2% 的轨迹加上 ψ 后才沿时间稳定
          （停驻/直行车辆的**世界系**朝向应近似恒定）。见 `scripts/ns_yaw_audit.py`。
    """

    def __init__(self, nusc, cfg_path=CFG, tol_s=0.06, corpus="nuscenes", split="test"):
        self.nusc = nusc
        self.corpus = corpus
        self.split = split
        self.cfg = OmegaConf.to_container(
            OmegaConf.load(NS_CFG if corpus == "navsim" else cfg_path), resolve=True)
        # `_rot` 里存的朝向在哪个坐标系 —— 决定 box3d_for 要不要减 ψ_ego
        self.yaw_frame = "ego" if corpus == "navsim" else "world"
        self.tol = float(tol_s)
        self._geo = {}
        self._scene = ({s["name"]: s for s in nusc.scene}
                       if corpus == "nuscenes" else None)
        self._log_cache = {}
        self.stats = {"frames_total": 0, "frames_visible": 0, "frames_no_box": 0,
                      "frames_off_grid": 0, "events_no_geo": 0}

    # ---- geo 后端 ----
    def _geo_navsim(self, ev):
        import pickle
        from collections import defaultdict as _dd
        import ns1_navsim_geometry as NS
        log_name = ev["log_name"]
        if log_name not in self._log_cache:
            if len(self._log_cache) > 2:                 # log pkl 很大，只留最近几个
                self._log_cache.pop(next(iter(self._log_cache)))
            lf = NS_LOGS / self.split / f"{log_name}.pkl"
            if not lf.exists():
                cand = list((NS_LOGS / self.split).glob(f"{log_name}*.pkl"))
                if not cand:
                    raise FileNotFoundError(f"NAVSIM log not found: {log_name}")
                lf = cand[0]
            by = _dd(list)
            for f in pickle.load(open(lf, "rb")):
                by[f["scene_token"]].append(f)
            self._log_cache[log_name] = by
        fl = self._log_cache[log_name].get(ev["scene_token"])
        if not fl:
            raise KeyError(ev["scene_token"])
        return NS.build_geo(sorted(fl, key=lambda z: z["timestamp"]), self.cfg, self.split)

    def geo(self, ev):
        """ev 可以是事件 dict，也可以是 scene_name 字符串（nuScenes 向后兼容）。"""
        if isinstance(ev, str):
            ev = {"scene_name": ev}
        key = ((ev.get("scene_token") or ev["scene_name"]) if self.corpus == "navsim"
               else ev["scene_name"])
        if key not in self._geo:
            if len(self._geo) > 3:                       # geo 很大，只留最近几个 scene
                self._geo.pop(next(iter(self._geo)))
            self._geo[key] = (self._geo_navsim(ev) if self.corpus == "navsim"
                              else G1.compute_scene_geometry(
                                  self.nusc, self._scene[ev["scene_name"]], self.cfg))
        return self._geo[key]

    def boxes_for(self, ev, timestamps_s):
        """timestamps_s: 窗口内每一帧的**绝对时间（秒）**，与 geo['grid_t'] 同一时基。

        返回与之等长的 list，每项是 [x0,y0,x1,y1] 或 None（该帧投影不出来/不可见）。
        """
        try:
            geo = self.geo(ev)
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

    def box3d_for(self, ev, t_s):
        """返回该实体在时刻 t_s 的 **ego 系 3D 框**：(center[3], (w, l, h), yaw) 或 None。

        取值与 `frame_bbox` 的 2D 投影**同源**：同一个 `p_ego` 中心、同一个 `_size`、
        同一条按最近关键帧取的朝向 `_rot`。故"删掉这个 3D 框里的点"与
        "涂掉这个 3D 框投影出的 2D 框"作用在同一个物理实体上，不是两套独立近似。
        """
        from pyquaternion import Quaternion
        try:
            geo = self.geo(ev)
        except Exception:                                              # noqa: BLE001
            return None
        o = geo["per_obj"].get(ev["object_token"])
        if o is None or not o.get("_rot") or o["_size"] is None:
            return None
        gt = geo["grid_t"]
        j = int(np.argmin(np.abs(gt - t_s)))
        if abs(gt[j] - t_s) > self.tol or not bool(o["valid"][j]):
            return None
        ti = int(np.argmin(np.abs(np.asarray(o["_rot_t"]) - gt[j])))
        yaw_s = Quaternion(o["_rot"][ti]).yaw_pitch_roll[0]
        # nuScenes：_rot 是世界系 ⇒ 减 ego 航向；NAVSIM：_rot 本就在 ego 系 ⇒ 不减（见类注释）
        yaw_e = (yaw_s if self.yaw_frame == "ego"
                 else yaw_s - Quaternion(matrix=geo["R_we"][j]).yaw_pitch_roll[0])
        w, l, h = (list(o["_size"]) + [0, 0, 0])[:3]
        c = np.asarray(o["p_ego"][j], float)
        if not np.all(np.isfinite(c)):
            return None
        return c, (float(w), float(l), float(h)), float(yaw_e)


def points_in_box(pts, center, size, yaw):
    """ego 系点云落在该 3D 框内的布尔掩码。size = (w, l, h)，x 方向长 l、y 方向宽 w
    （与 `g1_mine_events.box_corners_ego` 的约定逐字段一致）。"""
    if pts is None or len(pts) == 0:
        return np.zeros(0, bool)
    w, l, h = size
    d = np.asarray(pts, float)[:, :3] - np.asarray(center, float)[None, :]
    c, s = np.cos(-yaw), np.sin(-yaw)
    x = c * d[:, 0] - s * d[:, 1]
    y = s * d[:, 0] + c * d[:, 1]
    return (np.abs(x) <= l / 2) & (np.abs(y) <= w / 2) & (np.abs(d[:, 2]) <= h / 2)


def mirror_box3d(center, size, yaw):
    """把 3D 框沿 ego 纵轴镜像（y -> −y，yaw -> −yaw）—— 对照臂的 3D 体积。

    与 RGB 侧 `control_box` 的**水平镜像**同一几何构造：
    纵向距离、尺寸、俯仰不变，只把横向位置翻到另一侧 ⇒ 体积严格相等、成像离心率带相同。
    """
    c = np.asarray(center, float).copy()
    c[1] = -c[1]
    return c, size, -yaw
