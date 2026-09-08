"""Alpamayo 1 (R1-10B) 的模型解耦接口 —— 与 simlingo_runner 同约定。

工单 P（guide 附录 D）的阳性对照模型。复用 /data/Zhengyang/alpamayo 的仓库与依赖
（含已为 sm_120 编译好的 flash_attn 2.8.3）；权重在 /data/ruolin/alpamayo_ckpt。

对齐 SimLingo 的口径：
  * 行为量 v_plan —— Alpamayo 输出 64 航点 @10Hz；取 t=0.2→0.8 s 的位移求速度，
    与 SimLingo 的 commanded_speed（0.25→0.75 s 窗口）尽可能同窗（10 Hz 网格所限）；
  * b = v_plan(clean) − v_plan(ghost)，正 = 危险帧里规划得更慢（刹车）；
  * 语言端取 extra["cot"]（Chain-of-Causation），供 T2.5 同口径的提及率读数。

**适配偏离（必须随结果一起报）**：
  1. Alpamayo 训练在 NVIDIA PhysicalAI-AV 上，对 nuScenes 同样是 OOD —— 不是
     「按构造保证有货」的 nuScenes 原生规划器，故 P1 资格赛是硬闸门；
  2. 相机 FOV 失配：Alpamayo 期待 120° 广角 + 30° 长焦，nuScenes 六相机均为 70°，
     只能近似映射（CAM_FRONT_LEFT→cross_left、CAM_FRONT→front_wide/front_tele、
     CAM_FRONT_RIGHT→cross_right），长焦位复用 CAM_FRONT；
  3. 轨迹采样是随机的（扩散 + VLM rollout with temperature）——固定 seed，
     并在 clean/ghost 之间复用同一 seed，使配对差不被采样噪声主导。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# 依赖来自 Zhengyang 的 venv（site-packages + editable src），见 alp_env.sh
for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
           "/data/Zhengyang/alpamayo/src"):
    if _p not in sys.path:
        sys.path.append(_p)

DEFAULT_CKPT = "/data/ruolin/alpamayo_ckpt"
NUSC_ROOT = "/data/dataset/nuscenes/v1.0-trainval"

# v_plan 的时间窗（秒）—— 与 SimLingo commanded_speed 的 0.25–0.75 s 对齐
V_T0, V_T1 = 0.2, 0.8
DT = 0.1


@dataclass
class AlpaResult:
    traj: np.ndarray                     # (T, 3) 规划轨迹（ego 系，t0 处）
    v_plan: float                        # 规划目标速度 [m/s]
    cot: str = ""                        # Chain-of-Causation 推理文本
    hidden: dict = field(default_factory=dict)
    n_layers: int = 0


def plan_speed(traj: np.ndarray) -> float:
    """与 SimLingo 同口径的规划速度：取固定时间窗内的位移 / 时长。"""
    i0, i1 = int(round(V_T0 / DT)), int(round(V_T1 / DT))
    if len(traj) <= i1:
        return float("nan")
    return float(np.linalg.norm(traj[i1, :2] - traj[i0, :2]) / (V_T1 - V_T0))


class AlpamayoRunner:
    def __init__(self, ckpt: str = DEFAULT_CKPT, device: str = "cuda",
                 dataroot: str = NUSC_ROOT, seed: int = 42):
        from alpamayo_r1 import helper
        from alpamayo_r1.load_nuscenes import NuScenesDataInterface
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

        self.seed = seed
        self.device = device
        self.helper = helper
        self.model = AlpamayoR1.from_pretrained(ckpt, dtype=torch.bfloat16).to(device).eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        self.ndi = NuScenesDataInterface(dataroot=dataroot)
        self._hooks = []
        self._layer_out = []

    # ---------------- 数据 ----------------
    def _scene_range(self, scene_name: str):
        """场景的 ego_pose 时间覆盖 [t_min, t_max]（µs），带缓存。"""
        if not hasattr(self, "_rng_cache"):
            self._rng_cache = {}
        if scene_name not in self._rng_cache:
            tok = next(s["token"] for s in self.ndi.list_scenes() if s["name"] == scene_name)
            ts, _tr, _rot = self.ndi.get_scene_ego_poses(tok)
            self._rng_cache[scene_name] = (int(ts.min()), int(ts.max()))
        return self._rng_cache[scene_name]

    def covers(self, scene_name: str, t_sec: float,
               hist_s: float = 1.6, fut_s: float = 6.4) -> bool:
        """t0 前 hist_s、后 fut_s 是否都在 ego_pose 覆盖内。

        adapter 会无条件构造 6.4 s 的未来轨迹（用于 GT ADE），因此贴近场景末尾的
        事件不可用。**预筛比让模型跑完再抛异常便宜得多**，且可用率要随结果一起报。
        """
        try:
            lo, hi = self._scene_range(scene_name)
        except Exception:
            return False
        t0 = t_sec * 1e6
        return (t0 - hist_s * 1e6) >= lo and (t0 + fut_s * 1e6) <= hi

    def load(self, scene_name: str, t_sec: float):
        from alpamayo_r1.load_nuscenes import load_nuscenes
        return load_nuscenes(scene_name, t0_us=int(round(t_sec * 1e6)), ndi=self.ndi)

    # ---------------- 前向 ----------------
    @torch.no_grad()
    def infer(self, scene_name: str, t_sec: float, num_traj_samples: int = 1,
              max_generation_length: int = 256, capture_hidden: bool = False,
              ego_anchor_t: Optional[float] = None,
              data: Optional[dict] = None) -> AlpaResult:
        """ego_anchor_t 非 None 时，自车运动史取自该时刻，图像仍取 t_sec。

        与 SimLingo 的 prompt_anchor 同构：clean/ghost 两条件本就相隔 ~1.5 s，
        自车运动史本身不同，不扣掉的话 b 里混着「这两帧之间自车快了还是慢了」。
        SimLingo 侧把 prompt 的 Current speed 锚到 clean（commit 5c1366d），
        这里把 ego_history 锚到 clean，是**同一处理**，不是更友好的设定。
        """
        # data 非 None 时直接用它，跳过 nuScenes devkit —— `load_nuscenes()` 在这里
        # 只是把 nuScenes 的表结构解析成 image_frames / ego_history_*，模型本身
        # 不依赖 nuScenes。NAVSIM 侧由 alpa_navsim_loader.load_navsim() 构造同构字典。
        if data is None:
            data = self.load(scene_name, t_sec)
        if ego_anchor_t is not None:
            anchor = self.load(scene_name, ego_anchor_t)
            data["ego_history_xyz"] = anchor["ego_history_xyz"]
            data["ego_history_rot"] = anchor["ego_history_rot"]
        messages = self.helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        model_inputs = self.helper.to_device({
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }, self.device)

        # 配对两条件复用同一 seed —— 采样噪声不进 b
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if capture_hidden:
            self._layer_out = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.98, temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=max_generation_length, return_extra=True)

        traj = pred_xyz.float().cpu().numpy()[0, 0, 0]        # (T, 3)
        cot = ""
        if extra and "cot" in extra and extra["cot"]:
            c = extra["cot"][0]
            cot = c if isinstance(c, str) else (" ".join(map(str, c)) if isinstance(c, (list, tuple)) else str(c))
        res = AlpaResult(traj=traj, v_plan=plan_speed(traj), cot=cot)
        if capture_hidden and self._layer_out:
            res.n_layers = len(self._layer_out)
        return res

    # ---------------- 钩子（P2 用） ----------------
    def register_hooks(self, prefix: str = "vlm.model.language_model.layers"):
        """在骨干 decoder layer 上挂 forward hook 抓 hidden states。"""
        self.remove_hooks()
        layers = [(n, m) for n, m in self.model.named_modules()
                  if n.startswith(prefix) and m.__class__.__name__.endswith("DecoderLayer")]
        layers.sort(key=lambda x: int(x[0].rsplit(".", 1)[-1]))

        def mk(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                self._layer_out.append(h.detach())
            return hook
        self._hooks = [m.register_forward_hook(mk(i)) for i, (_n, m) in enumerate(layers)]
        self.n_layers = len(layers)
        return self.n_layers

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
