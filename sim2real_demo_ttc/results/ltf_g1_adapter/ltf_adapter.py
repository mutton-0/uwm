"""LTF（Latent TransFuser）× G1 语料 输入适配器 —— 复用 DiffusionDrive 适配器的全部前端。

**为什么几乎零成本**：本仓库的 `transfuser_agent.yaml` 设 `latent: True`，
即 Latent TransFuser；而 DiffusionDrive 的骨干正是同一个 `TransfuserBackbone(latent=True)`。
两者**共用同一个编码器**，只在**动作头**上不同：
    LTF            → `TrajectoryHead`：单个 trajectory query 过 MLP 直接回归 8×3 位姿（**连续回归头**）
    DiffusionDrive → anchored 扩散头（20 个 k-means 轨迹锚）
因此本适配器只替换 agent 构造，图像/状态/token 口径与 `diffusiondrive_g1_adapter/dd_adapter.py`
**逐像素、逐字段同一套**（直接 import 复用，不复制代码）。

**这构成一次受控实验**：同一编码器 + 不同动作头，可以直接检验
「F② 注入法的可测性是动作头的属性、不是编码器的属性」这一命题（见 axis_ltf_report_*）。
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

DEVKIT = "/data/ruolin/uwm"
sys.path.insert(0, os.path.join(DEVKIT, "sim2real_demo_ttc", "results", "diffusiondrive_g1_adapter"))
import dd_adapter as DD                     # 复用图像/状态/token/池化的全部前端

N_IMG_TOK = DD.N_IMG_TOK
image_to_camera_feature = DD.image_to_camera_feature
bbox_to_tokens = DD.bbox_to_tokens
status_feature = DD.status_feature

LTF_CKPT = "/data/ruolin/uwm/ckpt/ltf_sim_navtest.ckpt"
ANCHOR = f"{DEVKIT}/traj_final/kmeans_navsim_traj_20.npy"
PRED_DT = 0.5


def load_ltf_agent(ckpt: str = LTF_CKPT):
    """与 run_ghosthead_infer.load_agent 同构，只换 agent 类与 config。"""
    from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
    from navsim.agents.transfuser.transfuser_config import TransfuserConfig
    cfg = TransfuserConfig(bkb_path="unused", latent=True)
    agent = TransfuserAgent(config=cfg, lr=1e-4, checkpoint_path=ckpt)
    agent.initialize()
    agent.eval()
    if torch.cuda.is_available():
        agent = agent.cuda()
    return agent


class LTFRunner(DD.DDRunner):
    """继承 DDRunner，只改模型加载；钩子/池化/注入/行为量口径全部继承，保证跨模型同口径。"""

    def __init__(self, device="cuda:0", ckpt: str = LTF_CKPT):
        self.agent = load_ltf_agent(ckpt)
        self.device = next(self.agent.parameters()).device
        from navsim.agents.transfuser.transfuser_backbone import SelfAttention
        self.sas = [m for m in self.agent._transfuser_model._backbone.modules()
                    if isinstance(m, SelfAttention)]
        assert len(self.sas) == 8, f"期望 8 个 SelfAttention，实得 {len(self.sas)}"
        self._buf = [None] * len(self.sas)
        self._steer = None
        self._last_sigma = float("nan")
        for i, m in enumerate(self.sas):
            m.register_forward_hook(self._mk(i))

    @torch.no_grad()
    def run(self, img_rgb, speed_mps, region_tokens=None, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cam = image_to_camera_feature(img_rgb).to(self.device)
        st = status_feature(speed_mps).to(self.device)
        out = self.agent.forward({"camera_feature": cam, "status_feature": st})
        traj = out["trajectory"][0].cpu().numpy()          # (8, 3)
        pooled = {p: [] for p in self.POOLS}
        reg = sorted(set(region_tokens or []))
        for h in self._buf:
            img = h[:N_IMG_TOK]; lid = h[N_IMG_TOK:]
            bg = [i for i in range(N_IMG_TOK) if i not in set(reg)]
            pooled["vision_mean"].append(img.mean(0).cpu().numpy())
            pooled["lidar_mean"].append(lid.mean(0).cpu().numpy())
            pooled["all_mean"].append(h.mean(0).cpu().numpy())
            pooled["region_mean"].append((img[reg].mean(0) if reg else img.mean(0)).cpu().numpy())
            pooled["bg_mean"].append(img[bg].mean(0).cpu().numpy())
        pooled = {p: np.array([x.astype(np.float32) for x in v], dtype=object) for p, v in pooled.items()}
        return {"trajectory": traj, "pooled": pooled,
                "commanded_speed": float(np.linalg.norm(traj[0, :2]) / PRED_DT),
                "has_region": bool(reg), "n_region_tokens": len(reg)}
