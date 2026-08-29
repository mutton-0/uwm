"""DiffusionDrive × G1 语料 输入适配器 + 可读表征抽取。

工单依据:docs/g_axis_positive_calibration_diffusiondrive.md §3(工程依赖)。

**为什么需要它**:G 轴正向校准要求 DiffusionDrive 跑在与 SimLingo **完全相同**的刺激集上
(nuScenes G1 鬼探头语料 + N1 的 D2a/D2b/D2c/D2cV 负例),而不是它自己的 NAVSIM 评测集。
本文件把 nuScenes CAM_FRONT 单帧转成 DiffusionDrive 训练时的输入格式,并抓取可读表征。

口径对齐(协议 §4 跨模型可比性纪律):
  * 图像:与 `run_ghosthead_infer.py` **逐像素同一套** —— 居中裁 4:1 去天空 -> resize(2048,512)。
  * ego 状态:driving_command 固定直行 one-hot(与 ghosthead 同,nuScenes 无路由);
    速度/加速度用 **clean 帧锚定**(两条件共用同一份),使 clean/ghost 唯一差异是图像 ——
    与 SimLingo 侧 `prompt_anchor: clean` 的处理**同构**(demo commit 5c1366d 的同一个混淆)。
  * 可读层:TransFuser 编码器的 8 个 `SelfAttention` 输出,每层 320 token
    = 256 图像 token(8 行 × 32 列,行主序)+ 64 BEV/lidar-latent token。
    按协议 §4 规则 3「层按角色对齐、不按层号」,这 8 层即该模型的可读层集合。
  * 池化口径:
      vision_mean = 256 个图像 token 均值    <- SimLingo `vision_mean` 的同构物
      region_mean = bbox 内图像 token 均值   <- SimLingo `region_mean` 的同构物
      bg_mean     = bbox 外图像 token 均值
      lidar_mean  = 64 个 BEV latent token 均值
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

DEVKIT = "/data/ruolin/uwm"
sys.path.insert(0, os.path.join(DEVKIT, "scripts", "ghosthead_infer"))
import run_ghosthead_infer as G   # noqa: E402  (module import 触发 backbone monkeypatch)

IMG_VERT, IMG_HORZ = 8, 32        # transfuser_config: 256//32, 1024//32
N_IMG_TOK = IMG_VERT * IMG_HORZ   # 256
RESIZE_W, RESIZE_H = 2048, 512
DRIVING_COMMAND = G.DRIVING_COMMAND


# ------------------------------------------------------------------ 图像
def image_to_camera_feature(img_rgb):
    """任意 RGB 图 -> [1,3,512,2048]。与 ghosthead 前端逐像素同一套。"""
    return G.build_camera_feature(G.crop_4to1_no_sky(img_rgb))


def crop_geometry(im_wh):
    """返回 (top, target_h) —— crop_4to1_no_sky 在原图上截取的行范围。"""
    W, H = im_wh
    th = int(round(W / 4.0))
    top = max(0, min(G.CROP_CENTER_ROW - th // 2, H - th))
    return top, th


def bbox_to_tokens(bbox_xyxy, im_wh, dilate=1):
    """原图像素框 -> 图像 token 索引集合(0..255)。外扩 dilate 个 token 容错。"""
    if not bbox_xyxy:
        return []
    W, H = im_wh
    top, th = crop_geometry(im_wh)
    x0, y0, x1, y1 = bbox_xyxy
    y0, y1 = y0 - top, y1 - top
    if y1 <= 0 or y0 >= th:
        return []
    c0 = int(np.floor(x0 / W * IMG_HORZ)); c1 = int(np.ceil(x1 / W * IMG_HORZ))
    r0 = int(np.floor(max(y0, 0) / th * IMG_VERT)); r1 = int(np.ceil(min(y1, th) / th * IMG_VERT))
    c0 = max(0, c0 - dilate); r0 = max(0, r0 - dilate)
    c1 = min(IMG_HORZ, c1 + dilate); r1 = min(IMG_VERT, r1 + dilate)
    return [r * IMG_HORZ + c for r in range(r0, max(r0 + 1, r1)) for c in range(c0, max(c0 + 1, c1))]


def status_feature(speed_mps, accel_mps2=0.0):
    """[1,8] = driving_command(4) + v(2) + a(2)。nuScenes 只有纵向速率,横向置 0。"""
    v = np.array([float(speed_mps), 0.0], dtype=np.float32)
    a = np.array([float(accel_mps2), 0.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([DRIVING_COMMAND, v, a])).float().unsqueeze(0)


# ------------------------------------------------------------------ 模型 + 钩子
class DDRunner:
    """DiffusionDrive 推理封装:一次 forward 同时拿到 8 层表征与规划轨迹。"""

    POOLS = ("region_mean", "vision_mean", "bg_mean", "lidar_mean", "all_mean")

    def __init__(self, device="cuda:0"):
        self.agent = G.load_agent()
        self.device = next(self.agent.parameters()).device
        from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention
        self.sas = [m for m in self.agent._transfuser_model._backbone.modules()
                    if isinstance(m, SelfAttention)]
        assert len(self.sas) == 8, f"期望 8 个 SelfAttention，实得 {len(self.sas)}"
        self._buf = [None] * len(self.sas)
        self._steer = None
        self._last_sigma = float("nan")
        for i, m in enumerate(self.sas):
            m.register_forward_hook(self._mk(i))

    def _mk(self, i):
        def _h(mod, inp, out):
            st = self._steer
            if st is not None and st["layer"] == i:
                out = self._apply_steer(out, st)
            self._buf[i] = out.detach().float()[0]      # [320, C_l]
            return out
        return _h

    # ---------------- RepE 式操纵（与 SimLingo 侧 simlingo_runner.set_steering 同构） ----------------
    def set_steering(self, layer=None, vec=None, alpha=0.0, mode="add", tokens="image"):
        """在第 layer 个 encoder SelfAttention 的输出上操纵；layer=None 关闭。

        mode: add / project_out / recover —— 与 SimLingo 侧逐条同义。
        sigma = 本次前向中被注入 token 的激活标准差（逐帧自归一化，使 α 在不同层间可比）。
        tokens: image(前 256 个图像 token) / lidar(后 64 个 BEV latent) / all(全部 320)。
        默认 image：方向是从图像 token 池化的 δ 上提的，注回同一批 token 才同构。
        """
        if layer is None or vec is None:
            self._steer = None
            return
        v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        v = v / (v.norm() + 1e-8)
        self._steer = {"layer": int(layer), "v": v.to(self.device), "alpha": float(alpha),
                       "mode": mode, "tokens": tokens}

    def _apply_steer(self, out, st):
        n = out.shape[1]
        idx = {"image": slice(0, N_IMG_TOK), "lidar": slice(N_IMG_TOK, n),
               "all": slice(0, n)}[st["tokens"]]
        sub = out[0, idx, :].float()
        v = st["v"].to(sub.dtype)
        if st["mode"] == "add":
            sigma = float(sub.std()); self._last_sigma = sigma
            sub = sub + (st["alpha"] * sigma) * v
        elif st["mode"] == "project_out":
            sub = sub - torch.outer(sub @ v, v)
        elif st["mode"] == "recover":
            c = sub @ v
            sub = sub - torch.outer(c, v) + c.mean() * v
        else:
            raise ValueError(st["mode"])
        out = out.clone()
        out[0, idx, :] = sub.to(out.dtype)
        return out

    @staticmethod
    def lateral_offset(traj):
        """规划轨迹的横向偏移量（米）——特异性检查用，与 SimLingo 侧 lateral_offset 同义。"""
        return float(np.abs(traj[:, 1]).max())

    @staticmethod
    def comfort(traj):
        """纵向加加速度代理：二阶差分的均方根，越大越不舒适。"""
        d = np.diff(traj[:, 0], n=2) if len(traj) > 2 else np.zeros(1)
        return float(np.sqrt((d ** 2).mean()))

    @torch.no_grad()
    def run(self, img_rgb, speed_mps, region_tokens=None, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cam = image_to_camera_feature(img_rgb).to(self.device)
        st = status_feature(speed_mps).to(self.device)
        out = self.agent.forward({"camera_feature": cam, "status_feature": st})
        traj = out["trajectory"][0].cpu().numpy()        # [8,3] (x,y,heading)
        pooled = {p: [] for p in self.POOLS}
        reg = sorted(set(region_tokens or []))
        for h in self._buf:
            img = h[:N_IMG_TOK]; lid = h[N_IMG_TOK:]
            bg_idx = [i for i in range(N_IMG_TOK) if i not in set(reg)]
            pooled["vision_mean"].append(img.mean(0).cpu().numpy())
            pooled["lidar_mean"].append(lid.mean(0).cpu().numpy())
            pooled["all_mean"].append(h.mean(0).cpu().numpy())
            pooled["region_mean"].append((img[reg].mean(0) if reg else img.mean(0)).cpu().numpy())
            pooled["bg_mean"].append(img[bg_idx].mean(0).cpu().numpy())
        # 各层通道数不同(4 个尺度) -> 用 object 数组按层存
        pooled = {p: np.array([x.astype(np.float32) for x in v], dtype=object) for p, v in pooled.items()}
        return {"trajectory": traj, "pooled": pooled,
                "commanded_speed": float(np.linalg.norm(traj[0, :2]) / G.PRED_DT),
                "has_region": bool(reg), "n_region_tokens": len(reg)}
