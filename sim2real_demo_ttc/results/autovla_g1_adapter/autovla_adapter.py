"""AutoVLA × G1 语料 输入适配器（Qwen2.5-VL-3B 系 VLA）。

**环境说明（本轮最曲折的一处，见 amendments.md §CE/A35）**：
AutoVLA 在 `requirements.txt` 里钉死 `transformers==4.49.0`，其发布权重也是按 4.49 的
Qwen2.5-VL 模块布局保存的（`vlm.visual.*`）。本机唯一自带 Qwen2.5-VL 支持的环境是
Alpamayo 的 venv（transformers 4.57.1），而 4.57 把视觉塔挪到了 `vlm.model.visual.*` ⇒
直接加载会 **824 missing / 824 unexpected，等于一个权重都没加载上**。
解决办法：把 `transformers==4.49.0` 装进**隔离的 target 目录**并置于 sys.path 最前，
但**必须先 import torchvision**（用 venv 里那份匹配的 0.23.0+cu128 注册 C++ 算子），
否则 4.49 的 qwen2_5_vl 模块会因 `operator torchvision::nms does not exist` 而导入失败。
按此顺序加载后 **0 missing / 0 unexpected**。

**PDM 依赖的绕开**：`models/autovla.py` 顶层 import 了 `models.utils.score`，
后者 import navsim + nuplan 全家桶，只为 GRPO 训练包装器用的 `PDM_Reward`。
推理只需要 `AutoVLA` 类本身，故在 import 前把该模块替换为**桩模块**，
不改动外部仓库任何一行代码（与 DiffusionDriveV2 的 §CE/A30 同一手法）。

**输入构造**：AutoVLA 原生吃 **3 路相机 × 4 帧时序窗口**（`front / front_left / front_right`）
+ 车速 + 加速度。nuScenes 恰好有 CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT，
时序窗口由 sample_data 的 `prev` 链向前取 3 帧得到（窗口以目标帧结尾）。
**适配偏离（必须随结果一起报）**：AutoVLA 是**多帧模型**，clean/ghost 两条件因此是
两个**部分重叠的时序窗口**之间的对比，而不是两个单帧之间的对比；
且按 §CE/A34，D2cV 对多帧模型**不再是证伪地板**（相对速度对它可观测）。
"""
from __future__ import annotations

import os, sys, types

AUTOVLA_ROOT = "/data/ruolin/uwm/external/AutoVLA"
DEPS = "/data/ruolin/uwm/external/autovla_deps"
ALPA_SP = "/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages"
CKPT = "/data/ruolin/uwm/external/ckpts/AutoVLA_PDMS_89.ckpt"
EVAL_CFG = "config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml"
IMAGE_PAD_ID = 151655          # Qwen2.5-VL <|image_pad|>
CAMS = {"front_camera": "CAM_FRONT", "front_left_camera": "CAM_FRONT_LEFT",
        "front_right_camera": "CAM_FRONT_RIGHT"}
N_TEMPORAL = 4
POOLS = ("vision_mean", "last_token", "seq_mean")


def _bootstrap():
    """严格的导入顺序：venv → torchvision（注册算子）→ 隔离的 transformers 4.49 → 仓库根。"""
    if ALPA_SP not in sys.path:
        sys.path.append(ALPA_SP)
    import torch, torchvision, torchvision.ops   # noqa: F401  必须先于 transformers
    if DEPS not in sys.path:
        sys.path.insert(0, DEPS)
    if AUTOVLA_ROOT not in sys.path:
        sys.path.insert(0, AUTOVLA_ROOT)
    if "models.utils.score" not in sys.modules:
        stub = types.ModuleType("models.utils.score")
        for n in ("PDM_Reward", "TrajectorySampling", "Trajectory"):
            setattr(stub, n, type(n, (object,), {"__init__": lambda s, *a, **k: None}))
        sys.modules["models.utils.score"] = stub
    return torch


torch = _bootstrap()
import numpy as np                                   # noqa: E402


def _longest_run(ids):
    """返回 input_ids 中最长连续同值段的 (token_id, 长度) —— 即图像 pad token 段。"""
    best_id, best_len, i = None, 0, 0
    n = len(ids)
    while i < n:
        j = i
        while j + 1 < n and ids[j + 1] == ids[i]:
            j += 1
        if j - i + 1 > best_len:
            best_id, best_len = int(ids[i]), j - i + 1
        i = j + 1
    return best_id, best_len


class PooledCapture:
    """在钩子里当场池化（VLA 序列很长，不保留全张量）；只留 prefill 那一次。"""

    def __init__(self, model, prefix="vlm.model.layers"):
        layers = [(n, m) for n, m in model.named_modules()
                  if n.startswith(prefix) and m.__class__.__name__.endswith("DecoderLayer")]
        if not layers:      # 4.49 布局下语言塔可能挂在别处，退回全局搜索
            layers = [(n, m) for n, m in model.named_modules()
                      if m.__class__.__name__.endswith("DecoderLayer") and "visual" not in n]
        layers.sort(key=lambda x: int(x[0].rsplit(".", 1)[-1]))
        assert layers, "未找到语言塔 DecoderLayer"
        self.n_layers = len(layers)
        self.mask = None
        self.reset()
        self._h = [m.register_forward_hook(self._mk(i)) for i, (_n, m) in enumerate(layers)]

    def reset(self):
        self.pooled = {p: [None] * self.n_layers for p in POOLS}
        self.seen = [0] * self.n_layers

    def _mk(self, i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            S = h.shape[1]
            if S <= 1 or S <= self.seen[i]:
                return
            self.seen[i] = S
            x = h[0].float()
            self.pooled["last_token"][i] = x[-1].cpu().numpy()
            self.pooled["seq_mean"][i] = x.mean(0).cpu().numpy()
            if self.mask is not None and len(self.mask) == S and self.mask.any():
                m = torch.as_tensor(self.mask, device=x.device)
                self.pooled["vision_mean"][i] = x[m].mean(0).cpu().numpy()
        return hook


class AutoVLARunner:
    def __init__(self, device="cuda:1", ckpt=CKPT,
                 nuscenes_root="/data/dataset/nuscenes/v1.0-trainval"):
        import yaml
        cwd = os.getcwd(); os.chdir(AUTOVLA_ROOT)
        try:
            cfg = yaml.safe_load(open(EVAL_CFG))
            # prompt 含 3 相机 × 4 帧 ⇒ input_ids 已 ~1085 token，官方 eval 的 1024 不够；
            # 放宽到 2048；temperature 取 1e-4（transformers 拒绝 0.0）配 top_k=1，等价于贪心解码
            cfg.setdefault("inference", {})["sample"] = {
                "max_length": 2048, "temperature": 1e-4, "top_k": 1, "top_p": 1.0}
            from models.autovla import AutoVLA
            self.model = AutoVLA(cfg, inference=True, device=device)
            sd = torch.load(ckpt, map_location="cpu")["state_dict"]
            sd = {k.replace("autovla.", "").replace("drivevla.", ""): v for k, v in sd.items()}
            r = self.model.load_state_dict(sd, strict=False)
            assert not r.missing_keys and not r.unexpected_keys, \
                f"权重未对齐：missing {len(r.missing_keys)} / unexpected {len(r.unexpected_keys)}"
        finally:
            os.chdir(cwd)
        self.model = self.model.to(device).eval()
        self.device = device
        self.cap = PooledCapture(self.model)
        from nuscenes.nuscenes import NuScenes
        self.nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_root, verbose=False)
        self.root = nuscenes_root

    def temporal_paths(self, cam_sd_token):
        """由 CAM_FRONT 的 sd_token 取 3 路相机各 4 帧（时序窗口以目标帧结尾）的绝对路径。"""
        sd = self.nusc.get("sample_data", cam_sd_token)
        samples, tok = [], sd["sample_token"]
        for _ in range(N_TEMPORAL):
            s = self.nusc.get("sample", tok)
            samples.append(s)
            tok = s["prev"] if s["prev"] else tok
        samples = samples[::-1]                     # 旧 → 新
        out = {}
        for key, cam in CAMS.items():
            out[key] = [os.path.join(self.root,
                                     self.nusc.get("sample_data", s["data"][cam])["filename"])
                        for s in samples]
        return out

    @torch.no_grad()
    def run(self, cam_sd_token, speed_mps, accel=0.0):
        feats = {"images": self.temporal_paths(cam_sd_token), "sensor_data_path": None,
                 "vehicle_velocity": [float(speed_mps), 0.0],
                 "vehicle_acceleration": [float(accel), 0.0],
                 # 与其余候选一致的"直行"默认指令：DiffusionDrive/LTF/DDV2 用直行 one-hot，
                 # SimLingo 用固定 "Command: follow the road."，此处取语义等价的文本指令。
                 "driving_command": "go straight"}
        inputs = self.model.get_prompt(feats)
        ids = inputs["input_ids"][0].cpu().numpy()
        # 图像 pad token id 随 tokenizer 扩表而变（本仓库 resize 过词表），
        # 故不写死常量，按**最长连续同 id 段**自动识别，并把识别结果落盘供核查。
        tok, ln = _longest_run(ids)
        self.image_token_id = int(tok)
        self.cap.mask = (ids == tok)
        self.cap.reset()
        traj, cot = self.model.predict(feats)   # predict = 官方推理入口（forward 吃的是已 tokenize 的训练 batch）
        traj = np.asarray(traj.float().cpu() if hasattr(traj, "float") else traj, dtype=np.float32)
        pooled = {p: (np.stack(v).astype(np.float32) if all(x is not None for x in v) else None)
                  for p, v in self.cap.pooled.items()}
        return {"trajectory": traj, "pooled": pooled, "cot": cot,
                "commanded_speed": float(np.linalg.norm(np.asarray(traj)[0, :2]) / 0.5),
                "image_token_id": self.image_token_id,
                "n_image_tokens": int(self.cap.mask.sum()), "seq_len": int(len(ids))}
