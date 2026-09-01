"""DiffusionDriveV2 × G1 语料 输入适配器。

**与前两个 navsim 系候选的关键差异（必须随结果一起报）**：
DiffusionDriveV2 的两份官方配置（`diffusiondrivev2_{sel,rl}_agent.yaml`）都设 **`latent: False`**，
其发布权重里含 **232 个 `lidar_encoder.*` 张量、0 个 latent 张量** ——
即它**要求真实 lidar BEV 输入**，不像 DiffusionDrive/LTF 那样用可学习 latent 顶替 lidar 分支。

处理方式（见 amendments.md §CE/A29）：按"每个候选走自己的原生输入格式"这一适配器哲学，
从 nuScenes **LIDAR_TOP** 现场构造 TransFuser 式 BEV 直方图，复刻其
`_get_lidar_feature` 的每一步（256×256 网格、x/y ∈ [−32, 32]、4 px/m、
z > 0.2 m 的点、每像素上限 5 后归一化、`use_ground_plane=False` 故单通道）。
**适配偏离**：nuScenes 是 32 线单雷达，NAVSIM/nuPlan 是多雷达合并点云，点密度不同 ⇒
该直方图对 DiffusionDriveV2 属分布外输入。此偏离与 Alpamayo 的 FOV 失配同性质，随结果并列声明。
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

V2_ROOT = "/data/ruolin/uwm/external/DiffusionDriveV2"
DEVKIT = "/data/ruolin/uwm"
CKPT = "/data/ruolin/uwm/external/ckpts/diffusiondrivev2_sel.ckpt"
PRED_DT = 0.5

# V2 的 navsim 必须排在主仓库之前
if V2_ROOT not in sys.path:
    sys.path.insert(0, V2_ROOT)
sys.path.insert(0, os.path.join(DEVKIT, "sim2real_demo_ttc", "results", "diffusiondrive_g1_adapter"))
import dd_adapter as DD                      # 只借用图像/状态/token 前端，不借用模型

N_IMG_TOK = DD.N_IMG_TOK
image_to_camera_feature = DD.image_to_camera_feature
bbox_to_tokens = DD.bbox_to_tokens
status_feature = DD.status_feature

# ---- TransFuser lidar 直方图常量（逐条抄自 diffusiondrivev2_sel_config.py）----
LID_MIN, LID_MAX, PPM = -32.0, 32.0, 4.0
MAX_H, SPLIT_H, HIST_MAX = 100.0, 0.2, 5


def lidar_histogram(points_xyz: np.ndarray) -> torch.Tensor:
    """ego 系点云 (N,3) -> [1, 256, 256] BEV 直方图，复刻 V2 的 _get_lidar_feature。"""
    xb = np.linspace(LID_MIN, LID_MAX, int((LID_MAX - LID_MIN) * PPM) + 1)
    yb = np.linspace(LID_MIN, LID_MAX, int((LID_MAX - LID_MIN) * PPM) + 1)
    pc = points_xyz[points_xyz[:, 2] < MAX_H]
    above = pc[pc[:, 2] > SPLIT_H]
    hist = np.histogramdd(above[:, :2], bins=(xb, yb))[0]
    hist[hist > HIST_MAX] = HIST_MAX
    feat = (hist / HIST_MAX).astype(np.float32)[None]        # use_ground_plane=False -> 单通道
    return torch.from_numpy(feat).unsqueeze(0)               # [1,1,256,256]


class _TrajReady(Exception):
    """载体异常：把 forward_test_rl 已经算好的候选轨迹带出来。"""

    def __init__(self, traj):
        super().__init__("traj ready")
        self.traj = traj


def _patch_out_pdm_scoring():
    """绕开发布代码里对 nuPlan PDM metric cache 的硬依赖（见 amendments.md §CE/A30）。

    DiffusionDriveV2 `sel` 变体的推理路径 `TrajectoryHead.forward_test_rl` 在算完
    coarse + fine 精化、选出 `traj_to_score` 之后，**无条件**调用 `get_pdm_score_para(...)`
    去用官方 PDM 打分器给候选轨迹排序；该打分器需要逐 token 的 nuPlan metric cache
    （地图、agent 轨迹），而 nuScenes 语料没有、也不该有。
    作者自己在那一行正上方留了注释掉的官方评测出口：
        # for official eval
        # return {"trajectory": traj_to_score[:,-1]}
    我们**不改动作者代码**，而是把 `get_pdm_score_para` 换成抛出载体异常，
    在适配器里接住并取 `traj_to_score[:, -1]` —— 与那行注释掉的官方出口**逐字等价**，
    且发生在所有网络计算完成之后，不改变任何前向逻辑。
    """
    from navsim.agents.diffusiondrivev2 import diffusiondrivev2_model_sel as M
    head = None
    for name in dir(M):
        obj = getattr(M, name)
        if isinstance(obj, type) and hasattr(obj, "get_pdm_score_para"):
            head = obj
            break
    assert head is not None, "未找到带 get_pdm_score_para 的类"

    def _raise(self, trajectory, metric_cache_path):
        raise _TrajReady(trajectory)

    head.get_pdm_score_para = _raise
    return head.__name__


def load_ddv2_agent(ckpt: str = CKPT):
    # V2 的 TrajectoryHead 用**相对路径**读 kmeans_navsim_traj_20.npy，
    # 只在构造期临时切到仓库根目录，构造完立刻切回（否则后续图像相对路径会错）。
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_agent import Diffusiondrivev2_Sel_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_sel_config import TransfuserConfig
    cfg = TransfuserConfig()
    _cwd = os.getcwd()
    os.chdir(V2_ROOT)
    try:
        return _build(Diffusiondrivev2_Sel_Agent, cfg, ckpt)
    finally:
        os.chdir(_cwd)


def _build(cls, cfg, ckpt):
    agent = cls(config=cfg, lr=2e-4, checkpoint_path=ckpt)
    agent.initialize()
    agent.eval()
    if torch.cuda.is_available():
        agent = agent.cuda()
    return agent


class DDV2Runner(DD.DDRunner):
    """钩子/池化/注入/patching 全部继承 DDRunner，只改模型加载与 forward 的输入构造。"""

    def __init__(self, device="cuda:0", ckpt: str = CKPT):
        self.agent = load_ddv2_agent(ckpt)
        self.patched_class = _patch_out_pdm_scoring()
        self.device = next(self.agent.parameters()).device
        from navsim.agents.diffusiondrivev2.transfuser_backbone import SelfAttention
        self.sas = [m for m in self.agent._transfuser_model._backbone.modules()
                    if isinstance(m, SelfAttention)]
        assert self.sas, "未找到 SelfAttention"
        self._buf = [None] * len(self.sas)
        self._steer = None
        self._patch = None
        self._last_sigma = float("nan")
        for i, m in enumerate(self.sas):
            m.register_forward_hook(self._mk(i))

    @torch.no_grad()
    def run(self, img_rgb, speed_mps, region_tokens=None, seed=0, lidar_xyz=None):
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cam = image_to_camera_feature(img_rgb).to(self.device)
        st = status_feature(speed_mps).to(self.device)
        lid = lidar_histogram(np.zeros((0, 3), np.float32) if lidar_xyz is None else lidar_xyz).to(self.device)
        feats = {"camera_feature": cam, "status_feature": st, "lidar_feature": lid}
        try:
            out = self.agent._transfuser_model(feats, cal_pdm=False)
            traj = out["trajectory"][0].float().cpu().numpy()
        except _TrajReady as e:                      # 官方评测出口：取最后一次 fine 精化的轨迹
            traj = e.traj[0, -1].float().cpu().numpy()
        pooled = {p: [] for p in self.POOLS}
        reg = sorted(set(region_tokens or []))
        for h in self._buf:
            img = h[:N_IMG_TOK]; lidt = h[N_IMG_TOK:]
            bg = [i for i in range(N_IMG_TOK) if i not in set(reg)]
            pooled["vision_mean"].append(img.mean(0).cpu().numpy())
            pooled["lidar_mean"].append(lidt.mean(0).cpu().numpy() if lidt.shape[0] else img.mean(0).cpu().numpy())
            pooled["all_mean"].append(h.mean(0).cpu().numpy())
            pooled["region_mean"].append((img[reg].mean(0) if reg else img.mean(0)).cpu().numpy())
            pooled["bg_mean"].append(img[bg].mean(0).cpu().numpy())
        pooled = {p: np.array([x.astype(np.float32) for x in v], dtype=object) for p, v in pooled.items()}
        return {"trajectory": traj, "pooled": pooled,
                "commanded_speed": float(np.linalg.norm(traj[0, :2]) / PRED_DT),
                "has_region": bool(reg), "n_region_tokens": len(reg)}


class NavsimLidar:
    """NAVSIM/OpenScene 的点云读取器：读 MergedPointCloud 的 .pcd，返回 **ego 系** 点。

    比 nuScenes 侧简单一档，原因是数据源本身的便利（成本反转的一个具体例子）：
    NAVSIM 的 `lidar2ego` 是恒等变换（translation [0,0,0]、rotation [1,0,0,0]，已实测核对），
    故点云**天然就在 ego 系**，不需要 sensor→ego 的旋转平移。
    nuScenes 侧则必须先取 LIDAR_TOP 的 calibrated_sensor 再做一次变换。

    键用 CAM_F0 的 data_path（与其余适配器的 `sd_token` 位置同构），
    由 `event_id -> lidar_path` 的映射表在构造时一次性建好。
    """

    def __init__(self, blob_root="/data/dataset/navsim/dataset/sensor_blobs",
                 log_root="/data/dataset/navsim/dataset/navsim_logs", split="test"):
        import pathlib, pickle, glob
        self.blob = pathlib.Path(blob_root) / split
        self.map = {}                       # CAM_F0 data_path -> lidar_path
        for f in sorted(glob.glob(str(pathlib.Path(log_root) / split / "*.pkl"))):
            for fr in pickle.load(open(f, "rb")):
                self.map[fr["cams"]["CAM_F0"]["data_path"]] = fr["lidar_path"]
        self._cache = {}

    @staticmethod
    def _read_pcd(path):
        """二进制 PCD（FIELDS x y z intensity lidar_info ring，SIZE 4 4 4 1 1 1）。"""
        with open(path, "rb") as fh:
            raw = fh.read()
        i = raw.find(b"DATA binary\n")
        assert i >= 0, f"非二进制 PCD: {path}"
        head = raw[:i].decode("ascii", "ignore")
        n = int([l for l in head.splitlines() if l.startswith("POINTS")][0].split()[1])
        body = raw[i + len(b"DATA binary\n"):]
        rec = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("intensity", "u1"), ("lidar_info", "u1"), ("ring", "u1")])
        a = np.frombuffer(body[: n * rec.itemsize], dtype=rec, count=n)
        return np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float32)

    def ego_points(self, cam_path):
        if cam_path in self._cache:
            return self._cache[cam_path]
        lp = self.map.get(cam_path)
        if lp is None:
            return np.zeros((0, 3), np.float32)
        pts = self._read_pcd(self.blob / lp)      # 已在 ego 系，无需变换
        if len(self._cache) < 64:
            self._cache[cam_path] = pts
        return pts


class NuScenesLidar:
    """按 CAM_FRONT 的 sample_data token 取同 sample 的 LIDAR_TOP 点云，变换到 ego 系。

    DiffusionDriveV2 的发布权重要求真实 lidar（见头注）；任何**不喂 lidar** 的运行
    （全零直方图）都是把模型置于极端分布外，其读数不可与其余候选并列。
    因此凡是对 DDV2 做前向的脚本（缓存 / 注入 / patching）都必须走这个读取器。
    """

    def __init__(self, root="/data/dataset/nuscenes/v1.0-trainval", version="v1.0-trainval"):
        from nuscenes.nuscenes import NuScenes
        from pyquaternion import Quaternion
        import pathlib
        self.Q = Quaternion
        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)
        self.root = pathlib.Path(root)
        self._cache = {}          # 同一帧在注入实验里会被反复用到（169 个条件），必须缓存

    def ego_points(self, cam_sd_token):
        if cam_sd_token in self._cache:
            return self._cache[cam_sd_token]
        pts = self._ego_points(cam_sd_token)
        if len(self._cache) < 64:
            self._cache[cam_sd_token] = pts
        return pts

    def _ego_points(self, cam_sd_token):
        sd = self.nusc.get("sample_data", cam_sd_token)
        samp = self.nusc.get("sample", sd["sample_token"])
        lsd = self.nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
        pts = np.fromfile(self.root / lsd["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
        cs = self.nusc.get("calibrated_sensor", lsd["calibrated_sensor_token"])
        Rm = self.Q(cs["rotation"]).rotation_matrix
        return (pts @ Rm.T) + np.array(cs["translation"], np.float32)
