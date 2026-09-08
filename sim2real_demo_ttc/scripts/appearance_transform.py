"""构造"同场景不同光照"的配对刺激：模拟夜/雨，两种作用范围。

两个变体（用户 2026-09-07）：
  scope="sky"    只改天空区域 —— 天空是全图最大、最均匀、语义上与行车决策
                 最无关的一块。若模型对"只改天空"都有行为反应，那它对光照的
                 依赖是全局性的，不是靠某个局部线索。
  scope="global" 全图改 —— 更接近真实的夜/雨。

不用生成模型：生成模型自己的伪影会成为新的域移，与 CARLA 那条的病一样。
这里只做**可解析、可复现、参数写死**的光度变换。

天空分割用启发式而非分割网络：从顶部往下的连通区域，要求亮度高于全图中位、
且局部方差低（天空少纹理）。故意保守 —— 宁可漏掉一部分天空，也不要把建筑
或路面误判成天空，那会让"只改天空"这个对照失去意义。
"""
from __future__ import annotations

import numpy as np

# 夜：压亮度 + 降饱和 + 偏蓝 + 加噪；雨：降对比 + 灰蓝 + 轻微模糊 + 雨条
NIGHT = dict(gamma=2.2, sat=0.45, tint=(0.85, 0.92, 1.15), noise=7.0, gain=0.45)
RAIN = dict(gamma=1.15, sat=0.55, tint=(0.94, 0.97, 1.06), noise=3.0, gain=0.80,
            contrast=0.72, blur=1.2, streaks=True)


def sky_mask(img, top_frac=0.55, var_win=9):
    """启发式天空掩膜：顶部区域内，亮度高于全图中位且局部方差低的连通部分。"""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    m = np.zeros((H, W), np.uint8)
    band = int(H * top_frac)
    gb = g[:band].astype(np.float32)
    med = float(np.median(g))
    mu = cv2.blur(gb, (var_win, var_win))
    var = cv2.blur(gb * gb, (var_win, var_win)) - mu * mu
    cand = ((gb > med) & (var < 120.0)).astype(np.uint8)
    # 只保留与顶边相连的部分：天空必然连到画面上沿
    ff = np.zeros((band + 2, W + 2), np.uint8)
    seed = cand.copy()
    lab_n, lab = cv2.connectedComponents(seed)
    keep = set(np.unique(lab[0, :])) - {0}
    m[:band] = np.isin(lab, list(keep)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m.astype(bool)


def _photometric(a, p, rng):
    """a: float32 [H,W,3] in 0-255。就地返回变换后的副本。"""
    x = a / 255.0
    x = np.power(np.clip(x, 0, 1), p["gamma"]) * p["gain"]
    lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
    x = lum[..., None] + (x - lum[..., None]) * p["sat"]
    x = x * np.asarray(p["tint"], np.float32)[None, None, :]
    if "contrast" in p:
        x = (x - x.mean()) * p["contrast"] + x.mean()
    x = x * 255.0
    if p.get("noise", 0):
        x = x + rng.normal(0, p["noise"], x.shape)
    return x


def transform(img, kind="night", scope="global", seed=0, mask=None,
              noise_override=None):
    """img: RGB uint8 [H,W,3]。返回同尺寸 uint8。

    scope="sky" 时只在天空掩膜内生效，掩膜边缘做羽化，避免引入硬边
    —— 硬边本身是高频结构，会变成模型可以抓住的伪线索。
    """
    import cv2
    p = dict({"night": NIGHT, "rain": RAIN}[kind])
    if noise_override is not None:
        # 见 P-7：注入的高频噪声本身就是个强扰动，会盖过"光照变了"这个效应。
        p["noise"] = float(noise_override)
    rng = np.random.default_rng(seed)
    a = img.astype(np.float32)
    out = _photometric(a, p, rng)
    if p.get("blur"):
        out = cv2.GaussianBlur(out, (0, 0), p["blur"])
    if p.get("streaks"):
        n = rng.integers(120, 260)
        H, W = a.shape[:2]
        for _ in range(int(n)):
            x0 = int(rng.integers(0, W)); y0 = int(rng.integers(0, H))
            L = int(rng.integers(8, 26)); dx = int(rng.integers(-3, 4))
            cv2.line(out, (x0, y0), (x0 + dx, y0 + L), (215, 220, 230), 1)
    if scope == "sky":
        m = sky_mask(img) if mask is None else mask
        w = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6.0)[..., None]
        out = a * (1 - w) + out * w
    return np.clip(out, 0, 255).astype(np.uint8)
