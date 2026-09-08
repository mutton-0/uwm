"""把删掉的雷达点补回到局部地面上，而不是留一个洞。

## 为什么要补
只删点会在地面点云里留一个**空洞**。空洞本身是可探测的信号 ——
"这里刚才有个挡住地面的东西"。于是 clean 臂并不是干净的反事实：
危险物的形状没了，但它存在过的证据还在。

## 怎么补
在被删点的邻域里取地面点（半径 R 内、z 处于低分位的那些），最小二乘拟合
一个平面 z = ax + by + c，然后把每个被删点的 (x, y) 保留、z 换成平面上的值。
这样**点数和平面密度都不变**，只是把物体表面压回地面。

拟合失败（邻域地面点太少）时退回"只删不补"，并在返回值里标出来。
"""
from __future__ import annotations

import numpy as np


def fit_ground(pts, center, radius=12.0, z_lo=0.35, min_pts=40):
    """在 center 周围 radius 米内拟合地面平面。返回 (a, b, c) 或 None。"""
    if pts is None or len(pts) < min_pts:
        return None
    d = np.linalg.norm(pts[:, :2] - np.asarray(center, float)[:2], axis=1)
    near = pts[d <= radius]
    if len(near) < min_pts:
        return None
    # 地面 = 邻域里 z 最低的那一批（用分位数而非绝对阈值，兼容不同标定原点）
    thr = np.quantile(near[:, 2], z_lo)
    g = near[near[:, 2] <= thr]
    if len(g) < min_pts // 2:
        return None
    A = np.c_[g[:, 0], g[:, 1], np.ones(len(g))]
    try:
        coef, *_ = np.linalg.lstsq(A, g[:, 2], rcond=None)
    except Exception:                                        # noqa: BLE001
        return None
    return coef


def fill_to_ground(pts, mask, boxes_centers, radius=12.0):
    """删掉 mask 命中的点，再把它们压到局部地面上补回去。

    pts: [N,3+] ego 系点云；mask: [N] bool；boxes_centers: 每个框的中心，用于定位邻域。
    返回 (新点云, 补上的点数)。
    """
    if pts is None or not mask.any():
        return pts, 0
    kept = pts[~mask]
    removed = pts[mask]
    if not boxes_centers:
        return kept, 0
    filled = []
    for cc in boxes_centers:
        d = np.linalg.norm(removed[:, :2] - np.asarray(cc, float)[:2], axis=1)
        sel = removed[d <= radius]
        if len(sel) == 0:
            continue
        coef = fit_ground(kept, cc, radius)
        if coef is None:
            continue
        z = coef[0] * sel[:, 0] + coef[1] * sel[:, 1] + coef[2]
        new = sel.copy(); new[:, 2] = z
        filled.append(new)
    if not filled:
        return kept, 0
    F = np.concatenate(filled, 0)
    return np.concatenate([kept, F], 0), int(len(F))
