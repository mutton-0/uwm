"""把 3D 框内的激光点删掉 —— ddv2 是相机+激光双模态，只改图像等于制造模态冲突。

之前所有 ddv2 的移除结果都只改了相机：图像里人没了，点云里人还在。
模型看到"相机说没人、雷达说有人"，保守缩短轨迹是合理反应，不能解读成反因果。

3D 框是可信的（human 尺寸 0.76×0.74×1.71 m，就是人形；出问题的是 2D 投影关联）。
从 8 个角点还原有向包围盒：中心 + 三条正交轴 + 半长，然后在盒坐标系里做点判定。
margin 默认 0.15 m —— 稍微放宽，吃掉标注误差和边缘点，但别大到吃进地面。
删点之后那一格在 lidar_histogram 里变空，正是"这里没有障碍物"该有的样子。
"""
import numpy as np
def obb_from_corners(cor):
    """cor: (8,3)。返回 (center, R(3,3) 列为轴, half(3,))。
    nuScenes 角点顺序不保证，所以用 PCA 取轴 —— 对长方体，PCA 主轴就是盒轴。"""
    P=np.asarray(cor,float)
    c=P.mean(0)
    Q=P-c
    # 协方差的特征向量 = 盒的三条轴（长方体 8 顶点均匀分布时严格成立）
    w,V=np.linalg.eigh(Q.T@Q)
    proj=Q@V
    half=np.abs(proj).max(0)
    return c,V,half
def remove_points(xyz, corners_list, margin=0.15):
    """删掉落在任一 3D 框（外扩 margin）内的点。返回 (新点云, 删掉的点数)。"""
    if xyz is None or len(xyz)==0 or not corners_list: return xyz,0
    P=np.asarray(xyz,float)
    keep=np.ones(len(P),bool)
    for cor in corners_list:
        cor=np.asarray(cor,float)
        if cor.shape[0]<8: continue
        c,V,half=obb_from_corners(cor)
        d=np.abs((P[:,:3]-c)@V)
        keep &= ~np.all(d<=(half+margin),1)
    return P[keep], int((~keep).sum())

def scale_points(xyz, corners_list, k, margin=0.15):
    """把 3D 框内的点以**脚部接地点**为锚等比放大 k 倍 —— 图像里放大行人时点云要同步，
    否则又是"相机看到大人、雷达看到小人"的模态冲突（和只改相机的移除是同一个坑）。

    锚点取框底面中心：z 方向从底面往上放大，xy 从中心往外放大，
    与图像侧"以脚点为锚等比放大"几何一致。
    返回 (新点云, 被放大的点数)。
    """
    if xyz is None or len(xyz)==0 or not corners_list: return xyz,0
    P=np.asarray(xyz,float).copy()
    n=0
    for cor in corners_list:
        cor=np.asarray(cor,float)
        if cor.shape[0]<8: continue
        c,V,half=obb_from_corners(cor)
        d=(P[:,:3]-c)@V
        inb=np.all(np.abs(d)<=(half+margin),1)
        if not inb.any(): continue
        # 竖直轴 = 与全局 z 最对齐的那条盒轴；锚在该轴的底部
        iz=int(np.argmax(np.abs(V[2])))
        sgn=np.sign(V[2,iz]) or 1.0
        dd=d[inb].copy()
        dd[:,iz]=(dd[:,iz]+sgn*half[iz])*k-sgn*half[iz]     # 底面不动，往上拉伸
        for j in range(3):
            if j!=iz: dd[:,j]=dd[:,j]*k                      # 横向从中心往外
        P[inb,:3]=c+dd@V.T
        n+=int(inb.sum())
    return P,n
