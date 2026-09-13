"""导出每个体检单位对应的**真人驾驶轨迹**（0.5–2.5 s，当前自车坐标系），用于给危险敏感度提供真值参照。
做法：manifest 里每条记录给出 nuScenes 的 sample token；沿 sample.next 取后续关键帧，
读各帧 LIDAR_TOP 的 ego_pose，把全局位姿换算到当前帧自车系。
输出 human_path.json：{uid: [[x,y] × ≤5]}（x 向前，y 向左，与规划同一坐标系）。"""
import json,numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
R5="/home/boyuewang/120/uwm/sim2real_demo_ttc/results_5090"; V5=f"{R5}/paper_icra_v5"
NUSC=NuScenes(version="v1.0-trainval",dataroot="/data/dataset/nuscenes/v1.0-trainval",verbose=False)
MAN=json.load(open(f"{R5}/risk_card_manifest.json"))
def pose(sample_tok):
    s=NUSC.get("sample",sample_tok); sd=NUSC.get("sample_data",s["data"]["LIDAR_TOP"])
    e=NUSC.get("ego_pose",sd["ego_pose_token"])
    return np.array(e["translation"][:2]),Quaternion(e["rotation"])
out={}
for k in ("A","B"):
    for x in MAN[k]:
        try: t0,q0=pose(x["sample"])
        except Exception: continue
        R=q0.rotation_matrix[:2,:2]; pts=[]; s=NUSC.get("sample",x["sample"])
        for _ in range(5):
            if not s["next"]: break
            s=NUSC.get("sample",s["next"])
            try: t,_=pose(s["token"])
            except Exception: break
            pts.append((R.T@(t-t0)).tolist())      # 换到当前帧自车系
        if pts: out[x["uid"]]=pts
json.dump(out,open(f"{V5}/human_path.json","w"))
n=[len(v) for v in out.values()]
print(f"导出 {len(out)} 个单位；轨迹点数中位 {int(np.median(n))}；2.5 s 完整的占 {100*np.mean([v==5 for v in n]):.0f}%")
