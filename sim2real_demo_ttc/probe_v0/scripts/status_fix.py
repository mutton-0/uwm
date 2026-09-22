"""把 dd_adapter.status_feature 修正为官方口径。import 本模块即生效（不改仓库文件）。

官方口径 navsim/agents/diffusiondrive/transfuser_features.py:45
    concat[ driving_command(4), ego_velocity(2), ego_acceleration(2) ]
driving_command 顺序 = [左, 直, 右, unknown]
    依据 ① docs/agents.md:79 原文「towards the left, straight or right direction …
           a fourth command, representing 'unknown'」
         ② 实测 NAVSIM 未来3秒航向变化：索引0 +25.9° / 索引1 -0.0° / 索引2 -37.2°
注意 ego_status_mlp_agent.py:29 用的是 [v, a, cmd] 的**不同顺序**，本项目按 transfuser 口径。
"""
import numpy as np, torch
import dd_adapter as DD

CMD_ONEHOT = {"TURN_LEFT":  [1,0,0,0],
              "GO_STRAIGHT":[0,1,0,0],
              "TURN_RIGHT": [0,0,1,0],
              None:         [0,0,0,1]}   # unknown

def status_feature_official(driving_command, ego_velocity, ego_acceleration):
    """driving_command: 'GO_STRAIGHT'/'TURN_LEFT'/'TURN_RIGHT'/None 或长度4的 one-hot
       ego_velocity/ego_acceleration: 长度2 的 [x, y]，ego 系"""
    if isinstance(driving_command,(str,type(None))):
        cmd = CMD_ONEHOT.get(driving_command, CMD_ONEHOT[None])
    else:
        cmd = list(driving_command)
    return torch.from_numpy(np.concatenate([
        np.asarray(cmd, np.float32),
        np.asarray(ego_velocity, np.float32),
        np.asarray(ego_acceleration, np.float32)])).float().unsqueeze(0)

def run_official(runner, img_rgb, driving_command, ego_velocity, ego_acceleration, seed=0):
    """替代 runner.run(img, speed)：喂官方口径的真值 status。返回同样的 dict 结构。"""
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    cam = DD.image_to_camera_feature(img_rgb).to(runner.device)
    st  = status_feature_official(driving_command, ego_velocity, ego_acceleration).to(runner.device)
    out = runner.agent.forward({"camera_feature": cam, "status_feature": st})
    traj = out["trajectory"][0].detach().cpu().numpy()
    pooled = {"vision_mean": np.array(
        [h[:DD.N_IMG_TOK].mean(0).cpu().numpy().astype(np.float32) for h in runner._buf], dtype=object)}
    return {"trajectory": traj, "pooled": pooled}
