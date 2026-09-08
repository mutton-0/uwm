"""I 轴（新口径）第 1 步：导出逐事件、逐层的 原图 / 变外观 池化激活。

与 f_vfaith_extract.py **完全同构**：同一批事件、同一套池化、同一个 run_arm，
唯一区别是第二条臂不是"遮挡危险物"而是"改天空/全图的亮度饱和度噪声"
（scripts/appearance_transform.py）。同构是刻意的 —— 只有两条臂的激活活在
同一个空间、同一批事件上，v_bright 与 v_faith / v_decel 的夹角才有意义。

产出 results/vbright_acts_{corpus}_{scen}_{model}_{kind}_{scope}.npz：
    h_orig__L{l}   [n, d_l]      h_alt__L{l}   [n, d_l]
    eid, scene, v_orig, v_alt    （v_* = arc_full，即行为读数，用于 Δb）

**ctrl 臂**：另存 h_ctrl（对同一张图施加"亮度不变"的置换扰动 —— 把变换后的
图按像素随机重排回原亮度直方图？不。这里 ctrl = 不做任何变换重跑一次，
量的是模型自身的数值抖动地板）。

与 C 轴用的是**同一批事件、同一套 clean/ghost 输入**，区别只在于：
C 轴要逐层 patch（nL+2 次前向），这里只要两次前向（clean 一次、ghost 一次），
所以便宜 nL/2 倍。

产出 results/vfaith_acts_{corpus}_{scen}_{model}.npz：
    h_clean__L{l}  [n_events, d_l]      h_ghost__L{l}  [n_events, d_l]
    eid, scene, v_clean, v_ghost        （后者用于与标量 b 对齐）

不在这里算方向 —— 稀疏化、split-half 噪声地板、跨语料夹角都放在
f_vfaith_direction.py，这样重算方向不必重跑 GPU。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from c_axis_hazard_patch import arc_full, WP_DT                       # noqa: E402

LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
         "simlingo": "SimLingo"}
sys.path.insert(0, str(ROOT / "scripts"))
from appearance_transform import transform                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LABEL))
    ap.add_argument("--work", required=True, help="work_c_dep/{ghost,lead} 或 nuScenes 侧同构目录")
    ap.add_argument("--corpus", default="navsim", choices=["nuscenes", "navsim"])
    ap.add_argument("--scen", required=True, choices=["ghost", "lead"])
    ap.add_argument("--pos", default="A")
    ap.add_argument("--kind", default="night", choices=["night", "rain"])
    ap.add_argument("--scope", default="sky", choices=["sky", "global"])
    ap.add_argument("--no-noise", action="store_true",
                    help="**P-7 教训的直接搬运**：遮挡实验里 b 的暴涨全来自注入噪声，"
                         "不是来自'内容变了'。夜变换默认 noise=7.0，同一个陷阱在这里一样成立。"
                         "开此项把 noise 置 0，只留'变暗/降饱和/偏蓝'，"
                         "这样才能把'光照变了'和'画面变脏了'分开。")
    ap.add_argument("--pool", default="vision_mean",
                    help="池化方式；与 G/I 轴一致默认 vision_mean")
    ap.add_argument("--crop-center-row", type=int, default=0)
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--sl-config", default=str(ROOT / "configs/n1_d2.yaml"))
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    # **必须 resolve**：SimLingo 的加载链路里有 os.chdir（P-4），
    # 相对 --work 会在 chdir 之后解析失败（表现为 events_all.jsonl 找不到）。
    W = Path(A.work).resolve()
    lab = LABEL[A.model]
    tag = (f"{A.corpus}_{A.scen}_{A.model}_{A.kind}_{A.scope}"
           + ("_nonoise" if A.no_noise else ""))
    out = Path(A.out) if A.out else RES / f"vbright_acts_{tag}.npz"

    from PIL import Image

    def read(fn):
        p = fn if Path(fn).is_absolute() else None
        if p is None:                       # ghost 侧是相对 NAVSIM/nuScenes 根的路径
            base = ("/data/dataset/navsim/dataset/sensor_blobs" if A.corpus == "navsim"
                    else A.nuscenes_root)
            p = str(Path(base) / fn)
        return np.asarray(Image.open(p).convert("RGB"))

    IS_SL = A.model == "simlingo"
    if IS_SL:
        from omegaconf import OmegaConf
        from simlingo_runner import SimLingoRunner
        cfg = OmegaConf.to_container(OmegaConf.load(A.sl_config), resolve=True)
        cfg["model"]["device"] = A.device
        runner = SimLingoRunner(cfg, capture_hidden=True)
        nL = runner.n_layers
        import torch as _t

        def run_arm(img, spd, tok, delb):
            r = runner.infer(img, spd, pool_modes=("vision_mean",))
            hs = runner._layer_outputs[-nL:]
            ids = runner._adaptor_dict["language__ids"][0]
            vis = _t.nonzero(ids == runner.img_context_token_id).flatten()
            H = [h[0, vis, :].float().mean(0).cpu().numpy() for h in hs]
            return arc_full(r.waypoints, WP_DT["simlingo"]), H
    else:
        for d in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
            sys.path.insert(0, str(RES / d))
        if A.model == "dd":
            from dd_adapter import DDRunner as Runner
        elif A.model == "ltf":
            from ltf_adapter import LTFRunner as Runner
        else:
            from ddv2_adapter import DDV2Runner as Runner
        if A.crop_center_row:
            import dd_adapter as _DD
            _DD.set_crop_center_row(A.crop_center_row)
        runner = Runner(device=A.device)
        nL = len(runner.sas)
        lidar = None
        if A.model == "ddv2":
            if A.corpus == "navsim":
                from ddv2_adapter import NavsimLidar
                lidar = NavsimLidar()
            else:
                from ddv2_adapter import NuScenesLidar
                lidar = NuScenesLidar(A.nuscenes_root)

        def run_arm(img, spd, tok, delb):
            if lidar is None:
                o = runner.run(img, spd)
            else:
                pts = lidar.ego_points(tok)
                if delb and pts is not None:
                    # 与 C 轴 clean 臂完全同法：删掉 3D 框内的点（框各向外扩 0.25 m）。
                    # 不删点则"危险已移除"只对图像成立、雷达里仍在。
                    from f3_window_boxes import points_in_box
                    msk = np.zeros(len(pts), bool)
                    for b in delb:
                        msk |= points_in_box(pts, np.asarray(b["center"], float),
                                             tuple(x + 0.5 for x in b["size"]), b["yaw"])
                    pts = pts[~msk]
                o = runner.run(img, spd, lidar_xyz=pts)
            H = [np.asarray(x, np.float32) for x in o["pooled"][A.pool]]
            return arc_full(o["trajectory"], WP_DT[A.model]), H

    def _lidar_key(fr):
        return fr.get("lidar_key") or fr.get("sd_token")

    evs = [json.loads(l) for l in open(W / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == A.pos]
    print(f"[vbright/{lab}] {tag}: {len(evs)} 个事件，{nL} 层，池化={A.pool}，"
          f"变换={A.kind}/{A.scope}", flush=True)

    HC = [[] for _ in range(nL)]; HG = [[] for _ in range(nL)]
    eid, scene, vc_, vg_ = [], [], [], []
    for i, ev in enumerate(evs):
        try:
            spd = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            # **两条臂是同一帧**：origin 帧本身 + 它的外观变换版。
            # 不用 x_clean（那是遮挡图），否则 v_bright 会混进遮挡效应。
            fr = ev["x_ghost_frames"][0]
            ic = read(fr["filename"])
            ig = transform(ic, kind=A.kind, scope=A.scope, seed=0,
                           noise_override=(0.0 if A.no_noise else None))
            if not IS_SL:
                runner.set_steering(None)
            vc, Hc = run_arm(ic, spd, _lidar_key(fr), None)
            vg, Hg = run_arm(ig, spd, _lidar_key(fr), None)   # 点云不动：只改图像
        except Exception as e:                                   # noqa: BLE001
            print(f"[vbright/{lab}] skip {ev['event_id']}: {e}", flush=True)
            continue
        for l in range(nL):
            HC[l].append(Hc[l]); HG[l].append(Hg[l])
        eid.append(ev["event_id"]); scene.append(ev["scene_name"])
        vc_.append(vc); vg_.append(vg)
        if (i + 1) % 25 == 0:
            print(f"[vbright/{lab}] {i+1}/{len(evs)}", flush=True)

    D = {"eid": np.array(eid), "scene": np.array(scene),
         "v_orig": np.array(vc_, np.float32), "v_alt": np.array(vg_, np.float32),
         "sign_convention": "Δb = v_alt − v_orig = arc_full(变外观) − arc_full(原图)；"
                            "arc_full 是弧长/时域=规划速度，故 **Δb>0 = 变外观后开得更快**。"
                            "与 F 轴同序：b_model = arc_full(clean=危险物已移除) − arc_full(origin)，"
                            "b_model>0 = 移除危险后变快（期望方向）。"
                            "所以 Δb>0 意味着模型把'天黑了'读成了和'危险没了'同向的信号 —— 不安全方向。",
         "kind": A.kind, "scope": A.scope, "no_noise": bool(A.no_noise)}
    for l in range(nL):
        D[f"h_orig__L{l}"] = np.stack(HC[l]).astype(np.float32)
        D[f"h_alt__L{l}"] = np.stack(HG[l]).astype(np.float32)
    np.savez_compressed(out, **D)
    db = np.array(vg_, np.float32) - np.array(vc_, np.float32)
    faster = int((db > 0).sum())
    print(f"[vbright/{lab}] -> {out}  n={len(eid)}  "
          f"Δb mean={db.mean():+.4f} sd={db.std():.4f}  |Δb| mean={np.abs(db).mean():.4f}  "
          f"变快的事件 {faster}/{len(db)} ({faster/max(len(db),1)*100:.0f}%)"
          f"   [Δb>0 = 变外观后**开得更快**]", flush=True)


if __name__ == "__main__":
    main()
