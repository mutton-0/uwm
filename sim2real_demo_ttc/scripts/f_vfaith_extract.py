"""F 轴第二部分（表征侧）第 1 步：导出逐事件、逐层的 clean / ghost 池化激活。

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
         "simlingo": "SimLingo", "alpa": "Alpamayo-R1", "autovla": "AutoVLA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LABEL))
    ap.add_argument("--work", required=True, help="work_c_dep/{ghost,lead} 或 nuScenes 侧同构目录")
    ap.add_argument("--corpus", default="navsim", choices=["nuscenes", "navsim"])
    ap.add_argument("--scen", required=True, choices=["ghost", "lead"])
    ap.add_argument("--pos", default="A")
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
    tag = f"{A.corpus}_{A.scen}_{A.model}"
    out = Path(A.out) if A.out else RES / f"vfaith_acts_{tag}.npz"

    from PIL import Image

    def read(fn):
        p = fn if Path(fn).is_absolute() else None
        if p is None:                       # ghost 侧是相对 NAVSIM/nuScenes 根的路径
            base = ("/data/dataset/navsim/dataset/sensor_blobs" if A.corpus == "navsim"
                    else A.nuscenes_root)
            p = str(Path(base) / fn)
        return np.asarray(Image.open(p).convert("RGB"))

    IS_SL = A.model == "simlingo"
    IS_VLA = A.model in ("alpa", "autovla")
    if IS_VLA:
        # 两个 VLA 的 runner 接口与 TransFuser 系不同：读数走 cap.pooled[...]，
        # 输入组装完全复用 C 轴的 vla_run（同一批事件、同一套 clean/ghost 帧），
        # 保证 v_faith 与 C 轴、与其余四家活在同一个口径里。
        sys.path.insert(0, str(RES / ("alpamayo_g1_adapter" if A.model == "alpa"
                                      else "autovla_g1_adapter")))
        if A.model == "alpa":
            from alpa_patch import AlpaPatchRunner
            runner = AlpaPatchRunner(device=A.device)
        else:
            from autovla_adapter import AutoVLARunner
            runner = AutoVLARunner(device=A.device, nuscenes_root=A.nuscenes_root)
        nL = runner.cap.n_layers
        import importlib
        _cx = importlib.import_module("c_axis_hazard_patch")

        class _Args:                                  # 复用 c_axis 的 vla_run 需要的字段
            model = A.model; corpus = A.corpus; nuscenes_root = A.nuscenes_root
        _vin = {"v": None}

        def _navsim_in():
            if _vin["v"] is None:
                from vla_navsim_input import NavsimVLAInput
                _vin["v"] = NavsimVLAInput()
            return _vin["v"]

        # ---- 全窗口遮挡图的按需渲染（缓存到 tmp）----
        _BOXF = RES / "vla_boxes_deploy_pool_ghost_vp.json"
        _BOX = ({f"{b['split']}|{b['scene']}|{b['frame_idx']}": b
                 for b in json.load(open(_BOXF))} if _BOXF.exists() else {})
        _CAM2KEY = {"CAM_F0": "front_camera", "CAM_L0": "front_left_camera",
                    "CAM_R0": "front_right_camera"}
        _NSB = "/data/dataset/navsim/dataset/sensor_blobs"
        _TMP = Path("/tmp/claude-1001/-data-ruolin-uwm-sim2real-demo-ttc/"
                    "007773d2-b773-4b39-be8d-122fceddf6fe/scratchpad/vfaith_masks")
        _TMP.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(RES))
        from f3_occlusion_necessity import occlude as _occ

        def _win_masks(sp, scn, j):
            """返回 {"CAM|k": 遮挡图路径}，逐 (相机, 帧) 只在该格有框时渲染。"""
            rec = _BOX.get(f"{sp}|{scn}|{j}")
            if not rec or not rec.get("window"):
                return {}
            out = {}
            for slot, w in rec["window"].items():
                if not w["boxes"]:
                    continue
                op = _TMP / f"{scn}_{j}_{slot.replace('|','_')}.jpg"
                if not op.exists():
                    # data_path 不含 split 前缀，必须补上（与 f3_vla_navsim 同）
                    src = Path(_NSB) / sp / w["path"]
                    if not src.exists():
                        continue
                    im0 = np.asarray(Image.open(src).convert("RGB"))
                    for bb in w["boxes"]:
                        im0 = _occ(im0, bb)
                    Image.fromarray(im0).save(op, quality=95)
                out[slot] = str(op)
            return out

        def vla_run(ev, cond):
            """与 c_axis_hazard_patch.vla_run 同构（NAVSIM 侧不走 nuScenes devkit）。"""
            if A.corpus == "navsim":
                # ---- **全窗口遮挡**（§FM/A56 向提轴链路的补齐，2026-09-07）----
                # 旧版只换「前视当前帧」，16 格输入里只改 1 格，其余 15 格危险物仍在；
                # Alpamayo 更糟 —— 它有**两路前视**（camera_indices 1 与 6），
                # 旧版只遮索引 1 那一路，模型从 30° 长焦那一路照样看得见。
                # 于是提出来的方向量的是「图像上多了块补丁」，不是「场景里有危险」。
                vin = _navsim_in()
                sp = ev.get("split", "test"); scn = ev["scene_name"]
                j = ev["query_frame_idx"]
                need = (cond == "clean")
                pm = _win_masks(sp, scn, j) if need else {}
                if need and not pm:
                    return None
                if A.model == "autovla":
                    im = vin.images(sp, scn, j)
                    if im is None:
                        return None
                    if pm:
                        im = {k: list(v) for k, v in im.items()}
                        for slot, pth in pm.items():
                            cam, k = slot.split("|")
                            key = _CAM2KEY.get(cam)
                            if key in im and int(k) < len(im[key]):
                                im[key][int(k)] = pth
                    spd, acc = vin.ego(sp, scn, j)
                    return runner.run(None, spd, acc, images=im)
                from alpa_navsim_loader import load_navsim
                d = load_navsim(vin, sp, scn, j)
                if d is None:
                    return None
                if pm:
                    import torch as _t
                    # 槽位：排序后 camera_indices=[0,1,2,6] -> 0=前左,1=前(广角),2=前右,3=前(长焦)
                    # **两路前视都要遮**，它们看同一时刻同一方向，只是 FOV 不同。
                    C2S = {"CAM_L0": [0], "CAM_F0": [1, 3], "CAM_R0": [2]}
                    for slot, pth in pm.items():
                        cam, k = slot.split("|"); k = int(k)
                        arr = np.asarray(Image.open(pth).convert("RGB"))
                        for sl in C2S.get(cam, []):
                            if sl < d["image_frames"].shape[0] and k < d["image_frames"].shape[1]:
                                tgt = d["image_frames"][sl, k]
                                d["image_frames"][sl, k] = _t.from_numpy(
                                    arr.copy()).permute(2, 0, 1).to(tgt.dtype)
                return runner.run(scn, 0.0, data=d)
            # ---- nuScenes 侧 ----
            # **必须显式注入遮挡图**：新的 brake-first 池里 clean 与 ghost 是**同一帧**
            # （t 与 sd_token 完全相同），差别只在 filename —— clean 指向预渲染的遮挡 jpg。
            # 两个 VLA 的 nuScenes 入口只认 t / sd_token，不读 filename，
            # 若不注入，两条臂加载的是同一张原图，δ 恒为 0（实测 36 层全部零范数 -> θ=NaN）。
            patched = (ev["x_clean_frames"][0]["filename"] if cond == "clean" else None)
            if A.model == "alpa":
                d = runner.r.load(ev["scene_name"], ev[f"x_{cond}_frames"][0]["t"])
                if d is None:
                    return None
                if patched is not None:
                    import torch as _t
                    sl = int((d["camera_indices"] == 1).nonzero()[0][0])
                    tgt = d["image_frames"][sl, -1]                 # [3,H,W]
                    im = Image.open(patched).convert("RGB").resize(
                        (int(tgt.shape[-1]), int(tgt.shape[-2])), Image.BILINEAR)
                    d["image_frames"][sl, -1] = _t.from_numpy(
                        np.asarray(im)).permute(2, 0, 1).to(tgt.dtype)
                return runner.run(ev["scene_name"], ev[f"x_{cond}_frames"][0]["t"],
                                  ego_anchor_t=ev["x_clean_frames"][0]["t"], data=d)
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            tok = ev[f"x_{cond}_frames"][0]["sd_token"]
            im = runner.temporal_paths(tok)
            if patched is not None:
                im = dict(im); im["front_camera"] = list(im["front_camera"])
                im["front_camera"][-1] = patched                    # 只换前视当前帧
            return runner.run(tok, anchor, images=im)
    elif IS_SL:
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
    print(f"[vfaith/{lab}] {tag}: {len(evs)} 个事件，{nL} 层，池化={A.pool}", flush=True)

    HC = [[] for _ in range(nL)]; HG = [[] for _ in range(nL)]
    eid, scene, vc_, vg_ = [], [], [], []
    for i, ev in enumerate(evs):
        try:
            if IS_VLA:
                runner.set_capture_full(False)
                oc = vla_run(ev, "clean")
                og = vla_run(ev, "ghost")
                if oc is None or og is None:
                    print(f"[vfaith/{lab}] skip {ev['event_id']}: 输入缺失", flush=True)
                    continue
                vc = arc_full(oc["trajectory"], WP_DT[A.model])
                vg = arc_full(og["trajectory"], WP_DT[A.model])
                Hc = [np.asarray(x, np.float32) for x in oc["pooled"][A.pool]]
                Hg = [np.asarray(x, np.float32) for x in og["pooled"][A.pool]]
                for l in range(nL):
                    HC[l].append(Hc[l]); HG[l].append(Hg[l])
                eid.append(ev["event_id"]); scene.append(ev["scene_name"])
                vc_.append(vc); vg_.append(vg)
                if (i + 1) % 10 == 0:
                    print(f"[vfaith/{lab}] {i+1}/{len(evs)}", flush=True)
                continue
            spd = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            ic = read(ev["x_clean_frames"][0]["filename"])
            ig = read(ev["x_ghost_frames"][0]["filename"])
            if not IS_SL:
                runner.set_steering(None)
            vc, Hc = run_arm(ic, spd, _lidar_key(ev["x_clean_frames"][0]),
                             ev["x_clean_frames"][0].get("delete_boxes3d"))
            vg, Hg = run_arm(ig, spd, _lidar_key(ev["x_ghost_frames"][0]), None)
        except Exception as e:                                   # noqa: BLE001
            print(f"[vfaith/{lab}] skip {ev['event_id']}: {e}", flush=True)
            continue
        for l in range(nL):
            HC[l].append(Hc[l]); HG[l].append(Hg[l])
        eid.append(ev["event_id"]); scene.append(ev["scene_name"])
        vc_.append(vc); vg_.append(vg)
        if (i + 1) % 25 == 0:
            print(f"[vfaith/{lab}] {i+1}/{len(evs)}", flush=True)

    D = {"eid": np.array(eid), "scene": np.array(scene),
         "v_clean": np.array(vc_, np.float32), "v_ghost": np.array(vg_, np.float32)}
    for l in range(nL):
        D[f"h_clean__L{l}"] = np.stack(HC[l]).astype(np.float32)
        D[f"h_ghost__L{l}"] = np.stack(HG[l]).astype(np.float32)
    np.savez_compressed(out, **D)
    print(f"[vfaith/{lab}] -> {out}  n={len(eid)}  层维度="
          f"{[D[f'h_clean__L{l}'].shape[1] for l in range(nL)]}", flush=True)


if __name__ == "__main__":
    main()
