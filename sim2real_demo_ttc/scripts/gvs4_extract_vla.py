"""G-VS 第三步（VLA 版）|抽取 Alpamayo-R1 / AutoVLA 的 token 级视觉表征。

与 `gvs3_extract_tokens.py` 同一套任务、同一份 SAM 伪 GT、同一个探针，
差别只在**token ↔ 像素的映射**：两个 VLA 把多相机 × 多时刻打包成 Qwen2.5-VL 的 video token。

**AutoVLA 的布局（已实测解出）**：
    `get_prompt` 返回 `video_grid_thw = [[2,18,32]] × 3`（3 路相机各一个 video）。
    Qwen2.5-VL 的 patch 是 14 px、空间再做 2×2 merge ⇒ 每个 video
        2 个时间组 × (18/2) × (32/2) = 2 × 9 × 16 = 288 token，三路合计 864（与实测一致）。
    相机顺序 = `temporal_paths()` 的键序：front / front_left / front_right。
    **我们只取 front 相机的最后一个时间组**（即目标帧）的 9×16 = 144 个 token，
    它们均匀覆盖前视图像全幅（1600×900 → 448×252，宽高比一致，无裁剪）。

**为什么只取前视最后一个时间组**：SAM 伪 GT 是在**前视目标帧**上生成的。
取别的相机或别的时刻的 token 去预测这张图的掩码，是在制造一个无解的任务。
这与 TransFuser 系只取图像 token（不取 BEV latent token）是同一条纪律。
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

if importlib.util.find_spec("numpy") is None:
    for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
               "/data/Zhengyang/alpamayo/src"):
        if _p not in sys.path:
            sys.path.append(_p)
import numpy as np                                                    # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gvs3_extract_tokens import gt_to_grid                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["autovla", "alpa"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--layer", type=int, default=20, help="语言塔取第几层（AutoVLA 的 G 轴峰层是 L20）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--corpus", default="nuscenes", choices=["nuscenes", "navsim"],
                    help="navsim 时不走 nuScenes devkit —— devkit 在两个 VLA 的推理路径里"
                         "只做路径/位姿解析，换成同构的 NAVSIM 输入即可（与 F-3、C 轴同一套）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work); gt_dir = work / "sam_gt"
    from PIL import Image
    import torch

    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == "A" and (gt_dir / f"{e['event_id']}.png").exists()]
    if args.limit:
        evs = evs[: args.limit]

    if args.model != "autovla":
        raise SystemExit("Alpamayo 的 video token 布局未解出，见 §GF/A50，本轮不做")

    sys.path.insert(0, str(RES / "autovla_g1_adapter"))
    from autovla_adapter import AutoVLARunner
    runner = AutoVLARunner(device=args.device)
    rnd = AutoVLARunner(device=args.device)
    for m in rnd.model.modules():                       # 随机初始化对照臂
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    print(f"[GVS-EX/autovla] {len(evs)} 个事件有伪 GT")

    _VIN = {"v": None}

    def _vin():
        if _VIN["v"] is None:
            sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
            from vla_navsim_input import NavsimVLAInput
            _VIN["v"] = NavsimVLAInput()
        return _VIN["v"]

    def tok(rn, ev):
        """返回前视相机最后一个时间组的 token 特征 [144, C] 与 (rows, cols)。"""
        fr = ev["x_ghost_frames"][0]
        spd = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        if args.corpus == "navsim":
            v = _vin()
            sp = ev.get("split", "test")
            im = v.images(sp, ev["scene_name"], ev["query_frame_idx"])
            if im is None:
                return None, None, None
            spd2, acc = v.ego(sp, ev["scene_name"], ev["query_frame_idx"])
            out = rn.run(None, spd2, acc, images=im)
        else:
            out = rn.run(fr["sd_token"], spd)
        h = rn.cap.full[args.layer] if rn.cap.capture_full else None
        mask = rn.cap.mask
        idx = np.nonzero(mask)[0]                        # 图像 token 在序列里的绝对位置
        g = rn._last_grid                                # [(t,h,w)] × n_video
        per = [int(t * hh * ww / 4) for t, hh, ww in g]
        rows, cols = g[0][1] // 2, g[0][2] // 2
        n_t = g[0][0]
        start = 0 + per[0] - rows * cols                 # video0 的最后一个时间组
        sel = idx[start: start + rows * cols]
        return h[sel], rows, cols

    for rn in (runner, rnd):
        rn.set_capture_full(True)
        _orig = rn.model.get_prompt

        def _wrap(feats, _o=_orig, _r=rn):
            inp = _o(feats)
            _r._last_grid = inp["video_grid_thw"].cpu().numpy().tolist()
            return inp
        rn.model.get_prompt = _wrap

    feats, feats_r, gts, poss, scenes = [], [], [], [], []
    for i, ev in enumerate(evs):
        gt = np.array(Image.open(gt_dir / f"{ev['event_id']}.png"))
        try:
            with torch.no_grad():
                f, R_, C_ = tok(runner, ev)
                fr_, _, _ = tok(rnd, ev)
        except Exception as exc:                                       # noqa: BLE001
            print(f"[GVS-EX/autovla] FAIL {ev['event_id']}: {exc}"); continue
        if f is None:                    # NAVSIM 侧某相机/历史帧缺图 -> 跳过该事件
            continue
        g = gt_to_grid(gt, R_, C_)
        if g is None or f.shape[0] != R_ * C_:
            continue
        rr, cc = np.meshgrid(np.arange(R_), np.arange(C_), indexing="ij")
        feats.append(f); feats_r.append(fr_); gts.append(g.reshape(-1))
        poss.append(np.stack([rr.reshape(-1) / R_, cc.reshape(-1) / C_], 1))
        scenes += [ev["scene_name"]] * (R_ * C_)
        if (i + 1) % 25 == 0:
            print(f"[GVS-EX/autovla] {i+1}/{len(evs)} done={len(feats)}", flush=True)

    if not feats:
        raise SystemExit("没有抽到 token")
    Fm = np.concatenate(feats).astype(np.float32)
    np.savez_compressed(args.out, feat=Fm,
                        feat_rand=np.concatenate(feats_r).astype(np.float32),
                        gt=np.concatenate(gts).astype(np.int64),
                        pos=np.concatenate(poss).astype(np.float32),
                        scene=np.array(scenes),
                        meta=json.dumps({"grid": [int(R_), int(C_)], "layer": args.layer,
                                         "token_to_pixel": "前视相机最后一个时间组，9x16 token 均匀覆盖前视全幅",
                                         "random_init": "同架构重建 + 递归 reset_parameters()"},
                                        ensure_ascii=False))
    print(f"[GVS-EX/autovla] token {Fm.shape[0]}，维 {Fm.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
