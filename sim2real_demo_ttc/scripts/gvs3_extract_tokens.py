"""G-VS 第三步|抽取各候选的 **token 级**（未池化）视觉表征 + 对齐 SAM 伪 GT。

工单：docs/g_vs_f3_simplified_axes_workorder.md §1。

**为什么不能用已有缓存**：既有 `*_cache/*.npz` 存的是 vision_mean / region_mean 等**池化**表征，
空间结构已经被抹掉，无法做分割探针。本脚本重新前向一遍，取 token 级特征。

**跨候选统一的关键：token ↔ 像素的映射必须按各自的前端如实还原。**
  TransFuser 系（DiffusionDrive / LTF / DiffusionDriveV2）
      图像 token 是 8×32 的网格，覆盖 `crop_4to1_no_sky` 裁出的 4:1 带
      （原图 rows [top, top+W/4]，全宽）。峰层取该候选自己 C-hazard 的责任层 L6。
  SimLingo
      InternVL2 的 dynamic_preprocess 把图切成 n_patches 个 tile，
      每 tile 经 ViT patch14 → 32×32 → pixel_shuffle(0.5) → 16×16 = 256 token。
      复用 `simlingo_runner.image_to_token_grid` 的同一套几何做**逆映射**。

伪 GT 按各自网格做**面积加权下采样**：token 的标签 = 该 token 对应像素块里 object 像素占比 > 0.5。

`--random-init` 臂：同架构、随机初始化权重的模型，跑同一批图，取同一层同一网格的特征。
这是 selectivity 对照的来源（Hewitt & Liang 2019），**不是可选项**。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

# cv2 只在本脚本自己的抽取路径里用；VLA 版（gvs4）会 import 本模块的 gt_to_grid，
# 而 VLA 所在解释器没有 opencv，故延迟到用时再 import。
cv2 = None


def _cv2():
    global cv2
    if cv2 is None:
        import cv2 as _c
        cv2 = _c
    return cv2

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")


def gt_to_grid(gt, rows, cols, top=None, height=None, thr=0.5):
    """把原图分辨率的 object 掩码下采样到 rows×cols 的 token 网格。

    top/height 非空时先裁到该行带（TransFuser 系的 4:1 裁剪），与模型看到的画面一致。
    """
    g = gt if top is None else gt[top:top + height]
    if g.size == 0:
        return None
    H, Wd = g.shape
    ys = np.linspace(0, H, rows + 1).astype(int)
    xs = np.linspace(0, Wd, cols + 1).astype(int)
    out = np.zeros((rows, cols), np.int64)
    for r in range(rows):
        for c in range(cols):
            blk = g[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            out[r, c] = int(blk.mean() > thr) if blk.size else 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dd", "ltf", "ddv2", "simlingo"])
    ap.add_argument("--work", default=str(W))
    ap.add_argument("--nuscenes-root", default="/data/dataset/nuscenes/v1.0-trainval")
    ap.add_argument("--layer", type=int, default=6, help="取哪一层的 token 特征（TransFuser 系默认 L6 = C 轴责任层）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    work = Path(args.work)
    gt_dir = work / "sam_gt"

    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == "A" and (gt_dir / f"{e['event_id']}.png").exists()]
    if args.limit:
        evs = evs[: args.limit]
    print(f"[GVS-EX/{args.model}] {len(evs)} 个事件有伪 GT")

    feats, feats_r, gts, poss, scenes = [], [], [], [], []

    if args.model in ("dd", "ltf", "ddv2"):
        sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
        sys.path.insert(0, str(RES / "ltf_g1_adapter")); sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
        import dd_adapter as DD
        import torch
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        # 随机初始化对照：同架构、同前端，只把 encoder 权重重新随机初始化
        rnd = R(device=args.device)
        for m in rnd.agent.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        lidar = None
        if args.model == "ddv2":
            from ddv2_adapter import NuScenesLidar
            lidar = NuScenesLidar(args.nuscenes_root)
        ROWS, COLS = DD.IMG_VERT, DD.IMG_HORZ

        def tok_feats(rn, img, fr):
            kw = {} if lidar is None else {"lidar_xyz": lidar.ego_points(fr["sd_token"])}
            rn.run(img, float(np.mean([fr["ego_speed_mps"]])), **kw)
            h = rn._buf[args.layer]                       # [n_tok, C]
            return h[: DD.N_IMG_TOK].float().cpu().numpy()

        for i, ev in enumerate(evs):
            fr = ev["x_ghost_frames"][0]
            img = _cv2().cvtColor(_cv2().imread(str(Path(args.nuscenes_root) / fr["filename"])), cv2.COLOR_BGR2RGB)
            gt = _cv2().imread(str(gt_dir / f"{ev['event_id']}.png"), _cv2().IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            top, th = DD.crop_geometry((img.shape[1], img.shape[0]))
            g = gt_to_grid(gt, ROWS, COLS, top, th)
            if g is None:
                continue
            with torch.no_grad():
                f = tok_feats(runner, img, fr); fr_ = tok_feats(rnd, img, fr)
            rr, cc = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
            feats.append(f); feats_r.append(fr_); gts.append(g.reshape(-1))
            poss.append(np.stack([rr.reshape(-1) / ROWS, cc.reshape(-1) / COLS], 1))
            scenes += [ev["scene_name"]] * (ROWS * COLS)
            if (i + 1) % 50 == 0:
                print(f"[GVS-EX/{args.model}] {i+1}/{len(evs)}", flush=True)
        meta = {"grid": [ROWS, COLS], "layer": args.layer,
                "token_to_pixel": "8x32 网格覆盖 crop_4to1_no_sky 的 4:1 带（原图 rows [top, top+W/4]，全宽）",
                "random_init": "同架构重建 + 递归 reset_parameters()"}
    else:
        from omegaconf import OmegaConf
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        cfg = OmegaConf.to_container(OmegaConf.load(
            "/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        runner = SimLingoRunner(cfg, capture_hidden=True)
        rnd = SimLingoRunner(cfg, capture_hidden=True)
        for m in rnd.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        SIDE = 16                                  # 每 tile pixel_shuffle 后 16x16 token

        def tok_feats(rn, img, spd):
            rn.infer(img, spd, pool_modes=("vision_mean",))
            hs = rn._layer_outputs[-rn.n_layers:]
            ids = rn._adaptor_dict["language__ids"][0]
            vis = torch.nonzero(ids == rn.img_context_token_id).flatten()
            h = hs[args.layer][0, vis, :].float().cpu().numpy()
            return h, int(rn._n_patches)

        for i, ev in enumerate(evs):
            fr = ev["x_ghost_frames"][0]
            img = _cv2().cvtColor(_cv2().imread(str(Path(args.nuscenes_root) / fr["filename"])), cv2.COLOR_BGR2RGB)
            gt = _cv2().imread(str(gt_dir / f"{ev['event_id']}.png"), _cv2().IMREAD_GRAYSCALE)
            if img is None or gt is None:
                continue
            spd = float(np.mean([f_["ego_speed_mps"] for f_ in ev["x_clean_frames"]]))
            with torch.no_grad():
                f, npat = tok_feats(runner, img, spd)
                fr_, _ = tok_feats(rnd, img, spd)
            if f.shape[0] != npat * SIDE * SIDE:
                continue                                   # tile 数与 token 数不自洽，跳过
            # dynamic_preprocess 把整图重排成 (gh, gw) 个 tile；逐 tile 映射回原图区域
            gh = int(round(np.sqrt(npat * img.shape[0] / img.shape[1])))
            gh = max(1, min(gh, npat)); gw = max(1, npat // gh)
            if gh * gw != npat:
                continue
            g = gt_to_grid(gt, gh * SIDE, gw * SIDE)
            # token 顺序是 tile-major，需重排成 (gh*SIDE, gw*SIDE) 的行主序
            order = np.arange(npat * SIDE * SIDE).reshape(npat, SIDE, SIDE)
            order = order.reshape(gh, gw, SIDE, SIDE).transpose(0, 2, 1, 3).reshape(-1)
            feats.append(f[order]); feats_r.append(fr_[order]); gts.append(g.reshape(-1))
            rr, cc = np.meshgrid(np.arange(gh * SIDE), np.arange(gw * SIDE), indexing="ij")
            poss.append(np.stack([rr.reshape(-1) / (gh * SIDE), cc.reshape(-1) / (gw * SIDE)], 1))
            scenes += [ev["scene_name"]] * (gh * SIDE * gw * SIDE)
            if (i + 1) % 25 == 0:
                print(f"[GVS-EX/simlingo] {i+1}/{len(evs)}", flush=True)
        meta = {"grid": "per-image (gh*16, gw*16)", "layer": args.layer,
                "token_to_pixel": "InternVL2 dynamic_preprocess 的 tile 网格，每 tile 16x16 token",
                "random_init": "同架构重建 + 递归 reset_parameters()"}

    if not feats:
        raise SystemExit("没有抽到任何 token 特征")
    Fm = np.concatenate(feats).astype(np.float32)
    np.savez_compressed(args.out, feat=Fm,
                        feat_rand=np.concatenate(feats_r).astype(np.float32),
                        gt=np.concatenate(gts).astype(np.int64),
                        pos=np.concatenate(poss).astype(np.float32),
                        scene=np.array(scenes), meta=json.dumps(meta, ensure_ascii=False))
    print(f"[GVS-EX/{args.model}] token {Fm.shape[0]}，维 {Fm.shape[1]}，"
          f"object 占比 {np.concatenate(gts).mean():.3f} -> {args.out}")


if __name__ == "__main__":
    main()
