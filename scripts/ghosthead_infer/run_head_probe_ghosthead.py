#!/usr/bin/env python
"""
B: 内建头一致性探针（training-free, plan §4.2 O2）。
复用 DiffusionDrive 自带的 BEV 语义头(road/centerline/vehicles) + agent 检测头，
比较 sim(transfered) vs real(origin) 输入下模型"看到的场景"退化多少——尤其是否漏看 ghost。

BEV 语义类: 0 bg,1 road,2 walkway,3 centerline,4 static,5 vehicles,6 pedestrian。
无外部 GT，用 sim-vs-real **一致性** (per-class IoU) + vehicle 像素量 + agent 头在 GT actor 处置信度。
"""
import os, sys, csv, argparse, tempfile
import numpy as np
import torch
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import run_ghosthead_infer as G

CLASS_NAME = {1: "road", 3: "centerline", 5: "vehicles"}
BEV_COLORS = np.array([[30, 30, 40], [90, 90, 90], [60, 120, 60], [230, 210, 60],
                       [150, 100, 200], [230, 60, 60], [60, 180, 230]], dtype=np.uint8)  # 0..6


def infer_heads(agent, model, cam, status):
    dev = next(agent.parameters()).device
    with torch.no_grad():
        out = model({"camera_feature": cam.to(dev), "status_feature": status.to(dev)})
    bev = out["bev_semantic_map"][0].argmax(0).cpu().numpy()           # (128,256)
    ag = out["agent_states"][0].cpu().numpy()                           # (30,5)
    conf = torch.sigmoid(out["agent_labels"][0]).cpu().numpy()          # (30,)
    return bev, ag, conf


def build(scene, mats, mp4, fi, tmp):
    raw = f"{tmp}/r.png"; G.extract_frame(mp4, fi, raw)
    img = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB)
    cam = G.build_camera_feature(G.crop_4to1_no_sky(img))
    v, a = G.ego_status_at(mats, fi)
    status = torch.from_numpy(np.concatenate([G.DRIVING_COMMAND, v, a])).float().unsqueeze(0)
    return cam, status


def iou(a, b):
    inter = np.logical_and(a, b).sum(); uni = np.logical_or(a, b).sum()
    return float(inter / uni) if uni > 0 else 1.0


def actor_ego_centroid(scene, mats, fi, aid):
    R = mats[fi, :3, :3]; t = mats[fi, :3, 3]
    for a in scene["frames"][fi]["actors"]:
        if a["id"] == aid:
            c = np.array(a["corners_world"]).mean(0)
            e = (c - t) @ R
            return e[:2]
    return None


def nearest_conf(ag, conf, xy):
    d = np.linalg.norm(ag[:, :2] - xy[None, :], axis=1)
    k = int(np.argmin(d))
    return conf[k], float(d[k])


def select_scenes(summary_csv, k):
    rows = list(csv.DictReader(open(summary_csv)))
    per = {}
    for r in rows:
        if float(r["gt_coverage_s"]) <= 0:
            continue
        per.setdefault(r["scene"], {"transfered": [], "origin": []})[r["source"]].append(float(r["total"]))
    gaps = [(np.mean(d["transfered"]) - np.mean(d["origin"]), s)
            for s, d in per.items() if d["transfered"] and d["origin"]]
    gaps.sort(reverse=True)
    return [s for _, s in gaps[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--frames", default="1,2,3")
    ap.add_argument("--viz_frame", type=int, default=1)
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer/head_probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")]
    tmp = tempfile.mkdtemp()

    scenes = select_scenes("/data/ruolin/uwm/outputs/ghosthead_infer/_summary_all.csv", args.k)
    print(f">> 内建头探针: {len(scenes)} 场景, frames={frames}")

    agent = G.load_agent(); model = agent._transfuser_model.eval()

    rows = []
    agg = {c: [] for c in CLASS_NAME}        # per-class sim-vs-real IoU
    veh_sim, veh_real = [], []
    ghost_conf_sim, ghost_conf_real = [], []
    maxc_sim, maxc_real = [], []

    for sc in scenes:
        trans_mp4 = f"{G.DATA_ROOT}/renders/{sc}/frames.mp4"
        origin_mp4 = f"{G.DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"
        if not (os.path.exists(trans_mp4) and os.path.exists(origin_mp4)):
            continue
        scene = G.load_scene(f"{G.DATA_ROOT}/renders/{sc}"); mats = G.ego_mats(scene)
        ious = {c: [] for c in CLASS_NAME}
        vs, vr, gcs, gcr, mcs, mcr = [], [], [], [], [], []
        viz = {}
        for sec in frames:
            fi = min(int(round(sec * G.FPS)), len(mats) - 1)
            cam_s, st_s = build(scene, mats, trans_mp4, fi, tmp)
            cam_r, st_r = build(scene, mats, origin_mp4, fi, tmp)
            bev_s, ag_s, cf_s = infer_heads(agent, model, cam_s, st_s)
            bev_r, ag_r, cf_r = infer_heads(agent, model, cam_r, st_r)
            for c in CLASS_NAME:
                ious[c].append(iou(bev_s == c, bev_r == c))
            vs.append(int((bev_s == 5).sum())); vr.append(int((bev_r == 5).sum()))
            mcs.append(float(cf_s.max())); mcr.append(float(cf_r.max()))
            g = actor_ego_centroid(scene, mats, fi, "ghost")
            if g is not None:
                gcs.append(nearest_conf(ag_s, cf_s, g)[0]); gcr.append(nearest_conf(ag_r, cf_r, g)[0])
            if sec == args.viz_frame:
                viz = dict(bev_s=bev_s, bev_r=bev_r)
        for c in CLASS_NAME:
            agg[c].append(np.mean(ious[c]))
        veh_sim.append(np.mean(vs)); veh_real.append(np.mean(vr))
        maxc_sim.append(np.mean(mcs)); maxc_real.append(np.mean(mcr))
        if gcs:
            ghost_conf_sim.append(np.mean(gcs)); ghost_conf_real.append(np.mean(gcr))
        rows.append([sc] + [f"{np.mean(ious[c]):.3f}" for c in CLASS_NAME] +
                    [f"{np.mean(vs):.0f}", f"{np.mean(vr):.0f}",
                     f"{np.mean(gcs):.3f}" if gcs else "", f"{np.mean(gcr):.3f}" if gcr else "",
                     f"{np.mean(mcs):.3f}", f"{np.mean(mcr):.3f}"])
        print(f"   {sc}: road_IoU={np.mean(ious[1]):.2f} cl_IoU={np.mean(ious[3]):.2f} "
              f"veh_IoU={np.mean(ious[5]):.2f} veh_px sim={np.mean(vs):.0f}/real={np.mean(vr):.0f} "
              f"ghostconf sim={np.mean(gcs) if gcs else 0:.2f}/real={np.mean(gcr) if gcr else 0:.2f}")

        # ---- BEV 语义 sim vs real 并排
        if viz:
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
            for ax, key, ttl in [(axes[0], "bev_s", "sim(transfered)"), (axes[1], "bev_r", "real(origin)")]:
                rgb = BEV_COLORS[viz[key]]
                ax.imshow(rgb); ax.set_title(ttl, fontsize=10); ax.axis("off")
            fig.suptitle(f"{sc}  BEV semantic head (1road 3centerline 5vehicle)  t={args.viz_frame}s", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(os.path.join(args.out, f"bev_sem_{sc}.png"), dpi=120); plt.close(fig)

    # ---- CSV
    with open(os.path.join(args.out, "head_probe_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "road_IoU", "centerline_IoU", "vehicle_IoU",
                    "veh_px_sim", "veh_px_real", "ghost_conf_sim", "ghost_conf_real",
                    "maxconf_sim", "maxconf_real"])
        w.writerows(rows)

    # ---- 汇总图
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.3))
    cls = list(CLASS_NAME.values())
    means = [np.mean(agg[c]) for c in CLASS_NAME]
    a1.bar(cls, means, color=["#888", "#3a3", "#c33"])
    a1.set_ylim(0, 1); a1.set_title("BEV semantic sim-vs-real IoU\n(low = domain shift changes scene understanding)")
    for i, m in enumerate(means): a1.text(i, m + 0.02, f"{m:.2f}", ha="center")
    a2.bar(["sim", "real"], [np.mean(veh_sim), np.mean(veh_real)], color=["#1a7", "#c33"])
    a2.set_title("vehicle-class BEV pixels\n(fewer on real = misses other vehicles/ghost)")
    for i, m in enumerate([np.mean(veh_sim), np.mean(veh_real)]): a2.text(i, m, f"{m:.0f}", ha="center", va="bottom")
    if ghost_conf_sim:
        a3.bar(["sim", "real"], [np.mean(ghost_conf_sim), np.mean(ghost_conf_real)], color=["#1a7", "#c33"])
        a3.set_title("agent-head conf @ GT ghost\n(lower on real = less certain about ghost)")
        for i, m in enumerate([np.mean(ghost_conf_sim), np.mean(ghost_conf_real)]):
            a3.text(i, m, f"{m:.3f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "head_probe_bars.png"), dpi=130); plt.close(fig)

    print(f"\n>> BEV IoU 均值: " + " ".join(f"{CLASS_NAME[c]}={np.mean(agg[c]):.2f}" for c in CLASS_NAME))
    print(f">> vehicle 像素 sim={np.mean(veh_sim):.0f} real={np.mean(veh_real):.0f}")
    if ghost_conf_sim:
        print(f">> ghost 处置信度 sim={np.mean(ghost_conf_sim):.3f} real={np.mean(ghost_conf_real):.3f}")
    print(f">> 产物: {args.out}")


if __name__ == "__main__":
    main()
