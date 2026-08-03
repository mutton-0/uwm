#!/usr/bin/env python
"""
Ghosthead 因果定位：复用原 scripts/sim2real_analysis/activation_patching.py 的**内核**
(sa_modules / run_cache / run_patch / run + recovery 数学)，只替换输入前端为 ghosthead 图。

设定（与原脚本相反，因为本项目 sim 是 in-domain）：
  参考(好) = transfered(CARLA, sim, in-domain)
  退化      = origin(世界模型真实感, real, out-of-domain)
patch: 跑 real 前向，把第 L 层 encoder SelfAttention 输出换成 sim 的，看轨迹是否回到 sim。
  recovery(L)=1 -> 该层因果负责 real 输入引起的轨迹退化；=0 -> 只相关不因果。
"""
import os, sys, csv, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import run_ghosthead_infer as G
# 复用原脚本内核（import 只触发常量/函数定义，main() 不执行）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim2real_analysis"))
import activation_patching as AP


def build_batch(scene, mats, mp4, fi):
    import cv2, tempfile
    tmp = getattr(build_batch, "_tmp", None) or tempfile.mkdtemp()
    build_batch._tmp = tmp
    raw = os.path.join(tmp, "r.png")
    G.extract_frame(mp4, fi, raw)
    img = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB)
    cam = G.build_camera_feature(G.crop_4to1_no_sky(img))
    v, a = G.ego_status_at(mats, fi)
    status = torch.from_numpy(np.concatenate([G.DRIVING_COMMAND, v, a])).float().unsqueeze(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"camera_feature": cam.to(dev), "status_feature": status.to(dev)}


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
    ap.add_argument("--k", type=int, default=10, help="取 real 退化最狠的前 K 场景")
    ap.add_argument("--sec", type=int, default=1, help="用哪一秒的帧(cov 最大)")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer/patching")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    summ = "/data/ruolin/uwm/outputs/ghosthead_infer/_summary_all.csv"
    scenes = args.scenes.split(",") if args.scenes else select_scenes(summ, args.k)
    print(f">> 因果修补场景({len(scenes)}), 帧 t={args.sec}s, 参考=transfered(sim) 退化=origin(real)")

    agent = G.load_agent()
    model = agent._transfuser_model.eval()
    sas = AP.sa_modules(model)          # 8 个 SelfAttention（复用原脚本）
    nL = len(sas)
    print(f">> #SelfAttention = {nL}")

    rows = []
    for sc in scenes:
        trans_dir = f"{G.DATA_ROOT}/renders/{sc}"
        trans_mp4 = f"{trans_dir}/frames.mp4"
        origin_mp4 = f"{G.DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"
        if not (os.path.exists(trans_mp4) and os.path.exists(origin_mp4)):
            print(f"   [skip] {sc}"); continue
        scene = G.load_scene(trans_dir); mats = G.ego_mats(scene)
        fi = min(int(round(args.sec * G.FPS)), len(mats) - 1)

        b_sim = build_batch(scene, mats, trans_mp4, fi)      # 参考(好)
        b_real = build_batch(scene, mats, origin_mp4, fi)    # 退化

        # 缓存 sim 各层输出（run_cache 复用原脚本），real 轨迹
        traj_sim, cache_sim = AP.run_cache(model, b_sim, sas)
        traj_real = AP.run(model, b_real)
        xy_sim, xy_real = traj_sim[:, :2], traj_real[:, :2]
        gap = float(np.linalg.norm(xy_real - xy_sim))
        if gap < AP.GAP_MIN:
            print(f"   {sc}: gap={gap:.2f} < {AP.GAP_MIN} 跳过(轨迹几乎相同)")
            rows.append((sc, gap, None, None)); continue

        recs = []
        for L in range(nL):
            tp = AP.run_patch(model, b_real, sas, L, cache_sim)   # 复用原脚本
            recs.append(1 - float(np.linalg.norm(tp[:, :2] - xy_sim)) / gap)
        # patch ALL 自洽性检查
        hs = [sas[L].register_forward_hook((lambda c: (lambda m, i, o: c))(cache_sim[L])) for L in range(nL)]
        r_all = 1 - float(np.linalg.norm(AP.run(model, b_real)[:, :2] - xy_sim)) / gap
        for h in hs: h.remove()

        rows.append((sc, gap, recs, r_all))
        top = np.argsort(recs)[::-1][:3]
        print(f"   {sc}: gap={gap:.2f} top3=" + " ".join(f"L{i}={recs[i]:+.2f}" for i in top) + f" ALL={r_all:+.2f}")

    used = [r for r in rows if r[2] is not None]
    if not used:
        print("!! 无 gap>阈值 的场景"); return
    R = np.array([r[2] for r in used])
    posR = np.clip(R, 0, None)
    print(f"\n===== 汇总: {len(used)}/{len(rows)} 场景轨迹显著不同(gap>{AP.GAP_MIN}) =====")
    print("各层 recovery 均值:  " + " ".join(f"L{i}={R[:, i].mean():+.2f}" for i in range(nL)))
    print("各层 正recovery均值:" + " ".join(f"L{i}={posR[:, i].mean():+.2f}" for i in range(nL)))
    am = [int(np.argmax(R[j])) for j in range(len(used))]
    vals, cnts = np.unique(am, return_counts=True)
    print("argmax 因果层分布:", {f"L{v}": int(c) for v, c in zip(vals, cnts)})
    deep = sum(posR[j, 4:7].sum() / (posR[j].sum() + 1e-9) > 0.5 for j in range(len(used)))
    print(f'深层 L4-6 主导(占正recovery>50%) 场景: {deep}/{len(used)}')

    # CSV + 图
    with open(os.path.join(args.out, "recovery.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "gap"] + [f"L{i}" for i in range(nL)] + ["ALL"])
        for sc, gap, recs, r_all in rows:
            w.writerow([sc, f"{gap:.3f}"] + ([f"{v:.3f}" for v in recs] if recs else [""] * nL) +
                       [f"{r_all:.3f}" if r_all is not None else ""])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for j in range(len(used)):
        ax.plot(range(nL), R[j], color="gray", alpha=0.35, lw=1)
    ax.plot(range(nL), R.mean(0), "-o", color="#c33", lw=2.5, label="mean recovery")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("encoder self-attn layer"); ax.set_ylabel("recovery(L)  real→sim")
    ax.set_title(f"Causal activation patching (n={len(used)} real-degraded scenes)\nhigh=layer causally drives real degradation")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "recovery_by_layer.png"), dpi=130); plt.close(fig)
    print(f">> 产物: {args.out}")


if __name__ == "__main__":
    main()
