#!/usr/bin/env python
"""
E: 因果层强度 ↔ PDM 退化 相关性（plan §4.2 验收关卡）。
对全部场景在 t=1s：算 sim/real 轨迹 gap + 逐层激活修补 recovery（复用 activation_patching 内核），
与该帧 PDM Δtotal(=transfered_total - origin_total) 做 Pearson/Spearman 相关。
候选预测量：raw gap / L6 recovery / 深层L4-6正recovery占比 / gap×深层占比(带幅度) / gap×L6recovery。
"""
import os, sys, csv, argparse, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import run_ghosthead_infer as G
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim2real_analysis"))
import activation_patching as AP


def build_batch(scene, mats, mp4, fi, tmp):
    import cv2
    raw = f"{tmp}/r.png"; G.extract_frame(mp4, fi, raw)
    img = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB)
    cam = G.build_camera_feature(G.crop_4to1_no_sky(img))
    v, a = G.ego_status_at(mats, fi)
    st = torch.from_numpy(np.concatenate([G.DRIVING_COMMAND, v, a])).float().unsqueeze(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"camera_feature": cam.to(dev), "status_feature": st.to(dev)}


def pdm_delta_t1(csvp):
    """每场景 t=1s 的 transfered/origin total -> Δ。"""
    rows = list(csv.DictReader(open(csvp)))
    d = {}
    for r in rows:
        if r["frame_sec"] != "1":
            continue
        d.setdefault(r["scene"], {})[r["source"]] = float(r["total"])
    return {s: v for s, v in d.items() if "transfered" in v and "origin" in v}


def corr(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(x, y)[0, 1])
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return pear, spear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=int, default=1)
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer/causal_pdm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp()

    pdm = pdm_delta_t1("/data/ruolin/uwm/outputs/ghosthead_infer/_summary_all.csv")
    scenes = sorted(pdm.keys())
    print(f">> 场景 {len(scenes)}, t={args.sec}s")

    agent = G.load_agent(); model = agent._transfuser_model.eval()
    sas = AP.sa_modules(model); nL = len(sas)

    recs = []   # dict per scene
    for i, sc in enumerate(scenes):
        trans_mp4 = f"{G.DATA_ROOT}/renders/{sc}/frames.mp4"
        origin_mp4 = f"{G.DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"
        if not (os.path.exists(trans_mp4) and os.path.exists(origin_mp4)):
            continue
        scene = G.load_scene(f"{G.DATA_ROOT}/renders/{sc}"); mats = G.ego_mats(scene)
        fi = min(int(round(args.sec * G.FPS)), len(mats) - 1)
        b_sim = build_batch(scene, mats, trans_mp4, fi, tmp)
        b_real = build_batch(scene, mats, origin_mp4, fi, tmp)
        traj_sim, cache_sim = AP.run_cache(model, b_sim, sas)
        traj_real = AP.run(model, b_real)
        gap = float(np.linalg.norm(traj_real[:, :2] - traj_sim[:, :2]))
        d = dict(scene=sc, gap=gap, dpdm=pdm[sc]["transfered"] - pdm[sc]["origin"],
                 origin_total=pdm[sc]["origin"], rec=None)
        if gap >= AP.GAP_MIN:
            r = []
            for L in range(nL):
                tp = AP.run_patch(model, b_real, sas, L, cache_sim)
                r.append(1 - float(np.linalg.norm(tp[:, :2] - traj_sim[:, :2])) / gap)
            d["rec"] = r
        recs.append(d)
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(scenes)} ...")

    # ---- 构造预测量 ----
    allg = [d for d in recs]
    used = [d for d in recs if d["rec"] is not None]
    print(f"\n>> 有效(gap>{AP.GAP_MIN}): {len(used)}/{len(allg)}")

    def deep_share(r):
        p = np.clip(r, 0, None); s = p.sum()
        return float(p[4:7].sum() / (s + 1e-9)) if s > 0 else 0.0

    # y: PDM 退化幅度 = -Δtotal? Δ=transfered-origin>0 表示 real 更差。用 Δ 作"退化量"(越大越退化)
    Y = np.array([d["dpdm"] for d in allg])
    Gap = np.array([d["gap"] for d in allg])

    preds_all = {"traj_gap": Gap}
    # 因果类预测量只在 used 上
    Yu = np.array([d["dpdm"] for d in used])
    Gu = np.array([d["gap"] for d in used])
    L6 = np.array([max(d["rec"][6], 0) for d in used])
    DS = np.array([deep_share(d["rec"]) for d in used])
    preds_used = {
        "L6_recovery(norm)": L6,
        "deep_L4-6_share(norm)": DS,
        "gap×deep_share(abs)": Gu * DS,
        "gap×L6_recovery(abs)": Gu * L6,
        "traj_gap(used)": Gu,
    }

    print("\n===== 相关性: 预测量 ↔ PDM Δtotal(t=1s, 越大=real越退化) =====")
    print(f"[全部 {len(allg)} 场景]")
    for k, x in preds_all.items():
        p, s = corr(x, Y)
        print(f"  {k:24s} Pearson={p:+.2f} Spearman={s:+.2f}")
    print(f"[gap>{AP.GAP_MIN} 的 {len(used)} 场景 · 因果类]")
    for k, x in preds_used.items():
        p, s = corr(x, Yu)
        print(f"  {k:24s} Pearson={p:+.2f} Spearman={s:+.2f}")

    # ---- CSV + 散点 ----
    with open(os.path.join(args.out, "causal_pdm.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scene", "traj_gap", "dpdm_t1", "origin_total_t1",
                                       "L6_rec", "deep_share", "gapxdeep", "gapxL6"])
        for d in allg:
            if d["rec"] is not None:
                r = d["rec"]; ds = deep_share(r); l6 = max(r[6], 0)
                w.writerow([d["scene"], f"{d['gap']:.3f}", f"{d['dpdm']:.3f}", f"{d['origin_total']:.3f}",
                            f"{l6:.3f}", f"{ds:.3f}", f"{d['gap']*ds:.3f}", f"{d['gap']*l6:.3f}"])
            else:
                w.writerow([d["scene"], f"{d['gap']:.3f}", f"{d['dpdm']:.3f}", f"{d['origin_total']:.3f}",
                            "", "", "", ""])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # 1) traj_gap vs Δpdm (全部)
    axes[0].scatter(Gap, Y, alpha=0.6, color="#39c")
    p, s = corr(Gap, Y)
    axes[0].set_xlabel("trajectory gap ||real-sim|| (m)"); axes[0].set_ylabel("PDM Δtotal (transfered-origin)")
    axes[0].set_title(f"overall shift ↔ PDM degrade\nPearson={p:+.2f} Spearman={s:+.2f} (n={len(allg)})")
    axes[0].grid(True, ls=":", alpha=0.5)
    # 2) gap×deep_share vs Δpdm
    x2 = Gu * DS; p2, s2 = corr(x2, Yu)
    axes[1].scatter(x2, Yu, alpha=0.6, color="#c33")
    axes[1].set_xlabel("gap × deep(L4-6) recovery share (abs)"); axes[1].set_ylabel("PDM Δtotal")
    axes[1].set_title(f"deep-causal magnitude ↔ PDM degrade\nPearson={p2:+.2f} Spearman={s2:+.2f} (n={len(used)})")
    axes[1].grid(True, ls=":", alpha=0.5)
    # 3) deep_share(norm) vs Δpdm
    p3, s3 = corr(DS, Yu)
    axes[2].scatter(DS, Yu, alpha=0.6, color="#7a3")
    axes[2].set_xlabel("deep(L4-6) recovery share (normalized locus)"); axes[2].set_ylabel("PDM Δtotal")
    axes[2].set_title(f"causal LOCUS ↔ PDM degrade\nPearson={p3:+.2f} Spearman={s3:+.2f} (n={len(used)})")
    axes[2].grid(True, ls=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "causal_pdm_scatter.png"), dpi=120); plt.close(fig)
    print(f"\n>> 产物: {args.out}")


if __name__ == "__main__":
    main()
