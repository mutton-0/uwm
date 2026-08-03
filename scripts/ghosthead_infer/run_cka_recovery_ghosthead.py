#!/usr/bin/env python
"""
A: CKA-vs-recovery 三元对照（复用原 scripts/sim2real_analysis/cka_vs_recovery.py 内核）。
对每个 encoder self-attn 层比较：
  1) 注意力 JS(sim vs real)     —— 注意力级散度(症状)
  2) 1-CKA(sim_out vs real_out) —— 特征级散度 / PRH kernel 对齐
  3) recovery(L) 激活修补        —— 因果责任
看"表征对齐度(CKA)"是否比"注意力散度(JS)"更贴近因果。ghosthead: 参考=sim(transfered), 退化=real(origin)。
"""
import os, sys, csv, argparse, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import run_ghosthead_infer as G
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sim2real_analysis"))
import cka_vs_recovery as CKA           # 复用 linear_cka / js_div / run_cache / run_patch / run / sa_modules
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention

N_IMG = 256


def patch_attn():
    import torch.nn.functional as F
    def fwd(self, x):
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = F.softmax((q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1))), dim=-1)
        self.last_att = att.detach().cpu().numpy()
        return self.resid_drop(self.proj((self.attn_drop(att) @ v).transpose(1, 2).contiguous().view(b, t, c)))
    SelfAttention.forward = fwd


def build_batch(scene, mats, mp4, fi, tmp):
    import cv2
    raw = f"{tmp}/r.png"; G.extract_frame(mp4, fi, raw)
    img = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB)
    cam = G.build_camera_feature(G.crop_4to1_no_sky(img))
    v, a = G.ego_status_at(mats, fi)
    st = torch.from_numpy(np.concatenate([G.DRIVING_COMMAND, v, a])).float().unsqueeze(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {"camera_feature": cam.to(dev), "status_feature": st.to(dev)}


def key_imp(sas):
    out = []
    for m in sas:
        w = m.last_att[0].mean(0)                 # head 平均 (320,320)
        out.append(w[:N_IMG, :N_IMG].mean(0))     # (256,)
    return out


def select_scenes(csvp, k):
    rows = list(csv.DictReader(open(csvp)))
    per = {}
    for r in rows:
        if float(r["gt_coverage_s"]) <= 0:
            continue
        per.setdefault(r["scene"], {"transfered": [], "origin": []})[r["source"]].append(float(r["total"]))
    gaps = [(np.mean(d["transfered"]) - np.mean(d["origin"]), s)
            for s, d in per.items() if d["transfered"] and d["origin"]]
    gaps.sort(reverse=True)
    return [s for _, s in gaps[:k]]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--sec", type=int, default=1)
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer/cka_recovery")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp()

    scenes = select_scenes("/data/ruolin/uwm/outputs/ghosthead_infer/_summary_all.csv", args.k)
    patch_attn()
    agent = G.load_agent(); model = agent._transfuser_model.eval()
    sas = CKA.sa_modules(model); nL = len(sas)
    print(f">> CKA-vs-recovery: {len(scenes)} 场景, #self-attn={nL}")

    JS = []; OMC = []; REC = []          # per scene, per layer
    for sc in scenes:
        trans_mp4 = f"{G.DATA_ROOT}/renders/{sc}/frames.mp4"
        origin_mp4 = f"{G.DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"
        if not (os.path.exists(trans_mp4) and os.path.exists(origin_mp4)):
            continue
        scene = G.load_scene(f"{G.DATA_ROOT}/renders/{sc}"); mats = G.ego_mats(scene)
        fi = min(int(round(args.sec * G.FPS)), len(mats) - 1)
        b_sim = build_batch(scene, mats, trans_mp4, fi, tmp)
        b_real = build_batch(scene, mats, origin_mp4, fi, tmp)

        traj_sim, out_sim = CKA.run_cache(model, b_sim, sas); ki_sim = key_imp(sas)
        traj_real, out_real = CKA.run_cache(model, b_real, sas); ki_real = key_imp(sas)
        gap = float(np.linalg.norm(traj_real[:, :2] - traj_sim[:, :2]))
        if gap < CKA.SEED * 0 + 1.0:     # GAP_MIN=1.0
            print(f"   {sc}: gap={gap:.2f} 跳过"); continue

        js_l, omc_l, rec_l = [], [], []
        for L in range(nL):
            Xs = out_sim[L].reshape(-1, out_sim[L].shape[-1])
            Xr = out_real[L].reshape(-1, out_real[L].shape[-1])
            omc_l.append(1 - CKA.linear_cka(Xs, Xr))
            js_l.append(CKA.js_div(ki_sim[L], ki_real[L]))
            tp = CKA.run_patch(model, b_real, sas, L, out_sim)    # patch real 用 sim 输出
            rec_l.append(1 - float(np.linalg.norm(tp[:, :2] - traj_sim[:, :2])) / gap)
        JS.append(js_l); OMC.append(omc_l); REC.append(rec_l)
        print(f"   {sc}: gap={gap:.2f} argmax JS=L{int(np.argmax(js_l))} "
              f"1-CKA=L{int(np.argmax(omc_l))} recovery=L{int(np.argmax(rec_l))}")

    JS = np.array(JS); OMC = np.array(OMC); REC = np.array(REC)
    mJS, mOMC, mREC = JS.mean(0), OMC.mean(0), REC.mean(0)

    print(f"\n===== n={len(JS)} 场景, 各层均值 =====")
    print("layer   " + " ".join(f"L{i}" for i in range(nL)))
    print("JS      " + " ".join(f"{v:.2f}" for v in mJS))
    print("1-CKA   " + " ".join(f"{v:.2f}" for v in mOMC))
    print("recovery" + " ".join(f"{v:+.2f}" for v in mREC))
    # 相关性(逐层均值向量上)
    pj = np.corrcoef(mJS, mREC)[0, 1]; sj = spearman(mJS, mREC)
    pc = np.corrcoef(mOMC, mREC)[0, 1]; sc_ = spearman(mOMC, mREC)
    print(f"\nJS ↔ recovery:    Pearson={pj:+.2f} Spearman={sj:+.2f}")
    print(f"1-CKA ↔ recovery: Pearson={pc:+.2f} Spearman={sc_:+.2f}")
    print(f"argmax:  JS=L{int(np.argmax(mJS))}  1-CKA=L{int(np.argmax(mOMC))}  recovery=L{int(np.argmax(mREC))}")

    # CSV + 图
    with open(os.path.join(args.out, "cka_recovery.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["layer", "JS", "1-CKA", "recovery"])
        for L in range(nL):
            w.writerow([L, f"{mJS[L]:.4f}", f"{mOMC[L]:.4f}", f"{mREC[L]:.4f}"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(range(nL), norm(mJS), "-o", label="JS attn-divergence (symptom)", color="#e69")
    ax.plot(range(nL), norm(mOMC), "-o", label="1-CKA feature-divergence", color="#39c")
    ax.plot(range(nL), norm(mREC), "-o", label="recovery (cause)", color="#c33", lw=2.5)
    ax.set_xlabel("encoder self-attn layer"); ax.set_ylabel("normalized [0,1]")
    ax.set_title(f"JS vs 1-CKA vs causal recovery (n={len(JS)} real-degraded scenes)")
    ax.grid(True, ls=":", alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "cka_vs_recovery.png"), dpi=130); plt.close(fig)
    print(f">> 产物: {args.out}")


if __name__ == "__main__":
    main()
