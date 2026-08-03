#!/usr/bin/env python
"""
Ghosthead: real(origin) 退化场景下的 encoder self-attention 分布差异分析。

对 real 退化最狠的 top-K 场景，抓 DiffusionDrive 编码器 8 层 encoder_selfatt，
比较 sim(transfered) vs real(origin) 输入下的注意力分布：
  - 逐层 JS(sim‖real) 图像 key 重要度分布差异 -> 定位退化在哪层最强
  - 逐层图像 key vs lidar/BEV key 的注意力质量占比（sim vs real）
  - 逐层熵（注意力分散度）
  - 每场景 sim/real 各层图像注意力叠加到输入 RGB（2×8）
复用 scripts/build_bokeh_encoder_selfattn_rgb_analysis.py 的口径与
scripts/sim2real_analysis/capture_selfattn.py 的 patch/hook 逻辑。
"""
import os, sys, csv, argparse, tempfile
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import run_ghosthead_infer as G
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention

# token 布局 (transfuser_config): img 8x32=256, lidar 8x8=64, total 320
IMG_H, IMG_W = 8, 32
N_IMG = IMG_H * IMG_W        # 256
EPS = 1e-12


# ---------------- SelfAttention 抓取 patch ----------------
def patch_self_attention():
    def patched(self, x):
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach().cpu().numpy()
        y = self.resid_drop(self.proj((self.attn_drop(att) @ v).transpose(1, 2).contiguous().view(b, t, c)))
        return y
    SelfAttention.forward = patched


# ---------------- 分布度量 ----------------
def _prob(a):
    p = np.clip(a.astype(np.float64).reshape(-1), 0, None)
    s = p.sum()
    return p / s if s > 0 else np.ones_like(p) / len(p)


def kl(p, q):
    return float(np.sum(p * (np.log(p + EPS) - np.log(q + EPS))))


def js(a, b):
    p, q = _prob(a), _prob(b)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def entropy(a):
    p = _prob(a)
    return float(-np.sum(p * np.log(p + EPS)))


def layer_stats(att320):
    """(320,320) -> 图像 key 重要度(8x32) / 熵 / img|lid 质量占比。"""
    img_img = att320[:N_IMG, :N_IMG]
    img_lid = att320[:N_IMG, N_IMG:]
    key_imp = img_img.mean(axis=0)                  # (256,)
    img_mass = float(img_img.sum(axis=1).mean())
    lid_mass = float(img_lid.sum(axis=1).mean())
    return dict(key_imp=key_imp, grid=key_imp.reshape(IMG_H, IMG_W),
                entropy=entropy(key_imp),
                img_share=img_mass / (img_mass + lid_mass + EPS))


def head_mean(att):
    a = att
    if a.ndim == 4:
        a = a[0]
    a = a.mean(axis=0)          # over heads -> (320,320)
    return a


# ---------------- 前向 + 抓 8 层 ----------------
def capture_layers(agent, model, cam, status):
    dev = next(agent.parameters()).device
    with torch.no_grad():
        _ = model({'camera_feature': cam.to(dev), 'status_feature': status.to(dev)})
    sa = [m for m in model._backbone.modules() if isinstance(m, SelfAttention) and hasattr(m, 'last_att')]
    return [head_mean(m.last_att) for m in sa]        # list of 8 (320,320)


def build_input(scene, mats, mp4, fi):
    tmp = build_input._tmp
    raw = os.path.join(tmp, "r.png")
    G.extract_frame(mp4, fi, raw)
    img = cv2.cvtColor(cv2.imread(raw), cv2.COLOR_BGR2RGB)
    crop = G.crop_4to1_no_sky(img)
    cam = G.build_camera_feature(crop)
    v, a = G.ego_status_at(mats, fi)
    status = torch.from_numpy(np.concatenate([G.DRIVING_COMMAND, v, a])).float().unsqueeze(0)
    return cam, status, crop
build_input._tmp = tempfile.mkdtemp()


# ---------------- RGB 叠加 ----------------
def make_overlay(rgb, grid):
    g = grid.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + EPS)
    h, w = rgb.shape[:2]
    heat = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = (heat * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (0.55 * rgb + 0.45 * heat).astype(np.uint8)


# ---------------- 选场景 ----------------
def select_degraded_scenes(summary_csv, k):
    rows = list(csv.DictReader(open(summary_csv)))
    per = {}
    for r in rows:
        if float(r['gt_coverage_s']) <= 0:
            continue
        per.setdefault(r['scene'], {'transfered': [], 'origin': []})[r['source']].append(float(r['total']))
    gaps = []
    for sc, d in per.items():
        if d['transfered'] and d['origin']:
            gap = np.mean(d['transfered']) - np.mean(d['origin'])
            gaps.append((gap, sc, np.mean(d['transfered']), np.mean(d['origin'])))
    gaps.sort(reverse=True)
    return gaps[:k]


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6, help="取 real 退化最狠的前 K 个场景")
    ap.add_argument("--frames", default="1,2,3", help="用于统计的帧秒(cov>0)")
    ap.add_argument("--overlay_frame", type=int, default=1)
    ap.add_argument("--scenes", default="", help="逗号分隔手动指定场景(覆盖 --k)")
    ap.add_argument("--out", default="/data/ruolin/uwm/outputs/ghosthead_infer/attn_diff")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")]

    summ = "/data/ruolin/uwm/outputs/ghosthead_infer/_summary_all.csv"
    if args.scenes:
        scenes = [(None, s, None, None) for s in args.scenes.split(",")]
    else:
        scenes = select_degraded_scenes(summ, args.k)
    print(">> real 退化 top 场景 (gap=transfered-origin total):")
    for gap, sc, mt, mo in scenes:
        print(f"   {sc:22s} gap={gap:+.3f}  sim={mt:.2f} real={mo:.2f}" if gap is not None else f"   {sc}")

    patch_self_attention()
    print(">> 加载 agent ...")
    agent = G.load_agent()
    model = agent._transfuser_model

    # 收集：per scene -> per source -> per layer stats(对 frames 求平均)
    NL = 8
    js_by_scene = {}          # scene -> (8,) JS(sim‖real) per layer
    share_sim = {}; share_real = {}; ent_sim = {}; ent_real = {}
    rows_out = []

    for _, sc, _, _ in scenes:
        trans_dir = f"{G.DATA_ROOT}/renders/{sc}"
        trans_mp4 = f"{trans_dir}/frames.mp4"
        origin_mp4 = f"{G.DATA_ROOT}/ghosthead_result/{sc}/seg1p0/{sc}_seg1p0.mp4"
        if not (os.path.exists(trans_mp4) and os.path.exists(origin_mp4)):
            print(f"   [skip] {sc} 缺件"); continue
        scene = G.load_scene(trans_dir); mats = G.ego_mats(scene); n = len(mats)

        # 累加各 frame 的 grid / share / entropy
        acc = {'transfered': [np.zeros((IMG_H, IMG_W)) for _ in range(NL)],
               'origin': [np.zeros((IMG_H, IMG_W)) for _ in range(NL)]}
        acc_share = {'transfered': np.zeros(NL), 'origin': np.zeros(NL)}
        acc_ent = {'transfered': np.zeros(NL), 'origin': np.zeros(NL)}
        cnt = 0
        overlay_crop = {}
        for sec in frames:
            fi = min(int(round(sec * G.FPS)), n - 1)
            for src, mp4 in [('transfered', trans_mp4), ('origin', origin_mp4)]:
                cam, status, crop = build_input(scene, mats, mp4, fi)
                layers = capture_layers(agent, model, cam, status)
                for L in range(NL):
                    st = layer_stats(layers[L])
                    acc[src][L] += st['grid']
                    acc_share[src][L] += st['img_share']
                    acc_ent[src][L] += st['entropy']
                if sec == args.overlay_frame:
                    overlay_crop[src] = (crop, [layer_stats(layers[L])['grid'] for L in range(NL)])
            cnt += 1

        for src in ('transfered', 'origin'):
            for L in range(NL):
                acc[src][L] /= cnt
            acc_share[src] /= cnt
            acc_ent[src] /= cnt

        jsv = np.array([js(acc['transfered'][L], acc['origin'][L]) for L in range(NL)])
        js_by_scene[sc] = jsv
        share_sim[sc] = acc_share['transfered']; share_real[sc] = acc_share['origin']
        ent_sim[sc] = acc_ent['transfered']; ent_real[sc] = acc_ent['origin']
        for L in range(NL):
            rows_out.append([sc, L, f"{jsv[L]:.5f}", f"{acc_ent['transfered'][L]:.4f}",
                             f"{acc_ent['origin'][L]:.4f}", f"{acc_share['transfered'][L]:.4f}",
                             f"{acc_share['origin'][L]:.4f}"])
        print(f"   {sc}: argmax JS layer = L{int(np.argmax(jsv))} (JS={jsv.max():.4f})")

        # ---- 每场景 RGB 叠加 2x8
        if overlay_crop:
            fig, axes = plt.subplots(2, NL, figsize=(NL * 2.4, 4.2))
            for row, src in enumerate(['transfered', 'origin']):
                crop, grids = overlay_crop[src]
                for L in range(NL):
                    ov = make_overlay(crop, grids[L])
                    axes[row][L].imshow(ov); axes[row][L].axis('off')
                    if row == 0:
                        axes[row][L].set_title(f"L{L}", fontsize=9)
                axes[row][0].set_ylabel(src, fontsize=9)
            fig.suptitle(f"{sc}  encoder self-attn on RGB  (top=transfered/sim, bottom=origin/real)  t={args.overlay_frame}s",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(os.path.join(args.out, f"overlay_{sc}.png"), dpi=110)
            plt.close(fig)

    # ---------------- 汇总图1: 逐层 JS ----------------
    scenes_ok = list(js_by_scene.keys())
    if not scenes_ok:
        print("!! 无有效场景"); return
    JS = np.stack([js_by_scene[s] for s in scenes_ok])       # (K,8)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, s in enumerate(scenes_ok):
        ax.plot(range(NL), js_by_scene[s], color='gray', alpha=0.4, lw=1)
    ax.plot(range(NL), JS.mean(0), '-o', color='#c33', lw=2.5, label='mean over scenes')
    ax.set_xlabel('encoder self-attn layer'); ax.set_ylabel('JS( sim ‖ real )  image key-importance')
    ax.set_title(f'Per-layer attention divergence sim vs real (top-{len(scenes_ok)} real-degraded scenes)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "attn_layer_js.png"), dpi=130); plt.close(fig)

    # ---------------- 汇总图2: 图像注意力占比 + 熵 ----------------
    SS = np.stack([share_sim[s] for s in scenes_ok]).mean(0)
    SR = np.stack([share_real[s] for s in scenes_ok]).mean(0)
    ES = np.stack([ent_sim[s] for s in scenes_ok]).mean(0)
    ER = np.stack([ent_real[s] for s in scenes_ok]).mean(0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(range(NL), SS, '-o', label='sim(transfered)', color='#1a7')
    a1.plot(range(NL), SR, '-o', label='real(origin)', color='#c33')
    a1.set_title('image-key attention share (vs lidar/BEV)'); a1.set_xlabel('layer'); a1.set_ylabel('img share')
    a1.grid(True, ls=':', alpha=0.5); a1.legend()
    a2.plot(range(NL), ES, '-o', label='sim(transfered)', color='#1a7')
    a2.plot(range(NL), ER, '-o', label='real(origin)', color='#c33')
    a2.set_title('image key-importance entropy (spread)'); a2.set_xlabel('layer'); a2.set_ylabel('entropy')
    a2.grid(True, ls=':', alpha=0.5); a2.legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "attn_share_entropy.png"), dpi=130); plt.close(fig)

    # ---------------- CSV ----------------
    with open(os.path.join(args.out, "attn_diff_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "layer", "js_sim_real", "entropy_sim", "entropy_real", "img_share_sim", "img_share_real"])
        w.writerows(rows_out)

    ml = int(np.argmax(JS.mean(0)))
    print(f"\n>> 逐层平均 JS 峰值层 = L{ml} (JS={JS.mean(0)[ml]:.4f})")
    print(f">> 图像注意力占比 sim均值={SS.mean():.3f} real均值={SR.mean():.3f}")
    print(f">> 产物: {args.out}")


if __name__ == "__main__":
    main()
