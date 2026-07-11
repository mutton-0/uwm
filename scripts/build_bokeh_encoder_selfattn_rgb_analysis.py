"""
逐层编码器自注意力（encoder_selfatt_0..7）+ RGB 叠加交互分析。

针对缓存中含完整编码器自注意力权重的 2 个 token
(aa96f52b95b155e7, ca9e7281adce5212)，对 origin / transfer 两种输入：

1) 用统计折线图（Bokeh 交互）画出后几个（全部 8 个）transformer 层的注意力分布，
   以及每层 origin↔transfer 的差异（KL/JS/Wasserstein）随层号的变化；
2) 拆分 “ego / agent” 注意力：
   - 编码器内：图像 keys(256) vs lidar/BEV keys(64) 的注意力质量占比随层变化；
   - 解码器：diff_decoder 每层 cross_ego(1) / cross_agent(30) / cross_bev(8) 三部分分布；
3) 把每个 transformer 层的图像注意力(8x32)上采样叠到输入 RGB 上，origin/transfer 并排；
4) 文字自动给出差异最大的几个层并简析原因。

编码器自注意力权重信息取自：
  /data/ruolin_a6k/SimScale/outputs/analysis_cache_encoder/{token}_{mode}.pkl  (attn['encoder_selfatt_*'])
输入 RGB 缓存（首次运行时用 feature builder 重建，仅特征构建不跑前向）：
  /data/ruolin_a6k/SimScale/outputs/analysis_cache_encoder/rgb_inputs.npz
"""

import base64
import os
import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy.stats import wasserstein_distance

from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource, Div,
                          HoverTool, LinearColorMapper, Panel, Tabs)
from bokeh.palettes import Category10, Viridis256
from bokeh.plotting import figure

# ------------------ Paths & constants ------------------
ROOT = Path('/data/ruolin_a6k/SimScale')
CACHE_DIR = ROOT / 'outputs/analysis_cache_encoder'
RGB_NPZ = CACHE_DIR / 'rgb_inputs.npz'
HTML_OUT = ROOT / 'outputs/bokeh_encoder_selfattn_rgb_analysis.html'

TRANSFER_DIR = ROOT / 'outputs/my_diffusion_0_transfer2sim_scenarios'
TRANSFER_SUFFIX = 'after_transfer.jpg'

TRANS_CSV = Path('/data/ruolin_a6k/navsim_logs/exp_transferred_all/test_my_diffusiondrive_transferred_all/2026.06.30.10.33.55/final_pdm_after_transfer_2026.06.30.10.34.14.csv')
ORIG_CSV = Path('/data/ruolin_a6k/navsim_logs/exp_zero_cams/test_my_diffusiondrive_zero_cams/2026.06.30.10.48.53/final_pdm_origin_2026.06.30.10.49.12.csv')

# 只有这 2 个 token 缓存了完整的 encoder_selfatt 权重
TOKENS = ['aa96f52b95b155e7', 'ca9e7281adce5212']

# token 构成 (transfuser_config): img 8x32=256, lidar 8x8=64, total 320
IMG_H, IMG_W = 8, 32
N_IMG = IMG_H * IMG_W          # 256
LID_H, LID_W = 8, 8
N_LID = LID_H * LID_W          # 64

C10 = Category10[10]


# ------------------ divergence utils ------------------
def _prob(a: np.ndarray) -> np.ndarray:
    p = a.astype(np.float64).reshape(-1)
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def kl_div(p, q):
    p, q = _prob(p), _prob(q)
    return float(np.sum(p * np.log(p / q)))


def js_div(p, q):
    p, q = _prob(p), _prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def wass_div(p, q):
    p, q = _prob(p), _prob(q)
    x = np.arange(len(p), dtype=np.float64)
    return float(wasserstein_distance(x, x, u_weights=p, v_weights=q))


def entropy(p):
    p = _prob(p)
    return float(-np.sum(p * np.log(p)))


# ------------------ cache loading ------------------
def load_attn(token: str, mode: str) -> Dict[str, List[np.ndarray]]:
    with open(CACHE_DIR / f'{token}_{mode}.pkl', 'rb') as f:
        return pickle.load(f)['attn']


def selfatt_head_mean(arr: np.ndarray) -> np.ndarray:
    """(1,H,320,320) -> (320,320) 对 batch/heads 取平均."""
    a = arr
    if a.ndim == 4:
        a = a.mean(axis=1)  # over heads
    if a.ndim == 3:
        a = a.mean(axis=0)  # over batch
    return a


def selfatt_per_head(arr: np.ndarray) -> np.ndarray:
    """(1,H,320,320) / (H,320,320) -> (H,320,320) 保留每个 head，不做平均."""
    a = arr
    if a.ndim == 4:
        a = a[0]            # 去 batch -> (H,320,320)
    if a.ndim == 2:
        a = a[None]         # 单 head 兜底 -> (1,320,320)
    return a


def head_key_imp(att320: np.ndarray) -> np.ndarray:
    """从单个 head 的 (320,320) 抽取图像 key 被关注程度 (256,)."""
    img_img = att320[:N_IMG, :N_IMG]
    return img_img.mean(axis=0)


def merge_calls(arrs: List[np.ndarray]) -> np.ndarray:
    """多次前向捕获求平均 (diff decoder n=2)."""
    if len(arrs) == 1:
        return arrs[0]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def encoder_layer_keys(attn: Dict) -> List[str]:
    keys = [k for k in attn if k.startswith('encoder_selfatt_')]
    return sorted(keys, key=lambda k: int(k.rsplit('_', 1)[-1]))


# ------------------ per-layer statistics ------------------
def layer_stats(att320: np.ndarray) -> Dict:
    """从 (320,320) 自注意力矩阵抽取分析量."""
    img_img = att320[:N_IMG, :N_IMG]          # 图像 query -> 图像 key
    img_lid = att320[:N_IMG, N_IMG:]          # 图像 query -> lidar/BEV key

    key_imp = img_img.mean(axis=0)            # (256,) 每个图像 key 位置被关注程度
    key_imp_grid = key_imp.reshape(IMG_H, IMG_W)

    # 图像 query 的注意力质量在 image / lidar 上的划分（每行 sum=1）
    img_mass = float(img_img.sum(axis=1).mean())
    lid_mass = float(img_lid.sum(axis=1).mean())
    tot = img_mass + lid_mass + 1e-12

    return {
        'key_imp': key_imp,                        # (256,)
        'key_imp_grid': key_imp_grid,              # (8,32)
        'col_marg': key_imp_grid.sum(axis=0),      # (32,) 水平（全景宽）方向边缘分布
        'row_marg': key_imp_grid.sum(axis=1),      # (8,) 垂直方向
        'img_frac': img_mass / tot,                # 图像 key 注意力占比
        'lid_frac': lid_mass / tot,                # lidar/BEV key 注意力占比
        'entropy': entropy(key_imp),               # 图像 key 分布熵（越大越分散）
    }


# ------------------ RGB reconstruction (guarded) ------------------
def ensure_rgb_cache() -> Dict[str, np.ndarray]:
    if RGB_NPZ.exists():
        npz = np.load(RGB_NPZ)
        return {k: npz[k] for k in npz.files}

    print('[rgb] cache miss -> rebuilding camera_feature via feature builders ...')
    os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
    os.environ.setdefault('NUPLAN_MAPS_ROOT', '/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps')
    os.environ.setdefault('OPENSCENE_DATA_ROOT', '/data/Yuhao/world_model_yhl/navsim_workspace/dataset')
    os.environ.setdefault('NAVSIM_DEVKIT_ROOT', str(ROOT))

    import hydra
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneFilter, SceneLoader

    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        config_dir=str((ROOT / 'navsim/planning/script/config/pdm_scoring').resolve()),
        version_base=None,
    )
    cfg = hydra.compose(config_name='default_run_pdm_score_gpu',
                        overrides=['agent=my_diffusiondrive', 'train_test_split=navmini'])
    agent = instantiate(cfg.agent)
    agent.initialize()

    root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    loader = SceneLoader(
        original_sensor_path=root / 'sensor_blobs/mini',
        data_path=root / 'navsim_logs/mini',
        scene_filter=SceneFilter(tokens=TOKENS),
        sensor_config=agent.get_sensor_config(),
    )

    out = {}
    for token in TOKENS:
        agent_input = loader.get_agent_input_from_token(token)
        for mode, use_transfer in [('origin', False), ('transfer', True)]:
            if use_transfer:
                os.environ['TRANSFER_SCENARIO_IMAGE_DIR'] = str(TRANSFER_DIR)
                os.environ['TRANSFER_IMAGE_SUFFIX'] = TRANSFER_SUFFIX
                os.environ['NAVSIM_CURRENT_TOKEN'] = token
            else:
                os.environ.pop('TRANSFER_SCENARIO_IMAGE_DIR', None)
                os.environ.pop('TRANSFER_IMAGE_SUFFIX', None)
                os.environ.pop('NAVSIM_CURRENT_TOKEN', None)

            feat = {}
            for b in agent.get_feature_builders():
                feat.update(b.compute_features(agent_input))
            img = feat['camera_feature'].detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
            out[f'{token}_{mode}'] = img
            print(f'[rgb] {token}_{mode} -> {img.shape}')

    np.savez_compressed(RGB_NPZ, **out)
    print(f'[rgb] saved {RGB_NPZ}')
    return out


# ------------------ overlay -> base64 png ------------------
def make_overlay(rgb: np.ndarray, grid: np.ndarray) -> np.ndarray:
    g = grid.astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    heat = cv2.resize(g, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (0.55 * rgb + 0.45 * colored).astype(np.uint8)


def png_b64(img: np.ndarray, max_w: int = 960) -> str:
    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode('ascii')


def img_tag(img: np.ndarray, w: int = 460, title: str = '') -> str:
    b = png_b64(img)
    cap = f'<div style="font-size:12px;color:#555;text-align:center">{title}</div>' if title else ''
    return (f'<div style="display:inline-block;margin:4px;text-align:center">'
            f'<img src="data:image/png;base64,{b}" style="width:{w}px;border:1px solid #ccc"/>{cap}</div>')


# ------------------ score info ------------------
def get_score_info() -> Dict[str, str]:
    try:
        import pandas as pd
        a = pd.read_csv(TRANS_CSV)
        b = pd.read_csv(ORIG_CSV)
        a = a[a['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'transfer'})
        b = b[b['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'origin'})
        m = a.merge(b, on='token')
        m['delta'] = m['transfer'] - m['origin']
        return {r['token']: f"Origin={r['origin']:.4f}, Transfer={r['transfer']:.4f}, Delta={r['delta']:+.4f}"
                for _, r in m.iterrows()}
    except Exception:
        return {}


# ------------------ line chart helper ------------------
def line_chart(title, x, xlabels, series, width=560, height=300, ytt='0.0000'):
    data = {'x': list(range(len(x))), 'label': xlabels}
    for name, ys in series.items():
        data[name] = list(ys)
    src = ColumnDataSource(data=data)
    p = figure(width=width, height=height, title=title,
               tools='pan,wheel_zoom,box_zoom,reset,save')
    tips = [('x', '@label')]
    for i, (name, _) in enumerate(series.items()):
        col = C10[i % 10]
        p.line('x', name, source=src, color=col, line_width=2, legend_label=name)
        p.circle('x', name, source=src, color=col, size=5)
        tips.append((name, f'@{name}{{{ytt}}}'))
    p.xaxis.ticker = list(range(len(x)))
    p.xaxis.major_label_overrides = {i: str(xlabels[i]) for i in range(len(x))}
    p.xaxis.major_label_orientation = 0.9
    p.add_tools(HoverTool(tooltips=tips))
    p.legend.click_policy = 'hide'
    p.legend.label_text_font_size = '9px'
    return p


# ------------------ head x layer heatmap ------------------
def head_layer_heatmap(title, js_lh, layer_idx, width=430, height=320):
    """js_lh: (n_layer, n_head) 的 origin↔transfer 每 head JS 散度热力图."""
    n_layer, n_head = js_lh.shape
    xs, ys, vals = [], [], []
    for li in range(n_layer):
        for h in range(n_head):
            xs.append(h)
            ys.append(layer_idx[li])
            vals.append(float(js_lh[li, h]))
    src = ColumnDataSource(dict(head=xs, layer=ys, js=vals))
    mapper = LinearColorMapper(palette=Viridis256,
                               low=min(vals), high=max(vals) + 1e-12)
    p = figure(width=width, height=height, title=title,
               x_range=(-0.5, n_head - 0.5),
               y_range=(min(layer_idx) - 0.5, max(layer_idx) + 0.5),
               tools='save',
               tooltips=[('layer', '@layer'), ('head', '@head'), ('JS', '@js{0.0000}')])
    p.rect(x='head', y='layer', width=1, height=1, source=src,
           fill_color={'field': 'js', 'transform': mapper}, line_color='white')
    p.xaxis.axis_label = 'head'
    p.yaxis.axis_label = 'encoder_selfatt layer'
    p.xaxis.ticker = list(range(n_head))
    p.yaxis.ticker = layer_idx
    p.add_layout(ColorBar(color_mapper=mapper,
                          ticker=BasicTicker(desired_num_ticks=5)), 'right')
    return p


# ------------------ decoder ego/agent panel ------------------
def decoder_panels(token, attn_o, attn_t):
    plots = []
    txt = []
    for li in [0, 1]:
        for part, n in [('agent', 30), ('bev', 8), ('ego', 1)]:
            ko = f'diff_layer{li}_cross_{part}_attn'
            if ko not in attn_o or ko not in attn_t:
                continue
            A = merge_calls(attn_o[ko]).reshape(-1, n)  # (20,n)
            B = merge_calls(attn_t[ko]).reshape(-1, n)
            prof_o = A.mean(axis=0)
            prof_t = B.mean(axis=0)
            js = js_div(prof_o, prof_t)
            txt.append(f'diff_layer{li} · cross_{part}: JS={js:.4f}')
            if n == 1:
                continue  # ego 只有 1 维，无分布可画
            p = line_chart(
                f'{token} | decoder L{li} cross_{part} 分布 (JS={js:.3f})',
                list(range(n)), [f'{part[0]}{i}' for i in range(n)],
                {'origin': prof_o, 'transfer': prof_t},
                width=430, height=250)
            plots.append(p)
    return plots, txt


# ------------------ per-token tab ------------------
def make_tab(token, rgb_cache, score_info) -> Panel:
    attn_o = load_attn(token, 'origin')
    attn_t = load_attn(token, 'transfer')
    lkeys = encoder_layer_keys(attn_o)
    n_layer = len(lkeys)

    stats_o, stats_t = [], []
    for k in lkeys:
        stats_o.append(layer_stats(selfatt_head_mean(merge_calls(attn_o[k]))))
        stats_t.append(layer_stats(selfatt_head_mean(merge_calls(attn_t[k]))))

    layer_idx = [int(k.rsplit('_', 1)[-1]) for k in lkeys]

    # --- 1) 每层 origin↔transfer 差异（图像 key 分布） ---
    kl = [kl_div(so['key_imp'], st['key_imp']) for so, st in zip(stats_o, stats_t)]
    js = [js_div(so['key_imp'], st['key_imp']) for so, st in zip(stats_o, stats_t)]
    wa = [wass_div(so['key_imp'], st['key_imp']) for so, st in zip(stats_o, stats_t)]
    p_div = line_chart(f'{token} | 各 transformer 层 注意力分布差异 (origin vs transfer)',
                       layer_idx, [f'L{i}' for i in layer_idx],
                       {'KL': kl, 'JS': js, 'Wasserstein': wa})

    # --- 1b) head 级差异：每层保留每个 head，分别算图像 key 分布 JS ---
    kimp_o_h, kimp_t_h = [], []      # 每层: (n_head, 256)
    for k in lkeys:
        ao = selfatt_per_head(merge_calls(attn_o[k]))
        at = selfatt_per_head(merge_calls(attn_t[k]))
        kimp_o_h.append(np.stack([head_key_imp(ao[h]) for h in range(ao.shape[0])]))
        kimp_t_h.append(np.stack([head_key_imp(at[h]) for h in range(at.shape[0])]))
    n_head = kimp_o_h[0].shape[0]
    js_lh = np.array([[js_div(kimp_o_h[li][h], kimp_t_h[li][h])
                       for h in range(n_head)] for li in range(n_layer)])  # (n_layer,n_head)

    # per-head JS 随层折线（每个 head 一条 + 已对 head 平均的 JS 作对照）
    p_head_js = line_chart(
        f'{token} | 各 head 的图像 key 分布 JS（含 head 平均对照）',
        layer_idx, [f'L{i}' for i in layer_idx],
        {**{f'head{h}': js_lh[:, h] for h in range(n_head)},
         'head_avg(旧)': js},
        width=560, height=300)
    p_head_hm = head_layer_heatmap(
        f'{token} | JS 热力图 (layer × head)', js_lh, layer_idx)

    # 集中度：每层 max_head / mean_head，判断差异是否被某个 head 主导
    head_max = js_lh.max(axis=1)
    head_mean = js_lh.mean(axis=1)
    concen = head_max / (head_mean + 1e-12)         # (n_layer,) 越大越集中于单 head
    li_star, h_star = np.unravel_index(int(np.argmax(js_lh)), js_lh.shape)
    p_concen = line_chart(
        f'{token} | 每层 head 差异集中度 (max/mean，>~2 表示单 head 主导)',
        layer_idx, [f'L{i}' for i in layer_idx],
        {'concentration': concen}, width=560, height=240, ytt='0.00')

    # --- 2) ego/agent 划分：图像 keys vs lidar/BEV keys 注意力质量占比 ---
    p_split = line_chart(f'{token} | 图像 keys vs lidar/BEV keys 注意力占比（图像 query）',
                         layer_idx, [f'L{i}' for i in layer_idx],
                         {'origin_image': [s['img_frac'] for s in stats_o],
                          'origin_bev': [s['lid_frac'] for s in stats_o],
                          'transfer_image': [s['img_frac'] for s in stats_t],
                          'transfer_bev': [s['lid_frac'] for s in stats_t]},
                         ytt='0.000')

    # --- 3) 每层注意力熵（分散/聚焦程度） ---
    p_ent = line_chart(f'{token} | 各层图像注意力熵 (越大越分散)',
                       layer_idx, [f'L{i}' for i in layer_idx],
                       {'origin': [s['entropy'] for s in stats_o],
                        'transfer': [s['entropy'] for s in stats_t]},
                       ytt='0.000')

    # --- 4) 每层水平方向(全景宽 32)注意力边缘分布 小多图 ---
    col_plots = []
    for i, (so, st) in enumerate(zip(stats_o, stats_t)):
        p = line_chart(f'L{layer_idx[i]} 水平注意力分布',
                       list(range(IMG_W)), [f'c{j}' for j in range(IMG_W)],
                       {'origin': so['col_marg'], 'transfer': st['col_marg']},
                       width=300, height=190, ytt='0.000')
        p.legend.visible = False
        col_plots.append(p)
    col_grid = gridplot(col_plots, ncols=4)

    # --- 5) decoder ego/agent/bev ---
    dec_plots, dec_txt = decoder_panels(token, attn_o, attn_t)
    dec_grid = gridplot(dec_plots, ncols=2) if dec_plots else Div(text='(无 decoder attn)')

    # --- 6) RGB 叠加：每层 origin/transfer 并排 ---
    rgb_o = rgb_cache[f'{token}_origin']
    rgb_t = rgb_cache[f'{token}_transfer']
    base_html = ('<h3>输入 RGB</h3>'
                 + img_tag(rgb_o, 460, 'origin 输入')
                 + img_tag(rgb_t, 460, 'transfer 输入'))
    overlay_rows = ['<h3>各 transformer 层 图像注意力 叠加到 RGB（左 origin / 右 transfer）</h3>']
    for i, (so, st) in enumerate(zip(stats_o, stats_t)):
        ov_o = make_overlay(rgb_o, so['key_imp_grid'])
        ov_t = make_overlay(rgb_t, st['key_imp_grid'])
        overlay_rows.append(
            f'<div style="border-top:1px solid #eee;padding-top:6px">'
            f'<b>encoder_selfatt_{layer_idx[i]}</b> &nbsp; '
            f'JS(origin,transfer)={js[i]:.4f}, KL={kl[i]:.4f}<br>'
            + img_tag(ov_o, 460) + img_tag(ov_t, 460) + '</div>')
    rgb_div = Div(text=base_html + ''.join(overlay_rows), width=1000)

    # --- 6b) 最敏感层的逐 head 注意力叠加 RGB（定位到具体 head） ---
    head_rows = [f'<h3>最敏感层 L{layer_idx[li_star]} 的逐 head 图像注意力叠加 RGB'
                 f'（左 origin / 右 transfer，★=JS 最大 head{h_star}）</h3>']
    for h in range(n_head):
        go = kimp_o_h[li_star][h].reshape(IMG_H, IMG_W)
        gt = kimp_t_h[li_star][h].reshape(IMG_H, IMG_W)
        star = ' ★' if h == h_star else ''
        head_rows.append(
            f'<div style="border-top:1px solid #eee;padding-top:6px">'
            f'<b>head{h}{star}</b> &nbsp; JS(origin,transfer)={js_lh[li_star, h]:.4f}<br>'
            + img_tag(make_overlay(rgb_o, go), 460)
            + img_tag(make_overlay(rgb_t, gt), 460) + '</div>')
    head_rgb_div = Div(text=''.join(head_rows), width=1000)

    # --- 文字分析：差异最大的层 ---
    order = np.argsort(js)[::-1]
    top = order[:3]
    lines = [f'<b>{token}</b> &nbsp; {score_info.get(token, "")}',
             f'注意力矩阵 320 keys = 图像 {N_IMG}(8×32) + lidar/BEV {N_LID}(8×8)；共 {n_layer} 个编码器自注意力层，每层 4 heads（已对 heads 平均）。',
             '<b>差异最大的层（按图像 key 分布 JS 散度排序）：</b>']
    for rank, i in enumerate(top, 1):
        dfrac = (stats_t[i]['img_frac'] - stats_o[i]['img_frac'])
        dent = (stats_t[i]['entropy'] - stats_o[i]['entropy'])
        reason = []
        if abs(dfrac) > 0.02:
            reason.append(f"图像 key 注意力占比 {'上升' if dfrac > 0 else '下降'} {abs(dfrac)*100:.1f}%（transfer 后感知/BEV 关注比例改变）")
        if abs(dent) > 0.05:
            reason.append(f"注意力熵{'增大→更分散' if dent > 0 else '减小→更聚焦'}({dent:+.3f})")
        if not reason:
            reason.append('分布形态整体平移，占比/熵变化较小')
        lines.append(f'{rank}. <b>L{layer_idx[i]}</b>: JS={js[i]:.4f}, KL={kl[i]:.4f}, Wass={wa[i]:.3f} — ' + '；'.join(reason))
    lines.append('原因概述：transfer 改写输入图像的纹理/风格后，越深的融合层（后几层）语义整合越充分，'
                 '风格差异会被放大或抑制；JS 高的层说明该层对输入图像变化最敏感，其在 RGB 上的热点迁移最明显。')

    # head 级结论：差异是否集中在个别 head（决定 LoRA 掩码是否可下探到 head 粒度）
    hi = int(np.argmax(concen))
    lines.append('<b>head 级差异（新增，为 LoRA 掩码粒度提供依据）：</b>')
    lines.append(f'· 全局最敏感 head：<b>L{layer_idx[li_star]}·head{h_star}</b>，'
                 f'JS={js_lh[li_star, h_star]:.4f}（head 平均后仅 {js[li_star]:.4f}，'
                 f'平均掩掉了 {(1 - js[li_star] / (js_lh[li_star, h_star] + 1e-12)) * 100:.0f}% 的峰值差异）。')
    lines.append(f'· 集中度最高的层：<b>L{layer_idx[hi]}</b>，max/mean={concen[hi]:.2f}'
                 + ('（单 head 主导 → LoRA 可下探到 head 粒度，只解冻该 head 的 Q/K）'
                    if concen[hi] > 2.0 else
                    '（各 head 差异较均匀 → 该层宜整层适应，不宜按 head 稀疏）') + '。')
    lines.append('· 判读：集中度高的层，稀疏化到 head 粒度收益大；集中度接近 1 的层，'
                 '差异分散在所有 head，按 head mask 会漏信息，应保留整层低秩。')
    if dec_txt:
        lines.append('<b>decoder ego/agent/bev 差异：</b> ' + ' ｜ '.join(dec_txt))
    top_div = Div(text='<br>'.join(lines), width=1000)

    layout = column(
        top_div,
        p_div,
        Div(text='<h3>head 级差异（layer × head）——LoRA 掩码粒度依据</h3>', width=1000),
        gridplot([[p_head_js, p_head_hm]]),
        p_concen,
        p_split, p_ent,
        Div(text='<h3>各层 水平方向(全景宽32) 注意力边缘分布</h3>', width=1000),
        col_grid,
        Div(text='<h3>Decoder cross-attention: ego / agent / bev</h3>', width=1000),
        dec_grid,
        rgb_div,
        head_rgb_div,
    )
    return Panel(child=layout, title=token)


def main():
    np.random.seed(42)
    rgb_cache = ensure_rgb_cache()
    score_info = get_score_info()

    tabs = [make_tab(t, rgb_cache, score_info) for t in TOKENS]

    intro = Div(text='''
    <h2>编码器自注意力逐层分析 + RGB 叠加（Origin vs Transfer）</h2>
    <p>数据源：<code>outputs/analysis_cache_encoder/{token}_{mode}.pkl</code> 中的
    <code>encoder_selfatt_0..7</code>（8 层融合 transformer 自注意力，每层 4 heads，320 tokens = 图像256 + lidar/BEV64）。</p>
    <p>每个 token 一个 tab：1) 各层注意力分布差异曲线(KL/JS/Wasserstein)；1b) <b>head 级差异</b>（每 head 单独算 JS、
    layer×head 热力图、每层 max/mean 集中度）——用于判断 LoRA 掩码能否下探到 head 粒度；
    2) 图像 vs lidar/BEV 注意力占比(ego/agent 划分)；3) 各层注意力熵；4) 各层水平方向边缘分布小多图；
    5) decoder 的 ego/agent/bev cross-attention；6) 每层图像注意力叠加到输入 RGB + 最敏感层逐 head 叠加；
    并给出差异最大层与最敏感 head 的文字分析。</p>
    ''', width=1200)

    doc = column(intro, Tabs(tabs=tabs))
    output_file(str(HTML_OUT), title='Encoder SelfAttn RGB Analysis')
    save(doc)
    print(f'Saved HTML: {HTML_OUT}')


if __name__ == '__main__':
    main()
