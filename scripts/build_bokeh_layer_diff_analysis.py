import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE

from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Div, HoverTool, Tabs, Panel
from bokeh.palettes import Category10
from bokeh.plotting import figure

from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention

# ------------------ Paths & constants ------------------
ROOT = Path('/data/ruolin/SimScale')
CACHE_DIR = ROOT / 'outputs/analysis_cache_encoder'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HTML_OUT = ROOT / 'outputs/bokeh_diff_analysis_origin_vs_transfer.html'

TRANSFER_DIR = ROOT / 'outputs/my_diffusion_0_transfer2sim_scenarios'
TRANSFER_SUFFIX = 'after_transfer.jpg'

TRANS_CSV = Path('/data/ruolin/navsim_logs/exp_transferred_all/test_my_diffusiondrive_transferred_all/2026.06.30.10.33.55/final_pdm_after_transfer_2026.06.30.10.34.14.csv')
ORIG_CSV = Path('/data/ruolin/navsim_logs/exp_zero_cams/test_my_diffusiondrive_zero_cams/2026.06.30.10.48.53/final_pdm_origin_2026.06.30.10.49.12.csv')
HYDRA_CONFIG_PATH = str((ROOT / 'navsim/planning/script/config/pdm_scoring').resolve())

os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
os.environ['NUPLAN_MAPS_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps'
os.environ['OPENSCENE_DATA_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset'
os.environ['NAVSIM_DEVKIT_ROOT'] = '/data/ruolin/SimScale'

REQUIRED_FEATURE_KEYS = [
    'diff_layer0_cross_agent_out',
    'diff_layer0_cross_ego_out',
    'diff_layer1_cross_agent_out',
    'diff_layer1_cross_ego_out',
]


# ------------------ Utility math ------------------
def _matrix_from_tensor(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[None, :]
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 4:
        # BCHW -> BHW,C
        return np.transpose(x, (0, 2, 3, 1)).reshape(-1, x.shape[1])
    return x.reshape(x.shape[0], -1) if x.ndim > 1 else x[None, :]


def _reduce_feature_list(arr_list: List[np.ndarray], max_rows_per_item: int = 800) -> np.ndarray:
    mats = []
    for a in arr_list:
        m = _matrix_from_tensor(a)
        if m.shape[0] > max_rows_per_item:
            idx = np.random.choice(m.shape[0], max_rows_per_item, replace=False)
            m = m[idx]
        mats.append(m)
    if not mats:
        return np.zeros((1, 1), dtype=np.float32)
    return np.concatenate(mats, axis=0)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> float:
    x = X.mean(axis=0)
    y = Y.mean(axis=0)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = y / (np.linalg.norm(y) + 1e-9)
    return float(1.0 - np.dot(x, y))


def linear_cka(X: np.ndarray, Y: np.ndarray, n_samples: int = 512) -> float:
    n = min(X.shape[0], Y.shape[0], n_samples)
    if n < 4:
        return 0.0
    ix = np.random.choice(X.shape[0], n, replace=False)
    iy = np.random.choice(Y.shape[0], n, replace=False)
    Xs = X[ix]
    Ys = Y[iy]

    Xs = Xs - Xs.mean(0, keepdims=True)
    Ys = Ys - Ys.mean(0, keepdims=True)

    K = Xs @ Xs.T
    L = Ys @ Ys.T
    hsic = np.sum(K * L)
    norm = np.sqrt(np.sum(K * K) * np.sum(L * L)) + 1e-9
    return float(hsic / norm)


def mmd_rbf(X: np.ndarray, Y: np.ndarray, n_samples: int = 400) -> float:
    n = min(X.shape[0], Y.shape[0], n_samples)
    if n < 4:
        return 0.0
    ix = np.random.choice(X.shape[0], n, replace=False)
    iy = np.random.choice(Y.shape[0], n, replace=False)
    Xs = X[ix]
    Ys = Y[iy]

    Z = np.vstack([Xs, Ys])
    # median heuristic for sigma
    d2 = np.sum((Z[: min(200, len(Z))][:, None, :] - Z[: min(200, len(Z))][None, :, :]) ** 2, axis=-1)
    sigma2 = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    sigma2 = max(float(sigma2), 1e-6)

    def k(a, b):
        dist2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-dist2 / (2.0 * sigma2))

    kxx = k(Xs, Xs)
    kyy = k(Ys, Ys)
    kxy = k(Xs, Ys)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _flatten_prob(a: np.ndarray) -> np.ndarray:
    p = a.astype(np.float64).reshape(-1)
    p = np.maximum(p, 1e-12)
    p /= p.sum()
    return p


def kl_div(p: np.ndarray, q: np.ndarray) -> float:
    p = _flatten_prob(p)
    q = _flatten_prob(q)
    return float(np.sum(p * np.log(p / q)))


def js_div(p: np.ndarray, q: np.ndarray) -> float:
    p = _flatten_prob(p)
    q = _flatten_prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def wass_div(p: np.ndarray, q: np.ndarray) -> float:
    p = _flatten_prob(p)
    q = _flatten_prob(q)
    x = np.arange(len(p), dtype=np.float64)
    return float(wasserstein_distance(x, x, u_weights=p, v_weights=q))


# ------------------ Token resolution ------------------
def get_diff_tokens() -> List[str]:
    import pandas as pd

    a = pd.read_csv(TRANS_CSV)
    b = pd.read_csv(ORIG_CSV)
    a = a[a['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'score_transfer'})
    b = b[b['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'score_origin'})
    m = a.merge(b, on='token')
    diff = m[m['score_transfer'] > m['score_origin']].sort_values('token')
    return diff['token'].tolist()


# ------------------ Model extraction ------------------
def patch_self_attention_capture():
    def patched_forward(self, x):
        b, t, c = x.size()
        k = self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach().cpu().numpy()

        att_drop = self.attn_drop(att)
        y = att_drop @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y

    SelfAttention.forward = patched_forward


def build_agent_and_loader(tokens: List[str]):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name='default_run_pdm_score_gpu', overrides=['agent=my_diffusiondrive', 'train_test_split=navmini'])

    agent = instantiate(cfg.agent)
    agent.initialize()
    model = agent._transfuser_model.eval().cuda()

    root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    loader = SceneLoader(
        original_sensor_path=root / 'sensor_blobs/mini',
        data_path=root / 'navsim_logs/mini',
        scene_filter=SceneFilter(tokens=tokens),
        sensor_config=agent.get_sensor_config(),
    )
    return agent, model, loader


def extract_single(agent, model, loader, token: str, mode: str) -> Dict:
    cache_file = CACHE_DIR / f'{token}_{mode}.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        feat_keys = set(cached.get('features', {}).keys())
        if all(k in feat_keys for k in REQUIRED_FEATURE_KEYS):
            return cached

    # Prepare env for transfer mode
    if mode == 'transfer':
        os.environ['TRANSFER_SCENARIO_IMAGE_DIR'] = str(TRANSFER_DIR)
        os.environ['TRANSFER_IMAGE_SUFFIX'] = TRANSFER_SUFFIX
        os.environ['NAVSIM_CURRENT_TOKEN'] = token
    else:
        os.environ.pop('TRANSFER_SCENARIO_IMAGE_DIR', None)
        os.environ.pop('TRANSFER_IMAGE_SUFFIX', None)
        os.environ.pop('NAVSIM_CURRENT_TOKEN', None)

    scene_input = loader.get_agent_input_from_token(token)
    feat = {}
    for b in agent.get_feature_builders():
        feat.update(b.compute_features(scene_input))

    batch = {k: v.unsqueeze(0).cuda() for k, v in feat.items()}

    outputs: Dict[str, List[np.ndarray]] = {}
    attn: Dict[str, List[np.ndarray]] = {}
    hooks = []

    def save_out(name):
        def _h(mod, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(x):
                outputs.setdefault(name, []).append(x.detach().cpu().numpy())
        return _h

    # Encoder feature layers
    hooks.append(model._backbone.image_encoder.layer1.register_forward_hook(save_out('enc_layer1')))
    hooks.append(model._backbone.image_encoder.layer2.register_forward_hook(save_out('enc_layer2')))
    hooks.append(model._backbone.image_encoder.layer3.register_forward_hook(save_out('enc_layer3')))
    hooks.append(model._backbone.image_encoder.layer4.register_forward_hook(save_out('enc_layer4')))

    # Backbone and decoder outputs
    hooks.append(model._backbone.register_forward_hook(save_out('backbone_bev_upscale')))
    for i, layer in enumerate(model._tf_decoder.layers):
        hooks.append(layer.register_forward_hook(save_out(f'tf_decoder_layer{i}')))

    # Trajectory decoder outputs / attention
    for i, layer in enumerate(model._trajectory_head.diff_decoder.layers):
        hooks.append(layer.cross_bev_attention.register_forward_hook(save_out(f'diff_layer{i}_cross_bev_out')))
        hooks.append(layer.cross_agent_attention.register_forward_hook(save_out(f'diff_layer{i}_cross_agent_out')))
        hooks.append(layer.cross_ego_attention.register_forward_hook(save_out(f'diff_layer{i}_cross_ego_out')))

        def pre_bev(name):
            def _ph(mod, inp):
                q = inp[0]
                w = mod.attention_weights(q).softmax(-1)
                attn.setdefault(name, []).append(w.detach().cpu().numpy())
            return _ph

        hooks.append(layer.cross_bev_attention.register_forward_pre_hook(pre_bev(f'diff_layer{i}_cross_bev_attn')))

        def mh_hook(name):
            def _h(mod, inp, out):
                if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None and torch.is_tensor(out[1]):
                    attn.setdefault(name, []).append(out[1].detach().cpu().numpy())
            return _h

        hooks.append(layer.cross_agent_attention.register_forward_hook(mh_hook(f'diff_layer{i}_cross_agent_attn')))
        hooks.append(layer.cross_ego_attention.register_forward_hook(mh_hook(f'diff_layer{i}_cross_ego_attn')))

    # Run
    with torch.no_grad():
        _ = model(batch)

    # collect backbone transformer self-attentions from patched modules
    sa_idx = 0
    for m in model._backbone.modules():
        if isinstance(m, SelfAttention) and hasattr(m, 'last_att'):
            attn.setdefault(f'encoder_selfatt_{sa_idx}', []).append(m.last_att)
            sa_idx += 1

    for h in hooks:
        h.remove()

    data = {
        'token': token,
        'mode': mode,
        'features': outputs,
        'attn': attn,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

    return data


# ------------------ Analysis + plotting ------------------
def compute_feature_distances(origin: Dict, transfer: Dict) -> Dict[str, Dict[str, float]]:
    layers = sorted(set(origin['features'].keys()) & set(transfer['features'].keys()))
    out = {}
    for ly in layers:
        X = _reduce_feature_list(origin['features'][ly])
        Y = _reduce_feature_list(transfer['features'][ly])
        out[ly] = {
            'cosine': cosine_distance(X, Y),
            'mmd': mmd_rbf(X, Y),
            'cka': linear_cka(X, Y),
        }
    return out


def _merge_attn_list(arrs: List[np.ndarray]) -> np.ndarray:
    # stack on call-dim then mean
    if len(arrs) == 1:
        return arrs[0]
    return np.mean(np.stack(arrs, axis=0), axis=0)


def compute_attention_divergence(origin: Dict, transfer: Dict) -> Dict[str, Dict[str, float]]:
    layers = sorted(set(origin['attn'].keys()) & set(transfer['attn'].keys()))
    out = {}
    for ly in layers:
        A = _merge_attn_list(origin['attn'][ly])
        B = _merge_attn_list(transfer['attn'][ly])
        out[ly] = {
            'kl': kl_div(A, B),
            'js': js_div(A, B),
            'wasserstein': wass_div(A, B),
        }
    return out


def get_query_modality_profile(data: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
    # Prefer cross-BEV attention: expected shape close to (B, 20, 8).
    keys = [k for k in sorted(data['attn'].keys()) if 'cross_bev_attn' in k]
    if not keys:
        return np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32), ''

    key = keys[0]
    A = _merge_attn_list(data['attn'][key])

    # Normalize to (Q, M).
    if A.ndim == 3:
        # (B, Q, M)
        A = A.mean(axis=0)
    elif A.ndim == 4:
        # (B, H, Q, M)
        A = A.mean(axis=(0, 1))
    elif A.ndim != 2:
        A = A.reshape(A.shape[-2], A.shape[-1])

    A = np.maximum(A, 1e-12)
    A = A / A.sum(axis=-1, keepdims=True)

    per_modality = A.mean(axis=0)
    per_query_entropy = -np.sum(A * np.log(A), axis=-1)
    return per_modality, per_query_entropy, key


def pick_tsne_layers(feature_dist: Dict[str, Dict[str, float]], k: int = 3) -> List[str]:
    pairs = [(ly, vals['cosine']) for ly, vals in feature_dist.items()]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:k]]


def tsne_data_for_layer(origin: Dict, transfer: Dict, layer: str, max_points: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    X = _reduce_feature_list(origin['features'][layer], max_rows_per_item=max_points)
    Y = _reduce_feature_list(transfer['features'][layer], max_rows_per_item=max_points)

    # normalize
    Z = np.vstack([X, Y])
    Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-6)

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init='pca')
    E = tsne.fit_transform(Z)
    return E[: X.shape[0]], E[X.shape[0] :]


def make_token_tab(token: str, origin: Dict, transfer: Dict, score_info: str) -> Panel:
    fd = compute_feature_distances(origin, transfer)
    ad = compute_attention_divergence(origin, transfer)

    # Feature distances line plot
    layers_f = list(fd.keys())
    src_f = ColumnDataSource(data={
        'x': list(range(len(layers_f))),
        'layer': layers_f,
        'cosine': [fd[l]['cosine'] for l in layers_f],
        'mmd': [fd[l]['mmd'] for l in layers_f],
        'cka': [fd[l]['cka'] for l in layers_f],
    })
    p1 = figure(width=820, height=280, title=f'{token} | 分层特征距离曲线 (Origin vs Transfer)', tools='pan,wheel_zoom,box_zoom,reset,save')
    l1 = p1.line('x', 'cosine', source=src_f, color=Category10[10][0], line_width=2, legend_label='Cosine Distance')
    l2 = p1.line('x', 'mmd', source=src_f, color=Category10[10][1], line_width=2, legend_label='MMD (RBF)')
    l3 = p1.line('x', 'cka', source=src_f, color=Category10[10][2], line_width=2, legend_label='CKA')
    p1.circle('x', 'cosine', source=src_f, color=Category10[10][0], size=5)
    p1.circle('x', 'mmd', source=src_f, color=Category10[10][1], size=5)
    p1.circle('x', 'cka', source=src_f, color=Category10[10][2], size=5)
    p1.xaxis.ticker = list(range(len(layers_f)))
    p1.xaxis.major_label_overrides = {i: layers_f[i] for i in range(len(layers_f))}
    p1.xaxis.major_label_orientation = 1.1
    p1.add_tools(HoverTool(tooltips=[('layer', '@layer'), ('cosine', '@cosine{0.0000}'), ('mmd', '@mmd{0.0000}'), ('cka', '@cka{0.0000}')]))
    p1.legend.click_policy = 'hide'

    # Dedicated cross-agent/cross-ego feature bias panel
    ce_layers = [l for l in layers_f if ('cross_agent_out' in l or 'cross_ego_out' in l)]
    p_ce = figure(width=820, height=250, title=f'{token} | cross-agent / cross-ego 特征偏差', tools='pan,wheel_zoom,box_zoom,reset,save')
    if ce_layers:
        src_ce = ColumnDataSource(data={
            'x': list(range(len(ce_layers))),
            'layer': ce_layers,
            'cosine': [fd[l]['cosine'] for l in ce_layers],
            'mmd': [fd[l]['mmd'] for l in ce_layers],
            'cka': [fd[l]['cka'] for l in ce_layers],
        })
        p_ce.line('x', 'cosine', source=src_ce, color=Category10[10][0], line_width=2, legend_label='Cosine Distance')
        p_ce.line('x', 'mmd', source=src_ce, color=Category10[10][1], line_width=2, legend_label='MMD (RBF)')
        p_ce.line('x', 'cka', source=src_ce, color=Category10[10][2], line_width=2, legend_label='CKA')
        p_ce.circle('x', 'cosine', source=src_ce, color=Category10[10][0], size=5)
        p_ce.circle('x', 'mmd', source=src_ce, color=Category10[10][1], size=5)
        p_ce.circle('x', 'cka', source=src_ce, color=Category10[10][2], size=5)
        p_ce.xaxis.ticker = list(range(len(ce_layers)))
        p_ce.xaxis.major_label_overrides = {i: ce_layers[i] for i in range(len(ce_layers))}
        p_ce.xaxis.major_label_orientation = 1.1
        p_ce.add_tools(HoverTool(tooltips=[('layer', '@layer'), ('cosine', '@cosine{0.0000}'), ('mmd', '@mmd{0.0000}'), ('cka', '@cka{0.0000}')]))
        p_ce.legend.click_policy = 'hide'
    else:
        p_ce.title.text = f'{token} | 未检测到 cross-agent/cross-ego 特征层输出'

    # Attention divergence line plot
    layers_a = list(ad.keys())
    if layers_a:
        src_a = ColumnDataSource(data={
            'x': list(range(len(layers_a))),
            'layer': layers_a,
            'kl': [ad[l]['kl'] for l in layers_a],
            'js': [ad[l]['js'] for l in layers_a],
            'wasserstein': [ad[l]['wasserstein'] for l in layers_a],
        })
        p2 = figure(width=820, height=280, title=f'{token} | 分层注意力差分图 (KL/JS/Wasserstein)', tools='pan,wheel_zoom,box_zoom,reset,save')
        p2.line('x', 'kl', source=src_a, color=Category10[10][3], line_width=2, legend_label='KL')
        p2.line('x', 'js', source=src_a, color=Category10[10][4], line_width=2, legend_label='JS')
        p2.line('x', 'wasserstein', source=src_a, color=Category10[10][5], line_width=2, legend_label='Wasserstein')
        p2.circle('x', 'kl', source=src_a, color=Category10[10][3], size=5)
        p2.circle('x', 'js', source=src_a, color=Category10[10][4], size=5)
        p2.circle('x', 'wasserstein', source=src_a, color=Category10[10][5], size=5)
        p2.xaxis.ticker = list(range(len(layers_a)))
        p2.xaxis.major_label_overrides = {i: layers_a[i] for i in range(len(layers_a))}
        p2.xaxis.major_label_orientation = 1.1
        p2.add_tools(HoverTool(tooltips=[('layer', '@layer'), ('KL', '@kl{0.0000}'), ('JS', '@js{0.0000}'), ('Wasserstein', '@wasserstein{0.0000}')]))
        p2.legend.click_policy = 'hide'
    else:
        p2 = figure(width=820, height=280, title=f'{token} | 无可用注意力层数据')

    # 20-query x 8-modality weight profile from cross-BEV attention
    mod_o, ent_o, attn_key_o = get_query_modality_profile(origin)
    mod_t, ent_t, attn_key_t = get_query_modality_profile(transfer)
    mod_n = int(min(len(mod_o), len(mod_t)))
    qry_n = int(min(len(ent_o), len(ent_t)))

    p3_title = f'{token} | 输入源相机视角模态权重分布 (平均于 queries)'
    if attn_key_o:
        p3_title += f' | layer={attn_key_o}'
    p3 = figure(width=400, height=260, title=p3_title, tools='pan,wheel_zoom,box_zoom,reset,save')
    if mod_n > 0:
        x = np.arange(mod_n)
        src_m = ColumnDataSource(data={
            'x': x,
            'origin': mod_o[:mod_n],
            'transfer': mod_t[:mod_n],
            'label': [f'm{i}' for i in x],
        })
        p3.line('x', 'origin', source=src_m, line_width=2, color=Category10[10][0], legend_label='Origin')
        p3.line('x', 'transfer', source=src_m, line_width=2, color=Category10[10][1], legend_label='Transfer')
        p3.circle('x', 'origin', source=src_m, size=5, color=Category10[10][0])
        p3.circle('x', 'transfer', source=src_m, size=5, color=Category10[10][1])
        p3.xaxis.ticker = list(range(mod_n))
        p3.xaxis.major_label_overrides = {i: f'm{i}' for i in range(mod_n)}
        p3.add_tools(HoverTool(tooltips=[('modality', '@label'), ('origin', '@origin{0.0000}'), ('transfer', '@transfer{0.0000}')]))
        p3.legend.click_policy = 'hide'

    p4 = figure(width=400, height=260, title=f'{token} | 20-query 注意力熵分布', tools='pan,wheel_zoom,box_zoom,reset,save')
    if qry_n > 0:
        q = np.arange(qry_n)
        src_q = ColumnDataSource(data={
            'q': q,
            'origin': ent_o[:qry_n],
            'transfer': ent_t[:qry_n],
        })
        p4.line('q', 'origin', source=src_q, line_width=2, color=Category10[10][0], legend_label='Origin')
        p4.line('q', 'transfer', source=src_q, line_width=2, color=Category10[10][1], legend_label='Transfer')
        p4.circle('q', 'origin', source=src_q, size=4, color=Category10[10][0])
        p4.circle('q', 'transfer', source=src_q, size=4, color=Category10[10][1])
        p4.add_tools(HoverTool(tooltips=[('query', '@q'), ('origin_entropy', '@origin{0.0000}'), ('transfer_entropy', '@transfer{0.0000}')]))
        p4.legend.click_policy = 'hide'

    # Interactive t-SNE for top-k layers
    ts_layers = pick_tsne_layers(fd, k=min(3, len(fd)))
    ts_plots = []
    for idx, ly in enumerate(ts_layers):
        e_o, e_t = tsne_data_for_layer(origin, transfer, ly)
        src_o = ColumnDataSource(data={'x': e_o[:, 0], 'y': e_o[:, 1], 'grp': ['Origin'] * len(e_o)})
        src_t = ColumnDataSource(data={'x': e_t[:, 0], 'y': e_t[:, 1], 'grp': ['Transfer'] * len(e_t)})

        p = figure(width=380, height=330, title=f'{token} | t-SNE: {ly}', tools='pan,wheel_zoom,box_zoom,reset,save')
        p.scatter('x', 'y', source=src_o, size=5, alpha=0.35, color=Category10[10][0], legend_label='Origin')
        p.scatter('x', 'y', source=src_t, size=5, alpha=0.35, color=Category10[10][1], legend_label='Transfer')
        p.add_tools(HoverTool(tooltips=[('group', '@grp'), ('x', '@x{0.00}'), ('y', '@y{0.00}')]))
        p.legend.click_policy = 'hide'
        ts_plots.append(p)

    top_div = Div(text=f'<b>{token}</b><br>{score_info}<br>注：t-SNE横纵轴为高维特征的二维嵌入坐标，仅表示相对邻近关系。', width=820, height=60)

    ts_grid = gridplot(ts_plots, ncols=len(ts_plots) if ts_plots else 1)
    modality_grid = gridplot([p3, p4], ncols=2)
    layout = column(top_div, p1, p_ce, p2, modality_grid, ts_grid)
    return Panel(child=layout, title=token)


def get_score_info_map() -> Dict[str, str]:
    import pandas as pd

    a = pd.read_csv(TRANS_CSV)
    b = pd.read_csv(ORIG_CSV)
    a = a[a['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'transfer'})
    b = b[b['token'] != 'average_all_frames'][['token', 'score']].rename(columns={'score': 'origin'})
    m = a.merge(b, on='token')
    m['delta'] = m['transfer'] - m['origin']
    out = {}
    for _, r in m.iterrows():
        out[r['token']] = f"Origin={r['origin']:.4f}, Transfer={r['transfer']:.4f}, Delta={r['delta']:+.4f}"
    return out


def main():
    np.random.seed(42)
    patch_self_attention_capture()

    tokens = get_diff_tokens()
    score_info = get_score_info_map()

    agent, model, loader = build_agent_and_loader(tokens)

    tabs = []
    for t in tokens:
        origin = extract_single(agent, model, loader, t, 'origin')
        transfer = extract_single(agent, model, loader, t, 'transfer')
        tabs.append(make_token_tab(t, origin, transfer, score_info.get(t, '')))

    intro = Div(text='''
    <h2>Origin vs Transfer 差异分析（Bokeh 交互版）</h2>
    <p>包含：1) 全部分层特征距离曲线（Cosine/MMD/CKA）；2) cross-agent/cross-ego 特征偏差专门面板；3) 分层注意力差分图（KL/JS/Wasserstein）；4) 模态权重与query熵；5) t-SNE交互散点（可缩放/悬停/图例点击隐藏）。</p>
    <p>缓存策略：若存在 <code>outputs/analysis_cache_encoder/{token}_{mode}.pkl</code> 则直接复用；否则自动重跑并写入缓存。</p>
    ''', width=1200)

    doc = column(intro, Tabs(tabs=tabs))
    output_file(str(HTML_OUT), title='Bokeh Diff Analysis')
    save(doc)

    print(f'Saved HTML: {HTML_OUT}')
    print(f'Cache dir: {CACHE_DIR}')


if __name__ == '__main__':
    main()
