"""
CKA-vs-recovery 对照：对每个 encoder self-attn 层，比较三种量
  1) 注意力 JS 散度(origin vs transfer)        —— 注意力级散度
  2) 1-CKA(origin_out vs transfer_out)         —— 特征级散度(PRH kernel 对齐)
  3) recovery(L) 激活修补                        —— 因果责任
看"表征对齐度"是否比"注意力散度"更贴近因果，还是同样脱节。
"""
import os
from pathlib import Path
import numpy as np
import torch

# --- 迁移仓库时改这三个 env 即可(有默认值) ---
ROOT = Path(os.environ.get('SIMSCALE_ROOT', '/data/ruolin_a6k/SimScale'))
BKB_PATH = os.environ.get('BKB_PATH', '/data/ruolin_a6k/my_dataset/models/resnet34_model.bin')
CKPT_PATH = os.environ.get('CKPT_PATH', '/data/ruolin_a6k/ckpt/diffusiondrive_sim_navhard.ckpt')
CACHE_DIR = ROOT / 'outputs/analysis_cache_encoder'
TRANSFER_DIR = ROOT / 'outputs/my_diffusion_0_transfer2sim_scenarios'
TRANSFER_SUFFIX = 'after_transfer.jpg'
HYDRA_CONFIG_PATH = str((ROOT / 'navsim/planning/script/config/pdm_scoring').resolve())
os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
os.environ['NUPLAN_MAPS_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps'
os.environ['OPENSCENE_DATA_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset'
os.environ['NAVSIM_DEVKIT_ROOT'] = str(ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import hydra, pickle
from hydra.utils import instantiate
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention

TOKENS = ['aa96f52b95b155e7', 'ca9e7281adce5212']
SEED = 0
N_IMG = 256


def linear_cka(X, Y):
    """Kornblith 2019 线性 CKA. X:(n,d1) Y:(n,d2) -> [0,1], 1=表征相同."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X, 'fro') ** 2
    xx = np.linalg.norm(X.T @ X, 'fro')
    yy = np.linalg.norm(Y.T @ Y, 'fro')
    return float(xy / (xx * yy + 1e-12))


def js_div(p, q):
    p = np.maximum(p.reshape(-1).astype(np.float64), 1e-12); p /= p.sum()
    q = np.maximum(q.reshape(-1).astype(np.float64), 1e-12); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def build(tokens):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name='default_run_pdm_score_gpu',
                        overrides=['agent=my_diffusiondrive', 'train_test_split=navmini',
                                   f'agent.config.bkb_path={BKB_PATH}',
                                   f'agent.checkpoint_path={CKPT_PATH}'])
    agent = instantiate(cfg.agent); agent.initialize()
    model = agent._transfuser_model.eval().cuda()
    root = Path(os.environ['OPENSCENE_DATA_ROOT'])
    loader = SceneLoader(original_sensor_path=root / 'sensor_blobs/mini', data_path=root / 'navsim_logs/mini',
                         scene_filter=SceneFilter(tokens=tokens), sensor_config=agent.get_sensor_config())
    return agent, model, loader


def sa_modules(model):
    return [m for m in model._backbone.modules() if isinstance(m, SelfAttention)]


def make_batch(agent, loader, token, mode):
    if mode == 'transfer':
        os.environ['TRANSFER_SCENARIO_IMAGE_DIR'] = str(TRANSFER_DIR)
        os.environ['TRANSFER_IMAGE_SUFFIX'] = TRANSFER_SUFFIX
        os.environ['NAVSIM_CURRENT_TOKEN'] = token
    else:
        for k in ('TRANSFER_SCENARIO_IMAGE_DIR', 'TRANSFER_IMAGE_SUFFIX', 'NAVSIM_CURRENT_TOKEN'):
            os.environ.pop(k, None)
    ai = loader.get_agent_input_from_token(token)
    feat = {}
    for b in agent.get_feature_builders():
        feat.update(b.compute_features(ai))
    return {k: v.unsqueeze(0).cuda() for k, v in feat.items()}


def run(model, batch, seed=SEED):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        out = model(batch)
    return out['trajectory'][0].detach().cpu().numpy()


def run_cache(model, batch, sas, seed=SEED):
    cache = [None] * len(sas); hs = []
    for i, m in enumerate(sas):
        def _h(mod, inp, out, i=i):
            cache[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach().float().cpu().numpy()
        hs.append(m.register_forward_hook(_h))
    traj = run(model, batch, seed)
    for h in hs: h.remove()
    return traj, cache


def run_patch(model, batch, sas, layer, cached_origin, seed=SEED):
    t = torch.from_numpy(cached_origin[layer]).cuda()
    def _h(mod, inp, out, c=t):
        return c
    h = sas[layer].register_forward_hook(_h)
    traj = run(model, batch, seed); h.remove()
    return traj


def main():
    agent, model, loader = build(TOKENS)
    sas = sa_modules(model); nL = len(sas)
    print(f'#self-attn = {nL}\n')

    for tok in TOKENS:
        b_o = make_batch(agent, loader, tok, 'origin')
        b_t = make_batch(agent, loader, tok, 'transfer')
        traj_o, out_o = run_cache(model, b_o, sas)   # origin 输出缓存
        traj_t, out_t = run_cache(model, b_t, sas)   # transfer 输出缓存
        gap = float(np.linalg.norm(traj_t[:, :2] - traj_o[:, :2]))

        # 注意力 JS(从 pkl 缓存)
        ao = pickle.load(open(CACHE_DIR / f'{tok}_origin.pkl', 'rb'))['attn']
        at = pickle.load(open(CACHE_DIR / f'{tok}_transfer.pkl', 'rb'))['attn']

        rows = []
        for L in range(nL):
            Xo = out_o[L].reshape(-1, out_o[L].shape[-1])   # (320, C)
            Xt = out_t[L].reshape(-1, out_t[L].shape[-1])
            cka = linear_cka(Xo, Xt)
            wo = np.asarray(ao[f'encoder_selfatt_{L}'][0])[0].mean(0)   # head 平均 (320,320)
            wt = np.asarray(at[f'encoder_selfatt_{L}'][0])[0].mean(0)
            kimp_o = wo[:N_IMG, :N_IMG].mean(0); kimp_t = wt[:N_IMG, :N_IMG].mean(0)
            js = js_div(kimp_o, kimp_t)
            tp = run_patch(model, b_t, sas, L, out_o)
            rec = 1 - float(np.linalg.norm(tp[:, :2] - traj_o[:, :2])) / gap
            rows.append((L, js, 1 - cka, cka, rec))

        print(f'===== {tok}  (gap={gap:.2f}) =====')
        print(' L | 注意力JS | 1-CKA(特征散度) |  CKA  | recovery(因果)')
        for L, js, mis, cka, rec in rows:
            print(f'L{L} |  {js:.4f}  |     {mis:.4f}      | {cka:.3f} |   {rec:+.3f}')

        js_a = np.array([r[1] for r in rows]); mis_a = np.array([r[2] for r in rows]); rec_a = np.array([r[4] for r in rows])
        def corr(a, b):
            if a.std() < 1e-9 or b.std() < 1e-9: return float('nan')
            return float(np.corrcoef(a, b)[0, 1])
        def spear(a, b):
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
            return corr(ra.astype(float), rb.astype(float))
        print(f'  相关性(Pearson): JS↔recovery={corr(js_a,rec_a):+.2f} | (1-CKA)↔recovery={corr(mis_a,rec_a):+.2f}')
        print(f'  相关性(Spearman):JS↔recovery={spear(js_a,rec_a):+.2f} | (1-CKA)↔recovery={spear(mis_a,rec_a):+.2f}')
        print(f'  argmax: recovery=L{int(np.argmax(rec_a))}, 最大特征散度(1-CKA)=L{int(np.argmax(mis_a))}, 最大注意力JS=L{int(np.argmax(js_a))}\n')


if __name__ == '__main__':
    main()
