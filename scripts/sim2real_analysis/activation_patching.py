"""
激活修补(activation patching)：因果定位——transfer 前向时,把第 L 个 encoder
SelfAttention 的输出替换成 origin 的,看最终轨迹是否回到 origin。
recovery(L)=1 表示 patch 该层完全修复 transfer 引起的轨迹偏移(该层因果负责),
recovery(L)=0 表示该层与该偏移无因果关系(只是相关)。
"""
import os
from pathlib import Path
import numpy as np
import torch

# --- 迁移仓库时改这三个 env 即可(有默认值) ---
ROOT = Path(os.environ.get('SIMSCALE_ROOT', '/data/ruolin_a6k/SimScale'))
BKB_PATH = os.environ.get('BKB_PATH', '/data/ruolin_a6k/my_dataset/models/resnet34_model.bin')
CKPT_PATH = os.environ.get('CKPT_PATH', '/data/ruolin_a6k/ckpt/diffusiondrive_sim_navhard.ckpt')
TRANSFER_DIR = ROOT / 'outputs/my_diffusion_0_transfer2sim_scenarios'
TRANSFER_SUFFIX = 'after_transfer.jpg'
HYDRA_CONFIG_PATH = str((ROOT / 'navsim/planning/script/config/pdm_scoring').resolve())
os.environ['NUPLAN_MAP_VERSION'] = 'nuplan-maps-v1.0'
os.environ['NUPLAN_MAPS_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps'
os.environ['OPENSCENE_DATA_ROOT'] = '/data/Yuhao/world_model_yhl/navsim_workspace/dataset'
os.environ['NAVSIM_DEVKIT_ROOT'] = str(ROOT)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import hydra
from hydra.utils import instantiate
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention

import glob as _glob
TOKENS = sorted({os.path.basename(p).replace('_after_transfer.jpg', '')
                 for p in _glob.glob(str(TRANSFER_DIR / '*_after_transfer.jpg'))})
SEED = 0
GAP_MIN = 1.0   # 轨迹L2差 > 此值才算"真的不同", 纳入因果分析


def build(tokens):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(config_dir=HYDRA_CONFIG_PATH, version_base=None)
    cfg = hydra.compose(config_name='default_run_pdm_score_gpu',
                        overrides=['agent=my_diffusiondrive', 'train_test_split=navmini',
                                   f'agent.config.bkb_path={BKB_PATH}',
                                   f'agent.checkpoint_path={CKPT_PATH}'])
    agent = instantiate(cfg.agent)
    agent.initialize()
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
    return out['trajectory'][0].detach().cpu().numpy()  # (8,3)


def run_cache(model, batch, sas, seed=SEED):
    cache = [None] * len(sas)
    hs = []
    for i, m in enumerate(sas):
        def _h(mod, inp, out, i=i):
            cache[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach().clone()
        hs.append(m.register_forward_hook(_h))
    traj = run(model, batch, seed)
    for h in hs: h.remove()
    return traj, cache


def run_patch(model, batch, sas, layer, cached, seed=SEED):
    def _h(mod, inp, out, c=cached[layer]):
        return c  # 用 origin 的激活替换 transfer 的输出
    h = sas[layer].register_forward_hook(_h)
    traj = run(model, batch, seed)
    h.remove()
    return traj


def main():
    agent, model, loader = build(TOKENS)
    sas = sa_modules(model)
    nL = len(sas)
    print(f'#SelfAttention modules = {nL}; #tokens = {len(TOKENS)}')

    rows = []   # (token, gap, recs[nL], r_all)
    for tok in TOKENS:
        b_o = make_batch(agent, loader, tok, 'origin')
        b_t = make_batch(agent, loader, tok, 'transfer')
        traj_o, cache_o = run_cache(model, b_o, sas)
        traj_t = run(model, b_t)
        xy_o, xy_t = traj_o[:, :2], traj_t[:, :2]
        gap = float(np.linalg.norm(xy_t - xy_o))
        if gap < GAP_MIN:
            print(f'\n{tok}: gap={gap:.3f} < {GAP_MIN}  (轨迹几乎相同, 跳过因果分析)')
            rows.append((tok, gap, None, None))
            continue
        recs = []
        for L in range(nL):
            tp = run_patch(model, b_t, sas, L, cache_o)
            recs.append(1 - float(np.linalg.norm(tp[:, :2] - xy_o)) / gap)
        hs = [sas[L].register_forward_hook((lambda c: (lambda mod, inp, out: c))(cache_o[L])) for L in range(nL)]
        r_all = 1 - float(np.linalg.norm(run(model, b_t)[:, :2] - xy_o)) / gap
        for h in hs: h.remove()
        rows.append((tok, gap, recs, r_all))
        top = np.argsort(recs)[::-1][:3]
        print(f'\n{tok}: gap={gap:.3f}  top3=' + ' '.join(f'L{i}={recs[i]:+.2f}' for i in top) + f'  ALL={r_all:+.2f}')

    # ---- 汇总 ----
    used = [r for r in rows if r[2] is not None]
    print(f'\n===== 汇总: {len(used)}/{len(TOKENS)} 个 token 轨迹显著不同(gap>{GAP_MIN}) =====')
    print('token           gap   ' + ' '.join(f'L{i}   ' for i in range(nL)) + ' argmax  深层L4-6占比')
    R = np.zeros((len(used), nL))
    for j,(tok,gap,recs,r_all) in enumerate(used):
        R[j] = recs
        am = int(np.argmax(recs))
        pos = np.clip(recs,0,None); deep = pos[4:7].sum()/(pos.sum()+1e-9)
        print(f'{tok} {gap:6.2f}  ' + ' '.join(f'{v:+.2f}' for v in recs) + f'  L{am}    {deep:.2f}')
    print('\n各层 recovery 均值:  ' + ' '.join(f'L{i}={R[:,i].mean():+.2f}' for i in range(nL)))
    posR=np.clip(R,0,None)
    print('各层 正recovery均值:' + ' '.join(f'L{i}={posR[:,i].mean():+.2f}' for i in range(nL)))
    am_all=[int(np.argmax(R[j])) for j in range(len(used))]
    vals,cnts=np.unique(am_all,return_counts=True)
    print('argmax 因果层分布:', {f'L{v}':int(c) for v,c in zip(vals,cnts)})
    print(f'"深层L4-6主导(占正recovery>50%)"的token数: {sum(np.clip(R[j],0,None)[4:7].sum()/(np.clip(R[j],0,None).sum()+1e-9)>0.5 for j in range(len(used)))}/{len(used)}')


if __name__ == '__main__':
    main()
