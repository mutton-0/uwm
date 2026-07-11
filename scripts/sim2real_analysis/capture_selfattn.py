"""
自包含：为更多 token 抓取 encoder_selfatt（+完整 features/decoder attn），写回
outputs/analysis_cache_encoder/{token}_{mode}.pkl（超集，保留原 decoder 特征）。

复刻自 scripts/build_bokeh_layer_diff_analysis.py 的 patch + hook 逻辑，
但使用正确的 ruolin_a6k 路径，且只做抓取、不画图。
"""
import os
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

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

import hydra
from hydra.utils import instantiate
from navsim.common.dataloader import SceneFilter, SceneLoader
from navsim.agents.diffusiondrive.transfuser_backbone import SelfAttention


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


def build(tokens: List[str]):
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
    loader = SceneLoader(
        original_sensor_path=root / 'sensor_blobs/mini',
        data_path=root / 'navsim_logs/mini',
        scene_filter=SceneFilter(tokens=tokens),
        sensor_config=agent.get_sensor_config(),
    )
    return agent, model, loader


def capture(agent, model, loader, token: str, mode: str) -> int:
    if mode == 'transfer':
        os.environ['TRANSFER_SCENARIO_IMAGE_DIR'] = str(TRANSFER_DIR)
        os.environ['TRANSFER_IMAGE_SUFFIX'] = TRANSFER_SUFFIX
        os.environ['NAVSIM_CURRENT_TOKEN'] = token
    else:
        for k in ('TRANSFER_SCENARIO_IMAGE_DIR', 'TRANSFER_IMAGE_SUFFIX', 'NAVSIM_CURRENT_TOKEN'):
            os.environ.pop(k, None)

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

    hooks.append(model._backbone.image_encoder.layer1.register_forward_hook(save_out('enc_layer1')))
    hooks.append(model._backbone.image_encoder.layer2.register_forward_hook(save_out('enc_layer2')))
    hooks.append(model._backbone.image_encoder.layer3.register_forward_hook(save_out('enc_layer3')))
    hooks.append(model._backbone.image_encoder.layer4.register_forward_hook(save_out('enc_layer4')))
    hooks.append(model._backbone.register_forward_hook(save_out('backbone_bev_upscale')))
    for i, layer in enumerate(model._tf_decoder.layers):
        hooks.append(layer.register_forward_hook(save_out(f'tf_decoder_layer{i}')))
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

    with torch.no_grad():
        _ = model(batch)

    sa_idx = 0
    for m in model._backbone.modules():
        if isinstance(m, SelfAttention) and hasattr(m, 'last_att'):
            attn.setdefault(f'encoder_selfatt_{sa_idx}', []).append(m.last_att)
            sa_idx += 1

    for h in hooks:
        h.remove()

    data = {'token': token, 'mode': mode, 'features': outputs, 'attn': attn}
    with open(CACHE_DIR / f'{token}_{mode}.pkl', 'wb') as f:
        pickle.dump(data, f)
    return sa_idx


def main():
    patch_self_attention_capture()

    # 候选池 = 有 transfer 图的 token；只处理还缺 encoder_selfatt 的
    pool = sorted({p.name.replace(f'_{TRANSFER_SUFFIX}', '')
                   for p in TRANSFER_DIR.glob(f'*_{TRANSFER_SUFFIX}')})
    todo = []
    for tok in pool:
        need = False
        for mode in ('origin', 'transfer'):
            pk = CACHE_DIR / f'{tok}_{mode}.pkl'
            if not pk.exists():
                need = True
                break
            d = pickle.load(open(pk, 'rb'))
            if not any(k.startswith('encoder_selfatt_') for k in d.get('attn', {})):
                need = True
                break
        if need:
            todo.append(tok)

    print(f'[pool] {len(pool)} tokens; need selfatt capture: {len(todo)}')
    print('[todo]', todo)
    if not todo:
        print('nothing to do')
        return

    agent, model, loader = build(todo)
    for i, tok in enumerate(todo, 1):
        for mode in ('origin', 'transfer'):
            n = capture(agent, model, loader, tok, mode)
            print(f'[{i}/{len(todo)}] {tok}_{mode}: encoder_selfatt captured = {n}')


if __name__ == '__main__':
    main()
