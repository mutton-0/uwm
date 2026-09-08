"""Alpamayo-R1 的 activation patching 接口（C-hazard 用）。

与 AutoVLA 的 `PooledCapture` 逐条同构：同一组 decoder layer 钩子上同时支持
  * `capture_full` —— 保留 prefill 的全序列隐状态 [S, C]（CPU float32）当参考侧；
  * `set_patch(d, tokens)` —— 在 prefill 前向里把第 L 层输出替换为参考侧激活；
    decode step（S=1）一律不动，patch 通过 KV cache 影响后续所有生成步。

**Alpamayo 特有的一处必须先说清楚的东西**：它的推理是 `top_p=0.98, temperature=0.6`
的**随机采样** rollout（AutoVLA 是 top_k=1 的贪心解码，等价确定性）。
因此本模块强制在每次前向前重置 seed（与 `alpamayo_runner.infer` 同一手法），
使「同输入同 seed ⇒ 同输出」成立，patch 是两次运行之间**唯一**的差异。
但这只保证可复现，不保证**稳健**：patch 可能通过改变某一步采样到的 token
把 CoT 整条带偏，从而产生一个与「危险信息量」无关的大幅行为变化。
故 C-hazard 读数在 Alpamayo 上必须并列报告**采样噪声地板**
（同输入换 seed 的 v_plan 离散度），见 `scripts/c_axis_hazard_patch.py --seed-floor`。
"""
from __future__ import annotations

import sys

for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
           "/data/Zhengyang/alpamayo/src"):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
from alpamayo_runner import AlpamayoRunner, plan_speed   # noqa: E402

POOLS = ("vision_mean", "last_token", "seq_mean")
PREFIX = "vlm.model.language_model.layers"


def longest_run_token(ids):
    best, i = (None, 0, 0), 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]:
            j += 1
        if j - i + 1 > best[2]:
            best = (int(ids[i]), i, j - i + 1)
        i = j + 1
    return best


class PatchCapture:
    def __init__(self, model, prefix=PREFIX):
        layers = [(n, m) for n, m in model.named_modules()
                  if n.startswith(prefix) and m.__class__.__name__.endswith("DecoderLayer")]
        layers.sort(key=lambda x: int(x[0].rsplit(".", 1)[-1]))
        assert layers, f"未找到 DecoderLayer（prefix={prefix}）"
        self.n_layers = len(layers)
        self.mask = None
        self.capture_full = False
        self._patch = None
        self._patch_tokens = "all"
        self.reset()
        self._h = [m.register_forward_hook(self._mk(i)) for i, (_n, m) in enumerate(layers)]

    def reset(self):
        self.pooled = {p: [None] * self.n_layers for p in POOLS}
        self.full = [None] * self.n_layers
        self.seen = [0] * self.n_layers

    def set_patch(self, d, tokens="all"):
        self._patch = d
        self._patch_tokens = tokens

    def _mk(self, i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            S = h.shape[1]
            if S <= 1:
                return None
            ref = None if self._patch is None else self._patch.get(i)
            if ref is not None:
                r = torch.as_tensor(ref, device=h.device, dtype=h.dtype)
                # 序列长度不一致时**放弃 patch 而不是截断对齐** —— 截断会把「换掉了什么」
                # 变成不受控的量。调用方据此丢弃该事件。
                if r.shape[0] != S:
                    raise RuntimeError(f"patch seq_len {r.shape[0]} != run seq_len {S} @L{i}")
                h = h.clone()
                if self._patch_tokens == "image" and self.mask is not None and len(self.mask) == S:
                    m = torch.as_tensor(self.mask, device=h.device)
                    h[0, m] = r[m]
                else:
                    h[0] = r
                out = (h,) + tuple(out[1:]) if isinstance(out, (tuple, list)) else h
            if S <= self.seen[i]:
                return out if ref is not None else None
            self.seen[i] = S
            x = h[0].float()
            if self.capture_full:
                self.full[i] = x.cpu().numpy()
            self.pooled["last_token"][i] = x[-1].cpu().numpy()
            self.pooled["seq_mean"][i] = x.mean(0).cpu().numpy()
            if self.mask is not None and len(self.mask) == S and self.mask.any():
                m = torch.as_tensor(self.mask, device=x.device)
                self.pooled["vision_mean"][i] = x[m].mean(0).cpu().numpy()
            return out if ref is not None else None
        return hook

    def remove(self):
        for h in self._h:
            h.remove()


class AlpaPatchRunner:
    """把 `AlpamayoRunner` 包成与 C-hazard 脚本一致的 run/set_patch 接口。"""

    def __init__(self, device="cuda:0", seed=42):
        self.r = AlpamayoRunner(device=device, seed=seed)
        self.cap = PatchCapture(self.r.model)
        self.n_layers = self.cap.n_layers
        self.device = device

    def set_patch(self, d, tokens="all"):
        self.cap.set_patch(d, tokens)

    def set_capture_full(self, flag):
        self.cap.capture_full = bool(flag)

    def covers(self, scene, t):
        return self.r.covers(scene, t)

    @torch.no_grad()
    def run_from_data(self, data, seed=None, return_traj=False):
        """在**已构造好的 data**（可能已被遮挡过）上做一次前向 —— F-3 用。

        与 `run()` 共用同一条前向路径与同一个 seed 复位纪律，
        差别只是图像来自调用方而不是 `self.r.load()`。
        """
        messages = self.r.helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = self.r.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        mi = self.r.helper.to_device({"tokenized_data": inputs,
                                      "ego_history_xyz": data["ego_history_xyz"],
                                      "ego_history_rot": data["ego_history_rot"]}, self.device)
        self.cap.reset()
        sd = self.r.seed if seed is None else seed
        torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _r, _extra = self.r.model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True)
        traj = pred_xyz.float().cpu().numpy()[0, 0, 0]
        # return_traj=False 时行为与首版逐位一致（F-3 速度版口径不受影响）；
        # True 时额外交出完整规划轨迹，供距离读数使用。
        return (plan_speed(traj), traj) if return_traj else plan_speed(traj)

    @torch.no_grad()
    def run(self, scene_name, t_sec, ego_anchor_t=None, seed=None, data=None):
        """data 非 None 时直接用它，跳过 nuScenes devkit（NAVSIM 侧由
        alpa_navsim_loader.load_navsim() 构造同构字典）。其余路径逐字不变，
        故返回的 full / seq_len / n_image_tokens 与 nuScenes 侧同义。"""
        if data is None:
            data = self.r.load(scene_name, t_sec)
        if ego_anchor_t is not None and data is None:
            a = self.r.load(scene_name, ego_anchor_t)
            data["ego_history_xyz"] = a["ego_history_xyz"]
            data["ego_history_rot"] = a["ego_history_rot"]
        messages = self.r.helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = self.r.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        ids = inputs["input_ids"][0].cpu().numpy()
        tok, _st, _ln = longest_run_token(ids)
        self.cap.mask = (ids == tok)
        mi = self.r.helper.to_device({"tokenized_data": inputs,
                                      "ego_history_xyz": data["ego_history_xyz"],
                                      "ego_history_rot": data["ego_history_rot"]}, self.device)
        self.cap.reset()
        sd = self.r.seed if seed is None else seed
        torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _r, extra = self.r.model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True)
        traj = pred_xyz.float().cpu().numpy()[0, 0, 0]
        c = extra.get("cot", [""])[0] if extra else ""
        return {"trajectory": traj, "commanded_speed": plan_speed(traj),
                # pooled 与 AutoVLA 适配器对齐：F 轴的 v_faith 抽取要读它。
                # 纯新增字段，C 轴走的是 "full"，行为不变。
                "pooled": {k: [None if x is None else np.asarray(x, np.float32) for x in v]
                           for k, v in self.cap.pooled.items()},
                "cot": c if isinstance(c, str) else str(c),
                "full": self.cap.full if self.cap.capture_full else None,
                "seq_len": int(len(ids)), "n_image_tokens": int(self.cap.mask.sum())}
