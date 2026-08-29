"""SimLingo standalone 单帧推理封装（无 CARLA 依赖）。

严格复刻 team_code/agent_simlingo.py 的 tick()/run_step() 预处理与输入组装，
但把 CARLA 传感器/路由规划替换为参数化输入（图像文件 + ego speed + 固定 target point）。

手册对应：§4 G0 冒烟、§6 G2 批量前向、§10.1 预处理必须逐像素对齐。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image


# --------------------------------------------------------------------------------------
# 预处理：nuScenes CAM_FRONT -> SimLingo 输入张量
# --------------------------------------------------------------------------------------
def carla_post_crop_height(h: int) -> int:
    """SimLingo 自己的底部裁剪：h - (h*4.8)//16（约 30%），见 agent_simlingo.py:376。"""
    return int(h - (h * 4.8) // 16)


def preprocess_image(img_rgb: np.ndarray, pcfg: dict) -> Image.Image:
    """把任意来源的 RGB 图对齐到 SimLingo 的输入几何。

    默认 mode=resize_keep_aspect_then_crop:
        1600x900 --等比缩放--> 1024x576 --裁上部--> 1024x359
    359 = CARLA 1024x512 经 SimLingo 自身底部裁剪后的高度，保证送进
    dynamic_preprocess 的画布与训练时逐像素同尺度。
    """
    mode = pcfg["mode"]
    tw = int(pcfg["target_width"])
    ch = int(pcfg["carla_camera_height"])
    post_h = carla_post_crop_height(ch)  # 359

    h, w = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb)

    if mode == "resize_keep_aspect_then_crop":
        new_h = int(round(h * tw / w))
        pil = pil.resize((tw, new_h), Image.BICUBIC)
        pil = pil.crop((0, 0, tw, min(post_h, new_h)))
    elif mode == "stretch_to_carla_then_crop":
        pil = pil.resize((tw, ch), Image.BICUBIC)
        pil = pil.crop((0, 0, tw, post_h))
    elif mode == "carla_native":
        # 输入本身就是 CARLA 采集分辨率，只做 SimLingo 的底部裁剪
        pil = pil.crop((0, 0, w, carla_post_crop_height(h)))
    else:
        raise ValueError(f"unknown preprocess mode: {mode}")
    return pil


def image_to_token_grid(bbox_xyxy, im_wh, pcfg, n_patches, num_image_token):
    """原图像素框 -> vision token 索引集合（M3 region 池化）。

    链路（与 preprocess_image 严格同一套变换）：
      原图 (W,H) --等比缩放到宽 tw--> (tw, H*tw/W) --裁上部 post_h 行--> (tw, post_h)
      --dynamic_preprocess: 等比缩放到 (448*gw, 448*gh) 后切成 gw*gh 个 448 tile-->
      每 tile: ViT patch14 -> 32x32 -> pixel_shuffle(0.5) -> 16x16 = 256 token
    token 序号 = tile_idx * 256 + row * 16 + col（tile 按行主序，与 extract_feature 的拼接一致）。
    """
    W, H = im_wh
    tw = int(pcfg["target_width"])
    post_h = carla_post_crop_height(int(pcfg["carla_camera_height"]))
    input_size = int(pcfg["input_size"])
    side = int(np.sqrt(num_image_token))              # 每 tile 的 token 边长（16）

    sc = tw / W                                      # 等比缩放
    u0, v0, u1, v1 = [c * sc for c in bbox_xyxy]
    v0, v1 = v0, v1                                  # 裁上部：原点不变，仅截断
    if v0 >= post_h:
        return set()
    v1 = min(v1, post_h - 1)

    # dynamic_preprocess 的网格：n_patches 个 tile 排成 (gw, gh)
    gh = 1
    gw = n_patches // gh
    rw, rh = input_size * gw, input_size * gh        # 缩放后的画布
    fx, fy = rw / tw, rh / post_h
    U0, U1 = u0 * fx, u1 * fx
    V0, V1 = v0 * fy, v1 * fy

    cell = input_size / side                          # 每个 token 覆盖的像素边长（28）
    toks = set()
    for tile in range(n_patches):
        tx = (tile % gw) * input_size
        ty = (tile // gw) * input_size
        c0 = int(np.floor((U0 - tx) / cell)); c1 = int(np.ceil((U1 - tx) / cell))
        r0 = int(np.floor((V0 - ty) / cell)); r1 = int(np.ceil((V1 - ty) / cell))
        c0 = max(0, c0 - 1); r0 = max(0, r0 - 1)      # 外扩 1 patch 容错（§12.2-M3）
        c1 = min(side, c1 + 1); r1 = min(side, r1 + 1)
        for r in range(max(0, r0), max(0, r1)):
            for c in range(max(0, c0), max(0, c1)):
                toks.add(tile * num_image_token + r * side + c)
    return toks


def preprocess_hash(pcfg: dict) -> str:
    return hashlib.sha1(json.dumps(pcfg, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
@dataclass
class InferResult:
    waypoints: np.ndarray            # [F, 2] 速度航点（模型主输出 speed_wps）
    route: Optional[np.ndarray]      # [R, 2] 路径航点
    hidden: Dict[str, np.ndarray] = field(default_factory=dict)   # {"vision_mean": [L,C], "last_token": [L,C], ...}
    n_layers: int = 0
    hidden_dim: int = 0
    seq_len: int = 0
    n_region_tokens: int = 0
    n_query_tokens: int = 0        # driving query 段长度（T1-Q schema v3）
    vision_tokens: Optional[np.ndarray] = None      # debug: [n_vis, C]
    prompt: str = ""
    language: str = ""            # 模型贪心生成的文本（部署路径的副产物）


class SimLingoRunner:
    def __init__(self, cfg: dict, capture_hidden: bool = True):
        self.cfg = cfg
        paths = cfg["paths"]
        mcfg = cfg["model"]
        self.repo = Path(paths["simlingo_repo"])
        self.device = torch.device(mcfg["device"])
        self.capture_hidden = capture_hidden
        self.pcfg = mcfg["preprocess"]

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        # simlingo 的相对路径依赖（pretrained/ 缓存）以 repo 根为准
        os.chdir(self.repo)

        self.hydra_cfg = OmegaConf.load(paths["simlingo_hydra_cfg"])
        self.hydra_cfg.model.vision_model.use_global_img = self.hydra_cfg.data_module.use_global_img

        self._build_model(paths["simlingo_ckpt"])
        self._build_tokenizer_bits()
        if capture_hidden:
            self._register_hooks()

    # ---------------- 模型 ----------------
    def _build_model(self, ckpt_path: str):
        import hydra
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        variant = self.hydra_cfg.model.vision_model.variant
        cache_dir = str(self.repo / "pretrained" / variant.split("/")[1])
        self.cache_dir = cache_dir

        # 与 agent_simlingo.py 一致：不加 use_fast=False（闭环 eval 用的是 fast tokenizer）
        try:
            processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)
        except Exception:
            processor = AutoTokenizer.from_pretrained(variant, trust_remote_code=True)
        self.processor = processor
        self.tokenizer = processor.tokenizer if "tokenizer" in processor.__dict__ else processor
        self.tokenizer.add_special_tokens({"additional_special_tokens": [
            "<WAYPOINTS>", "<WAYPOINTS_DIFF>", "<ORG_WAYPOINTS_DIFF>", "<ORG_WAYPOINTS>",
            "<WAYPOINT_LAST>", "<ROUTE>", "<ROUTE_DIFF>", "<TARGET_POINT>"]})
        self.tokenizer.padding_side = "left"

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        model = hydra.utils.instantiate(
            self.hydra_cfg.model,
            cfg_data_module=self.hydra_cfg.data_module,
            processor=processor,
            cache_dir=cache_dir,
            _recursive_=False,
        ).to(self.device)
        torch.set_default_dtype(default_dtype)

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        model.eval()
        # 沿用 agent_simlingo.py 的部署路径 predict_language=True：
        # 先贪心生成语言，再把 [prompt+生成文本 | driving queries] 整条序列过一次 LM 得到 waypoints。
        # repo 的 predict_language=False 分支在 split_outputs_by_adaptor 处有 bug（未被官方 eval 走过）。
        model.predict_language = True
        self.model = model

        self.ckpt_sha1 = self._file_sha1(ckpt_path)

    @staticmethod
    def _file_sha1(path: str, chunk: int = 1 << 22) -> str:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()[:16]

    def _build_tokenizer_bits(self):
        from transformers import AutoConfig
        variant = self.hydra_cfg.model.vision_model.variant
        conf = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        image_size = conf.force_image_size or conf.vision_config.image_size
        patch_size = conf.vision_config.patch_size
        self.num_image_token = int((image_size // patch_size) ** 2 * (conf.downsample_ratio ** 2))

        spec = importlib.util.spec_from_file_location("get_conv_template", f"{self.cache_dir}/conversation.py")
        conv_module = importlib.util.module_from_spec(spec)
        sys.modules["get_conv_template"] = conv_module
        spec.loader.exec_module(conv_module)
        self.conv_module = conv_module
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    # ---------------- T2 manipulation：表征注入 ----------------
    def set_steering(self, layer=None, vec=None, alpha=0.0, mode="add", tokens="vision",
                     region_local=None):
        """在第 `layer` 层的输出上做 RepE 式操纵；layer=None 关闭。

        mode:
          add          Z' = Z + α·σ_L·v̂          （guide T2.1 剂量注入；α<0 即负向注入对照）
          project_out  Z' = Z − (Zᵀv̂)v̂           （T2.4 termination：把该方向整体剔除）
          recover      先剔除再按原投影量加回      （T2.4 recovery，应回到基线）
        σ_L = 本次前向中被注入 token 的激活标准差 —— 逐帧自归一化，
        使 α 在不同层/不同帧之间可比（guide "α 以该层激活标准差为单位"）。

        tokens: vision(全部 vision token) / region(目标框内 token) / all(整条序列)。
        默认 vision：方向是从 vision token 池化的 δ 上提的，注回同一批 token 才同构；
        且 causal attention 下 vision token 在序列最前，改动必然传到下游 driving query。
        """
        if layer is None or vec is None:
            self._steer = None
            return
        v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
        v = v / (v.norm() + 1e-8)
        self._steer = {"layer": int(layer), "v": v.to(self.device), "alpha": float(alpha),
                       "mode": mode, "tokens": tokens, "region_local": region_local}

    def set_patch(self, layers=None):
        """layers: {layer_idx: [n_vision_tokens, C] 参考侧激活}；None 关闭。

        只替换 **vision token 段**：语言段在两个条件之间长度与内容都会变，
        对应关系无定义（与 g2_cache schema v2 的同一条纪律）。
        """
        self._patch = None if not layers else {int(k): np.asarray(v, np.float32) for k, v in layers.items()}

    def _apply_patch(self, h: torch.Tensor, values) -> Optional[torch.Tensor]:
        """把 h 的 vision token 位置整体替换为 values；token 数不匹配时跳过（返回 None）。"""
        if h.shape[1] <= 1 or self._adaptor_dict is None:
            return None
        ids = self._adaptor_dict["language__ids"][0]
        if h.shape[1] < int(ids.shape[0]):
            return None
        vis = torch.nonzero(ids == self.img_context_token_id).flatten()
        if len(vis) == 0 or len(vis) != len(values):
            return None
        h = h.clone()
        h[0, vis, :] = torch.as_tensor(values, device=h.device).to(h.dtype)
        return h

    def _steer_positions(self, seq_len: int):
        """本次前向中要注入的 token 位置；返回 None 表示这次前向不注入。

        贪心解码期间 hook 会被触发多次：prefill(整条 prompt) / 每个 decode step(seq_len=1) /
        最后的整条前向。decode step 没有 vision token（且 KV 已含 prefill 的注入结果），跳过。
        """
        st = self._steer
        if seq_len <= 1 or self._adaptor_dict is None:
            return None
        ids = self._adaptor_dict["language__ids"][0]
        n_prompt = int(ids.shape[0])
        if seq_len < n_prompt:
            return None
        if st["tokens"] == "all":
            return torch.arange(seq_len, device=ids.device)
        if st["tokens"] == "query":
            # T1-L Step 2：轴在 query 位置定义，就注回 query 位置（同构）
            n_drv = int(self._len_driving) if self._len_driving else 0
            if not n_drv:
                return None
            return torch.arange(seq_len - n_drv, seq_len, device=ids.device)
        vis = torch.nonzero(ids == self.img_context_token_id).flatten()
        if st["tokens"] == "vision":
            return vis
        if st["tokens"] == "region":
            loc = st["region_local"]
            if loc is None or len(loc) == 0:
                return None
            return vis[torch.as_tensor(np.asarray(loc), device=vis.device)]
        raise ValueError(st["tokens"])

    def _apply_steer(self, h: torch.Tensor, st: dict) -> torch.Tensor:
        idx = self._steer_positions(h.shape[1])
        if idx is None or len(idx) == 0:
            return h
        v = st["v"]
        sub = h[0, idx, :].float()
        if st["mode"] == "add":
            sigma = float(sub.std())
            self._last_sigma = sigma
            sub = sub + (st["alpha"] * sigma) * v
        elif st["mode"] == "project_out":
            sub = sub - torch.outer(sub @ v, v)
        elif st["mode"] == "recover":
            # 先剔除逐 token 的投影，再把**该帧的平均投影量**统一加回。
            # 逐 token 原样加回是恒等式、什么也测不到；改成加回聚合量后，
            # "剔除→行为变化，加回→行为复原" 才构成对方向本身的检验（guide T2.4 recovery）。
            c = sub @ v
            sub = sub - torch.outer(c, v) + c.mean() * v
        else:
            raise ValueError(st["mode"])
        h = h.clone()
        h[0, idx, :] = sub.to(h.dtype)
        return h

    def _register_hooks(self):
        """在 LLM 每个 decoder layer 上挂 forward hook 抓 hidden states（并按需注入）。"""
        self._layer_outputs: List[torch.Tensor] = []
        self._adaptor_dict = None
        self._steer = None
        self._patch = None
        self._last_sigma = float("nan")
        layers = []
        for name, mod in self.model.language_model.model.named_modules():
            if mod.__class__.__name__.endswith("DecoderLayer"):
                layers.append((name, mod))
        # 只保留最内层的 DecoderLayer（PEFT 包装不会重复命名，但保险起见去重）
        self.layer_names = [n for n, _ in layers]
        assert layers, "未找到 DecoderLayer，无法抓取 hidden states"

        def make_hook(idx):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                # C 轴 activation patching：把该层的 vision token 段整体换成参考侧（sim）的激活。
                # 与 DiffusionDrive 侧 patching 的 corruption 口径同构（配对真实输入互换，禁用噪声破坏）。
                pt = getattr(self, "_patch", None)
                if pt and idx in pt:
                    h2 = self._apply_patch(h, pt[idx])
                    if h2 is not None:
                        self._layer_outputs.append(h2.detach())
                        if isinstance(out, (tuple, list)):
                            return (h2,) + tuple(out[1:])
                        return h2
                st = self._steer
                if st is not None and st["layer"] == idx:
                    # T2 manipulation：改写该层输出后再往下传（forward hook 返回值即新输出）
                    h2 = self._apply_steer(h, st)
                    self._layer_outputs.append(h2.detach())
                    if isinstance(out, (tuple, list)):
                        return (h2,) + tuple(out[1:])
                    return h2
                self._layer_outputs.append(h.detach())
            return hook

        self._hook_handles = [m.register_forward_hook(make_hook(i)) for i, (_, m) in enumerate(layers)]
        self.n_layers = len(layers)

        def adaptor_hook(_m, _inp, out):
            self._adaptor_dict = out
        self._hook_handles.append(self.model.adaptors.register_forward_hook(adaptor_hook))

        def driving_hook(_m, _inp, out):
            self._len_driving = out["inputs"].shape[1]
        self._hook_handles.append(self.model.adaptors.driving.register_forward_hook(driving_hook))

    # ---------------- 输入组装 ----------------
    def build_pixel_values(self, img_rgb: np.ndarray) -> torch.Tensor:
        from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
        pil = preprocess_image(img_rgb, self.pcfg)
        transform = build_transform(input_size=int(self.pcfg["input_size"]))
        images = dynamic_preprocess(
            pil,
            image_size=int(self.pcfg["input_size"]),
            use_thumbnail=bool(self.pcfg["use_global_img"]),
            max_num=int(self.pcfg["max_num_patches"]),
        )
        pv = torch.stack([transform(im) for im in images])           # [P, 3, 448, 448]
        p, c, h, w = pv.shape
        self._n_patches = p
        return pv.view(1, 1, p, c, h, w)                              # [B=1, T=1, P, C, H, W]

    def build_prompt(self, speed_mps: float, n_patches: int, context: str = ""):
        """context = 插在速度句之后的一句场景描述（T1-L 的语言配对刺激）。

        评分回路里 context 必须为空(中性 prompt) —— 危险措辞只允许出现在
        T1-L Step 1 的离线刺激集里，见 guide 附录 A"测量不污染红线"。
        """
        mcfg = self.cfg["model"]
        speed = round(float(speed_mps), 1)
        tp = np.asarray(mcfg["target_point_m"], dtype=np.float32)     # [[x1,y1],[x2,y2]]

        if mcfg["prompt_mode"] in ("target_point", "target_point_command"):
            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        elif mcfg["prompt_mode"] == "command":
            prompt_tp = mcfg["fixed_command"]
        else:
            raise ValueError(mcfg["prompt_mode"])

        ctx = f"{context.strip()} " if context and context.strip() else ""
        if mcfg["use_cot"]:
            prompt = f"Current speed: {speed} m/s. {ctx}{prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {ctx}{prompt_tp} Predict the waypoints."

        conv = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Waypoints:"},
        ]
        template = self.conv_module.get_conv_template("internlm2-chat")
        for i, part in enumerate(conv):
            if part["role"] == "assistant":
                template.append_message(template.roles[1], None)
            else:
                content = part["content"]
                if i == 0 and "<image>" not in content:
                    content = "<image>\n" + content
                template.append_message(template.roles[0], content)
        query = template.get_prompt()
        system_prompt = template.system_template.replace("{system_message}", template.system_message) + template.sep
        query = query.replace(system_prompt, "")

        image_tokens = "<img>" + "<IMG_CONTEXT>" * self.num_image_token * n_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

        tok = self.tokenizer([query], padding=True, return_tensors="pt", add_special_tokens=False)
        ids = tok["input_ids"]
        valid = ids != self.tokenizer.pad_token_id

        placeholder = {self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>"): tp}
        return query, prompt, ids, valid, [placeholder], tp

    def build_driving_input(self, img_rgb: np.ndarray, speed_mps: float, context: str = ""):
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.projection import get_camera_extrinsics, get_camera_intrinsics

        pv = self.build_pixel_values(img_rgb)
        n_patches = pv.shape[2]
        b, t, p, c, H, W = pv.shape
        query, prompt, ids, valid, placeholder, tp = self.build_prompt(speed_mps, n_patches, context)

        ll = LanguageLabel(
            phrase_ids=ids.to(self.device),
            phrase_valid=valid.to(self.device),
            phrase_mask=valid.to(self.device),
            placeholder_values=placeholder,
            language_string=[query],
            loss_masking=None,
        )
        fov = float(self.cfg["model"]["fov_deg"])
        di = DrivingInput(
            camera_images=pv.to(self.device).bfloat16(),
            image_sizes=None,
            camera_intrinsics=get_camera_intrinsics(W, H, fov).unsqueeze(0).view(1, 3, 3).float().to(self.device),
            camera_extrinsics=get_camera_extrinsics().unsqueeze(0).view(1, 4, 4).float().to(self.device),
            vehicle_speed=torch.FloatTensor([float(speed_mps)]).unsqueeze(0).to(self.device),
            target_point=torch.from_numpy(tp).unsqueeze(0).to(self.device),
            prompt=ll,
            prompt_inference=ll,
        )
        return di, prompt

    # ---------------- 前向 ----------------
    @torch.no_grad()
    def infer(self, img_rgb: np.ndarray, speed_mps: float, pool_modes=("vision_mean", "last_token"),
              prompt_speed: Optional[float] = None, bbox_xyxy=None, im_wh=None,
              context: str = "") -> InferResult:
        """prompt_speed 非 None 时，prompt 里写的速度与该帧真实 ego 速度解耦。

        手册 §10.4 要求 clean/ghost 之间 prompt 完全一致。但 prompt 模板里含
        `Current speed: X m/s`，两帧的真实自车速度本就不同——若各写各的，
        条件间就多了一个语言差异，同时污染 δ 与行为量 b。
        实测 d(指令速度)/d(prompt速度) = 0.725，而危险本身造成的 |b| 中位仅 0.52 m/s，
        即这个污染项与待测信号同量级。
        """
        torch.manual_seed(int(self.cfg["model"]["seed"]))
        di, prompt = self.build_driving_input(
            img_rgb, speed_mps if prompt_speed is None else prompt_speed, context)

        if self.capture_hidden:
            self._layer_outputs = []
            self._adaptor_dict = None
            self._len_driving = None

        speed_wps, route, lang = self.model(di)
        wps = speed_wps.float().cpu().numpy()[0] if speed_wps is not None else None
        rt = route.float().cpu().numpy()[0] if route is not None else None

        res = InferResult(waypoints=wps, route=rt, prompt=prompt)
        res.language = lang[0] if lang else ""
        if not self.capture_hidden:
            return res

        # 贪心生成期间 hook 会被反复触发（prefill + 每个 decode step + 最后的整条前向）。
        # 只取最后 n_layers 次（= 最后一次全序列前向）。
        hs = self._layer_outputs[-self.n_layers:]     # list[L] of [1, S, C]
        assert len(hs) == self.n_layers, "hook 未捕获到完整层"
        seq_lens = {h.shape[1] for h in hs}
        assert len(seq_lens) == 1 and hs[0].shape[1] > 1, f"最后一次前向的 seq 长度异常: {seq_lens}"
        res.n_layers = len(hs)
        res.hidden_dim = hs[0].shape[-1]
        res.seq_len = hs[0].shape[1]

        vis_mask, last_idx, last_lang_idx = self._token_masks(hs[0].shape[1])
        # T1-Q schema v3：driving query token 段 = 序列末尾 n_driving 个位置。
        # 动作由这些位置预测 => 行为中介态最可能住在这里，此前从未抓过。
        n_drv = int(self._len_driving) if self._len_driving else 0
        q_lo = hs[0].shape[1] - n_drv
        res.n_query_tokens = n_drv

        # M3：目标区域 token 掩码。vision token 在序列中按图像顺序排列，
        # 取其位置排序后的第 k 个即第 k 个图像 token。
        vis_pos = torch.nonzero(vis_mask).flatten()
        region_local = None
        if bbox_xyxy is not None and im_wh is not None:
            toks = image_to_token_grid(bbox_xyxy, im_wh, self.pcfg,
                                       self._n_patches, self.num_image_token)
            toks = sorted(t for t in toks if 0 <= t < len(vis_pos))
            if toks:
                region_local = torch.as_tensor(toks, device=vis_pos.device)
        res.n_region_tokens = 0 if region_local is None else len(region_local)

        pooled = {m: np.zeros((len(hs), hs[0].shape[-1]), dtype=np.float32) for m in pool_modes}
        for li, h in enumerate(hs):
            hf = h[0].float()
            if "vision_mean" in pooled:
                pooled["vision_mean"][li] = hf[vis_mask].mean(0).cpu().numpy()
            if "last_token" in pooled:
                pooled["last_token"][li] = hf[last_idx].cpu().numpy()
            if "last_lang_token" in pooled:
                pooled["last_lang_token"][li] = hf[last_lang_idx].cpu().numpy()
            if "query_mean" in pooled and n_drv:
                pooled["query_mean"][li] = hf[q_lo:].mean(0).cpu().numpy()
            if "query_first" in pooled and n_drv:
                pooled["query_first"][li] = hf[q_lo].cpu().numpy()
            if "seq_mean" in pooled:
                pooled["seq_mean"][li] = hf.mean(0).cpu().numpy()
            if region_local is not None and {"region_mean", "region_max", "bg_mean"} & set(pooled):
                vt = hf[vis_pos]                                   # [512, C] 按图像顺序
                m = torch.zeros(len(vis_pos), dtype=torch.bool, device=vt.device)
                m[region_local] = True
                if "region_mean" in pooled:
                    pooled["region_mean"][li] = vt[m].mean(0).cpu().numpy()
                if "region_max" in pooled:
                    pooled["region_max"][li] = vt[m].max(0).values.cpu().numpy()
                if "bg_mean" in pooled:
                    pooled["bg_mean"][li] = vt[~m].mean(0).cpu().numpy()
        res.hidden = pooled
        if getattr(self, "_debug_return_vision_tokens", False):
            li = self._debug_layer
            res.vision_tokens = hs[li][0].float()[vis_pos].cpu().numpy()   # [512, C]
        return res

    def _token_masks(self, seq_len: int):
        """定位 vision token 在最终整条前向序列中的位置。

        predict_language=True 路径下，最后一次前向的输入是
            [prompt tokens (含 <IMG_CONTEXT>) | 贪心生成的 token | driving queries]
        batch=1 无左 padding，因此 prompt 段位置与 language__ids 一一对应。
        """
        ad = self._adaptor_dict
        ids = ad["language__ids"][0]                       # [L_prompt]
        n_prompt = int(ids.shape[0])
        assert n_prompt <= seq_len, f"prompt 长度 {n_prompt} > 序列长度 {seq_len}"
        vis_mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
        vis_mask[:n_prompt] = ids == self.img_context_token_id
        assert vis_mask.any(), "未定位到 vision token"

        last_idx = seq_len - 1                             # 最末 driving query（直接喂 waypoint head）
        n_driving = int(self._len_driving) if self._len_driving else 0
        last_lang_idx = seq_len - n_driving - 1            # 语言段最后一个 token
        return vis_mask, last_idx, last_lang_idx


# --------------------------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
