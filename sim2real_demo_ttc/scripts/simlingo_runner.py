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

    def _register_hooks(self):
        """在 LLM 每个 decoder layer 上挂 forward hook 抓 hidden states。"""
        self._layer_outputs: List[torch.Tensor] = []
        self._adaptor_dict = None
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
        return pv.view(1, 1, p, c, h, w)                              # [B=1, T=1, P, C, H, W]

    def build_prompt(self, speed_mps: float, n_patches: int):
        mcfg = self.cfg["model"]
        speed = round(float(speed_mps), 1)
        tp = np.asarray(mcfg["target_point_m"], dtype=np.float32)     # [[x1,y1],[x2,y2]]

        if mcfg["prompt_mode"] in ("target_point", "target_point_command"):
            prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        elif mcfg["prompt_mode"] == "command":
            prompt_tp = mcfg["fixed_command"]
        else:
            raise ValueError(mcfg["prompt_mode"])

        if mcfg["use_cot"]:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} What should the ego do next?"
        else:
            prompt = f"Current speed: {speed} m/s. {prompt_tp} Predict the waypoints."

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

    def build_driving_input(self, img_rgb: np.ndarray, speed_mps: float):
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.projection import get_camera_extrinsics, get_camera_intrinsics

        pv = self.build_pixel_values(img_rgb)
        n_patches = pv.shape[2]
        b, t, p, c, H, W = pv.shape
        query, prompt, ids, valid, placeholder, tp = self.build_prompt(speed_mps, n_patches)

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
    def infer(self, img_rgb: np.ndarray, speed_mps: float, pool_modes=("vision_mean", "last_token")) -> InferResult:
        torch.manual_seed(int(self.cfg["model"]["seed"]))
        di, prompt = self.build_driving_input(img_rgb, speed_mps)

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
        pooled = {m: np.zeros((len(hs), hs[0].shape[-1]), dtype=np.float32) for m in pool_modes}
        for li, h in enumerate(hs):
            hf = h[0].float()
            if "vision_mean" in pooled:
                pooled["vision_mean"][li] = hf[vis_mask].mean(0).cpu().numpy()
            if "last_token" in pooled:
                pooled["last_token"][li] = hf[last_idx].cpu().numpy()
            if "last_lang_token" in pooled:
                pooled["last_lang_token"][li] = hf[last_lang_idx].cpu().numpy()
        res.hidden = pooled
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
