"""Stage A1|CoRL 式权重空间投影(网络驱动 / 数据无关的候选方向生成)。

计划依据:docs/direction_vector_discovery_validation_plan.md Stage A1。

做法(Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories" 的 logit lens 变体):
  FFN 第二个矩阵 W_down: nn.Linear(intermediate=4864 -> hidden=896) 的**列向量**
  即 value vector v_{l,j} ∈ R^896 —— FFN 的输出是这些列向量的加权和。
  把 v 过最终 RMSNorm 再乘输出嵌入 E,得到该 value vector "推高哪些 token" 的 logit 剖面。

三条纪律(缺一不可,否则读数无效):
  1. **必须先 merge LoRA**。SimLingo 的 LoRA 用 target_modules="all-linear",
     down_proj 本身被包了 LoRA;不 merge 就不是部署时的权重。
  2. **不做任何行为断言**。本步只产出"结构标签像不像某个概念"的候选清单。
  3. **必须报多重比较零分布**。24 层 x 4864 = 116,736 个候选取 max 必然虚高,
     单次比较基线(1/sqrt(896)=0.033 之类)在这里是错的。用同规模随机方向池的
     极值零分布 + Bonferroni 高斯尾阈值,两者并列报告。
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_vocab import VRU_RE, SLOW_RE, FAST_RE  # noqa: E402

# 概念库:复用 lang_vocab 的封闭词表(与 T2.5/P1 字面同源)+ 计划 Stage A1 点名的两组
CONCEPT_RE = {
    "vru":       VRU_RE,
    "brake":     SLOW_RE,
    "fast":      FAST_RE,          # 反向对照组,用于构造 brake-fast 对比
    "hazard":    re.compile(r"\b(danger\w*|hazard\w*|risk\w*|unsafe|collision|collide|crash\w*|"
                            r"emergency|accident|threat\w*)\b", re.I),
    "occlusion": re.compile(r"\b(occlu\w*|blocked|blocking|hidden|obscur\w*|blind\s*spot|"
                            r"behind\s+the|out\s+of\s+sight)\b", re.I),
}


def build_concept_tokens(tokenizer, vocab_size):
    """token id -> 是否属于各概念组。对 BPE 前缀空格(Ġ/▁)做归一化后整词匹配。"""
    ids = list(range(vocab_size))
    strs = tokenizer.convert_ids_to_tokens(ids)
    groups = {k: [] for k in CONCEPT_RE}
    for i, s in enumerate(strs):
        if s is None:
            continue
        w = s.replace("Ġ", " ").replace("▁", " ").replace("Ċ", " ").strip()
        if len(w) < 3 or not w.isascii():
            continue
        for k, rgx in CONCEPT_RE.items():
            if rgx.fullmatch(w) or rgx.fullmatch(w.lower()):
                groups[k].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


@torch.no_grad()
def score_block(U, E, gidx, topk=8, chunk=512):
    """U:[n,H] 单位方向; E:[V,H] 输出嵌入 -> 每组 z 分数 + top-k token id。

    z 分数 = (该组 token 的平均 logit - 全词表 logit 均值) / 全词表 logit 标准差。
    对 -U 而言 z 恰好取负,故只算 +U,符号在外面处理。
    """
    n = U.shape[0]
    out = {k: np.zeros(n, dtype=np.float32) for k in gidx}
    top_pos = np.zeros((n, topk), dtype=np.int64)
    top_neg = np.zeros((n, topk), dtype=np.int64)
    for s in range(0, n, chunk):
        u = U[s:s + chunk]
        lg = (u @ E.T).float()                        # [c, V]
        mu = lg.mean(1, keepdim=True); sd = lg.std(1, keepdim=True) + 1e-6
        z = (lg - mu) / sd
        for k, idx in gidx.items():
            out[k][s:s + chunk] = z[:, idx].mean(1).cpu().numpy()
        top_pos[s:s + chunk] = lg.topk(topk, dim=1).indices.cpu().numpy()
        top_neg[s:s + chunk] = (-lg).topk(topk, dim=1).indices.cpu().numpy()
    return out, top_pos, top_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--out-dir", default="/data/ruolin/uwm/sim2real_demo_ttc/results")
    ap.add_argument("--n-null", type=int, default=5000)
    ap.add_argument("--keep-per-group", type=int, default=40)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["model"]["device"] = args.device
    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=False)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 定位 LLM 并 merge LoRA ----------
    # SimLingo 的 LLM 包装:runner.model.language_model 是自定义 `LLM` 壳,
    # 真正的 Qwen2ForCausalLM(含 lm_head) 在 .model 上(且被 PEFT 包了一层)。
    lm = runner.model.language_model.model
    n_lora = sum(1 for _, m in lm.named_modules() if "lora" in m.__class__.__name__.lower())
    merged = False
    if hasattr(lm, "merge_and_unload"):
        lm = lm.merge_and_unload()
        merged = True
    print(f"[A1] LoRA 子模块 {n_lora} 个 -> merge_and_unload={merged}")

    head = lm
    while not hasattr(head, "lm_head"):
        head = head.model if hasattr(head, "model") else head.base_model
    core = head.model
    while not hasattr(core, "layers"):
        core = core.model
    layers = core.layers
    final_norm = core.norm
    E = head.lm_head.weight.detach()                     # [V, H]
    V, H = E.shape
    print(f"[A1] layers={len(layers)}  hidden={H}  vocab={V}  intermediate={layers[0].mlp.down_proj.weight.shape[1]}")

    dev = torch.device(args.device)
    E = E.to(dev)
    gidx_np = build_concept_tokens(runner.tokenizer, V)
    print("[A1] 概念组 token 数:", {k: len(v) for k, v in gidx_np.items()})
    gidx = {k: torch.from_numpy(v).to(dev) for k, v in gidx_np.items() if len(v) >= 3}

    nw = final_norm.weight.detach().to(dev).float()
    eps = getattr(final_norm, "variance_epsilon", 1e-6)

    def rmsnorm_unit(Vmat):
        """RMSNorm 后再归一化为单位向量(方向本身无尺度含义)。"""
        x = Vmat.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * nw
        return (x / (x.norm(dim=-1, keepdim=True) + 1e-8)).to(E.dtype)

    # ---------- 随机方向的极值零分布 ----------
    g = torch.Generator(device=dev).manual_seed(20260829)
    R = torch.randn(args.n_null, H, generator=g, device=dev, dtype=torch.float32)
    R = rmsnorm_unit(R)
    null_scores, _, _ = score_block(R, E, gidx, topk=1)
    del R; torch.cuda.empty_cache()

    # ---------- 逐层扫描 value vectors ----------
    n_layers = len(layers)
    n_int = layers[0].mlp.down_proj.weight.shape[1]
    N_total = n_layers * n_int
    all_scores = {k: np.zeros((n_layers, n_int), dtype=np.float32) for k in gidx}
    all_top_pos = np.zeros((n_layers, n_int, args.topk), dtype=np.int32)
    all_top_neg = np.zeros((n_layers, n_int, args.topk), dtype=np.int32)
    for l in range(n_layers):
        Wd = layers[l].mlp.down_proj.weight.detach().to(dev)     # [H, n_int]
        U = rmsnorm_unit(Wd.T)                                   # [n_int, H]
        sc, tp, tn = score_block(U, E, gidx, topk=args.topk)
        for k in gidx:
            all_scores[k][l] = sc[k]
        all_top_pos[l] = tp; all_top_neg[l] = tn
        print(f"[A1] layer {l:2d}/{n_layers}  " +
              "  ".join(f"{k}:max|z|={np.abs(sc[k]).max():.2f}" for k in gidx))
        del Wd, U; torch.cuda.empty_cache()

    # ---------- 阈值:极值零分布 + Bonferroni 高斯尾 ----------
    from scipy import stats as st
    null_stat, cands = {}, []
    tokid2str = {}
    for k in gidx:
        a = np.abs(null_scores[k])
        mu, sd = float(np.abs(null_scores[k]).mean()), float(null_scores[k].std())
        z_bonf = float(st.norm.ppf(1 - 0.05 / (2 * N_total)))
        thr_bonf = z_bonf * sd
        # 极值零分布:n_null 抽样的 |z| 经验最大值,再按 Gumbel 外推到 N_total 规模
        emp_max = float(a.max())
        extrap = float(sd * st.norm.ppf(1 - 0.5 / N_total))
        thr = max(thr_bonf, extrap)
        null_stat[k] = {"n_null": args.n_null, "null_sd": sd, "null_abs_mean": mu,
                        "null_abs_max_empirical": emp_max,
                        "bonferroni_thresh_abs_z": thr_bonf,
                        "extreme_value_thresh_abs_z": extrap,
                        "threshold_used_abs_z": thr,
                        "n_candidates_tested": N_total}
        A = np.abs(all_scores[k])
        flat = np.argsort(A.ravel())[::-1][: args.keep_per_group]
        for f in flat:
            l, j = int(f // n_int), int(f % n_int)
            z = float(all_scores[k][l, j]); sign = 1 if z > 0 else -1
            tops = (all_top_pos if sign > 0 else all_top_neg)[l, j]
            for t in tops:
                tokid2str.setdefault(int(t), runner.tokenizer.convert_ids_to_tokens(int(t)))
            cands.append({"group": k, "layer": l, "unit": j, "sign": sign,
                          "abs_z": abs(z), "z": z,
                          "passes_null": bool(abs(z) > thr),
                          "top_tokens": [tokid2str[int(t)] for t in tops],
                          "score_by_group": {g2: float(all_scores[g2][l, j]) for g2 in gidx}})

    # 去重(同一 value vector 可能在多组入选),按 |z| 排序
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda d: -d["abs_z"]):
        key = (c["layer"], c["unit"])
        if key in seen:
            continue
        seen.add(key); uniq.append(c)

    # 落盘候选方向本体([N,896],已 RMSNorm+单位化,含符号)
    dirs = np.zeros((len(uniq), H), dtype=np.float32)
    for i, c in enumerate(uniq):
        w = layers[c["layer"]].mlp.down_proj.weight.detach()[:, c["unit"]].to(dev)
        dirs[i] = (c["sign"] * rmsnorm_unit(w[None, :])[0]).float().cpu().numpy()
    np.save(out_dir / "candidate_dirs.npy", dirs)

    n_pass = sum(c["passes_null"] for c in uniq)
    payload = {
        "stage": "A1 权重空间投影(网络驱动/数据无关)",
        "model": "SimLingo (InternVL2-1B / Qwen2-0.5B)",
        "lora_merged": merged, "n_lora_submodules": n_lora,
        "n_layers": n_layers, "hidden": H, "intermediate": n_int,
        "n_candidates_tested": N_total, "vocab": V,
        "concept_groups": {k: int(len(v)) for k, v in gidx_np.items()},
        "multiple_comparison_null": null_stat,
        "n_kept": len(uniq), "n_passing_null": n_pass,
        "layer_profile_max_abs_z": {k: np.abs(all_scores[k]).max(1).tolist() for k in gidx},
        "candidates": uniq,
        "discipline": "本步不对任何候选做行为断言;是否可信由 Stage C/D/E 的数据驱动验证决定。",
    }
    (out_dir / "candidate_directions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[A1] 保留 {len(uniq)} 个候选,其中 {n_pass} 个超出多重比较零分布阈值")
    for k in gidx:
        print(f"  [{k}] 阈值|z|={null_stat[k]['threshold_used_abs_z']:.2f} "
              f"(Bonferroni {null_stat[k]['bonferroni_thresh_abs_z']:.2f} / "
              f"极值外推 {null_stat[k]['extreme_value_thresh_abs_z']:.2f}); "
              f"观测 max|z|={np.abs(all_scores[k]).max():.2f}")
    print(f"[A1] wrote results/candidate_directions.json + candidate_dirs.npy {dirs.shape}")


if __name__ == "__main__":
    main()
