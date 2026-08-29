"""Stage C|交叉印证:Stage A1 权重候选 × Stage B 数据驱动方向 的余弦矩阵 + 优先级队列。

计划依据:direction_vector_discovery_validation_plan.md Stage C + H2。

比较基:
  v_hazard_clean (G 轴, nuScenes region_mean)   v_hazard_carla (G 轴, CARLA in-domain)
  v_danger_lang  (G 轴语言通道)                  v_brake        (F 轴行为锚)
  v_domain       (I 轴输入; 其与其余方向的夹角**即 I 轴干涉角**, 是本矩阵的直接产出)

多重比较控制(计划明写的硬要求):
  单次比较基线 1/sqrt(896)=0.033 在这里是**错的** —— 我们在 N_cand × N_axis 个格子里取 max。
  两条零分布并列:
    ① **结构匹配零分布**:从同一批 FFN value vector 里随机抽 N_null 个(未被概念筛中的),
       走同一套 RMSNorm+单位化,与同一条数据驱动方向算 |cos| —— 这是"概念筛选是否带来额外共线"的正确对照;
    ② **各向同性零分布**:随机高斯单位方向。两者差异本身就说明 value vector 的各向异性有多强。
  阈值 = 对应零分布在 N_cand × N_axis 次抽取下的极值分位(Bonferroni 高斯尾 + 经验极值,取较严者)。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def get_layers_and_norm(device):
    sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
    cfg = OmegaConf.to_container(OmegaConf.load("/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml"),
                                 resolve=True)
    cfg["model"]["device"] = device
    from simlingo_runner import SimLingoRunner
    r = SimLingoRunner(cfg, capture_hidden=False)
    lm = r.model.language_model.model
    if hasattr(lm, "merge_and_unload"):
        lm = lm.merge_and_unload()
    head = lm
    while not hasattr(head, "lm_head"):
        head = head.model if hasattr(head, "model") else head.base_model
    core = head.model
    while not hasattr(core, "layers"):
        core = core.model
    return core.layers, core.norm


def rms_unit(V, nw, eps=1e-6):
    """logit-lens 读数基:过最终 RMSNorm 后单位化(A1 打概念分用的就是这个基)。"""
    x = V.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * nw
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()


def raw_unit(V):
    """**残差流基**:FFN 直接把 value vector 写进残差流,数据驱动方向(v_hazard/v_brake…)
    也是在逐层原始隐状态上拟合的 —— 因此两者可比的基是**原始基**,不是 RMSNorm 后的基。
    修正案 DV/A15:首版 Stage C 误用 RMSNorm 基做余弦,已改为原始基为主、RMSNorm 基作敏感性。"""
    x = V.float()
    return (x / (x.norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-null", type=int, default=116736)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-model", action="store_true", help="跳过结构匹配零分布(只用各向同性零分布)")
    ap.add_argument("--basis", default="raw", choices=["raw", "rmsnorm"],
                    help="raw=残差流基(主口径，与数据驱动方向同基)；rmsnorm=logit-lens 读数基(敏感性)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cand = json.load(open(RES / "candidate_directions.json"))
    C = 896
    AX = {}
    for nm, fn in (("v_hazard_clean", RES / "v_hazard_clean.npy"),
                   ("v_hazard_carla", RES / "v_hazard_carla.npy"),
                   ("v_domain", RES / "v_domain.npy"),
                   ("v_danger_lang", RES.parent / "variants/n1_d2/results/v_danger_lang_query_mean.npy"),
                   ("v_brake", RES.parent / "variants/n1_d2/results/v_brake_query_mean.npy")):
        if Path(fn).exists():
            w = np.load(fn)
            if w.ndim == 2 and w.shape[1] == C:
                AX[nm] = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)
    print(f"[C] 候选 {len(cand['candidates'])} 条；比较基 {list(AX)}；余弦基={args.basis}")

    # 候选向量在所选基下重建（含 A1 定下的符号）
    layers = norm = None
    # 结构匹配零分布：**用全部 24x4864 = 116,736 个 value vector 作为零分布总体**。
    # 修正案 DV/A14：首版用 3000 个随机抽样 + 高斯尾外推，但实测零分布尾部远重于高斯
    # （经验 q99.9 = 0.10~0.15 > 高斯尾阈值 0.071~0.084），高斯外推**偏松**、会虚增命中数。
    # 全总体枚举后，α = 0.05/(N_cand × N_axis) = 5.3e-5 对应总体第 ~6 名，可直接经验取分位，
    # 不再需要任何分布假设。
    null_struct = None
    if not args.no_model:
        layers, norm = get_layers_and_norm(args.device)
        dev = next(layers[0].parameters()).device
        nw = norm.weight.detach().to(dev).float()
        n_int = layers[0].mlp.down_proj.weight.shape[1]; n_L = len(layers)
        taken = {(c["layer"], c["unit"]) for c in cand["candidates"]}
        conv = (lambda W: raw_unit(W)) if args.basis == "raw" else (lambda W: rms_unit(W, nw))
        dirs = np.zeros((len(cand["candidates"]), C), dtype=np.float32)
        for i, c in enumerate(cand["candidates"]):
            W = layers[c["layer"]].mlp.down_proj.weight.detach()[:, [c["unit"]]].T.to(dev)
            dirs[i] = c["sign"] * conv(W)[0]
        np.save(RES / f"candidate_dirs_{args.basis}.npy", dirs)
        null_struct = {}
        for l in range(n_L):
            keep = [j for j in range(n_int) if (l, j) not in taken]
            W = layers[l].mlp.down_proj.weight.detach()[:, keep].T.to(dev)
            null_struct[l] = conv(W)
        n_pop = sum(v.shape[0] for v in null_struct.values())
        print(f"[C] 结构匹配零分布总体 = {n_pop} 个 value vector（全枚举，无抽样）")
        del layers; torch.cuda.empty_cache()

    # 余弦矩阵：候选在**自己的层**上与各轴同层比较；同时报跨层最大
    rows, nulls = [], {}
    for nm, W in AX.items():
        same, best = [], []
        for i, c in enumerate(cand["candidates"]):
            l = c["layer"]
            same.append(float(dirs[i] @ W[l]))
            best.append(float(np.max(np.abs(dirs[i] @ W.T))))
        rows.append({"axis": nm, "cos_same_layer": same, "max_abs_cos_any_layer": best})
        # 零分布
        iso = np.random.default_rng(11).normal(size=(min(args.n_null, 200000), C))
        iso /= np.linalg.norm(iso, axis=1, keepdims=True)
        iso_cos = np.abs(iso @ W[cand["candidates"][0]["layer"]])
        st = None
        if null_struct:
            st = np.concatenate([np.abs(v @ W[l]) for l, v in null_struct.items()])
        n_tests = len(dirs) * len(AX)
        alpha = 0.05 / n_tests
        def thr(a, exact):
            sd = float(a.std())
            d = {"n_null": int(len(a)), "sd": sd, "abs_mean": float(a.mean()),
                 "emp_q999": float(np.percentile(a, 99.9)),
                 "gauss_bonferroni_REJECTED": float(stats.norm.ppf(1 - alpha / 2) * sd),
                 "resolvable": bool(len(a) * alpha >= 5)}
            # 经验分位（唯一被采纳的阈值口径；总体足够大时精确，否则退回经验最大值并标注保守）
            d["empirical_quantile"] = float(np.quantile(a, 1 - alpha)) if d["resolvable"] else float(a.max())
            d["threshold_used"] = d["empirical_quantile"]
            return d
        nulls[nm] = {"isotropic": thr(iso_cos, False), "n_tests": n_tests, "alpha_per_test": alpha}
        if st is not None:
            nulls[nm]["structure_matched"] = thr(st, True)

    # 优先级队列
    queue = []
    for i, c in enumerate(cand["candidates"]):
        per = {}
        for r in rows:
            nm = r["axis"]
            n = nulls[nm].get("structure_matched") or nulls[nm]["isotropic"]
            t = max(n["threshold_used"], nulls[nm]["isotropic"]["threshold_used"])
            per[nm] = {"cos_same_layer": r["cos_same_layer"][i],
                       "max_abs_cos_any_layer": r["max_abs_cos_any_layer"][i],
                       "threshold": t, "passes": abs(r["cos_same_layer"][i]) > t}
        hit = [k for k, v in per.items() if v["passes"]]
        queue.append({"layer": c["layer"], "unit": c["unit"], "sign": c["sign"], "group": c["group"],
                      "abs_z_concept": c["abs_z"], "top_tokens": c["top_tokens"][:5],
                      "cos": per, "double_corroborated_axes": hit,
                      "evidence": "双重印证" if hit else "仅结构标签",
                      "priority_score": max(abs(v["cos_same_layer"]) / max(v["threshold"], 1e-9)
                                            for v in per.values())})
    queue.sort(key=lambda q: -q["priority_score"])
    n_double = sum(q["evidence"] == "双重印证" for q in queue)

    out = {"n_candidates": len(dirs), "axes": list(AX), "nulls": nulls,
           "matrix": rows, "queue": queue, "n_double_corroborated": n_double,
           "H2": ("成立：存在超出多重比较零分布的双重印证方向" if n_double else
                  "不成立：A1 候选与数据驱动方向全部落在零分布内 ⇒ 权重投影对本任务信息量有限，下次可跳过"),
           "discipline": "候选选择(A1)与比较基(B)相互独立；阈值按 N_cand × N_axis 次比较控制。"}
    out["cosine_basis"] = args.basis
    (RES / f"cosine_matrix{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = ["# Stage C｜验证优先级队列（交叉印证后）", "",
             f"> 候选 {len(dirs)} 条 × 比较基 {len(AX)} 条 = {len(dirs)*len(AX)} 次比较；",
             f"> 阈值取 结构匹配零分布 与 各向同性零分布 的较严者（Bonferroni / 极值外推取大）。",
             f"> **双重印证候选：{n_double} 条**。H2 判定：{out['H2']}", "",
             "| # | 层 | unit | 概念组 | \\|z\\| | 最近邻 token | " +
             " | ".join(AX) + " | 证据来源 |",
             "|---|---|---|---|---|---|" + "---|" * (len(AX) + 1)]
    for i, q in enumerate(queue[:40], 1):
        cs = " | ".join(f"{q['cos'][a]['cos_same_layer']:+.3f}" +
                        ("**" if q["cos"][a]["passes"] else "") for a in AX)
        lines.append(f"| {i} | L{q['layer']} | {q['unit']} | {q['group']} | {q['abs_z_concept']:.2f} | "
                     f"`{' '.join(q['top_tokens'][:3])}` | {cs} | {q['evidence']} |")
    lines += ["", f"阈值（|cos| 超过即记 `**`）：全总体经验分位，α = 0.05/{len(dirs)*len(AX)} = "
                  f"{0.05/(len(dirs)*len(AX)):.2e}（**不使用高斯尾外推** —— 零分布尾部远重于高斯，见 DV/A14）",
              "", "| 轴 | 结构匹配零分布 n | sd | 经验阈值 | 各向同性零分布 经验阈值 | 采用阈值 |",
              "|---|---|---|---|---|---|"]
    for nm in AX:
        s = nulls[nm].get("structure_matched"); iso = nulls[nm]["isotropic"]
        used = max(s["threshold_used"], iso["threshold_used"]) if s else iso["threshold_used"]
        lines.append(f"| {nm} | " + (f"{s['n_null']} | {s['sd']:.4f} | {s['threshold_used']:.4f} | "
                                     if s else "— | — | — | ") +
                     f"{iso['threshold_used']:.4f} | **{used:.4f}** |")
    (RES / f"priority_queue{args.tag}.md").write_text("\n".join(lines) + "\n")
    print(f"[C] 双重印证 {n_double}/{len(queue)} 条；H2：{out['H2']}")
    print(f"[C] wrote results/cosine_matrix{args.tag}.json + results/priority_queue{args.tag}.md")
    for q in queue[:8]:
        print(f"    L{q['layer']:2d} u{q['unit']:4d} {q['group']:9s} score={q['priority_score']:.2f} "
              f"{q['evidence']} hit={q['double_corroborated_axes']} :: {' '.join(q['top_tokens'][:3])}")


if __name__ == "__main__":
    main()
