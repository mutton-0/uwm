"""T1-L Step 1｜语义配对臂:用**语言对**定义危险轴(guide 附录 A)。

为什么换语言配对:RepE 的 LAT 刺激对本来就是语言对——同一段上下文只差一个概念词,
逐 token 可控。我们此前的视觉配对(clean/ghost 时序切片 + 几何匹配)天然是脏的:
两帧之间自车动了、光照变了、目标之外的一切都在变。E1/E2 都 FAIL 之后,
本臂承担两个职能:**仪器自证**(Step 2)与**失效定位**(Step 3)。

Step 1 做的事:
  同一张图过两遍,只改 prompt 里那一句场景描述(danger vs safe),
  δ = h(danger) − h(safe) 取在 **driving query token**(动作预测前的最后位置,
  同时看得到图像与 prompt)→ 逐层 PCA → v_danger^lang,报 EVR₁。

三个对照(缺一不可):
  ① 置换配对 selectivity —— 把 danger/safe 标签打乱后走同一套 PCA,量"无信息时能造出多少";
  ② 语义特异性 —— 另提一条天气轴 v_weather(晴/雨措辞对),两轴夹角应明显偏离 0,
     否则说明提到的只是"任意语义差"方向,不是危险概念;
  ③ 措辞变体 —— 5–10 组措辞,轴在变体间应稳定(逐变体轴的两两夹角)。

红线:危险措辞只出现在本脚本(离线刺激集)。Step 3 评分回路一律中性 prompt。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 语言刺激对(英文:SimLingo/InternLM2 的训练语言;guide 里的中文示例是意译) ----
# 每组 (danger, safe);token 长度在运行时对齐(不足的一侧补中性填充词并记录)
DANGER_SAFE = [
    ("A pedestrian is about to cross right in front of us, very dangerous.",
     "The road right in front of us is completely clear, very safe."),
    ("There is an immediate collision risk with a person ahead.",
     "There is no collision risk with anything ahead."),
    ("Someone has suddenly stepped into our lane, we must brake now.",
     "Nobody has stepped into our lane, we can keep going now."),
    ("A vulnerable road user is dangerously close to our path.",
     "A wide empty margin is comfortably far from our path."),
    ("Hazard ahead: a pedestrian is crossing and we could hit them.",
     "Clear ahead: the lane is empty and we cannot hit anyone."),
    ("This is an emergency, a person is directly in our way.",
     "This is a normal drive, the way ahead is entirely open."),
]
# 语义特异性对照:天气语义(与危险无关),用来排除"任意语义差"方向
WEATHER_PAIRS = [
    ("It is raining heavily and the road surface is wet.",
     "It is sunny and bright and the road surface is dry."),
    ("Thick fog covers the scene and visibility is poor.",
     "Clear air fills the scene and visibility is good."),
    ("A heavy snowstorm is falling across the whole road.",
     "A warm sunshine is shining across the whole road."),
]


def align_tokens(tok, a: str, b: str, filler: str = " now"):
    """把两句话补到同样的 token 长度 —— 长度差本身会造成 δ 里的位置偏移。"""
    la = len(tok(a, add_special_tokens=False)["input_ids"])
    lb = len(tok(b, add_special_tokens=False)["input_ids"])
    pad = 0
    while la != lb and pad < 24:
        if la < lb:
            a = a.rstrip(".") + filler + "."
            la = len(tok(a, add_special_tokens=False)["input_ids"])
        else:
            b = b.rstrip(".") + filler + "."
            lb = len(tok(b, add_special_tokens=False)["input_ids"])
        pad += 1
    return a, b, la, lb


def pca1(D):
    """[N, C] -> PC1 单位向量 + EVR₁。**不去均值**（这是 RepE 配方的要点，不是笔误）。

    δ 的均值方向**就是**概念方向；先去均值再取 PC1 恰好把要找的东西减掉了。
    RepE 的做法是对"随机顺序的配对差"做 PCA —— 随机顺序使总体均值≈0，
    于是不去均值的 PC1 = 配对差里最一致的那个方向。我们这里配对顺序固定
    （danger − safe），因此直接对未去均值的 δ 取 PC1，
    并用**随机翻转配对顺序**作为置换零分布（对照①）。
    """
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    evr = float(S[0] ** 2 / max(1e-12, (S ** 2).sum()))
    return Vt[0] / (np.linalg.norm(Vt[0]) + 1e-8), evr


def axis_from_deltas(D):
    """[N, L, C] δ -> 逐层 PC1 [L, C] + 逐层 EVR₁ [L]，符号按"danger 端投影为正"定。"""
    N, L, C = D.shape
    V = np.zeros((L, C), dtype=np.float32)
    E = np.zeros(L, dtype=np.float32)
    for l in range(L):
        w, evr = pca1(D[:, l, :])
        if (D[:, l, :] @ w).mean() < 0:
            w = -w
        V[l], E[l] = w, evr
    return V, E


def consistency(D):
    """逐层"配对一致性" r = ‖mean δ‖ / mean‖δ‖ ∈ [0,1]。

    这才是"存不存在一条一致的危险方向"的统计量。**不能用 EVR₁ 做置换检验**:
    不去均值的 SVD 对逐行符号翻转严格不变(D^T D 不变),置换零分布会恒等于观测——
    冒烟跑到的 0.766 vs 0.766 就是这个数学事实,不是巧合。
    r 则对翻转敏感:无一致成分时 mean δ→0。
    """
    m = np.linalg.norm(D.mean(0), axis=-1)                    # [L]
    s = np.linalg.norm(D, axis=-1).mean(0)                    # [L]
    return m / (s + 1e-12)


def heldout_projection(D, seed=0):
    """刺激集对半split:一半提轴,另一半的 δ 投影上去。

    无一致概念方向时，held-out 投影的符号应当是随机的 => 均值≈0、正比例≈0.5。
    这是 Step 1 的**主选择性检验**，也用来选层（选 held-out 一致性最高的层）。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(D))
    a, b = idx[: len(D) // 2], idx[len(D) // 2:]
    V, _ = axis_from_deltas(D[a])
    P = np.einsum("nlc,lc->nl", D[b], V)                      # [n_b, L]
    nrm = np.linalg.norm(D[b], axis=-1) + 1e-12
    cos = P / nrm
    frac_pos = (P > 0).mean(0)                                # [L]
    return cos.mean(0), frac_pos


def cos_by_layer(A, B):
    return np.array([float(np.dot(A[l], B[l]) /
                           (np.linalg.norm(A[l]) * np.linalg.norm(B[l]) + 1e-12))
                     for l in range(A.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-frames", type=int, default=200, help="刺激帧数(clean/ghost/D2 混采)")
    ap.add_argument("--pool-mode", default="query_mean",
                    choices=["query_mean", "query_first", "last_token", "last_lang_token",
                             "vision_mean", "seq_mean"])
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--perm-seeds", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if args.device:
        cfg["model"]["device"] = args.device
    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])

    # ---- 刺激帧:混采,让轴由**语言对比**定义而非图像内容 ----
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    rng = np.random.default_rng(args.seed)
    frames = []
    for e in evs:
        for cond in ("x_clean_frames", "x_ghost_frames"):
            if e[cond]:
                frames.append({"event_id": e["event_id"], "type": e["event_type"],
                               "cond": cond[2:-7], **e[cond][0]})
    rng.shuffle(frames)
    # 按事件类型分层,避免刺激集被某一类主导
    by_t = {}
    for f in frames:
        by_t.setdefault(f["type"], []).append(f)
    per = max(1, args.n_frames // len(by_t))
    stim = [f for t in sorted(by_t) for f in by_t[t][:per]][: args.n_frames]
    print(f"[T1-L] 刺激帧 n={len(stim)}  类型分布 "
          f"{ {t: sum(1 for f in stim if f['type']==t) for t in sorted(by_t)} }")

    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=True)
    tok = runner.tokenizer

    pairs = DANGER_SAFE[: args.variants]
    aligned = []
    for a, b in pairs:
        a2, b2, la, lb = align_tokens(tok, a, b)
        aligned.append((a2, b2))
        print(f"[T1-L] 措辞对 token 长度 {la}/{lb} {'✅' if la == lb else '❌未对齐'}"
              f"   danger={a2!r}")
    wpairs = [align_tokens(tok, a, b)[:2] for a, b in WEATHER_PAIRS]

    def deltas(pair_list, label):
        """对每帧 × 每组措辞跑两次前向，返回 δ [N*V, L, C] 与逐变体索引。"""
        D, vidx = [], []
        for i, fr in enumerate(stim):
            img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
            sp = fr["ego_speed_mps"]
            for vi, (a, b) in enumerate(pair_list):
                ha = runner.infer(img, sp, pool_modes=(args.pool_mode,), context=a)
                hb = runner.infer(img, sp, pool_modes=(args.pool_mode,), context=b)
                D.append(ha.hidden[args.pool_mode] - hb.hidden[args.pool_mode])
                vidx.append(vi)
            if (i + 1) % 20 == 0 or i + 1 == len(stim):
                print(f"[T1-L] {label} {i+1}/{len(stim)}")
        return np.stack(D), np.array(vidx)

    D, vidx = deltas(aligned, "danger/safe")
    V, E = axis_from_deltas(D)
    r = consistency(D)
    ho_cos, ho_pos = heldout_projection(D, seed=args.seed)
    # 选层准则 = held-out 投影一致性最高层（不是 EVR₁——EVR₁ 不含选择性信息）
    peak = int(np.argmax(ho_cos))
    print(f"\n[Step1] v_danger^lang 提出  δ n={len(D)}")
    print(f"        选层(held-out 投影 cos 最大) L*={peak}: "
          f"cos={ho_cos[peak]:.3f}  正比例={ho_pos[peak]:.3f}  "
          f"一致性 r={r[peak]:.3f}  EVR₁={E[peak]:.3f}")
    print(f"        逐层 held-out cos 前五 " +
          ", ".join(f"L{l}:{ho_cos[l]:.3f}" for l in np.argsort(-ho_cos)[:5]))

    # ---- 对照①:置换配对 selectivity(随机翻转 danger/safe 极性) ----
    perm_r = []
    for s in range(args.perm_seeds):
        rs = np.random.default_rng(100 + s)
        sign = rs.choice([-1.0, 1.0], size=len(D))[:, None, None]
        perm_r.append(float(consistency(D * sign)[peak]))
    sel_pass = bool(r[peak] > max(perm_r))
    print(f"[对照①] 置换配对一致性 r@L{peak}: " + " ".join(f"{x:.3f}" for x in perm_r)
          + f"   观测 {r[peak]:.3f}  ->  "
          + ("✅ 高于置换零分布" if sel_pass else "❌ 与置换零分布无法区分"))

    # ---- 对照②:语义特异性(天气轴) ----
    DW, _ = deltas(wpairs, "weather")
    VW, EW = axis_from_deltas(DW)
    cs = cos_by_layer(V, VW)
    print(f"[对照②] cos(v_danger, v_weather)@L{peak} = {cs[peak]:+.3f}   "
          f"全层 |cos| 中位 {np.median(np.abs(cs)):.3f}  ->  "
          + ("✅ 两轴明显不同向" if abs(cs[peak]) < 0.5 else "❌ 与任意语义差方向混同"))

    # ---- 对照③:措辞稳定性 ----
    per_var = []
    for vi in range(len(aligned)):
        Vi, _ = axis_from_deltas(D[vidx == vi])
        per_var.append(Vi)
    cc = [float(np.dot(per_var[i][peak], per_var[j][peak]))
          for i in range(len(per_var)) for j in range(i + 1, len(per_var))]
    print(f"[对照③] 措辞变体两两 cos@L{peak}: 中位 {np.median(cc):+.3f}  "
          f"范围 [{min(cc):+.3f}, {max(cc):+.3f}]  ->  "
          + ("✅ 轴在措辞间稳定" if np.median(cc) > 0.5 else "❌ 轴随措辞漂移"))

    out_dir = work / "results"
    np.save(out_dir / f"v_danger_lang_{args.pool_mode}{args.tag}.npy", V)
    np.save(out_dir / f"v_weather_lang_{args.pool_mode}{args.tag}.npy", VW)
    (out_dir / f"t1l_axis_{args.pool_mode}{args.tag}.json").write_text(json.dumps({
        "pool_mode": args.pool_mode, "n_stim_frames": len(stim), "n_deltas": int(len(D)),
        "variants": [a for a, _ in aligned], "peak_layer": peak,
        "peak_criterion": "held-out 投影 cos 最大",
        "evr1_by_layer": E.tolist(), "evr1_peak": float(E[peak]),
        "consistency_by_layer": r.tolist(), "consistency_peak": float(r[peak]),
        "heldout_cos_by_layer": ho_cos.tolist(), "heldout_frac_pos_by_layer": ho_pos.tolist(),
        "perm_consistency_peak": perm_r,
        "cos_danger_weather_by_layer": cs.tolist(),
        "wording_pairwise_cos_at_peak": cc,
        "selectivity_pass": sel_pass,
        "specificity_pass": bool(abs(cs[peak]) < 0.5),
        "wording_stable": bool(np.median(cc) > 0.5),
    }, indent=2, ensure_ascii=False))
    print(f"[T1-L] wrote results/t1l_axis_{args.pool_mode}{args.tag}.json + v_danger_lang_*.npy")


if __name__ == "__main__":
    main()
