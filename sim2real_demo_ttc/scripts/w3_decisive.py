"""W3｜决定性实验:视觉危险在 $v_{brake}$ 上的投影差（guide 附录 B W3）。

问题:模型的**行为**对危险有微弱但显著的梯度（b-AUC ~0.555）。
那么在**行为中介态**（query token × 行为定义轴 $v_{brake}$）上读，
能不能读到比外部行为更灵敏的信号？

主读数（预注册）= 投影差的**逐事件 AUC(A vs D2a)**，与行为端 AUC 对标。

判定（guide 写死）:
  AUC 显著 > 行为端  ⇒ 内部读数更灵敏，评分轴就是它，进 board.json；
  AUC ≈ 行为端       ⇒ 内部无更多存货，SimLingo 终局画像定案；
  AUC < 0.5 或不显著 ⇒ 行为的中介不在该线性方向上，记录后收束。

纪律:$v_{brake}$ 必须**折外拟合**——用同一批事件既提方向又报数是自证。
因此走 scene 级 K 折：折内提方向 + 嵌套选层，折外投影，最后汇总所有 held-out 投影算一次 AUC。
并列报随机方向地板与标签置换零分布（同一 CV+选层流程）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402
from n1_readout import auc  # noqa: E402
from t1q_axes import brake_direction  # noqa: E402


def hv_se(A, n1, n2):
    """Hanley–McNeil 的 AUC 标准误。"""
    Q1 = A / (2 - A); Q2 = 2 * A * A / (1 + A)
    return np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A * A) + (n2 - 1) * (Q2 - A * A)) / (n1 * n2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="query_mean")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rand-seeds", type=int, default=5)
    ap.add_argument("--perm-seeds", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))
    matched = {t: set((work / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a", "D2b", "D2c") if (work / "mining" / f"matched_{t}.txt").exists()}

    by_type = defaultdict(list)
    for eid, e in items.items():
        t = evmap[eid]["event_type"]
        if t in matched and eid not in matched[t]:
            continue
        by_type[t].append(e)

    scenes = sorted({e["scene"] for v in by_type.values() for e in v})
    rng = np.random.default_rng(args.seed)
    fold_of = {s: int(i % args.folds) for i, s in enumerate(rng.permutation(scenes))}
    ALL = [e for t in by_type for e in by_type[t]]

    def run(mode: str, rseed: int = 0):
        """mode: brake / random / permuted。返回 {type: held-out 投影数组} 与逐折峰层。"""
        held = defaultdict(list)
        peaks = []
        for f in range(args.folds):
            tr = [e for e in ALL if fold_of[e["scene"]] != f]
            te = [e for e in ALL if fold_of[e["scene"]] == f]
            if len(tr) < 50 or not te:
                continue
            # 折内再切一块做选层（嵌套），不碰当前 held-out 折
            insc = sorted({e["scene"] for e in tr})
            selsc = set(insc[: max(1, len(insc) // 4)])
            fit = [e for e in tr if e["scene"] not in selsc]
            sel = [e for e in tr if e["scene"] in selsc] or tr

            if mode == "random":
                rs = np.random.default_rng(7000 + 100 * rseed + f)
                L_, C_ = fit[0]["h_clean"].shape
                v = rs.normal(size=(L_, C_)).astype(np.float32)
                v /= np.linalg.norm(v, axis=1, keepdims=True)
            elif mode == "permuted":
                # 打乱刹/不刹标签后走同一套提取：量"方向不含行为信息时能造出多少"
                rs = np.random.default_rng(9000 + 100 * rseed + f)
                shuf = list(fit)
                idx = rs.permutation(len(shuf))
                fake = [dict(e, v_clean=shuf[idx[i]]["v_clean"], v_ghost=shuf[idx[i]]["v_ghost"])
                        for i, e in enumerate(shuf)]
                v, _ = brake_direction(fake, seed=args.seed)
            else:
                v, _ = brake_direction(fit, seed=args.seed)

            # 选层：S_sel 上 |ρ(投影, v_plan)| 最大（与 T1-Q 同准则）
            rho = []
            for l in range(v.shape[0]):
                x = np.array([float(e["h_clean"][l] @ v[l]) for e in sel]
                             + [float(e["h_ghost"][l] @ v[l]) for e in sel])
                y = np.array([e["v_clean"] for e in sel] + [e["v_ghost"] for e in sel])
                r_ = stats.spearmanr(x, y).correlation
                rho.append(abs(r_) if np.isfinite(r_) else np.nan)
            L = int(np.nanargmax(rho))
            peaks.append(L)
            for e in te:
                t = evmap[e["meta"]["event_id"]]["event_type"]
                held[t].append(float((e["h_ghost"][L] - e["h_clean"][L]) @ v[L]))
        return {k: np.array(v_) for k, v_ in held.items()}, peaks

    print(f"[W3] pool={args.pool_mode}  {args.folds} 折 scene 级 CV  "
          f"事件 A/{len(by_type['A'])} D2a/{len(by_type['D2a'])}")

    H, peaks = run("brake")
    A, N = H.get("A", np.zeros(0)), H.get("D2a", np.zeros(0))
    a_main, p_main = auc(A, N)
    se = hv_se(a_main, len(A), len(N))
    print(f"[W3] 逐折峰层={peaks}")
    print(f"\n[主读数] 投影差 AUC(A vs D2a) = {a_main:.3f}  "
          f"95% CI [{a_main-1.96*se:.3f}, {a_main+1.96*se:.3f}]  p={p_main:.3g}  "
          f"n={len(A)}/{len(N)}")

    # 行为端对标
    bA = np.array([e["b"] for e in by_type["A"]])
    bN = np.array([e["b"] for e in by_type["D2a"]])
    a_beh, p_beh = auc(bA, bN)
    se_b = hv_se(a_beh, len(bA), len(bN))
    print(f"[对标]   行为端 b   AUC(A vs D2a) = {a_beh:.3f}  "
          f"95% CI [{a_beh-1.96*se_b:.3f}, {a_beh+1.96*se_b:.3f}]  p={p_beh:.3g}")

    # 地板
    rand = []
    for s in range(args.rand_seeds):
        Hr, _ = run("random", s)
        rand.append(auc(Hr.get("A", np.zeros(0)), Hr.get("D2a", np.zeros(0)))[0])
    perm = []
    for s in range(args.perm_seeds):
        Hp, _ = run("permuted", s)
        perm.append(auc(Hp.get("A", np.zeros(0)), Hp.get("D2a", np.zeros(0)))[0])
    print(f"\n[地板①] 随机方向   " + " ".join(f"{x:.3f}" for x in rand)
          + f"   最大 {max(rand):.3f}")
    print(f"[地板②] 标签置换   " + " ".join(f"{x:.3f}" for x in perm)
          + f"   最大 {max(perm):.3f}")

    # 其余负类（敏感性）
    extra = {}
    for t in ("D2b", "D2c"):
        if t in H and len(H[t]) >= 20:
            a_, p_ = auc(A, H[t])
            extra[t] = {"n": int(len(H[t])), "auc": float(a_), "p": float(p_)}
            print(f"[敏感性] AUC(A vs {t}) = {a_:.3f}  p={p_:.3g}  n={len(H[t])}")

    floor = max(max(rand), max(perm))
    if a_main - 1.96 * se > a_beh + 1.96 * se_b:
        verdict = "内部读数显著优于行为端 ⇒ 评分轴成立"
    elif a_main - 1.96 * se > floor and p_main < 0.05:
        verdict = "显著高于地板但不优于行为端 ⇒ 内部无更多存货"
    elif a_main < 0.5:
        verdict = "AUC<0.5 ⇒ 行为的中介不在该线性方向上"
    else:
        verdict = "不显著 ⇒ 行为的中介不在该线性方向上（或功效不足）"
    print(f"\n[W3 判定] 主读数 {a_main:.3f} vs 行为端 {a_beh:.3f} vs 地板 {floor:.3f}\n"
          f"          => {verdict}")

    out = {"pool_mode": args.pool_mode, "folds": args.folds, "peaks": peaks,
           "main": {"auc": float(a_main), "ci95": [float(a_main - 1.96 * se), float(a_main + 1.96 * se)],
                    "p": float(p_main), "n_pos": int(len(A)), "n_neg": int(len(N))},
           "behavior_ref": {"auc": float(a_beh),
                            "ci95": [float(a_beh - 1.96 * se_b), float(a_beh + 1.96 * se_b)],
                            "p": float(p_beh)},
           "floor_random": [float(x) for x in rand],
           "floor_permuted": [float(x) for x in perm],
           "sensitivity": extra, "verdict": verdict}
    (work / "results" / f"w3_decisive_{args.pool_mode}{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[W3] wrote results/w3_decisive_{args.pool_mode}{args.tag}.json")


if __name__ == "__main__":
    main()
