"""P1｜资格赛：先证明阳性对照候选「有货」（guide 附录 D）。

只测外部，不碰内部。两项达标才算通过：
  ① 行为端 b-AUC(A vs D2a) 显著 > 0.5 且逐样本可判别
     —— SimLingo 恰恰此项 0.534（p=0.157）不显著；
  ② 危险提及率 A 类显著高于 D2a（CoC 文本，沿用 T2.5 的封闭词表）。

预注册口径与 SimLingo 完全一致，**不得为阳性对照换更友好的设定**：
  * 正例 = A（VRU 突现），负例 = 几何匹配后的 D2a（matched_D2a.txt）；
  * b = v_plan(clean) − v_plan(ghost)，正 = 危险帧规划得更慢；
  * scene 级 bootstrap；AUC 用 Mann–Whitney。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lang_vocab import VRU_RE, SLOW_RE  # noqa: E402  与 T2.5 字面同一份词表

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "variants" / "n1_d2"


def boot(vals, scenes, n_boot=4000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(float(v))
    keys = list(by)
    if not keys:
        return float("nan"), [float("nan")] * 2
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(np.mean([float(v) for v in vals])), \
        [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def auc_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan"), [float("nan")] * 2
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    A = float(u.statistic / (len(a) * len(b)))
    Q1, Q2 = A / (2 - A), 2 * A * A / (1 + A)
    se = np.sqrt((A * (1 - A) + (len(a) - 1) * (Q1 - A * A) + (len(b) - 1) * (Q2 - A * A))
                 / (len(a) * len(b)))
    return A, float(u.pvalue), [A - 1.96 * se, A + 1.96 * se]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ego-anchor", default="clean", choices=["clean", "per_frame"],
                    help="clean=自车运动史锚定 clean 帧（与 SimLingo 的 prompt_anchor 同构，"
                         "主读数口径）；per_frame=各用各的（敏感性）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out_path = Path(args.out).resolve() if args.out else \
        (WORK / "results" / f"p1_alpamayo_raw_{args.ego_anchor}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    evs = [json.loads(l) for l in open(WORK / "mining" / "events_all.jsonl")]
    matched = set((WORK / "mining" / "matched_D2a.txt").read_text().split())
    want = [e for e in evs if e["event_type"] == "A"
            or (e["event_type"] == "D2a" and e["event_id"] in matched)]

    from alpamayo_runner import AlpamayoRunner
    r = AlpamayoRunner(device=args.device)

    # 覆盖预筛（adapter 需 t0 前 1.6s / 后 6.4s 的 ego_pose）
    usable, dropped = [], defaultdict(int)
    for e in want:
        ok = all(r.covers(e["scene_name"], e[c][0]["t"])
                 for c in ("x_clean_frames", "x_ghost_frames"))
        if ok:
            usable.append(e)
        else:
            dropped[e["event_type"]] += 1
    cov = {t: {"usable": sum(1 for e in usable if e["event_type"] == t),
               "total": sum(1 for e in want if e["event_type"] == t)} for t in ("A", "D2a")}
    print(f"[P1] 覆盖预筛：" + "　".join(
        f"{t} {v['usable']}/{v['total']} ({v['usable']/max(1,v['total']):.1%})" for t, v in cov.items()))
    if args.limit:
        usable = usable[: args.limit]

    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [e for e in usable if e["event_id"] not in done]
    print(f"[P1] 待跑 {len(todo)} / {len(usable)}（已有 {len(done)}）")
    t_start = time.time()
    for i, e in enumerate(todo):
        rec = {"type": e["event_type"], "scene": e["scene_name"]}
        try:
            t_anchor = (e["x_clean_frames"][0]["t"] if args.ego_anchor == "clean" else None)
            for cond in ("clean", "ghost"):
                res = r.infer(e["scene_name"], e[f"x_{cond}_frames"][0]["t"],
                              ego_anchor_t=t_anchor)
                rec[f"v_{cond}"] = res.v_plan
                rec[f"cot_{cond}"] = res.cot
            rec["b"] = rec["v_clean"] - rec["v_ghost"]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        done[e["event_id"]] = rec
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            out_path.write_text(json.dumps(done, ensure_ascii=False))
            el = time.time() - t_start
            print(f"[P1] {i+1}/{len(todo)}  {el:.0f}s ({el/(i+1):.2f}s/event)")
    out_path.write_text(json.dumps(done, ensure_ascii=False))

    # ---------------- 判定 ----------------
    rows = [dict(v, id=k) for k, v in done.items() if "b" in v]
    A = [x for x in rows if x["type"] == "A"]
    N = [x for x in rows if x["type"] == "D2a"]
    print(f"\n[P1] 有效读数 A={len(A)}  D2a={len(N)}（失败 {sum(1 for v in done.values() if 'error' in v)}）")

    a, p, ci = auc_ci([x["b"] for x in A], [x["b"] for x in N])
    print(f"\n=== ① 行为端 b-AUC(A vs D2a) ===")
    print(f"  AUC = {a:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  p={p:.3g}")
    print(f"  参照 SimLingo 同口径：0.534（p=0.157，不显著）")
    hit1 = bool(np.isfinite(a) and p < 0.05 and ci[0] > 0.5)
    print(f"  -> {'✅ 达标（显著且 CI 下界 >0.5）' if hit1 else '❌ 未达标'}")

    print(f"\n=== ② CoC 危险提及率（ghost 帧，沿用 T2.5 封闭词表）===")
    rate = {}
    for key, rx, lab in (("vru", VRU_RE, "提到 VRU"), ("slow", SLOW_RE, "说要减速/让行")):
        va = [bool(rx.search(x.get("cot_ghost", ""))) for x in A]
        vn = [bool(rx.search(x.get("cot_ghost", ""))) for x in N]
        ma, cia = boot(va, [x["scene"] for x in A])
        mn, cin = boot(vn, [x["scene"] for x in N])
        pf = stats.fisher_exact([[sum(va), len(va) - sum(va)], [sum(vn), len(vn) - sum(vn)]])[1]
        rate[key] = {"A": ma, "A_ci": cia, "D2a": mn, "D2a_ci": cin, "fisher_p": float(pf)}
        print(f"  {lab:14s} A {ma:.3f} [{cia[0]:.3f},{cia[1]:.3f}]　"
              f"D2a {mn:.3f} [{cin[0]:.3f},{cin[1]:.3f}]　Fisher p={pf:.3g}")
    hit2 = bool(rate["vru"]["fisher_p"] < 0.05 and rate["vru"]["A"] > rate["vru"]["D2a"])
    print(f"  -> {'✅ 达标（A 显著高于 D2a）' if hit2 else '❌ 未达标'}")
    print(f"  参照 SimLingo：VRU 提及 0.205 vs 0.092（p=1.5e-4，达标）")

    verdict = ("资格赛通过 ——「有货」经验证实，可作阳性对照" if (hit1 and hit2)
               else "资格赛未过 —— 不能作阳性对照，结果入档并回 P0")
    print(f"\n[P1 判定] 行为端={hit1}  语言端={hit2}  => **{verdict}**")

    res = {
        "model": "Alpamayo-1 (R1-10B)", "ego_anchor": args.ego_anchor, "n": {"A": len(A), "D2a": len(N)},
        "coverage": cov,
        "behavior": {"auc": a, "ci95": ci, "p": p, "hit": hit1,
                     "simlingo_ref": {"auc": 0.534, "p": 0.157, "hit": False},
                     "b_mean_A": float(np.mean([x["b"] for x in A])),
                     "b_mean_D2a": float(np.mean([x["b"] for x in N])),
                     "pass_rate_A": float(np.mean([x["b"] >= 0.5 for x in A]))},
        "language": rate, "language_hit": hit2,
        "verdict": verdict, "qualified": bool(hit1 and hit2),
        "deviations": [
            "Alpamayo 训练于 NVIDIA PhysicalAI-AV，对 nuScenes 同样是 OOD —— 非「按构造保证有货」",
            f"相机 FOV 失配：Alpamayo 期待 120°广角+30°长焦，nuScenes 均为 70°，只能近似映射",
            f"覆盖预筛：adapter 需 t0 前 1.6s/后 6.4s 的 ego_pose，"
            f"A 可用 {cov['A']['usable']}/{cov['A']['total']}、D2a {cov['D2a']['usable']}/{cov['D2a']['total']}",
            "轨迹采样随机（扩散+VLM rollout），clean/ghost 复用同一 seed 以免采样噪声进入 b",
        ],
    }
    (WORK / "results" / f"p1_alpamayo_{args.ego_anchor}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[P1] wrote results/p1_alpamayo_{args.ego_anchor}.json")


if __name__ == "__main__":
    main()
