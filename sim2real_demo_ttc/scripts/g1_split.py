"""G1 划分（手册 §5.3 + §7.1）。

- 估计集 / 真值集（Tier-S 的 "500 vs 10k" 代码演练）：默认按 **scene** 划分，杜绝泄漏；
  分层目标 = 事件类型 × 日夜配比与全集一致。
- 估计集内部再按 scene 三分：S_dir(提方向) / S_sel(选层) / S_test(报数)，选层与报数分离。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


def composition(events):
    c = Counter(e["event_type"] for e in events)
    n = max(1, len(events))
    v = np.array([c.get(t, 0) for t in "ABCD"] + [sum(e["is_night"] for e in events)], dtype=float)
    return v / n


def scene_split(by_scene, target_frac, seed, all_events, max_trials=200000):
    """在所有 scene 二分方案里找分层最好的一个。

    scene 数少时穷举，多时随机采样。代价 =
        两侧 (类型分布, 夜间比例) 与全集的 L1 距离 + 事件量偏离目标比例的惩罚。
    贪心逐场景分配在这里不可用：空的一侧代价恒为 0，会把所有 scene 都塞给同一侧。
    """
    scenes = sorted(by_scene)
    n = len(scenes)
    ref = composition(all_events)
    n_total = len(all_events)

    def cost_of(mask):
        est_s = [s for i, s in enumerate(scenes) if mask >> i & 1]
        tr_s = [s for i, s in enumerate(scenes) if not (mask >> i & 1)]
        if not est_s or not tr_s:
            return np.inf, None, None
        ev_e = [e for s in est_s for e in by_scene[s]]
        ev_t = [e for s in tr_s for e in by_scene[s]]
        if not ev_e or not ev_t:
            return np.inf, None, None
        c = np.abs(composition(ev_e) - ref).sum() + np.abs(composition(ev_t) - ref).sum()
        c += 4.0 * abs(len(ev_e) / n_total - target_frac)
        return c, est_s, tr_s

    candidates = range(1, 2 ** n - 1) if n <= 18 else \
        (random.Random(seed).getrandbits(n) for _ in range(max_trials))
    best = (np.inf, None, None)
    for m in candidates:
        c, e, t = cost_of(m)
        if c < best[0]:
            best = (c, e, t)
    assert best[1] is not None, "无法划分：至少需要 2 个含事件的 scene"
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "tier_s.yaml"))
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    scfg = cfg["split"]
    work = Path(cfg["paths"]["work_dir"])
    mining = work / "mining"
    events = [json.loads(l) for l in open(mining / "events_all.jsonl")]

    by_scene = defaultdict(list)
    for e in events:
        by_scene[e["scene_name"]].append(e)

    if scfg["unit"] == "scene":
        est_scenes, truth_scenes = scene_split(by_scene, scfg["estimate_frac"], scfg["seed"], events)
        est = [e for s in est_scenes for e in by_scene[s]]
        truth = [e for s in truth_scenes for e in by_scene[s]]
    else:  # event 级分层随机（有 scene 泄漏，仅作对照）
        rng = random.Random(scfg["seed"])
        est, truth = [], []
        strata = defaultdict(list)
        for e in events:
            strata[(e["event_type"], e["is_night"])].append(e)
        for lst in strata.values():
            rng.shuffle(lst)
            k = int(round(len(lst) * scfg["estimate_frac"]))
            est += lst[:k]
            truth += lst[k:]
        est_scenes = sorted({e["scene_name"] for e in est})
        truth_scenes = sorted({e["scene_name"] for e in truth})

    # 估计集内部三分（scene 级，穷举分配使**事件量**比例贴近 dir/sel/test 目标）
    pcfg = scfg["probe_split"]
    es = sorted({e["scene_name"] for e in est})
    keys = ["dir", "sel", "test"]
    target = np.array([pcfg[k] for k in keys], dtype=float)
    n_est = len(est)
    best = (np.inf, None)
    for code in range(3 ** len(es)):
        assign, c = [], code
        for _ in es:
            assign.append(c % 3)
            c //= 3
        parts = [[e for s, a in zip(es, assign) if a == i for e in by_scene[s]] for i in range(3)]
        counts = np.array([len(p) for p in parts], dtype=float)
        if (counts == 0).any():          # 三份都必须非空
            continue
        # 每份都必须同时含正例(A/B/C)与负例(D)，否则 §7.1 的 ρ_TTC / Acc(ghost vs D) 算不出来
        ok = all(any(e["event_type"] in "ABC" for e in p) and any(e["event_type"] == "D" for e in p)
                 for p in parts)
        if not ok:
            continue
        ref_est = composition(est)
        cost = np.abs(counts / n_est - target).sum()
        cost += 0.5 * sum(np.abs(composition(p) - ref_est).sum() for p in parts)
        if cost < best[0]:
            best = (cost, assign)
    assert best[1] is not None, "估计集 scene 数不足以三分（S_dir/S_sel/S_test 需各 ≥1 个 scene）"
    probe = {k: [s for s, a in zip(es, best[1]) if a == i] for i, k in enumerate(keys)}

    def summarize(evs):
        return {"n": len(evs), "by_type": dict(Counter(e["event_type"] for e in evs)),
                "night": sum(e["is_night"] for e in evs),
                "scenes": sorted({e["scene_name"] for e in evs})}

    out = {
        "unit": scfg["unit"],
        "estimate": summarize(est),
        "truth": summarize(truth),
        "probe_split_scenes": probe,
        "probe_split_events": {k: summarize([e for e in est if e["scene_name"] in v]) for k, v in probe.items()},
    }
    (mining / "splits.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (mining / "split_estimate.txt").write_text("\n".join(e["event_id"] for e in est) + "\n")
    (mining / "split_truth.txt").write_text("\n".join(e["event_id"] for e in truth) + "\n")

    print(f"[split] unit={scfg['unit']}")
    print(f"[split] estimate: {out['estimate']['n']} events {out['estimate']['by_type']} "
          f"night={out['estimate']['night']} scenes={out['estimate']['scenes']}")
    print(f"[split] truth   : {out['truth']['n']} events {out['truth']['by_type']} "
          f"night={out['truth']['night']} scenes={out['truth']['scenes']}")
    for k in ("dir", "sel", "test"):
        s = out["probe_split_events"][k]
        print(f"[split] S_{k:<4}: {s['n']} events {s['by_type']} scenes={s['scenes']}")
    print(f"[split] wrote {mining/'splits.json'}")


if __name__ == "__main__":
    main()
