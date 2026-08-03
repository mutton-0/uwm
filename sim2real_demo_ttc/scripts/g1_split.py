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


def sized_scene_split(by_scene, estimate_n, truth_n, seed, trials=4000):
    """按**目标事件量**做 scene 级划分（Tier-M/L：500 估计集 + 2k/10k 真值集）。

    scene 数上百时无法穷举，改随机重启：每轮打乱 scene 顺序，
    先攒够 estimate_n 个事件当估计集，再从剩下的攒 truth_n 个当真值集，
    取两侧类型/日夜配比与全集最接近的一轮。多出来的 scene 直接丢弃（手册允许，两集不相交）。
    """
    scenes = sorted(by_scene)
    all_events = [e for s in scenes for e in by_scene[s]]
    ref = composition(all_events)
    rng = random.Random(seed)
    best = (np.inf, None, None)
    for _ in range(trials):
        order = scenes[:]
        rng.shuffle(order)
        est_s, tr_s, n_e, n_t = [], [], 0, 0
        for s in order:
            k = len(by_scene[s])
            if n_e < estimate_n:
                est_s.append(s); n_e += k
            elif n_t < truth_n:
                tr_s.append(s); n_t += k
            else:
                break
        if not est_s or not tr_s:
            continue
        ev_e = [e for s in est_s for e in by_scene[s]]
        ev_t = [e for s in tr_s for e in by_scene[s]]
        cost = (np.abs(composition(ev_e) - ref).sum() + np.abs(composition(ev_t) - ref).sum()
                + abs(len(ev_e) - estimate_n) / estimate_n
                + abs(len(ev_t) - truth_n) / truth_n)
        if cost < best[0]:
            best = (cost, est_s, tr_s)
    assert best[1] is not None, "sized_scene_split 未找到可行划分"
    return best[1], best[2]


def probe_three_way(es, by_scene, n_est, pcfg, seed, trials=4000):
    """估计集内部三分（scene 级）。scene 少时穷举 3^n，多时随机重启。"""
    keys = ["dir", "sel", "test"]
    target = np.array([pcfg[k] for k in keys], dtype=float)
    est_events = [e for s in es for e in by_scene[s]]
    ref_est = composition(est_events)

    def cost_of(assign):
        parts = [[e for s, a in zip(es, assign) if a == i for e in by_scene[s]] for i in range(3)]
        counts = np.array([len(p) for p in parts], dtype=float)
        if (counts == 0).any():
            return np.inf
        # 每份都必须同时含正例(A/B/C)与负例(D)，否则 §7.1 的读数算不出来
        if not all(any(e["event_type"] in "ABC" for e in p) and any(e["event_type"] == "D" for e in p)
                   for p in parts):
            return np.inf
        c = np.abs(counts / n_est - target).sum()
        c += 0.5 * sum(np.abs(composition(p) - ref_est).sum() for p in parts)
        return c

    best = (np.inf, None)
    if len(es) <= 12:
        for code in range(3 ** len(es)):
            assign, c = [], code
            for _ in es:
                assign.append(c % 3); c //= 3
            cost = cost_of(assign)
            if cost < best[0]:
                best = (cost, assign)
    else:
        rng = random.Random(seed)
        # 随机重启 + 单点爬山
        for _ in range(max(1, trials // 20)):
            assign = [rng.randrange(3) for _ in es]
            cost = cost_of(assign)
            for _ in range(20):
                i = rng.randrange(len(es))
                old = assign[i]
                assign[i] = rng.randrange(3)
                c2 = cost_of(assign)
                if c2 < cost:
                    cost = c2
                else:
                    assign[i] = old
            if cost < best[0]:
                best = (cost, assign[:])
    assert best[1] is not None, "估计集 scene 数不足以三分（S_dir/S_sel/S_test 需各 ≥1 个 scene）"
    return {k: [s for s, a in zip(es, best[1]) if a == i] for i, k in enumerate(keys)}


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
        if scfg.get("estimate_n"):
            est_scenes, truth_scenes = sized_scene_split(
                by_scene, int(scfg["estimate_n"]), int(scfg["truth_n"]), scfg["seed"],
                int(scfg.get("search_trials", 4000)))
        else:
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

    # 估计集内部三分（scene 级）：S_dir 提方向 / S_sel 选层 / S_test 报数
    pcfg = scfg["probe_split"]
    es = sorted({e["scene_name"] for e in est})
    probe = probe_three_way(es, by_scene, len(est), pcfg, scfg["seed"] + 1,
                            int(scfg.get("search_trials", 4000)))

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
          f"night={out['estimate']['night']} scenes={len(out['estimate']['scenes'])}")
    print(f"[split] truth   : {out['truth']['n']} events {out['truth']['by_type']} "
          f"night={out['truth']['night']} scenes={len(out['truth']['scenes'])}")
    for k in ("dir", "sel", "test"):
        s = out["probe_split_events"][k]
        print(f"[split] S_{k:<4}: {s['n']} events {s['by_type']} scenes={len(s['scenes'])}")
    print(f"[split] wrote {mining/'splits.json'}")


if __name__ == "__main__":
    main()
