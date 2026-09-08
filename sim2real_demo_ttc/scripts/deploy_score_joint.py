"""联合打分：把四候选在同一 token 上的轨迹作为**同一批 proposal** 送进官方 PDMScorer。

## 为什么必须联合打
`pdm_scorer.py:257-260`：
```
norm_constant_progress = np.max(masked_progress)      # 对 proposal 集合取 max
normalized_progress    = clip(progress_raw / norm_constant_progress, 0, 1)
```
`ego_progress` 是**在 proposal 集合内归一化**的。一次只打一条 ⇒ max 就是它自己
⇒ 恒等于 1.0，既丢掉全部区分度，又抬高 pdm_score。
四家一起打，progress 才在四者之间可比。

（官方 NAVSIM 用 PDM-Closed 规划器的进展做归一化常数；我们没有那条基线，
  故改用"四候选之内取 max"。这是**相对**进展，只可在本文四家之间比较，
  不可与他人发布的绝对 PDM 分数比。这一条与非反应式 v1 口径一并声明。）
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, "/data/ruolin/uwm")
CACHE = "/data/dataset/navsim/exp/metric_cache"
MODELS = ["simlingo", "dd", "ltf", "ddv2"]


def comfort_metrics(poses, step=0.5):
    p = np.vstack([[0.0, 0.0], np.asarray(poses, float)[:, :2]])
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / step
    a = np.diff(v) / step
    j = np.diff(a) / step if len(a) > 1 else np.array([0.0])
    return {"jerk_rms": float(np.sqrt(np.mean(j ** 2))),
            "max_decel": float(max(-a.min(), 0.0)) if len(a) else 0.0,
            "max_accel": float(max(a.max(), 0.0)) if len(a) else 0.0}


def main():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer, PDMScorerConfig)
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

    trajs = {m: json.load(open(RES / f"deploy_traj_{m}.json")) for m in MODELS}
    by_tok = defaultdict(dict)
    for m, d in trajs.items():
        for r in d["records"]:
            by_tok[r["token"]][m] = r
    toks = [t for t, v in by_tok.items() if len(v) == len(MODELS)]
    print(f"[JOINT] 四家都成功的 token: {len(toks)} / {len(by_tok)}")

    loader = MetricCacheLoader(Path(CACHE))
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    sim = PDMSimulator(ps); scorer = PDMScorer(ps, PDMScorerConfig())
    fs = TrajectorySampling(num_poses=8, interval_length=0.5)

    out = defaultdict(list); fails = defaultdict(int)
    for i, tok in enumerate(toks):
        try:
            mc = loader.get_from_token(tok)
            arr = []
            for m in MODELS:
                poses = np.asarray(by_tok[tok][m]["poses"], float)
                it = transform_trajectory(Trajectory(poses, fs), mc.ego_state)
                arr.append(get_trajectory_as_array(it, ps, mc.ego_state.time_point))
            # 必须先过 PDMSimulator（LQR + 自行车模型），再打分 ——
            # 直接把插值轨迹送进 scorer 等于跳过运动学可行性，各项都会变。
            sim_states = sim.simulate_proposals(np.stack(arr, 0), mc.ego_state)
            res = scorer.score_proposals(sim_states, mc.observation, mc.centerline,
                                         mc.route_lane_ids, mc.drivable_area_map)
            for k, m in enumerate(MODELS):
                row = res[k].iloc[0].to_dict()
                out[m].append({"token": tok, "scene": by_tok[tok][m]["scene"],
                               **{kk: (float(v) if np.size(v) == 1 else float(np.mean(v)))
                                  for kk, v in row.items()},
                               **comfort_metrics(by_tok[tok][m]["poses"])})
        except Exception as e:                                          # noqa: BLE001
            fails[f"{type(e).__name__}: {str(e)[:70]}"] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(toks)}", flush=True)

    keys = [k for k in out[MODELS[0]][0] if k not in ("token", "scene")]
    rng = np.random.default_rng(0); summ = {}
    print(f"\n联合打分  n={len(out[MODELS[0]])} token  失败 {sum(fails.values())}\n")
    print(f"{'指标':<32}" + "".join(f"{m:>21}" for m in MODELS))
    print("-" * (32 + 21 * len(MODELS)))
    for k in keys:
        line = f"{k:<32}"
        for m in MODELS:
            a = np.array([r[k] for r in out[m]], float)
            bs = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)])
            ci = np.percentile(bs, [2.5, 97.5])
            summ.setdefault(m, {})[k] = {"mean": float(a.mean()), "ci": list(ci)}
            line += f"{a.mean():>9.3f}[{ci[0]:.2f},{ci[1]:.2f}]"
        print(line)
    (RES / "deploy_rank_joint.json").write_text(json.dumps(
        {"models": MODELS, "n_tokens": len(out[MODELS[0]]),
         "protocol": {"pdm": "non-reactive (v1 cache)",
                      "progress_normalisation": "max over the four candidates (no PDM-Closed baseline)",
                      "scorer": "official PDMScorer.score_proposals, 4 proposals per call",
                      "note": "相对进展：只可在本文四家之间比较，不可与他人发布的绝对 PDM 分数比"},
         "summary": summ, "failures": dict(fails),
         "per_model": {m: out[m] for m in MODELS}}, indent=2, ensure_ascii=False))
    print(f"\n[JOINT] wrote {RES/'deploy_rank_joint.json'}")


if __name__ == "__main__":
    main()
