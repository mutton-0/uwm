"""N1 几何匹配采样：构造"标签与成像几何解耦"的评测集。

背景（q_audit §Q3/§Q6）：旧 D 类按定义在走廊外 → 成像必然小而偏心，
与正例的几何重叠只有 13%，caliper 配对 75% 失败，标签-几何相关无法事后消除。

做法：在 (log10 表观面积, 离心率) 二维上，把每个负类**下采样**到与正例同分布
（1:1 最近邻配对，带 caliper；scene 级不重复使用同一负例）。
匹配后再算读数，标签与几何的相关在**源头**被打断。

产出：
  mining/matched_{neg}.txt  —— 匹配后的事件 id 清单（正例 + 该负类）
  results/n1_matching.json  —— 匹配质量报告（SMD、caliper 内比例、丢弃量）
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

POSITIVE_TYPES = ("A", "B", "C")


def load_events(work: Path):
    evs = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    out = []
    for e in evs:
        if e.get("area_px") is None or e.get("ecc") is None:
            continue
        e["_la"] = float(np.log10(e["area_px"] + 1.0))
        e["_ecc"] = float(e["ecc"])
        out.append(e)
    return out


def smd(a, b):
    """标准化均值差（Cohen's d 式），|SMD| < 0.1 通常视为平衡良好。"""
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / s)


def match(pos, neg, cal_la, cal_ecc, seed=0):
    """1:1 最近邻 + caliper，负例无放回。返回 (配上的正例, 配上的负例)。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pos))          # 随机顺序，避免系统性优先
    la_n = np.array([e["_la"] for e in neg])
    ec_n = np.array([e["_ecc"] for e in neg])
    used = np.zeros(len(neg), dtype=bool)
    mp, mn = [], []
    for i in order:
        p = pos[i]
        d_la = np.abs(la_n - p["_la"])
        d_ec = np.abs(ec_n - p["_ecc"])
        ok = (~used) & (d_la <= cal_la) & (d_ec <= cal_ecc)
        if not ok.any():
            continue
        cost = (d_la / cal_la) ** 2 + (d_ec / cal_ecc) ** 2
        cost[~ok] = np.inf
        j = int(np.argmin(cost))
        used[j] = True
        mp.append(p)
        mn.append(neg[j])
    return mp, mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--caliper-la", type=float, default=0.15, help="log10 面积 caliper（dex）")
    ap.add_argument("--caliper-ecc", type=float, default=0.06, help="离心率 caliper")
    ap.add_argument("--pos-types", default="A", help="正例类型（默认只用 A：VRU 突现，§12.4 M2 主读数）")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evs = load_events(work)
    by_type = defaultdict(list)
    for e in evs:
        by_type[e["event_type"]].append(e)

    pos_types = list(args.pos_types)
    pos = [e for t in pos_types for e in by_type[t]]
    print(f"[N1] 正例类型={pos_types}  n={len(pos)}   caliper: Δlog面积≤{args.caliper_la} dex, Δ离心率≤{args.caliper_ecc}")

    report = {"pos_types": pos_types, "n_pos_total": len(pos),
              "caliper": {"log_area": args.caliper_la, "ecc": args.caliper_ecc}, "negatives": {}}
    keep_ids = set(e["event_id"] for e in pos)

    VRU = ("human.", "vehicle.bicycle", "vehicle.motorcycle")
    # D2cV / D2bV：把负类限制到与正例**同类别**(VRU)。
    # D2cV 是唯一纯净的证伪控制：与 A 同类别、同几何，唯一差异是相对速度（单帧不可见）=> 预测恰好 0.5。
    # D2c 混了 55% 车辆，类别差异本身单帧可见，会把地板抬高。
    for src, dst in (("D2c", "D2cV"), ("D2b", "D2bV")):
        by_type[dst] = [e for e in by_type.get(src, []) if e["object_class"].startswith(VRU)]

    for neg_type in ("D", "D2a", "D2b", "D2bV", "D2c", "D2cV"):
        neg = by_type.get(neg_type, [])
        if len(neg) < 20:
            print(f"[N1] {neg_type}: 样本不足({len(neg)})，跳过")
            continue
        la_p = np.array([e["_la"] for e in pos]); ec_p = np.array([e["_ecc"] for e in pos])
        la_n = np.array([e["_la"] for e in neg]); ec_n = np.array([e["_ecc"] for e in neg])
        pre = {"smd_log_area": smd(la_p, la_n), "smd_ecc": smd(ec_p, ec_n)}
        mp, mn = match(pos, neg, args.caliper_la, args.caliper_ecc)
        if len(mp) < 20:
            print(f"[N1] {neg_type}: 匹配后仅 {len(mp)} 对，几何重叠不足")
            report["negatives"][neg_type] = {"n_neg_pool": len(neg), "n_pairs": len(mp),
                                             "pre_match": pre, "usable": False}
            continue
        la_mp = np.array([e["_la"] for e in mp]); ec_mp = np.array([e["_ecc"] for e in mp])
        la_mn = np.array([e["_la"] for e in mn]); ec_mn = np.array([e["_ecc"] for e in mn])
        post = {"smd_log_area": smd(la_mp, la_mn), "smd_ecc": smd(ec_mp, ec_mn)}
        print(f"[N1] {neg_type:4s} 池={len(neg):5d}  配对={len(mp):4d} ({len(mp)/len(pos)*100:.0f}% 正例被配上)"
              f"  SMD 面积 {pre['smd_log_area']:+.2f}→{post['smd_log_area']:+.2f}"
              f"  离心率 {pre['smd_ecc']:+.2f}→{post['smd_ecc']:+.2f}")
        report["negatives"][neg_type] = {"n_neg_pool": len(neg), "n_pairs": len(mp),
                                         "pre_match": pre, "post_match": post, "usable": True}
        ids = [e["event_id"] for e in mp] + [e["event_id"] for e in mn]
        (work / "mining" / f"matched_{neg_type}.txt").write_text("\n".join(ids) + "\n")
        keep_ids.update(ids)

    (work / "mining" / "cache_needed.txt").write_text("\n".join(sorted(keep_ids)) + "\n")
    (work / "results").mkdir(exist_ok=True)
    (work / "results" / "n1_matching.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[N1] 需缓存事件总数 = {len(keep_ids)}（正例 {len(pos)} + 各负类匹配子集）")
    print(f"[N1] wrote mining/cache_needed.txt, results/n1_matching.json")


if __name__ == "__main__":
    main()
