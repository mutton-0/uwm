"""把并行 G2 的分片报告合并成 results/g2_cache_report.json（make_report 读这一个）。

两个 worker 是并行跑的，因此 elapsed_s 取**墙钟最大值**而不是求和，
s/事件 按 (墙钟 / 总缓存数) 计——这是两卡合计的实际吞吐，报告里会如实标注。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tags", nargs="+", default=["_a", "_b"])
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    res = Path(cfg["paths"]["work_dir"]) / "results"
    parts = []
    for t in args.tags:
        p = res / f"g2_cache_report{t}.json"
        if p.exists():
            parts.append(json.loads(p.read_text()))
    assert parts, "没找到任何分片报告"

    n_events = sum(p["n_events"] for p in parts)
    cached = sum(p["cached"] for p in parts)
    skipped = sum(p["skipped_existing"] for p in parts)
    failed = [f for p in parts for f in p["failed"]]
    merged = {
        "n_events": n_events, "cached": cached, "skipped_existing": skipped,
        "failed": failed, "completeness": (cached + skipped) / max(1, n_events),
        "ckpt_sha1": parts[0]["ckpt_sha1"], "preprocess_hash": parts[0]["preprocess_hash"],
        "pool_modes": parts[0]["pool_modes"], "waypoint_dt_s": parts[0]["waypoint_dt_s"],
        "elapsed_s": max(p["elapsed_s"] for p in parts),
        "parallel_workers": len(parts),
        "note": f"{len(parts)} 个 GPU worker 并行；elapsed_s 为墙钟最大值，"
                f"s/事件 = 墙钟/总事件数（两卡合计吞吐，非单卡）",
    }
    (res / "g2_cache_report.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    print(f"[merge] {cached} cached + {skipped} skipped / {n_events}  "
          f"完整率={merged['completeness']:.3f}  墙钟={merged['elapsed_s']:.0f}s  "
          f"失败={len(failed)}")
    print(f"[merge] wrote {res/'g2_cache_report.json'}")


if __name__ == "__main__":
    main()
