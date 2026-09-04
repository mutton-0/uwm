"""通用语料池落定|应用 a_req>=0.4 主集 + 低速标签 + QA 留痕（场景/语料通用）。"""
import json, sys
from pathlib import Path
import numpy as np
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
A_REQ_MIN, LOW_V0 = 0.4, 3.0


def main():
    src, out, qa_note = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
    d = json.load(open(RES / src)); cands = d["candidates"]
    keep = [c for c in cands if c["a_vru_max"] >= A_REQ_MIN]
    drop = [c for c in cands if c["a_vru_max"] < A_REQ_MIN]
    for c in cands:
        c["low_speed_v0"] = {"is_low": bool(c["ego_v0"] < LOW_V0), "ego_v0_mps": c["ego_v0"]}
    for c in drop:
        c["status"] = "deferred: a_req<0.4；不删，保留待定"
    lo_k = [c for c in keep if c["low_speed_v0"]["is_low"]]
    o = {"scenario": d.get("scenario"), "corpus": d.get("corpus"),
         "hazard_classes": d.get("hazard_classes"),
         "static_corridor_m": d.get("static_corridor_m"),
         "explained_ratio_bounds": d.get("explained_ratio_bounds"),
         "scan": {"n_scenes": d["n_scenes_scanned"], "n_brake_episodes": d["n_brake_episodes"]},
         "n_events": len(keep), "n_scenes": len({c["scene"] for c in keep}),
         "low_speed_stats": {"n_low": len(lo_k), "n_total": len(keep),
                             "frac_low": round(len(lo_k) / max(len(keep), 1), 3)},
         "qa": {"method": "分层抽样目视（按 a_req 分位均匀取）ghost 帧 + 遮挡后并排",
                "note": qa_note},
         "candidates": sorted(keep, key=lambda z: -z["a_vru_max"]),
         "deferred_low_a_req": sorted(drop, key=lambda z: -z["a_vru_max"])}
    (RES / out).write_text(json.dumps(o, indent=2, ensure_ascii=False))
    a = np.array([c["a_vru_max"] for c in keep])
    print(f"{out}: {o['n_events']} 事件 / {o['n_scenes']} scene  "
          f"a_req 中位 {np.median(a):.2f} 最大 {a.max():.2f}  "
          f"低速 {len(lo_k)}/{len(keep)}  deferred {len(drop)}")


if __name__ == "__main__":
    main()
