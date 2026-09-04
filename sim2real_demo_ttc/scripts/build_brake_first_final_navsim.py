"""NAVSIM brake-first 最终池|与 nuScenes 同一套决定与 QA 机制，但**两语料独立不合并**。"""
import json
from pathlib import Path

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
A_REQ_MIN = 0.4
LOW_SPEED_V0 = 3.0

QA = {
 "log-0020-scene-0005": {"verdict": "keep", "note":
   "拉斯维加斯大道。3 人横穿 37.2m，自车 8.09->0.08（完全停住），零非VRU竞争，3/3 全遮。"
   "**但路口有交通信号灯**（geo 不标注信号灯）—— 这次停车可能部分是为红灯，属已登记的"
   "未标注交通管制混淆。另有若干走廊外未遮 VRU。"},
 "log-0089-scene-0006": {"verdict": "keep", "note":
   "骑行者横穿 19.8m，自车 2.33->0.0。**旁边紧挨着第二个骑行者未被遮**（在 2m 走廊外）—— "
   "遮挡组记 1，但视觉上'有人在横穿'的线索未被完全移除。属已登记的残留偏差。"},
 "log-0126-scene-0010": {"verdict": "keep-marginal", "note":
   "**边缘**：正前方本车道内有一辆黑色 SUV，行人在 29.4m（部分被 SUV 遮挡）。"
   "a_non_vru 仅 0.07 —— 物理上正确（该车与自车同速前行，v_close≈0 ⇒ 无需刹车），"
   "但目视这是典型的跟车场景。归因公式与人的直觉在此分歧，标记为边缘供后续留意。"
   "另 explains 1.649（超解释）。"},
 "log-0111-scene-0024": {"verdict": "keep", "note":
   "两人横穿 33.3m、lat 0.18m（几乎正对车道中线），自车 7.32->2.44，2/2 全遮。"
   "前方有车，非VRU 0.427 vs VRU 1.013（比值 2.37×，过 1.5× 余量）。"},
 "log-0061-scene-0003": {"verdict": "keep", "note":
   "干净。行人横穿 28.6m，自车 6.01->2.96，零非VRU竞争。"
   "**a_obs 1.014 为本语料最高**，是唯一超过 1.0 m/s² 的真刹车。"},
 "log-0110-scene-0030": {"verdict": "keep", "note":
   "住宅街，行人横穿 32.1m，自车 5.42->0.01（完全停住），2/2 全遮，零非VRU竞争。"
   "**但路口有 STOP 标志**（geo 不标注）—— 停车可能是为该标志，属未标注交通管制混淆。"},
 "log-0062-scene-0003": {"verdict": "keep", "note":
   "干净。行人横穿 27.5m，自车 4.81->1.09，零非VRU竞争，explains 0.956。"},
 "log-0061-scene-0004": {"verdict": "keep", "note":
   "有人行横道标志的标线路口，两人横穿 13.9m、lat 0.5m，自车 2.52->0.02，2/2 全遮。"
   "**画面内另有 2 人正在横穿但在走廊外、未被遮** —— 残留偏差在本例明显。"},
}


def main():
    d = json.load(open(RES / "brake_first_pool_navsim.json"))
    cands = d["candidates"]
    keep = [c for c in cands if c["a_vru_max"] >= A_REQ_MIN]
    drop = [c for c in cands if c["a_vru_max"] < A_REQ_MIN]
    for c in cands:
        c["low_speed_v0"] = {"is_low": bool(c["ego_v0"] < LOW_SPEED_V0),
                             "ego_v0_mps": c["ego_v0"], "threshold": LOW_SPEED_V0}
    for c in keep:
        c["qa"] = QA.get(c["scene"], {"verdict": "not_reviewed", "note": ""})
    for c in drop:
        c["status"] = ("deferred: a_req<0.4；不删，等 SimLingo 指令服从探针分支结果再定去留")
    lo_k = [c for c in keep if c["low_speed_v0"]["is_low"]]
    lo_d = [c for c in drop if c["low_speed_v0"]["is_low"]]
    out = {
      "corpus": "NAVSIM / OpenScene (nuPlan-derived), split=test",
      "independent_of_nuscenes_pool": True,
      "note": "与 nuScenes 池**不合并**，两语料各自独立（用户 2026-09-04 决定）",
      "design": "刹车优先挖矿：先找人类减速片段，再归因到走廊内 VRU（判据与 nuScenes 逐条相同）",
      "scan": {"n_scenes": d["n_scenes_scanned"], "n_brake_episodes": d["n_brake_episodes"],
               "full_scan": True, "n_logs": 147},
      "n_events": len(keep), "n_scenes": len({c["scene"] for c in keep}),
      "low_speed_stats": {
        "threshold_mps": LOW_SPEED_V0, "note": "仅统计与标签，未用于排除",
        "primary_pool": {"n_low": len(lo_k), "n_total": len(keep),
                         "frac_low": round(len(lo_k) / max(len(keep), 1), 3),
                         "v0_low": {c["scene"]: c["ego_v0"] for c in lo_k}},
        "deferred_pool": {"n_low": len(lo_d), "n_total": len(drop),
                          "frac_low": round(len(lo_d) / max(len(drop), 1), 3)},
        "all_candidates": {"n_low": len(lo_k) + len(lo_d), "n_total": len(cands),
                           "frac_low": round((len(lo_k) + len(lo_d)) / max(len(cands), 1), 3)},
      },
      "qa": {"method": "逐候选目视 ghost 帧 + 遮挡后图像并排（results/figures/brake_first_navsim/）",
             "n_reviewed": sum(1 for c in keep if c["qa"]["verdict"] != "not_reviewed"),
             "n_rejected_at_qa": 0,
             "marginal": [c["scene"] for c in keep if c["qa"]["verdict"] == "keep-marginal"]},
      "open_diagnostics": [
        "**未标注交通管制**：NAVSIM 的 geo 同样不标注信号灯 / STOP 标志。"
        "log-0020（信号灯）与 log-0110（STOP 标志）的停车可能部分归因于此，无法用几何量剥离。",
        "走廊外未遮 VRU 残留：log-0089（并排的第二个骑行者）、log-0061-scene-0004（另 2 人横穿）最明显。",
        "跟车场景与归因公式的分歧：log-0126-scene-0010 的前车因与自车同速而 a_req≈0，"
        "物理正确但与人的直觉不符。",
      ],
      "candidates": sorted(keep, key=lambda z: -z["a_vru_max"]),
      "deferred_low_a_req": sorted(drop, key=lambda z: -z["a_vru_max"]),
    }
    (RES / "brake_first_pool_navsim_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"NAVSIM 最终池: {out['n_events']} 事件 / {out['n_scenes']} scene")
    print(f"  扫描 {d['n_scenes_scanned']} scene / {d['n_brake_episodes']} 减速片段（147 log 全量）")
    print(f"  排除(a_req<{A_REQ_MIN}): {len(drop)}")
    print(f"  QA: 目视 {out['qa']['n_reviewed']}/{out['n_events']}，拒绝 0，边缘 {out['qa']['marginal']}")
    ls = out["low_speed_stats"]
    print(f"  低速 v0<3: 主集 {ls['primary_pool']['n_low']}/{ls['primary_pool']['n_total']}，"
          f"全部候选 {ls['all_candidates']['n_low']}/{ls['all_candidates']['n_total']}")


if __name__ == "__main__":
    main()
