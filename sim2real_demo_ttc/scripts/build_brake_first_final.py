"""brake-first 最终语料池|应用三条待决决定 + 分层 QA 留痕。

决定（用户 2026-09-04）：
 1. a_req < 0.4 的 5 个（慢车滑行到停）**排除** —— 该阈值本就是物理判据里偏宽松的下限。
 2. scene-1055（夜间，归因最强但遮挡对比最弱）**保留**。
    归因基于真实减速行为独立算出，与遮挡实验是两件事；
    **拿要测量的结果去筛样本是选择性偏差**，不能这么做。
    其"遮挡对比为何弱"单列为诊断问题，不影响入池。
 3. 旧池 f3_candidate_pool.json（T1 9 事件/5 scene）**不合并**，两池独立使用。

追加（用户 2026-09-04）：低速（v0 < 3 m/s）候选**不删**，打 `low_speed_v0` 标签保留并统计。
理由：另一条独立分支正在测 SimLingo 的**指令服从探针**（走训练时用过的语言指令通道下
"加速到 X"，看行人在场时会不会被拒绝）—— 那不涉及谎报观测状态，对低速事件理论上仍可能有效。

**两条决定的调和（本模块的处理，属判断call，明确写出）**：
被 a_req<0.4 排除的 5 个**恰好全部是 v0<3 的低速事件**。故：
  * `candidates`（主分析集）= a_req >= 0.4，每条仍带 `low_speed_v0` 标签；
  * `deferred_low_a_req` = 那 5 个，**保留在文件内**、打标签、标注 pending，
    等指令服从探针分支出结果再定去留 —— 不现在删死。
"""
import json
from pathlib import Path

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
A_REQ_MIN = 0.4
LOW_SPEED_V0 = 3.0      # 仅打标签与统计，**不作为排除条件**（用户 2026-09-04 追加）

# 分层目视 QA 留痕：13 个候选逐个看过 results/figures/brake_first/*.jpg
QA = {
  "scene-1055": {"verdict": "keep", "note": "夜间 34.2m。归因最强(a_req 2.36)、刹车最硬(1.63 m/s²)、"
                 "走廊内零非VRU竞争。但行人在暗处成像极弱，**遮挡前后图像差异很小** —— "
                 "记为诊断问题（灰斑取全图均值色，夜景下与背景对比度低），不作为排除理由。"},
  "scene-0003": {"verdict": "keep", "note": "三人横穿人行横道 8.8m，STOP 标志，零竞争，3/3 全遮。教科书案例。"},
  "scene-0007": {"verdict": "keep", "note": "33.9m，零竞争。"},
  "scene-0862": {"verdict": "keep", "note": "13.6m，零竞争。"},
  "scene-0151": {"verdict": "keep", "note": "自行车 37.7m，零竞争。本池唯一非行人类目标。"},
  "scene-0071": {"verdict": "keep", "note": "17.2m。非VRU 竞争 0.30 vs VRU 0.64，比值 2.1× —— "
                 "过 1.5× 余量但不宽裕，标记为边缘。"},
  "scene-1088": {"verdict": "keep", "note": "23.9m，零竞争。"},
  "scene-1016": {"verdict": "keep", "note": "37.1m，零竞争。"},
  "scene-0717": {"verdict": "keep-marginal",
                 "note": "11.9m。**边缘**：正前方有停驻 SUV、右侧 Eversource 工程车与锥桶，"
                 "非VRU 竞争 0.211 vs VRU 0.556（比值 2.63×，过线）。另：mask group 记为 1，"
                 "但该 3D 框的 2D 投影**目视覆盖了两个人** —— 遮挡效果上两人都被盖住，"
                 "账面数与实际覆盖不一致，需在分析时留意。"},
  "scene-0187": {"verdict": "keep", "note": "两人横穿 17.5m / 22.4m，2/2 全遮，零竞争。干净。"},
  "scene-0427": {"verdict": "keep", "note": "两人于人行横道标志处 38.1m，2/2 全遮，零竞争。"
                 "距离偏远、成像框小。"},
  "scene-0917": {"verdict": "keep", "note": "27.8m 横穿，零竞争。但**画面内另有 4-5 个未遮 VRU**"
                 "（走廊外，按判据不遮）—— 属已登记的残留偏差（R 会被低估），本例尤为明显。"},
  "scene-0544": {"verdict": "keep", "note": "10.7m 人行横道，绿灯，零竞争。"
                 "**遛狗场景：狗属 animal 类、不在 VRU 定义内，遮挡后仍可见** —— "
                 "'有东西在横穿'的线索未被完全移除，记为该事件的残留。"},
}


def main():
    d = json.load(open(RES / "brake_first_pool.json"))
    cands = d["candidates"]
    keep = [c for c in cands if c["a_vru_max"] >= A_REQ_MIN]
    drop = [c for c in cands if c["a_vru_max"] < A_REQ_MIN]
    for c in cands:
        c["low_speed_v0"] = {"is_low": bool(c["ego_v0"] < LOW_SPEED_V0),
                             "ego_v0_mps": c["ego_v0"], "threshold": LOW_SPEED_V0}
    for c in keep:
        c["qa"] = QA.get(c["scene"], {"verdict": "not_reviewed", "note": ""})
    for c in drop:
        c["status"] = ("deferred: a_req<0.4 且 v0 偏低；不删，等 SimLingo 指令服从探针"
                       "分支结果再定去留")
    lo_k = [c for c in keep if c["low_speed_v0"]["is_low"]]
    lo_d = [c for c in drop if c["low_speed_v0"]["is_low"]]
    out = {
      "corpus": "nuScenes v1.0-trainval (G1)",
      "design": "刹车优先挖矿：先找人类减速片段，再归因到走廊内 VRU",
      "scan": {"n_scenes": d["n_scenes_scanned"], "n_brake_episodes": d["n_brake_episodes"],
               "full_scan": True},
      "decisions_2026_09_04": {
        "a_req_min": A_REQ_MIN,
        "excluded_low_a_req": [{"scene": c["scene"], "a_req": c["a_vru_max"]} for c in drop],
        "scene_1055": "保留。不以遮挡实验结果反向筛样本（选择性偏差）；"
                      "其遮挡对比弱另记为诊断问题",
        "merge_with_old_pool": False,
      },
      "n_events": len(keep), "n_scenes": len({c["scene"] for c in keep}),
      "low_speed_stats": {
        "threshold_mps": LOW_SPEED_V0,
        "note": "仅统计与标签，**未用于排除**；去留待 SimLingo 指令服从探针分支",
        "primary_pool": {"n_low": len(lo_k), "n_total": len(keep),
                         "frac_low": round(len(lo_k) / max(len(keep), 1), 3),
                         "scenes_low": [c["scene"] for c in lo_k],
                         "v0_low": {c["scene"]: c["ego_v0"] for c in lo_k}},
        "deferred_pool": {"n_low": len(lo_d), "n_total": len(drop),
                          "frac_low": round(len(lo_d) / max(len(drop), 1), 3),
                          "v0_low": {c["scene"]: c["ego_v0"] for c in lo_d}},
        "all_candidates": {"n_low": len(lo_k) + len(lo_d), "n_total": len(cands),
                           "frac_low": round((len(lo_k) + len(lo_d)) / max(len(cands), 1), 3)},
      },
      "qa": {"method": "逐候选目视 ghost 帧 + 遮挡后图像并排（results/figures/brake_first/）",
             "n_reviewed": sum(1 for c in keep if c["qa"]["verdict"] != "not_reviewed"),
             "n_rejected_at_qa": 0,
             "marginal": [c["scene"] for c in keep if c["qa"]["verdict"] == "keep-marginal"]},
      "open_diagnostics": [
        "夜间灰斑对比度：灰斑取全图均值色，暗场景下与背景差异小（scene-1055）。"
        "是否改用局部均值/噪声填充需单独实验，**不得据此筛样本**。",
        "走廊外未遮 VRU 残留：R 方向性低估，scene-0917 最明显（见 unmasked_vru_residual.json）。",
        "非 VRU 类横穿目标（狗等 animal 类）不在遮挡组内（scene-0544）。",
      ],
      "candidates": sorted(keep, key=lambda z: -z["a_vru_max"]),
      "deferred_low_a_req": sorted(drop, key=lambda z: -z["a_vru_max"]),
    }
    (RES / "brake_first_pool_final.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"最终池: {out['n_events']} 事件 / {out['n_scenes']} scene")
    print(f"  排除(a_req<{A_REQ_MIN}): " + ", ".join(
        f"{c['scene']}({c['a_vru_max']:.2f})" for c in drop))
    print(f"  QA: 目视 {out['qa']['n_reviewed']}/{out['n_events']}，"
          f"QA 阶段拒绝 {out['qa']['n_rejected_at_qa']}，边缘 {out['qa']['marginal']}")
    ls = out["low_speed_stats"]
    print(f"\n低速标签统计（v0 < {LOW_SPEED_V0} m/s，仅统计不排除）：")
    print(f"  主分析集   低速 {ls['primary_pool']['n_low']}/{ls['primary_pool']['n_total']} "
          f"({ls['primary_pool']['frac_low']*100:.1f}%)  " + ", ".join(
              f"{k} v0={v}" for k, v in ls['primary_pool']['v0_low'].items()))
    print(f"  deferred  低速 {ls['deferred_pool']['n_low']}/{ls['deferred_pool']['n_total']} "
          f"({ls['deferred_pool']['frac_low']*100:.1f}%)")
    print(f"  全部候选   低速 {ls['all_candidates']['n_low']}/{ls['all_candidates']['n_total']} "
          f"({ls['all_candidates']['frac_low']*100:.1f}%)")
    print(f"  wrote {RES/'brake_first_pool_final.json'}")


if __name__ == "__main__":
    main()
