"""brake-first 新语料（13 事件）指令探针分析。只读 newpool 产物，不碰旧语料任何文件。

读数层级与旧版方法一致（方法不变，只换数据源）：
  L1 主对比  ghost − occ   同一帧，只差走廊内 VRU 整组在不在
  L2 对照    ctrl − occ    两臂同面积灰斑，只差涂的是不是那些 VRU
  L3 地板    grey          整幅均值灰图；危险类拒绝若仍出现 ⇒ 文本先验驱动
  clean      参考臂（本脚本构造的 t-1.5s 帧，非新语料设计），不作主结论

n=13 时 bootstrap CI 极宽，故每格**并列给出配对不一致计数 n10/n01**。
三态判定纪律：CI 跨 0 一律记"不可估/无效应"，不因样本小而放宽。
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from simlingo_instruction_probe_newpool import boot_scene, paired_counts  # noqa: E402

INSTR = [("acc_target_speed", "up"), ("acc_faster", "up"),
         ("dec_slower", "down"), ("dec_stop", "down")]


def rate(recs, arm, key="hazard_reject"):
    return [1.0 if recs_i["arms"][arm][key] else 0.0 for recs_i in recs]


def fmt(b):
    return "—" if not b else "%+.3f [%+.3f, %+.3f]" % (b["mean"], *b["ci95"])


def block(recs, key, title):
    print(f"\n===== {title}（读数 = {key}）n={len(recs)} =====")
    print("%-18s %7s %7s %7s %7s | %-24s %-9s | %-24s" % (
        "指令", "clean", "ghost", "occ", "ctrl", "L1 ghost−occ", "n10/n01", "L2 ctrl−occ"))
    res = {}
    sc = [r["scene"] for r in recs]
    cl = [r for r in recs if r["clean_available"]]
    for name, direction in INSTR:
        g, o, c = (rate(recs, f"{a}/{name}", key) for a in ("ghost", "occ", "ctrl"))
        cln = rate(cl, f"clean/{name}", key) if cl else []
        l1 = boot_scene([x - y for x, y in zip(g, o)], sc)
        l2 = boot_scene([x - y for x, y in zip(c, o)], sc)
        pc = paired_counts(g, o)
        res[name] = {"direction": direction,
                     "rate": {"clean": float(np.mean(cln)) if cln else None,
                              "ghost": float(np.mean(g)), "occ": float(np.mean(o)),
                              "ctrl": float(np.mean(c))},
                     "n_clean_events": len(cl),
                     "L1_ghost_minus_occ": l1, "L1_paired_counts": pc,
                     "L2_ctrl_minus_occ": l2,
                     "L2_paired_counts": paired_counts(c, o)}
        print("%-18s %7s %6.1f%% %6.1f%% %6.1f%% | %-24s %-9s | %-24s" % (
            name, ("%.1f%%" % (100 * np.mean(cln))) if cln else "n/a",
            100 * np.mean(g), 100 * np.mean(o), 100 * np.mean(c),
            fmt(l1), f"{pc['n10_only_first']}/{pc['n01_only_second']}", fmt(l2)))
    return res


def main():
    d = json.load(open(RES / "simlingo_instruction_probe_newpool.json"))
    recs = d["per_event"]
    out = {"corpus": d["corpus"], "source_pool": d["source_pool"],
           "occlusion_logic": d["occlusion_logic"], "clean_arm": d["clean_arm"],
           "merge_with_old_pool": False, "n_events": len(recs),
           "n_scenes": len({r["scene"] for r in recs}),
           "n_clean_available": sum(1 for r in recs if r["clean_available"])}

    print(f"语料：{d['corpus']}  来源：{d['source_pool']}")
    print(f"事件 {out['n_events']} / scene {out['n_scenes']}"
          f"（每 scene 恰好 1 事件 ⇒ scene 级 bootstrap = 事件级）")
    print(f"clean 臂可用 {out['n_clean_available']}/{out['n_events']}；遮挡逻辑：{d['occlusion_logic']}")

    # 场景清点
    op = np.array([r["other_peds_in_view"] for r in recs])
    nv = np.array([r["vehicles_in_view"] for r in recs])
    out["scene_census"] = {
        "other_pedestrians_in_view": {"mean": float(op.mean()), "median": float(np.median(op)),
                                      "frac_zero": float((op == 0).mean()),
                                      "per_event": {r["eid"]: r["other_peds_in_view"] for r in recs}},
        "vehicles_in_view": {"mean": float(nv.mean()), "median": float(np.median(nv)),
                             "frac_zero": float((nv == 0).mean())}}
    print(f"\n遮挡组之外仍可见：行人 均值 {op.mean():.2f} 中位 {np.median(op):.0f}，"
          f"为 0 的事件 {(op==0).sum()}/{len(op)}；车辆 均值 {nv.mean():.2f} 中位 {np.median(nv):.0f}")

    out["hazard_reject_all"] = block(recs, "hazard_reject", "全样本 · 危险类拒绝")
    out["pedestrian_reject_all"] = block(recs, "pedestrian_reject",
                                         "全样本 · 只数 'because of the pedestrian'")
    solo = [r for r in recs if r["other_peds_in_view"] == 0]
    if len(solo) >= 5:
        out["hazard_reject_no_other_ped"] = block(
            solo, "hazard_reject", f"子集：遮挡组外再无行人 · 危险类拒绝")
    else:
        out["hazard_reject_no_other_ped"] = {
            "skipped": f"该子集仅 {len(solo)} 个事件，低于 bootstrap 最低 5 个 scene 的门槛，不出读数"}
        print(f"\n[子集] 遮挡组外再无行人的事件仅 {len(solo)} 个 < 5，不做 bootstrap（如实记为功效不足）")

    # 轨迹方向服从
    print("\n===== 轨迹方向服从率 =====")
    tj = {}
    sc = [r["scene"] for r in recs]
    for name, direction in INSTR:
        row = {a: float(np.mean(rate(recs, f"{a}/{name}", "traj_complies")))
               for a in ("ghost", "occ", "ctrl", "grey")}
        g, o = rate(recs, f"ghost/{name}", "traj_complies"), rate(recs, f"occ/{name}", "traj_complies")
        b = boot_scene([x - y for x, y in zip(g, o)], sc)
        tj[name] = {"comply_rate": row, "ghost_minus_occ": b,
                    "paired_counts": paired_counts(g, o)}
        print("%-18s ghost %5.1f%% occ %5.1f%% ctrl %5.1f%% grey %5.1f%% | ghost−occ %s" % (
            name, *[100 * row[a] for a in ("ghost", "occ", "ctrl", "grey")], fmt(b)))
    out["trajectory_compliance"] = tj

    # L3 地板 + 基线活性
    print("\n===== L3 选择性地板（整幅均值灰图）与探针活性 =====")
    fl = {}
    for name, _ in INSTR:
        fl[name] = {a: float(np.mean(rate(recs, f"{a}/{name}"))) for a in ("ghost", "occ", "grey")}
        print("%-18s ghost %5.1f%%  occ %5.1f%%  **grey %5.1f%%**" % (
            name, 100 * fl[name]["ghost"], 100 * fl[name]["occ"], 100 * fl[name]["grey"]))
    out["L3_selectivity_floor"] = fl

    nb = sum(1 for r in recs for a in ("clean", "ghost", "occ", "ctrl", "grey")
             if f"{a}/baseline" in r["arms"] and "ignore instruction" in
             (r["arms"][f"{a}/baseline"]["lang"] or "").lower())
    nr = sum(1 for r in recs for k, v in r["arms"].items()
             if not k.endswith("baseline") and v.get("hazard_reject"))
    out["probe_liveness"] = {"baseline_arms_with_refusal": nb, "hazard_rejects_total": nr}
    print(f"\n无指令基线臂出现拒绝的次数：{nb}（应为 0）")
    print(f"全部指令臂的危险类拒绝总次数：{nr}"
          f" ⇒ {'拒绝逻辑可被触发' if nr else '**从未触发**（本语料上探针无分辨力）'}")

    p = RES / "simlingo_instruction_probe_newpool_analysis.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
