"""指令探针的合并分析：主对比 + 场景分层 + 选择性地板。

分三层读数，强度递减：
  L1 主对比  ghost − occ（同一张帧，只差目标实体在不在）—— 真正的在场/不在场操作
  L2 对照    ctrl − occ（两臂都有同面积灰斑，只差灰斑盖住的是不是那个实体）
             若 L1 与 L2 同号同量级 ⇒ 效应来自"画面被涂了一块"，与实体无关
  L3 地板    grey 臂（整幅均值灰图）—— 危险类拒绝若在这里仍高比例出现，
             说明它由 prompt 文本先验驱动，任何 L1 差值都不能读作"用到了行人"

**分层的必要性**：occ 只抹掉挖掘锁定的那一个实体。census 实测 77.4% 的 ghost 帧
里还有别的行人。在那些帧上，模型完全可以因为**别人**继续拒绝，L1 的零效应
只能读作"没用到**这一个**行人"。故 `other_peds_in_view == 0` 的子集
（唯一满足"画面里再无行人"的子集）单独报一次。
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simlingo_instruction_probe import classify, HAZARD, boot_scene

INSTR = ["acc_target_speed", "acc_faster", "dec_slower", "dec_stop"]
DIRN = {"acc_target_speed": "up", "acc_faster": "up", "dec_slower": "down", "dec_stop": "down"}


def hz(recs, arm):
    return [1.0 if classify(r["arms"][arm]["lang"]) in HAZARD else 0.0 for r in recs]


def fmt(b):
    return "—" if not b else "%+.3f [%+.3f, %+.3f]" % (b["mean"], *b["ci95"])


def contrast(recs, a, b, name):
    sc = [r["scene"] for r in recs]
    return boot_scene([x - y for x, y in zip(hz(recs, f"{a}/{name}"), hz(recs, f"{b}/{name}"))], sc)


def main():
    occ = json.load(open(RES / "simlingo_instruction_probe_occ.json"))
    recs = occ["per_event"]
    census = {r["event_id"]: r for r in
              json.load(open(RES / "simlingo_instruction_probe_scene_census.json"))["per_event"]}
    for r in recs:
        r["other_peds"] = census.get(r["eid"], {}).get("other_peds_in_view")
    solo = [r for r in recs if r["other_peds"] == 0]

    out = {"n_events": len(recs), "n_events_no_other_pedestrian": len(solo),
           "n_scenes": len({r["scene"] for r in recs}),
           "n_scenes_solo": len({r["scene"] for r in solo})}

    print(f"事件 {len(recs)}（场景 {out['n_scenes']}）；其中"
          f"「画面里再无其它行人」子集 {len(solo)}（场景 {out['n_scenes_solo']}）\n")

    for tag, sub in (("全样本", recs), ("无其它行人子集", solo)):
        print(f"===== {tag} n={len(sub)} =====")
        print("%-18s %7s %7s %7s %7s | %-24s %-24s" % (
            "指令", "clean", "ghost", "occ", "ctrl", "L1 ghost−occ(主)", "L2 ctrl−occ(对照,应≈0)"))
        blk = {}
        for name in INSTR:
            rates = {a: float(np.mean(hz(sub, f"{a}/{name}"))) for a in ("clean", "ghost", "occ", "ctrl")}
            l1, l2 = contrast(sub, "ghost", "occ", name), contrast(sub, "ctrl", "occ", name)
            blk[name] = {"direction": DIRN[name], "hazard_reject_rate": rates,
                         "L1_ghost_minus_occ": l1, "L2_ctrl_minus_occ": l2,
                         "clean_minus_ghost": contrast(sub, "clean", "ghost", name)}
            print("%-18s %6.1f%% %6.1f%% %6.1f%% %6.1f%% | %-24s %-24s" % (
                name, *[100 * rates[a] for a in ("clean", "ghost", "occ", "ctrl")], fmt(l1), fmt(l2)))
        out["strata_" + ("all" if tag == "全样本" else "no_other_pedestrian")] = blk
        print()

    # ---- 轨迹方向服从（与拒绝文本相互独立的第二读数）----
    print("===== 轨迹服从率（Δv 相对同帧无指令基线的符号是否与指令方向一致）=====")
    tj = {}
    for name in INSTR:
        row = {}
        for a in ("clean", "ghost", "occ", "ctrl"):
            row[a] = float(np.mean([1.0 if r["arms"][f"{a}/{name}"]["traj_complies"] else 0.0 for r in recs]))
        sc = [r["scene"] for r in recs]
        d = boot_scene([(1.0 if r["arms"][f"ghost/{name}"]["traj_complies"] else 0.0)
                        - (1.0 if r["arms"][f"occ/{name}"]["traj_complies"] else 0.0) for r in recs], sc)
        tj[name] = {"comply_rate": row, "ghost_minus_occ": d}
        print("%-18s clean %5.1f%%  ghost %5.1f%%  occ %5.1f%%  ctrl %5.1f%%  | ghost−occ %s" % (
            name, *[100 * row[a] for a in ("clean", "ghost", "occ", "ctrl")], fmt(d)))
    out["trajectory_compliance"] = tj

    # ---- L3 选择性地板 ----
    gp = RES / "simlingo_instruction_probe_grey.json"
    if gp.exists():
        g = json.load(open(gp))["per_event"]
        print("\n===== L3 选择性地板：整幅均值灰图（画面里什么都没有）n=%d =====" % len(g))
        fl = {}
        for name in INSTR:
            fl[name] = {a: float(np.mean(hz(g, f"{a}/{name}"))) for a in ("clean", "ghost", "grey")}
            print("%-18s clean %5.1f%%  ghost %5.1f%%  **grey %5.1f%%**" % (
                name, 100 * fl[name]["clean"], 100 * fl[name]["ghost"], 100 * fl[name]["grey"]))
        out["L3_selectivity_floor"] = fl

    p = RES / "simlingo_instruction_probe_analysis.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
