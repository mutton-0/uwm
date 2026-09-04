"""F-3 危险方向轴（GT 版）|用人类司机的真实行为定义"正确的 b"，再给模型打分。

工单：2026-09-04（用户提出）。

## 思路

原 F-3 的三臂全是**模型输出**，没有真值参照 ⇒ 只能做"显不显著"的二值判定。
本模块引入一条**与模型无关**的危险方向轴：

* $v^{GT}_{origin}$ = **人类司机的真实未来速度**（在与该模型 commanded_speed 完全相同的
  时间窗上、从自车真实位姿算出）—— 这是"危险在场时的正确动作"。
* $v^{GT}_{clean}$ = **查询帧的当前速度 $v_0$** —— prompt 是车道保持任务，
  危险移除后的正确动作就是"维持当前速度"。
* $b^{GT} = v^{GT}_{clean} - v^{GT}_{origin} = v_0 - v_{GT}$
  ⇒ **正 = 移除危险后应当提速**，其大小就是这次危险"应该"引起多大的减速。

模型得分（与 F-3 主读数同符号约定 $b = v_{clean}-v_{origin}$）：

* **方向分** $\\mathrm{sign}(b)\\cdot\\mathrm{sign}(b^{GT})$
* **比值分** $b / b^{GT}$ —— 1.0 = 幅度刚好，0 = 无响应，负 = 反向
* **逐臂误差** $v_{origin}-v^{GT}_{origin}$（危险在场时开得对不对）、
  $v_{clean}-v_0$（危险移除后是否维持速度）

## 两个必须声明的假设（不是测量，是建模）

1. **"危险移除后 = 维持当前速度"不可观测**，是对反事实的建模。
   本语料**天然支持**它：brake-first 的查询帧按定义取在**减速起点**（速度局部极大），
   即自车在该帧之前是稳态行驶 ⇒ $v_0$ 是合理的"未受扰动速度"。
   但若该帧还存在**未标注的交通管制**（信号灯/STOP，见 brake_first_report §3.2），
   人类即使没有行人也会减速 ⇒ $b^{GT}$ 会**高估**危险应引起的响应。
2. **归因不是 100%**：$b^{GT}$ 的全部减速未必都该记在危险组头上。
   故并列给出按 `vru_class_share` 缩放的 $b^{GT}_{attr}$。

## 时间窗必须逐模型对齐

`commanded_speed` 各候选定义不同（SimLingo: $|wp_0-wp_2|\\times2$，
$wp_k$ 对应 $t=(k{+}1)\\times0.25$s ⇒ 窗口 0.25–0.75 s；DD 家族: 第一步 / PRED_DT=0.5s）。
GT 必须在**同一窗口**上算，否则比的是两个不同的量。
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from f3_occlusion_necessity import boot_scene                          # noqa: E402

# 各候选 commanded_speed 的时间窗 (t_a, t_b)：速度 = |p(t_b)-p(t_a)| / (t_b-t_a)
READOUT_WINDOW = {
    "simlingo": (0.25, 0.75),      # |wp[0]-wp[2]|*2，wp_k -> t=(k+1)*0.25s
    "dd": (0.0, 0.5), "ltf": (0.0, 0.5), "ddv2": (0.0, 0.5),   # 第一步 / PRED_DT
}


def gt_speed(geo, j, t_a, t_b):
    """人类司机在 [t_a, t_b] 窗口上的真实速度（与 commanded_speed 同一算法）。"""
    gt, exyz = geo["grid_t"], geo["ego_xyz"]
    t0 = gt[j]
    ka = int(np.argmin(np.abs(gt - (t0 + t_a))))
    kb = int(np.argmin(np.abs(gt - (t0 + t_b))))
    dt = float(gt[kb] - gt[ka])
    if dt <= 1e-6:
        return None, None
    return float(np.linalg.norm(exyz[kb, :2] - exyz[ka, :2]) / dt), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simlingo")
    ap.add_argument("--f3", default="")
    ap.add_argument("--pool", default=str(RES / "brake_first_pool_final.json"))
    ap.add_argument("--min-bgt", type=float, default=0.20,
                    help="b_GT 分母守卫：|b_GT| 低于此值的事件不进比值分（仍进方向分）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    f3p = Path(args.f3) if args.f3 else RES / f"f3_brakefirst_{args.model}.json"
    args.out = str(Path(args.out).resolve() if args.out
                   else RES / f"f3_gt_axis_{args.model}.json")

    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    G1.set_include_animal(True)
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval",
                    verbose=False)
    scmap = {s["name"]: s for s in nusc.scene}
    f3 = json.load(open(f3p))
    pool = {c["scene"]: c for c in json.load(open(args.pool))["candidates"]}
    t_a, t_b = READOUT_WINDOW[args.model]
    print(f"[GTAXIS/{args.model}] {len(f3['per_event'])} 事件  读数窗口 {t_a}-{t_b}s")

    recs = []
    for r in f3["per_event"]:
        geo = G1.compute_scene_geometry(nusc, scmap[r["scene"]], cfg)
        j = r["frame_idx"]
        v_gt, dt = gt_speed(geo, j, t_a, t_b)
        if v_gt is None:
            continue
        v0 = float(geo["ego_speed"][j])
        share = pool[r["scene"]]["vru_class_share"]
        b_gt = v0 - v_gt                       # 正 = 移除危险后应提速
        b_gt_attr = b_gt * share
        b = r["b"]
        recs.append({
            "scene": r["scene"], "v0_mps": v0, "v_gt_origin": v_gt,
            "v_gt_clean": v0, "b_gt": b_gt, "vru_class_share": share,
            "b_gt_attributed": b_gt_attr,
            "v_origin": r["v_origin"], "v_clean": r["v_clean"], "v_ctrl": r["v_ctrl"],
            "b_model": b, "b_ctrl": r["b_ctrl"],
            "err_origin": r["v_origin"] - v_gt,        # 危险在场时开得对不对
            "err_clean": r["v_clean"] - v0,            # 移除危险后是否维持速度
            "sign_match": int(np.sign(b) == np.sign(b_gt)) if abs(b_gt) > 1e-9 else None,
            "ratio": (b / b_gt) if abs(b_gt) >= args.min_bgt else None,
        })
        print(f"  {r['scene']:14s} v0 {v0:5.2f} GT {v_gt:5.2f} b_GT {b_gt:+6.3f} | "
              f"模型 b {b:+6.3f}  方向{'对' if recs[-1]['sign_match'] else '错'}  "
              f"err_origin {recs[-1]['err_origin']:+6.3f} err_clean {recs[-1]['err_clean']:+6.3f}")

    sc = [r["scene"] for r in recs]
    sm = [r["sign_match"] for r in recs if r["sign_match"] is not None]
    rt = [r["ratio"] for r in recs if r["ratio"] is not None]
    out = {"model": args.model, "readout_window_s": [t_a, t_b],
           "gt_axis_def": "b_GT = v0 - v_GT(人类真实未来速度，同窗口)；"
                          "v_GT_clean = v0（车道保持 ⇒ 危险移除后维持当前速度）",
           "assumptions": [
             "'危险移除后维持当前速度'是对反事实的建模，不可观测。"
             "brake-first 查询帧按定义取在减速起点（速度局部极大）⇒ v0 是合理的未受扰动速度；"
             "但若该帧存在未标注交通管制，b_GT 会高估危险应引起的响应。",
             "b_GT 的全部减速未必都该记在危险组头上，故并列给出按 vru_class_share 缩放的版本。"],
           "n_events": len(recs), "n_scenes": len(set(sc)),
           "min_bgt_guard": args.min_bgt, "n_used_for_ratio": len(rt),
           "sign_agreement": {"n_match": int(sum(sm)), "n": len(sm),
                              "frac": round(sum(sm) / max(len(sm), 1), 3)},
           "per_event": recs}
    for k in ("b_gt", "b_gt_attributed", "b_model", "err_origin", "err_clean"):
        out[k] = boot_scene([r[k] for r in recs], sc)
    if rt:
        out["ratio"] = boot_scene(rt, [r["scene"] for r in recs if r["ratio"] is not None])
    # 汇总均值之比（对小分母稳健）
    mb = np.mean([r["b_model"] for r in recs]); mg = np.mean([r["b_gt"] for r in recs])
    out["ratio_of_means"] = float(mb / mg) if abs(mg) > 1e-9 else None
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    g = lambda s: "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
    print(f"\n[GTAXIS/{args.model}] n={len(recs)}/{len(set(sc))}scene")
    for k in ("b_gt", "b_gt_attributed", "b_model", "err_origin", "err_clean"):
        print(f"  {k:18s} {g(out[k])}")
    print(f"  方向一致 {out['sign_agreement']['n_match']}/{out['sign_agreement']['n']} "
          f"= {out['sign_agreement']['frac']*100:.1f}%")
    if rt:
        print(f"  比值分(逐事件, |b_GT|>={args.min_bgt}, n={len(rt)}) {g(out['ratio'])}")
    print(f"  比值分(汇总均值之比) {out['ratio_of_means']:+.4f}")
    print(f"[GTAXIS] wrote {args.out}")


if __name__ == "__main__":
    main()
