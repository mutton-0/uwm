"""F-3 复核|把"擦除效应"从被污染的 clean 臂里解耦出来。

**为什么需要这一步**（本轮实测发现，§FC/A61）：
F-3 把 clean 臂描述为"危险实体本来就不在场"，但 G1 A 类 288 个事件里
**219 个（76.0%）在 clean 帧上实体就已经可见**——clean 帧取自 emergence 前 1.0~1.5 s，
而 `t_emergence` 标记的是"进走廊 / TTC 越阈"，不是"变得可见"（§FM/A57 同一条教训）。
两帧都可见的 218 例里，成像面积比 ghost/clean 中位 **1.67**、纵距中位 31.3 m → 25.2 m。

⇒ $b_{ghost} = v(\\text{ghost}) - v(\\text{clean})$ 量的**不是**"危险出现引起的响应"，
而是"危险**走近了 6 m**引起的响应"——一个弱得多的刺激。
而三态判定的门恰恰是 $b_{ghost}$ 显著性，于是"无基线响应"这个读数可能**部分是弱对比的产物**。

**本模块补的读数**（不需要重跑任何前向，只重新分析已落盘的 per_event）：

    d_occ  = v_ghost − v_occ     把实体从**同一帧**里擦掉引起的动作变化
    d_ctrl = v_ghost − v_ctrl    在**同一帧**别处涂同样大小的灰斑（必需对照）

两者都在**同一帧内**比较，不经过 clean 臂，因此完全不受上述污染影响。
注意 $R = 1 - b_{occ}/b_{ghost} = d_{occ}/b_{ghost}$ ——
即原设计的**分子本来就是 $d_{occ}$**，被污染的只是分母（同时也是那道门）。

判定（预注册）：$d_{occ}$ 的 scene 级 bootstrap CI 不含 0 ⇒ 擦除确实改变了动作；
并要求 $|d_{occ}|$ 显著大于 $|d_{ctrl}|$（配对差 CI 不含 0），否则可能只是"画面多了块灰斑"。
**这不替代 F-3 的主判定**，是在主判定判"无基线响应"时补一条不依赖 clean 臂的证据。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from f3_occlusion_necessity import boot_scene                          # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def sig(s):
    return bool(s and (s["ci95"][0] > 0 or s["ci95"][1] < 0))


def fmt(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=None)
    ap.add_argument("--out", default=str(RES / "f3_within_frame_effect.json"))
    args = ap.parse_args()
    files = ([Path(f) for f in args.files] if args.files
             else sorted(RES.glob("f3_occlusion_*.json")))

    rows = []
    for p in files:
        try:
            d = json.load(open(p))
        except Exception:                                              # noqa: BLE001
            continue
        pe = d.get("per_event")
        if not pe or "v_occ" not in pe[0]:
            continue
        sc = [r["scene"] for r in pe]
        d_occ = boot_scene([r["v_ghost"] - r["v_occ"] for r in pe], sc)
        d_ctrl = boot_scene([r["v_ghost"] - r["v_ctrl"] for r in pe], sc)
        d_diff = boot_scene([abs(r["v_ghost"] - r["v_occ"]) - abs(r["v_ghost"] - r["v_ctrl"])
                             for r in pe], sc)
        b_ghost = d.get("b_ghost")
        rows.append({
            "file": p.name, "model": d.get("model"), "n": len(pe),
            "b_ghost": b_ghost, "b_ghost_significant": sig(b_ghost),
            "d_occ": d_occ, "d_occ_significant": sig(d_occ),
            "d_ctrl": d_ctrl, "d_ctrl_significant": sig(d_ctrl),
            "abs_occ_minus_abs_ctrl": d_diff, "occ_exceeds_ctrl": sig(d_diff) and d_diff["mean"] > 0,
            "verdict_original": d.get("verdict", "")[:40],
            "rescued": bool(sig(d_occ) and not sig(b_ghost))})
        # 方向与特异性：v_plan 是**指令速度**，擦掉危险应当让模型**开快**
        # ⇒ v_occ > v_ghost ⇒ d_occ = v_ghost − v_occ **为负**才是预期方向。
        r = rows[-1]
        r["direction"] = ("预期（擦掉危险→提速）" if (d_occ and d_occ["mean"] < 0)
                          else "反向（擦掉危险→反而减速）")
        r["specific"] = bool(sig(d_occ) and not sig(d_ctrl)
                             and sig(d_diff) and d_diff["mean"] > 0)
        r["wf_verdict"] = (
            "不显著：同帧擦除未改变动作" if not sig(d_occ) else
            ("显著但**不特异**：同面积灰斑（ctrl 臂）也显著 ⇒ 可能只是画面多了块灰斑"
             if sig(d_ctrl) else
             ("显著且特异，方向符合预期" if d_occ["mean"] < 0
              else "显著且特异，但**方向与预期相反**（擦掉危险反而减速）")))
        print(f"{p.name:46s} n={len(pe):4d}  b_ghost {fmt(b_ghost):34s} "
              f"{'显著' if sig(b_ghost) else '跨0 '}  |  d_occ {fmt(d_occ):34s} "
              f"{'显著' if sig(d_occ) else '跨0 '}  d_ctrl {fmt(d_ctrl):34s}"
              f"{'  ← ' + rows[-1]['wf_verdict'] if (sig(d_occ) and not sig(b_ghost)) else ''}")

    n_res = sum(r["rescued"] for r in rows)
    n_spec = sum(r["rescued"] and r["specific"] for r in rows)
    n_exp = sum(r["rescued"] and r["specific"] and r["d_occ"]["mean"] < 0 for r in rows)
    out = {"design": "同一帧内擦除效应 d_occ = v_ghost − v_occ，绕开被污染的 clean 臂",
           "context": ("G1 A 类 219/288 (76.0%) 的事件在 clean 帧上实体已可见 ⇒ "
                       "b_ghost 量的是'走近 6m'而非'出现'；d_occ 不受此影响"),
           "n_cells": len(rows), "n_rescued": n_res,
           "n_rescued_and_specific": n_spec,
           "n_rescued_specific_expected_direction": n_exp,
           "sign_convention": ("v_plan 是指令速度；擦掉危险应让模型提速 ⇒ "
                               "d_occ = v_ghost − v_occ **为负**才是预期方向"),
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[WF] {len(rows)} 格；b_ghost 判'无基线响应'但 d_occ 显著：**{n_res}** 格；"
          f"其中通过 ctrl 特异性检验的 **{n_spec}** 格；方向也符合预期的 **{n_exp}** 格")
    print(f"[WF] wrote {args.out}")


if __name__ == "__main__":
    main()
