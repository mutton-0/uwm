"""汇总多帧平均降噪的结果：逐候选把"单帧 vs 多帧"的读数、CI 半宽与判定并排列出。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def hw(s):
    return None if not s else float((s["ci95"][1] - s["ci95"][0]) / 2)


def f(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["dd", "ltf", "ddv2", "simlingo"])
    ap.add_argument("--out", default=str(RES / "f3_tavg_summary.json"))
    args = ap.parse_args()
    rows = []
    for m in args.models:
        p = RES / f"f3_tavg_{m}.json"
        if not p.exists():
            print(f"[TS] 缺 {p.name}"); continue
        d = json.load(open(p))
        r = {"model": d["model"], "file": p.name, "n": d["n_events"],
             "window_mean_len": d["window"]["mean_len"],
             "clean_contam_frac": d.get("clean_arm_contamination_frac"),
             "verdict_single": d["verdict_singleframe"], "verdict_multi": d["verdict"],
             "verdict_changed": d["verdict_changed"],
             "noise_reduction_ratio": d.get("noise_reduction_ratio"),
             "sd_within_ghost_mean": float(np.mean([e["sd_within_ghost"] for e in d["per_event"]])),
             "b_ghost_absent_subset": d.get("b_ghost_entity_absent_subset")}
        for k in ("b_ghost", "b_occ", "b_ctrl", "d_occ", "d_ctrl"):
            r[k + "_multi"], r[k + "_single"] = d.get(k), d.get(k + "_singleframe")
            r[k + "_hw_multi"], r[k + "_hw_single"] = hw(d.get(k)), hw(d.get(k + "_singleframe"))
        # 同帧擦除效应的特异性检验（与 f3_within_frame_effect.py 同一判据）：
        # 要求 d_occ 显著、d_ctrl 不显著、且 |d_occ| 配对地大于 |d_ctrl|
        from f3_occlusion_necessity import boot_scene
        pe, sc = d["per_event"], [e["scene"] for e in d["per_event"]]
        for suf, ka, kb in (("_multi", "d_occ", "d_ctrl"), ("_single", "d_occ_single", "d_ctrl_single")):
            diff = boot_scene([abs(e[ka]) - abs(e[kb]) for e in pe], sc)
            do, dc = (d.get("d_occ"), d.get("d_ctrl")) if suf == "_multi" else \
                     (d.get("d_occ_singleframe"), d.get("d_ctrl_singleframe"))
            s_ = lambda x: bool(x and (x["ci95"][0] > 0 or x["ci95"][1] < 0))   # noqa: E731
            r["d_occ_specific" + suf] = bool(s_(do) and not s_(dc) and s_(diff) and diff["mean"] > 0)
            r["d_occ_sig" + suf], r["d_ctrl_sig" + suf] = s_(do), s_(dc)
            r["abs_diff" + suf] = diff
            r["d_occ_direction" + suf] = ("预期（擦掉危险→提速）" if (do and do["mean"] < 0)
                                          else "反向")
        rows.append(r)
        print(f"\n=== {d['model']}  n={d['n_events']}  窗口均长 {d['window']['mean_len']:.1f}"
              f"  窗内 sd(v_ghost) 均值 {r['sd_within_ghost_mean']:.4f}"
              f"  clean 臂污染 {d.get('clean_arm_contamination_frac'):.1%} ===")
        for k in ("b_ghost", "b_occ", "d_occ", "d_ctrl"):
            a, b = r[k + "_hw_multi"], r[k + "_hw_single"]
            print(f"  {k:8s} 单帧 {f(r[k+'_single']):34s} (半宽 {b:.4f})  ->  "
                  f"多帧 {f(r[k+'_multi']):34s} (半宽 {a:.4f})  比值 {a/b:.3f}" if a and b else "")
        print(f"  d_occ 特异性  单帧: 显著={r['d_occ_sig_single']} 特异={r['d_occ_specific_single']} "
              f"方向={r['d_occ_direction_single']}")
        print(f"  d_occ 特异性  多帧: 显著={r['d_occ_sig_multi']} 特异={r['d_occ_specific_multi']} "
              f"方向={r['d_occ_direction_multi']}")
        print(f"  判定 单帧: {r['verdict_single'][:34]}")
        print(f"  判定 多帧: {r['verdict_multi'][:34]}   变了吗: {r['verdict_changed']}")

    chg = [r["model"] for r in rows if r["verdict_changed"]]
    gained = [r["model"] for r in rows
              if r["d_occ_specific_multi"] and not r["d_occ_specific_single"]]
    nr = [r["noise_reduction_ratio"] for r in rows if r["noise_reduction_ratio"]]
    out = {"n_models": len(rows), "verdict_changed_models": chg,
           "d_occ_became_specific_under_averaging": gained,
           "noise_reduction_ratio": {"values": nr,
                                     "mean": float(np.mean(nr)) if nr else None,
                                     "min": float(np.min(nr)) if nr else None,
                                     "max": float(np.max(nr)) if nr else None},
           "primary": "b_ghost 的 scene 级 bootstrap CI 半宽（多帧 / 单帧）= 降噪比值，<1 为降噪",
           "rows": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[TS] 降噪比值 {['%.3f' % x for x in nr]}（均值 {np.mean(nr):.3f}）"
          if nr else "\n[TS] 无可用比值")
    print(f"[TS] 判定发生变化的候选：{chg if chg else '无'}")
    print(f"[TS] 多帧平均后 d_occ 由不显著/不特异转为显著且特异的候选：{gained if gained else '无'}")
    print(f"[TS] wrote {args.out}")


if __name__ == "__main__":
    main()
