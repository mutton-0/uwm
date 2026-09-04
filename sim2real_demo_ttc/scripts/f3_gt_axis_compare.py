"""四候选 arc_full 并排表|GT 危险方向轴下的响应充分度对比。

**跨模型只能比"响应比" $b_{model}/b^{GT}$**：各候选规划时域不同
（SimLingo 10×0.25=2.5s，DD 家族 8×0.5=4s）⇒ $b^{GT}$ 本身不同，
$b$ 的绝对值不可横向比。
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from f3_occlusion_necessity import boot_scene                          # noqa: E402

MODELS = [("SimLingo", "simlingo"), ("LTF", "ltf"),
          ("DiffusionDrive", "dd"), ("DiffusionDriveV2", "ddv2")]
K = "arc_full"


def ci(s):
    return f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def sig(s):
    return s["ci95"][0] * s["ci95"][1] > 0


rows = []
for name, m in MODELS:
    d = json.load(open(RES / f"f3_gt_axis_full_{m}.json"))
    o = d[K]; pe = d["per_event"]; sc = [r["scene"] for r in pe]
    spec = boot_scene([abs(r[f"b_model__{K}"]) - abs(r[f"b_ctrl__{K}"]) for r in pe], sc)
    ec = boot_scene([r[f"err_clean__{K}"] for r in pe], sc)
    rows.append({
        "model": name, "horizon_s": d["horizon_s"], "n": len(pe),
        "b_gt": o["b_gt"], "b_model": o["b_model"], "b_ctrl": o["b_ctrl"],
        "ratio_of_means": o["ratio_of_means"],
        "sign_match": o["sign_agreement"]["n_match"], "sign_n": o["sign_agreement"]["n"],
        "paired_specificity": spec, "err_origin": o["err_origin"], "err_clean": ec,
        "b_model_significant": sig(o["b_model"]),
        "b_ctrl_significant": sig(o["b_ctrl"]),
        "specificity_significant": sig(spec),
    })

W = "| {:<17} | {:>5} | {:>26} | {:>26} | {:>8} | {:>7} |"
print("\n### arc_full 主表\n")
print(W.format("候选", "时域", "b_GT [scene CI]", "b_model [scene CI]", "响应比", "方向"))
print("|" + "-"*19 + "|" + "-"*7 + "|" + "-"*28 + "|" + "-"*28 + "|" + "-"*10 + "|" + "-"*9 + "|")
for r in rows:
    print(W.format(r["model"], f"{r['horizon_s']:g}s", ci(r["b_gt"]), ci(r["b_model"]),
                   f"{r['ratio_of_means']:+.3f}", f"{r['sign_match']}/{r['sign_n']}"))
print("\n### 对照臂 / 特异性 / 逐臂误差\n")
W2 = "| {:<17} | {:>26} | {:>26} | {:>26} |"
print(W2.format("候选", "b_ctrl", "配对特异性 |b|-|b_ctrl|", "err_origin"))
print("|" + "-"*19 + "|" + "-"*28 + "|" + "-"*28 + "|" + "-"*28 + "|")
for r in rows:
    print(W2.format(r["model"], ci(r["b_ctrl"]), ci(r["paired_specificity"]),
                    ci(r["err_origin"])))
print("\n### 显著性一览（CI 是否跨 0）\n")
print(f"{'候选':<18}{'b_model':>10}{'b_ctrl':>10}{'特异性':>10}")
for r in rows:
    y = lambda b: "显著" if b else "跨0"
    print(f"{r['model']:<18}{y(r['b_model_significant']):>10}"
          f"{y(r['b_ctrl_significant']):>10}{y(r['specificity_significant']):>10}")
(RES / "f3_gt_axis_compare.json").write_text(
    json.dumps({"readout": K, "note": "跨模型只比响应比；b 绝对值因时域不同不可横向比",
                "rows": rows}, indent=2, ensure_ascii=False))
print(f"\nwrote {RES/'f3_gt_axis_compare.json'}")
