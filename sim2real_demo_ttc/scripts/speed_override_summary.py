"""速度覆盖 F-3 结果|原始 vs 覆盖并排对比。"""
import json
from pathlib import Path
RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")

PAIRS = [
    ("DiffusionDrive", "f3_occlusion_dd.json", "f3_occlusion_dd_spdctrl.json"),
    ("LTF", "f3_occlusion_ltf.json", "f3_occlusion_ltf_spdctrl.json"),
    ("DiffusionDriveV2", "f3_occlusion_ddv2_lidar.json",
     "f3_occlusion_ddv2_lidar_spdctrl.json"),
    ("SimLingo", "f3_occlusion_simlingo.json", "f3_occlusion_simlingo_spdctrl.json"),
]


def g(s):
    return "—" if not s else f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"


def tag(d):
    v = d.get("verdict", "")
    if v.startswith("PASS"):
        return "PASS"
    if v.startswith("FAIL"):
        return "FAIL"
    return "不可估"


rows = []
for name, fo, fn in PAIRS:
    po, pn = RES / fo, RES / fn
    if not (po.exists() and pn.exists()):
        print(f"[skip] {name}: 缺 {fo if not po.exists() else fn}")
        continue
    a, b = json.load(open(po)), json.load(open(pn))
    hw = lambda s: (s["ci95"][1] - s["ci95"][0]) / 2 if s else float("nan")
    rows.append({
        "model": name,
        "orig": {"b_ghost": a.get("b_ghost"), "b_occ": a.get("b_occ"),
                 "b_ctrl": a.get("b_ctrl"), "R": a.get("necessity_ratio"),
                 "gate": bool(a.get("baseline_response_significant")),
                 "verdict": tag(a), "n": a.get("n_events")},
        "spdctrl": {"b_ghost": b.get("b_ghost"), "b_occ": b.get("b_occ"),
                    "b_ctrl": b.get("b_ctrl"), "R": b.get("necessity_ratio"),
                    "gate": bool(b.get("baseline_response_significant")),
                    "verdict": tag(b), "n": b.get("n_events")},
        "ci_halfwidth_ratio": (hw(b.get("b_ghost")) / hw(a.get("b_ghost"))
                               if a.get("b_ghost") else None),
        "effect_ratio": (abs(b["b_ghost"]["mean"]) / abs(a["b_ghost"]["mean"])
                         if a.get("b_ghost") and abs(a["b_ghost"]["mean"]) > 1e-9 else None),
        "verdict_changed": tag(a) != tag(b),
    })

print(f"{'候选':18s}{'口径':9s}{'b_ghost [scene CI]':>30s}{'门':>5s}{'判定':>8s}")
for r in rows:
    for k, lab in (("orig", "原始"), ("spdctrl", "floor8")):
        e = r[k]
        print(f"  {r['model'] if k=='orig' else '':16s}{lab:9s}{g(e['b_ghost']):>30s}"
              f"{'开' if e['gate'] else '关':>5s}{e['verdict']:>8s}")
    print(f"  {'':16s}{'→':9s}效应量比 {r['effect_ratio']:.3f}   "
          f"CI 半宽比 {r['ci_halfwidth_ratio']:.3f}   "
          f"判定{'**变了**' if r['verdict_changed'] else '未变'}")
(RES / "speed_override_summary.json").write_text(
    json.dumps({"floor_mps": 8.0, "rows": rows}, indent=2, ensure_ascii=False))
print(f"\nwrote {RES/'speed_override_summary.json'}")
