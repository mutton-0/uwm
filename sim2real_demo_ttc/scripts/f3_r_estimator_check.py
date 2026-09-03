"""核查|F-3 的必要性比 R 换一种估计量会不会改判（§FE/A67）。

R 自设计以来一直用**逐事件先算比值、再求 scene 级均值**估计。逐事件分母 b_ghost 可以很小
（分母守卫只是部分缓解），比值重尾，且 E[X/Y] != E[X]/E[Y]。
本脚本并列算"汇总均值之比" R = mean(b_ghost − b_occ) / mean(b_ghost)（同一套 scene 级 bootstrap），
逐格比较两者的三态判定是否一致。

**注意**：只有"门开着"（b_ghost 显著）的格子里 R 才进入判定；门关着的格子两估计量不一致无后果。
"""
import json, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")
from f3_occlusion_necessity import boot_scene
RES = "/data/ruolin/uwm/sim2real_demo_ttc/results/"

def ratio_of_means(pe, sc, min_b):
    use = [r for r in pe if abs(r["b_ghost"]) >= min_b]
    if len(use) < 20: return None, 0
    d = defaultdict(lambda: ([], []))
    for r in use:
        d[r["scene"]][0].append(r["b_ghost"] - r["b_occ"]); d[r["scene"]][1].append(r["b_ghost"])
    k = sorted(d); rng = np.random.default_rng(0); v = []
    for _ in range(5000):
        idx = rng.integers(0, len(k), len(k))
        a = [x for i in idx for x in d[k[i]][0]]; b = [x for i in idx for x in d[k[i]][1]]
        if not a or not b or abs(np.mean(b)) < 1e-12: continue
        v.append(np.mean(a) / np.mean(b))
    A = [x for kk in k for x in d[kk][0]]; B = [x for kk in k for x in d[kk][1]]
    return {"mean": float(np.mean(A)/np.mean(B)),
            "ci95": [float(np.percentile(v,2.5)), float(np.percentile(v,97.5))]}, len(use)

vd = lambda c: "PASS" if c[0] > 0.5 else ("FAIL" if c[1] < 0.5 else "INDET")
print(f"{'文件':40s} {'逐事件比值均值 R':>34s} {'判':>6s} {'汇总均值之比 R':>34s} {'判':>6s}  一致?")
for f in ("f3_occlusion_ltf.json", "f3_occlusion_simlingo.json", "f3_occlusion_navsim_ltf.json",
          "f3_occlusion_navsim_simlingo.json", "f3_occlusion_dd.json", "f3_occlusion_ddv2_lidar.json"):
    d = json.load(open(RES + f)); pe = d["per_event"]; sc = [r["scene"] for r in pe]
    r1 = d.get("necessity_ratio")
    r2, n = ratio_of_means(pe, sc, d.get("min_b_gate", 0.02))
    if not r1 or not r2:
        print(f"{f:40s} {'—':>34s}"); continue
    v1, v2 = vd(r1["ci95"]), vd(r2["ci95"])
    g = lambda r: f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"
    print(f"{f:40s} {g(r1):>34s} {v1:>6s} {g(r2):>34s} {v2:>6s}  {'是' if v1==v2 else '**否**'}")
