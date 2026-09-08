"""汇总 steer_repe 的全部结果，按预设逻辑给判定。"""
import json, glob, os
from pathlib import Path
RES=Path(__file__).resolve().parents[1]/"results"
AX={"speed":"速度轴 −ŝ","decel":"v_decel","bright":"v_bright (I 轴)"}
rows=[]
for f in sorted(glob.glob(str(RES/"steer_repe_*.json"))):
    d=json.load(open(f))
    if "stim_side" not in d: continue
    rows.append(d)
if not rows:
    print("没有带 stim_side 的结果；请先跑 run_causal_transfer.sh"); raise SystemExit
print(f"{'候选':<10}{'轴':<16}{'算子':<11}{'刺激':<6}{'斜率':>10}{'随机中位':>10}{'p':>8}{'有效':>6}")
print("-"*80)
for r in sorted(rows,key=lambda x:(x["axis"],x["model"],x["mode"],x["stim_side"])):
    import numpy as np
    rs=np.median(np.abs(r["rand_slopes"])) if r.get("rand_slopes") else float("nan")
    ok = (r["slope"]<0) and (r["p"]<0.05)
    print(f"{r['model']:<10}{AX.get(r['axis'],r['axis']):<16}{r['mode']:<11}{r['stim_side']:<6}"
          f"{r['slope']:>+10.4f}{rs:>10.4f}{r['p']:>8.3f}{('✓' if ok else '✗'):>6}")
print()
print("判定：")
for ax in sorted({r["axis"] for r in rows}):
    for m in sorted({r["model"] for r in rows if r["axis"]==ax}):
        for md in sorted({r["mode"] for r in rows if r["axis"]==ax and r["model"]==m}):
            g={r["stim_side"]:r for r in rows if r["axis"]==ax and r["model"]==m and r["mode"]==md}
            if "LHD" not in g: continue
            L=(g["LHD"]["slope"]<0) and (g["LHD"]["p"]<0.05)
            if not L:
                v="**方法未建立**（左舵注入即无效）⇒ 右舵结果不可解读"
            elif "RHD" not in g:
                v="左舵有效，右舵未跑"
            else:
                R=(g["RHD"]["slope"]<0) and (g["RHD"]["p"]<0.05)
                v=("**因果不变**（两侧都有效）" if R else "**因果偏移**（左舵有效、右舵无效）")
            print(f"  {m:<9}{AX.get(ax,ax):<16}{md:<11}-> {v}")
