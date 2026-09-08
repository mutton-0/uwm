"""GT 危险方向轴|三种读数并排：0.5s 弦长（现行）/ 全轨迹弧长 / 全轨迹终点弦长。

工单：2026-09-04（用户指出现行读数只用了规划轨迹的 2 个点）。

现行 `commanded_speed` = $|wp_0-wp_2|\\times2$，只覆盖 **0.25–0.75 s**，
而 SimLingo 的规划轨迹是 **10 点 × 0.25 s = 2.5 s** ⇒ **只用了 1/5 的时域、且是弦长不是弧长**。
代价已量化：$b^{GT}$ 在 0.25–0.75 s 上是 0.332，在 0–4 s 上是 1.232（截断约 3.7 倍）。

三种读数（模型与 GT **用同一定义、同一时域**）：

| 读数 | 模型侧 | GT 侧 |
| `cs_0.5s` | $\\lvert wp_0-wp_2\\rvert\\times2$ | $\\lvert p(0.75)-p(0.25)\\rvert\\times2$ |
| `arc_full` | (从原点起的全轨迹弧长) / T | 自车真实路径 0→T 的弧长 / T |
| `chord_full` | $\\lvert wp_{-1}\\rvert$ / T | $\\lvert p(T)-p(0)\\rvert$ / T |

**GT clean 臂 = $v_0$**（车道保持 ⇒ 危险移除后维持当前速度）。
三种读数都是速度量纲，故 clean 侧同为 $v_0$，可直接比。

**坐标系说明**：弧长与弦长都是刚体变换下的不变量，故 GT 轨迹**不需要**转到查询帧 ego 系。
（若将来做横向/纵向分解，则必须按 §FE/A69 转到查询帧朝向。）
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                            # noqa: E402
from f3_occlusion_necessity import boot_scene                          # noqa: E402

WP_DT = {"simlingo": 0.25, "dd": 0.5, "ltf": 0.5, "ddv2": 0.5,
         "alpa": 0.1, "autovla": 0.5}   # Alpamayo 64x0.1=6.4s；AutoVLA 10x0.5=5s
# 各候选**原生** commanded_speed 的定义不同，必须逐模型对齐，否则比的是两个量：
#   simlingo: |wp[0]-wp[2]|*2        窗口 0.25-0.75s（agent_simlingo.py:815）
#   DD 家族 : |wp[0]| / PRED_DT      窗口 0-0.5s（第一步位移/dt）
NATIVE = {"simlingo": ("pair02", 0.25, 0.75), "dd": ("first", 0.0, 0.5),
          "ltf": ("first", 0.0, 0.5), "ddv2": ("first", 0.0, 0.5),
          # Alpamayo 原生 v_plan = |p(0.8)-p(0.2)|/0.6（alpamayo_runner.py:55-59）
          "alpa": ("pair", 0.2, 0.8),
          # AutoVLA 原生 commanded_speed = |traj[0]|/0.5（autovla_adapter.py:227）
          "autovla": ("first", 0.0, 0.5)}


def model_readouts(wp, dt, native="pair02"):
    """wp: [N,2] 规划轨迹（不含原点，wp[k] 对应 t=(k+1)*dt）。返回三种速度读数。"""
    w = np.asarray(wp, float)[:, :2]
    T = len(w) * dt
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
    arc = float(np.linalg.norm(w[0]) + seg)          # 含原点->首点那一段
    if native == "pair02" and len(w) > 2:
        cs = float(np.linalg.norm(w[0] - w[2]) * 2.0)
    elif native == "pair":            # Alpamayo：|p(0.8)-p(0.2)| / 0.6
        i0, i1 = int(round(0.2 / dt)) - 1, int(round(0.8 / dt)) - 1
        cs = (float(np.linalg.norm(w[i1] - w[i0]) / 0.6) if len(w) > i1
              else float(np.linalg.norm(w[0]) / dt))
    else:
        cs = float(np.linalg.norm(w[0]) / dt)
    return {"cs_0.5s": cs,
            "arc_full": arc / T, "chord_full": float(np.linalg.norm(w[-1])) / T,
            "horizon_s": T}


def gt_readouts(geo, j, T, dt, ta=0.25, tb=0.75):
    """自车真实未来在同一时域 T 上的三种读数（与 model_readouts 同定义）。"""
    gt, exyz = geo["grid_t"], geo["ego_xyz"]
    t0 = gt[j]
    idx = [int(np.argmin(np.abs(gt - (t0 + k * dt)))) for k in range(int(round(T / dt)) + 1)]
    P = exyz[idx, :2] - exyz[j, :2]
    ka = int(np.argmin(np.abs(gt - (t0 + ta)))); kb = int(np.argmin(np.abs(gt - (t0 + tb))))
    dt_ab = float(gt[kb] - gt[ka])
    arc = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return {"cs_0.5s": (float(np.linalg.norm(exyz[kb, :2] - exyz[ka, :2]) / dt_ab)
                        if dt_ab > 1e-6 else np.nan),
            "arc_full": arc / T, "chord_full": float(np.linalg.norm(P[-1])) / T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simlingo",
                    choices=["simlingo","dd","ltf","ddv2","alpa","autovla"])
    ap.add_argument("--corpus", default="nuscenes", choices=["nuscenes","navsim"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--f3", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    sfx = "" if args.corpus == "nuscenes" else "_navsim"
    f3p = Path(args.f3) if args.f3 else RES / f"f3_brakefirst{sfx}_{args.model}_traj.json"
    args.out = str(Path(args.out).resolve() if args.out
                   else RES / f"f3_gt_axis_full{sfx}_{args.model}.json")

    from omegaconf import OmegaConf
    G1.set_include_animal(True)
    if args.corpus == "nuscenes":
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes("v1.0-trainval", dataroot="/data/dataset/nuscenes/v1.0-trainval",
                        verbose=False)
        scmap = {s["name"]: s for s in nusc.scene}

        def build_geo(n, split=None):      # split 仅为签名对齐，nuScenes 不分 split
            return G1.compute_scene_geometry(nusc, scmap[n], cfg)
    else:
        import pickle
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"), resolve=True)
        _fr = {}

        def build_geo(n, split=None):
            """**必须带 split**：scene_name 跨 split 不唯一（test 的 65.8% 与 trainval
            重名但指向不同 log）。裸名做键会把两段无关数据拼到一起。"""
            sp0 = split or (args.split if args.split != "auto" else "test")
            if not _fr:
                for sp in (["test", "trainval"] if args.split == "auto" else [args.split]):
                    d0 = NS.NS_ROOT / "navsim_logs" / sp
                    if not d0.exists():
                        continue
                    for lf in sorted(d0.glob("*.pkl")):
                        for f in pickle.load(open(lf, "rb")):
                            _fr.setdefault((sp, f["scene_name"]), []).append(f)
            return NS.build_geo(sorted(_fr[(sp0, n)], key=lambda z: z["timestamp"]),
                                cfg, sp0)

    f3 = json.load(open(f3p)); dt = WP_DT[args.model]
    nat, t_a, t_b = NATIVE[args.model]
    KEYS = ("cs_0.5s", "arc_full", "chord_full")

    recs = []
    for r in f3["per_event"]:
        geo = build_geo(r["scene"], r.get("split"))
        j = r["frame_idx"]; v0 = float(geo["ego_speed"][j])
        mo = model_readouts(r["traj_origin"], dt, nat)
        mc = model_readouts(r["traj_clean"], dt, nat)
        mt = model_readouts(r["traj_ctrl"], dt, nat)
        g = gt_readouts(geo, j, mo["horizon_s"], dt, t_a, t_b)
        rec = {"scene": r["scene"], "v0": v0, "horizon_s": mo["horizon_s"]}
        for k in KEYS:
            rec[f"gt_origin__{k}"] = g[k]
            rec[f"b_gt__{k}"] = v0 - g[k]                 # GT clean = v0
            rec[f"b_model__{k}"] = mc[k] - mo[k]
            rec[f"b_ctrl__{k}"] = mt[k] - mo[k]
            rec[f"v_origin__{k}"] = mo[k]; rec[f"v_clean__{k}"] = mc[k]
            rec[f"err_origin__{k}"] = mo[k] - g[k]
            rec[f"err_clean__{k}"] = mc[k] - v0
        recs.append(rec)

    sc = [r["scene"] for r in recs]
    out = {"model": args.model, "n_events": len(recs), "n_scenes": len(set(sc)),
           "horizon_s": recs[0]["horizon_s"] if recs else None, "wp_dt": dt,
           "note": "三种读数模型与 GT 用同一定义同一时域；GT clean 臂 = v0（车道保持）",
           "per_event": recs}
    out["native_readout"] = {"kind": nat, "window_s": [t_a, t_b]}
    print(f"[GTFULL/{args.model}] n={len(recs)}/{len(set(sc))}scene  时域 {out['horizon_s']}s  "
          f"原生读数 {nat} 窗口 {t_a}-{t_b}s\n")
    hdr = f"{'读数':14s}{'b_GT [scene CI]':>30s}{'b_model [scene CI]':>30s}{'方向一致':>10s}{'均值比':>9s}"
    print(hdr)
    for k in KEYS:
        bg = boot_scene([r[f"b_gt__{k}"] for r in recs], sc)
        bm = boot_scene([r[f"b_model__{k}"] for r in recs], sc)
        bc = boot_scene([r[f"b_ctrl__{k}"] for r in recs], sc)
        eo = boot_scene([r[f"err_origin__{k}"] for r in recs], sc)
        ec = boot_scene([r[f"err_clean__{k}"] for r in recs], sc)
        sm = sum(1 for r in recs if np.sign(r[f"b_model__{k}"]) == np.sign(r[f"b_gt__{k}"]))
        pos = sum(1 for r in recs if r[f"b_gt__{k}"] > 0)
        rom = bm["mean"] / bg["mean"] if abs(bg["mean"]) > 1e-9 else None
        out[k] = {"b_gt": bg, "b_model": bm, "b_ctrl": bc,
                  "err_origin": eo, "err_clean": ec,
                  "sign_agreement": {"n": len(recs), "n_match": sm,
                                     "frac": round(sm / len(recs), 3)},
                  "b_gt_all_positive": f"{pos}/{len(recs)}", "ratio_of_means": rom}
        f = lambda s: f"{s['mean']:+.4f} [{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
        print(f"{k:14s}{f(bg):>30s}{f(bm):>30s}{str(sm)+'/'+str(len(recs)):>10s}{rom:+9.3f}")
    print()
    for k in KEYS:
        o = out[k]
        print(f"  {k:12s} b_GT 全正 {o['b_gt_all_positive']:>6s}  "
              f"b_ctrl {o['b_ctrl']['mean']:+.4f}  "
              f"err_origin {o['err_origin']['mean']:+.4f}  err_clean {o['err_clean']['mean']:+.4f}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[GTFULL] wrote {args.out}")


if __name__ == "__main__":
    main()
