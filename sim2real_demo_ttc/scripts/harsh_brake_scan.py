"""全量扫描：人类**实际**踩出的最大减速度。纯行为口径。

## 为什么换这个
按 TTC 筛在正常运营数据上连踩两类误检（详见 ttc_scan.py 的门注释）：
路侧家具、路口横穿车。加满门之后 nuScenes 850 场景里 TTC<1.5 s **一个都没有**。

改成直接量「人踩了多重的刹车」：不需要走廊、不需要检测框、不需要归因，
上面两类误检从构造上不可能发生。人急刹本身就是危险最直接的证据。

## 阈值（自然驾驶研究常用）
    |a| ≥ 2 m/s²   一般制动
    |a| ≥ 3 m/s²   harsh braking（自然驾驶研究里的常用事件阈）
    |a| ≥ 4 m/s²   重刹
    |a| ≥ 5 m/s²   接近紧急制动（干路面轮胎极限约 8–9）
"""
from __future__ import annotations
import argparse, json, pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
import sys                                                            # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
MAP = json.load(open(RES / "driveside_map.json"))
CUTS = [2.0, 3.0, 4.0, 5.0]
SMOOTH = 3          # 速度先做 3 帧滑动平均，避免单帧位姿抖动造成的假峰值
MIN_V0 = 2.0        # 起刹速度下限：从静止附近的抖动不算


def peak_decel(t, v):
    """返回 (峰值减速度 m/s², 发生帧, 该处起刹速度)。"""
    t = np.asarray(t, float); v = np.asarray(v, float)
    if len(v) < SMOOTH + 2:
        return 0.0, -1, 0.0
    # **两端必须复制padding**：np.convolve(mode="same") 是补零，会把首末帧的
    # 平滑速度拉向 0，制造出物理上不可能的假峰值（实测首个场景 42 m/s²，
    # 峰值恰在最后一帧）。此外峰值搜索再掐掉两端各 SMOOTH 帧。
    k = np.ones(SMOOTH) / SMOOTH
    pad = SMOOTH // 2
    vp = np.concatenate([np.repeat(v[0], pad), v, np.repeat(v[-1], pad)])
    vs = np.convolve(vp, k, mode="valid")
    dt = np.gradient(t)
    a = np.gradient(vs) / np.where(np.abs(dt) > 1e-6, dt, np.nan)
    a = np.where(np.isfinite(a), a, 0.0)
    a[vs < MIN_V0] = 0.0                       # 低速段不计
    m = max(SMOOTH, 2)
    if len(a) > 2 * m:
        a[:m] = 0.0; a[-m:] = 0.0              # 端点效应区不参与峰值搜索
    j = int(np.argmin(a))
    return float(max(-a[j], 0.0)), j, float(vs[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["nuscenes", "navsim"])
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    rows = []
    if A.corpus == "navsim":
        import ns1_navsim_geometry as NS
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"),
                                     resolve=True)
        n = 0
        for sp in ("test", "trainval"):
            d0 = NS.NS_ROOT / "navsim_logs" / sp
            if not d0.exists():
                continue
            for lf in sorted(d0.glob("*.pkl")):
                by = defaultdict(list)
                for f in pickle.load(open(lf, "rb")):
                    by[f["scene_name"]].append(f)
                for scn, fl in by.items():
                    try:
                        geo = NS.build_geo(sorted(fl, key=lambda z: z["timestamp"]), cfg, sp)
                    except Exception:                                  # noqa: BLE001
                        continue
                    rows.append(_row(geo, scn, sp, MAP["navsim_side"].get(f"{sp}|{scn}")))
                    n += 1
                    if n % 1000 == 0:
                        print(f"  navsim {n}", flush=True)
    else:
        import g1_mine_events as G1
        from nuscenes.nuscenes import NuScenes
        cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/n1_d2.yaml"), resolve=True)
        nusc = NuScenes(version="v1.0-trainval",
                        dataroot="/data/dataset/nuscenes/v1.0-trainval", verbose=False)
        G1.set_include_animal(True)
        for i, sc in enumerate(nusc.scene):
            try:
                geo = G1.compute_scene_geometry(nusc, sc, cfg)
            except Exception:                                          # noqa: BLE001
                continue
            rows.append(_row(geo, sc["name"], "trainval",
                             MAP["nuscenes_side"].get(sc["name"])))
            if (i + 1) % 200 == 0:
                print(f"  nuscenes {i+1}", flush=True)
    out = Path(A.out) if A.out else RES / f"harsh_brake_{A.corpus}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"\n===== {A.corpus}：{len(rows)} 场景。峰值减速度 ≥ 阈值的场景数 =====")
    print(f"{'舵位':<6}{'场景':>7}" + "".join(f"{'≥'+str(c):>9}" for c in CUTS))
    for side in ("LHD", "RHD"):
        g = [r for r in rows if r["side"] == side]
        if not g:
            continue
        print(f"{side:<6}{len(g):>7}"
              + "".join(f"{sum(1 for r in g if r['peak_decel'] >= c):>9}" for c in CUTS))
    a = np.array([r["peak_decel"] for r in rows])
    print(f"\n峰值减速度分位：50%={np.percentile(a,50):.2f} 90%={np.percentile(a,90):.2f} "
          f"99%={np.percentile(a,99):.2f} 最大={a.max():.2f} m/s²")
    print(f"-> {out}")


def _row(geo, scn, split, side):
    pk, j, v0 = peak_decel(geo["grid_t"], geo["ego_speed"])
    r = {"scene": scn, "split": split, "side": side,
         "peak_decel": pk, "peak_frame": j, "v_at_peak": v0}
    if pk >= 2.0 and j >= 0:
        ft = np.asarray(geo["frame_ttc"], float)
        r["ttc_at_peak"] = (float(ft[j]) if j < len(ft) and np.isfinite(ft[j]) else None)
    return r


if __name__ == "__main__":
    main()
