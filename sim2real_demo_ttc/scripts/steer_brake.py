"""因果干预：往**本来空无一物**的巡航场景注入方向，看会不会长出刹车。

## 为什么换成注入
之前都是相关性读数（方向亮不亮）。急刹验证显示 v_faith 是混合方向
（既含"有危险"也含"该刹车"，DDv2/SimLingo 上前者根本检不出）。
但**混了什么其实不重要** —— 真正要问的是：这根轴能不能**驱动**刹车。
注入实验直接回答，且不依赖任何语义解释。

  Z' = Z + α·σ_L·v̂        于第 L* 层图像 token 段
  读数 y(α) = arc_full(轨迹)  即规划速度
  主量 = 整条 α 阶梯的最小二乘斜率 dy/dα（负 = 注入使其减速）

刺激集 = `cruise_pool_navsim_*.json` 的 cruise 组：挖矿时已要求
**前方 15 m 无物、视野开阔、匀速**。所以任何减速都不可能来自场景里的东西。

必需的对照（缺一不可）：
  ① 随机方向：同层 N 个随机单位向量走同一阶梯，给出斜率的零分布
  ② 反向注入：−α 应给出相反符号（否则是幅度效应不是方向效应）
  ③ 特异性：横向偏移不应随 α 同步劣化
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from c_axis_hazard_patch import arc_full, WP_DT                       # noqa: E402
from f_vfaith_direction import load_by_side, pick_layers              # noqa: E402

LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2"}
NSB = "/data/dataset/navsim/dataset/sensor_blobs"
ALPHAS = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
BANDS = ["0.5-4", "4-8", "8-12", "12-30"]


def band_of(v0):
    for i, (lo, hi) in enumerate([(0.5, 4), (4, 8), (8, 12), (12, 30)]):
        if lo <= v0 < hi:
            return i
    return None


def vspeed_weighted(model, nL, side="LHD"):
    """**多层加权速度轴**：逐层回归方向，层权重 = 该层投影与 v0 的 Spearman。

    这是本项目唯一验证过跨舵位迁移的轴：左舵 5 折 CV +0.58~+0.72，
    右舵留出 +0.52~+0.66（且稳定优于单层）。见 results/speed_axis_weighted.json。
    返回 {层: (单位方向, 权重)}，注入时逐层按权重同时施加。
    """
    f = RES / f"vdecel_acts_navsim_test_{model}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    sel = (z["side"] == side) & (z["kind"] == "cruise")
    v0 = z["v0"][sel].astype(float); vc = v0 - v0.mean()

    def sp(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean(); d = np.sqrt((rx @ rx) * (ry @ ry))
        return float(rx @ ry / d) if d > 0 else 0.0

    out = {}
    for l in range(nL):
        X = z[f"h__L{l}"][sel].astype(float)
        d = (X - X.mean(0)).T @ vc / max(vc @ vc, 1e-12)
        n = np.linalg.norm(d)
        if n < 1e-12:
            continue
        d = d / n
        out[l] = (d, sp(X @ d, v0))
    return out


def vspeed_dir(model, L, side="LHD"):
    """速度方向 ŝ：巡航样本的激活对 v0 做线性回归的斜率向量。

    **不能用 band 均值差分**：那是两个带噪均值相减，实测 split-half 地板 81–86°
    （= 随机方向），估不出来。回归用全部样本，地板降到 19.5–25.4°。
    """
    f = RES / f"vdecel_acts_navsim_test_{model}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    sel = (z["side"] == side) & (z["kind"] == "cruise")
    X = z[f"h__L{L}"][sel].astype(float); v0 = z["v0"][sel].astype(float)
    vc = v0 - v0.mean()
    return (X - X.mean(0)).T @ vc / max(vc @ vc, 1e-12)


def vdecel_dir(model, L, band, side="LHD"):
    """v_decel(band, L) = mean(h|brake) − mean(h|cruise)，与刺激场景**同速度段**。"""
    f = RES / f"vdecel_acts_navsim_test_{model}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    sel = (z["side"] == side) & (z["band"] == band)
    br = sel & (z["kind"] == "brake"); cr = sel & (z["kind"] == "cruise")
    if br.sum() < 5 or cr.sum() < 5:
        return None
    H = z[f"h__L{L}"]
    return H[br].mean(0) - H[cr].mean(0)


def slope(al, y):
    al = np.asarray(al, float); y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    A = np.stack([al[m], np.ones(m.sum())], 1)
    return float(np.linalg.lstsq(A, y[m], rcond=None)[0][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LABEL))
    ap.add_argument("--n", type=int, default=60, help="巡航场景数")
    ap.add_argument("--n-rand", type=int, default=50, help="随机方向对照数")
    ap.add_argument("--axis", default="decel", choices=["decel", "faith", "speed", "speed_w"],
                    help="decel = v_decel（刹车组 − 巡航组，**按速度段与场景匹配**，"
                         "与刺激集同源）；faith = v_faith（提自 brake-first 池，非同源，作对照）；"
                         "speed = **−ŝ**，ŝ 由巡航样本对 v0 做回归得到（用全部 390 个样本，"
                         "地板 19.5–25.4°；用差分估计地板是 81–86°=随机，不可用）。"
                         "**这是本套注入方法的阳性对照**：速度是模型的显式输入，"
                         "若连它的表征方向都推不动，则注入路径整体不通，"
                         "v_faith 的阴性结果无法区分'轴没用'与'方法没用'。")
    ap.add_argument("--auto-layer", action="store_true",
                    help="**因果可达性选层**：先用随机方向在每层做小剂量注入，量 |Δy|/α，"
                         "取可达性最高且过 PR/地板门的层。表征侧判据（SNR/PR/地板）只保证"
                         "「方向估得准」，完全不保证「这一层能影响输出」—— 实测 DD 的 L7 "
                         "可达性精确为 0（轨迹头不消费它），而三重门选出的 L5 只有 L4 的 1/100。")
    ap.add_argument("--layer", type=int, default=-1,
                    help="注入层。**默认 -1 = 最后一层**：在中间层注入会被下游归一化吃掉"
                         "（实测 DD 在 L5 注入 64%%，传到 L6 只剩 2.3%%，衰减 27 倍），"
                         "那样的实验根本没做出干预。最后一层往下就是动作头，注入必然被传递。")
    ap.add_argument("--device", default="cuda:1")
    A = ap.parse_args()

    # ---- 轴与层：左舵 brake-first 池，三重门。不看本刺激集 ----
    bs = load_by_side("lead", A.model); b = bs["LHD"]
    idx, snr, pr, fl = pick_layers(b["d"], b["scene"], np.random.default_rng(7), 1)
    L = (len(b["d"]) - 1) if A.layer < 0 else A.layer
    AUTO = A.auto_layer
    print(f"[{LABEL[A.model]}] 表征侧三重门选出 L={idx[0]}"
          f"（SNR={snr[idx[0]]:.1f} PR={pr[idx[0]]:.1f} 地板={fl[idx[0]]:.1f}°）", flush=True)

    pool = []
    for sp in ("test", "trainval"):
        p = RES / f"cruise_pool_navsim_{sp}.json"
        if p.exists():
            pool += [c for c in json.load(open(p))["cruise"] if c.get("side") == "LHD"]
    # **本地只下了 test split 的传感器包**，trainval 的图不在盘上 —— 先按文件存在性过滤，
    # 否则探测循环会全部抛 FileNotFoundError 并被 except 吞掉，reach 全零看着像"无因果"。
    n0 = len(pool)
    pool = [c for c in pool if (Path(NSB) / c["filename"]).exists()]
    print(f"  巡航池 {n0} -> 图在盘上的 {len(pool)}", flush=True)
    rng = np.random.default_rng(0)
    pool = [pool[i] for i in rng.choice(len(pool), min(A.n, len(pool)), replace=False)]
    cl = [c["clear_m"] for c in pool if c.get("clear_m") is not None]
    nf = [c["n_front"] for c in pool if c.get("n_front") is not None]
    print(f"  刺激集：{len(pool)} 个巡航场景"
          + (f"（前方净空中位 {np.median(cl):.0f} m" if cl else "（净空未记")
          + (f"，前方物体数中位 {np.median(nf):.0f}）" if nf else "）"), flush=True)

    from PIL import Image
    for d_ in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
        sys.path.insert(0, str(RES / d_))
    if A.model == "dd":
        from dd_adapter import DDRunner as R_
    elif A.model == "ltf":
        from ltf_adapter import LTFRunner as R_
    else:
        from ddv2_adapter import DDV2Runner as R_
    import dd_adapter as _DD
    _DD.set_crop_center_row(560)
    runner = R_(device=A.device)
    lidar = None
    if A.model == "ddv2":
        from ddv2_adapter import NavsimLidar
        lidar = NavsimLidar()

    def run(img, spd, key, vec, alpha):
        runner.set_steering(None)
        if vec is not None and alpha != 0:
            runner.set_steering(layer=L, vec=vec, alpha=float(alpha) * sig, mode="add")
        o = (runner.run(img, spd) if lidar is None
             else runner.run(img, spd, lidar_xyz=lidar.ego_points(key)))
        runner.set_steering(None)
        t = np.asarray(o["trajectory"], float)
        return arc_full(t, WP_DT[A.model]), float(np.abs(t[:, 1]).max())

    # ---- 因果可达性探测：每层用随机方向小剂量注入，量 |Δy|/α ----
    if AUTO:
        probe = pool[:min(6, len(pool))]
        reach = []
        for l in range(len(b["d"])):
            dd = len(b["d"][l][0]); acc = []
            for k in range(3):
                rv = rng.normal(size=dd); rv /= np.linalg.norm(rv)
                for c in probe:
                    try:
                        im = np.asarray(Image.open(f"{NSB}/{c['filename']}").convert("RGB"))
                        key = c["filename"].split("/", 1)[1]
                        y0 = run(im, float(c["v0"]), key, None, 0)[0]
                        runner.set_steering(layer=l, vec=rv, alpha=10.0, mode="add")
                        o = (runner.run(im, float(c["v0"])) if lidar is None
                             else runner.run(im, float(c["v0"]), lidar_xyz=lidar.ego_points(key)))
                        runner.set_steering(None)
                        y1 = arc_full(np.asarray(o["trajectory"], float), WP_DT[A.model])
                        acc.append(abs(y1 - y0) / 10.0)
                    except Exception as _ex:                           # noqa: BLE001
                        if k == 0:
                            print(f"      L{l} 探测异常: {type(_ex).__name__}: {_ex}", flush=True)
                        continue
            reach.append(float(np.median(acc)) if acc else 0.0)
        print(f"  逐层因果可达性 |Δy|/α（随机方向，3 向 × {len(probe)} 场景）：")
        for l, x in enumerate(reach):
            g = "✓" if (pr[l] >= 3 and (fl[l] is None or fl[l] <= 30)) else "×"
            print(f"    L{l}  reach={x:.5f}   PR={pr[l]:5.1f} 地板={(fl[l] or float('nan')):5.1f}° 表征门{g}")
        ok = [l for l in range(len(reach))
              if pr[l] >= 3 and (fl[l] is None or fl[l] <= 30)]
        if not ok:
            ok = list(range(len(reach)))
        L = max(ok, key=lambda l: reach[l])
        print(f"  ⇒ 选 L={L}（可达性 {reach[L]:.5f}，是三重门原选 L{idx[0]} 的 "
              f"{reach[L]/max(reach[idx[0]],1e-12):.0f} 倍）", flush=True)

    D = b["d"][L]
    sig = float(np.linalg.norm(D.std(0)))
    VF = D.mean(0); VF = VF / max(np.linalg.norm(VF), 1e-12)
    if A.axis == "faith":
        VB = {bd: VF for bd in range(4)}
        print(f"  轴 = v_faith（非同源对照）", flush=True)
    elif A.axis == "speed_w":
        WD = vspeed_weighted(A.model, len(b["d"]))
        if not WD:
            print("  加权速度轴估不出，退出"); return
        tot = sum(abs(w) for _, w in WD.values()) or 1.0
        print(f"  轴 = **多层加权 −ŝ**  层={sorted(WD)}  "
              f"权重={[round(WD[l][1],3) for l in sorted(WD)]}   **阳性对照**", flush=True)

        def run(img, spd, key, vec, alpha):        # 覆盖：逐层同时注入
            runner.set_steering(None)
            if alpha != 0:
                for l, (d, w) in WD.items():
                    # 适配器内部已乘该层本次前向的 sigma，故这里**不再乘 sig**
                    runner.set_steering(layer=l, vec=-d, alpha=float(alpha) * w / tot,
                                        mode="add", accumulate=True)
            o = (runner.run(img, spd) if lidar is None
                 else runner.run(img, spd, lidar_xyz=lidar.ego_points(key)))
            runner.set_steering(None)
            t = np.asarray(o["trajectory"], float)
            return arc_full(t, WP_DT[A.model]), float(np.abs(t[:, 1]).max())
        VB = {bd: VF for bd in range(4)}
    elif A.axis == "speed":
        sp = vspeed_dir(A.model, L)
        if sp is None or np.linalg.norm(sp) < 1e-12:
            print("  速度轴估不出，退出"); return
        sp = -sp / np.linalg.norm(sp)          # **取反**：速度降低方向
        VB = {bd: sp for bd in range(4)}
        print(f"  轴 = −ŝ（速度降低方向，回归估计）  **阳性对照**", flush=True)
    else:
        VB = {}
        for bd in range(4):
            u = vdecel_dir(A.model, L, bd)
            if u is not None and np.linalg.norm(u) > 1e-12:
                VB[bd] = u / np.linalg.norm(u)
        print(f"  轴 = v_decel，可用速度段 {sorted(VB)}（band 3 刹车样本为 0，不做）",
              flush=True)
        pool = [c for c in pool if band_of(float(c["v0"])) in VB]
        print(f"  刺激集按速度段过滤后 {len(pool)} 个", flush=True)
    v = VF
    print(f"  注入层 L={L}  dim={len(VF)}  σ_L={sig:.4f}", flush=True)
    rands = [rng.normal(size=len(v)) for _ in range(A.n_rand)]
    rands = [r / np.linalg.norm(r) for r in rands]

    Y = {a: [] for a in ALPHAS}; LATM = {a: [] for a in ALPHAS}
    RS = [[] for _ in rands]
    for i, c in enumerate(pool):
        try:
            img = np.asarray(Image.open(f"{NSB}/{c['filename']}").convert("RGB"))
            runner.set_steering(None)
            key = c["filename"].split("/", 1)[1]
            vv = VB.get(band_of(float(c["v0"])), v)      # 按速度段取对应方向
            for a in ALPHAS:
                y, lat = run(img, float(c["v0"]), key, vv, a)
                Y[a].append(y); LATM[a].append(lat)
            for k, rv in enumerate(rands):
                RS[k].append([run(img, float(c["v0"]), key, rv, a)[0] for a in ALPHAS])
        except Exception as ex:                                       # noqa: BLE001
            print(f"  skip {c['scene']}: {type(ex).__name__}: {ex}", flush=True); continue
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pool)}", flush=True)

    ys = [float(np.mean(Y[a])) for a in ALPHAS]
    s_obs = slope(ALPHAS, ys)
    s_rand = [slope(ALPHAS, np.asarray(r).mean(0)) for r in RS if r]
    p = float((np.abs(s_rand) >= abs(s_obs)).mean()) if s_rand else float("nan")
    lat = [float(np.mean(LATM[a])) for a in ALPHAS]

    print(f"\n[{LABEL[A.model]}] 空场景注入 v_faith，α 阶梯上的规划速度 arc_full：")
    print("  α      " + "".join(f"{a:>8.1f}" for a in ALPHAS))
    print("  速度   " + "".join(f"{y:>8.3f}" for y in ys))
    print("  横偏   " + "".join(f"{x:>8.3f}" for x in lat))
    print(f"\n  斜率 dy/dα = {s_obs:+.4f}   （负 = 注入使其减速）")
    print(f"  随机方向零分布：中位 |斜率| = {np.median(np.abs(s_rand)):.4f}，"
          f"{A.n_rand} 个中 {sum(1 for x in s_rand if abs(x)>=abs(s_obs))} 个 ≥ 实测  ⇒ p={p:.3f}")
    print(f"  单调性：α<0 段斜率 {slope([a for a in ALPHAS if a<=0],[y for a,y in zip(ALPHAS,ys) if a<=0]):+.4f}"
          f"   α>0 段 {slope([a for a in ALPHAS if a>=0],[y for a,y in zip(ALPHAS,ys) if a>=0]):+.4f}")
    s_lat = slope(ALPHAS, lat)
    pos = slope([a for a in ALPHAS if a >= 0], [y for a, y in zip(ALPHAS, ys) if a >= 0])
    neg = slope([a for a in ALPHAS if a <= 0], [y for a, y in zip(ALPHAS, ys) if a <= 0])
    ratio = (pos / neg) if abs(neg) > 1e-12 else float("inf")
    i8 = ALPHAS.index(8)
    C1 = (s_obs < 0) and (p < 0.05)
    C2 = (pos * neg > 0) and (1/3 <= abs(ratio) <= 3)
    C3 = abs(s_lat) <= abs(s_obs) / 3
    C4 = abs(ys[i8] - ys[ALPHAS.index(0)]) >= 0.5 * abs(ys[-1] - ys[ALPHAS.index(0)])
    print(f"\n  预注册判定（results/PREREG_steer_brake.md）")
    print(f"    C1 显著(斜率<0 且 p<0.05)         {'✓' if C1 else '✗'}   斜率={s_obs:+.4f} p={p:.3f}")
    print(f"    C2 单调(两段同号、比值∈[1/3,3])    {'✓' if C2 else '✗'}   α>0={pos:+.4f} α<0={neg:+.4f}")
    print(f"    C3 特异(横偏斜率 ≤ 1/3)           {'✓' if C3 else '✗'}   横偏斜率={s_lat:+.5f}")
    print(f"    C4 剂量(α≤8 已出现一半以上效应)    {'✓' if C4 else '✗'}")
    print(f"    ⇒ {'**四条全过：该方向因果驱动刹车**' if all([C1,C2,C3,C4]) else '未证实'}")
    json.dump({"model": A.model, "axis": A.axis, "layer": int(L), "sigma": sig, "alphas": ALPHAS,
               "C1": C1, "C2": C2, "C3": C3, "C4": C4, "slope_lat": s_lat,
               "slope_pos": pos, "slope_neg": neg,
               "arc_full": ys, "lat_max": lat, "slope": s_obs,
               "rand_slopes": s_rand, "p": p, "n_scenes": len(Y[0])},
              open(RES / f"steer_brake_{A.model}.json", "w"), ensure_ascii=False, indent=1)
    print(f"-> {RES}/steer_brake_{A.model}.json")


if __name__ == "__main__":
    main()
