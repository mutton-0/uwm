"""预注册检验：v_faith 能否区分「危险引起的急刹」与「非危险引起的急刹」。

见 results/PREREG_vfaith_explains_harshbrake.md（写于任何模型跑之前）。
读数 score_i = < h(急刹帧_i, L*) , v̂_faith^LHD(L*) >，比较正/负例两组。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
from c_axis_hazard_patch import arc_full, WP_DT                       # noqa: E402
from f_vfaith_direction import load_by_side, pick_layers              # noqa: E402

LABEL = {"dd": "DiffusionDrive", "ltf": "LTF", "ddv2": "DiffusionDriveV2",
         "simlingo": "SimLingo"}
NUSC = "/data/dataset/nuscenes/v1.0-trainval"


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (len(a) * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LABEL))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--pool", default="vision_mean")
    ap.add_argument("--sl-config", default=str(ROOT / "configs/n1_d2.yaml"))
    A = ap.parse_args()
    ev = json.load(open(RES / "harsh_brake_labeled_nusc.json"))

    # ---- 轴：左舵 brake-first 池，三重门选层。**不看本验证集** ----
    bs = load_by_side("lead", A.model)
    b = bs["LHD"]
    idx, snr, pr, fl = pick_layers(b["d"], b["scene"], np.random.default_rng(7), 1)
    L = idx[0]
    v = b["d"][L].mean(0); v = v / max(np.linalg.norm(v), 1e-12)
    print(f"[{LABEL[A.model]}] L*={L}  SNR={snr[L]:.1f} PR={pr[L]:.1f} "
          f"地板={fl[L]:.1f}°  维度={len(v)}  轴来自左舵 {len(b['scene'])} 个 brake-first 事件",
          flush=True)

    from PIL import Image
    IS_SL = A.model == "simlingo"
    if IS_SL:
        from omegaconf import OmegaConf
        from simlingo_runner import SimLingoRunner
        import torch as _t
        cfg = OmegaConf.to_container(OmegaConf.load(A.sl_config), resolve=True)
        cfg["model"]["device"] = A.device
        runner = SimLingoRunner(cfg, capture_hidden=True)
        nL = runner.n_layers

        def act(img, spd, tok):
            r = runner.infer(img, spd, pool_modes=("vision_mean",))
            hs = runner._layer_outputs[-nL:]
            ids = runner._adaptor_dict["language__ids"][0]
            vis = _t.nonzero(ids == runner.img_context_token_id).flatten()
            return hs[L][0, vis, :].float().mean(0).cpu().numpy(), \
                arc_full(r.waypoints, WP_DT["simlingo"])
    else:
        for d_ in ("diffusiondrive_g1_adapter", "ltf_g1_adapter", "ddv2_g1_adapter"):
            sys.path.insert(0, str(RES / d_))
        if A.model == "dd":
            from dd_adapter import DDRunner as R_
        elif A.model == "ltf":
            from ltf_adapter import LTFRunner as R_
        else:
            from ddv2_adapter import DDV2Runner as R_
        runner = R_(device=A.device)
        lidar = None
        if A.model == "ddv2":
            from ddv2_adapter import NuScenesLidar
            lidar = NuScenesLidar(NUSC)

        def act(img, spd, tok):
            if lidar is None:
                o = runner.run(img, spd)
            else:
                o = runner.run(img, spd, lidar_xyz=lidar.ego_points(tok))
            return np.asarray(o["pooled"][A.pool][L], np.float32), \
                arc_full(o["trajectory"], WP_DT[A.model])

    rows, skip, HH = [], 0, []
    for i, e in enumerate(ev):
        try:
            img = np.asarray(Image.open(f"{NUSC}/{e['filename']}").convert("RGB"))
            if not IS_SL:
                runner.set_steering(None)
            h, arc = act(img, float(e["v_at_peak"]), e["sd_token"])
        except Exception as ex:                                       # noqa: BLE001
            skip += 1
            print(f"  skip {e['scene']}: {type(ex).__name__}: {ex}", flush=True); continue
        rows.append({**{k: e[k] for k in ("scene", "side", "group", "peak_decel", "v_at_peak")},
                     "score": float(np.asarray(h, float) @ v), "arc_full": arc})
        HH.append(np.asarray(h, np.float32))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(ev)}", flush=True)

    out = RES / f"harsh_probe_{A.model}.json"
    hz = [r["score"] for r in rows if r["group"] == "hazard"]
    nh = [r["score"] for r in rows if r["group"] == "no_hazard"]
    rng = np.random.default_rng(0)
    allv = np.array(hz + nh); n1 = len(hz)
    obs = np.mean(hz) - np.mean(nh)
    null = np.array([(lambda p: allv[p[:n1]].mean() - allv[p[n1:]].mean())(rng.permutation(len(allv)))
                     for _ in range(20000)])
    p1 = float((null >= obs).mean())                        # 单边：H1 预测正例更高
    dlt = cliffs_delta(hz, nh)
    res = {"model": A.model, "layer": int(L), "n_hazard": len(hz), "n_no_hazard": len(nh),
           "skipped": skip, "mean_hazard": float(np.mean(hz)), "mean_no_hazard": float(np.mean(nh)),
           "diff": float(obs), "perm_p_onesided": p1, "cliffs_delta": dlt, "rows": rows}
    by = {}
    for s in ("LHD", "RHD"):
        a = [r["score"] for r in rows if r["group"] == "hazard" and r["side"] == s]
        c = [r["score"] for r in rows if r["group"] == "no_hazard" and r["side"] == s]
        if len(a) >= 3 and len(c) >= 3:
            by[s] = {"n": [len(a), len(c)], "diff": float(np.mean(a) - np.mean(c)),
                     "cliffs_delta": cliffs_delta(a, c)}
    res["by_side"] = by
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    # **另存原始激活**：供「左舵急刹提轴 → 右舵急刹检验」这类训练/测试切分使用
    np.savez_compressed(RES / f"harsh_acts_{A.model}.npz",
                        h=np.stack(HH).astype(np.float32),
                        scene=np.array([r["scene"] for r in rows]),
                        side=np.array([r["side"] for r in rows]),
                        group=np.array([r["group"] for r in rows]),
                        arc_full=np.array([r["arc_full"] for r in rows], np.float32),
                        v_at_peak=np.array([r["v_at_peak"] for r in rows], np.float32),
                        layer=np.array([int(L)]))

    print(f"\n[{LABEL[A.model]}] L*={L}  跳过 {skip}")
    print(f"  正例(前方有物体) n={len(hz)}  score 均值 {np.mean(hz):+.4f}")
    print(f"  负例(红灯等)     n={len(nh)}  score 均值 {np.mean(nh):+.4f}")
    print(f"  差 {obs:+.4f}   单边置换 p={p1:.4f}   Cliff's δ={dlt:+.3f}")
    print(f"  H1 (p<0.05): {'成立' if p1 < 0.05 else '**不成立**'}   "
          f"H2 (δ≥0.33): {'成立' if dlt >= 0.33 else '**不成立**'}")
    for s, d in by.items():
        print(f"  H3 {s}: 差 {d['diff']:+.4f}  δ={d['cliffs_delta']:+.3f}  n={d['n']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
