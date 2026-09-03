"""速度覆盖合理性检查|把喂给模型的"当前速度观测"覆盖成受控值，先看输出还正不正常。

工单：2026-09-03。**这不是给模型下达目标速度**（接口调查已确认除 SimLingo 外
都没有目标速度通道），而是**控制一个协变量**：跨事件的背景速度差异是 F-3 读数里
最大的噪声源之一（LTF 的 v_plan 与输入速度 corr = 0.9996，图像可调动幅度仅
0.0855 m/s，见 ltf_sanity_check_report_zh.md §4.2）。把它压平后，
臂间差异更接近纯图像驱动。

**先做合理性检查再信**：覆盖后的输出必须
  (a) 不退化（不是常数、不全零、不乱跳）；
  (b) 仍随图像内容变化（否则说明模型只是在复读输入速度）；
  (c) 对覆盖值本身不过度敏感（故并列测 5 / 8 / 11 m/s，不只测一个数）。
只跑 ghost 帧单臂，不跑四臂 —— 本模块只回答"输出正不正常"。
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

# AutoVLA 跑在没有 numpy 的 py312 解释器里，依赖来自 Alpamayo 的 venv；
# 必须在 import numpy 之前挂上（append 而非 insert，其余环境自带的 numpy 仍优先）。
# 与 f3_occlusion_vla.py:32-36 同一处理。
if importlib.util.find_spec("numpy") is None:
    for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
               "/data/Zhengyang/alpamayo/src"):
        if _p not in sys.path:
            sys.path.append(_p)

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["dd", "ltf", "ddv2", "simlingo", "autovla"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--floors", type=float, nargs="+", default=[0.0, 5.0, 8.0, 11.0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--work", default=str(ROOT / "variants" / "n1_d2"))
    ap.add_argument("--sl-config", default=str(ROOT / "configs" / "n1_d2.yaml"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = str(RES / f"speed_probe_{args.model}.json")

    evs = [json.loads(l) for l in open(Path(args.work) / "mining" / "events_all.jsonl")]
    evs = [e for e in evs if e["event_type"] == "A"
           and e.get("x_ghost_frames") and e["x_ghost_frames"][0].get("bbox_xyxy")]
    # 均匀取样，覆盖不同真实速度
    idx = np.linspace(0, len(evs) - 1, min(args.n, len(evs))).astype(int)
    evs = [evs[i] for i in idx]
    print(f"[PROBE/{args.model}] {len(evs)} 事件 × {len(args.floors)} 个覆盖值")

    real_spd = [float(np.mean([f["ego_speed_mps"] for f in e["x_clean_frames"]])) for e in evs]

    # ---------------- 构造 runner（与 f3_occlusion_* 同一套） ----------------
    if args.model in ("dd", "ltf", "ddv2"):
        import cv2
        sys.path.insert(0, str(RES / "diffusiondrive_g1_adapter"))
        sys.path.insert(0, str(RES / "ltf_g1_adapter"))
        sys.path.insert(0, str(RES / "ddv2_g1_adapter"))
        import dd_adapter as DD                                     # noqa: F401
        R = {"dd": lambda: __import__("dd_adapter").DDRunner,
             "ltf": lambda: __import__("ltf_adapter").LTFRunner,
             "ddv2": lambda: __import__("ddv2_adapter").DDV2Runner}[args.model]()
        runner = R(device=args.device)
        lid = None
        if args.model == "ddv2":
            from ddv2_adapter import NuScenesLidar
            lid = NuScenesLidar(NUSC)

        def infer(ev, v):
            fg = ev["x_ghost_frames"][0]
            img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / fg["filename"])), cv2.COLOR_BGR2RGB)
            kw = {} if lid is None else {"lidar_xyz": lid.ego_points(fg.get("sd_token"))}
            r = runner.run(img, v, **kw)
            return float(r["commanded_speed"]), np.asarray(r["trajectory"], float)

    elif args.model == "simlingo":
        import cv2
        from omegaconf import OmegaConf
        cfg = OmegaConf.to_container(OmegaConf.load(args.sl_config), resolve=True)
        cfg["model"]["device"] = args.device
        from simlingo_runner import SimLingoRunner
        from g2_cache import commanded_speed
        runner = SimLingoRunner(cfg, capture_hidden=False)

        def infer(ev, v):
            fg = ev["x_ghost_frames"][0]
            img = cv2.cvtColor(cv2.imread(str(Path(NUSC) / fg["filename"])), cv2.COLOR_BGR2RGB)
            res = runner.infer(img, v, pool_modes=())
            return float(commanded_speed(res.waypoints)), np.asarray(res.waypoints, float)

    else:                                                            # autovla
        # 不要手动插 autovla_deps —— 适配器的 _bootstrap() 有严格的导入顺序
        # （venv -> torchvision 注册算子 -> 隔离的 transformers 4.49 -> 仓库根，见 §CE/A35）。
        # 提前插会导致 "operator torchvision::nms does not exist"。
        sys.path.insert(0, str(RES / "autovla_g1_adapter"))
        from autovla_adapter import AutoVLARunner
        runner = AutoVLARunner(device=args.device, nuscenes_root=NUSC)

        def infer(ev, v):
            o = runner.run(ev["x_ghost_frames"][0]["sd_token"], v)
            return float(o["commanded_speed"]), np.asarray(o["trajectory"], float)

    # ---------------- 逐覆盖值跑 ----------------
    out = {"model": args.model, "n_events": len(evs), "floors": args.floors,
           "real_speed": {"mean": float(np.mean(real_spd)), "min": float(np.min(real_spd)),
                          "max": float(np.max(real_spd))}, "runs": {}}
    for fl in args.floors:
        vs, arcs, lats, fed = [], [], [], []
        for e, rv in zip(evs, real_spd):
            v_in = max(rv, fl) if fl > 0 else rv
            try:
                cs, tj = infer(e, v_in)
            except Exception as ex:                                  # noqa: BLE001
                print(f"    [skip] {e['event_id']}: {ex}"); continue
            fed.append(v_in); vs.append(cs)
            w = np.asarray(tj, float)[:, :2]
            arcs.append(float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()))
            lats.append(float(abs(w[-1, 1])))
        vs, fed = np.array(vs), np.array(fed)
        key = "real" if fl == 0 else f"floor{fl:g}"
        r = {"n": len(vs), "fed_speed_mean": float(fed.mean()),
             "v_plan": {"mean": float(vs.mean()), "std": float(vs.std()),
                        "min": float(vs.min()), "max": float(vs.max()),
                        "n_unique": int(len(np.unique(np.round(vs, 4))))},
             "traj_arclen_mean": float(np.mean(arcs)),
             "endpoint_lat_mean": float(np.mean(lats)),
             "fed_is_constant": bool(np.std(fed) < 1e-9),
             "corr_vplan_fed": (float(np.corrcoef(fed, vs)[0, 1])
                                if np.std(fed) > 1e-9 else None),
             # 关键：扣掉输入速度后还剩多少"图像驱动"的变异。
             # **注意**：fed 恒定时（如 floor 高于所有真实速度）无从回归，
             # 此值退化为原始 std —— 但那恰是最干净的图像驱动变异度量（输入完全一致）。
             "resid_std_after_regressing_fed": (
                 float(np.std(vs - np.polyval(np.polyfit(fed, vs, 1), fed)))
                 if np.std(fed) > 1e-9 else float(vs.std())),
             "degenerate": {"all_same": bool(len(np.unique(np.round(vs, 3))) == 1),
                            "all_zero": bool(np.all(vs < 0.05)),
                            "all_straight": bool(np.all(np.array(lats) < 0.2))},
             "per_event_v_plan": [round(float(x), 4) for x in vs]}
        out["runs"][key] = r
        print(f"  {key:9s} 喂入均值 {fed.mean():5.2f}  v_plan {vs.mean():6.3f}±{vs.std():5.3f} "
              f"[{vs.min():.3f},{vs.max():.3f}] 唯一值 {r['v_plan']['n_unique']:3d}  "
              f"扣输入后残差 std {r['resid_std_after_regressing_fed']:.4f}  "
              f"弧长 {np.mean(arcs):5.2f}  退化={any(r['degenerate'].values())}")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[PROBE/{args.model}] wrote {args.out}")


if __name__ == "__main__":
    main()
