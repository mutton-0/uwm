"""Stage D 解析臂|通过动作头的 Jacobian 解析预测 steering 效应,与经验 α 扫描比对(H3)。

计划依据:direction_vector_discovery_validation_plan.md Stage D「新增(仅适用于连续/近线性动作头)」。

**为什么 SimLingo 满足做解析投影的架构条件**:
  speed_wps_head = Linear(896->256) -> SiLU -> Linear(256->2, bias=False),预测再 cumsum。
  行为量 commanded_speed(wp) = ||wp[0] - wp[2]|| * 2 = 2*||head(f_1) + head(f_2)||,
  只依赖 speed_wps query 段的第 1、2 个位置,且从最终隐状态到该量只隔一层 SiLU MLP。
  ⇒ ∂v_cmd/∂h_23 可由 autograd **精确**求出(不是有限差分近似)。

**解析预测**(注入 Z' = Z + α·σ_L·v̂ 于第 L 层的 query 位置):
  残差流恒等路径给出 Δh_23 ≈ Δh_L = α·σ_L·v̂ ⇒
      Δv_cmd_pred(α) = α · σ_L · Σ_{t∈query} ⟨ g_t , v̂ ⟩ ,   g = ∂v_cmd/∂h_23
  L=23 时该式**仅**忽略 head 的二阶项(SiLU 曲率);L<23 时还额外忽略 L→23 之间
  注意力/MLP 对该扰动的加工 —— 两者的差距正是 H3 要测的东西。

  对 tokens=vision 的注入,恒等路径**不直接到达** query 位置,解析预测为 0;
  经验效应若显著非零,说明效应完全由注意力搬运,解析捷径在该设定下不适用 —— 这是
  H3 的边界结论,不是失败。
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache        # noqa: E402
from n1_readout import supervised_direction, auc, proj   # noqa: E402

RES = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")


def head_grad(runner, h_last):
    """g = ∂ commanded_speed / ∂ h_23[query 段]  —— autograd 精确解。"""
    n_drv = int(runner._len_driving)
    adaptors = runner.model.adaptors
    drv = adaptors.driving
    core = runner.model.language_model.model
    while not hasattr(core, "norm"):
        core = core.model if hasattr(core, "model") else core.base_model
    final_norm = core.norm
    n_route = drv.sizes.get("route", 0) if "route" in drv.order else 0
    x = h_last[0, -n_drv:, :].detach().float().clone().requires_grad_(True)
    f = final_norm(x.to(h_last.dtype)).float()
    sp = f[n_route:n_route + drv.sizes["speed_wps"]]
    d = drv.heads["speed_wps"](sp.to(next(drv.heads["speed_wps"].parameters()).dtype)).float()
    wp = d.cumsum(0)
    v_cmd = torch.linalg.vector_norm(wp[0] - wp[2]) * 2.0
    g, = torch.autograd.grad(v_cmd, x)
    return g.detach().cpu().numpy(), float(v_cmd.detach()), n_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/data/ruolin/uwm/sim2real_demo_ttc/configs/n1_d2.yaml")
    ap.add_argument("--pool-mode", default="region_mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-events", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vecs", nargs="+", required=True,
                    help="形如 name:path.npy:layer:tokens")
    ap.add_argument("--out", default=str(RES / "d2_analytic.json"))
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["model"]["device"] = args.device
    work = Path(cfg["paths"]["work_dir"]); root = Path(cfg["paths"]["nuscenes_root"])
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap), verbose=False)
    by = defaultdict(list)
    for eid, e in items.items():
        by[evmap[eid]["event_type"]].append(e)
    scenes = sorted({e["scene"] for v in by.values() for e in v})
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(scenes))
    n_dir, n_sel = int(len(scenes) * .5), int(len(scenes) * .25)
    test = {scenes[k] for i, k in enumerate(perm) if i >= n_dir + n_sel}
    test_a = [e for e in by["A"] if e["scene"] in test][: args.max_events]
    print(f"[D-ana] S_test A 事件 n={len(test_a)}（与 t2_steer 同一 seed/同一三分）")

    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=True)

    specs = []
    for s in args.vecs:
        nm, path, L, tok = s.split(":")
        v = np.load(path).astype(np.float32)[int(L)]
        specs.append({"name": nm, "layer": int(L), "tokens": tok,
                      "v": v / (np.linalg.norm(v) + 1e-8)})

    per_event = []
    for i, e in enumerate(test_a):
        ev = evmap[e["meta"]["event_id"]]
        clean = ev["x_clean_frames"]
        anchor = float(np.mean([f["ego_speed_mps"] for f in clean]))
        runner.set_steering(None)
        gs, sigmas, vcmds = [], [], []
        for fr in clean:
            img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
            r = runner.infer(img, fr["ego_speed_mps"], pool_modes=("vision_mean",), prompt_speed=anchor)
            hs = runner._layer_outputs[-runner.n_layers:]
            g, vc, n_route = head_grad(runner, hs[-1])
            n_drv = int(runner._len_driving)
            gs.append(g); vcmds.append(vc)
            sigmas.append({sp["layer"]: float(hs[sp["layer"]][0, -n_drv:, :].float().std())
                           for sp in specs})
        G = np.mean(np.stack(gs), 0)                    # [n_drv, 896]
        row = {"event_id": ev["event_id"], "scene": ev["scene_name"],
               "v_cmd_base": float(np.mean(vcmds)), "grad_norm": float(np.linalg.norm(G))}
        for sp in specs:
            sg = float(np.mean([s[sp["layer"]] for s in sigmas]))
            slope = sg * float(G @ sp["v"]).__float__() if G.ndim == 1 else sg * float((G @ sp["v"]).sum())
            row[sp["name"]] = {"sigma_L": sg,
                               "analytic_slope_dv_dalpha": (slope if sp["tokens"] == "query" else 0.0),
                               "analytic_slope_identity_path": slope,
                               "applicable": sp["tokens"] == "query"}
        per_event.append(row)
        if (i + 1) % 10 == 0:
            print(f"[D-ana] {i+1}/{len(test_a)}")

    out = {"n_events": len(per_event), "pool_mode": args.pool_mode, "seed": args.seed,
           "specs": [{k: v for k, v in s.items() if k != "v"} for s in specs],
           "per_event": per_event,
           "method": "g = ∂commanded_speed/∂h_23[query]（autograd 精确）；"
                     "Δv_pred = α·σ_L·Σ_t⟨g_t, v̂⟩（残差流恒等路径）",
           "caveat": "tokens=vision 的注入不经恒等路径到达 query 位置，解析预测记 0（applicable=false）。"}
    for sp in specs:
        a = np.array([r[sp["name"]]["analytic_slope_identity_path"] for r in per_event])
        out.setdefault("summary", {})[sp["name"]] = {
            "mean_analytic_slope": float(a.mean()), "sd": float(a.std()),
            "frac_negative": float((a < 0).mean()), "layer": sp["layer"], "tokens": sp["tokens"]}
        print(f"[D-ana] {sp['name']:24s} L{sp['layer']:2d} tokens={sp['tokens']:6s} "
              f"解析斜率 dΔv/dα = {a.mean():+.4f} ± {a.std():.4f} m/s per α  "
              f"(负号占比 {(a<0).mean():.2f})")
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[D-ana] wrote {args.out}")


if __name__ == "__main__":
    main()
