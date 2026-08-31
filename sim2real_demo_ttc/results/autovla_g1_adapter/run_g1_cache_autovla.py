"""把 G1 语料喂给 AutoVLA，缓存 36 层池化表征 + 规划速度（G/F 轴原料）。

行为量 `commanded_speed = ‖traj[0]‖ / 0.5 s`，与 DiffusionDrive/LTF/DDV2 同口径。
ego 速度**两条件共用 clean 帧锚定**（与其余候选同构）。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

R = Path("/data/ruolin/uwm/sim2real_demo_ttc/results")
W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
sys.path.insert(0, str(R / "autovla_g1_adapter"))
from autovla_adapter import AutoVLARunner            # noqa: E402
import numpy as np                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["A", "D2a"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = W / "autovla_cache"; out_dir.mkdir(exist_ok=True)
    evs = [json.loads(l) for l in open(W / "mining" / "events_all.jsonl")]
    matched = {t: set((W / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a", "D2b", "D2c") if (W / "mining" / f"matched_{t}.txt").exists()}
    keep = [e for e in evs if e["event_type"] in args.types
            and not (e["event_type"] in matched and e["event_id"] not in matched[e["event_type"]])]
    if args.limit:
        keep = keep[: args.limit]
    print(f"[AVLA] {len(keep)} 事件待缓存 -> {out_dir}", flush=True)

    runner = AutoVLARunner(device=args.device)
    t0, done, fail = time.time(), 0, []
    for i, ev in enumerate(keep):
        p = out_dir / f"{ev['event_id']}.npz"
        if p.exists():
            continue
        try:
            anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
            store, extra = {}, {}
            for cond in ("clean", "ghost"):
                o = runner.run(ev[f"x_{cond}_frames"][0]["sd_token"], anchor)
                for pool, v in o["pooled"].items():
                    if v is not None:
                        store[f"{cond}/{pool}"] = v
                store[f"{cond}/traj"] = np.asarray(o["trajectory"], np.float32)
                store[f"{cond}/v_plan"] = np.array([o["commanded_speed"]], np.float32)
                extra[f"{cond}_n_image_tokens"] = o["n_image_tokens"]
                extra[f"{cond}_image_token_id"] = o["image_token_id"]
                extra[f"{cond}_seq_len"] = o["seq_len"]
            meta = {k: v for k, v in ev.items() if k != "ttc_curve"}
            meta.update(extra)
            np.savez_compressed(p, meta=json.dumps(meta, ensure_ascii=False), **store)
            done += 1
        except Exception as exc:                                    # noqa: BLE001
            fail.append({"event_id": ev["event_id"], "error": f"{type(exc).__name__}: {exc}"})
            if len(fail) <= 5:
                print(f"[AVLA] FAIL {ev['event_id']}: {exc}", flush=True)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[AVLA] {i+1}/{len(keep)} done={done} fail={len(fail)} "
                  f"{el:.0f}s ({el/max(1,done):.2f}s/event)", flush=True)
    (W / "results" / "autovla_g1_cache_report.json").write_text(json.dumps(
        {"n_events": len(keep), "cached": done, "failed": fail[:20], "n_failed": len(fail),
         "elapsed_s": time.time() - t0}, indent=2, ensure_ascii=False))
    print(f"[AVLA] 完成 {done}/{len(keep)}，失败 {len(fail)}")


if __name__ == "__main__":
    main()
