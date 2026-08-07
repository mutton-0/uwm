"""预测-验证：在两个 held-out 集上采 CoT 文本（只取 ghost 帧，语言读数只需要它）。

集的定义与三条预测已冻结在 `results/amendments.md` §PV，本脚本不改口径。
  V-A  = tier_m 的 A 类 312（独立挖掘轮，CoT 从未采过）
  V-BC = n1_d2 的 B+C 类 657（b 与 CoT 都是完全 held-out）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--types", required=True, help="逗号分隔的事件类型，如 A 或 B,C")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg["model"]["use_cot"] = True
    if args.device:
        cfg["model"]["device"] = args.device
    work = Path(cfg["paths"]["work_dir"])
    root = Path(cfg["paths"]["nuscenes_root"])
    types = tuple(args.types.split(","))

    events = [json.loads(l) for l in open(work / "mining" / "events_all.jsonl")]
    events = [e for e in events if e["event_type"] in types]
    # SimLingoRunner 初始化会 os.chdir 到 simlingo repo —— 相对路径必须在此之前定死
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [e for e in events if e["event_id"] not in done]
    print(f"[PV] {work.name} 类型={types}  共 {len(events)}，待跑 {len(todo)}（已有 {len(done)}）")
    if not todo:
        return

    from simlingo_runner import SimLingoRunner
    runner = SimLingoRunner(cfg, capture_hidden=False)
    for i, ev in enumerate(todo):
        # prompt 锚定沿用 G2/T2.5 口径：写 clean 帧均速，唯一差异是图像
        anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        txt = []
        for fr in ev["x_ghost_frames"]:
            img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
            r = runner.infer(img, fr["ego_speed_mps"], pool_modes=(), prompt_speed=anchor)
            txt.append(r.language)
        done[ev["event_id"]] = {"ghost": txt, "type": ev["event_type"]}
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            out_path.write_text(json.dumps(done, ensure_ascii=False))
            print(f"[PV] {i+1}/{len(todo)}")
    out_path.write_text(json.dumps(done, ensure_ascii=False))
    print(f"[PV] wrote {out_path}")


if __name__ == "__main__":
    main()
