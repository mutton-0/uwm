"""Alpamayo-R1 × G1 语料：逐层表征 + 规划速度缓存（G/F/C 轴共用原料）。

方法纪律（工单 §0）：**用现有 G1 nuScenes 鬼探头语料 + N1 负例体系发现并测量**，
不在 Alpamayo 自己的原生域（PhysicalAI-AV）里找方向再迁移。
本脚本只做一件事：把 G1 事件的 clean/ghost 两条件各跑一次前向，
把**预填充（prefill）阶段**逐层的池化激活与规划速度落盘。

实现要点：
  * Alpamayo 的推理是 VLM rollout（先生成 CoT 再采轨迹），钩子会被触发很多次。
    我们要的是**整条 prompt 的那一次前向**（prefill），即 seq_len 最大的那一批调用，
    逐层取其首次出现；decode step（seq_len=1）一律丢弃。
  * 隐状态很大（6 路相机 ⇒ S 可达数千），**在钩子内当场池化**，不保留全张量。
  * 池化口径与其余候选对齐：vision_mean（图像 token 均值）/ last_token / seq_mean。
    图像 token 通过 input_ids 中**最长连续同 id 段**自动识别（Qwen2-VL 系的 image pad token），
    识别结果随缓存落盘，供核查。
  * ego 运动史**锚定到 clean 帧**（与 SimLingo 的 prompt_anchor、DiffusionDrive 的 status 锚定同构），
    使两条件唯一差异是图像。
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

# 依赖来自 Zhengyang 的 alpamayo venv（本机 py312 环境本身没有 numpy/torch）；
# 必须在 import numpy 之前挂上，否则先 import 就找不到。
for _p in ("/data/Zhengyang/alpamayo/ar1_venv/lib/python3.12/site-packages",
           "/data/Zhengyang/alpamayo/src"):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch

W = Path("/data/ruolin/uwm/sim2real_demo_ttc/variants/n1_d2")
sys.path.insert(0, "/data/ruolin/uwm/sim2real_demo_ttc/scripts")

POOLS = ("vision_mean", "last_token", "seq_mean")


def longest_run_token(ids: np.ndarray):
    """返回 input_ids 中最长连续同值段的 (token_id, 起点, 长度) —— 即图像 pad token 段。"""
    best = (None, 0, 0)
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[i]:
            j += 1
        if j - i + 1 > best[2]:
            best = (int(ids[i]), i, j - i + 1)
        i = j + 1
    return best


class PooledCapture:
    """在钩子里当场池化，只留 [n_layers, C] 三份，不保留全序列张量。"""

    def __init__(self, model, prefix="vlm.model.language_model.layers"):
        layers = [(n, m) for n, m in model.named_modules()
                  if n.startswith(prefix) and m.__class__.__name__.endswith("DecoderLayer")]
        layers.sort(key=lambda x: int(x[0].rsplit(".", 1)[-1]))
        assert layers, f"未找到 DecoderLayer（prefix={prefix}）"
        self.n_layers = len(layers)
        self.mask = None            # bool[S] 图像 token
        self.reset()
        self._h = [m.register_forward_hook(self._mk(i)) for i, (_n, m) in enumerate(layers)]

    def reset(self):
        self.pooled = {p: [None] * self.n_layers for p in POOLS}
        self.seen_len = [0] * self.n_layers

    def _mk(self, i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            S = h.shape[1]
            if S <= 1 or S <= self.seen_len[i]:       # 只留最长的一次（prefill）
                return
            self.seen_len[i] = S
            x = h[0].float()
            self.pooled["last_token"][i] = x[-1].cpu().numpy()
            self.pooled["seq_mean"][i] = x.mean(0).cpu().numpy()
            if self.mask is not None and len(self.mask) == S and self.mask.any():
                m = torch.as_tensor(self.mask, device=x.device)
                self.pooled["vision_mean"][i] = x[m].mean(0).cpu().numpy()
        return hook

    def remove(self):
        for h in self._h:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["A", "D2a"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--events", default="", help="事件 id 清单；给定时忽略 --types")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(W / "alpa_cache"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    evs = [json.loads(l) for l in open(W / "mining" / "events_all.jsonl")]
    matched = {t: set((W / "mining" / f"matched_{t}.txt").read_text().split())
               for t in ("D2a",) if (W / "mining" / f"matched_{t}.txt").exists()}
    if args.events:
        want = set(Path(args.events).read_text().split())
        keep = [e for e in evs if e["event_id"] in want]
    else:
        keep = [e for e in evs if e["event_type"] in args.types
                and not (e["event_type"] in matched and e["event_id"] not in matched[e["event_type"]])]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    from alpamayo_runner import AlpamayoRunner, plan_speed
    runner = AlpamayoRunner(device=args.device)
    cap = PooledCapture(runner.model)
    print(f"[ALPA] 可读层 n_layers={cap.n_layers}；候选事件 {len(keep)}")

    # 覆盖预筛：adapter 无条件构造 6.4 s 未来轨迹，贴近场景末尾的事件不可用
    usable = [e for e in keep
              if all(runner.covers(e["scene_name"], e[f"x_{c}_frames"][0]["t"])
                     for c in ("clean", "ghost"))]
    print(f"[ALPA] 覆盖预筛后可用 {len(usable)}/{len(keep)}")
    if args.limit:
        usable = usable[: args.limit]

    done, fail, t0 = 0, [], time.time()
    for i, ev in enumerate(usable):
        p = out_dir / f"{ev['event_id']}.npz"
        if p.exists():
            continue
        anchor_t = ev["x_clean_frames"][0]["t"]
        store, meta_extra = {}, {}
        try:
            for cond in ("clean", "ghost"):
                t_sec = ev[f"x_{cond}_frames"][0]["t"]
                data = runner.load(ev["scene_name"], t_sec)
                if cond == "ghost":
                    a = runner.load(ev["scene_name"], anchor_t)
                    data["ego_history_xyz"] = a["ego_history_xyz"]
                    data["ego_history_rot"] = a["ego_history_rot"]
                messages = runner.helper.create_message(data["image_frames"].flatten(0, 1))
                inputs = runner.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    continue_final_message=True, return_dict=True, return_tensors="pt")
                ids = inputs["input_ids"][0].cpu().numpy()
                tok, st, ln = longest_run_token(ids)
                cap.mask = (ids == tok)
                meta_extra[f"{cond}_image_token_id"] = tok
                meta_extra[f"{cond}_n_image_tokens"] = int(cap.mask.sum())
                meta_extra[f"{cond}_seq_len"] = int(len(ids))
                mi = runner.helper.to_device({"tokenized_data": inputs,
                                              "ego_history_xyz": data["ego_history_xyz"],
                                              "ego_history_rot": data["ego_history_rot"]}, args.device)
                cap.reset()
                torch.manual_seed(runner.seed); torch.cuda.manual_seed_all(runner.seed)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, _r, extra = runner.model.sample_trajectories_from_data_with_vlm_rollout(
                        data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                        max_generation_length=256, return_extra=True)
                traj = pred_xyz.float().cpu().numpy()[0, 0, 0]
                store[f"{cond}/traj"] = traj.astype(np.float32)
                store[f"{cond}/v_plan"] = np.array([plan_speed(traj)], np.float32)
                for pool in POOLS:
                    v = cap.pooled[pool]
                    if all(x is not None for x in v):
                        store[f"{cond}/{pool}"] = np.stack(v).astype(np.float32)
                c = extra.get("cot", [""])[0] if extra else ""
                store[f"{cond}/cot"] = np.array([c if isinstance(c, str) else str(c)])
            meta = {k: v for k, v in ev.items() if k != "ttc_curve"}
            meta.update(meta_extra)
            np.savez_compressed(p, meta=json.dumps(meta, ensure_ascii=False), **store)
            done += 1
        except Exception as exc:                                        # noqa: BLE001
            fail.append({"event_id": ev["event_id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ALPA] FAIL {ev['event_id']}: {exc}")
        if args.smoke and done >= 2:
            break
        if (i + 1) % 10 == 0 or args.smoke:
            el = time.time() - t0
            print(f"[ALPA] {i+1}/{len(usable)} done={done} fail={len(fail)} "
                  f"{el:.0f}s ({el/max(1,done):.1f}s/event)", flush=True)

    rep = {"n_candidate": len(keep), "n_usable": len(usable), "cached": done, "failed": fail,
           "n_layers": cap.n_layers, "pools": list(POOLS),
           "coverage_note": "覆盖预筛：需 t0 前 1.6 s 与后 6.4 s 都在 ego_pose 覆盖内"}
    (W / "results" / "alpa_g1_cache_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"[ALPA] 完成 {done}，失败 {len(fail)}")


if __name__ == "__main__":
    main()
