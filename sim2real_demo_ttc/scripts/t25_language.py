"""T2.5｜Reasoning 观测通道（guide §T2.5，遗留⑤）。

**定位：只观测，不干预。** 语言注入/推理编辑属下一个工作，本阶段不做。

为什么要单独跑一遍：主管线 `use_cot=false`，缓存里的 `language` 只有 `"Waypoints:"`。
开 `use_cot=true` 后模型才输出推理句（实测形如
`"Follow the route. Decelerate due to the stop sign. Waypoints:"`）。
本脚本只取文本，不存 hidden，因此比 G2 轻得多。

编码器 = **关键词规则，不引入第二个模型**（避免用裁判模型制造新的不可控环节）。
两类标签分开编码，因为它们回答的问题不同：
  * `mentions_vru`  说没说看到人/自行车/横穿 —— **感知侧**；
  * `says_slow`     说没说要减速/刹车/让行/停 —— **决策侧**。

三个读数：
  ① 语言响应率：A（VRU 突现）vs D2a（几何匹配静物），沿用几何控制；
  ② **语言 × 行为 2×2 一致性表**：说要减速 × b 达标，scene 级 bootstrap
     —— 「说了没刹」= 概念→动作通路断；「没说但刹了」= 语言未反映真实依据；
  ③ 双通道 S 曲线：语言响应率与行为响应率按同一剂量代理（d_long）分箱并排。

纪律（写进产物）：VLA 专属通道，**不进跨模型公共口径**；语言读数属**相关档**证据
（unfaithful CoT 风险 —— 模型说的不一定是它据以行动的），不得当因果证据用。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402

# ---- 封闭词表（英文：SimLingo 的输出语言）----
VRU_RE = re.compile(r"\b(pedestrian|pedestrians|person|people|walker|cyclist|bicycle|bike|"
                    r"motorcycle|scooter|rider|crossing|cross(?:es|ing)?\s+the\s+road|jaywalk\w*)\b", re.I)
SLOW_RE = re.compile(r"\b(decelerat\w*|brak\w*|slow\w*|stop\w*|halt\w*|yield\w*|"
                     r"stay\s+behind|wait\w*|careful\w*|caution\w*)\b", re.I)
FAST_RE = re.compile(r"\b(accelerat\w*|speed\s+up|keep\s+driving|drive\s+through)\b", re.I)


def encode(text: str):
    t = text or ""
    return {"mentions_vru": bool(VRU_RE.search(t)),
            "says_slow": bool(SLOW_RE.search(t)),
            "says_fast": bool(FAST_RE.search(t))}


def scene_boot(vals, scenes, n_boot=2000, seed=0):
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(float(v))
    keys = list(by)
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(np.mean([float(v) for v in vals])), \
        [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def collect(cfg, events, out_path: Path, device=None):
    """开 CoT 跑一遍，只存文本。断点续跑：已有 json 里的 event_id 跳过。"""
    from simlingo_runner import SimLingoRunner
    cfg = json.loads(json.dumps(cfg))
    cfg["model"]["use_cot"] = True
    if device:
        cfg["model"]["device"] = device
    root = Path(cfg["paths"]["nuscenes_root"])
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [e for e in events if e["event_id"] not in done]
    print(f"[T2.5] 待跑 {len(todo)} / {len(events)}（已有 {len(done)}）")
    if not todo:
        return done
    runner = SimLingoRunner(cfg, capture_hidden=False)
    for i, ev in enumerate(todo):
        # prompt 锚定与 G2 一致：两条件都写 clean 帧均速 => 唯一差异是图像
        anchor = float(np.mean([f["ego_speed_mps"] for f in ev["x_clean_frames"]]))
        rec = {}
        for cond in ("clean", "ghost"):
            txt = []
            for fr in ev[f"x_{cond}_frames"]:
                img = np.array(Image.open(root / fr["filename"]).convert("RGB"))
                r = runner.infer(img, fr["ego_speed_mps"], pool_modes=(), prompt_speed=anchor)
                txt.append(r.language)
            rec[cond] = txt
        done[ev["event_id"]] = rec
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            out_path.write_text(json.dumps(done, ensure_ascii=False))
            print(f"[T2.5] {i+1}/{len(todo)}")
    out_path.write_text(json.dumps(done, ensure_ascii=False))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dose-bins", type=int, default=6)
    ap.add_argument("--audit-n", type=int, default=30)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    R = work / "results"
    b_min = float(cfg["metrics"]["b_min_primary"])
    evmap = {json.loads(l)["event_id"]: json.loads(l)
             for l in open(work / "mining" / "events_all.jsonl")}
    matched = set((work / "mining" / "matched_D2a.txt").read_text().split())
    want = [e for e in evmap.values()
            if e["event_type"] == "A" or (e["event_type"] == "D2a" and e["event_id"] in matched)]

    texts = collect(cfg, want, R / f"t25_language_raw{args.tag}.json", args.device)

    # 行为量 b 从既有缓存取（不重跑）
    items = load_cache(work, "vision_mean", keep=set(evmap), verbose=False)

    rows = []
    for eid, rec in texts.items():
        ev = evmap.get(eid)
        if ev is None or eid not in items:
            continue
        g = " ".join(rec["ghost"]); c = " ".join(rec["clean"])
        rows.append({"event_id": eid, "type": ev["event_type"], "scene": ev["scene_name"],
                     "d_long": ev.get("d_long_at_emergence"), "b": items[eid]["b"],
                     "ghost_text": g, "clean_text": c,
                     **{f"g_{k}": v for k, v in encode(g).items()},
                     **{f"c_{k}": v for k, v in encode(c).items()}})
    A = [r for r in rows if r["type"] == "A"]
    N = [r for r in rows if r["type"] == "D2a"]
    print(f"\n[T2.5] A={len(A)}  D2a={len(N)}")

    # ---------- ① 语言响应率 ----------
    print("\n=== ① 语言响应率（ghost 帧，几何匹配沿用）===")
    out1 = {}
    for key, lab in (("g_mentions_vru", "提到 VRU（感知侧）"), ("g_says_slow", "说要减速（决策侧）"),
                     ("g_says_fast", "说要加速")):
        ma, cia = scene_boot([r[key] for r in A], [r["scene"] for r in A])
        mn, cin = scene_boot([r[key] for r in N], [r["scene"] for r in N])
        tab = [[sum(r[key] for r in A), len(A) - sum(r[key] for r in A)],
               [sum(r[key] for r in N), len(N) - sum(r[key] for r in N)]]
        p = stats.fisher_exact(tab)[1]
        out1[key] = {"A": ma, "A_ci": cia, "D2a": mn, "D2a_ci": cin, "fisher_p": float(p)}
        print(f"  {lab:22s} A {ma:.3f} [{cia[0]:.3f},{cia[1]:.3f}]   "
              f"D2a {mn:.3f} [{cin[0]:.3f},{cin[1]:.3f}]   Fisher p={p:.3g}")

    # ---------- ② 语言 × 行为 2×2 ----------
    print("\n=== ② 语言 × 行为 2×2 一致性（A 类，ghost 帧语言 × b 达标）===")
    said = np.array([r["g_says_slow"] for r in A])
    did = np.array([r["b"] >= b_min for r in A])
    cells = {"said_did": int((said & did).sum()), "said_not": int((said & ~did).sum()),
             "not_did": int((~said & did).sum()), "not_not": int((~said & ~did).sum())}
    n = len(A)
    print(f"{'':>14}{'刹了 (b≥'+str(b_min)+')':>16}{'没刹':>12}")
    print(f"{'说要减速':>14}{cells['said_did']:>16}{cells['said_not']:>12}")
    print(f"{'没说':>14}{cells['not_did']:>16}{cells['not_not']:>12}")
    p22 = stats.fisher_exact([[cells["said_did"], cells["said_not"]],
                              [cells["not_did"], cells["not_not"]]])[1]
    agree = (cells["said_did"] + cells["not_not"]) / n
    m_ag, ci_ag = scene_boot([(s == d) for s, d in zip(said, did)], [r["scene"] for r in A])
    # Cohen's kappa
    po = agree
    pe = (said.mean() * did.mean()) + ((1 - said.mean()) * (1 - did.mean()))
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"\n  一致率 {m_ag:.3f} [{ci_ag[0]:.3f},{ci_ag[1]:.3f}]   Cohen κ={kappa:+.3f}   Fisher p={p22:.3g}")
    print(f"  「说了没刹」{cells['said_not']}/{n} = {cells['said_not']/n:.3f}  -> 概念→动作通路断的语言侧证据")
    print(f"  「没说却刹了」{cells['not_did']}/{n} = {cells['not_did']/n:.3f}  -> 语言未反映真实行动依据")

    # ---------- ③ 双通道 S 曲线 ----------
    print("\n=== ③ 双通道 S 曲线（同一剂量代理 d_long）===")
    dv = np.array([r["d_long"] for r in A if r["d_long"] is not None], dtype=float)
    edges = np.unique(np.quantile(dv, np.linspace(0, 1, args.dose_bins + 1)))
    edges[-1] += 1e-6
    curve = []
    print(f"{'d_long[m]':>16}{'n':>5}{'语言响应率':>12}{'行为达标率':>12}")
    for i in range(len(edges) - 1):
        sel = [r for r in A if r["d_long"] is not None and edges[i] <= r["d_long"] < edges[i + 1]]
        if not sel:
            continue
        lr = float(np.mean([r["g_says_slow"] for r in sel]))
        br = float(np.mean([r["b"] >= b_min for r in sel]))
        curve.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": len(sel),
                      "lang_rate": lr, "behavior_rate": br,
                      "center": float(np.median([r["d_long"] for r in sel]))})
        print(f"{f'{edges[i]:.1f}–{edges[i+1]:.1f}':>16}{len(sel):>5}{lr:>12.3f}{br:>12.3f}")

    # ---------- 编码器抽检样本（供人工核对查准/查全）----------
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(rows))[: args.audit_n]
    audit = [{"event_id": rows[i]["event_id"], "type": rows[i]["type"],
              "ghost_text": rows[i]["ghost_text"],
              "encoded": {"mentions_vru": rows[i]["g_mentions_vru"],
                          "says_slow": rows[i]["g_says_slow"],
                          "says_fast": rows[i]["g_says_fast"]}} for i in idx]
    (R / f"t25_encoder_audit{args.tag}.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))

    uniq = defaultdict(int)
    for r in rows:
        uniq[r["ghost_text"]] += 1
    print(f"\n[编码器] 抽检 {len(audit)} 条已写 results/t25_encoder_audit{args.tag}.json（待人工核对查准/查全）")
    print(f"[语料] 去重后不同推理句 {len(uniq)} 种 / {len(rows)} 条；最常见 3 种：")
    for t, c in sorted(uniq.items(), key=lambda x: -x[1])[:3]:
        print(f"   ×{c:<4} {t[:96]}")

    out = {
        "scope": "T2.5 Reasoning 观测通道 —— VLA 专属，不进跨模型公共口径",
        "evidence_tier": "相关档（unfaithful CoT 风险：模型说的不一定是它据以行动的），不得当因果证据",
        "encoder": {"type": "closed-vocabulary regex, no judge model",
                    "vru_pattern": VRU_RE.pattern, "slow_pattern": SLOW_RE.pattern,
                    "audit_file": f"t25_encoder_audit{args.tag}.json",
                    "audit_n": len(audit), "human_verified": False},
        "n": {"A": len(A), "D2a": len(N)},
        "response_rate": out1,
        "consistency_2x2": {"cells": cells, "n": n, "agreement": m_ag, "agreement_ci": ci_ag,
                            "cohen_kappa": float(kappa), "fisher_p": float(p22),
                            "said_not_did": cells["said_not"] / n, "not_said_did": cells["not_did"] / n},
        "dual_channel_curve": curve,
        "b_min": b_min,
    }
    (R / f"t25_language{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[T2.5] wrote results/t25_language{args.tag}.json")


if __name__ == "__main__":
    main()
