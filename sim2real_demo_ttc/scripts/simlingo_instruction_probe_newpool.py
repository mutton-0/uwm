"""SimLingo 指令服从/拒绝探针 —— **brake-first 新语料版**（13 事件 / 13 scene）。

数据源：`results/brake_first_pool_final.json`（另一会话的刹车优先挖矿产出，
先找人类减速片段、再归因到走廊内 VRU）。**本脚本只用这 13 个事件**，
不读取、不合并、不引用 `variants/n1_d2/mining/events_all.jsonl` 那批旧语料的任何数据。

方法与 `simlingo_instruction_probe.py` 完全一致（直接 import 复用其
prompt 组装、指令模板、拒绝理由分类、bootstrap），**只换数据源**。

--------------------------------------------------------------------------
新旧语料的三处结构差异，以及本脚本的处理（不糊弄，逐条写明）

**① 遮挡逻辑：改用 brake-first 的"整组遮挡"，不用旧探针的"只遮单个目标"。**
新语料每个事件带 `f3_mask_group` —— 走廊内**全部** VRU 的列表，设计上要一起遮。
旧探针只遮挖掘锁定的那一个实体。两者在 13 个事件里 **10 个 group 大小为 1（等价）**，
另 3 个为 2/2/3（新逻辑遮得更干净）。既然新语料的挖掘设计就是整组遮，
这里**服从新语料的设计**：occ 臂把 `f3_mask_group` 里每个框都涂掉。
投影用 `g1_mine_events.frame_bbox()`、涂色用 `f3_occlusion_necessity.occlude()`，
与 `brake_first_export.py` 出图时逐像素同一套。

**② ctrl 臂（必需对照）：逐框镜像。**
组里每个框各自找一个同面积、同离心率带的对照框，且**不与组内任何框、
也不与已放置的对照框重叠**。这样"被涂掉的总面积"与"每块的离心率"在 occ/ctrl 两臂可比，
只差涂的是不是那些 VRU。任一框找不到合法对照 ⇒ 整个事件跳过，不做单臂近似。

**③ clean 臂：新语料没有定义，由本脚本构造，须单独看待。**
新语料每个事件只有一帧（刹车起始帧 `frame_idx`），没有旧语料的 clean/ghost 双窗口。
这里取**同场景往前 1.5 s（10 Hz ⇒ 15 帧）**的那一帧作为 clean，
与旧语料 `clean_window_s` 的时间尺度对齐。**这是本脚本的构造，不是新语料的设计**，
且它同样有"VRU 那时多半已经可见"的老问题，故 clean 相关读数只作参考、不作主结论。
前导帧不足 1.5 s 的事件（`frame_idx < 15`）标记为 `clean_unavailable`，
其 clean 臂不参与统计，但 ghost/occ/ctrl/grey 照常跑。

**统计功效**：13 事件 / 13 scene，每个 scene 恰好 1 个事件 ⇒
scene 级 bootstrap 退化为事件级 bootstrap。CI 会非常宽，
故除 bootstrap 外**另报配对不一致计数**（n01/n10），在这个样本量下它比 CI 更可读。
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc")
RES = ROOT / "results"
NUSC = "/data/dataset/nuscenes/v1.0-trainval"
sys.path.insert(0, str(ROOT / "scripts"))

from simlingo_instruction_probe import (          # noqa: E402  方法复用，只换数据源
    HAZARD, REASONS, build_runner, classify, instructions)

CLEAN_LEAD_FRAMES = 15          # 10 Hz ⇒ 1.5 s，与旧语料 clean_window_s 的时间尺度对齐
ARMS = ("clean", "ghost", "occ", "ctrl", "grey")


def boot_scene(vals, scenes, n=10000, seed=0):
    """scene 级 bootstrap。新语料每 scene 恰好 1 事件，故等价于事件级重采样。"""
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        if v is not None and np.isfinite(v):
            by[s].append(v)
    keys = list(by)
    if len(keys) < 5:
        return None
    rng = np.random.default_rng(seed)
    o = [np.mean([x for i in rng.integers(0, len(keys), len(keys)) for x in by[keys[i]]])
         for _ in range(n)]
    return {"mean": float(np.mean([x for k in keys for x in by[k]])),
            "ci95": [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))],
            "n": int(sum(len(v) for v in by.values())), "n_scenes": len(keys)}


def paired_counts(a, b):
    """配对二值不一致计数：n10 = a=1,b=0；n01 = a=0,b=1。n=13 时比 CI 可读。"""
    n10 = int(sum(1 for x, y in zip(a, b) if x > y))
    n01 = int(sum(1 for x, y in zip(a, b) if x < y))
    return {"n10_only_first": n10, "n01_only_second": n01,
            "n_concordant": int(len(a) - n10 - n01)}


def control_boxes(img, boxes, rng, tries=400):
    """逐框找对照框：同面积、同离心率带（水平镜像优先），不与任何组内框/已放对照框重叠。"""
    H, W = img.shape[:2]
    placed, out = [b[:] for b in boxes], []
    for bb in boxes:
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        mx0 = W - x1
        cand = [(mx0, y0, mx0 + w, y0 + h)]
        for dy in (-2 * h, 2 * h, -3 * h, 3 * h, -4 * h, 4 * h):
            cand.append((x0, y0 + dy, x1, y1 + dy))
        for dx in (-2 * w, 2 * w, -3 * w, 3 * w):
            cand.append((x0 + dx, y0, x1 + dx, y1))
        got = None
        for c in cand:
            cx0, cy0, cx1, cy1 = c
            if cx0 < 0 or cy0 < 0 or cx1 > W or cy1 > H:
                continue
            if any(not (cx1 <= p[0] or cx0 >= p[2] or cy1 <= p[1] or cy0 >= p[3]) for p in placed):
                continue
            got = list(c); break
        if got is None:
            return None
        placed.append(got); out.append(got)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(RES / "brake_first_pool_final.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/n1_d2.yaml"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out", default=str(RES / "simlingo_instruction_probe_newpool.json"))
    args = ap.parse_args()

    import cv2
    from omegaconf import OmegaConf
    from nuscenes.nuscenes import NuScenes
    import g1_mine_events as G1
    from f3_occlusion_necessity import occlude
    from g2_cache import commanded_speed

    pool = json.load(open(args.pool))
    cands = pool["candidates"]
    print(f"[NP] brake-first 新语料：{pool['n_events']} 事件 / {pool['n_scenes']} scene；"
          f"corpus = {pool['corpus']}")
    assert len(cands) == 13, f"预期 13 个事件，实到 {len(cands)}"

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    nusc = NuScenes("v1.0-trainval", dataroot=NUSC, verbose=False)
    scenes = {s["name"]: s for s in nusc.scene}
    runner = build_runner(args.config, args.device)

    recs, skipped = [], defaultdict(int)
    for i, c in enumerate(cands):
        geo = G1.compute_scene_geometry(nusc, scenes[c["scene"]], cfg)
        j = int(c["frame_idx"])
        spd = float(geo["ego_speed"][j])          # 与 pool 的 ego_v0 一致，所有臂共用同一 prompt 速度

        def read(k):
            return cv2.cvtColor(cv2.imread(str(Path(NUSC) / geo["frames"][k]["filename"])),
                                cv2.COLOR_BGR2RGB)

        gh = read(j)
        # ① 遮挡组：走廊内全部 VRU（brake-first 设计）
        boxes, missing = [], []
        for g in c["f3_mask_group"]:
            ob = geo["per_obj"].get(g["token"])
            bb = G1.frame_bbox(geo, ob, j) if ob is not None else None
            (boxes if bb is not None else missing).append(list(bb) if bb is not None else g["token"])
        if not boxes:
            skipped["no_projectable_box"] += 1; continue
        cbs = control_boxes(gh, boxes, np.random.default_rng(1234 + i))
        if cbs is None:
            skipped["no_control_box"] += 1; continue

        occ, ctrl = gh.copy(), gh.copy()
        for bb in boxes:
            occ = occlude(occ, bb)
        for bb in cbs:
            ctrl = occlude(ctrl, bb)

        imgs = {"ghost": gh, "occ": occ, "ctrl": ctrl,
                "grey": np.full_like(gh, gh.reshape(-1, 3).mean(0).astype(gh.dtype))}
        # ③ clean：本脚本构造的 t-1.5s 帧；前导不足则整臂缺席
        clean_ok = j >= CLEAN_LEAD_FRAMES
        if clean_ok:
            imgs["clean"] = read(j - CLEAN_LEAD_FRAMES)

        # 画面里除遮挡组外还剩多少行人/车辆（用 geo 自身可见性，与投影同源）
        grp = {g["token"] for g in c["f3_mask_group"]}
        vis = [(t, o["cat"]) for t, o in geo["per_obj"].items()
               if t not in grp and bool(np.asarray(o["visible"])[j])]
        n_ped = sum(1 for _, cat in vis if cat.startswith("human.pedestrian"))
        n_veh = sum(1 for _, cat in vis if cat.startswith("vehicle."))

        rec = {"eid": f"{c['scene']}_f{j}", "scene": c["scene"], "frame_idx": j,
               "ego_speed": spd, "ego_v0_pool": c["ego_v0"], "a_obs": c["a_obs"],
               "ttc_s": c.get("ttc_s"), "lead_vru": c["lead_vru"],
               "n_mask_group": c["n_mask_group"], "n_boxes_projected": len(boxes),
               "unprojectable_tokens": missing,
               "mask_boxes": [[round(v, 1) for v in b] for b in boxes],
               "control_boxes": [[round(v, 1) for v in b] for b in cbs],
               "clean_available": clean_ok,
               "clean_frame_idx": (j - CLEAN_LEAD_FRAMES) if clean_ok else None,
               "other_peds_in_view": n_ped, "vehicles_in_view": n_veh,
               "qa_note": c["qa"]["note"], "arms": {}}

        for frame, img in imgs.items():
            runner.instruction = ""
            r0 = runner.infer(img, spd, pool_modes=())
            v0 = float(commanded_speed(r0.waypoints))
            rec["arms"][f"{frame}/baseline"] = {
                "v_plan": v0, "lang": r0.language, "reason": classify(r0.language),
                "prompt": r0.prompt}
            for name, direction, text in instructions(spd):
                runner.instruction = text
                r = runner.infer(img, spd, pool_modes=())
                v = float(commanded_speed(r.waypoints))
                lang = r.language or ""
                rec["arms"][f"{frame}/{name}"] = {
                    "v_plan": v, "dv_vs_baseline": v - v0, "direction": direction,
                    "instruction": text, "lang": lang, "reason": classify(lang),
                    "hazard_reject": bool(classify(lang) in HAZARD),
                    "pedestrian_reject": bool(classify(lang) == "hazard_pedestrian"),
                    "traj_complies": bool((v - v0) > 0) if direction == "up" else bool((v - v0) < 0),
                    "prompt": r.prompt}
        runner.instruction = ""
        recs.append(rec)
        print(f"[NP] {i+1}/{len(cands)} {rec['eid']:18s} group {len(boxes)}/{c['n_mask_group']}"
              f"  其它行人 {n_ped} 车辆 {n_veh}  clean {'有' if clean_ok else '**无**'}", flush=True)

    out = {"corpus": pool["corpus"], "source_pool": Path(args.pool).name,
           "design": pool["design"], "merge_with_old_pool": False,
           "occlusion_logic": "brake-first f3_mask_group 整组遮挡（走廊内全部 VRU）",
           "clean_arm": f"本脚本构造：同场景 frame_idx-{CLEAN_LEAD_FRAMES}（-1.5s）；新语料未定义 clean",
           "prompt_format": "Current speed: {v} m/s. Target waypoint: <TP><TP>. {instruction}",
           "prompt_source": "dataset_dreamer.py:129",
           "n_events": len(recs), "skipped": dict(skipped), "per_event": recs}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[NP] wrote {args.out}  （n={len(recs)}，skipped={dict(skipped)}）")


if __name__ == "__main__":
    main()
