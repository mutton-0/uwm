"""把**被归因否掉**的场景事件画出来：所有危险类目标 + 它们到自车未来路径的横向距离。

用途：回答「按现在的原则到底能不能选出有效事件」。
绿框 = 落在走廊内（可被归因，标 a_req）；红框 = 走廊外（公式看不见它）。
"""
from __future__ import annotations
import json, sys, pickle
from pathlib import Path
from collections import defaultdict

import cv2, numpy as np
from omegaconf import OmegaConf

ROOT = Path("/data/ruolin/uwm/sim2real_demo_ttc"); RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))
import g1_mine_events as G1                                             # noqa: E402
import ns1_navsim_geometry as NS                                        # noqa: E402
from brake_first_miner import (point_to_polyline, D_SAFE, SCENARIO_HAZARD,
                               STATIC_PREFIX, STATIC_CORRIDOR)          # noqa: E402

NS_BLOBS = "/data/dataset/navsim/dataset/sensor_blobs"
# OpenCV 的 Hershey 字体画不了 CJK，图上一律用英文
REASON_EN = {
    "走廊内没有任何危险类目标(a>0)": "no hazard-class object inside the corridor (a>0)",
    "危险类不占主导(share<=0.6 或 无1.5倍优势)": "hazard class not dominant (share<=0.6 or no 1.5x margin)",
    "解释度越界(explained_ratio 不在 (0.3,10])": "explained_ratio outside (0.3, 10]",
    "归因目标投影不到图像上(无法遮挡)": "attributed target not projectable (cannot be masked)",
    "太远且不紧迫(TTC>5s 且 距离>40m)": "too far and not urgent (TTC>5s and range>40m)",
    "归因需求太弱(a_req<0.4)": "attribution demand too weak (a_req<0.4)",
}
CORRIDOR = 2.0


def main():
    rej_file, scenario, outdir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    HAZ = SCENARIO_HAZARD[scenario]
    G1.set_include_animal(True)
    d = json.load(open(RES / rej_file))
    R = [r for r in d["rejects"]]
    # 分层：每个拒绝原因都取到
    by = defaultdict(list)
    for r in R:
        by[r["reason"]].append(r)
    pick, i = [], 0
    while len(pick) < n and any(by.values()):
        for k in list(by):
            if by[k] and len(pick) < n:
                pick.append(by[k].pop(len(by[k]) // 2))
        i += 1
        if i > 50:
            break

    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/navsim_corpus.yaml"), resolve=True)
    cache = defaultdict(list)
    for lf in sorted((NS.NS_ROOT / "navsim_logs" / "test").glob("*.pkl")):
        for f in pickle.load(open(lf, "rb")):
            cache[f["scene_name"]].append(f)
    for kk in cache:
        cache[kk].sort(key=lambda z: z["timestamp"])

    OUT = RES / "figures" / outdir; OUT.mkdir(parents=True, exist_ok=True)
    meta = []
    for r in pick:
        fl = cache[r["scene"]]
        geo = NS.build_geo(fl, cfg, "test")
        j = r["frame_idx"]; gt = geo["grid_t"]; exyz = geo["ego_xyz"]
        es = geo["ego_speed"]; v0 = float(es[j])
        # 复建自车未来路径（与挖矿同法）：走到速度谷底 + 沿末帧朝向外推 20m
        k = int(np.argmin(es[j:]) + j)
        k = max(k, min(j + 1, len(gt) - 1))
        Rm = geo["R_we"][j]
        poly = (exyz[j:k + 1, :2] - exyz[j, :2]) @ Rm[:2, :2]
        fwd = geo["R_we"][k][:2, 0] @ Rm[:2, :2]; fwd /= (np.linalg.norm(fwd) + 1e-12)
        poly = np.vstack([poly, poly[-1] + np.outer(np.linspace(0, 20, 20)[1:], fwd)])

        fr = geo["frames"][j]
        img = cv2.cvtColor(cv2.imread(str(Path(NS_BLOBS) / fr["filename"])), cv2.COLOR_BGR2RGB)
        rows, n_in, n_out = [], 0, 0
        for tok, o in geo["per_obj"].items():
            if not bool(o["valid"][j]) or not o["cat"].startswith(HAZ):
                continue
            p = np.asarray(o["p_ego"][j], float)[:2]
            dmin, seg, tang = point_to_polyline(p, poly)
            lim = STATIC_CORRIDOR if o["cat"].startswith(STATIC_PREFIX) else CORRIDOR
            s = (float(np.linalg.norm(np.diff(poly[:seg + 1], axis=0), axis=1).sum())
                 if seg > 0 else 0.0)
            va = float(np.asarray(o["v_obj_ego"][j], float)[:2] @ tang)
            a = (max(v0 - va, 0.0) ** 2 / (2.0 * max(s - D_SAFE, 0.5))) if dmin < lim else 0.0
            if np.linalg.norm(p) > 60:
                continue
            bb = G1.frame_bbox(geo, o, j)
            inside = dmin < lim
            n_in += inside; n_out += (not inside)
            rows.append((bb, o["cat"], dmin, s, a, inside))
        vis = img.copy()
        for bb, cat, dmin, s, a, inside in rows:
            if bb is None:
                continue
            x0, y0, x1, y1 = [int(round(v)) for v in bb]
            col = (0, 235, 0) if inside else (255, 60, 60)
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 3)
            txt = (f"{s:.0f}m lat{dmin:.1f} a={a:.2f}" if inside
                   else f"{s:.0f}m lat{dmin:.1f} OUTSIDE")
            cv2.putText(vis, txt, (x0, max(16, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        bar = np.zeros((160, vis.shape[1], 3), np.uint8)
        for i2, t in enumerate([
                f"{r['scene']} frame {j}   [{scenario}]  REJECTED",
                f"REASON: {REASON_EN.get(r['reason'], r['reason'])}",
                f"human: v0 {r['ego_v0']} -> {r['ego_vmin']} m/s   a_obs {r['a_obs']} m/s2",
                f"hazard-class objects within 60m: {len(rows)}"
                f"   IN corridor(green) {n_in}   OUTSIDE(red) {n_out}"]):
            cv2.putText(bar, t, (12, 32 + 36 * i2), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                        (255, 255, 255) if i2 == 0 else
                        ((120, 200, 255) if i2 == 1 else (180, 230, 255)), 2)
        p_ = OUT / f"{r['scene']}_f{j}.jpg"
        cv2.imwrite(str(p_), cv2.cvtColor(np.vstack([bar, vis]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 84])
        meta.append({**r, "file": p_.name, "n_hazard_in": int(n_in), "n_hazard_out": int(n_out)})
        print(f"  {r['scene']:22s} f{j:<3d} {r['reason'][:28]:30s} 走廊内 {n_in} 外 {n_out}")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
