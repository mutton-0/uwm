"""敏感性网格汇总：扫描一个 work_dir 下所有 metrics_estimate_*.json，按主/次读数列表。

预注册纪律（results/tier_m_preregistration.md）：主读数只有一个，其余全部是敏感性分析，
**不得因为某个次要格子更好看就替换主结论**。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


def fmt(x, n=3):
    return "n/a" if x is None or x != x else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="sensitivity_grid.md")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    res = Path(cfg["paths"]["work_dir"]) / "results"
    mcfg = cfg["metrics"]
    primary = (mcfg.get("direction_method", "pca"), mcfg.get("deconfound", "none"),
               mcfg.get("primary_pool_mode", "vision_mean"))

    rows = []
    for p in sorted(res.glob("metrics_estimate_*.json")):
        m = json.loads(p.read_text())
        key = (m.get("direction_method", "pca"), m.get("deconfound", "none"), m["pool_mode"])
        rows.append((key, m, p.name))
    # 同一组合可能既有带 tag 也有不带 tag 的文件，去重（保留文件名最短的 = 主读数那份）
    seen, uniq = {}, []
    for key, m, name in rows:
        if key not in seen or len(name) < len(seen[key][2]):
            seen[key] = (key, m, name)
    uniq = sorted(seen.values(), key=lambda r: (r[0] != primary, r[0]))

    lines = [
        f"# Tier-{uniq[0][1]['tier']} 敏感性网格",
        "",
        "> 主读数（★）在看到任何本档结果之前已冻结，见 `results/tier_m_preregistration.md`。",
        "> 其余行是敏感性分析。**任何一行更好看都不构成替换主结论的理由。**",
        "",
        "所有读数取自 **truth holdout**（该集合从未参与提方向、选层、调参）与 **S_test**（手册指定主判定集）。",
        "",
        "| | 方向法 | 解混淆 | 池化 | L* | S_test AUC(正例/D) | truth AUC(正例/D) | p | truth ρ_TTC | p(置换) | truth ρ(投影,行为) | p |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m, _ in uniq:
        ro_t, ro_h = m["readouts"]["S_test"], m["readouts"]["truth_holdout"]
        star = "★" if key == primary else ""
        lines.append(
            f"| {star} | {key[0]} | {key[1]} | {key[2]} | {m['probe']['peak_layer']} | "
            f"{fmt(ro_t['auc_positive_vs_D'])} | {fmt(ro_h['auc_positive_vs_D'])} | "
            f"{fmt(ro_h['p_positive_vs_D'])} | {fmt(ro_h['rho_ttc'])} | {fmt(ro_h['p_permutation'])} | "
            f"{fmt(ro_h['rho_projection_behavior'])} | {fmt(ro_h['p_projection_behavior'])} |")

    n_h = uniq[0][1]["readouts"]["truth_holdout"]["n"]
    sig = [k for k, m, _ in uniq
           if m["readouts"]["truth_holdout"]["p_positive_vs_D"] < 0.05
           and m["readouts"]["truth_holdout"]["auc_positive_vs_D"] > 0.5]
    lines += [
        "",
        "## 判读",
        "",
        f"- truth holdout 有 **{n_h}** 个事件，AUC 的标准误约 ±0.015，"
        "因此 0.5 附近的偏离在这个样本量下是**可以判定为“没有效应”**的，而不是“功效不足”。",
        f"- {len(sig)}/{len(uniq)} 个格子在 truth 上给出显著且方向正确的 AUC(正例 vs D)"
        + ("。" if not sig else f"：{sig}。"),
        "- 与 Tier-S 的对照见 `../../results/deconfound_ablation.md`。",
        "",
    ]
    out = res / args.out
    out.write_text("\n".join(lines))
    print("\n".join(lines[7:9 + len(uniq)]))
    print(f"\n[sensitivity] wrote {out}")


if __name__ == "__main__":
    main()
