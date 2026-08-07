"""T1-Q｜寻轴 v2:$v_{brake}$ 行为定义参照轴 + 夹角矩阵（guide §T1-Q）。

战略前提（guide 写死的）：行为对危险有微弱但显著的梯度（A 类 b-AUC 0.555、S1 单调 p=0.031）
=> 行为不是随机的 => **必然存在某个内部中介状态**，问题只是它住在哪个位置。
所以先造一条"按构造必然存在"的轴当**站内上界标定**：

  $v_{brake}$ = 用模型**自己的刹/不刹**当标签，在 driving query 位置上提的判别式方向。

它有三个用途：
  ① steering 它应当**平凡通过** —— 若连它都推不动行为，那是仪器/注入机制有病，
     而不是"危险表征不存在"。这是 T2 FAIL 的关键分流器；
  ② 关键测试：**ghost 帧激活在 $v_{brake}$ 上的投影是否高于 clean** ——
     视觉危险是否在亚阈值地把模型往刹车方向推（行为端看不出，表征端也许看得出）；
  ③ 夹角矩阵 cos($v_{danger}^{lang}$, $v_{brake}$, 视觉判别式方向)：
     语言概念轴与行为轴近正交 = "概念→动作断"的几何证据。

纪律：位置×层×来源×池化是大网格，S_dir 提方向 / S_sel 选层 / S_test 报数三分沿用，
主格预注册 = query_mean × $v_{brake}$ 投影差，S_test 只报一次。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_metrics import load_cache  # noqa: E402
from n1_readout import auc, supervised_direction  # noqa: E402


def brake_direction(events, seed=0):
    """标签 = 模型自身刹不刹。

    b = v_plan(clean) − v_plan(ghost) 是**配对差**，直接拿它当标签、再用 δ 当特征，
    等于用同一对活动预测自己，必然自证。因此这里：
      特征 = 逐条件的 query 激活（clean 与 ghost 各算一条样本）；
      标签 = 该条件下 pred_speed 对 ego_speed 回归后的残差是否低于中位数
             （低残差 = 相对该车速"刹得多"）。
    去掉 ego_speed 主效应是必须的：prompt 里明写 Current speed，
    不扣掉的话提到的是"车速轴"而不是"刹车轴"。
    """
    from sklearn.linear_model import LogisticRegression
    X, y_raw, ego = [], [], []
    for e in events:
        for cond in ("clean", "ghost"):
            X.append(e[f"h_{cond}"])
            y_raw.append(e[f"v_{cond}"])
            ego.append(e["meta"].get("ego_speed_mps", np.nan))
    X = np.stack(X)                                   # [N, L, C]
    y_raw = np.asarray(y_raw, float)
    ego = np.asarray(ego, float)
    ok = np.isfinite(ego)
    if ok.sum() > 10:
        A = np.vstack([ego[ok], np.ones(ok.sum())]).T
        coef, *_ = np.linalg.lstsq(A, y_raw[ok], rcond=None)
        resid = y_raw - (coef[0] * np.nan_to_num(ego, nan=np.nanmean(ego)) + coef[1])
    else:
        resid = y_raw - y_raw.mean()
    y = (resid < np.median(resid)).astype(int)        # 1 = 刹得多
    n, L, C = X.shape
    v = np.zeros((L, C), dtype=np.float32)
    for l in range(L):
        Z = X[:, l, :]
        mu, sd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit((Z - mu) / sd, y)
        w = clf.coef_[0] / sd[0]
        v[l] = w / (np.linalg.norm(w) + 1e-8)
    return v, y


def boot_ci(vals, scenes, n_boot=2000, seed=0):
    vals = np.asarray(vals, float)
    ok = np.isfinite(vals)
    vals, scenes = vals[ok], np.asarray(scenes)[ok]
    if len(vals) < 5:
        return float("nan"), (float("nan"), float("nan"))
    by = defaultdict(list)
    for v, s in zip(vals, scenes):
        by[s].append(v)
    keys = list(by)
    rng = np.random.default_rng(seed)
    st = [np.mean([v for i in rng.choice(len(keys), len(keys), replace=True) for v in by[keys[i]]])
          for _ in range(n_boot)]
    return float(vals.mean()), (float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pool-mode", default="query_mean")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang-axis", default="", help="Step 1 的 v_danger_lang_*.npy（算夹角用）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    work = Path(cfg["paths"]["work_dir"])
    evmap = {json.loads(l)["event_id"]: json.loads(l) for l in open(work / "mining" / "events_all.jsonl")}
    items = load_cache(work, args.pool_mode, keep=set(evmap))
    matched = {}
    for t in ("D2a",):
        f = work / "mining" / f"matched_{t}.txt"
        if f.exists():
            matched[t] = set(f.read_text().split())

    by_type = defaultdict(list)
    for eid, e in items.items():
        t = evmap[eid]["event_type"]
        if t in matched and eid not in matched[t]:
            continue
        by_type[t].append(e)

    scenes = sorted({e["scene"] for v_ in by_type.values() for e in v_})
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(scenes))
    n_dir, n_sel = int(len(scenes) * 0.5), int(len(scenes) * 0.25)
    part = {"dir": set(), "sel": set(), "test": set()}
    for i, k in enumerate(perm):
        part["dir" if i < n_dir else ("sel" if i < n_dir + n_sel else "test")].add(scenes[k])

    def grp(t, p):
        return [e for e in by_type[t] if e["scene"] in part[p]]

    allev = lambda p: [e for t in by_type for e in grp(t, p)]           # noqa: E731

    # ---- v_brake：S_dir 上提，S_sel 上选层 ----
    v_br, _ = brake_direction(allev("dir"), seed=args.seed)
    sel = allev("sel")
    sel_auc = []
    for l in range(v_br.shape[0]):
        lo = [float(e["h_clean"][l] @ v_br[l]) for e in sel]
        hi = [float(e["h_ghost"][l] @ v_br[l]) for e in sel]
        # 选层准则：该层能否把"刹得多/少"分开 —— 用 clean/ghost 内部的 pred_speed 排序代理
        y = np.array([e["v_clean"] for e in sel] + [e["v_ghost"] for e in sel])
        x = np.array(lo + hi)
        r = stats.spearmanr(x, y).correlation
        sel_auc.append(abs(r) if np.isfinite(r) else np.nan)
    Lb = int(np.nanargmax(sel_auc))
    print(f"[T1-Q] v_brake 提出（{args.pool_mode}）  选层 L*={Lb}  "
          f"S_sel |ρ(投影, v_plan)|={sel_auc[Lb]:.3f}")

    # ---- 关键测试：ghost 激活是否比 clean 更靠刹车方向 ----
    test = grp("A", "test")
    d = np.array([float((e["h_ghost"][Lb] - e["h_clean"][Lb]) @ v_br[Lb]) for e in test])
    sc = [e["scene"] for e in test]
    m, ci = boot_ci(d, sc)
    print(f"\n[关键测试] A 类 ghost−clean 在 v_brake 上的投影差 = {m:+.4f}  "
          f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  n={len(d)}")
    print("           " + ("✅ 视觉危险**亚阈值地**把模型推向刹车方向" if ci[0] > 0 else
                           ("⚠️ 显著推向油门方向" if ci[1] < 0 else
                            "❌ 无推动 —— 视觉危险在行为中介态上也读不到")))

    # 负类对照：D2a 应当更弱
    d2 = grp("D2a", "test")
    if len(d2) >= 20:
        dn = np.array([float((e["h_ghost"][Lb] - e["h_clean"][Lb]) @ v_br[Lb]) for e in d2])
        a_, p_ = auc(d, dn)
        mn, cin = boot_ci(dn, [e["scene"] for e in d2])
        print(f"[负类对照] D2a 投影差 = {mn:+.4f} [{cin[0]:+.4f}, {cin[1]:+.4f}]  "
              f"AUC(A vs D2a)={a_:.3f}  p={p_:.3g}")

    # ---- 夹角矩阵 ----
    axes = {"v_brake": v_br}
    if args.lang_axis and Path(args.lang_axis).exists():
        axes["v_danger_lang"] = np.load(args.lang_axis).astype(np.float32)
    vis = supervised_direction(grp("A", "dir"), grp("D2a", "dir"), seed=args.seed)
    axes["v_visual_disc"] = vis
    names = list(axes)
    print(f"\n[夹角矩阵] cos @ L={Lb}")
    cosm = {}
    for i, a in enumerate(names):
        for bnm in names[i + 1:]:
            x, y = axes[a][Lb], axes[bnm][Lb]
            c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
            cosm[f"{a}|{bnm}"] = c
            note = "近正交" if abs(c) < 0.15 else ("同向" if c > 0.5 else "弱相关")
            print(f"           cos({a}, {bnm}) = {c:+.3f}   {note}")

    out = {"pool_mode": args.pool_mode, "brake_layer": Lb,
           "sel_rho_by_layer": [float(x) for x in sel_auc],
           "projdiff_mean": m, "projdiff_ci95": list(ci), "n_test": len(d),
           "cos_matrix": cosm,
           "verdict": ("亚阈值推动" if ci[0] > 0 else
                       ("反向" if ci[1] < 0 else "无推动"))}
    np.save(work / "results" / f"v_brake_{args.pool_mode}{args.tag}.npy", v_br)
    (work / "results" / f"t1q_axes_{args.pool_mode}{args.tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[T1-Q] wrote results/t1q_axes_{args.pool_mode}{args.tag}.json + v_brake_*.npy")


if __name__ == "__main__":
    main()
