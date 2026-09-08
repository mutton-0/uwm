# 因果迁移实验 —— 远程机跑法

## 要回答的问题

在**左舵（benchmark）**上学到的方向，注入进模型能不能引发减速？
同一根轴注入**右舵（deployment）**场景，还灵不灵？

## 判定逻辑（顺序不可颠倒）

```
第 1 步  左舵注入必须先有效
         └─ 无效 ⇒ 方法未建立，实验到此为止。
                  右舵的阴性结果**不可解读** —— 分不清"轴不驱动行为"和"方法测不出驱动"

第 2 步  左舵有效后再看右舵
         ├─ 右舵也有效 ⇒ **因果不变**（invariance）
         └─ 右舵无效   ⇒ **因果偏移**（causal shift）  ← 本实验要找的
```

## 跑法

```bash
cd <repo>/sim2real_demo_ttc
PY=/path/to/python DEV=cuda:0 N=60 NR=30 bash remote_pkg/run_causal_transfer.sh
```

`N` = 每格刺激场景数，`NR` = 随机方向对照数（p 值分辨率 = 1/NR，30 → 0.033）。
四步跑完约 `3 模型 × 6 格 × N × 9 档 × (NR+1)` 次前向。单卡独占时 N=60/NR=30 约 1–2 小时。

## 前置数据（必须已在 `results/`）

| 文件 | 来源 |
|---|---|
| `vdecel_acts_navsim_test_{dd,ltf,ddv2}.npz` | `scripts/f_vdecel_extract.py` |
| `vbright_acts_navsim_lead_{m}_night_global.npz` | `scripts/i_bright_extract.py` |
| `cruise_pool_navsim_{test,trainval}.json` | `scripts/cruise_miner.py` |
| `driveside_map.json` | `scripts/driveside_map.py` |

图像需在 `/data/dataset/navsim/dataset/sensor_blobs`（脚本会自动跳过不在盘上的场景）。

## 三根轴

| `--axis` | 方向 | 角色 |
|---|---|---|
| `speed` | $-\hat s$，巡航样本对 $v_0$ 回归 | **阳性对照**。速度是显式输入，它都推不动则方法不通 |
| `decel` | $v_{\text{decel}}$ = 刹车组 − 巡航组 | F 轴的动作方向 |
| `bright` | $v_{\text{bright}}$ = 原图 → 变夜的表征差 | **I 轴**。预期：注入它应使模型**加速** |

**三者的方向与层权重一律只在左舵学**；`--stim-side` 只换被注入的场景。

## 判据（每格四条）

| | 判据 |
|---|---|
| C1 | 斜率为负，且在 `NR` 个随机方向的零分布里 p < 0.05 |
| C2 | 单调：α>0 段与 α<0 段斜率同号，比值 ∈ [1/3, 3] |
| C3 | 特异：横向偏移的斜率 ≤ 速度斜率的 1/3 |
| C4 | 剂量：α ≤ 8 时已出现一半以上效应（防"把模型推坏"被当成方向效应） |

## 已知的坑

1. **中间层注入会被下游归一化吃掉**：本机实测 DD 在 L5 注入 64%，传到 L6 只剩 2.3%。
   脚本按 RepE 口径用"读取性能"选层（`PR≥3` 且 `|Spearman(投影, v0)| ≥ 0.3`），
   DD 选出 L2,L4,L5,L6,L7。
2. **L7 可能是死层**：DD 的 L7 因果可达性精确为 0（轨迹头不消费它）。
3. **适配器内部已乘 σ**，脚本外面不要再乘。
4. **对固定的群体读取向量，`add` 算子的级联是恒等的**（v 不随 R 变，σ 按当次前向算）。
   只有 `piecewise` 的 `sign(Rᵀv)` 依赖当前激活，已内联进钩子，仍只需一次前向。
5. **本机当前状态：第 1 步尚未通过**（dd/decel 那格 p=0.800，随机方向的斜率中位
   0.0075 还比真实方向的 0.0031 大）。所以远程第一件事就是确认阳性对照能不能过。

## 输出

- 每格一个 `results/steer_repe_{model}_{axis}_{mode}_{side}.json`
- `remote_pkg/summarize_causal.py` 汇总并按上面的逻辑给判定
