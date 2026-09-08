# 预注册：左舵急刹提轴 → 右舵急刹检验（训练/测试切分）

写于 2026-09-07，激活重抽中，`harsh_acts_*.npz` 尚未用于任何计算。

## 设计

轴**只用左舵**估：
$$\hat v_{\text{harsh}}=\text{unit}\big(\text{mean}(h\mid\text{hazard},\text{LHD})-\text{mean}(h\mid\text{no\_hazard},\text{LHD})\big)$$
（左舵 19 个危险引起的急刹 vs 29 个红灯/让行引起的急刹）

在**右舵**上检验：把右舵 31 个急刹事件（11 hazard / 20 no_hazard）投到这根轴上，
看两组能否分开。**轴完全没见过右舵数据**，是真正的 out-of-sample。

对比之前那个检验（`PREREG_vfaith_explains_harshbrake.md`）：那次轴来自
brake-first 池、用全部 79 个事件一起判定；这次轴来自急刹事件本身、且左右舵严格切分。

## 预测（写在计算之前）

- **T1**　右舵上两组可分：单边置换检验 p < 0.05
- **T2**　效应量 Cliff's δ ≥ 0.33
- **T3**　右舵的 δ **不超过**左舵自身（留一交叉验证）的 δ 的 1.5 倍
  —— 若右舵反而明显更强，说明不是"轴迁移成功"，而是右舵样本的某种巧合

**我的具体预期**：T1 成立但 T2 不成立（能分开但弱）。依据是上一轮
用 v_faith 做同一件事时，左舵 δ 只有 0.005–0.205、右舵 0.100–0.664。

**判定**
- T1 ∧ T2 ⇒ 左舵学到的「危险 vs 非危险」表征方向**能迁移到右舵**，
  这是本文关于「少量 deploy 场景即可测量」最直接的支持证据。
- T1 不成立 ⇒ 方向不迁移，「用 bench 轴测 deploy」这条路在急刹口径上不通。

## 口径（写死）

- 激活：`results/harsh_acts_{model}.npz` 的 `h`，层 = 各模型三重门选出的 L*
- 事件表：`results/harsh_brake_labeled_nusc.json`（79 个，group 由「急刹帧前方
  2–60 m、|lat|≤2.5 m 有可见非家具物体」判定，逐个调图核过其中 6 个）
- 候选：dd / ltf / ddv2 / simlingo
