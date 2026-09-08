# 预注册：ghost 场景的 θ 能否预测 lead 场景的官方 PDM 排名

**写于 2026-09-07 17:5x，Alpamayo / AutoVLA 的 ghost 激活仍在抽取中，
其 θ 尚未计算、尚未被任何人看到。** 这份文件的存在就是为了让下面的检验
是事前的而不是事后的。

## 假设（用户提出的设计意图）

用**针对性危险场景**（ghost = 遮挡的易受伤害道路使用者）在**我们自己的
表征指标**下的分数，预测**另一个更泛化场景**（lead = 前车急刹）在
**官方闭环 PDM** 下的排名。

若成立，其价值在于：θ 只需两次前向 + 一个夹角，而 PDM 需要闭环仿真 +
metric cache；且 θ 完全不依赖仿真器。

## 已知（假设的来源，不计入检验）

四家在 ghost θ 与 lead PDM 上排名完全一致（Spearman = +1.00）：

| 候选 | lead PDM(危险事件) | ghost θ |
|---|---|---|
| LTF | 0.7546 | 29.5° |
| DiffusionDriveV2 | 0.7527 | 30.9° |
| DiffusionDrive | 0.6900 | 54.3° |
| SimLingo | 0.3297 | 62.9° |

**这四家是假设的生成集，不能同时当作检验集。** 且这个配对是在先试过
lead θ（Spearman +0.40，不匹配）之后才找到的，属事后挑选。

## 检验（新信息只有两家）

lead PDM 六家已算完（n=32 事件，见 `hazard_avoidance_dep_lead_6m.json`）：

| 候选 | lead PDM | 名次 |
|---|---|---|
| AutoVLA | 0.771 | 1 |
| DiffusionDriveV2 | 0.747 | 2 |
| LTF | 0.745 | 3 |
| DiffusionDrive | 0.683 | 4 |
| Alpamayo-R1 | 0.476 | 5 |
| SimLingo | 0.328 | 6 |

**逐条可证伪的预测（在看到数据前写下）：**

- **P1**　AutoVLA 的 ghost θ 应当**小于 DiffusionDrive 的 54.3°**。
  （它 PDM 第一，理应比 PDM 第四的 DD 表征更稳。）
- **P2**　Alpamayo 的 ghost θ 应当**落在 54.3° 与 62.9° 之间**，
  即比 DD 差、比 SimLingo 好，对应它 PDM 第五。
- **P3**　六家的 Spearman(ghost θ 升序, lead PDM 降序) **≥ +0.83**
  （即最多错一个相邻对换）。

**判定**：P1 与 P2 同时成立 ⇒ 假设通过本轮检验；只成立一条 ⇒ 存疑；
两条皆不成立 ⇒ 四家那次是巧合，假设否定。

六家排名完全命中的随机概率 = 1/6! = 1/720 ≈ 0.0014；
但真正的新信息只有两家插入四家已定序列中的位置，随机命中概率
= 1/(5×6) ≈ 0.033（P1∧P2 的严格版），故本轮**至多**是中等强度证据。
要更强的证据必须加候选或换场景对。

## 口径（写死，事后不得更改）

- θ = `f_vfaith_direction.py --pooled --topk 3`，选层依据 = **左舵侧** SNR
  （与跨域比较无关），各层单位化后拼接，右舵 vs 左舵夹角。
- ghost 池 = `work_c_dep/ghost`（NAVSIM）+ `work_c_new/ghost_nusc`（nuScenes），
  按 P-1 合并后切舵位。
- lead PDM = `results/hazard_avoidance_dep_lead_6m.json` 的 `pdm_score.mean`。
