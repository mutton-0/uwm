# 完成标志：G-VS（分割一致性）+ F-3（遮挡必要性检验）

> 工单：[`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md)。
> 完成日期：2026-09-01。修正案：[`amendments.md`](amendments.md) §GF/A49–A52（累计 52 条）。
> **可行性检查通过**（半天时间盒，实际 < 2 小时）。

---

## 1. 可行性检查结论（工单 §1）

| 待查项 | 结论 |
| --- | --- |
| TransFuser 系是否已有现成分割头？ | **有，但不能用作本轮 G-VS**：它是 **BEV 空间**（`bev_semantic_head`，7 类），而 SAM 伪 GT 是图像空间；且只覆盖 3/6 候选，用它测三个、用探针测另外三个会让 mIoU 跨候选不可比，直接违背"六候选统一"的目的；且它是训练目标不是探针，不需要 selectivity 对照，回答的不是同一个问题。**但它证明了三个 TransFuser 候选的表征里一定含分割信息**，这为本轮低 mIoU 的归因提供了独立锚点。 |
| 本机能否跑 SAM？ | **能**。`segment_anything` pip 安装成功，`sam_vit_b_01ec64.pth`（375 MB）下载成功，**0.9 s/图**，291 帧 < 5 分钟。伪 GT 质量：每图掩码中位 45，object 像素占比均值 0.244（分位 0.116/0.235/0.381）。 |

## 2. 六候选统一矩阵（v1 主结果）

| Policy | **G-VS** selectivity | G-VS 判定 | **F-3** 必要性比 $R$ | F-3 判定 |
| --- | --- | --- | --- | --- |
| SimLingo | **+0.0273 [+0.0128, +0.0424]** | **PASS** | **+0.461 [+0.166, +0.789]** | 不可估（CI 跨 0.5） |
| DiffusionDrive | **+0.0381 [+0.0211, +0.0551]** | **PASS** | 基线响应不存在 | 不可估 |
| LTF | +0.0191 [−0.0006, +0.0385] | 不可估 | **+0.113 [+0.044, +0.191]** | **FAIL** |
| DiffusionDriveV2 | +0.0159 [−0.0029, +0.0353] | 不可估 | 基线响应不存在 | 不可估 |
| Alpamayo-R1 | n.m. | — | 基线响应不存在 | 不可估 |
| AutoVLA | n.m. | — | 基线响应不存在 | 不可估 |

**F-3 覆盖 6/6；G-VS 覆盖 4/6**（两个 VLA 的 video token 布局未接完，§GF/A50）。

### 三条最值得看的结论

1. **LTF 是唯一的 FAIL，也是唯一的 blind action 签名**：遮住危险实体只移除 **11%** 的响应
   （$R$ = +0.113 [+0.044, +0.191]），而在别处涂同样大小的灰斑移除 3%（CI 跨 0）。
   即"遮挡"这个操作有效，无效的是"遮住**这个实体**"——动作有响应，但不是被该实体驱动的。
2. **其余四个是"不可估：基线响应本身不存在"，不是 FAIL**（§GF/A52）。
   两者修复处方完全不同：FAIL 要查驱动源并重新接线，无响应要先让它有响应。
   **把两者都写成"没通过"会把两种处方压成一个数字。**
3. **DiffusionDrive 一格同时给出两件事**：G-VS selectivity 全表最高（+0.038，表征里确有物体性信息），
   F-3 基线响应与 0 不可区分（动作对危险帧根本没反应）。
   **"看得见"与"据此行动"独立** —— 与旧 G/F 反复给出的结论一致，
   但这次不依赖任何"危险"判据。

### 两条必需对照臂都起了作用

* **G-VS 的 `position_only` 地板 = 0.333**，而 trained mIoU 只有 0.35 ~ 0.40 ⇒
  **绝对 mIoU 的绝大部分来自空间先验，表征净贡献只有 0.02 ~ 0.04**。
  selectivity 是配对差故仍可信，但净贡献小必须并报。
* **F-3 的 $R_{ctrl}$** 六个候选全部落在 [−0.09, +0.16]、CI 跨 0 ⇒
  "加灰斑"本身不会让动作退回基线。**这是 LTF 的 FAIL 判定得以成立的前提**（§GF/A51）。

## 3. 产出物路径

| 路径 | 说明 |
| --- | --- |
| `scripts/gvs1_sam_pseudo_gt.py` | SAM 伪 GT 生成（二类物体性，面积阈值 5%） |
| `scripts/gvs2_probe.py` | 线性探针 + 三条臂（trained / random_init / position_only）+ selectivity |
| `scripts/gvs3_extract_tokens.py` | token 级特征抽取（TransFuser 系 8×32 网格、SimLingo tile 网格） |
| `scripts/gvs4_extract_vla.py` | VLA 版抽取（AutoVLA 布局已解出，实现未通，§GF/A50） |
| `scripts/f3_occlusion_necessity.py` | F-3 四臂（clean/ghost/occ/ctrl），非 VLA 候选 |
| `scripts/f3_occlusion_vla.py` | F-3 四臂，Alpamayo / AutoVLA |
| `results/alpamayo_g1_adapter/alpa_patch.py`（改） | 新增 `run_from_data()`（在已遮挡的 data 上前向） |
| `variants/n1_d2/sam_gt/*.png` | 291 帧伪 GT；`variants/n1_d2/results/sam_pseudo_gt_report.json` 质检 |
| `variants/n1_d2/gvs/tokens_*.npz` | 四候选的 token 特征（含随机初始化臂） |
| `results/g_vs_feasibility_report_{zh,en}.md` | **可行性评估**（分割头为何不用、SAM 成本、任务定义） |
| `results/g_vs_axis_{simlingo,dd,ltf,ddv2}_{zh,en}.md` | G-VS 逐候选读数（含三臂对照） |
| `results/f3_occlusion_axis_{simlingo,dd,ltf,ddv2,alpa,autovla}_{zh,en}.md` | F-3 逐候选读数 |
| `results/g_vs_f3_unified_matrix_report_{zh,en}.md` | **核心交付**：六候选统一矩阵，不分组 |
| `results/g_vs_axis_*.json`、`results/f3_occlusion_*.json` | 全部数值产出物 |

## 4. 并入论文（工单 §5）

`results/paper_experiments_section_{zh,en}.md`：

* **Table 1(v1) 新增**：G-VS / F-3 六候选统一，作为 **v1 主读数**，置于 Table 1(a)–(c) 之前；
* **§4.2.9 新增**：复杂 G/F①/F② **一个数字不删**，改列为
  "为什么围绕语义构念构造的轴难以量化"的二级证据体系，
  并给出五条实证（跨场景 0/3、跨数据源 0/4、地板反超主读数、操作化不可跨族比较、判据本身人工）；
* **§4.6 Future Work 新增**三项，锚定文献：
  **G-VL**（语义化 G-VS）、**F-1**（CoC 事后合理化，Embodied Interpretability arXiv 2605.00321）、
  **F-2**（虚假相关/注意力，Causal Imitative Model, Samsami et al. 2021, arXiv:2112.03908 的
  causal inversion 框架与 inertia/collision 两种失效签名）；
* 修正案计数 48 → **52**。

## 5. 自我更正记录（4 条，全文见 amendments.md）

1. **§GF/A49**：G-VS 探针首版手写 GD **未收敛**，三臂读数几乎同值、
   mIoU 0.2971 恰好等于"全预测背景"的解析值。换 lbfgs 后三臂分离。
   **教训：探针实验里"三臂读数相同"要先怀疑优化，再怀疑表征。**
   若不核对退化解，会得出"六个候选表征里都没有物体性信息"这个错误结论。
2. **§GF/A50**：G-VS 只覆盖 4/6，如实标 n.m. 而非硬凑。
   AutoVLA 的 token 布局本轮**已解出**（3 相机 × `[2,18,32]`，2×2 merge ⇒ 前视最后时间组
   9×16 = 144 token 覆盖全幅），写进修正案供后续直接接。
3. **§GF/A51**：F-3 的 ctrl 对照臂设为必跑而非可选，理由与判定效果已记录。
4. **§GF/A52**：F-3 的"不可估（无基线响应）"与"FAIL"必须分开写，两者修复处方不同。

## 6. 未完成项（如实记录）

| 项 | 状态 | 理由 |
| --- | --- | --- |
| G-VS 的两个 VLA 候选 | **未测（n.m.）** | AutoVLA 布局已解、实现卡在随机初始化对照臂；Alpamayo 6 相机分组未核对。不用"取平均 token"凑格。 |
| G-VS 的网格分辨率消融 | 未做 | trained mIoU 0.35~0.40 vs position_only 0.333，净贡献小，归因偏读出侧但未做消融验证。 |
| G-VL（语义类别版） | 未做 | 列入 Future Work §4.6。 |
| F-1 / F-2 | 未做 | 列入 Future Work §4.6。F-1 需先有非零基线响应的语言候选；F-2 应优先做 LTF。 |
| 前车急刹 / NAVSIM 上的 G-VS/F-3 复现 | 未做 | 工单 §3 明确列为"有余力再做的加分项"，非硬性要求。 |
| F-3 的 inpainting 版遮挡 | 未做 | 本轮用灰斑填充；ctrl 臂控制了"加灰斑"但未控制"改变全局图像统计"。 |
