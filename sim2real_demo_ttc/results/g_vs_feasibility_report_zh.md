# G-VS 可行性评估：现成分割头 / SAM 伪 GT

> 工单：[`../docs/g_vs_f3_simplified_axes_workorder.md`](../../docs/g_vs_f3_simplified_axes_workorder.md) §1 的半天时间盒。
> 实际用时 < 2 小时。自行决策见 [`amendments.md`](amendments.md) §GF/A49–A52。

---

## 1. 两条待查项的结论

| 待查项 | 结论 |
| --- | --- |
| TransFuser 系是否已有分割辅助头？ | **有，但不能用作本轮的 G-VS** |
| 本机能否跑 SAM 生成伪 GT？ | **能**（需下载，已完成；0.9 s/图） |

### 1.1 TransFuser 系确实带一个分割头，但它是 **BEV 空间**的

`navsim/agents/transfuser/transfuser_model.py:41` 有 `_bev_semantic_head`，
`transfuser_config.py` 里 `use_bev_semantic = True`、`bev_semantic_weight = 10.0`，
即三个 TransFuser 系候选在**原始训练目标里就带语义分割**，7 个类：

| label | 类 | 来源 |
| --- | --- | --- |
| 1 | road（LANE + INTERSECTION） | 地图多边形 |
| 2 | walkways | 地图多边形 |
| 3 | centerline | 地图线串 |
| 4 | static_objects（锥桶/护栏/施工牌/通用物） | 标注框 |
| 5 | vehicles | 标注框 |
| 6 | pedestrians | 标注框 |

**为什么不能直接读它当 G-VS**，三条理由缺一不可：

1. **空间不同**：它输出的是**鸟瞰图（BEV）**语义图（`bev_pixel_size = 0.25 m`），
   而 SAM 伪 GT 是**图像空间**的。两者不是同一个预测任务，mIoU 不可互换。
2. **只覆盖 3/6 候选**：SimLingo / Alpamayo-R1 / AutoVLA 没有这个头。
   工单要求"六个候选统一测，不分单帧/多帧、有无语言"——
   用现成头测三个、用探针测另外三个，得到的 mIoU 跨候选不可比，直接违背统一的目的。
3. **它是训练目标，不是探针**：读它等于问"模型在它被显式训练的任务上做得好不好"，
   而 G-VS 要问的是"表征里**是否线性可读出**场景结构"。前者不需要 selectivity 对照，
   后者必须有——两者回答的不是同一个问题。

**但它有一处实质用途，已记入**：它证明**三个 TransFuser 系候选的表征里一定含有分割信息**
（否则训练目标学不动）。因此若它们的 G-VS 探针读数接近随机初始化地板，
可以确定问题出在**探针/网格分辨率**一侧，而不是"表征里没有"。
这为本轮的低 mIoU 提供了一个独立的解释锚点（见统一矩阵报告的讨论）。

### 1.2 SAM 可行

| 项 | 结果 |
| --- | --- |
| 依赖 | `segment_anything` 本机原无，pip 安装成功 |
| 权重 | `sam_vit_b_01ec64.pth`（375 MB），从 `dl.fbaipublicfiles.com` 下载成功 |
| 推理成本 | **0.9 s / 图**（ViT-B，`points_per_side=16`，cuda:0） |
| 全量成本 | G1 的 A 类 291 帧 ⇒ **< 5 分钟** |

**伪 GT 质量抽检（291 帧全量统计）**：每图掩码数中位 **45**；
object 像素占比均值 **0.244**，分位 [0.116, 0.235, 0.381]。
分布合理（既没有塌成全背景，也没有把整幅图糊成物体）。

---

## 2. 由此确定的 G-VS 任务定义（预注册）

SAM 的自动掩码是**无类别的实例掩码**，要变成良定义的密集预测任务必须给出客观标签规则。
本轮取**二类「物体性（object-ness）」分割**：

    class 1 (object)     ：被 SAM 掩码覆盖，且掩码面积 < 画幅的 5%
    class 0 (background) ：其余（无掩码，或路面/天空/建筑这类超大 stuff 区域）

* **只用 SAM**，不引入任何人工语义判据 —— 这正是本工单要摆脱"危险"这个构念的动机；
* 面积阈值把 stuff 与 thing 分开，是 SAM 自动掩码的标准用法，阈值随结果落盘；
* 二类 mIoU 良定义、跨候选可比，随机初始化对照有明确含义。

**它不是什么**（必须写明，否则 mIoU 会被过度解读）：
这不是语义分割（没有类别），也不是实例分割（不区分个体）。
G-VS 问的是"**哪里有物体**这件事能不能从表征里线性读出"，
不是"模型认得出这是行人还是锥桶"。后者留给 Future Work 的 **G-VL**。

## 3. 三条对照臂（缺一不可）

| 臂 | 含义 |
| --- | --- |
| trained | 训练好的模型的 token 特征 |
| **random_init** | **同架构、随机初始化**的模型，同一批图、同一层、同一网格（Hewitt & Liang 2019 的 control task） |
| **position_only** | 只用 token 的 (row, col) 坐标 —— 隔离"物体多在画面下半部"这个空间先验 |

**主读数 = selectivity = mIoU(trained) − mIoU(random_init)**，scene 级配对差 + bootstrap。
只报 mIoU 绝对值是不够的：token 网格自带空间先验，`position_only` 臂实测就能到 0.33，
不扣掉它与随机初始化，"模型表征里有信息"这个结论就不成立。

---

## 4. 可行性判定

**通过。** SAM 路线可行且成本极低；现成分割头不适用于本轮目的，理由已逐条记录。
本轮按"SAM 伪 GT + 线性探针 + 双地板对照"执行。
