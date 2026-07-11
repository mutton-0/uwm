# sim2real 分析脚本 (eva_ckpts_by_pdm)

针对 DiffusionDrive 规划器的 **sim2real 差异定位** 分析脚本集合。目标：用尽量少的诊断
成本，定位 transfer(sim) 相对 origin(real) 的能力退化**因果层**，为低成本 LoRA 对齐提供
依据。所有脚本共用同一个 ckpt / 数据加载路径，通过环境变量配置。

## 迁移到新仓库时需要改的路径

三个脚本头部都读同一组环境变量（有默认值，指向本机 `ruolin_a6k`）。迁移只需 export：

```bash
export SIMSCALE_ROOT=/path/to/SimScale                         # 仓库根
export BKB_PATH=/path/to/my_dataset/models/resnet34_model.bin  # ResNet34 backbone 权重
export CKPT_PATH=/path/to/ckpt/diffusiondrive_sim_navhard.ckpt # 待评估 ckpt
```

另外脚本内**硬编码**了 NavSim 数据集根（`OPENSCENE_DATA_ROOT` /
`NUPLAN_MAPS_ROOT`，当前指向 `/data/Yuhao/.../navsim_workspace/dataset`）。若数据集位置
不同，改脚本顶部这几行，或在运行前 export 覆盖。

依赖 transfer 图目录：`$SIMSCALE_ROOT/outputs/my_diffusion_0_transfer2sim_scenarios/`
（`{token}_after_transfer.jpg`）。origin 用原始 sensor 图，transfer 用这里的注入图
（经 `TRANSFER_SCENARIO_IMAGE_DIR` / `TRANSFER_IMAGE_SUFFIX` 环境变量注入，见
`outputs/evaluation_guide.md` 第 7 章）。

## 脚本与运行顺序

| # | 脚本 | 作用 | 关键输出 |
|---|------|------|----------|
| 1 | `capture_selfattn.py` | 为有 transfer 图的 token 抓取 encoder 8 层 self-attn（+完整 decoder 特征/attn），写 `outputs/analysis_cache_encoder/{token}_{mode}.pkl` 超集 | pkl 缓存，供 head 统计与 CKA 用 |
| 2 | `activation_patching.py` | **因果定位**：transfer 前向时逐层把第 L 个 encoder SelfAttention 输出替换成 origin 的，测轨迹恢复度 `recovery(L)` | 每层 recovery、argmax 因果层分布、深层 L4-6 占比 |
| 3 | `cka_vs_recovery.py` | 对照三种量：注意力 JS 散度 / 特征级 1-CKA(Kornblith 2019) / recovery，看"表征对齐度"是否比"注意力散度"更贴近因果 | 每层三量表 + Pearson/Spearman 相关性 |

```bash
cd $SIMSCALE_ROOT
python scripts/sim2real_analysis/capture_selfattn.py      # 先补齐 pkl 缓存
python scripts/sim2real_analysis/activation_patching.py   # 因果 recovery
python scripts/sim2real_analysis/cka_vs_recovery.py       # 对照分析
```

## 已得结论（2 token，待扩样验证）

- **L6 因果主导**（recovery 0.82 / 0.80），L4–L6 为因果层；L0/L1/L3 ≈ 0；L7 弱（0.16/0.05）。
- patch 全部 8 层 → recovery ≈ 1.000（8 个 self-attn 是充分割集）。
- **推翻**"L1 导致路面误判"假设（L1 recovery ≈ 0），也推翻"按注意力散度放 LoRA"
  （L6 的 JS 近最低却因果最高）。
- CKA↔recovery（Spearman +0.48/+0.21，正）优于 JS↔recovery（-0.19/-0.40，负），
  但 argmax 仍落在 L7（错），**CKA 也不是可靠因果定位器**。

> 注意：当前样本含早期 warmup 的全 0 origin 坏样本。需按 `outputs/evaluation_guide.md`
> 重跑 navmini 全 token PDM 生成干净 origin 分后，扩到 ≥30 个 real>sim 且轨迹不同的
> 退化 token 复核 L6/深层主导结论。

## 相关产物

- `outputs/sim2real_research_plan.md` / `.html` — 研究计划与理论背景（论文大纲式）
- `outputs/evaluation_guide.md` — NavSim 运行 / 自定义图注入 / attention 分析指南
- `scripts/build_bokeh_layer_diff_analysis.py` — 逐层差异抓取+绘图（本目录脚本的源）
- `scripts/build_bokeh_encoder_selfattn_rgb_analysis.py` — encoder self-attn head 级 RGB 叠加分析
