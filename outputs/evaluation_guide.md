# NAVSIM 测评与打分执行指南

本文档记录了使用本地权进行评估打分、生成测试结果以及可视化分析低分场景的完整操作流程。

## 1. 核心环境变量配置

在执行任何命令前，需要确保配置并激活正确的运行环境（尤其是 `NAVSIM_EXP_ROOT` 已经修改为新的路径，所有的运行日志和结果都将存放在该目录下）。

```bash
# 激活 python 虚拟环境
conda activate ipad

# 基础数据和仓库路径
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"
export NAVSIM_DEVKIT_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset"

# [修改项] 输出路径（所有的评测 csv, submission.pkl 等都会保存在这里）
# export NAVSIM_EXP_ROOT="/data/ruolin/iPad/navsim_logs/exp"
export NAVSIM_EXP_ROOT="/data/ruolin/navsim_logs/exp"
```

> **提示**：在使用 VS Code 左侧面板（运行和调试）启动任务时，我已经为你写好了相关的配置在 `.vscode/launch.json` 中，系统会自动加载上述环境变量，不需要手动在命令行输入。

---

## 2. 本地评估跑分流程 (Warmup)

在正式提交前，通常先在小规模的 `warmup` 集合（基于 `mini` 数据）中做快速打分测试。需要按照先后顺序执行以下两步：

### 步骤 A：缓存数据特质 (Metric Caching)
只做场景的交通规则和背景信息的分析缓存，为防冲突需手动指定保存的绝对路径。
* **命令**：
  ```bash
  python navsim/planning/script/run_metric_caching.py \
      train_test_split=warmup_test_e2e \
      +cache_path=$NAVSIM_EXP_ROOT/metric_cache
  ```
* **VS Code 快捷键**：选择 `1. Cache Warmup Metrics (Local)` 并运行。

### 步骤 B：执行打分 (Run PDM Score)
在上一步预计算好的约束基础上，跑一遍你的指定模型权重并进行具体的计算打分。
* **命令**：
  ```bash
  python navsim/planning/script/run_pdm_score.py \
      train_test_split=warmup_test_e2e \
      agent=navsim_agent
  ```
* **VS Code 快捷键**：选择 `2. Run Warmup PDM Score (Local)` 并运行。

运行完毕后，这将会输出测试摘要，并在 `/data/ruolin/iPad/navsim_logs/exp/ke/ke/[时间戳]/` 目录下生成一个包含每个场景 Token 各项明细分的 **CSV 文件**。

---

## 3. 生成 Submission 文件 (测试集预测)

此步骤不会在本地出分数，仅仅是在对应的 Split 上跑模型并生成待提交到榜单的 `submission.pkl` 包：

* **命令**：
  ```bash
  python navsim/planning/script/run_create_submission_pickle.py
  ```
* **VS Code 快捷键**：选择 `Debug run_create_submission_pickle` 并运行。
该 Pkl 文件会生成在类似 `/data/ruolin/iPad/navsim_logs/exp/ke/ke/[时间戳]/submission.pkl` 的位置，可以将此文件上传至 Hugging Face 用于 EvalAI 评估。

---

## 4. 低分案例可视化分析

我们提供了一个额外的定制化脚本，用于读取第二步生成的 CSV 文件，找寻得分为 `0`（或很低）的特定场景（token），然后抽出场景前后的 Camera 贴图和 BEV 地图画出来，用来查错和优化。

* **命令**：
  ```bash
  python extract_zero_score.py
  ```
* **流程说明**：
  打开 `extract_zero_score.py`，将 `csv_path` 变量更换为你最近一次由于第二步新生成的 `.csv` 路径（此文件现在会存在新配置的 `$NAVSIM_EXP_ROOT` 下面），然后运行代码。
* **输出**：
  查找到的问题场景将输出一对 `{token}_cameras.png` 与 `{token}_bev.png`，存放在 `/data/ruolin/iPad/outputs/low_score_pics` 目录下。

---

## 5. 本地评估跑分流程 - LTF 模型及验证集 (navtest)

针对 LTF (Latent Transfuser) 模型在 `navtest` 验证集上的专门打分流程，需要特别注意 `PYTHONPATH` 的设置以及 Hydra 的强制参数 `experiment_name`，具体完整执行步骤如下：

### 核心环境与路径前置准备
确保在终端执行以下命令，避免模块查找错误和找不到文件：

```bash
# 1. 必须进入项目根目录
cd /data/ruolin/navsim

# 2. 激活虚拟环境
conda activate navsim

# 3. 设置数据集绝对路径与输出路径
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"
export NAVSIM_DEVKIT_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset"
export NAVSIM_EXP_ROOT="/data/ruolin/navsim_logs/exp"

# 4. 指定 PYTHONPATH，确保优先加载本地修改过的 navsim 代码包
export PYTHONPATH="/data/ruolin/navsim:$PYTHONPATH"
```

### 步骤 A：缓存数据特质 (针对 navtest 集合)
如果是第一次在 `navtest` 数据集上跑，必须先构建 Metric 缓存（注意正确的 Hydra 语法需要使用 `+` 号）：
```bash
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    +cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### 步骤 B：执行打分 (Run PDM Score)
在上一步预计算好约束后，调用刚才写好的 `ltf_agent` 并传入必填参数 `experiment_name` 运行评估：
```bash
python navsim/planning/script/run_pdm_score.py \
    train_test_split=navtest \
    agent=ltf_agent \
    experiment_name=ltf_eval
```
此时日志与生成的 CSV 文件将存放在 `$NAVSIM_EXP_ROOT/ltf_eval/` 对应的子目录内。


---

## 6. SimScale 环境端到端模型 (DiffusionDrive) 单阶段评测流程 (navmini)

针对 SimScale 的预训练模型（如 DiffusionDrive），为了在不下载庞大的 `navhard_two_stage` 等带合成数据的测试集的情况下，快速在本地走通推理（Inference）流水线，我们可以直接借用本地现有的 `navmini` 集合，采用单阶段（one_stage）专用脚本进行测试。

### 步骤 A：核心环境与前置权重准备
因为模型骨干网络有自己的固定依赖，首先需要确保当前环境已配置无误并下载模型所必须的 ResNet34 初始化权重至独立目录（避免干扰共享数据）。
```bash
#仅执行一次的创建conda 
#参考 https://github.com/OpenDriveLab/SimScale/blob/main/README.md#-getting-started
conda env create --name simscale -f environment.yml
conda activate simscale
pip install -e .

# 激活环境与目录
cd /data/ruolin/SimScale
conda activate simscale

# 导出必要的 SimScale 路径与挂载数据集映射
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset"
export NAVSIM_EXP_ROOT="/data/ruolin/navsim_logs/exp"
export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale"
export PYTHONPATH="/data/ruolin/SimScale:$PYTHONPATH"

# 建立私有目录并下载 Pytorch 官方版缺失的预训练 ResNet34 backbone
mkdir -p /data/ruolin/my_dataset/models
cd /data/ruolin/my_dataset/models
wget -O resnet34_model.bin https://download.pytorch.org/models/resnet34-b627a593.pth
```

### 步骤 B：缓存 `navmini` 集合特征 (Metric Caching)
回到项目根目录生成 `navmini` 验证集前置评估需要比对的分数背景（Metric Cache）：
```bash
cd /data/ruolin/SimScale
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navmini \
    +cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

### 步骤 C：执行小规模评测 (Run PDM Score Inference)
**关键**：在纯 `navmini` 子集只能跑 `one_stage` 脚本。启动时务必显式重写并下挂 Backbone、Anchor 路径以便模型加载。
```bash
ckpt_path="'/data/ruolin/ckpt/diffusiondrive_sim_navhard.ckpt'"
experiment_name="test_diffusiondrive_hf_ckpt_navmini"

# 使用 SimScale 特定提供的 one_stage 单阶段测试脚本
python navsim/planning/script/run_pdm_score_one_stage_gpu_diffusiondrive.py \
    agent=diffusiondrive_agent \
    dataloader.params.batch_size=32 \
    agent.checkpoint_path=$ckpt_path \
    experiment_name=$experiment_name \
    +cache_path=null \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
    train_test_split=navmini \
    agent.config.bkb_path=/data/ruolin/my_dataset/models/resnet34_model.bin \
    agent.config.plan_anchor_path=${NAVSIM_DEVKIT_ROOT}/traj_final/kmeans_navsim_traj_20.npy
```
运行完成后即可完成验证与 Pipeline 的首尾跑通。测试输出的 `.csv` 或者 `.pkl` 日志评估成绩都将会保存在 `$NAVSIM_EXP_ROOT/test_diffusiondrive_hf_ckpt_navmini/` 子目录里。

### 新增：预制 YAML 简化命令行运行！
为了避免在终端里拼写一大堆类似 `agent.config.bkb_path=...` 的配置，我已经在你的 `/data/ruolin/SimScale/navsim/planning/script/config/common/agent/` 目录中，创建了针对各个模型已经固定好权重路径（尤其是本地 `mini` 跑通需要的补丁）的定制 `yaml` 配置文件：
1. **`my_diffusiondrive.yaml`**
2. **`my_gtrs_dense.yaml`**
3. **`my_ltf.yaml`**

因此，如果你想在 `navmini` 测试集上运行 **GTRS-Dense** 或者是 **LTF** 的单阶段测评（不需要显性挂一大串参数）：

**运行 GTRS-Dense (One Stage on Navmini):**
```bash
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT/test_my_gtrs_dense_navmini/navmini_subscore.pkl
python navsim/planning/script/run_pdm_score_one_stage_gpu.py \
    agent=my_gtrs_dense \
    +combined_inference=false \
    dataloader.params.batch_size=16 \
    trainer.params.po'gision=32 \
    experiment_name="test_my_gtrs_dense_navmini" \
    +cache_path=null \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
    train_test_split=navmini
```

**运行 LTF (Transfuser One Stage on Navmini):**
```bash
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT/test_my_ltf_navmini/navmini_subscore.pkl
python navsim/planning/script/run_pdm_score_one_stage_gpu_transfuser.py \
    agent=my_ltf \
    dataloader.params.batch_size=32 \
    experiment_name="test_my_ltf_navmini" \
    +cache_path=null \
    metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache \
    train_test_split=navmini
```

有了 `my_xxx.yaml` 以后，代码会自动读取其中绑好的 `/data/ruolin/ckpt/xxx.ckpt` 模型权重以及从我们在 `my_dataset` 下载好的 ResNet backbone 初始化它！

### 一键执行脚本 (run_navmini_all_models.sh)
为了进一步简化操作，我已经把所有环节（设置环境变量 -> 环境缓存检查 -> 分别评测三个模型）合并写进了一键执行的 shell 脚本里。它集成了所有的最佳实践参数绑定。

脚本路径在：`/data/ruolin/SimScale/scripts/run_navmini_all_models.sh`

你只需要在终端激活环境后运行：
```bash
cd /data/ruolin/SimScale
conda activate simscale
bash scripts/run_navmini_all_models.sh
```
它会自动按顺序跑完 DiffusionDrive、GTRS-Dense 和 LTF 三个模型，并把它们各自的 `navmini_subscore.pkl` 打分日志存放在 `$NAVSIM_EXP_ROOT` 下对应的子目录里！

---

## 7. 提供外置自定义图片替换并评估输出轨迹可视化

如果要对 DiffusionDrive 模型在推理时注入外置合成的 RGB 贴图（并自动适配裁剪为 3 视野，而不是使用默认提取的场景原始数据集相机流），我们提供了完整流程：

### 第一步：修改支持图片挂载后缀名
这部分支持已经加入在基础模块中了：
它依靠导出环境变量 `TRANSFER_SCENARIO_IMAGE_DIR` 和 `TRANSFER_IMAGE_SUFFIX` 的存在情况在推理期劫持数据装载流程。

### 第二步：批量评测脚本 (Custom Images)
以挂载 `outputs/my_diffusiondrive_zero_cams/` 下结尾是 `*_stitched_input.png` 批量评测为例。
```bash
cd /data/ruolin/SimScale
conda activate simscale

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"
export OPENSCENE_DATA_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset"
export NAVSIM_EXP_ROOT="/data/ruolin/navsim_logs/exp_zero_cams"
export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale"
export PYTHONPATH="/data/ruolin/SimScale:$PYTHONPATH"

# 指定修改后的图片输入位置及名称后缀
export TRANSFER_SCENARIO_IMAGE_DIR="/data/ruolin/SimScale/outputs/my_diffusiondrive_zero_cams"
export TRANSFER_IMAGE_SUFFIX="stitched_input.png"
export CUDA_VISIBLE_DEVICES=0

# 若批量评测清空此变量防止加载同一图片
unset NAVSIM_CURRENT_TOKEN
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT/test_my_diffusiondrive_zero_cams/navmini_subscore.pkl

# 限定测试你的 Tokens:
python navsim/planning/script/run_pdm_score_one_stage_gpu_diffusiondrive.py \
    agent=my_diffusiondrive \
    dataloader.params.batch_size=1 \
    experiment_name="test_my_diffusiondrive_zero_cams" \
    +cache_path=null \
    metric_cache_path=/data/ruolin/navsim_logs/exp/metric_cache \
    train_test_split=navmini \
    train_test_split.scene_filter.tokens=["1d05dbff3a245c6b","258325ee3fe65b51","2e0ec9c9c8fa51ba"]
```
*以上 token 列出部分作为例子，执行后可以在 `$SUBSCORE_PATH` 找到预测轨迹矩阵以及相关的评测 CSV*。

### 第三步：轨迹可视化叠加 BEV
评估跑出 pkl 后，我们可以通过运行写好的可视化映射脚本 `plot_bev_zero_cams.py`（针对上个步骤 `test_my_diffusiondrive_zero_cams/navmini_subscore.pkl` 等日志）
```bash
# 激活同等上文环境变量后 (见 /data/ruolin/SimScale/plot_bev_zero_cams.py) 
python /data/ruolin/SimScale/plot_bev_zero_cams.py 
```
结果会输出类似于 `{token}_bev_agent_traj_all.png` 这样的图片保存在：
`/data/ruolin/SimScale/outputs/my_diffusiondrive_zero_cams_bev_agent_traj_all` 目录下，包含地图路网原貌并叠加上原真实驾驶人轨迹（红线）以及我们替换图片后 Agent 规划产生的新轨迹（绿线）。

---

## 8. 编码器注意力可视化与每 Token 一页总览（Transfer JPG）

本节用于分析 `outputs/my_diffusion_0_transfer2sim_scenarios/*_after_transfer.jpg` 输入下，DiffusionDrive **编码器侧**注意力在 RGB 上的权重分布，并自动生成“每个 token 一页”的图文稿和 PDF。

### 步骤 A：运行编码器注意力图批量生成
```bash
cd /data/ruolin/SimScale
conda activate simscale

export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale"
export NUPLAN_MAPS_ROOT="/data/Yuhao/world_model_yhl/navsim_workspace/dataset/maps"

# 指向 transfer2sim 目录与后缀
export TRANSFER_SCENARIO_IMAGE_DIR="/data/ruolin/SimScale/outputs/my_diffusion_0_transfer2sim_scenarios"
export TRANSFER_IMAGE_SUFFIX="after_transfer.jpg"

# 自动按目录中图片名解析 token
export ENCODER_ATTN_USE_DIR_TOKENS=1

# 输出目录（建议独立）
export ENCODER_ATTN_OUT_DIR="/data/ruolin/SimScale/outputs/attention_viz_encoder_rgb_transfer2sim_all"

python /data/ruolin/SimScale/visualize_encoder_attention_rgb.py
```

输出：
- `*_origin_encoder_attention_rgb.png`
- `*_transfer_encoder_attention_rgb.png`
- `encoder_attention_summary.csv`

目录：
`/data/ruolin/SimScale/outputs/attention_viz_encoder_rgb_transfer2sim_all`

### 步骤 B：生成每个 Token 一页的 Markdown 与 PDF
```bash
cd /data/ruolin/SimScale
conda activate simscale
python /data/ruolin/SimScale/scripts/generate_encoder_attention_overview.py
```

输出文件：
- 图文稿（Markdown）：`/data/ruolin/SimScale/outputs/report_sci_rgb_style_effect_token_pages.md`
- 总览 PDF（每 token 一页，origin vs transfer 并排）：`/data/ruolin/SimScale/outputs/report_sci_rgb_style_effect_token_pages.pdf`

### 步骤 C：主报告自动关联
上一步脚本会自动在主报告末尾追加“每个 Token 一页可投屏版面”的入口信息。

主报告路径：
`/data/ruolin/SimScale/outputs/report_sci_rgb_style_effect.md`

---

## 9. 编码器自注意力逐层分析 + RGB 叠加（Origin vs Transfer）

对缓存中含完整编码器自注意力权重的 2 个 token（`aa96f52b95b155e7`、`ca9e7281adce5212`），
把后几个（全部 8 个）融合 transformer 层 `encoder_selfatt_0..7` 的注意力分布用统计折线图画出，
并把每层图像注意力(8×32)上采样叠加到输入 RGB 上，origin/transfer 并排。

### 步骤 A：运行脚本
```bash
cd /data/ruolin/SimScale
conda activate simscale
export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale"

python /data/ruolin/SimScale/scripts/build_bokeh_encoder_selfattn_rgb_analysis.py
```

说明：
- 编码器自注意力权重取自 `outputs/analysis_cache_encoder/{token}_{mode}.pkl` 的 `attn['encoder_selfatt_*']`。
- 输入 RGB 首次运行时用 feature builder 重建（仅特征构建、不跑前向），缓存到
  `outputs/analysis_cache_encoder/rgb_inputs.npz`，之后秒出。

### 步骤 B：查看输出
- 交互 HTML：`/data/ruolin/SimScale/outputs/bokeh_encoder_selfattn_rgb_analysis.html`

### 页面内容
每个 token 一个 tab，包含：
- 各 transformer 层注意力分布差异曲线（KL / JS / Wasserstein）
- 图像 keys(256) vs lidar/BEV keys(64) 注意力占比（编码器内的 ego/agent 划分）
- 各层图像注意力熵、各层水平方向(全景宽32)边缘分布小多图
- decoder 的 cross_agent(30) / cross_bev(8) 分布
- **每层图像注意力叠加到输入 RGB**（origin/transfer 并排）+ 差异最大层的文字分析

结论（两 token 一致）：差异最大的层集中在**首尾**——L7（最深融合层）JS 最大，L0（最浅、直接吃像素/风格）次之，中间层稳定。

---

## 10. DiffusionDrive 网络结构与逐层维度（HTML）

自包含结构文档，所有张量维度取自一次**真实前向**（batch=1，推理模式，钩子记录 713 个子模块 I/O）。

### 步骤 A：抓取真实维度（可选，已生成 json 可跳过）
```bash
cd /data/ruolin/SimScale
conda activate simscale
export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale" CUDA_VISIBLE_DEVICES=0
python /data/ruolin/SimScale/scripts/_capture_model_shapes.py   # -> outputs/_model_shapes.json
```

### 步骤 B：查看
- 结构 HTML：`/data/ruolin/SimScale/outputs/diffusiondrive_model_structure.html`

### 页面内容
① 端到端总管线 ② Backbone（图像 resnet34 + LiDAR 可学习 latent + FPN）
③ GPT 融合（4 尺度×2 Block = 8 个 `encoder_selfatt`，token 320 = 图256+lidar64）
④ BEV 解码（`_tf_decoder` 3 层 8 头，query 31=1 ego+30 agents）
⑤ 扩散轨迹头（DDIM 2 步 × diff_decoder 2 层，cross_bev/agent/ego 逐个列维度）
⑥ 输出头 ⑦ **全部 Attention 层维度总表**（含缓存覆盖）⑧ 为何只有 encoder 能叠 RGB。

关键口径：`cross_bev` 的 “8” 是**轨迹自身 8 个采样点**（grid_sample），非 8 个 BEV token；
缓存里 `n=2` 是 **DDIM 2 步**，非 2 层。

---

## 11. Decoder cross-attention 的 BEV 空间可视化

`cross_bev` / `cross_agent` 的 key 在**车体 BEV 坐标**（ego 米），不在相机像素空间，
故画在俯视图上才准确（不能直接叠到拼接全景 RGB）。

### 步骤 A：抓取 BEV 几何 + cross-attention
```bash
cd /data/ruolin/SimScale
conda activate simscale
export NAVSIM_DEVKIT_ROOT="/data/ruolin/SimScale" CUDA_VISIBLE_DEVICES=0
python /data/ruolin/SimScale/scripts/_capture_bev_geometry.py   # -> outputs/analysis_cache_encoder/bev_geometry.npz
```
抓取内容（2 token × origin/transfer，取最后一层 `diff_layer1`）：
`poses_reg(20,8,3)` / `cross_bev(20,8)` / `cross_agent(20,30)` / `agent_states(30,5)` / `agent_labels(30)`。

### 步骤 B：绘图
```bash
python /data/ruolin/SimScale/scripts/build_bev_cross_attention_viz.py
```

### 步骤 C：查看输出
- HTML：`/data/ruolin/SimScale/outputs/bev_cross_attention_viz.html`

### 页面内容
每个 token 2×2（左 origin / 右 transfer）：
- 上排 **cross_agent**：30 个 agent slot 位置，颜色/大小=对 20 模式平均的关注度，青框=有效检测(sigmoid>0.5)，标注 top-3；
- 下排 **cross_bev**：20 条候选轨迹 + 每条 8 个未来采样点，颜色=grid-sample 关注权重；
- 附文字：top1 关注 agent 的 origin→transfer 迁移、关注熵、JS 散度、有效检测数。
