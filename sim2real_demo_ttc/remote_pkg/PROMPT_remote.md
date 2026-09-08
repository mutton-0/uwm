# 任务：因果迁移实验（左舵学轴 → 注入左舵/右舵场景）

## 代码与数据

代码：`git@github.com:mutton-0/uwm.git`，分支 `sim2real/demo-v3-repe-mainline`，
工作目录 `sim2real_demo_ttc/`。先 `git pull`。

**数据在 `/data/ruolin/uwm/sim2real_demo_ttc/` 下，直接读，不要重建**：
`results/vdecel_acts_navsim_test_*.npz`、`results/vbright_acts_navsim_lead_*_night_global.npz`、
`work_c_dep/lead/`。这些在 `.gitignore` 里（约 500 MB），git 上没有。
不是同一目录时 `ln -s` 或 `rsync` 过来。访问不到才跑 `remote_pkg/prepare_data.sh`
重建（要 GPU，约 1 小时）。

图像：`/data/dataset/navsim/dataset/sensor_blobs`（脚本自动跳过不在盘上的场景）。
环境：`/home/mut0/.conda/envs/simscale/bin/python`。

## 判定逻辑 —— 顺序不可颠倒

    第 1 步  左舵注入必须先有效（阳性对照）
             └─ 无效 ⇒ 方法未建立，**立即停止，不要跑右舵**
                      右舵阴性结果不可解读：分不清"轴不驱动行为"与"方法测不出驱动"

    第 2 步  左舵有效后才看右舵
             ├─ 右舵也有效 ⇒ 因果不变（invariance）
             └─ 右舵无效   ⇒ 因果偏移（causal shift）  ← 要找的

**本机第 1 步尚未通过**：dd/v_decel 那格 p=0.800，随机方向斜率中位 0.0075 反而大于
真实方向的 0.0031。可能是本机被其他用户占满（118 GB 显存、两卡 99%），
也可能是方法在 TransFuser 架构上不成立。**第一件事就是在独占卡下确认这一点。**

## 跑法

    cd <repo>/sim2real_demo_ttc
    PY=/home/mut0/.conda/envs/simscale/bin/python DEV=cuda:0 N=60 NR=30 \
      bash remote_pkg/run_causal_transfer.sh 2>&1 | tee /tmp/causal.log

四步：① 左舵速度轴（阳性对照）② 右舵速度轴 ③ I 轴 v_bright 两侧 ④ 分段算子复核。

## 三根轴

| --axis | 方向 | 预期 |
|---|---|---|
| speed | −ŝ，巡航样本对 v0 回归 | **阳性对照**，注入应致**减速** |
| decel | 刹车组 − 巡航组 | 注入应致减速 |
| bright | 原图→变夜的表征差（I 轴） | **注入应致加速**（与另两根相反，cos(v_bright,v_faith)=−0.54） |

方向与层权重**一律只在左舵学**，`--stim-side` 只换被注入的场景。

## 四条判据

C1 斜率符号符合预期且 p<0.05 ｜ C2 α>0 与 α<0 段斜率同号、比值∈[1/3,3]
C3 横偏斜率 ≤ 速度斜率的 1/3 ｜ C4 α≤8 已出现一半以上效应

## 自主执行到结束

1. 跑 `run_causal_transfer.sh`，全程 tee 到日志
2. **第 1 步跑完就检查**：三个模型的左舵速度轴注入若全部 C1 不成立，
   **停下**，报告"方法未建立"，不跑第 2–4 步
3. 至少一个模型第 1 步通过则跑完，再 `python remote_pkg/summarize_causal.py` 出判定表
4. 报错就修；常见坑见 `remote_pkg/README.md`
5. 显存不够降 N，但 NR 不低于 20

## 报告

- 每个 (模型 × 轴 × 算子 × 舵位) 的斜率、随机零分布中位、p、C1–C4
- **最重要的一句：第 1 步过了没有**
- 若出现"左舵有效、右舵无效"，把两侧的 α 阶梯完整贴出
