# 轴名对齐记录（旧 R/S/I/C/D → 新 G/F/I/C/D）

> 生成时间：2026-08-29。触发原因：`docs/vla_base_model_selection_protocol.md` §1「命名规则(2026-08-29 定稿)」
> 把五轴字母从 R/S/I/C/D 改为 **G/F/I/C/D**。本文件是执行新工单
> (`docs/four_axis_proof_experiment_workorder.md`) 之前的一次性对齐，目的只有一个：
> **让此前所有产出物里的旧记号能被无歧义地读成新记号，后续报告一律只用新记号。**
> 本文件不改动任何历史文件的内容（历史读数不重跑、不改写），只提供映射表。

---

## 1. 轴字母映射

| 旧字母 | 旧轴名 | **新字母** | **新轴名** | 是否改变定义 | 改名理由（协议 §1） |
|---|---|---|---|---|---|
| R | Readability / 概念可读性 | **G** | Grounding（语义奠基性） | **是（实质性收窄）** | "可读性"与新定义方向性矛盾：一个被 shortcut 驱动的表征可能"可读性"很高但奠基错误。新定义要求表征因果地奠基于任务相关实体 |
| S | Steering / Steering 响应性 | **F** | Faithfulness（感知-动作忠实度） | 否（操作含义保留） | 避免与三分 split 记号 $S_{dir}/S_{sel}/S_{test}$ 混淆 |
| I | 域不变性 | **I** | Invariance（域不变性） | 否 | 字母不变 |
| C | 失效集中度 | **C** | Concentration（失效集中度） | 否 | Localizability 的 "L" 与层号记号 $L/L^*$ 冲突，保留 C |
| D | 行为倾向校准度 | **D** | Disposition（行为倾向） | 否（但已降级） | 明确不在 G→F→I→C 核心因果链上，仅探索性附加维度 |

**一句话记法**：**R→G、S→F，I/C/D 字母不变**；其中只有 R→G 伴随定义收窄，其余为纯改名。

---

## 2. 我历史产出物的轴归属（"我历史上的 X = 新命名的 Y"）

| 历史产出物 | 文件 | 历史里的称呼 | **新命名下属于哪个轴** | 说明 |
|---|---|---|---|---|
| $v_{hazard}$（vision_mean / last_token 版） | `results/v_hazard_*.npy` | "感知端危险方向"、"R 轴方向" | **G 轴实例**（nuScenes 版，已知落在证伪地板上） | Tier-M 主线产出 |
| $v_{hazard}^{clean}$ | 本轮 Stage B1 产出 | 计划中的"清白感知方向" | **G 轴实例**（region_mean + 中性 prompt + D2a 负例） | 本轮新增 |
| $v_{hazard}^{carla}$ | 本轮 Stage B1 `--arm carla` 产出 | M1 的 "Hcar vs Ncar 方向" | **G 轴实例**（CARLA in-domain，唯一已知有信号者） | 本轮新增 |
| $v_{brake}$ | `variants/n1_d2/results/v_brake_query_mean.npy` | "行为定义方向"、"S 轴锚" | **F 轴实例**（已过 W1 steering 四件套） | T1-Q 产出 |
| $v_{brake}^{obs}$ | 本轮 T-F 产出 | — | **F 轴实例**（观测法，TTC/a_brake 梯度回归） | 新工单 T-F |
| $v_{danger}^{lang}$ | `variants/n1_d2/.../v_danger_lang_query_mean.npy` | "语言概念方向" | **G 轴实例（语言通道版）** | 与 F 轴解耦是 T1-L 的核心负结果 |
| $v_{weather}^{lang}$ | `.../v_weather_lang_query_mean.npy` | "天气语言方向" | **对照方向**，不属任何轴（用于检验语言方向提取管线本身） | — |
| $v_{domain}$ | 本轮 B6 产出 | 计划中"域方向"（此前未提取） | **I 轴所需输入** | 本轮新增 |
| 逐层 recovery（DiffusionDrive patching） | `outputs/ghosthead_infer/patching/recovery.csv` | "因果定位/激活修补" | **C 轴读数原料** | 已存在，T-C 直接整理 |

### 2.1 需要特别注意的"次级记号"

`scripts/n1_readout.py` / `results/n1_report.md` / `results/m1_report.md` 里出现的 **R① / R② / R③**
**不是旧的轴字母 R**，而是旧 G 轴（当时叫 R 轴）内部的三项组合读数（协议 §3② 修订条 ③）：

| 旧记号 | 含义 | 新记号（本轮起统一使用） |
|---|---|---|
| R① | 类别特异性（同几何：VRU vs 静物） | **G-① 类别特异性** |
| R② | 上下文调制（同目标：走廊内 vs 走廊外） | **G-② 上下文调制** |
| R③ | 行为联动（投影 → 行为响应） | **G-③ 行为联动** |

历史文件中的 R①/R②/R③ 一律按上表读作 G-①/G-②/G-③；**历史文件不做批量替换**（改写既往报告会破坏
与既有 JSON 读数的可追溯对应），只在本轮及以后的新报告里使用新记号。

---

## 3. 本轮起生效的书写纪律

1. 新报告（中英双语）一律只用 **G/F/I/C/D**，不出现 R/S 轴字母；
2. 引用历史读数时，写成"（历史文件中记为 R①，即本报告的 G-①）"这种带括号注解的形式，**不静默替换**；
3. 英文报告中轴名统一写作 **G (Grounding) / F (Faithfulness) / I (Invariance) / C (Concentration) / D (Disposition)**；
4. 任何新场景类型复用同一套五轴，**不为新场景发明新轴名**（协议 §1 纪律①）。
