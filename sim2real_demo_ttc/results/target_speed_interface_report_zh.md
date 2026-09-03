# 调查报告：六个候选是否存在"可控目标速度"输入通道

> 工单：2026-09-03（插队，纯调查）。**未改任何代码、未跑任何推理。**
> 问题：各候选的输入接口里，有没有一个**独立于当前观测自车速度**、可外部注入的
> 期望/目标速度？接口能塞 ≠ 训练见过，两者分别求证。

## 0. 结论表

| 候选 | 接口是否存在独立目标速度通道 | 证据（文件:行号） | 训练分布见过这类条件 |
| --- | --- | --- | --- |
| **SimLingo** | **是** | `scripts/simlingo_runner.py:390,409-411`（`context` 形参插入自由文本）；模板见下 | **是**（强证据） |
| **DiffusionDrive** | **否** | `results/diffusiondrive_g1_adapter/dd_adapter.py:82-86` | 否 |
| **LTF** | **否** | `results/ltf_g1_adapter/ltf_adapter.py:27`（复用 DD） | 否 |
| **DiffusionDriveV2** | **否** | `results/ddv2_g1_adapter/ddv2_adapter.py:35`（复用 DD） | 否 |
| **Alpamayo-R1** | **否**（接口不暴露文本通道） | `scripts/alpamayo_runner.py:124`；`helper.py:28-57` | 不确定（无正面证据） |
| **AutoVLA** | **接口是，语义否** | `results/autovla_g1_adapter/autovla_adapter.py:205`；`models/autovla.py:629` | **否**（词表纯方向性） |

**一句话**：**只有 SimLingo 这条路真正走得通**；AutoVLA 接口能塞但训练没见过速度数值；
TransFuser 三兄弟根本没有这个字段；Alpamayo 连文本通道都没暴露。

---

## 1. SimLingo —— 通，且训练时明确用过

### 1.1 先更正工单里的一个前提

工单说"语言指令现在是写死的 `Command: follow the road.`"。**实际不是。**

`scripts/simlingo_runner.py:400-405`：

```python
if mcfg["prompt_mode"] in ("target_point", "target_point_command"):
    prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
elif mcfg["prompt_mode"] == "command":
    prompt_tp = mcfg["fixed_command"]
```

而 `configs/n1_d2.yaml:81` 是 `prompt_mode: target_point_command` ⇒ 走**第一个分支**，
`fixed_command`（`configs/n1_d2.yaml:82`）**根本没被用上**。实际 prompt 是：

```
Current speed: {speed} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints.
```

（`simlingo_runner.py:409-411`，`use_cot=False` 时取后一支）

### 1.2 接口：`context` 形参可注入任意文本

`scripts/simlingo_runner.py:390`：

```python
def build_prompt(self, speed_mps: float, n_patches: int, context: str = ""):
```

`:406-411`：

```python
ctx = f"{context.strip()} " if context and context.strip() else ""
prompt = f"Current speed: {speed} m/s. {ctx}{prompt_tp} Predict the waypoints."
```

该形参已由 `build_driving_input`（`:440`）与 `infer`（`:472-474`）一路透传，**无需改动即可用**。

### 1.3 另有一个**已实测**的数值通道：`prompt_speed`

`scripts/simlingo_runner.py:472-482`：

```python
def infer(self, img_rgb, speed_mps, pool_modes=(...), prompt_speed: Optional[float] = None, ...):
    """prompt_speed 非 None 时，prompt 里写的速度与该帧真实 ego 速度解耦。"""
```

注释里记着**已测出的敏感度**：`d(指令速度)/d(prompt速度) = 0.725`
（`simlingo_runner.py:480`）。即 prompt 里的速度数字每变 1 m/s，输出指令速度变 0.725 m/s。

> **注意语义**：`prompt_speed` 写的是"**当前**速度"，不是"目标速度"。
> 它证明模型对 prompt 里的速度数值高度敏感，但它本身**不是**目标速度通道。
> 真正的目标速度要走 §1.2 的 `context`。

### 1.4 训练分布：**明确训练过**，证据强

SimLingo 的 dreamer（指令跟随）子集里有独立的 `target_speed` 模式。

`/data/ruolin/simlingo/dataset_generation/dreamer_data/dreamer_instructions.py:177`：

```
mode (str): The mode of the instruction, e.g., 'lane_change', 'faster', 'slower',
            'stop', 'target_speed', 'crash'.
```

`:394-405` 生成指令文本，**km/h 与 m/s 各半**：

```python
elif 'target_speed' in mode:
    instruction = random.choice(dreamer_templates['target_speed'])
    target_speed_ms = info['target_speed']
    target_speed_kmh = round(target_speed_ms * 3.6, 1)
    if random.random() < 0.5:
        instruction = instruction.replace('<TARGET_SPEED>', f'{str(target_speed_kmh)} km/h')
    else:
        instruction = instruction.replace('<TARGET_SPEED>', f'{str(target_speed_ms)} m/s')
```

模板取自 `/data/ruolin/simlingo/data/augmented_templates/dreamer.json` 的 `target_speed` 键，
**共 20 条**，前 8 条：

```
Drive at <TARGET_SPEED>.          Aim for <TARGET_SPEED> speed.
Maintain a speed of <TARGET_SPEED>.   Keep your speed at <TARGET_SPEED>.
Target a speed of <TARGET_SPEED>.     Drive with a steady <TARGET_SPEED>.
Set your speed to <TARGET_SPEED>.     Try to reach <TARGET_SPEED>.
```

同文件另有 `faster` / `slower` / `stop_now` / `walker` / `redlight` 等 19 个模板类别。

### 1.5 训练时对"行人临近"有专门的拒绝逻辑 —— 与本项目直接相关

`dreamer_instructions.py:70-73`：

```python
if walker_close and new_speed > current_measurement['speed']:
    safe_to_execute = False
    dreamer_answer = 'Ignore instruction as it might lead to a dangerous situation because of the pedestrian. Waypoints:'
elif walker_close and new_speed < current_measurement['speed']:
    safe_to_execute = True
```

`:97-101` 对 `faster` / `slower` 同理。

> **这条对 F-3 有直接价值**：SimLingo 被**显式训练**成"行人近时拒绝加速指令、接受减速指令"。
> 于是可以构造一个**远比遮挡更强的探针**：给定"加速到 X"的指令，
> 看模型是否因为画面里的行人而拒绝执行。这把"模型有没有用到这个行人"
> 从**被动读数差**变成**主动的指令服从/拒绝二分**，信噪比高得多。

### 1.6 但有两处语序偏差，必须记

训练时的 dreamer prompt（`simlingo_training/dataloader/dataset_dreamer.py:129-131`）：

```python
if random.random() < 0.8:
    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {chosen_option['dreamer_instruction']}"
else:
    prompt = f"Current speed: {speed_rounded} m/s. {chosen_option['dreamer_instruction']}"
```

对比我们的 runner（`simlingo_runner.py:411`）：

```python
prompt = f"Current speed: {speed} m/s. {ctx}{prompt_tp} Predict the waypoints."
```

| 差异 | 训练 | 我们的 runner |
| --- | --- | --- |
| 指令位置 | 在 target 之**后** | `context` 在 target 之**前** |
| 结尾 | dreamer 分支**无** `Predict the waypoints.` | 有 |

普通驾驶分支（`dataset_driving.py:259`）确实是
`f"Current speed: ... {target_options} Predict the waypoints."`，与我们的空 context 情形逐字一致；
但**一旦注入指令，就应改用 dreamer 的语序**（指令放在 target 之后、去掉结尾句），
否则是训练分布外的写法。**这是一处需要改代码的地方，本轮未改。**

---

## 2. TransFuser 三兄弟（DiffusionDrive / LTF / DiffusionDriveV2）—— 不通

`results/diffusiondrive_g1_adapter/dd_adapter.py:82-86`：

```python
def status_feature(speed_mps, accel_mps2=0.0):
    """[1,8] = driving_command(4) + v(2) + a(2)。nuScenes 只有纵向速率,横向置 0。"""
    v = np.array([float(speed_mps), 0.0], dtype=np.float32)
    a = np.array([float(accel_mps2), 0.0], dtype=np.float32)
    return torch.from_numpy(np.concatenate([DRIVING_COMMAND, v, a])).float().unsqueeze(0)
```

`DRIVING_COMMAND` 定义在 `/data/ruolin/uwm/scripts/ghosthead_infer/run_ghosthead_infer.py:47`：

```python
DRIVING_COMMAND = np.array([0, 1, 0, 0], dtype=np.float32)  # 直行 one-hot(假设)
```

LTF 与 DDv2 **原样复用**该函数：
`results/ltf_g1_adapter/ltf_adapter.py:27` 与 `results/ddv2_g1_adapter/ddv2_adapter.py:35`
均为 `status_feature = DD.status_feature`。

### 与官方实现对齐核验

`/data/ruolin/uwm/navsim/agents/transfuser/transfuser_features.py:45-49`：

```python
features["status_feature"] = torch.concatenate([
    torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
    torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
    torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
])
```

`EgoStatus` 数据类（`/data/ruolin/uwm/navsim/common/dataclasses.py:138-145`）字段为
`ego_pose / ego_velocity / ego_acceleration / driving_command`。

> **我们的 adapter 与官方逐字段一致，不是简化版。** 整个 8 维状态里：
> * `driving_command(4)` = 方向性 one-hot（左/直/右/未知），**无速度语义**；
> * `ego_velocity(2)` = **实测**自车速度，是**观测量**；
> * `ego_acceleration(2)` = 实测加速度，同为观测量。
>
> **没有任何字段具备"期望速度"语义。**

**"往 `ego_velocity` 槽里塞假值"不算一条出路**：那是谎报当前状态，
不是下达目标；模型会把它当成"我现在开这么快"，与"我希望开这么快"是不同的条件。
两者在训练分布里对应完全不同的样本。**这条路对这三个候选不通。**

---

## 3. Alpamayo-R1 —— 接口未暴露文本通道

`results/alpamayo_g1_adapter/alpa_patch.py:135`（`run_from_data`）与 `:161`（`run`），
以及 `scripts/alpamayo_runner.py:124`（`infer`），三处调用完全相同：

```python
messages = self.r.helper.create_message(data["image_frames"].flatten(0, 1))
```

上游 `/data/Zhengyang/alpamayo/src/alpamayo_r1/helper.py:28`：

```python
def create_message(frames: torch.Tensor):
    """Construct the message using images and cot."""
```

**形参只有 `frames`，没有任何文本/指令入口。** 用户文本是硬编码的（`helper.py:52-55`）：

```python
{"type": "text",
 "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."}
```

模型的三路输入是：`tokenized_data`（由图像+固定模板得到）、`ego_history_xyz`、`ego_history_rot`
（`scripts/alpamayo_runner.py:128-132`）——自车信息走 `<|traj_history|>` token 通道，
是**历史轨迹观测**，不是目标。

**结构上可以改**（messages 就是一个 list of dict，追加一段 text 即可），但：

1. `helper.py:32` 的注释写明 `# NOTE: we expand the padding tokens to match training,
   so we can directly apply native processor from VLM.` —— 该模板是**对齐训练**的固定写法；
2. 未找到任何证据表明训练数据里有速度指令（**无正面证据，也无反面证据，故判"不确定"**）。

**如实说：这条路对 Alpamayo 需要改上游代码，且训练是否支持无法从现有代码确认。
本轮不设计变通方案。** 列为待决项。

---

## 4. AutoVLA —— 接口通，但训练词表里没有速度

### 4.1 接口：`driving_command` 是自由文本

`results/autovla_g1_adapter/autovla_adapter.py:199-206`：

```python
def run(self, cam_sd_token, speed_mps, accel=0.0):
    feats = {"images": self.temporal_paths(cam_sd_token), "sensor_data_path": None,
             "vehicle_velocity": [float(speed_mps), 0.0],
             "vehicle_acceleration": [float(accel), 0.0],
             "driving_command": "go straight"}
    inputs = self.model.get_prompt(feats)
```

上游 `/data/ruolin/uwm/external/AutoVLA/models/autovla.py:571`：

```python
instruction = input_features["driving_command"].lower()
```

`:628-629` 拼进 prompt：

```python
f"The current velocity of the vehicle is {velocity:.3f} m/s, and the current acceleration is {acceleration:.3f} m/s². "
f"The driving instruction is: {instruction}. Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds."
```

⇒ **传 `"maintain 30 km/h"` 这类字符串，语法上完全通得过，无需改任何代码。**

### 4.2 但训练分布里没有速度数值

指令取值来自 `/data/ruolin/uwm/external/AutoVLA/dataset_utils/preprocessing/waymo_e2e_dataset.py:78-79`：

```python
intent_map = {0: "unknown", 1: "go straight", 2: "go left", 3: "go right"}
instruction = intent_map.get(frame.intent, "unknown")
```

nuScenes 侧（`tools/preprocessing/nusc_sample_generation.py:225,230`）为
`"turn left"` / `"turn right"`；Waymo 侧另有
`"change lane to left"` / `"change lane to right"`（`waymo_e2e_dataset.py:508-510`）。

> **全部是方向性指令，一条带速度数值的都没有。**

### 4.3 一个替代通道：纵向动作词表

系统 prompt 里（`models/autovla.py:654`）：

```
- **Longitudinal actions** (choose exactly one): [stop, deceleration to zero,
  maintain constant speed, quick deceleration, deceleration, quick acceleration, acceleration]
```

这是模型**输出**端的动作词表（训练过），不是输入端的目标速度。
但它说明模型对"纵向动作"这个概念有明确表示，
**塞入 `"maintain constant speed"` 或 `"deceleration"` 这类词条，比塞数值更贴近训练分布。**
不过这仍是**定性**档位而非可控目标速度，且它出现在系统 prompt 的输出规范里、
不是 `driving_command` 的训练取值 —— **属于推测，未经实验验证，如实标注为不确定。**

---

## 5. 待决项（不设计变通方案）

| 项 | 状态 |
| --- | --- |
| **SimLingo 语序需对齐 dreamer**（指令放 target 之后、去掉 `Predict the waypoints.`） | 需改 `simlingo_runner.py:409-411`，**本轮未改** |
| **Alpamayo 无文本通道** | 需改上游 `helper.py:create_message` 加形参；且训练是否支持速度指令**无法从代码确认** ⇒ **这条路对 Alpamayo 不通**，除非另找证据 |
| **AutoVLA 数值指令属分布外** | 接口通但训练没见过；塞进去可能无效或不可预测。**不建议**当作可控目标速度使用 |
| **TransFuser 三兄弟无此字段** | **这条路对 DiffusionDrive / LTF / DiffusionDriveV2 完全不通**，不存在无损的变通 |
| SimLingo `prompt_speed` 的 0.725 敏感度 | 是**当前速度**通道的实测值，不能直接当目标速度通道的预期效应 |

## 6. 本轮未做

未改任何代码、未跑任何推理、未修改任何既有结果文件。
本报告全部结论均给出文件名与行号，可逐条复核。
