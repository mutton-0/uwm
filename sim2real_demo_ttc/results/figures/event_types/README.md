# 挖矿事件类型图库
每个子目录 = 一个事件类型，取自 `configs/n1_d2.yaml` 的 `mining.event_*`。
图为该事件的 **ghost 帧**（危险帧），绿框=正例类 / 红框=负例·对照类，框内是该事件的目标。

| 类型 | 含义 | 角色 | 判据 | 库内总数 | 本次抽样 |
| --- | --- | --- | --- | --- | --- |
| **D2b** | distant large target | GEOMETRY-BALANCED NEG | top 3 by apparent area -- deliberately geometry-matched to positives | 1680 | 6 |
| **D** | harmless appearance | NEGATIVE | never enters corridor AND own TTC > 6s throughout (max 2 per scene) | 1423 | 6 |
| **D2a** | parked / stationary vehicle | GEOMETRY-BALANCED NEG | large and central target, but harmless by state (parked/stopped) | 1003 | 6 |
| **D2c** | in corridor but not urgent | FALSIFICATION CONTROL | enters corridor but TTC > 8s throughout | 906 | 6 |
| **D2aP** | D2a placebo | CONTROL | window shifted 2.5s earlier so both frames precede corridor entry | 823 | 6 |
| **C** | generic TTC drop | POSITIVE | TTC drops >= 1.5s within 1s and ends < 5s (catch-all) | 524 | 6 |
| **A** | VRU emergence | POSITIVE | VRU enters corridor, TTC < 5s (1.0s window) | 288 | 6 |
| **D2ctxP** | D2c context placebo | CONTROL | window shifted 2.5s earlier | 203 | 6 |
| **B** | close cut-in | POSITIVE | d_long <= 25m, TTC < 6s, lateral velocity toward ego lane -- the lead-brake corpus | 103 | 6 |

> 注：`lat (OLD instantaneous)` 是**旧的瞬时朝向横向偏移**，已证实在弯道下失真（见 `results/lane_path_filter_report_zh.md`）。
