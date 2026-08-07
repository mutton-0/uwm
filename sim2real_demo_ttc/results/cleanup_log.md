# Cleanup Log —— guide v3 §0 执行记录

> 日期：2026-08-05
> 依据：`docs/demo_guide_v3_repe_mainline.md` §0
> **执行口径：只归档，不删除、不回退**（用户决定）。guide 原文的 `rm` 与 `git reset --hard`
> 未执行——M1 的 in-domain 结论（"读数方法没病"）是有效资产，删掉 commit 会一并丢掉它。

## 1. 停止 CARLA 相关工作 ✅

| 对象 | 处理 |
|---|---|
| CARLA 服务端 `/opt/carla-0.9.13`（pid 2119994，05:15 起跑，占 GPU0 3.4 GB / CPU 121%） | `kill -9` 停止 |
| N2-sim 采集（`scripts/n2sim_collect.py`） | 停止，不再续跑（最后一轮已完成 140 次采集） |

`/opt/carla-0.9.13` 是系统目录、非本项目安装，**未移动**（可能有他人共用）。

## 2. 数据归档（mv，未删除） ✅

```
/data/ruolin/n2sim        1.6 G  ->  /data/ruolin/_parked/n2sim
/data/ruolin/n2sim_test    43 M  ->  /data/ruolin/_parked/n2sim_test
/data/ruolin/carla-0.9.15  15 G  ->  /data/ruolin/_parked/carla-0.9.15
```

合计释放 ~16.6 G（`/data` 余量 5.5 T，本来也不紧张；归档是为了口径干净，不是为了空间）。
**全部可原地 mv 回来。**

未移动、继续保留的 CARLA 相关资产：

- `/data/ruolin/simlingo_carla`（81 G）—— 这是 **SimLingo 官方 CARLA 训练数据**，
  不是我们捏造的场景；M1 的 in-domain 阳性对照建立在它上面，属 §0"保留资产"。
- `variants/m1_carla/`（mining + 3373 事件 cache + results）—— M1 结论的原始产物。

## 3. 代码归档 ✅

```
git branch park/carla-sideline    # 指向 11c444a，CARLA 侧线全部 commit 就地存档
```

主线**未回退**，`park/carla-sideline` 与主线当前同点；日后若真要回退，commit 已有独立分支托住。

工作区里仍未入库的 CARLA 侧线文件（保持原样，未删）：

```
sim2real_demo_ttc/scripts/n2sim_collect.py
sim2real_demo_ttc/results/n2sim_pairing_check.png
sim2real_demo_ttc/results/n2sim_category_match.png
```

## 4. 保留资产清单（guide §0 要求的"全部继续用，不重跑"）

| 资产 | 位置 | 状态 |
|---|---|---|
| 事件挖掘管线 | `scripts/g1_mine_events.py` + `configs/n1_d2.yaml` | ✅ 在用（T1 重跑） |
| G2 缓存管线 | `scripts/g2_cache.py` | ✅ 在用（T1 重跑，升 v2 五向量 schema） |
| V1 / Tier-M / q_audit 结论 | `results/q_audit.md`、`results/tier_m_*.md` | ✅ 存档，纪律继承 |
| N1 D2 负例设计 + 证伪控制 | `results/n1_report.md` | ✅ T1 直接沿用 |
| M1 in-domain 阳性对照 | `results/m1_report.md`、`variants/m1_carla/` | ✅ 保留 |
| 模型解耦接口 | `scripts/simlingo_runner.py` | ✅ 在用（T2 在其上加注入钩子） |
