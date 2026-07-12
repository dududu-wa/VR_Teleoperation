# Selective-Walk 抗扰退化因果归因实验设计

日期：2026-07-12
状态：已确认设计，待实施计划
范围：`E:\codebase\VR_Teleoperation`

## 1. 研究问题

Jul11 `selective_walk_profile_teacher_retention_disturb100_profile_guard_recovery` 从正确的 Jul08_12 `model_16000.pt` 恢复训练，严格 profile gate 也按配置工作，但 `model_20000.pt` 的 forced-disturbance robustness 显著退化：forced `0.925` 的平均 task return / fall rate 从源 checkpoint 的 `29.55 / 0.092` 变为 `21.00 / 0.248`，forced `1.0` 从 `22.26 / 0.208` 变为 `10.02 / 0.462`。重点增强的 `stand` 和 `jump` 反而成为最严重的失败项，且诊断仍由 `contact:base_link` 主导。

Jul11 相对 Jul08_12 同时改变了多个因素：

1. `commands.resampling_time` 从继承默认值 `10s` 改为 `30s`；
2. `stand/jump` profile 权重分别提高到 `0.25`；
3. staged schedule 从 `[0.75, 0.85, 0.925, 1.0]` 改为 `[0.925, 0.95, 0.975, 1.0]`，同时改变 gate 阈值和窗口；
4. 监控范围从七 profile aggregate 改为 `stand/jump`，并启用 strict per-profile gate。

因此，Jul11 结果只能证明这组联合干预失败，不能识别哪一个因素导致退化。本设计优先进行因果归因，不以一次训练直接产生更强 checkpoint 为目标。

## 2. 设计原则与理论依据

- 使用共同 warm-start checkpoint 和 one-factor-at-a-time screening，降低初始化差异并给每个实验明确的因果解释。
- 保留 continuation control，用来测量单纯继续 PPO fine-tune 的遗忘与随机漂移；否则无法把所有变化都归因于新增配置。
- 第一阶段只筛查可独立操纵的因素。strict per-profile gate 受现有代码契约约束，要求 `resampling_time` 大于最大 episode 长度，因此不能在 `10s` resampling 下单独打开。
- 先进行无扰动 preservation gate，再进行 forced-disturbance 测试，避免把已经退化的基础策略误解释为抗扰动问题。
- 主要结论使用 terminal checkpoint；best-task checkpoint 仅作为次要参考，避免基于训练窗口最大值产生选择偏差。
- 单 seed screening 只识别大效应；发现大效应后再增加 seed 复现。该策略符合序贯实验设计中先筛查、后确认的资源分配思想。

训练机制依据沿用项目现有配置引用：PPO 使用 Schulman et al. (2017)，teacher-policy retention 依据 Li & Hoiem (2016)，课程学习依据 Bengio et al. (2009)，domain randomization 依据 Tobin et al. (2017) 与 OpenAI et al. (2019)。本轮不新增 reward、网络或算法机制。

## 3. 共同训练合同

所有第一阶段实验必须从同一 checkpoint 开始：

```text
logs/r2_amp/Jul08_12/Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe/model_16000.pt
```

共同条件：

| 项目 | 固定值 |
|---|---|
| task | `r2amp` |
| training seed | `0` |
| resume iteration | `16000` |
| additional iterations | `2000` |
| expected terminal | `model_18000.pt` |
| save interval | `250` |
| teacher retention | `0.25` |
| AMP style weight | `0.0` |
| PPO / reward / command anchors | 与 Jul08_12 原配置一致 |

每个 JSON 都必须写入 `train.runner.max_iterations=2000`，正式命令同时显式传 `--max_iterations=2000`。训练启动后必须先检查 `train.log` 第一条 `Loading model from`；路径不指向上述 Jul08_12 `model_16000.pt` 时立即停止，该 run 不进入比较。

## 4. 第一阶段：四组单变量筛查

### 4.1 C0：continuation control

建议配置名：

```text
configs/ablation/selective_walk_disturb100_causal_control.json
```

相对 Jul08_12 原配置不改变环境和算法行为，只改变自描述的 run name、notes、追加预算与保存间隔。该组测量从 `model_16000.pt` 继续 2000 iterations 本身造成的漂移。

### 4.2 H：hold-only

建议配置名：

```text
configs/ablation/selective_walk_disturb100_hold30_only.json
```

唯一行为变化：

```json
{"env": {"commands": {"resampling_time": 30.0}}}
```

profile 权重、staged schedule、monitor profiles 和 aggregate gate 均保持 Jul08_12 原值。该组测量整回合固定一个 command profile 是否降低策略对扰动时序和命令切换的适应能力。

### 4.3 W：weights-only

建议配置名：

```text
configs/ablation/selective_walk_disturb100_stand_jump_weights_only.json
```

唯一行为变化是采用 Jul11 profile 权重：

| profile | weight |
|---|---:|
| stand | 0.25 |
| walk_slow | 0.10 |
| walk_fast | 0.12 |
| run | 0.12 |
| jump | 0.25 |
| turn_left | 0.08 |
| strafe_right | 0.08 |

`resampling_time` 继续使用 `10s`，staged schedule 和 aggregate gate 保持 Jul08_12 原值。该组测量针对 `stand/jump` 的过采样是否造成其他 profile 遗忘，或使这两个 profile 对特定扰动分布过拟合。

### 4.4 S：schedule-only

建议配置名：

```text
configs/ablation/selective_walk_disturb100_high_start_schedule_only.json
```

唯一行为变化是采用 Jul11 staged schedule 参数：

```text
stage_levels             = [0.925, 0.95, 0.975, 1.0]
stage_min_episodes       = 1024
stage_min_task_return    = [18.0, 20.0, 22.0, 24.0]
stage_max_fall_rate      = [0.20, 0.16, 0.12, 0.10]
stage_regress_patience   = 2
```

仍使用原始 profile 权重、`10s` resampling、原始七 profile monitor scope 和 aggregate gate；`stage_require_all_monitor_profiles=false`。该组测量高强度 `0.925` 起步和更细 schedule 是否导致 catastrophic robustness forgetting。

## 5. 第一阶段评估合同

评估统一使用 WSL CPU PhysX / CPU policy：

| 项目 | 固定值 |
|---|---|
| evaluation seed | `1` |
| presets | 默认 full7 |
| num envs | `64` |
| episodes per preset | `64` |
| episode length | `10s` |
| DTW | 关闭 |

### 5.1 无扰动 preservation gate

四组均评估 `model_best_task.pt` 和 `model_18000.pt`。主要 gate 使用 terminal checkpoint，并要求同时满足：

- full7 平均 task return `>= 30`；
- full7 平均 fall rate `<= 0.15`；
- 任一 preset fall rate `<= 0.35`。

失败组停止后续 forced-disturbance 评估，结论记为“该单变量已经破坏基础能力”。

### 5.2 Forced `0.925`

通过 preservation gate 的 terminal checkpoint 执行 forced `0.925` full7。H/W/S 均与 C0 做配对比较：

- 若相对 C0，平均 task return 下降至少 `5`，且平均 fall rate 上升至少 `0.10`，标记为“明显有害”；
- 若两个指标以相同幅度向好，标记为“候选有益”；
- 其余标记为“screening 未观察到大效应”，不得写成“证明无影响”。

### 5.3 Forced `1.0`

仅当 forced `0.925` 同时满足平均 task return `>= 27`、平均 fall rate `<= 0.15` 时运行 forced `1.0` full7。

任一 forced 测试中，若某 preset fall rate `>= 0.50`，只对失败 preset 追加：

```text
--record_termination_reasons
--record_state_trace
--state_trace_window_steps=50
```

诊断必须按 preset 汇总 `contact:base_link`、orientation、timeout、terminal base z 和 contact force；不能只报告 aggregate fall rate。

## 6. 复现与统计解释

第一阶段使用 seed `0` 进行大效应筛查。如果 H、W 或 S 被标记为“明显有害”或“候选有益”，增加训练 seed `1` 和 `2`，但只复现“对应变量 + 同 seed C0”配对，不重复整个四组矩阵。也就是说，每个新增 seed 都必须同时训练一条 continuation control，不能把 seed `1/2` 的处理组直接与 seed `0` 的 C0 比较。

正式结论要求：

- 至少两个训练 seed 相对各自 C0 对照呈相同方向；
- 报告每个 seed 的原始指标和均值，不只报告跨 seed 平均；
- 单 seed 结果使用“候选机制”措辞，不使用“已证明”。

本设计以大效应和方向复现为决策依据，不在仅三个 seed 的条件下声称精确估计总体方差或统计显著性。

## 7. 第二阶段：strict-gate 因果拆分

第二阶段仅在以下任一条件成立时启动：

1. 第一阶段没有识别出明显有害因素；或
2. hold-only H 保持健康，需要继续判断 strict gate 本身是否有害。

strict gate 不能直接与 C0 比较，因为它必须搭配 `30s` profile hold。为避免把 monitor scope 与 strictness 混在一起，第二阶段增加两个从 Jul08_12 `model_16000.pt` 开始的 2000-iteration 配对组：

### T：target-monitor aggregate control

- `resampling_time=30s`；
- 原始 profile 权重；
- 原始 staged schedule；
- `stage_monitor_profiles=["stand", "jump"]`；
- `stage_require_all_monitor_profiles=false`。

### G：target-monitor strict gate

除下列字段外与 T 完全一致：

```text
stage_require_all_monitor_profiles=true
```

比较关系：

- H vs T：识别“monitor scope 从 full7 改为 stand/jump aggregate”的影响；
- T vs G：识别 strict per-profile pass requirement 的增量影响。

T/G 使用与第一阶段相同的训练预算、preservation gate、forced `0.925` 和条件式 forced `1.0` 协议。

## 8. 决策树

1. 若 C0 自身明显退化：暂停 H/W/S 机制结论；先把问题归类为 continuation forgetting，并评估是否需要更强 retention 或更短预算。
2. 若仅 H 退化：`30s` 整回合 hold 是首要嫌疑；后续 strict gate 设计必须寻找不依赖长 hold 的 episode attribution 机制，不能继续复用 Jul11 方案。
3. 若仅 W 退化：恢复原 profile 权重；后续可考虑 loss reweighting 或按失败率采样，但不能直接提高 episode sampling proportion。
4. 若仅 S 退化：恢复 Jul08_12 schedule，从 `0.75/0.85` 逐级进入高扰动；不再从 `0.925` 冷启动 continuation。
5. 若多个因素退化：按单变量效应大小排序，先移除最大有害因素；不立即做交互组合。
6. 若 H/W/S 均健康：执行第二阶段 T/G，检查 monitor scope 和 strict gate；Jul11 失败更可能来自交互效应而非单个主效应。
7. 无论哪组训练窗口指标更高，只要 forced-disturbance 指标没有改善，就不晋升为新主 checkpoint。

## 9. 实施产物与验证

实施阶段预计新增四个第一阶段 JSON；第二阶段 JSON 仅在触发条件满足后创建。每个 JSON 必须：

- 包含明确的 hypothesis、唯一变量、resume source、追加预算和预期 terminal checkpoint；
- 通过 `python -m json.tool`；
- 在 `tests/test_amp_training_contracts.py` 中增加 focused semantic contract，验证共同字段与各组唯一差异；
- 更新 `CODE_STRUCTURE.md` 中相关配置职责；
- 在 `docs/experiments/r2_amp_experiment_progress.md` 标记为 `not trained`，训练后再补真实 run/checkpoint/metrics；
- 完成配置与测试修改后同步 codegraph。

建议输出目录使用稳定标签而不是训练日期猜测，例如：

```text
outputs/eval/causal_C0_18000_baseline_full7
outputs/eval/causal_H_18000_baseline_full7
outputs/eval/causal_W_18000_baseline_full7
outputs/eval/causal_S_18000_baseline_full7
outputs/eval/causal_<arm>_18000_full7_disturb0925
outputs/eval/causal_<arm>_18000_full7_disturb100
```

## 10. 明确不做的事项

- 不从 Jul11 `model_20000.pt` 继续训练；
- 不修改 reward scale、termination threshold、network architecture 或 AMP style weight；
- 不在第一阶段启用 strict gate；
- 不把 train-window task reward 当作 robustness 结论；
- 不在 baseline 已失败时继续 forced-disturbance sweep；
- 不因单 seed 小幅差异提出新机制。
