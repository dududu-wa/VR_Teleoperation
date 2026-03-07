# VR 多步态人形遥操作 -- 实现计划

> 基于已有规划 (`recursive-imagining-willow.md`)，按 7 波次增量实现全部模块。
> **默认采用 Isaac Gym GPU 训练方案**：物理仿真、观测/奖励、rollout 与策略更新常驻 GPU；MuJoCo 仅保留为 sim2sim、部署前验证与对齐后备。

---

## 当前状态

- 项目骨架已创建：目录结构 + `requirements.txt` + stub `__init__.py`（含断裂 import）
- **无任何实现代码**，所有 `.py` 模块、YAML 配置、脚本均为空
- 参考项目可用：FALCON（G1 配置/解耦 PPO/GPU rollout）、HugWBC（MlpAdaptModel/PPO/干预训练/GPU pipeline）、unitree_mujoco（G1 MJCF 模型）

---

## 架构决策

| 项目 | 决策 |
|------|------|
| 物理仿真 | **Isaac Gym GPU tensor pipeline 为默认**；MuJoCo 仅用于 sim2sim、对齐验证与部署前回归 |
| 数据流 | Isaac Gym GPU state tensors → PyTorch GPU policy & PPO → GPU action tensor → Isaac Gym step |
| 环境数量 | 默认 4096-16384 GPU 并行环境；MuJoCo fallback 仅用于 1-64 env 调试 |
| 配置系统 | Hydra + OmegaConf（参考 FALCON 模式） |
| NN 架构 | HugWBC MlpAdaptModel（HistoryEncoder + StateEstimator + LowLevelController） |
| 动作空间 | 策略输出 15-DOF（下肢+腰）；上肢链路采用“手目标位姿→重定向/滤波/IK→14-DOF” |

---

## 实现波次

### Wave 1: 基础设施层
**目标**: 工具函数 + 机器人配置 + YAML 配置体系

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 1 | `vr_teleop/utils/math_utils.py` | FALCON `utils/torch_utils.py` + `utils/math.py` | 替代 `isaacgym.torch_utils`：quat_mul, quat_rotate, quat_rotate_inverse, projected_gravity, wrap_to_pi, euler↔quat |
| 2 | `vr_teleop/utils/config_utils.py` | FALCON `utils/config_utils.py` | OmegaConf resolver 注册 + 配置加载合并 |
| 3 | `vr_teleop/utils/logger.py` | HugWBC `runners/on_policy_runner.py:148-209` | TensorBoard 日志包装 |
| 4 | `vr_teleop/robot/g1_config.py` | FALCON `config/robot/g1/g1_29dof_waist_fakehand.yaml` | G1 29-DOF dataclass：关节名/索引/限位/PD增益/对称索引/默认姿态 |
| 5 | `vr_teleop/utils/symmetry.py` | FALCON yaml `symmetric_dofs_idx` + HugWBC PPO 对称损失 | G1 15-DOF 下肢对称置换矩阵 + 观测置换矩阵 |
| 6 | `configs/base.yaml` | FALCON `config/base.yaml` | Hydra defaults 入口 |
| 7 | `configs/robot/g1.yaml` | FALCON G1 yaml | 关节名/限位/PD增益 |
| 8 | `configs/algo/ppo.yaml` | HugWBC PPO 默认参数 | clip=0.2, gamma=0.99, lam=0.95, lr=1e-4, symmetry_coef=0.5, entropy_coef=0.03 |
| 9 | `configs/obs/g1_obs.yaml` | 规划文档 2.3 节 | actor 单步 58 维；history 采用 proprio 51 维 × H=5（255）；critic ~99 维 |
| 10 | `configs/rewards/g1_rewards.yaml` | 规划文档 2.6 节 | 13 项奖励权重（已修复平衡：alive=1.5, standing_still=-0.5, base_height=-8, torso_orientation=-5） |
| 11 | `configs/env/g1_multigait.yaml` | — | episode_length, decimation=4, sim_dt=0.005, action_scale |
| 12 | `configs/domain_rand/default.yaml` | FALCON `domain_rand_rl_gym.yaml` | 摩擦/质量/PD增益/延迟随机化范围 |
| 13 | `configs/curriculum/phase.yaml` | 规划文档 2.7 节 | Phase 0-4 阈值 + ADR 参数 |
| 14 | `configs/intervention/disturb.yaml` | 规划文档 2.8 节 | 干预参数范围 |
| 15 | 修复所有 `__init__.py` | — | 清空断裂 import，改为延迟导入或空文件 |

**验证**: `python -c "from vr_teleop.utils.math_utils import quat_rotate; print('OK')"` + Hydra 配置加载测试

---

### Wave 2: Isaac Gym 环境核心
**目标**: 以 Isaac Gym 为主的 GPU 向量化环境跑通，MuJoCo 环境仅用于校验

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 16 | `vr_teleop/envs/g1_base_env.py` | unitree_mujoco + HugWBC `h1.py:40-126` | MuJoCo 单实例对齐环境：用于 sim2sim、观测一致性检查、部署前回放 |
| 17 | `vr_teleop/envs/observation.py` | HugWBC `observation_buffer.py` + 规划 2.3 节 | ActorObs(58) 与 CriticObs(~99) 构建 + 历史缓冲；优先直接消费 Isaac Gym GPU tensor |
| 18 | `vr_teleop/envs/reward.py` | HugWBC `h1interrupt.py:430-481` + 规划 2.6 节 | 13 项奖励函数，默认基于 PyTorch GPU tensor 实现 |
| 19 | `vr_teleop/envs/termination.py` | HugWBC `h1interrupt.py:378-385` | 接触终止 + 姿态终止 + 超时，兼容 Isaac Gym batched GPU rollout |
| 20 | `vr_teleop/envs/domain_rand.py` | FALCON `domain_rand_rl_gym.yaml` | 摩擦/质量/PD增益/推力随机化，支持 Isaac Gym 侧批量更新 |
| 21 | `vr_teleop/envs/isaac_vec_env.py` | HugWBC/FALCON env pipeline | **主环境后端**：基于 Isaac Gym tensor API 实现 batched step、reset、reward、termination，obs/action 保持在 GPU |
| 22 | `vr_teleop/envs/mujoco_vec_env.py` | HugWBC `vec_env.py` 接口 | MuJoCo fallback / 对齐验证后端，不作为主训练路径 |

**验证**: IsaacGymVecEnv(4096) step 形状正确、GPU 显存稳定、奖励值有限；MuJoCo 对齐环境与 Isaac Gym 单步统计误差可控

---

### Wave 3: 多步态训练环境
**目标**: 命令采样 + 步态时钟 + 干预接口

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 23 | `vr_teleop/envs/g1_multigait_env.py` | HugWBC `h1interrupt.py` 全文 | 继承 Isaac Gym GPU VecEnv：gait_id 采样(stand/walk/run)，速度命令采样，步态相位时钟(sin/cos)；干预注入采用“手目标位姿→重定向/滤波/IK→14-DOF upper”，再与 15-DOF leg 拼接为 29-DOF；并执行腿优先安全约束与回退逻辑 |

**验证**: stand 命令→零速度 + walk 命令→前进 + 奖励分量符号正确

---

### Wave 4: RL 算法层
**目标**: PPO 训练循环可运行

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 24 | `vr_teleop/agents/networks.py` | HugWBC `net_model.py:104-131` | MlpAdaptModel（不依赖 isaacgym），HistoryEncoder(255→32), StateEstimator(32→3), LowLevelController(93→15)；默认部署在 CUDA |
| 25 | `vr_teleop/agents/actor_critic.py` | HugWBC `actor_critic.py:39-147` | Actor=MlpAdaptModel, Critic=MLP(99→1)，可学习 std；forward/update 全程 GPU |
| 26 | `vr_teleop/agents/rollout_storage.py` | HugWBC `rollout_storage.py` | 经验缓冲 + GAE 计算 + mini-batch 生成器；默认显存缓存 rollout |
| 27 | `vr_teleop/agents/ppo.py` | HugWBC `ppo.py` | PPO + 对称损失（G1 15-DOF 置换矩阵）+ 特权信息重建损失 + adaptive KL lr 调度 |
| 28 | `vr_teleop/agents/runner.py` | HugWBC `on_policy_runner.py:45-226` | 训练循环：Isaac Gym GPU rollout → PPO update → curriculum → log → checkpoint |

**验证**: 100 次迭代 surrogate loss 下降 + 对称矩阵正交自逆 + 梯度有限

---

### Wave 5: 干预系统（组件 A）
**目标**: 上肢干预信号生成 + 可行性过滤

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 29 | `vr_teleop/intervention/intervention_generator.py` | HugWBC `h1interrupt.py:74-109,247-344` | 生成手目标位姿（高斯/均匀/正弦/结构化末端轨迹），duty_mask，transition_stress 放大；支持 batched GPU 生成 |
| 30 | `vr_teleop/intervention/feasibility_filter.py` | 规划 2.9 节 | 关节限位裁剪 + 速度限幅 + 低通滤波 + 回退策略；优先实现 GPU 张量版 |
| 31 | `vr_teleop/intervention/ik_solver.py` | 规划 2.10 节 | 阻尼最小二乘 IK（`mj_jac` 或等价 batched Jacobian）；优先提供 batched GPU 版本 |
| 32 | `vr_teleop/intervention/motion_retarget.py` | FALCON motion_lib | VR 手柄位姿 → G1 上肢关节角映射 |
| 33 | `vr_teleop/intervention/motion_library.py` | — | 可行动作片段存储 + 采样（Phase 4 用） |

**验证**: 干预目标在关节限位内 + 可行性过滤后轨迹平滑 + Phase 1 训练不发散

---

### Wave 6: 课程系统
**目标**: Phase 0-4 自动升级 + 可选 ADR

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 34 | `vr_teleop/curriculum/phase_curriculum.py` | HugWBC `h1interrupt.py:196-232` + 规划 2.7 节 | Phase 0-4 管理：tracking/fall_rate 指标 → 升级阈值 → 干预参数范围扩大 |
| 35 | `vr_teleop/curriculum/adr_scheduler.py` | 规划文档第 8 节 | ADR：SR > τ_high → 加难，SR < τ_low → 保持/降低 |
| 36 | `configs/experiment/phase0_baseline.yaml` ~ `phase4_vr_align.yaml` | — | 各 Phase 实验预设 |

**验证**: 长时间训练中 Phase 自动提升 + ADR 参数随 SR 变化

---

### Wave 7: 训练脚本 + 部署 + 评估
**目标**: 端到端可用

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 37 | `scripts/train.py` | FALCON `train_agent.py` | Hydra 入口：默认创建 Isaac Gym GPU 环境 → Runner.learn() |
| 38 | `scripts/play.py` | HugWBC `sim2sim_deploy.py` | 加载 checkpoint + MuJoCo viewer 可视化回放 |
| 39 | `scripts/eval.py` | — | 加载模型 → 批量跑 episode → 统计 fall_rate / tracking / transition_failure |
| 40 | `vr_teleop/deploy/policy_wrapper.py` | — | 策略推理封装（去掉训练辅助模块） |
| 41 | `vr_teleop/deploy/sim2sim_runner.py` | — | 不同物理参数下策略验证 |
| 42 | `vr_teleop/deploy/dds_bridge.py` | unitree_mujoco `unitree_sdk2py_bridge.py` | lowcmd/lowstate DDS 通信桥接 |
| 43 | `vr_teleop/eval/evaluator.py` | — | 指标计算模块 |
| 44 | `scripts/build_motion_library.py` | — | 构建可行动作库 |

**验证**: `python scripts/train.py sim_backend=isaacgym num_envs=4096 device=cuda:0` 完整运行 → checkpoint 保存 → `play.py` 可视化 → Phase 0 站/走成功率 > 95%

---

## 实现顺序与依赖

```
Wave 1 (无依赖) ──→ Wave 2 (依赖 Wave 1) ──→ Wave 3 (依赖 Wave 2)
                                                      ↓
Wave 4 (依赖 Wave 1, 独立于 Wave 2/3 的网络部分) ←────┘
                                                      ↓
Wave 5 (依赖 Wave 2) ──→ Wave 6 (依赖 Wave 5) ──→ Wave 7 (依赖全部)
```

Wave 4 的网络/PPO 代码可与 Wave 2/3 并行开发，最终在 Wave 7 集成。

---

## 关键参考文件路径

> 使用可配置根目录（如环境变量），避免硬编码本地绝对路径。

| 用途 | 路径 |
|------|------|
| G1 MJCF 模型 | `${UNITREE_MUJOCO_ROOT}/unitree_robots/g1/g1_29dof.xml` |
| G1 场景 | `${UNITREE_MUJOCO_ROOT}/unitree_robots/g1/scene.xml` |
| G1 关节配置 | `${FALCON_ROOT}/humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml` |
| MlpAdaptModel | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/modules/net_model.py` |
| PPO+对称损失 | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/algorithms/ppo.py` |
| ActorCritic | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/modules/actor_critic.py` |
| RolloutStorage | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/storage/rollout_storage.py` |
| OnPolicyRunner | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/runners/on_policy_runner.py` |
| 干预训练环境 | `${HUGWBC_ROOT}/legged_gym/envs/h1/h1interrupt.py` |
| ObservationBuffer | `${HUGWBC_ROOT}/legged_gym/legged_utils/observation_buffer.py` |
| VecEnv 接口 | `${HUGWBC_ROOT}/rsl_rl/rsl_rl/env/vec_env.py` |
| torch_utils | `${FALCON_ROOT}/humanoidverse/utils/torch_utils.py` |
| DDS 桥接 | `${UNITREE_MUJOCO_ROOT}/simulate_python/unitree_sdk2py_bridge.py` |
| 域随机化 | `${FALCON_ROOT}/humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml` |

---

## 端到端验证清单

1. **环境冒烟**: `G1BaseEnv` 零动作站立 1000 步不摔；`IsaacGymVecEnv(num_envs=4096)` 单步输出稳定
2. **向量化**: `IsaacGymVecEnv(num_envs=4096)` step 形状正确，无 NaN；GPU 显存无异常增长
3. **训练冒烟**: `train.py sim_backend=isaacgym num_envs=4096 device=cuda:0` 运行 100 迭代，loss 有限且下降
4. **Phase 0 收敛**: 10k 迭代后 stand 成功率 > 95%，walk tracking > 0.8
5. **干预鲁棒**: Phase 1-3 训练后 tracking > 0.65，transition_failure < 10%
6. **可视化**: `play.py` 加载 checkpoint 显示稳定行走
7. **Sim2Sim**: GPU 训练策略迁移到独立 MuJoCo/Unitree 环境，不同摩擦/阻尼参数下 fall_rate < 5%

---

## Code Sync (2026-03-06)

本项目当前保留两套训练体系：

- `phase`：手工阶段课程（Phase 0-4）。
- `lp_teacher`：LP-Teacher 自动难度采样。

统一入口：

```bash
python scripts/train.py --curriculum-system phase --use-intervention
python scripts/train.py --curriculum-system lp_teacher --use-intervention
```

代码与配置同步点：

- `scripts/train.py` 新增 `--curriculum-system {phase, lp_teacher}`。
- `vr_teleop/curriculum/lp_teacher_curriculum.py` 新增 LP-Teacher 到环境参数的映射层。
- `configs/curriculum/phase.yaml` 保留 `curriculum.system: phase`。
- `configs/curriculum/lp_teacher.yaml` 提供 LP-Teacher 参数模板。

训练建议：

- 要可控、好复现：选 `phase`。
- 要自动探索任务前沿：选 `lp_teacher`。

---

## Code Sync (2026-03-07): 训练失败修复

两次训练（phase / lp_teacher 各 10K 迭代）均 100% 摔倒，根因为奖励权重失衡导致策略学会"立即摔倒"。

### 修复内容

**奖励权重重新平衡** (`configs/rewards/g1_rewards.yaml` + `vr_teleop/envs/reward.py`)：

| 参数 | 修改前 | 修改后 | 说明 |
|------|--------|--------|------|
| `alive` | 0.2 | **1.5** | 增强存活激励 |
| `torso_orientation` | -20.0 | **-5.0** | 降低以容许早期探索 |
| `base_height` | -40.0 | **-8.0** | 降低以容许早期探索 |
| `standing_still` | -10.0 | **-0.5** | 消除"摔倒优于存活"激励 |

**standing_still 归一化**：惩罚按动作/关节维度数归一化，消除 DOF 数量对惩罚幅度的影响。

**熵系数** (`configs/algo/ppo.yaml` + `vr_teleop/agents/ppo.py`)：0.01 → **0.03**，防止噪声崩溃。

**噪声日志修复** (`vr_teleop/agents/runner.py`)：报告 clamp 后实际值而非原始参数。

---

## 代码审查修复 (2026-03-07)

代码审查发现并修复了以下 Critical 与 Medium 级别问题：

### Critical 修复

- **C1**: `_compute_rewards` 终止信号不完整 -- 改为 contact | orientation | height 完整组合 (`g1_multigait_env.py`)
- **C2**: 缺少 `extras['time_outs']` -- PPO 无法对超时做 value bootstrapping (`g1_multigait_env.py`)
- **C3**: 奖励分量仅记录单步快照 -- 改为 episode 级累积 (`g1_multigait_env.py`)
- **C4**: `InterventionGenerator.set_phase()` 未设置 `curriculum_factor` -- 按 phase 映射 (`intervention_generator.py`)
- **C5**: PD 力矩在 decimation 循环外计算（50Hz）-- 移入循环内（200Hz）(`g1_multigait_env.py`)
- **C6**: checkpoint 不保存课程状态 -- 增加课程状态保存/加载 (`runner.py`)
- **C7**: `DomainRandConfig` 字段名不匹配 -- 修正 `eval.py` 和 `sim2sim_runner.py` 中的字段名

### Medium 修复

- **M1**: `dof_vel` 观测噪声 `noise_scale` 1.5 → 0.2 (`observation.py`)
- **M4**: `rewbuffer`/`lenbuffer` 从 `learn()` 移至 `__init__` 以保持跨 chunk 统计 (`runner.py`)
- **M5**: mini-batch 随机排列改为每个 PPO epoch 重新生成 (`rollout_storage.py`)
