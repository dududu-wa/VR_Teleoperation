# VR 多步态人形遥操作 -- 实现计划

> 基于已有规划 (`recursive-imagining-willow.md`)，按 7 波次增量实现全部模块。
> GPU 用于 PyTorch 训练；MuJoCo 物理仿真在 CPU 运行（同步批量），MJX GPU 加速作为未来扩展。

---

## 当前状态

- 项目骨架已创建：目录结构 + `requirements.txt` + stub `__init__.py`（含断裂 import）
- **无任何实现代码**，所有 `.py` 模块、YAML 配置、脚本均为空
- 参考项目可用：FALCON（G1 配置/解耦 PPO）、HugWBC（MlpAdaptModel/PPO/干预训练）、unitree_mujoco（G1 MJCF 模型）

---

## 架构决策

| 项目 | 决策 |
|------|------|
| 物理仿真 | MuJoCo CPU 同步批量（1 进程内 N 个 MjModel/MjData），PyTorch NN 在 GPU |
| 数据流 | CPU numpy → `.to(cuda)` → GPU 训练 → `.cpu().numpy()` → 回写 MuJoCo |
| 环境数量 | 初始 256-2048（CPU 同步），后续可扩展 MJX GPU 4096+ |
| 配置系统 | Hydra + OmegaConf（参考 FALCON 模式） |
| NN 架构 | HugWBC MlpAdaptModel（HistoryEncoder + StateEstimator + LowLevelController） |
| 动作空间 | 策略输出 15-DOF（下肢+腰），上肢 14-DOF 由干预系统驱动 |

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
| 8 | `configs/algo/ppo.yaml` | HugWBC PPO 默认参数 | clip=0.2, gamma=0.99, lam=0.95, lr=1e-4, symmetry_coef=0.5 |
| 9 | `configs/obs/g1_obs.yaml` | 规划文档 2.3 节 | actor 58 维 × H=5 history, critic ~99 维 |
| 10 | `configs/rewards/g1_rewards.yaml` | 规划文档 2.6 节 | 13 项奖励权重 |
| 11 | `configs/env/g1_multigait.yaml` | — | episode_length, decimation=4, sim_dt=0.005, action_scale |
| 12 | `configs/domain_rand/default.yaml` | FALCON `domain_rand_rl_gym.yaml` | 摩擦/质量/PD增益/延迟随机化范围 |
| 13 | `configs/curriculum/phase.yaml` | 规划文档 2.7 节 | Phase 0-4 阈值 + ADR 参数 |
| 14 | `configs/intervention/disturb.yaml` | 规划文档 2.8 节 | 干预参数范围 |
| 15 | 修复所有 `__init__.py` | — | 清空断裂 import，改为延迟导入或空文件 |

**验证**: `python -c "from vr_teleop.utils.math_utils import quat_rotate; print('OK')"` + Hydra 配置加载测试

---

### Wave 2: MuJoCo 环境核心
**目标**: 单实例环境 + 向量化环境，机器人能站立

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 16 | `vr_teleop/envs/g1_base_env.py` | HugWBC `h1.py:40-126` | 加载 `unitree_mujoco/.../g1_29dof.xml`，PD 控制（tau=Kp*(target-q)-Kd*dq），decimation=4，状态提取（注意 MuJoCo wxyz 四元数） |
| 17 | `vr_teleop/envs/observation.py` | HugWBC `observation_buffer.py` + 规划 2.3 节 | ActorObs(58) 与 CriticObs(~99) 构建 + 历史缓冲 |
| 18 | `vr_teleop/envs/reward.py` | HugWBC `h1interrupt.py:430-481` + 规划 2.6 节 | 13 项奖励函数，支持 interrupt_mask |
| 19 | `vr_teleop/envs/termination.py` | HugWBC `h1interrupt.py:378-385` | 接触终止 + 姿态终止 + 超时 |
| 20 | `vr_teleop/envs/domain_rand.py` | FALCON `domain_rand_rl_gym.yaml` | 摩擦/质量/PD增益/推力随机化 |
| 21 | `vr_teleop/envs/mujoco_vec_env.py` | HugWBC `vec_env.py` 接口 | 同步批量方式（同一进程 N 个 MjModel/MjData），obs → GPU，actions → CPU |

**验证**: 零动作 1000 步机器人不摔 + VecEnv(8) step 形状正确 + 奖励值有限

---

### Wave 3: 多步态训练环境
**目标**: 命令采样 + 步态时钟 + 干预接口

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 22 | `vr_teleop/envs/g1_multigait_env.py` | HugWBC `h1interrupt.py` 全文 | 继承 VecEnv：gait_id 采样(stand/walk/run)，速度命令采样，步态相位时钟(sin/cos)，干预注入(15-DOF leg + 14-DOF upper 拼接29-DOF)，episode 逻辑 |

**验证**: stand 命令→零速度 + walk 命令→前进 + 奖励分量符号正确

---

### Wave 4: RL 算法层
**目标**: PPO 训练循环可运行

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 23 | `vr_teleop/agents/networks.py` | HugWBC `net_model.py:104-131` | MlpAdaptModel（不依赖 isaacgym），HistoryEncoder(255→32), StateEstimator(32→3), LowLevelController(93→15) |
| 24 | `vr_teleop/agents/actor_critic.py` | HugWBC `actor_critic.py:39-147` | Actor=MlpAdaptModel, Critic=MLP(99→1)，可学习 std |
| 25 | `vr_teleop/agents/rollout_storage.py` | HugWBC `rollout_storage.py` | 经验缓冲 + GAE 计算 + mini-batch 生成器 |
| 26 | `vr_teleop/agents/ppo.py` | HugWBC `ppo.py` | PPO + 对称损失（G1 15-DOF 置换矩阵）+ 特权信息重建损失 + adaptive KL lr 调度 |
| 27 | `vr_teleop/agents/runner.py` | HugWBC `on_policy_runner.py:45-226` | 训练循环：rollout → update → curriculum → log → checkpoint |

**验证**: 100 次迭代 surrogate loss 下降 + 对称矩阵正交自逆 + 梯度有限

---

### Wave 5: 干预系统（组件 A）
**目标**: 上肢干预信号生成 + 可行性过滤

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 28 | `vr_teleop/intervention/intervention_generator.py` | HugWBC `h1interrupt.py:74-109,247-344` | 高斯/均匀/正弦/结构化干预，duty_mask，transition_stress 放大 |
| 29 | `vr_teleop/intervention/feasibility_filter.py` | 规划 2.9 节 | 关节限位裁剪 + 速度限幅 + 低通滤波 + 回退策略 |
| 30 | `vr_teleop/intervention/ik_solver.py` | 规划 2.10 节 | 阻尼最小二乘 IK（mj_jac），7-DOF 手臂 |
| 31 | `vr_teleop/intervention/motion_retarget.py` | FALCON motion_lib | VR 手柄位姿 → G1 上肢关节角映射 |
| 32 | `vr_teleop/intervention/motion_library.py` | — | 可行动作片段存储 + 采样（Phase 4 用） |

**验证**: 干预目标在关节限位内 + 可行性过滤后轨迹平滑 + Phase 1 训练不发散

---

### Wave 6: 课程系统
**目标**: Phase 0-4 自动升级 + 可选 ADR

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 33 | `vr_teleop/curriculum/phase_curriculum.py` | HugWBC `h1interrupt.py:196-232` + 规划 2.7 节 | Phase 0-4 管理：tracking/fall_rate 指标 → 升级阈值 → 干预参数范围扩大 |
| 34 | `vr_teleop/curriculum/adr_scheduler.py` | 规划文档第 8 节 | ADR：SR > τ_high → 加难，SR < τ_low → 保持/降低 |
| 35 | `configs/experiment/phase0_baseline.yaml` ~ `phase4_vr_align.yaml` | — | 各 Phase 实验预设 |

**验证**: 长时间训练中 Phase 自动提升 + ADR 参数随 SR 变化

---

### Wave 7: 训练脚本 + 部署 + 评估
**目标**: 端到端可用

| # | 文件 | 参考 | 说明 |
|---|------|------|------|
| 36 | `scripts/train.py` | FALCON `train_agent.py` | Hydra 入口：环境创建 → Runner.learn() |
| 37 | `scripts/play.py` | HugWBC `sim2sim_deploy.py` | 加载 checkpoint + MuJoCo viewer 可视化回放 |
| 38 | `scripts/eval.py` | — | 加载模型 → 批量跑 episode → 统计 fall_rate / tracking / transition_failure |
| 39 | `vr_teleop/deploy/policy_wrapper.py` | — | 策略推理封装（去掉训练辅助模块） |
| 40 | `vr_teleop/deploy/sim2sim_runner.py` | — | 不同物理参数下策略验证 |
| 41 | `vr_teleop/deploy/dds_bridge.py` | unitree_mujoco `unitree_sdk2py_bridge.py` | lowcmd/lowstate DDS 通信桥接 |
| 42 | `vr_teleop/eval/evaluator.py` | — | 指标计算模块 |
| 43 | `scripts/build_motion_library.py` | — | 构建可行动作库 |

**验证**: `python scripts/train.py num_envs=256` 完整运行 → checkpoint 保存 → `play.py` 可视化 → Phase 0 站/走成功率 > 95%

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

| 用途 | 路径 |
|------|------|
| G1 MJCF 模型 | `/home/ubuntu/lzxworkspace/codespace/unitree_mujoco/unitree_robots/g1/g1_29dof.xml` |
| G1 场景 | `/home/ubuntu/lzxworkspace/codespace/unitree_mujoco/unitree_robots/g1/scene.xml` |
| G1 关节配置 | `/home/ubuntu/lzxworkspace/codespace/FALCON/humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml` |
| MlpAdaptModel | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/modules/net_model.py` |
| PPO+对称损失 | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/algorithms/ppo.py` |
| ActorCritic | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/modules/actor_critic.py` |
| RolloutStorage | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/storage/rollout_storage.py` |
| OnPolicyRunner | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/runners/on_policy_runner.py` |
| 干预训练环境 | `/home/ubuntu/lzxworkspace/codespace/HugWBC/legged_gym/envs/h1/h1interrupt.py` |
| ObservationBuffer | `/home/ubuntu/lzxworkspace/codespace/HugWBC/legged_gym/legged_utils/observation_buffer.py` |
| VecEnv 接口 | `/home/ubuntu/lzxworkspace/codespace/HugWBC/rsl_rl/rsl_rl/env/vec_env.py` |
| torch_utils | `/home/ubuntu/lzxworkspace/codespace/FALCON/humanoidverse/utils/torch_utils.py` |
| DDS 桥接 | `/home/ubuntu/lzxworkspace/codespace/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py` |
| 域随机化 | `/home/ubuntu/lzxworkspace/codespace/FALCON/humanoidverse/config/domain_rand/domain_rand_rl_gym.yaml` |

---

## 端到端验证清单

1. **环境冒烟**: `G1BaseEnv` 零动作站立 1000 步不摔
2. **向量化**: `MujocoVecEnv(num_envs=256)` step 形状正确，无 NaN
3. **训练冒烟**: `train.py num_envs=256` 运行 100 迭代，loss 有限且下降
4. **Phase 0 收敛**: 10k 迭代后 stand 成功率 > 95%，walk tracking > 0.8
5. **干预鲁棒**: Phase 1-3 训练后 tracking > 0.65，transition_failure < 10%
6. **可视化**: `play.py` 加载 checkpoint 显示稳定行走
7. **Sim2Sim**: 不同摩擦/阻尼参数下 fall_rate < 5%
