# VR 多步态人形遥操作 -- 模型代码实现计划

> 基于 readme 的 A+B 架构，参考 HugWBC（算法/网络架构）、FALCON（G1 机器人配置/解耦设计）、unitree_mujoco（MuJoCo G1 模型）

---

## 0. 设计决策总结

| 决策项 | 选择 |
|--------|------|
| 仿真器 | **Isaac Gym GPU tensor pipeline 为默认**；MuJoCo 仅作 sim2sim、单元测试与部署前对齐验证 |
| 机器人 | Unitree G1 29DOF（复用 `unitree_mujoco/unitree_robots/g1/g1_29dof.xml`） |
| 策略架构 | readme A+B 解耦：π_leg 控制下肢 15DOF；上肢走“手目标位姿→重定向/滤波/IK→14DOF”链路 |
| 网络模型 | HugWBC 的 MlpAdaptModel（history encoder + state estimator + low-level controller） |
| RL 算法 | PPO + 对称损失 + 特权信息重建损失 |
| 课程系统 | Phase 0-4 手动课程 + 可选 ADR 自动调度 |
| 数据通路 | Isaac Gym GPU state tensors → PyTorch GPU policy/PPO → GPU action tensor → Isaac Gym step |
| 开发顺序 | A+B 并行开发，且以 Isaac Gym GPU rollout 主链路优先 |

---

## 1. 项目目录结构

```
VR_Teleoperation/
├── configs/                          # YAML 配置文件
│   ├── robot/g1_29dof.yaml           # G1 关节定义、PD 增益、力矩限制
│   ├── env/g1_intervention.yaml      # 环境参数：episode 长度、decimation、仿真步长
│   ├── algo/ppo_leg.yaml             # PPO 超参数
│   ├── obs/g1_leg_obs.yaml           # 观测空间组成、缩放、噪声
│   ├── rewards/g1_multigait.yaml     # 奖励函数与权重
│   ├── curriculum/phase_0_to_4.yaml  # 课程 Phase 定义与阈值
│   ├── intervention/upper_body.yaml  # 干预生成器参数
│   ├── domain_rand/g1_rand.yaml      # 域随机化范围
│   └── experiment/                   # 实验级组合配置
│       ├── phase0_baseline.yaml
│       ├── phase1_low.yaml
│       ├── phase2_medium.yaml
│       ├── phase3_strong.yaml
│       └── phase4_vr_align.yaml
│
├── vr_teleop/                        # 主 Python 包
│   ├── envs/                         # MuJoCo 环境
│   │   ├── g1_base_env.py            # 单实例 MuJoCo G1 环境封装
│   │   ├── g1_multigait_env.py       # 多步态训练环境（核心）
│   │   ├── mujoco_vec_env.py         # MuJoCo 对齐/调试向量化环境（可选多进程后端）
│   │   ├── isaac_vec_env.py          # Isaac Gym GPU 主训练向量化环境
│   │   ├── observation.py            # 观测构建器
│   │   ├── reward.py                 # 奖励函数实现
│   │   ├── termination.py            # 终止条件
│   │   └── domain_rand.py            # 域随机化模块
│   │
│   ├── agents/                       # RL 算法
│   │   ├── ppo.py                    # PPO（含对称损失 + 特权重建损失）
│   │   ├── actor_critic.py           # Actor-Critic 封装
│   │   ├── networks.py               # MLP、HistoryEncoder、StateEstimator
│   │   ├── rollout_storage.py        # Rollout 经验缓冲
│   │   └── runner.py                 # 训练循环编排器
│   │
│   ├── intervention/                 # 组件 A：上肢干预系统
│   │   ├── intervention_generator.py # 参数化干预信号生成器
│   │   ├── feasibility_filter.py     # H2O 式可行性筛选器
│   │   ├── motion_retarget.py        # VR/mocap → G1 上肢重定向
│   │   ├── ik_solver.py              # G1 手臂逆运动学
│   │   └── motion_library.py         # 可行动作库存储与采样
│   │
│   ├── curriculum/                   # 课程学习
│   │   ├── phase_curriculum.py       # Phase 0-4 手动课程管理
│   │   ├── adr_scheduler.py          # ADR 自动难度调度
│   │   └── lp_teacher.py             # LP-Teacher（可选）
│   │
│   ├── robot/                        # G1 机器人定义
│   │   ├── g1_config.py              # G1 关节索引、限位、默认姿态、PD增益
│   │   └── g1_kinematics.py          # 正/逆运动学
│   │
│   ├── utils/                        # 工具
│   │   ├── math_utils.py             # 四元数、旋转矩阵、重力投影
│   │   ├── config_utils.py           # YAML 配置加载
│   │   ├── logger.py                 # TensorBoard 日志
│   │   └── symmetry.py               # G1 左右对称置换矩阵
│   │
│   ├── deploy/                       # 部署
│   │   ├── sim2sim_runner.py         # sim2sim 验证
│   │   ├── policy_wrapper.py         # 策略推理封装
│   │   └── dds_bridge.py             # unitree_mujoco DDS 接口桥接
│   │
│   └── eval/                         # 评估
│       ├── evaluator.py              # 指标计算
│       └── visualization.py          # 轨迹可视化
│
├── scripts/                          # 入口脚本
│   ├── train.py                      # 训练入口
│   ├── play.py                       # 策略可视化回放
│   ├── eval.py                       # 评估
│   ├── sim2sim.py                    # sim2sim 部署
│   └── build_motion_library.py       # 构建可行动作库
│
├── data/motions/                     # 动作数据
├── tests/                            # 单元测试
├── logs/                             # 训练日志
├── checkpoints/                      # 模型检查点
├── requirements.txt
└── readme.md
```

---

## 2. 关键模块实现计划

### 2.1 机器人配置 -- `vr_teleop/robot/g1_config.py`

**数据来源**: FALCON `g1_29dof_waist_fakehand.yaml` + unitree_mujoco `g1_29dof.xml`

```python
class G1Config:
    NUM_DOF = 29
    NUM_ACTIONS_LEG = 15    # 12 leg + 3 waist
    NUM_ACTIONS_ARM = 14    # 7 per arm

    # 关节索引（严格按照 MJCF actuator 顺序）
    LEG_INDICES   = [0..11]   # left_hip_pitch → right_ankle_roll
    WAIST_INDICES = [12,13,14]
    LEFT_ARM_INDICES  = [15..21]
    RIGHT_ARM_INDICES = [22..28]
    LOWER_BODY_INDICES = [0..14]  # leg + waist（策略输出）
    UPPER_BODY_INDICES = [15..28] # 两臂（干预驱动）

    # 关节限位（rad）-- 来自 FALCON config dof_pos_lower/upper_limit_list
    # 力矩限位（Nm）-- [88,88,88,139,50,50, 88,88,88,139,50,50, 88,50,50, 25x5,5,5, 25x5,5,5]
    # 速度限位（rad/s）-- [32,32,32,20,37,37, ...]
    # PD 增益  -- stiffness/damping 来自 FALCON control section
    # 默认关节角 -- 来自 FALCON init_state.default_joint_angles
    # 对称索引 -- 来自 FALCON symmetric_dofs_idx
    # 初始位置 -- pos=[0, 0, 0.8], standing height ~0.793m
```

### 2.2 训练/验证环境 -- `vr_teleop/envs/`

#### 2.2.1 `g1_base_env.py` -- 单实例环境

- 加载 `unitree_mujoco/unitree_robots/g1/g1_29dof.xml`（通过 scene.xml）
- 封装 `mujoco.MjModel` / `mujoco.MjData`
- 提供 `reset()` / `step(action_29dof)` / `get_sensor_data()`
- **PD 控制**（在 Python 层计算，写入 `mj_data.ctrl`）：
  ```
  q_des = action_scale * action + default_angles
  tau = kp * (q_des - q) - kd * dq
  tau = clip(tau, -torque_limit, torque_limit)
  mj_data.ctrl[:] = tau
  ```
- **Decimation=4**: 每次 `step()` 内执行 4 次 `mj_step()`（仿真 200Hz，策略 50Hz）

#### 2.2.2 `mujoco_vec_env.py` -- MuJoCo 对齐/回归后端

- 仅用于调试、单元测试、Isaac Gym 对齐验证、部署前观测一致性检查
- 默认模式：单进程同步批量（同一进程维护 N 个 `MjModel/MjData`）
- 可选后端：`multiprocessing.Pool` 或 `concurrent.futures.ProcessPoolExecutor`
- 多进程模式下每个 worker 持有独立 `g1_base_env`，通过共享内存传递 action/observation 批量张量
- 目标: 1-64 环境验证，不作为主训练吞吐来源

#### 2.2.3 `isaac_vec_env.py` -- Isaac Gym GPU 主训练后端

- 基于 Isaac Gym / Legged Gym 风格 GPU tensor API 构建 G1 批量环境
- 通过 `gymtorch.wrap_tensor(...)` 直接读取 root states、dof states、contact forces 等 GPU tensor
- 观测构建、奖励、终止、域随机化全部在 GPU 端批量完成
- 主目标: 4096-16384 env 规模下获得接近 HugWBC/FALCON 的 GPU rollout 吞吐

#### 2.2.4 `g1_multigait_env.py` -- 训练环境（核心）

继承向量化环境，增加:
- **多步态命令采样**: gait_id ∈ {stand=0, walk=1, run=2} + 速度命令 (vx, vy, wz)
- **上肢干预注入**: 从 `InterventionGenerator` 获取手目标位姿，经重定向/滤波/IK 得到上肢 14DOF 目标；与策略 15DOF 输出拼接为 29DOF，并执行腿优先安全约束与回退逻辑
- **奖励计算**: 调用 `reward.py` 中各奖励函数
- **终止检测**: 调用 `termination.py`
- **课程更新**: 每次 PPO update 后检查是否升级 Phase
- **后端策略**: 默认走 `isaac_vec_env.py`；`mujoco_vec_env.py` 只用于 debug / eval 对齐

### 2.3 观测空间 -- `vr_teleop/envs/observation.py`

#### Actor 观测（部分可观测，可部署到真机）

| 分量 | 维度 | 来源 | 缩放 |
|------|------|------|------|
| 基座角速度（body frame） | 3 | IMU gyro | ×0.25 |
| 投影重力向量（body frame） | 3 | IMU quat 计算 | ×1.0 |
| 下肢关节角（相对默认值） | 15 | jointpos [0:15] | ×1.0 |
| 下肢关节角速度 | 15 | jointvel [0:15] | ×0.05 |
| 上一步动作 | 15 | 策略输出缓冲 | ×1.0 |
| **小计 proprioception** | **51** | | |
| 命令: vx, vy | 2 | 外部命令 | ×1.0 |
| 命令: wz | 1 | 外部命令 | ×1.0 |
| 命令: gait_id | 1 | 外部命令 | ×1.0 |
| 干预指示 I(t) | 1 | 干预生成器 | ×1.0 |
| **小计 command** | **5** | | |
| 步态时钟: sin(phase), cos(phase) | 2 | 步态时钟 | ×1.0 |
| **单步总计** | **58** | | |
| **含历史 (H=5)** | **51 × 5 = 255** | 仅对 proprioception 做历史堆叠并送入 HistoryEncoder；历史缓存默认保存在 GPU | |

#### Critic 观测（特权信息，仅训练时使用）

| 分量 | 维度 |
|------|------|
| 当前 Actor 观测（单步） | 58 |
| 基座线速度（body frame） | 3 |
| 基座离地高度 | 1 |
| 足底接触力（左、右） | 2 |
| 摩擦系数（随机化） | 1 |
| 质量偏移 | 1 |
| 电机强度乘数 | 1 |
| PD 增益乘数 (kp, kd) | 2 |
| 上肢关节角 | 14 |
| 上肢关节角速度 | 14 |
| 干预参数 (amp, freq) | 2 |
| **总计** | **~99** |

### 2.4 网络架构 -- `vr_teleop/agents/networks.py`

参考 HugWBC `rsl_rl/modules/net_model.py` 第 104-131 行的 `MlpAdaptModel`:

```
Actor 网络（MlpAdaptModel 架构）:
┌─────────────────────────────────────────────────────┐
│ HistoryEncoder:                                      │
│   输入: obs_history [B, 5, 51] → flatten → [B, 255] │
│   Linear(255, 256) → ELU → Linear(256, 128)         │
│   → ELU → Linear(128, 32) → latent [B, 32]          │
├──────────────────────────────────────────────────────┤
│ StateEstimator:                                       │
│   Linear(32, 64) → ELU → Linear(64, 32) → ELU       │
│   → Linear(32, 3) → privileged_pred [B, 3]           │
│   （重建: base linear velocity）                      │
├──────────────────────────────────────────────────────┤
│ LowLevelController:                                   │
│   输入: concat(latent, priv_pred, current_proprio,    │
│           command+clock) = [B, 32+3+51+7=93]          │
│   Linear(93, 512) → ELU → Linear(512, 256) → ELU     │
│   → Linear(256, 128) → ELU → Linear(128, 15)         │
│   → action [B, 15]                                   │
└──────────────────────────────────────────────────────┘

动作分布: Gaussian(mean=action, std=learnable)
  初始 std=0.8, 范围 clamp [0.1, 1.2]

Critic 网络:
  输入: critic_obs [B, ~99]
  Linear(99, 512) → ELU → Linear(512, 256) → ELU
  → Linear(256, 128) → ELU → Linear(128, 1)
  → value [B, 1]
```

### 2.5 PPO 算法 -- `vr_teleop/agents/ppo.py`

参考 HugWBC `rsl_rl/algorithms/ppo.py`:

**超参数**:
- `clip_param=0.2`, `gamma=0.99`, `lam=0.95`
- `num_learning_epochs=5`, `num_mini_batches=4`
- `learning_rate=1e-4`, adaptive KL schedule (`desired_kl=0.01`)
- `value_loss_coef=1.0`, `entropy_coef=0.01`
- `max_grad_norm=1.0`
- `use_symmetry_loss=True`, `symmetry_loss_coef=0.5`
- `sync_update=True`（启用特权信息重建损失）

**总损失**:
```
L = L_surrogate + value_loss_coef * L_value - entropy_coef * H
    + symmetry_loss_coef * L_symmetry
    + L_privileged_reconstruction
```

**对称损失**: 使用 FALCON `symmetric_dofs_idx` 中的左右对称索引，构建置换矩阵，强制策略对称输入产生对称输出。

**执行策略**:
- rollout storage、GAE、mini-batch 采样、PPO update 默认全部在 CUDA 上完成
- 避免 CPU↔GPU 往返拷贝成为瓶颈；Isaac Gym 状态、奖励、动作全部维持在 GPU tensor 上

### 2.6 奖励函数 -- `vr_teleop/envs/reward.py`

| 奖励项 | 类型 | 权重 | 说明 |
|--------|------|------|------|
| tracking_lin_vel | 正向 | +2.0 | exp(-‖v_cmd - v_actual‖²/σ²), σ=0.25 |
| tracking_ang_vel | 正向 | +3.0 | exp(-(wz_cmd - wz_actual)²/σ²) |
| alive | 正向 | +0.2 | 存活奖励 |
| torso_orientation | 惩罚 | -20.0 | 躯干 roll/pitch 偏离 |
| ang_vel_xy | 惩罚 | -0.5 | 横滚/俯仰角速度 |
| base_height | 惩罚 | -40.0 | 离目标高度偏差 |
| action_rate | 惩罚 | -0.01 | 动作变化率 + 二阶变化 |
| torques | 惩罚 | -5e-6 | 下肢力矩平方和 |
| dof_acc | 惩罚 | -2.5e-7 | 关节加速度 |
| foot_slip | 惩罚 | -0.2 | 足地接触时水平滑移 |
| feet_contact_forces | 惩罚 | -0.2 | 过大接触力 |
| transition_stability | 惩罚 | -5.0 | 步态切换窗口额外稳定性惩罚 |
| standing_still | 惩罚 | -10.0 | stand 命令时关节不应动 |
| termination | 惩罚 | -200.0 | 非超时终止惩罚 |

### 2.7 课程系统 -- `vr_teleop/curriculum/phase_curriculum.py`

**Phase 0 -- 无干预（基础）**
- `I(t)=0`, 仅学: 稳站、稳走、基本切换
- 速度: [-0.3, 0.3], 步态: {stand, walk}
- 升级条件: tracking > 0.80, fall_rate < 0.05

**Phase 1 -- 低频小幅低占空比**
- `amp=[0.05, 0.2]`, `freq=[0.3, 1.0]`, `duty=[0.1, 0.3]`, `hold_T=[1.0, 3.0]`
- 升级条件: tracking > 0.75, fall_rate < 0.10

**Phase 2 -- 结构化干预 + 中幅**
- `amp=[0.1, 0.5]`, `freq=[0.5, 2.0]`, `duty=[0.2, 0.5]`, `transition_stress=1.5`
- 加入结构化模式: arm_swing, turning_gesture
- 速度扩大, 步态: {stand, walk, run}
- 升级条件: tracking > 0.70, transition_failure < 0.15

**Phase 3 -- 强干预 + 切换窗口持续干预**
- `amp=[0.2, 1.0]`, `freq=[0.5, 3.0]`, `duty=[0.5, 0.8]`, `transition_stress=2.0`
- 切换窗口强制 `I(t)=1`
- 升级条件: tracking > 0.65, transition_failure < 0.10, time_to_fall > 15s

**Phase 4 -- 真实 VR/人体动作分布对齐**
- 启用 motion_library 回放
- mixing_ratio: 20% → 50% → 80%（逐步增加真实数据比例）
- 合成干预仍保留用于多样性

### 2.8 干预生成器 -- `vr_teleop/intervention/intervention_generator.py`

**干预参数向量**: `d = [amp, freq, duty, transition_stress, hold_T]`

**信号生成模式**:
1. **正弦振荡**: `target[j] = default[j] + amp * sin(2π*freq*t + φ_j) * duty_mask(t)`
2. **均匀随机噪声**: 在关节限位内采样，按课程缩放（参考 HugWBC `h1interrupt.py`）
3. **结构化模式**: 摆臂（反相 shoulder_pitch）、伸手（工作空间随机目标）
4. **动作库回放**: Phase 4 使用经过可行性筛选的动作片段

**切换窗口应力放大**:
```python
if in_transition_window:
    amp_effective = amp * transition_stress  # 例如 2x
    force intervention_active = True         # 忽略 duty
```

### 2.9 可行性筛选器 -- `vr_teleop/intervention/feasibility_filter.py`

处理流水线:
```
原始目标 → [1] 关节限位检查（夹到限位±5%余量）
         → [2] 速度限位检查（(target-current)/dt < vel_limit）
         → [3] 自碰撞检查（`mj_forward` 后读取 `data.contact/ncon`）
         → [4] 躯干稳定性检查（预测 torso roll/pitch < 0.3 rad）
         → [5] 信号整形（低通滤波 + rate/jerk 限幅）
         → 可行目标 or 回退到中性姿态
```

### 2.10 IK 求解器 -- `vr_teleop/intervention/ik_solver.py`

- G1 单臂 7-DOF（shoulder_pitch/roll/yaw + elbow + wrist_roll/pitch/yaw）
- 阻尼最小二乘法（DLS）迭代 IK
- 使用 MuJoCo `mj_jac()` 计算雅可比矩阵
- 输入: 手目标位姿 `[p_des, R_des]` → 输出: 7 个关节角

---

## 3. 训练流程

### 3.1 训练循环

```python
# scripts/train.py
cfg = load_config(experiment="phase0_baseline")
env = G1MultiGaitEnv(cfg, num_envs=4096, device="cuda:0", sim_backend="isaacgym")
actor_critic = ActorCritic(env.num_partial_obs, env.num_obs, env.num_actions, cfg.algo)
ppo = PPO(actor_critic, **cfg.algo)
ppo.init_storage(env.num_envs, num_steps_per_env=24, ...)

obs, critic_obs = env.reset()
for iteration in range(max_iterations):
    # Rollout 收集
    with torch.inference_mode():
        for step in range(num_steps_per_env):
            actions = ppo.act(obs, critic_obs)
            obs, critic_obs, rewards, dones, infos = env.step(actions)
            ppo.process_env_step(rewards, dones, infos)
        ppo.compute_returns(critic_obs)

    # PPO 更新
    metrics = ppo.update()

    # 课程更新
    env.update_curriculum()

    # 日志 & 检查点
    log(iteration, metrics)
    if iteration % save_interval == 0:
        save_checkpoint(ppo, iteration)
```

### 3.2 与 HugWBC/FALCON 的关键差异

1. **采用 Isaac Gym**: 直接复用 HugWBC / FALCON 的 GPU 仿真范式与 tensor API 组织方式
2. **GPU 向量化**: 基于 `create_sim`、`acquire_*_tensor`、`refresh_*_tensor` 实现大规模并行环境
3. **观测/奖励/重置**: 全部基于 GPU tensor 完成，避免 host copy
4. **对称损失置换矩阵**: 按 G1 的 15-action 下肢关节排列重新计算

---

## 4. Sim2Sim 部署流程

### 4.1 独立 MuJoCo 验证

```python
# scripts/sim2sim.py
policy = PolicyWrapper.load("checkpoints/best.pt")
env = G1BaseEnv("unitree_mujoco/unitree_robots/g1/scene.xml")
# 使用不同物理参数（摩擦、时间步、solver 参数）回放策略
```

### 4.2 unitree_mujoco DDS 桥接

- 通过 `dds_bridge.py` 将策略输出转换为 `rt/lowcmd` 消息
- 策略读取 `rt/lowstate` 获取观测
- 验证与 unitree_mujoco 仿真器的通信管线

### 4.3 验证协议

- **物理参数变化**: 摩擦 [0.3, 0.5, 1.0, 1.5, 2.0], 阻尼 ±50%
- **观测一致性**: 训练环境与部署环境的观测对齐校验
- **核心指标**: fall_rate, transition_failure, time_to_fall, torso_peak_deviation

---

## 5. 需修改/引用的关键文件

| 参考文件 | 路径 | 用途 |
|----------|------|------|
| G1 MJCF 模型 | `unitree_mujoco/unitree_robots/g1/g1_29dof.xml` | 直接复用为仿真资产 |
| G1 场景 | `unitree_mujoco/unitree_robots/g1/scene.xml` | 环境加载入口 |
| MlpAdaptModel | `HugWBC/rsl_rl/rsl_rl/modules/net_model.py:104-131` | 网络架构参考 |
| PPO 算法 | `HugWBC/rsl_rl/rsl_rl/algorithms/ppo.py` | PPO+对称损失参考 |
| 干预训练 | `HugWBC/legged_gym/envs/h1/h1interrupt.py` | 干预注入逻辑参考 |
| G1 配置 | `FALCON/humanoidverse/config/robot/g1/g1_29dof_waist_fakehand.yaml` | 关节限位/PD增益/对称索引 |
| 解耦 PPO | `FALCON/humanoidverse/agents/decouple/ppo_decoupled_wbc_ma.py` | 解耦架构思路参考 |
| DDS 桥接 | `unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py` | sim2sim 通信参考 |

---

## 6. 实现步骤（按模块拆分）

### Step 1: 基础设施搭建
1. 创建项目目录结构
2. 编写 `requirements.txt` / `environment.yml`（Isaac Gym、PyTorch CUDA、numpy、omegaconf、tensorboard 等）
3. 实现 `g1_config.py`（从 FALCON yaml 提取所有数据）
4. 实现 `math_utils.py`（替代 `isaacgym.torch_utils`）
5. 实现 `config_utils.py`（YAML 加载合并）

### Step 2: 环境层
6. 实现 `g1_base_env.py`（MuJoCo 单实例封装 + PD 控制，用于 sim2sim）
7. 实现 `isaac_vec_env.py`（Isaac Gym GPU 主训练后端）
8. 实现 `mujoco_vec_env.py`（MuJoCo 对齐/调试后端）
9. 实现 `observation.py`（观测构建）
10. 实现 `reward.py`（所有奖励函数）
11. 实现 `termination.py`（终止条件）
12. 实现 `domain_rand.py`（域随机化）
13. 实现 `g1_multigait_env.py`（组装完整训练环境）

### Step 3: RL 算法层
13. 实现 `networks.py`（MLP + HistoryEncoder + StateEstimator）
14. 实现 `actor_critic.py`
15. 实现 `rollout_storage.py`
16. 实现 `ppo.py`（含对称损失 + 特权重建损失）
17. 实现 `runner.py`（训练循环）

### Step 4: 干预系统（组件 A）
18. 实现 `intervention_generator.py`（参数化干预信号）
19. 实现 `feasibility_filter.py`（可行性筛选）
20. 实现 `ik_solver.py`（手臂 IK）
21. 实现 `motion_retarget.py`（VR → G1 重定向）
22. 实现 `motion_library.py`（动作库存储采样）

### Step 5: 课程系统
23. 实现 `phase_curriculum.py`（Phase 0-4）
24. 实现 `adr_scheduler.py`（ADR 自动调度）
25. 编写所有 YAML 配置文件

### Step 6: 训练脚本与集成
26. 实现 `scripts/train.py`
27. 实现 `scripts/play.py`
28. 实现 `symmetry.py`（G1 对称置换矩阵）
29. 实现 `logger.py`

### Step 7: 部署与评估
30. 实现 `policy_wrapper.py`
31. 实现 `sim2sim_runner.py`
32. 实现 `dds_bridge.py`
33. 实现 `evaluator.py`
34. 实现 `scripts/eval.py`、`scripts/sim2sim.py`

### Step 8: 测试
35. 编写各模块单元测试
36. 集成测试: 完整训练 Phase 0 → 验证收敛

---

## 7. 验证方案

### 7.1 环境验证
- `step()` 返回有效形状的 obs/reward/done
- 零动作不崩溃，随机动作产生有限奖励
- Isaac Gym 与 MuJoCo 单步/短时 rollout 对齐
- PD 控制输出与 unitree_mujoco 一致

### 7.2 训练验证
- Phase 0: `sim_backend=isaacgym, device=cuda:0, num_envs>=4096` 下 10k iterations 后 stand 成功率 >95%, walk tracking >80%
- Phase 1-3: 逐步引入干预，tracking 不低于 0.65, transition_failure < 10%

### 7.3 Sim2Sim 验证
- 训练环境 vs 独立 MuJoCo 的观测一致性 (atol < 1e-4)
- 不同物理参数下 fall_rate < 5%

### 7.4 消融实验
- 有/无课程对比
- 有/无可行性筛选器（A）对比
- Phase 0-3 vs 加 Phase 4 对比
- 手动 Phase vs ADR 对比
