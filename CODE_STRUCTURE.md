# AMP 相关文件速览

AMP 是本项目在普通 PPO 任务外加入参考 motion 判别器的训练路径。它不是一个完全独立的环境，而是通过 `r2amp` 任务配置打开：环境继续使用 `R2InterruptRobot`，但会额外加载 motion 数据、生成 `amp_obs`，并在 `rsl_rl` 侧把普通 `PPO` 替换成 `AMPPPO`。

最该先看的 AMP 文件：

- `legged_gym/envs/r2/r2_amp_config.py`：AMP 配置入口。定义 `R2AmpCfg/R2AmpCfgPPO`，打开 `amp.enable`，指定 motion 目录、AMP observation 维度、关键 body、discriminator 超参和 best checkpoint 保存策略。
- `legged_gym/envs/r2/r2.py`：环境侧 AMP 实现。`compute_amp_obs()` 拼单帧 AMP 观测；`R2Robot._init_buffers()` 在 `cfg.amp.enable=True` 时创建 `MotionLoader`、AMP history buffer 和 body/dof 索引；`compute_amp_observations()` 把当前策略状态写入 `infos["amp_obs"]`；`collect_reference_motions()` 从参考 motion 中采样 discriminator 的真样本。
- `legged_gym/utils/motion_loader.py`：AMP motion 加载器。扫描 `legged_gym/motions/*.npz`，校验多 clip 的 `dof_names/body_names/dt` 一致，并按时间插值采样 DOF、body pose 和速度。
- `legged_gym/motions/README.md`：AMP motion 数据契约。说明 `.npz` 必须包含哪些字段、shape 要求、多 clip 约束，以及如何生成数据。
- `legged_gym/motions/r2_walk.npz`：本地默认参考 motion 数据，会被 `R2AmpCfg.amp.motion_file` 指向的目录扫描加载。
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`：训练总控中的 AMP 接入点。`_init_amp()` 创建 `AMPDiscriminator`、`AMPReplayBuffer`，并把算法从 `PPO` 替换成 `AMPPPO`；日志和 checkpoint 中记录 task/style reward。
- `rsl_rl/rsl_rl/algorithms/amp_ppo.py`：AMP-PPO 算法。用 discriminator 对 `infos["amp_obs"]` 计算 style reward 作为日志信号，PPO 仍使用环境 task reward，并在 update 后训练 discriminator。
- `rsl_rl/rsl_rl/modules/discriminator.py`：AMP 判别器网络。输入 flattened AMP observation history，输出真假风格 logit，并提供 gradient penalty。
- `rsl_rl/rsl_rl/storage/amp_storage.py`：策略生成的 AMP observation replay buffer。给 discriminator 提供 agent 样本。
- `legged_gym/scripts/retarget_motion.py`：把外部 G1 `.npz` motion 转成 R2 AMP `.npz` 格式。
- `scripts/convert_lafan1_to_amp.py`：把 LaFAN1 风格 R2V2 `.npz` 转成当前项目 AMP motion 格式，保留 `base_link`、左右 `arm_yaw_link`、左右 `ankle_roll_link` 等 AMP key body。
- `legged_gym/scripts/train.py`：启动 AMP 训练的入口，典型命令是 `python legged_gym/scripts/train.py --task=r2amp --headless`。
- `legged_gym/scripts/play.py`：回放 AMP checkpoint 的入口，支持 `--checkpoint -3` 加载 `model_best_mixed.pt`。

AMP 数据和训练的最短路径是：

```text
参考 motion .npz
  -> legged_gym/motions/
  -> MotionLoader
  -> R2Robot.collect_reference_motions()
  -> AMPDiscriminator 真样本

策略 rollout
  -> R2Robot.compute_amp_observations()
  -> infos["amp_obs"]
  -> AMPPPO.process_env_step()
  -> task reward 写入 PPO storage，style reward 只记录日志
  -> PPO update + discriminator update
```

# VR_Teleoperation 代码结构说明

本文按“文件夹做什么 -> 文件做什么 -> 关键类/函数 -> 在训练链路中的位置”展开。

## 1. 项目总览

这个仓库是一个基于 Isaac Gym 的 R2 人形机器人训练、回放和 AMP 模仿学习项目。整体可以分成四层：

- `legged_gym/`：具体机器人环境层。负责 Isaac Gym 仿真、R2 任务配置、地形、观测、奖励、motion loader、训练/回放脚本。
- `rsl_rl/`：强化学习框架层。负责 PPO、AMP-PPO、Actor-Critic、runner、rollout storage、AMP replay buffer。
- `r2_v2_with_shell_no_hand/` 和 `resources/`：机器人资产层。提供 R2 和 H1 的 URDF/MJCF/XML/STL 模型。
- `scripts/`、`logs/`、`imgs/`：项目辅助层。包括 motion 转换脚本、训练日志/checkpoint、说明图片。

当前 README 里主推的任务名是：

- `r2int`：标准 PPO 任务，环境类是 `R2InterruptRobot`，配置是 `R2InterruptCfg/R2InterruptCfgPPO`。
- `r2amp`：AMP 版本任务，环境类仍是 `R2InterruptRobot`，但配置换成 `R2AmpCfg/R2AmpCfgPPO`，会打开 AMP motion loader 和 discriminator 训练。

## 2. 核心运行链路

### 2.1 普通训练链路

```text
python legged_gym/scripts/train.py --task=r2int --headless
    -> import legged_gym.envs
    -> legged_gym/envs/__init__.py 注册 r2int/r2amp
    -> task_registry.make_env(...)
    -> R2InterruptRobot(cfg=R2InterruptCfg)
    -> BaseTask/R2Robot 创建 Isaac Gym sim、terrain、env actor、buffer
    -> task_registry.make_alg_runner(...)
    -> rsl_rl.runners.OnPolicyRunner
    -> rsl_rl.algorithms.PPO
    -> OnPolicyRunner.learn()
```

训练时每个 iteration 的主循环是：

```text
env.get_observations()
env.get_privileged_observations()
PPO.act()
env.step(actions)
PPO.process_env_step()
PPO.compute_returns()
PPO.update()
env.training_curriculum()
runner.log/save()
```

### 2.2 AMP 训练链路

```text
python legged_gym/scripts/train.py --task=r2amp --headless
    -> R2AmpCfg.amp.enable=True
    -> R2Robot 初始化 MotionLoader 和 amp_observation_buffer
    -> env.step() 在 extras/infos 中放入 infos["amp_obs"]
    -> OnPolicyRunner 检测 train_cfg["amp"]
    -> 创建 AMPDiscriminator + AMPReplayBuffer
    -> 用 AMPPPO 替换 PPO
```

AMP 每步会根据 discriminator 计算 style reward，但当前 PPO storage 写入的仍是环境 task reward：

```text
ppo_reward = task_reward
style_reward = f(discriminator(amp_obs))
```

`AMPPPO.update()` 先做普通 PPO update，再从 replay buffer 取 agent AMP obs、从 `env.collect_reference_motions()` 取参考 motion obs，训练 discriminator。

### 2.3 回放链路

```text
python legged_gym/scripts/play.py --task=r2amp --checkpoint -3
    -> 单环境、plane 地形、关闭随机化/噪声
    -> 加载 checkpoint
    -> 使用 DEMO_PRESETS 写入 env.commands
    -> policy.act_inference()
    -> env.step(actions)
    -> Isaac Gym viewer 显示，可选录制 mp4
```

## 3. 顶层目录与文件

### `README.md`

项目主说明文档。内容包括：

- 项目定位：R2/HugWBC 的 Isaac Gym 训练和回放栈。
- 安装步骤：Conda、PyTorch、外部 Isaac Gym Preview 4、`pip install -e rsl_rl`。
- 训练命令：`r2int` 和 `r2amp`。
- 回放命令：`play.py` 加载最新 checkpoint、best task checkpoint，兼容旧 run 的 best mixed checkpoint。
- AMP motion 数据契约和转换脚本入口。
- 顶层 repository map。

### `CODE_STRUCTURE.md`

本文件。用于写完整项目结构说明。

### `.gitignore`

忽略本地生成物和缓存：

- Python 缓存：`__pycache__/`、`*.py[cod]`、`.pytest_cache/`、`.mypy_cache/`
- 训练输出：`logs/`、`runs/`、`wandb/`、`events.out.tfevents*`、`*.log`
- 编辑器/系统文件：`.idea/`、`.vscode/`、`.claude/`、`.DS_Store`、`Thumbs.db`
- 本地 AMP 数据：`legged_gym/motions/`

### `.git/`

Git 元数据目录，不属于运行逻辑。

### `.claude/settings.local.json`

本地 Claude/Codex 工具配置。当前只看到允许 `WebSearch` 的本地权限配置。和训练代码无直接关系，且被 `.gitignore` 忽略。

## 4. `legged_gym/`

`legged_gym` 是项目的机器人环境与脚本层。它把 Isaac Gym 仿真、R2 任务配置、reward、terrain、motion 数据、训练入口和回放入口组织在一起。

### `legged_gym/__init__.py`

定义两个路径常量：

- `LEGGED_GYM_ROOT_DIR`：仓库根目录。
- `LEGGED_GYM_ENVS_DIR`：`legged_gym/envs` 目录。

这些常量被 R2 asset 路径、motion 路径、日志路径反复使用，例如 `{LEGGED_GYM_ROOT_DIR}/r2_v2_with_shell_no_hand/r2v2_with_shell.urdf`。

### `legged_gym/LICENSE`

`legged_gym` 相关代码许可证文件。

### `legged_gym/envs/`

具体环境与配置目录。

#### `legged_gym/envs/__init__.py`

任务注册入口。导入 R2 环境和配置，并注册：

- `r2int` -> `R2InterruptRobot`, `R2InterruptCfg`, `R2InterruptCfgPPO`
- `r2amp` -> `R2InterruptRobot`, `R2AmpCfg`, `R2AmpCfgPPO`

注意：`R2Robot/R2Cfg` 被导入，但没有直接注册为可运行 task。

#### `legged_gym/envs/base/`

环境基类和通用配置。

##### `base_config.py`

提供 `BaseConfig`。它会递归实例化配置类内部的嵌套 class，让这种写法：

```python
class R2Cfg(LeggedRobotCfg):
    class env:
        num_envs = 4096
```

在 `R2Cfg()` 实例化后变成 `cfg.env.num_envs` 可直接访问的对象属性。

##### `legged_robot_config.py`

基础默认配置文件。

- `LeggedRobotCfg`：环境数量、观测维度、地形、命令、初始姿态、控制参数、asset、domain randomization、reward、normalization、noise、viewer、sim 默认值。
- `LeggedRobotCfgPPO`：PPO 默认训练配置，包括 runner class、policy 网络、algorithm 超参、保存/恢复参数。

R2 的配置都继承并覆盖这里。

##### `base_task.py`

Isaac Gym 环境基类 `BaseTask`。

主要职责：

- 获取 `gymapi.acquire_gym()`。
- 解析 sim device 和 graphics device。
- 初始化 `obs_buf`、`rew_buf`、`reset_buf`、`episode_length_buf`、`time_out_buf` 等通用 buffer。
- 创建 sim 和 viewer。
- 注册 viewer 键盘事件，用键盘模拟手柄按键/摇杆。
- 提供 `render()`。
- 定义子类必须实现的 `reset_idx()` 和 `step()`。

`R2Robot` 继承它，补上具体机器人仿真逻辑。

##### `curriculum.py`

命令课程学习工具。

- `Curriculum`：建立多维命令网格、维护每个 bin 的采样权重、按权重采样。
- `SumCurriculum`：记录成功次数和尝试次数，用于成功率统计。
- `RewardThresholdCurriculum`：根据 reward 是否超过阈值提升当前 bin 和邻域权重。

`R2Robot` 使用它扩展线速度和 yaw 命令采样范围。

#### `legged_gym/envs/r2/`

R2 机器人环境和配置。

##### `r2_config.py`

R2 基础配置。

关键常量：

- `NUM_ACTIONS = 24`
- `PROPRIOCEPTION_DIM = 6 + 3 * NUM_ACTIONS`
- `CMD_DIM = 3 + 4 + 1 + 1`
- `TERRAIN_DIM = 221`
- `PRIVILEGED_DIM = 13`
- `CLOCK_INPUT = 2`

关键类：

- `R2Cfg`：定义 24 个控制 DOF、默认关节角、PD stiffness/damping/torque limits、trimesh 地形、gait/body 命令、奖励项、R2 URDF asset 路径、domain randomization。
- `R2CfgPPO`：定义 policy 为 `MlpAdaptModel`，配置 proprioception/cmd/privileged/terrain 的维度分块，critic 网络和 PPO 训练参数。默认实验名 `r2_teacher`。

##### `r2interrupt_config.py`

在 `R2Cfg` 上叠加 interrupt/disturb 配置。

当前顶部常量：

- `INTERRUPT_IN_CMD = False`
- `NOISE_IN_PRIVILEGE = False`
- `EXECUTE_IN_PRIVILEGE = False`
- `DISTURB_DIM = 0`

所以默认配置下干扰机制结构存在，但实际干扰维度为 0，且 `use_disturb=False`。

关键类：

- `R2InterruptCfg`：调整 observation/command/privileged 维度，新增 `disturb` 子配置，修改部分 reward scale。
- `R2InterruptCfgPPO`：实验名 `r2_interrupt`，`max_iterations=40000`，继续使用 `MlpAdaptModel`。

##### `r2_amp_config.py`

AMP 配置层。

- `R2AmpCfg` 继承 `R2InterruptCfg`，新增 `amp.enable=True`、motion 目录、AMP obs 维度、history 步数、key body 名。
- `R2AmpCfgPPO` 继承 `R2InterruptCfgPPO`，新增 AMP 判别器和 style reward 统计相关超参。

重要字段：

- `motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"`
- `amp_obs_dim = 73`
- `num_amp_obs_steps = 2`
- `key_body_names = ["left_arm_yaw_link", "right_arm_yaw_link", "left_ankle_roll_link", "right_ankle_roll_link"]`
- `reference_body_name = "base_link"`
- `disc_hidden_dims = [1024, 512]`

##### `r2.py`

核心环境实现文件，定义 `R2Robot(BaseTask)`。

模块级函数：

- `_polynomial_planer(...)`：生成五次多项式轨迹系数，用在脚部 clearance reward。
- `compute_amp_obs(...)`：把单帧状态拼成 AMP observation，内容包括 DOF pos/vel、root height、root tangent/normal、root lin/ang vel、关键 body 相对位置。

`R2Robot` 的主要职责：

- 创建 Isaac Gym sim、terrain、env、actor。
- 加载 R2 URDF asset。
- 管理 DOF/root/contact/rigid body tensor。
- 执行 PD 控制和 decimation simulation。
- 采样命令和 gait clock。
- 计算观测、privileged 信息、terrain height scan。
- 计算 reward 和 reset。
- 管理地形课程和命令课程。
- 支持 control latency、domain randomization、随机推力。
- 在 AMP 打开时初始化 motion loader、AMP buffer，并提供 `collect_reference_motions()`。

关键方法按生命周期看：

- `__init__`：解析 cfg，调用 `BaseTask` 创建 sim/env，初始化 buffer、reward、command curriculum、history observation。
- `create_sim`：创建 sim，按 cfg 创建 plane/trimesh terrain，再创建 env actors。
- `_create_envs`：加载 URDF，设置 asset options，创建多环境 actor，收集 body/dof 名字、feet/contact/base 索引。
- `_init_buffers`：wrap Isaac Gym GPU tensor，初始化动作、命令、PD gains、randomization buffer、gait clock、reward sums、AMP 数据结构。
- `step`：处理 action，按控制 decimation 循环计算 torque、simulate，最后调用 `post_physics_step()`。
- `post_physics_step`：刷新状态，计算 base/foot/contact 状态，更新命令和地形高度，检查终止，计算奖励，reset 结束环境，重新计算观测。
- `reset_idx`：更新课程、重置 DOF/root、重采样命令、应用随机化、清 episode 统计。
- `compute_observations/_preprocess_obs`：拼接 proprioception、command、clock、privileged info、terrain height；可加 latency/noise/history stack。
- `_compute_torques`：PD 控制，支持 randomized gains、motor strength、motor offset，并按 torque limit 裁剪。
- `_resample_commands`：采样线速度、yaw、gait frequency、phase、duration、foot swing height、body height、body pitch 等命令。
- `_init_command_distribution/update_command_curriculum_grid`：使用 `RewardThresholdCurriculum` 扩展命令空间。
- `_update_terrain_curriculum`：根据移动距离、tracking reward、姿态失败情况升降 terrain level。
- `compute_amp_observations/_bootstrap_amp_buffer/collect_reference_motions`：AMP 观测和参考 motion 采样。

reward 约定：

- `_prepare_reward_function()` 会扫描 cfg 中非零 reward scale，并调用对应的 `_reward_<name>()`。
- 文件末尾包含大量 reward：速度跟踪、角速度跟踪、站立、base height、orientation、torque、dof vel/acc、action rate、contact force、collision、termination、dof/torque limit、feet stumble、feet slip、feet clearance、gait contact、no fly、alive 等。

##### `r2interrupt.py`

`R2InterruptRobot(R2Robot)`，在基础 R2 环境上增加 interrupt/disturb 动作机制。

关键方法：

- `initial_disturb`：初始化 disturb action、mask、interrupt mask、executed action、课程半径等。
- `_create_envs`：在父类创建 env 后，补充 disturb termination body 索引和 disturb mode mask。
- `_resample_commands`：继承命令采样，并在需要时更新 disturb curriculum。
- `calculate_action`：核心逻辑。先 clip policy action，再根据 disturb mask 替换或叠加末尾 `disturb_dim` 维动作，保存真实执行动作。
- `_preprocess_obs/add_other_privilege`：可把 interrupt flag、目标扰动、实际执行动作拼入观测或 privileged obs。
- `check_termination`：干扰模式下对部分 termination 做豁免。
- reward override：中断时对肩部偏差、upper/lower action rate、碰撞、DOF limits/acc/vel 等做特殊处理。

当前默认 `DISTURB_DIM=0`、`use_disturb=False`，因此 `r2int/r2amp` 默认主要使用它的基础训练能力，干扰逻辑是预留框架。

### `legged_gym/utils/`

环境和脚本共享工具。

#### `helpers.py`

通用配置、命令行和 checkpoint 工具。

- `class_to_dict`：把嵌套配置对象转换成 dict，供 runner 使用。
- `update_class_from_dict`：用 dict 覆盖配置对象。
- `set_seed`：设置 Python、NumPy、Torch、CUDA 随机种子。
- `parse_sim_params`：把配置和 CLI 参数合成 Isaac Gym `SimParams`。
- `get_load_path`：解析 checkpoint 路径，支持：
  - `-1` 最新数字 checkpoint，如 `model_2000.pt`
  - `-2` `model_best_task.pt`
  - `-3` `model_best_mixed.pt`
- `update_cfg_from_args`：用 CLI 参数覆盖 env/train cfg。
- `get_args`：封装 Isaac Gym 参数解析，增加 `--task`、`--resume`、`--checkpoint`、`--num_envs`、`--sim_joystick` 等参数。

#### `task_registry.py`

任务注册和创建中心。

关键类 `TaskRegistry`：

- `register(name, task_class, env_cfg, train_cfg)`：注册任务。
- `get_task_class(name)`：取环境类。
- `get_cfgs(name)`：取 env/train cfg，并同步 seed。
- `make_env(...)`：解析参数、覆盖 cfg、设置 seed、解析 sim params、实例化环境。
- `make_alg_runner(...)`：创建 `OnPolicyRunner`，建立日志目录，处理 resume checkpoint 加载。

全局对象：

- `task_registry = TaskRegistry()`

#### `motion_loader.py`

AMP motion 数据加载器。

职责：

- 支持单个 `.npz` 文件或目录下多个 `.npz` clip。
- 读取 DOF/body 状态、名称、`dt/fps`。
- 校验多 clip 的 `dof_names`、`body_names`、`dt` 一致。
- 建立 name -> index 映射。
- 按时间插值采样 DOF、body pos、body rot、body velocity。
- `sample_times/ensure_time_margin` 保证 AMP history 不跨 clip。

被 `R2Robot` 的 AMP 初始化和 `collect_reference_motions()` 使用。

#### `terrain.py`

heightfield/trimesh 地形生成器。

关键类 `Terrain`：

- 根据 cfg 创建多行多列 terrain map。
- 维护 `env_origins`、`height_field_raw`、`heightsamples`。
- trimesh 模式下生成 `vertices/triangles` 给 Isaac Gym。

关键方法：

- `randomized_terrain`：随机地形。
- `curiculum`：按 row 难度递增的课程地形。
- `selected_terrain`：固定指定地形。
- `make_terrain`：根据 proportions 生成 random uniform、slope、stairs、step 等地形。
- `add_terrain_to_map`：把单块地形写入大地图，并记录 env origin。

辅助函数：

- `gap_terrain`
- `pit_terrain`

#### `math.py`

Torch 数学辅助：

- `quat_apply_yaw`：只保留 yaw 后应用 quaternion。
- `wrap_to_pi`：角度归一到 `[-pi, pi]`。
- `torch_rand_sqrt_float`：平方根分布随机采样。

#### `isaacgym_utils.py`

Isaac Gym quaternion 辅助：

- `copysign`
- `get_euler_xyz`：从 Isaac Gym `xyzw` quaternion 计算 roll/pitch/yaw。

#### `logger.py`

轻量状态/reward logger。

- `log_state/log_states`
- `log_rewards`
- `print_rewards`
- `reset`

当前主训练链路主要用 runner 的 TensorBoard/text log，这个 logger 更偏调试/播放辅助。

#### `__init__.py`

集中导出：

- helpers 中的配置和参数工具。
- `task_registry`
- `Logger`
- math 工具。
- `Terrain`

### `legged_gym/legged_utils/`

更偏 legged locomotion 的辅助模块。

#### `observation_buffer.py`

历史观测 buffer。

关键类 `ObservationBuffer`：

- `reset(reset_idxs, new_obs)`：重置指定环境的历史观测。
- `insert(new_obs)`：左移历史并追加最新观测。
- `get_obs_tensor_3D(history_ids=None)`：返回 `[num_envs, history, obs_dim]` 和 attention mask。

被 `R2Robot` 用于：

- stacked partial observation。
- control latency sensor buffer。

#### `curriculum.py`

和 `envs/base/curriculum.py` 基本重复的课程学习工具。当前 R2 主环境实际导入的是 `legged_gym/envs/base/curriculum.py`，不是这个文件。

### `legged_gym/scripts/`

训练、回放、motion 转换和 asset 检查脚本。

#### `train.py`

最小训练入口。

流程：

1. `from legged_gym.envs import *` 触发任务注册。
2. `task_registry.make_env(name=args.task, args=args)` 创建环境。
3. `task_registry.make_alg_runner(...)` 创建 runner。
4. `ppo_runner.learn(...)` 开始训练。

常用命令：

```powershell
python legged_gym/scripts/train.py --task=r2int --headless
python legged_gym/scripts/train.py --task=r2amp --headless
```

#### `play.py`

模型回放和视频录制脚本。

主要行为：

- 强制单环境。
- 使用 plane 地形。
- 关闭噪声、随机化、课程学习。
- 加载 checkpoint。
- 使用 `DEMO_PRESETS` 生成 `stand`、`jump`、`walk`、`fast_walk` 命令。
- 通过 `R2_PLAY_INITIAL_GAIT` 选择初始 gait。
- 用 `policy.act_inference()` 输出动作均值。
- 可通过 Isaac Gym viewer 截帧，并用 ffmpeg/imageio-ffmpeg 导出 mp4。

关键函数：

- `_build_command_tensor`
- `_get_demo_phase`
- `_update_camera`
- `_apply_deterministic_reset_pose`
- `_init_recording/_capture_record_frame/_finalize_recording`
- `play(args)`

#### `retarget_motion.py`

把外部 G1 `.npz` motion 转成 R2 AMP motion 格式。

主要逻辑：

- 定义 `R2_DOF_NAMES`。
- 定义 `G1_TO_R2_DOF_MAP`。
- 定义 `BODY_MAP`。
- 读取 source npz 的 DOF/body 状态。
- 重排 DOF 到 R2 24 DOF 顺序。
- 选择 AMP 需要的 body。
- 按目标 base height 缩放 body z 和 z 方向速度。
- 输出 `legged_gym/motions` 能直接读取的 `.npz`。

注意：README 中提到当前 `BODY_MAP` 输出的手臂 body 名和 `r2amp` 期望的 `left_arm_yaw_link/right_arm_yaw_link` 有差异，使用前需要检查生成的 `body_names`。

#### `pkl_to_npz.py`

把旧的 LaFAN1 风格 R2V2 `.pkl` motion 转成 AMP `.npz`。

特点：

- 输入 26 DOF，去掉两个 head DOF，输出训练环境的 24 DOF。
- 根据 root pose 和近似 body offsets 构造 `base_link`、手、脚踝 body 状态。
- 支持多个 pkl merge 成一个 `.npz`。

这个脚本更像历史/辅助转换工具。

#### `view_static_asset.py`

Isaac Gym 静态 asset 查看器。

用途：

- 加载指定 URDF/MJCF/XML asset。
- 固定 base。
- 关闭重力。
- GUI 预览，或 `--headless` 快速检查 asset 是否可加载。

默认 asset 是 `r2_v2_with_shell_no_hand/r2v2_with_shell.xml`。

### `legged_gym/motions/`

AMP motion 数据目录。该目录被 `.gitignore` 忽略，但本地存在。

#### `README.md`

AMP motion 数据契约说明。

要求每个 `.npz` 包含：

- `dof_names`
- `body_names`
- `dof_positions`
- `dof_velocities`
- `body_positions`
- `body_rotations`
- `body_linear_velocities`
- `body_angular_velocities`
- `dt` 或 `fps`

多 clip 必须共享相同 `dof_names`、`body_names`、`dt`。

#### `r2_walk.npz`

本地默认 AMP motion 数据之一。`R2AmpCfg.amp.motion_file` 指向整个 `legged_gym/motions` 目录，`MotionLoader` 会扫描这里的 `.npz`。

### `legged_gym/__pycache__/` 及子目录中的 `__pycache__/`

Python 运行生成的 bytecode 缓存。属于生成物，不参与源码结构。

## 5. `rsl_rl/`

`rsl_rl` 是强化学习框架层，提供 PPO、AMP-PPO、Actor-Critic、runner、storage 和 VecEnv 接口。

### `rsl_rl/setup.py`

Python 包安装配置。

- 包名：`rsl_rl`
- 版本：`1.0.2`
- 依赖：`torch`、`torchvision`、`numpy`、`tensorboard`、`tqdm`、`matplotlib`

README 中通过 `pip install -e rsl_rl` 安装这个包。

### `rsl_rl/LICENSE`

`rsl_rl` 许可证文件。

### `rsl_rl/licenses/dependencies/`

依赖库许可证目录。

- `torch_license.txt`：PyTorch license。
- `numpy_license.txt`：NumPy license。

### `rsl_rl/rsl_rl/__init__.py`

包初始化文件，当前只有版权头，没有导出对象。

### `rsl_rl/rsl_rl/algorithms/`

算法层。

#### `__init__.py`

导出：

- `PPO`
- `AMPPPO`

#### `ppo.py`

核心 PPO 实现。

关键类 `PPO`：

- `__init__`：保存 PPO 超参，创建 `AdamW` 优化器，持有 `ActorCritic`。
- `init_storage`：创建 `RolloutStorage`。
- `act`：采样动作并记录 value、log prob、action mean/std、obs。
- `process_env_step`：处理 reward/done/timeouts，把 transition 写入 storage。
- `compute_returns`：用 critic 估计 last value，调用 storage 计算 GAE。
- `update`：计算 clipped surrogate loss、value loss、entropy bonus、可选 adaptation loss、可选 symmetry loss，并优化网络。

特别点：

- `sync_update=True` 时会训练 actor 内部的 privileged reconstruction。
- `use_wbc_sym_loss=True` 时有一套 legacy 镜像 symmetry loss。
- recurrent 分支接口存在，但当前 storage 没有对应 generator，默认 `ActorCritic.is_recurrent=False`。

#### `amp_ppo.py`

AMP 版本 PPO，继承 `PPO`。

关键类 `AMPPPO`：

- `process_env_step`：要求 `infos["amp_obs"]`，用 discriminator 计算 style reward；PPO storage 写入原始 task reward。
- `update`：先跑普通 PPO update，再把 agent AMP obs 放进 replay buffer，统计 AMP reward，训练 discriminator。
- `_update_discriminator`：从 replay buffer 采 agent 样本，从 `env.collect_reference_motions()` 采 reference 样本，训练 LSGAN 风格 discriminator，并加 gradient penalty/logit regularization。

### `rsl_rl/rsl_rl/modules/`

神经网络模块。

#### `__init__.py`

导出：

- `ActorCritic`
- `AMPDiscriminator`

#### `actor_critic.py`

Actor-Critic 封装。

关键类 `ActorCritic`：

- actor 通过 `model_name` 动态创建，例如 `MlpAdaptModel`。
- critic 是普通 MLP，输入 critic obs，输出 value。
- 动作分布是 `Normal(mean, std)`，其中 `std` 是可学习参数。
- `act`：训练时采样动作。
- `act_inference`：推理时返回动作均值和 actor latent。
- `evaluate`：计算 value。
- `get_actions_log_prob`：计算动作 log prob。

#### `net_model.py`

策略网络构件。

函数：

- `get_activation`
- `MLP`

关键类：

- `BaseAdaptModel`：适应型 actor 基类。用历史 proprioception 编码 latent，再预测一小段 privileged 信息，最后输出动作。
- `MlpAdaptModel`：用 MLP 编码历史 proprioception。R2 配置默认使用它。

`sync_update=True` 时，`BaseAdaptModel` 会计算 `privileged_recon_loss`，帮助 actor 从历史观测估计隐含状态。

#### `discriminator.py`

AMP 判别器。

关键类 `AMPDiscriminator`：

- `forward`：输出 AMP observation 的 logit。
- `compute_grad_penalty`：对输入求梯度范数平方，用于 discriminator 正则。

### `rsl_rl/rsl_rl/runners/`

训练总控层。

#### `__init__.py`

导出 `OnPolicyRunner`。

#### `on_policy_runner.py`

训练 runner。它把 env、policy、algorithm、storage、log、checkpoint 串起来。

关键类 `OnPolicyRunner`：

- 初始化时创建 `ActorCritic`。
- 默认创建 `PPO`。
- 如果 `train_cfg` 里有 `amp`，调用 `_init_amp()` 创建 discriminator、AMP replay buffer，并把算法替换成 `AMPPPO`。
- `learn`：执行 rollout、PPO/AMP update、课程学习、日志和 checkpoint 保存。
- `log`：写 TensorBoard 和 `train.log`。
- `_maybe_save_best_checkpoints`：保存 `model_best_task.pt`。
- `save/load`：保存/加载模型、optimizer、AMP discriminator。
- `get_inference_policy`：返回 eval 模式 policy。

### `rsl_rl/rsl_rl/storage/`

数据缓存层。

#### `__init__.py`

导出：

- `RolloutStorage`
- `AMPReplayBuffer`

#### `rollout_storage.py`

PPO rollout buffer。

关键类 `RolloutStorage`：

- 存 `observations`、`privileged_observations`、`actions`、`rewards`、`dones`、`values`、`returns`、`advantages`、`mu`、`sigma`、`actions_log_prob`。
- `Transition` 是单步临时容器。
- `add_transitions` 写入一步 rollout。
- `compute_returns` 计算 GAE 和 advantage normalization。
- `mini_batch_generator` 生成 PPO update mini-batch。

#### `amp_storage.py`

AMP agent observation replay buffer。

关键类 `AMPReplayBuffer`：

- 环形 buffer。
- `insert`：插入 flattened AMP obs。
- `sample`：随机采样 agent AMP obs 给 discriminator。

### `rsl_rl/rsl_rl/env/`

环境抽象接口。

#### `__init__.py`

导出 `VecEnv`。

#### `vec_env.py`

最小向量化环境接口 `VecEnv`。

声明环境需要提供：

- `num_envs`
- `num_obs`
- `num_privileged_obs`
- `num_actions`
- `max_episode_length`
- `obs_buf`
- `privileged_obs_buf`
- `rew_buf`
- `reset_buf`
- `episode_length_buf`
- `extras`
- `device`

抽象方法：

- `step`
- `reset`
- `get_observations`
- `get_privileged_observations`

`legged_gym` 的环境实际比这个接口提供更多属性，例如 `num_partial_obs`、`include_history_steps`、`training_curriculum`、`collect_reference_motions`。

### `rsl_rl/rsl_rl/utils/`

RSL-RL 通用工具。

#### `__init__.py`

导出：

- `split_and_pad_trajectories`
- `unpad_trajectories`

#### `utils.py`

轨迹 padding/unpadding 工具，主要为 recurrent policy 准备。

- `split_and_pad_trajectories(tensor, dones)`：根据 done 切分 `[time, env, ...]` 轨迹并 pad。
- `unpad_trajectories(trajectories, masks)`：把 padded trajectories 还原。

当前默认 policy 非 recurrent，但 `ActorCritic` 中保留了 mask/unpad 支持。

## 6. `r2_v2_with_shell_no_hand/`

R2V2 机器人资产目录，是当前 R2 训练配置实际引用的 asset 来源。

### `r2v2_with_shell.urdf`

R2V2 URDF asset。`R2Cfg.asset.file` 默认指向它。

内容包括：

- `base_link`
- 左右腿：hip pitch/roll/yaw、knee、ankle pitch/roll
- 腰：waist yaw/pitch
- 双臂：shoulder pitch/roll/yaw、arm pitch/yaw
- 手部壳体 fixed link
- 头部 fixed link
- IMU fixed link

训练环境会通过 Isaac Gym 读取这个 URDF，获取 DOF、body、collision、visual、joint limit 等。

### `r2v2_with_shell.xml`

R2V2 MuJoCo XML asset。

用途更偏：

- MuJoCo 资产描述。
- 静态 asset 检查。
- 跨仿真或可视化参考。

`view_static_asset.py` 默认加载的就是这个 XML。

### `HEAD_LOCK_REVIEW.md`

记录头部关节锁定说明：

- `head_yaw_joint`：从 revolute 改 fixed。
- `head_pitch_joint`：从 revolute 改 fixed。

目标是无灵巧手情况下锁头部 2 DOF，保持身体 active DOF 设定。

### `meshes_shell/`

R2V2 shell STL 网格目录。每个 `.STL` 基本对应 URDF/XML 中同名 link 的可视化或碰撞几何。

根、腰、头：

- `base_link.STL`
- `waist_yaw_link.STL`
- `waist_pitch_link.STL`
- `head_yaw_link.STL`
- `head_pitch_link.STL`

左腿：

- `left_hip_pitch_link.STL`
- `left_hip_roll_link.STL`
- `left_hip_yaw_link.STL`
- `left_knee_link.STL`
- `left_ankle_pitch_link.STL`
- `left_ankle_roll_link.STL`

右腿：

- `right_hip_pitch_link.STL`
- `right_hip_roll_link.STL`
- `right_hip_yaw_link.STL`
- `right_knee_link.STL`
- `right_ankle_pitch_link.STL`
- `right_ankle_roll_link.STL`

左臂和左手：

- `left_shoulder_pitch_link.STL`
- `left_shoulder_roll_link.STL`
- `left_shoulder_yaw_link.STL`
- `left_arm_pitch_link.STL`
- `left_arm_yaw_link.STL`
- `left_hand_pitch_link.STL`
- `left_hand_roll_link.STL`
- `left_hand_link.STL`

右臂和右手：

- `right_shoulder_pitch_link.STL`
- `right_shoulder_roll_link.STL`
- `right_shoulder_yaw_link.STL`
- `right_arm_pitch_link.STL`
- `right_arm_yaw_link.STL`
- `right_hand_pitch_link.STL`
- `right_hand_roll_link.STL`
- `right_hand_link.STL`
- `right_hand_link copy.STL`

## 7. `resources/`

外部/参考机器人资产目录。目前主要是 H1。

### `resources/robots/h1/urdf/h1.urdf`

H1 机器人 URDF。

包含 pelvis、腿部、torso、双臂、IMU、足端固定点等定义。当前 README 的 R2 任务不直接使用它，但它可以作为仿真资产或 motion retargeting/参考机器人资产。

### `resources/robots/h1/meshes/`

H1 STL 网格目录。

腿部：

- `left_hip_pitch_link.STL`
- `left_hip_roll_link.STL`
- `left_hip_yaw_link.STL`
- `left_knee_link.STL`
- `left_ankle_link.STL`
- `left_ankle_link_modified.STL`
- `right_hip_pitch_link.STL`
- `right_hip_roll_link.STL`
- `right_hip_yaw_link.STL`
- `right_knee_link.STL`
- `right_ankle_link.STL`
- `right_ankle_link_modified.STL`

躯干：

- `pelvis.STL`
- `torso_link.STL`
- `logo_link.STL`

手臂：

- `left_shoulder_pitch_link.STL`
- `left_shoulder_roll_link.STL`
- `left_shoulder_yaw_link.STL`
- `left_elbow_link.STL`
- `right_shoulder_pitch_link.STL`
- `right_shoulder_roll_link.STL`
- `right_shoulder_yaw_link.STL`
- `right_elbow_link.STL`

## 8. `scripts/`

顶层项目辅助脚本目录，区别于 `legged_gym/scripts`。

### `convert_lafan1_to_amp.py`

把 LaFAN1 风格 R2V2 `.npz` motion 转成当前项目 AMP motion 格式。

输入特点：

- `dof_positions` 是 26 DOF，包含 head yaw/pitch。
- `body_positions` 是完整 R2V2 body。
- `dt` 是源数据 timestep。

输出特点：

- 去掉 head 两个 DOF，得到 24 DOF。
- 只保留 AMP body：
  - `base_link`
  - `left_arm_yaw_link`
  - `right_arm_yaw_link`
  - `left_ankle_roll_link`
  - `right_ankle_roll_link`
- 按目标 base height 缩放 z 和 z 速度。
- 写入 `legged_gym/motions` 可直接被 `MotionLoader` 读取。

关键函数：

- `convert_file(src_path, dst_path, target_base_height=0.78)`
- `main()`

### `scripts/__pycache__/`

Python 运行缓存。属于生成物，不是源码。

## 9. `imgs/`

文档/展示图片资源，不参与训练运行。

- `framework.png`：项目/论文框架图。
- `share-logo.png`：分享或展示 logo。
- `sjtu.png`：上海交大相关图片。

## 10. `logs/`

训练日志、TensorBoard event、checkpoint 和分析产物。按 `.gitignore` 设计它应当是本地生成物，但当前工作区里存在这些文件。

### `logs/train_r2v2_amp_version4_console.log`

R2 AMP version4 的控制台训练日志。记录 iteration、FPS、reward、episode length、curriculum 等。

### `logs/analysis/`

训练结果分析目录。

- `r2_amp_training_report.md`：R2 AMP 训练分析报告，包括训练时长、迭代数、环境步数、reward/loss 变化、最终奖励分解。
- `r2_amp_scalar_summary.csv`：TensorBoard 标量汇总表，便于继续筛选指标或画图。

### `logs/r2_amp/`

AMP 任务实验输出。

#### `Apr12_05-06-14_r2v2_amp_hopeful/`

- `events.out.tfevents...`：TensorBoard event 文件，记录一次 AMP 训练过程。

#### `Apr17_15-18-11_r2v2_amp_version4/`

- `events.out.tfevents...`：TensorBoard event 文件。
- `model_best_mixed.pt`：旧训练 run 可能留下的 mixed reward 最佳 checkpoint，可用 `--checkpoint -3` 加载。
- `train.log`：runner 写出的训练文本日志。

### `logs/r2_interrupt/`

标准 interrupt/PPO 任务实验输出。

#### `Apr02_08-46-31_/`

- `events.out.tfevents...`：TensorBoard event 文件。当前文件很小，可能是一次未完整运行的实验。

## 11. 缺失或外部依赖目录

### `IsaacGymEnvs/`

当前仓库顶层没有 `IsaacGymEnvs/` 目录。Isaac Gym 按 README 要求从外部 Isaac Gym Preview 4 安装，不随仓库提供。

### `tests/`

当前仓库未发现显式 `tests/` 目录，也未发现独立测试文件。验证主要依赖：

- 能否创建环境。
- 能否启动训练。
- 回放是否正常。
- TensorBoard/log 指标。
- asset viewer 是否能加载资产。

## 12. 主要继承与配置关系

配置继承：

```text
BaseConfig
  -> LeggedRobotCfg
      -> R2Cfg
          -> R2InterruptCfg
              -> R2AmpCfg

BaseConfig
  -> LeggedRobotCfgPPO
      -> R2CfgPPO
          -> R2InterruptCfgPPO
              -> R2AmpCfgPPO
```

环境继承：

```text
BaseTask
  -> R2Robot
      -> R2InterruptRobot
```

RL 框架关系：

```text
OnPolicyRunner
  -> ActorCritic
      -> MlpAdaptModel actor
      -> critic MLP
  -> PPO
      -> RolloutStorage

OnPolicyRunner with amp config
  -> AMPPPO
      -> PPO behavior
      -> AMPDiscriminator
      -> AMPReplayBuffer
      -> env.collect_reference_motions()
```

## 13. 最重要的改动入口

如果要改 R2 任务行为，通常看这些文件：

- 观测/reward/仿真逻辑：`legged_gym/envs/r2/r2.py`
- 干扰/中断动作逻辑：`legged_gym/envs/r2/r2interrupt.py`
- 基础 R2 参数：`legged_gym/envs/r2/r2_config.py`
- `r2int` 参数：`legged_gym/envs/r2/r2interrupt_config.py`
- `r2amp` 参数：`legged_gym/envs/r2/r2_amp_config.py`
- AMP motion 读取：`legged_gym/utils/motion_loader.py`
- PPO/AMP-PPO 算法：`rsl_rl/rsl_rl/algorithms/ppo.py`、`rsl_rl/rsl_rl/algorithms/amp_ppo.py`
- 训练循环和 checkpoint：`rsl_rl/rsl_rl/runners/on_policy_runner.py`
- 策略网络：`rsl_rl/rsl_rl/modules/actor_critic.py`、`rsl_rl/rsl_rl/modules/net_model.py`
- R2 asset：`r2_v2_with_shell_no_hand/r2v2_with_shell.urdf`
