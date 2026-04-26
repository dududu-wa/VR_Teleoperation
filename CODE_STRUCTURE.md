# Code Structure Walkthrough

这份文档不是只列目录，而是按代码真实调用链从上往下读：先看入口和任务注册，再看 runner，再进入 R2 环境生命周期，最后单独拆 AMP。为了避免“顺着代码讲故事讲得太顺”，这里保留一个质疑者视角，专门指出哪些说法需要代码证据、哪些地方容易误导。

## Review Method

本次走读按四个 subagent 视角交叉检查：

- 入口层 subagent：检查 `train.py`、`play.py`、`task_registry.py`、任务注册和 CLI 覆盖。
- 环境层 subagent：检查 `R2Robot`、`R2InterruptRobot`、reset、step、obs、reward、torque。
- AMP/RL subagent：检查 `OnPolicyRunner`、`PPO`、`AMPPPO`、discriminator、AMP replay buffer、motion loader。
- 质疑者 subagent：专门找文档中没有证据、路径不准、命令复制后可能失败的地方。

## 0. Top-Level Layout

- `README.md`: 快速启动、训练、播放、AMP motion 数据说明。
- `legged_gym`: Isaac Gym 环境、R2 机器人逻辑、任务注册、训练和播放脚本、motion loader。
- `rsl_rl`: 本仓库使用的 RL 包，包含 PPO、AMP PPO、runner、actor-critic、storage、discriminator。
- `legged_gym/motions`: `r2amp` 默认读取的 AMP `.npz` 参考动作目录。
- `scripts`: 项目级数据转换脚本，不属于 upstream `legged_gym/scripts` 的标准入口。
- `r2_v2_with_shell_no_hand`: R2 URDF、MJCF、mesh 等资产。
- `logs`: 本地训练输出，实际 run 目录形如 `logs/<experiment_name>/<MonDD_HH-MM-SS>_<run_name>`。

质疑者：`logs/<experiment_name>/<run_name>` 这种写法不够准。代码在 `task_registry.make_alg_runner()` 里会给 run name 前面拼时间戳。

## 1. Entry Layer

训练入口是 `legged_gym/scripts/train.py`。

真实调用链：

```text
python legged_gym/scripts/train.py --task=r2int/r2amp
-> train.py imports legged_gym.envs *
-> legged_gym/envs/__init__.py 执行 task_registry.register(...)
-> get_args()
-> train(args)
-> task_registry.make_env(name=args.task, args=args)
-> task_registry.make_alg_runner(env=env, name=args.task, args=args)
-> ppo_runner.learn(...)
```

关键代码点：

- `train.py` 的 `from legged_gym.envs import *` 会触发任务注册副作用。
- `train(args)` 只做三件事：创建 env，创建 runner，调用 `learn()`。
- `helpers.get_args()` 解析 `--task`、`--num_envs`、`--resume`、`--load_run`、`--checkpoint` 等 CLI 参数。
- `helpers.update_cfg_from_args()` 只覆盖一部分字段，例如 env 的 `num_envs` 和 train config 的 resume/checkpoint 参数。

质疑者：CLI 脚本本身并不知道有哪些 task。task 能不能找到，取决于导入 `legged_gym.envs` 后注册表里有没有对应名字。

播放入口是 `legged_gym/scripts/play.py`。

真实调用链：

```text
python legged_gym/scripts/play.py --task=r2amp --load_run ... --checkpoint ...
-> play.py imports legged_gym.envs *
-> task_registry.get_cfgs(args.task)
-> 原地修改 env_cfg 为单环境、plane、无噪声、无随机化、长 episode
-> task_registry.make_env(..., env_cfg=env_cfg)
-> train_cfg.runner.resume = True
-> task_registry.make_alg_runner(env=env, train_cfg=train_cfg)
-> runner.load(checkpoint)
-> ppo_runner.get_inference_policy()
-> env.reset()
-> _apply_deterministic_reset_pose(env)
-> 循环写 env.commands，policy.act_inference()，env.step()
```

质疑者：`play.py` 会强制 `resume=True`，所以刚克隆仓库直接运行 play 可能失败。必须先训练出 checkpoint，或者显式传入已有 `--load_run/--checkpoint`。

## 2. Task Registration Layer

任务注册在 `legged_gym/envs/__init__.py`。

当前直接注册的任务只有：

- `r2int`: `R2InterruptRobot` + `R2InterruptCfg` + `R2InterruptCfgPPO`
- `r2amp`: `R2InterruptRobot` + `R2AmpCfg` + `R2AmpCfgPPO`

也就是说，当前训练入口不是直接实例化纯 `R2Robot/R2Cfg`。真实结构是：

```text
R2InterruptRobot
-> inherits R2Robot
-> inherits BaseTask
```

质疑者：不要把 `R2Robot` 写成当前直接训练 task。它是主体父类，当前注册 task 实例化的是 `R2InterruptRobot`。

`TaskRegistry` 在 `legged_gym/utils/task_registry.py` 里维护三张表：

- `task_classes[name]`: task 名到环境类。
- `env_cfgs[name]`: task 名到环境配置对象。
- `train_cfgs[name]`: task 名到训练配置对象。

关键函数：

- `register(name, task_class, env_cfg, train_cfg)`: 把三元组注册进去。
- `get_cfgs(name)`: 返回注册好的 env config 和 train config，并把 train seed 同步到 env。
- `make_env(name, args, env_cfg)`: 应用 CLI 覆盖，解析 sim 参数，实例化环境类。
- `make_alg_runner(env, name, args, train_cfg)`: 应用 CLI 覆盖，创建 runner，必要时加载 checkpoint。

质疑者：`get_cfgs()` 返回的是注册对象本身，不是深拷贝。`play.py` 对 env config 的改动是原地改配置对象，这在单进程脚本里通常没问题，但文档不要暗示它是不可变快照。

## 3. Config Layer

R2 配置按继承关系叠加：

- `legged_gym/envs/r2/r2_config.py`: `R2Cfg` 和 `R2CfgPPO`，定义 24 action、观测维度、初始姿态、PD 参数、命令范围、reward scale、地形、domain randomization、PPO 默认项。
- `legged_gym/envs/r2/r2interrupt_config.py`: `R2InterruptCfg` 和 `R2InterruptCfgPPO`，继承 R2 配置，增加 interrupt/disturb 相关字段。当前默认 `use_disturb=False` 且 `DISTURB_DIM=0`。
- `legged_gym/envs/r2/r2_amp_config.py`: `R2AmpCfg` 和 `R2AmpCfgPPO`，继承 interrupt 配置，增加环境侧 AMP motion 设置和训练侧 AMP PPO/discriminator 设置。

常见 CLI 覆盖：

- `--num_envs`: 覆盖 `env_cfg.env.num_envs`。
- `--seed`: 覆盖 train seed。
- `--max_iterations`: 覆盖 runner 最大训练迭代。
- `--resume --load_run --checkpoint`: 控制 checkpoint 加载。

质疑者：`runner_class_name` 是配置里的字符串，`TaskRegistry.make_alg_runner()` 用 `eval(train_cfg.runner_class_name)` 创建 runner。当前配置指向 `OnPolicyRunner`，但文档最好写“当前配置创建 OnPolicyRunner”，不要写成永远只能创建它。

## 4. Runner And PPO Layer

runner 在 `rsl_rl/rsl_rl/runners/on_policy_runner.py`。

初始化流程：

```text
OnPolicyRunner.__init__
-> read train_cfg["runner"], ["algorithm"], ["policy"]
-> env.num_partial_obs 给 actor
-> env.num_obs 给 critic
-> ActorCritic(...)
-> PPO(actor_critic, ...)
-> alg.init_storage(...)
-> if "amp" in train_cfg: _init_amp(train_cfg["amp"])
-> env.reset()
```

训练循环在 `OnPolicyRunner.learn()`。

每个 iteration 的大结构：

```text
obs = env.get_observations()
critic_obs = env.get_privileged_observations()

for it:
  for rollout step:
    actions = alg.act(obs, critic_obs)
    obs, critic_obs, rewards, dones, infos = env.step(actions)
    alg.process_env_step(rewards, dones, infos)

  alg.compute_returns(critic_obs)
  metrics = alg.update()
  env.training_curriculum()
  log / save checkpoint
```

PPO 本体在 `rsl_rl/rsl_rl/algorithms/ppo.py`。

关键函数：

- `PPO.act(obs, critic_obs)`: actor 采样 action，critic 估值，记录 transition。
- `PPO.process_env_step(rewards, dones, infos)`: 写 rollout storage，并处理 timeout bootstrap。
- `PPO.compute_returns(last_critic_obs)`: 用 critic value 计算 GAE/returns。
- `PPO.update()`: 按 mini-batch 做 policy loss、value loss、entropy、可选 symmetry loss，然后 optimizer step。

质疑者：actor 用的是 `env.num_partial_obs`，critic 用的是 `env.num_obs`。不要把观测说成一个单一向量。

## 5. R2 Environment Layer

主体环境在 `legged_gym/envs/r2/r2.py` 的 `R2Robot`。当前注册 task 用的是 `legged_gym/envs/r2/r2interrupt.py` 的 `R2InterruptRobot`，它继承并覆盖部分行为。

初始化链路：

```text
R2InterruptRobot.__init__
-> R2Robot.__init__
-> _parse_cfg(cfg)
-> BaseTask.__init__
-> create_sim()
-> _create_envs()
-> _init_buffers()
-> _prepare_reward_function()
-> _init_command_distribution(...)
-> interrupt-specific initial_disturb(...)
```

### Simulation Creation

`R2Robot.create_sim()` 负责创建 Isaac Gym sim 和地形。随后 `_create_envs()` 负责：

- 读取 `cfg.asset.file` 指向的 R2 URDF。
- 创建 `gymapi.AssetOptions`。
- `gym.load_asset()` 加载 robot asset。
- 读取 DOF/body 名称。
- 创建 `num_envs` 个 Isaac Gym env。
- 为每个 env 创建 actor、设置 DOF 属性、刚体属性、摩擦等。
- 缓存 feet、penalized contact、termination contact、base body 索引。

质疑者：`--num_envs` 只改变并行 env 数，不会改变单个机器人结构。速度提升受 GPU 显存、sim step、policy update batch 大小一起限制。

### Step Flow

每个环境 step 从 `R2Robot.step(actions)` 开始：

```text
step(actions)
-> calculate_action(actions)
-> render()
-> repeat cfg.control.decimation times:
     _compute_torques(actions_after_push)
     gym.set_dof_actuation_force_tensor(...)
     gym.simulate(...)
     gym.refresh_dof_state_tensor(...)
-> post_physics_step()
-> return partial_obs_buf, obs_buf, rew_buf, reset_buf, extras
```

`R2InterruptRobot.calculate_action()` 会先走基础 action clip，然后在 disturb 开启时替换或融合末尾若干 action 维度。当前默认 `use_disturb=False` 和 `disturb_dim=0`，所以默认训练里 interrupt 代码壳存在，但扰动机制基本不生效。

### Post Physics Flow

`post_physics_step()` 是环境每步的中枢：

```text
refresh root/contact/dof/body tensors
-> 更新 base_pos、base_quat、rpy、base_lin_vel、base_ang_vel、projected_gravity
-> 更新 foot contact、collision state、foot pose
-> AMP buffer bootstrap if needed
-> _post_physics_step_callback()
-> check_termination()
-> compute_reward()
-> compute_amp_observations() if AMP enabled
-> reset_idx(done_envs)
-> compute_observations(done_envs)
-> 滚动 last_actions、last_dof_vel、last_root_vel
```

`_post_physics_step_callback()` 处理每步辅助逻辑：

- 周期性重采样 commands。
- heading 模式下把 heading error 转成 yaw command。
- 更新 gait phase、clock inputs、desired contact states。
- 测地形高度和 foot scan。
- domain randomization push 或 teleport。

### Reset Flow

`reset()` 会重置所有 env，然后额外执行一次零动作 `step()` 来生成初始 observation。

`reset_idx(env_ids)` 做的事比“重置状态”多：

- 可选更新 terrain curriculum。
- `_reset_dofs(env_ids)`: 关节位置随机到默认姿态的 `0.5~1.5` 倍，速度随机到 `[-1, 1]`。
- `_reset_root_states(env_ids)`: 根据 env origin 放置 base，随机 yaw，随机 base 线速度和角速度。
- `_resample_commands(env_ids)`: 重采样速度、yaw、gait、body height、body pitch 等命令。
- `apply_randomizations(env_ids)`: 应用摩擦、增益、link 属性、mass 等随机化。
- 清空 last action、foot air time、episode length、episode sums。
- 写 `extras["episode"]` 给 runner logging。

质疑者：不要说 reset 只是恢复初始姿态。它还采样命令、随机化环境、更新课程学习，并且 `reset()` 会再走一次零动作 step。

### Observation Flow

`_preprocess_obs()` 先拼接完整 `obs_buf`：

- base angular velocity。
- projected gravity。
- joint position error 和 joint velocity。
- 当前 action。
- commands。
- gait clock。
- privileged 信息，例如 base linear velocity、base height error、foot clearance、friction、contact forces。
- terrain height samples。

`compute_observations(reset_env_ids)` 继续处理：

- 可选 control latency。
- 可选 observation noise。
- clip observation。
- `partial_obs_buf = obs_buf[..., :num_partial_obs]` 给 actor。
- 如果 `stack_history_obs=True`，actor obs 会变成历史堆叠后的 3D tensor。

质疑者：actor 和 critic 看到的信息不同。actor 主要用 `partial_obs_buf`，critic 用完整 `obs_buf`。

### Reward Flow

`_prepare_reward_function()` 会读取 `cfg.rewards.scales`，只保留非零 reward scale，并动态绑定同名 `_reward_<name>()` 函数。

`compute_reward()` 每步执行：

```text
rew_buf = 0
for each active reward function:
  rew = reward_fn() * reward_scale
  if reward is curriculum-controlled: rew *= curriculum_scale
  rew_buf += rew
  episode_sums[name] += rew
  command_sums[name] += rew when needed
if only_positive_rewards: clip rew_buf >= 0
if termination scale exists: add _reward_termination()
```

典型 `_reward_*` 覆盖速度跟踪、角速度、姿态、base height、torque、dof velocity/acceleration、action rate、碰撞、接触力、足端高度、足端滑移、站立等。

质疑者：不要在文档里手写一个固定 reward 列表说“训练一定用这些”。真正启用哪些 reward 取决于 `cfg.rewards.scales` 的非零项。

### Torque Flow

`_compute_torques(actions)` 当前只支持 P 控制：

```text
actions_scaled = actions * cfg.control.action_scale
target_pos = actions_scaled + default_dof_pos + motor_offsets
torque = p_gain * (target_pos - dof_pos) - d_gain * dof_vel
torque *= motor_strength
torque = clip(torque, -custom_torque_limits, custom_torque_limits)
```

质疑者：最终控制输出 clip 用的是 `custom_torque_limits`，来自 `R2Cfg.control.torque_limits` 按关节名匹配；不能简单说“只由 URDF effort 决定”。

## 6. AMP Layer

AMP 有两套相关开关，必须分清。

环境侧开关：

- `R2AmpCfg.amp.enable = True`。
- `R2Robot._init_buffers()` 看到 `hasattr(cfg, "amp") and cfg.amp.enable` 后，创建 AMP observation buffer、motion loader、DOF/body 映射。

训练侧开关：

- `R2AmpCfgPPO` 有顶层 `class amp`。
- `class_to_dict(train_cfg)` 后 runner 会看到 `train_cfg["amp"]`。
- `OnPolicyRunner.__init__()` 检查 `"amp" in train_cfg`，然后调用 `_init_amp()`。

质疑者：只写 `enable=True` 会误导。环境侧 AMP buffer 和训练侧 AMPPPO 是两个来源，一个在 env config，一个在 train config。

### AMP Config

`legged_gym/envs/r2/r2_amp_config.py` 当前关键值：

- `motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"`。
- `amp_obs_dim = 73`。
- `num_amp_obs_steps = 2`。
- `key_body_names = left_arm_yaw_link, right_arm_yaw_link, left_ankle_roll_link, right_ankle_roll_link`。
- `reference_body_name = "base_link"`。
- `task_reward_weight = 0.6`。
- `style_reward_weight = 0.4`。

### AMP Observation

单帧 AMP observation 由 `compute_amp_obs()` 拼接：

```text
dof_pos                  24
dof_vel                  24
root height               1
root tangent + normal     6
root linear velocity      3
root angular velocity     3
4 key bodies relative xyz 12
total                    73
```

live observation 流程：

```text
R2Robot.post_physics_step()
-> compute_amp_observations()
-> compute_amp_obs(live robot state)
-> shift amp_observation_buffer history
-> extras["amp_obs"] = flattened history
```

reference observation 流程：

```text
MotionLoader loads .npz clips
-> validates same dof_names/body_names/dt across clips
-> collect_reference_motions(num_samples)
-> sample current and historical times
-> convert motion quaternion wxyz to IsaacGym xyzw
-> compute_amp_obs(reference state)
-> return [num_samples, num_amp_obs_steps, amp_obs_dim]
```

Motion files are expected to include DOF names, body names, DOF state, body state, and preferably `dt` or `fps`. If both `dt` and `fps` are missing, `MotionLoader` assumes 30 FPS.

质疑者：`base_link` 这里是 simulator/root link frame 和 AMP reference body，不等于 URDF inertial COM offset。

### AMPPPO Runtime

`OnPolicyRunner._init_amp()` 做三件事：

- 创建 `AMPDiscriminator`。
- 创建 `AMPReplayBuffer`。
- 用 `AMPPPO` 替换默认 `PPO`，但保留同一个 actor-critic。

rollout 每步：

```text
OnPolicyRunner.learn()
-> actions = self.alg.act(obs, critic_obs)
-> env.step(actions)
-> infos["amp_obs"] comes from env.extras
-> AMPPPO.process_env_step(rewards, dones, infos)
```

`AMPPPO.process_env_step()` 做的关键事：

```text
disc_logit = discriminator(amp_obs)
style_reward = -log(1 - sigmoid(disc_logit) + 1e-7) * disc_reward_scale
mixed_reward = task_reward_weight * task_reward + style_reward_weight * style_reward
PPO.process_env_step(mixed_reward, dones, infos)
```

质疑者：mixed reward 不是只用于 logging。它会进入 PPO rollout storage，后面的 returns、value loss、policy loss 都基于 mixed reward。

每轮 update：

```text
AMPPPO.update()
-> super().update() 先更新 PPO actor-critic
-> insert collected agent amp_obs into AMPReplayBuffer
-> _update_discriminator()
```

`_update_discriminator()`：

- 从 replay buffer 采 agent AMP obs。
- 调 `env.collect_reference_motions()` 采 reference AMP obs。
- agent logit 目标是 0，reference logit 目标是 1。
- 加 gradient penalty 和 logit regularization。
- 用独立 AdamW 更新 discriminator。

质疑者：discriminator 不是和 actor-critic 在同一个 loss 里一起反传。代码顺序是先 PPO update，再单独 discriminator update。

### AMP Motion Conversion Caveat

`scripts/convert_lafan1_to_amp.py` 是当前更贴近 `r2amp` body set 的转换路径，因为它会选择 `base_link`、`left_arm_yaw_link`、`right_arm_yaw_link`、左右 ankle。

`legged_gym/scripts/retarget_motion.py` 需要额外检查。当前 `BODY_MAP` 会输出 `left_hand_roll_link` 和 `right_hand_roll_link`，但 `r2amp` 配置需要 `left_arm_yaw_link` 和 `right_arm_yaw_link`。如果直接拿这个脚本输出去训练 `r2amp`，`MotionLoader.get_body_index()` 可能找不到 key body，或者动作语义不一致。

质疑者：这不是文档洁癖，是会直接影响 AMP 是否读到正确参考 body 的问题。

## 7. Play And Checkpoint Details

`play.py` 会修改环境为演示模式：

- `num_envs = 1`。
- 地形改为 plane。
- 关闭 terrain curriculum。
- 关闭 observation noise。
- 关闭 friction/load/gain/link/base mass randomization。
- 设置超长 episode。
- `env.use_disturb = False`，演示时基本关闭 interrupt。
- 通过 `_build_command_tensor()` 手动写 `env.commands`。

checkpoint 加载由 `helpers.get_load_path()` 决定：

- `--checkpoint -1`: 找最新 numeric `model_<iteration>.pt`。
- `--checkpoint -2`: 加载 `model_best_task.pt`。
- `--checkpoint -3`: 加载 `model_best_mixed.pt`。

质疑者：`-2/-3` 只有对应文件存在才有效。`r2amp` 默认保存 best task/mixed，其他 task 不一定有这些文件。

## 8. Debugging Pointers

- 入口断点：先看 `train.py` 的 `train(args)`，确认 task 名有没有进 `task_registry`。
- 注册断点：看 `legged_gym/envs/__init__.py`，确认 `r2amp` 绑定的是 `R2InterruptRobot + R2AmpCfg + R2AmpCfgPPO`。
- runner 断点：看 `OnPolicyRunner.__init__()`，确认 `self.use_amp = "amp" in train_cfg` 是否为 true。
- env 断点：看 `R2Robot._init_buffers()`，确认是否创建 `_motion_loader` 和 `amp_observation_buffer`。
- reward 断点：看 `AMPPPO.process_env_step()`，确认 `infos["amp_obs"]` 是否存在，以及 mixed reward 权重是否符合预期。
- motion 断点：看 `MotionLoader.get_body_index()`，确认 `.npz` 的 `body_names` 覆盖 `R2AmpCfg.amp.key_body_names`。
- 日志断点：`train.log` 明确记录 task/mixed/style reward、episode length、best checkpoint 信息；当前代码没有直接记录 termination rate 字段。

质疑者：如果机器人学成下跪，不要只看最终 play 画面。需要同时检查 task reward、style reward、mixed reward、AMP motion 的 base/key body 统计、以及 `r2_config.py` 里的 base height 和初始姿态。

## 9. Common Modification Points

- 改并行环境数：CLI 用 `--num_envs`，代码会覆盖 `env_cfg.env.num_envs`。
- 改训练长度：CLI 用 `--max_iterations`，代码会覆盖 `train_cfg.runner.max_iterations`。
- 改命令范围：`R2Cfg.commands.ranges`。
- 改默认站姿：`R2Cfg.init_state.default_joint_angles`。
- 改初始 base 高度：`R2Cfg.init_state.pos`。
- 改 base height reward target：`R2Cfg.rewards.base_height_target`。
- 改 PD 和 torque：`R2Cfg.control`。
- 改 reward 权重：`R2Cfg.rewards.scales` 或子类 config 覆盖。
- 改 AMP task/style 比例：`R2AmpCfgPPO.amp.task_reward_weight` 和 `style_reward_weight`。
- 改 AMP key bodies：`R2AmpCfg.amp.key_body_names`，同时必须保证 motion `.npz` 的 `body_names` 匹配。
