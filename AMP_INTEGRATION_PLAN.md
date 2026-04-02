# AMP + Discriminator 集成到 VR_Teleoperation 计划

## Context

当前 VR_Teleoperation 项目使用纯 PPO + 手工奖励函数训练 R2 人形机器人行走。需要从 `humanoid_amp` 项目中引入 AMP (Adversarial Motion Priors) + Discriminator 方法，让机器人学习更自然的运动风格。

两个项目使用不同的仿真引擎（IsaacGym vs IsaacLab）和 RL 框架（rsl_rl vs skrl），因此需要在 IsaacGym + rsl_rl 框架内重新实现 AMP 机制，而非直接迁移代码。

核心思路：**AMP 作为插件模块**，在现有 PPO 训练循环上叠加 discriminator 训练和 style reward，保留全部现有功能。

---

## 修改概览

### 新增文件（6个）

| 文件 | 职责 |
|------|------|
| `rsl_rl/rsl_rl/modules/discriminator.py` | AMP Discriminator 网络，复用已有 `net_model.py` 中的 `MLP()` 构建器 |
| `rsl_rl/rsl_rl/algorithms/amp_ppo.py` | 继承现有 `PPO` 类，增加 discriminator 训练逻辑 |
| `rsl_rl/rsl_rl/storage/amp_storage.py` | AMP observation 的 replay buffer |
| `legged_gym/utils/motion_loader.py` | 参考运动数据加载器（从 `humanoid_amp/motions/motion_loader.py` 适配，去掉 IsaacLab 依赖） |
| `legged_gym/envs/r2/r2_amp_config.py` | AMP 训练专用配置，继承 `R2InterruptCfg` / `R2InterruptCfgPPO` |
| `legged_gym/scripts/retarget_motion.py` | G1→R2 运动数据重定向脚本 |

### 修改文件（3个）

| 文件 | 修改内容 |
|------|----------|
| `legged_gym/envs/r2/r2.py` | `R2Robot` 中添加 amp_observation_buffer + `compute_amp_observations()` + reset 后延迟 bootstrap history |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` | 添加 `_init_amp()` + save/load 中增加 discriminator |
| `legged_gym/envs/__init__.py` | 增加 1 行 AMP 任务注册 |

---

## 详细实现步骤

### 第1步：Discriminator 网络

新建 `rsl_rl/rsl_rl/modules/discriminator.py`

复用 `net_model.py` 中已有的 `MLP()` 函数和 `get_activation()` 函数构建网络。

```python
from rsl_rl.modules.net_model import MLP  # 复用已有 MLP 构建器

class AMPDiscriminator(nn.Module):
    def __init__(self, amp_obs_dim, hidden_dims=[1024, 512], activation='relu'):
        super().__init__()
        # 复用 net_model.py 的 MLP(): amp_obs_dim → 1024 → 512 → 1
        self.net = nn.Sequential(*MLP(amp_obs_dim, 1, hidden_dims, activation))

    def forward(self, amp_obs):
        return self.net(amp_obs)  # 输出 logit，不加 sigmoid

    def compute_grad_penalty(self, amp_obs):
        amp_obs.requires_grad_(True)
        disc_out = self.forward(amp_obs)
        grad = torch.autograd.grad(disc_out.sum(), amp_obs, create_graph=True)[0]
        return (grad.norm(2, dim=-1) ** 2).mean()
```

### 第2步：AMP Observation 定义与 History Buffer

R2 的 AMP obs（适配 28 DOF，参照 `g1_amp_env.py` 的 `compute_obs()` 函数）：

```
R2 AMP Observation (单帧 81 维):
- dof_pos:           28 维（关节位置）
- dof_vel:           28 维（关节速度）
- root_height:        1 维（base_link 高度）
- root_orientation:   6 维（四元数→切线+法线，复用 g1_amp_env.py 的 quaternion_to_tangent_and_normal）
- root_lin_vel:       3 维（世界坐标系）
- root_ang_vel:       3 维（世界坐标系）
- key_body_rel_pos: 4×3=12 维（左右手+左右脚相对 base 位置）
总计: 28+28+1+6+3+3+12 = 81 维
```

关键体（从 URDF 确认的准确名称）：
- `left_hand_roll_link`（R2 URDF 第 743 行）
- `right_hand_roll_link`（R2 URDF 第 993 行）
- `left_ankle_roll_link`（R2 URDF 第 192 行）
- `right_ankle_roll_link`（R2 URDF 第 415 行）

AMP history buffer 采用与参考实现 `g1_amp_env.py:60-62` 相同的设计：
- shape: `(num_envs, num_amp_observations=2, amp_obs_dim=81)`
- 每步滚动：旧帧后移，新帧写入 `[:, 0]`
- flatten 后通过 `extras["amp_obs"]` 传出，shape `(num_envs, 162)`

### 第3步：环境侧修改 — `r2.py`

当前 `post_physics_step()` 的执行顺序是 `r2.py:170-173`：
```
check_termination()
compute_reward()
env_ids = reset_buf.nonzero().flatten()
reset_idx(env_ids)          ← 先 reset（随机关节 + 随机根速度）
compute_observations(env_ids) ← 再算 obs
```

#### 问题1：reset 时 AMP history 跨 episode 污染

如果只在 step 里简单保存 prev/cur 拼接，reset 后的第一帧会把「上一回合终止帧」和「新回合初始帧」拼成假 transition，污染 discriminator。

#### 问题2：不能用 reference motion 回填 buffer

参考实现 `g1_amp_env.py:183,198` 之所以能用 reference motion 回填 AMP buffer，是因为它同时把机器人状态也 reset 到了同一条 sampled motion（`_reset_strategy_random`）。R2 的 reset 是自己的随机/默认 reset（`_reset_dofs` + `_reset_root_states`），机器人实际状态和 reference motion 不匹配。如果用 reference motion 回填 buffer，discriminator 的 agent batch 里会混入 reference 数据。

#### 解决方案：终止帧先产出 `amp_obs`，reset 只做延迟初始化标记

当前 `r2.py` 的 `post_physics_step()` 会先 `refresh_*_tensor()`，然后 `check_termination()/compute_reward()`，再 `reset_idx()`。这意味着两件事要显式区分：

1. 本训练步要送进 discriminator / style reward 的 `amp_obs`，必须来自 **reset 之前** 的当前 post-physics 状态；
2. reset 后不能立即依赖 `rigid_body_states` 回填 history，因为当前代码路径没有在 `set_*_indexed()` 之后再次拿到刚体状态的权威 refresh。

因此 AMP history 采用“两阶段”处理：

- 在 `reset_idx()` 之前先 `compute_amp_observations()`，把本步真实 terminal/non-terminal 状态写入 `extras["amp_obs"]`
- `reset_idx(env_ids)` 只负责常规 reset，并把 `amp_reset_pending[env_ids] = True`
- 下一次 `post_physics_step()` 开头、完成 `refresh_*_tensor()` 之后，再对 `amp_reset_pending` 的 env 用当前权威 live state 填满整个 history buffer，然后清掉 pending 标记

这样处理后：

- 不会跨 episode 污染（旧 history 不会混进新 episode）
- `infos["amp_obs"]` 的语义正确，对 done env 也仍然对应本步 terminal 状态
- 不依赖“reset 后同帧 `rigid_body_states` 一定已刷新”这个错误前提
- 刚 reset 的 env 在第一条 AMP history 上表现为“两帧相同”，语义上等价于“无历史”
- 不需要改变 R2 的 reset 语义

#### 问题3：四元数约定不一致

**已确认**：
- IsaacGym `root_states[:, 3:7]` 和 `isaacgym.torch_utils.quat_apply` 使用 **xyzw**（`isaacgym_utils.py:12`）
- motion_loader 返回的 `body_rotations` 存储为 **wxyz**（`data_convert.py:283`）
- `g1_amp_env.py` 的 `quaternion_to_tangent_and_normal` 使用 `isaaclab.utils.math.quat_apply`，期望 **wxyz**

因此 `_compute_amp_obs` 中的 `quat_apply` 必须区分来源：
- live sim 数据：xyzw → 直接用 `isaacgym.torch_utils.quat_apply`
- motion 数据：wxyz → 先转换为 xyzw 再调用同一个 `quat_apply`

转换公式：`wxyz [w,x,y,z]` → `xyzw [x,y,z,w]` = `quat[:, [1,2,3,0]]`

#### 实现代码

```python
# ==== _init_buffers() 末尾新增 ====
if hasattr(self.cfg, 'amp') and self.cfg.amp.enable:
    self.amp_obs_dim = self.cfg.amp.amp_obs_dim  # 81
    self.num_amp_obs_steps = self.cfg.amp.num_amp_obs_steps  # 2

    # key body indices（从 URDF body_names 查找）
    amp_key_body_names = self.cfg.amp.key_body_names
    self.amp_key_body_indices = torch.zeros(len(amp_key_body_names), dtype=torch.long, device=self.device)
    for i, name in enumerate(amp_key_body_names):
        self.amp_key_body_indices[i] = self.body_names.index(name)

    # 2帧 history buffer，与 g1_amp_env.py:60-62 相同结构
    self.amp_observation_buffer = torch.zeros(
        self.num_envs, self.num_amp_obs_steps, self.amp_obs_dim, device=self.device)

    # reset 后延迟到下一次权威 refresh 再初始化 AMP history
    self.amp_reset_pending = torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device)

    # motion loader（环境侧持有，用于 reference 采样）
    from legged_gym.utils.motion_loader import MotionLoader
    motion_file = self.cfg.amp.motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    self._motion_loader = MotionLoader(motion_file, self.device)

    # 绑定 motion file 中的 dof/body 索引（参照 g1_amp_env.py:50-55）
    self.motion_dof_indices = self._motion_loader.get_dof_index(list(self.dof_names))
    self.motion_ref_body_index = self._motion_loader.get_body_index(
        [self.cfg.amp.reference_body_name])[0]
    self.motion_key_body_indices = self._motion_loader.get_body_index(
        self.cfg.amp.key_body_names)

# ==== 新增 _compute_amp_obs（统一使用 IsaacGym 的 xyzw 约定） ====
# 注意：不能用 @torch.jit.script 装饰静态方法（IsaacGym 的 quat_apply 不兼容），
# 改为普通函数放在文件顶层。
def compute_amp_obs(dof_pos, dof_vel, root_pos, root_quat_xyzw,
                    root_lin_vel, root_ang_vel, key_body_pos):
    """计算单帧 AMP observation。root_quat_xyzw 必须是 xyzw 格式。"""
    # quaternion_to_tangent_and_normal（使用 isaacgym 的 quat_apply，期望 xyzw）
    ref_tangent = torch.zeros_like(root_quat_xyzw[..., :3])
    ref_normal = torch.zeros_like(root_quat_xyzw[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(root_quat_xyzw, ref_tangent)  # isaacgym.torch_utils.quat_apply, xyzw
    normal = quat_apply(root_quat_xyzw, ref_normal)
    root_tn = torch.cat([tangent, normal], dim=-1)

    return torch.cat([
        dof_pos, dof_vel,
        root_pos[:, 2:3],  # height
        root_tn,
        root_lin_vel, root_ang_vel,
        (key_body_pos - root_pos.unsqueeze(-2)).view(key_body_pos.shape[0], -1),
    ], dim=-1)

# ==== R2Robot 新增方法 ====
def compute_amp_observations(self):
    """从 live sim 状态计算当前帧 AMP obs 并滚动更新 history buffer"""
    key_body_pos = self.rigid_body_states[:, self.amp_key_body_indices, :3]
    cur_obs = compute_amp_obs(
        self.dof_pos, self.dof_vel,
        self.root_states[:, :3],
        self.root_states[:, 3:7],   # IsaacGym xyzw，直接传入
        self.root_states[:, 7:10], self.root_states[:, 10:13],
        key_body_pos)
    # 滚动 history
    for i in reversed(range(self.num_amp_obs_steps - 1)):
        self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
    self.amp_observation_buffer[:, 0] = cur_obs

def _bootstrap_amp_buffer(self, env_ids):
    """在 refresh_*_tensor() 之后，用当前权威 live state 填满 AMP history。
    只用于处理刚 reset 完、等待初始化 history 的 env。
    这样可以避免在 reset 同帧直接读取可能 stale 的 rigid_body_states。"""
    key_body_pos = self.rigid_body_states[env_ids][:, self.amp_key_body_indices, :3]
    cur_obs = compute_amp_obs(
        self.dof_pos[env_ids], self.dof_vel[env_ids],
        self.root_states[env_ids, :3],
        self.root_states[env_ids, 3:7],
        self.root_states[env_ids, 7:10], self.root_states[env_ids, 10:13],
        key_body_pos)
    # 所有 history 槽填充同一帧（"无历史"语义）
    for i in range(self.num_amp_obs_steps):
        self.amp_observation_buffer[env_ids, i] = cur_obs

def collect_reference_motions(self, num_samples, current_times=None):
    """采样参考运动的多帧 AMP obs，仅用于 discriminator 训练。
    注意：motion_loader 返回 wxyz 四元数，需转换为 xyzw。"""
    import numpy as np
    if current_times is None:
        current_times = self._motion_loader.sample_times(num_samples)
    times = (
        np.expand_dims(current_times, axis=-1)
        - self._motion_loader.dt * np.arange(0, self.num_amp_obs_steps)
    ).flatten()
    (dof_pos, dof_vel, body_pos, body_rot_wxyz, body_lin_vel, body_ang_vel
     ) = self._motion_loader.sample(
         num_samples=num_samples * self.num_amp_obs_steps, times=times)

    # 四元数转换：wxyz → xyzw
    body_rot_xyzw = body_rot_wxyz[:, :, [1, 2, 3, 0]]

    amp_obs = compute_amp_obs(
        dof_pos[:, self.motion_dof_indices],
        dof_vel[:, self.motion_dof_indices],
        body_pos[:, self.motion_ref_body_index],
        body_rot_xyzw[:, self.motion_ref_body_index],   # 转换后的 xyzw
        body_lin_vel[:, self.motion_ref_body_index],
        body_ang_vel[:, self.motion_ref_body_index],
        body_pos[:, self.motion_key_body_indices])
    return amp_obs.view(num_samples, self.num_amp_obs_steps, -1)

# ==== post_physics_step() 开头新增 ====
# 在 refresh_*_tensor() 之后、check_termination() 之前添加：
if hasattr(self, 'amp_observation_buffer'):
    pending_env_ids = self.amp_reset_pending.nonzero(as_tuple=False).flatten()
    if len(pending_env_ids) > 0:
        self._bootstrap_amp_buffer(pending_env_ids)
        self.amp_reset_pending[pending_env_ids] = False

# ==== reset_idx() 末尾新增 ====
# 在现有 reset_idx() 的 self.gait_indices[env_ids] = 0 之后添加：
if hasattr(self, 'amp_observation_buffer') and len(env_ids) > 0:
    self.amp_reset_pending[env_ids] = True

# ==== post_physics_step() 中修改 ====
# 在 compute_reward() 之后、reset_idx(env_ids) 之前添加：
if hasattr(self, 'amp_observation_buffer'):
    self.compute_amp_observations()
    amp_obs_size = self.num_amp_obs_steps * self.amp_obs_dim
    self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, amp_obs_size)
```

#### 数据流（修正后）

```
reset_idx(env_ids):
  → _reset_dofs(env_ids)         ← R2 原有随机 reset
  → _reset_root_states(env_ids)  ← R2 原有随机 reset
  → amp_reset_pending[env_ids] = True
                                ← 只做“待初始化”标记，不在同帧读取 rigid_body_states

post_physics_step():
  → refresh_*_tensor()
  → _bootstrap_amp_buffer(pending_env_ids)
                                ← 对刚 reset 的 env，用当前权威 live state 填满 history
  → check_termination / compute_reward
  → compute_amp_observations()   ← 在 reset 之前记录本步真实状态（含 terminal 帧）
  → extras["amp_obs"] = buffer   ← 传给 AMPPPO

collect_reference_motions():     ← 仅在 discriminator 训练时调用
  → motion_loader.sample()       ← 返回 wxyz 四元数
  → wxyz → xyzw 转换             ← 对齐 IsaacGym 约定
  → compute_amp_obs()            ← 统一使用 xyzw + isaacgym.quat_apply
```

### 第4步：AMP Replay Buffer

新建 `rsl_rl/rsl_rl/storage/amp_storage.py`（与原计划相同，无变化）

```python
class AMPReplayBuffer:
    """环形缓冲区，存储 agent 的 amp_obs history"""
    def __init__(self, buffer_size, amp_obs_size, device):
        self.buffer = torch.zeros(buffer_size, amp_obs_size, device=device)
        self.buffer_size = buffer_size
        self.insert_idx = 0
        self.count = 0

    def insert(self, amp_obs_batch):
        n = amp_obs_batch.shape[0]
        if self.insert_idx + n > self.buffer_size:
            overflow = (self.insert_idx + n) - self.buffer_size
            self.buffer[self.insert_idx:] = amp_obs_batch[:n - overflow]
            self.buffer[:overflow] = amp_obs_batch[n - overflow:]
        else:
            self.buffer[self.insert_idx:self.insert_idx + n] = amp_obs_batch
        self.insert_idx = (self.insert_idx + n) % self.buffer_size
        self.count = min(self.count + n, self.buffer_size)

    def sample(self, batch_size):
        idx = torch.randint(0, self.count, (batch_size,))
        return self.buffer[idx]
```

### 第5步：运动数据加载器

新建 `legged_gym/utils/motion_loader.py`

直接从 `humanoid_amp/motions/motion_loader.py` 复制全部代码（`MotionLoader` 类本身不依赖 IsaacLab，只用了 numpy + torch）。

**不新增 `sample_amp_obs_pairs()` 方法**。reference motion 采样统一由环境侧的 `collect_reference_motions()` 完成（该方法已经绑定了正确的 dof/body 索引），AMPPPO 通过 `env.collect_reference_motions()` 调用。

### 第6步：AMP PPO 算法

新建 `rsl_rl/rsl_rl/algorithms/amp_ppo.py`

```python
from collections import defaultdict
from rsl_rl.algorithms.ppo import PPO

class AMPPPO(PPO):
    def __init__(self, actor_critic, discriminator, amp_replay_buffer,
                 env,  # 持有 env 引用，用于调用 collect_reference_motions
                 task_reward_weight=0.3, style_reward_weight=0.7,
                 disc_learning_rate=5e-5, disc_grad_penalty=5.0,
                 disc_logit_reg=0.05, disc_weight_decay=1e-4,
                 disc_reward_scale=2.0, disc_batch_size=4096,
                 **ppo_kwargs):
        super().__init__(actor_critic, **ppo_kwargs)
        self.discriminator = discriminator
        self.disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=disc_learning_rate, weight_decay=disc_weight_decay)
        self.amp_replay_buffer = amp_replay_buffer
        self.env = env
        self.task_reward_weight = task_reward_weight
        self.style_reward_weight = style_reward_weight
        self.disc_grad_penalty = disc_grad_penalty
        self.disc_logit_reg = disc_logit_reg
        self.disc_reward_scale = disc_reward_scale
        self.disc_batch_size = disc_batch_size
        self.amp_obs_collector = []
        self.style_reward_collector = []
        self.mixed_reward_collector = []

    def process_env_step(self, rewards, dones, infos):
        if 'amp_obs' not in infos:
            raise KeyError("AMP enabled but infos['amp_obs'] is missing")

        # 1. 归一化到 RL device（当前 on_policy_runner.learn() 不会搬运 infos）
        amp_obs = infos['amp_obs']
        if amp_obs.device != self.device:
            amp_obs = amp_obs.to(self.device)

        # 2. 收集 amp_obs
        self.amp_obs_collector.append(amp_obs.clone())

        # 3. 计算 style reward
        with torch.no_grad():
            disc_logit = self.discriminator(amp_obs)
            style_reward = -torch.log(
                1 - torch.sigmoid(disc_logit) + 1e-7) * self.disc_reward_scale
            style_reward = style_reward.squeeze(-1)
        self.style_reward_collector.append(style_reward.detach())

        # 4. 混合奖励
        mixed_rewards = self.task_reward_weight * rewards + self.style_reward_weight * style_reward
        self.mixed_reward_collector.append(mixed_rewards.detach())

        # 5. 调用父类
        super().process_env_step(mixed_rewards, dones, infos)

    def update(self):
        # 1. PPO 更新
        metrics = super().update()

        # 2. 存入 replay buffer
        if self.amp_obs_collector:
            all_amp_obs = torch.cat(self.amp_obs_collector, dim=0)
            self.amp_replay_buffer.insert(all_amp_obs)
            self.amp_obs_collector.clear()

        # 3. 记录 AMP 奖励日志（runner 会写到 Loss/style_reward, Loss/mixed_reward）
        if self.style_reward_collector:
            metrics['style_reward'] = torch.cat(self.style_reward_collector).mean().item()
            self.style_reward_collector.clear()
        if self.mixed_reward_collector:
            metrics['mixed_reward'] = torch.cat(self.mixed_reward_collector).mean().item()
            self.mixed_reward_collector.clear()

        # 4. 更新 discriminator
        if self.amp_replay_buffer.count > 0:
            disc_metrics = self._update_discriminator()
            metrics.update(disc_metrics)

        return metrics

    def _update_discriminator(self):
        metrics = defaultdict(float)
        half_batch = self.disc_batch_size // 2

        # agent 数据（从 replay buffer 采样）
        agent_amp_obs = self.amp_replay_buffer.sample(half_batch)

        # reference 数据（通过 env.collect_reference_motions 采样，闭环调用）
        ref_amp_obs_3d = self.env.collect_reference_motions(half_batch)
        # flatten (half_batch, num_steps, obs_dim) → (half_batch, num_steps*obs_dim)
        ref_amp_obs = ref_amp_obs_3d.view(half_batch, -1)
        if ref_amp_obs.device != self.device:
            ref_amp_obs = ref_amp_obs.to(self.device)

        # Discriminator loss (least-squares GAN)
        agent_logit = self.discriminator(agent_amp_obs)
        ref_logit = self.discriminator(ref_amp_obs)
        disc_loss = 0.5 * (agent_logit ** 2).mean() + 0.5 * ((ref_logit - 1) ** 2).mean()

        # Gradient penalty
        grad_penalty = self.discriminator.compute_grad_penalty(
            torch.cat([agent_amp_obs, ref_amp_obs], dim=0))

        # Logit regularization
        logit_reg = (agent_logit ** 2).mean() + (ref_logit ** 2).mean()

        total_loss = disc_loss + self.disc_grad_penalty * grad_penalty + self.disc_logit_reg * logit_reg

        self.disc_optimizer.zero_grad()
        total_loss.backward()
        self.disc_optimizer.step()

        metrics['disc_loss'] = disc_loss.item()
        metrics['disc_grad_penalty'] = grad_penalty.item()
        metrics['disc_agent_logit'] = agent_logit.mean().item()
        metrics['disc_ref_logit'] = ref_logit.mean().item()
        return metrics
```

### 第7步：Runner 扩展

修改 `rsl_rl/rsl_rl/runners/on_policy_runner.py`：

```python
# __init__() 末尾新增:
self.use_amp = 'amp' in train_cfg
if self.use_amp:
    self._init_amp(train_cfg['amp'])

def _init_amp(self, amp_cfg):
    from rsl_rl.modules.discriminator import AMPDiscriminator
    from rsl_rl.storage.amp_storage import AMPReplayBuffer
    from rsl_rl.algorithms.amp_ppo import AMPPPO

    amp_obs_size = amp_cfg['amp_obs_dim'] * amp_cfg.get('num_amp_obs_steps', 2)
    self.discriminator = AMPDiscriminator(
        amp_obs_dim=amp_obs_size,
        hidden_dims=amp_cfg.get('disc_hidden_dims', [1024, 512])
    ).to(self.device)

    self.amp_replay_buffer = AMPReplayBuffer(
        buffer_size=amp_cfg.get('replay_buffer_size', 1000000),
        amp_obs_size=amp_obs_size,
        device=self.device
    )

    # 用 AMPPPO 替换已创建的 PPO，传入 env 引用
    actor_critic = self.alg.actor_critic
    self.alg = AMPPPO(
        actor_critic, self.discriminator, self.amp_replay_buffer,
        env=self.env,  # 传入 env，用于 collect_reference_motions
        task_reward_weight=amp_cfg.get('task_reward_weight', 0.3),
        style_reward_weight=amp_cfg.get('style_reward_weight', 0.7),
        disc_learning_rate=amp_cfg.get('disc_learning_rate', 5e-5),
        disc_grad_penalty=amp_cfg.get('disc_grad_penalty', 5.0),
        disc_logit_reg=amp_cfg.get('disc_logit_reg', 0.05),
        disc_weight_decay=amp_cfg.get('disc_weight_decay', 1e-4),
        disc_reward_scale=amp_cfg.get('disc_reward_scale', 2.0),
        disc_batch_size=amp_cfg.get('disc_batch_size', 4096),
        device=self.device, **self.alg_cfg
    )
    self.alg.init_storage(self.env.num_envs, self.num_steps_per_env,
        [self.env.include_history_steps, self.env.num_partial_obs]
            if self.env.include_history_steps else [self.env.num_partial_obs],
        [self.env.num_obs], [self.env.num_actions])

# save() — 当前实现是 torch.save({...}, path) 一次性构造 dict（on_policy_runner.py:212-217）。
# 修改方式：在现有 dict 字面量中追加 discriminator 字段，而非事后 append。
def save(self, path, infos=None):
    save_dict = {
        'model_state_dict': self.alg.actor_critic.state_dict(),
        'optimizer_state_dict': self.alg.optimizer.state_dict(),
        'iter': self.current_learning_iteration,
        'infos': infos,
    }
    if self.use_amp:
        save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
        save_dict['disc_optimizer_state_dict'] = self.alg.disc_optimizer.state_dict()
    torch.save(save_dict, path)

# load() — 同理，在现有 load 逻辑之后追加 discriminator 恢复：
def load(self, path, load_optimizer=True, load_adaptation=False):
    loaded_dict = torch.load(path, map_location=self.device)
    self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    if load_optimizer:
        self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
    self.current_learning_iteration = loaded_dict['iter']
    if self.use_amp and 'discriminator_state_dict' in loaded_dict:
        self.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        if load_optimizer:
            self.alg.disc_optimizer.load_state_dict(loaded_dict['disc_optimizer_state_dict'])
    return loaded_dict['infos']
```

rollout 循环（`learn()` 方法）**无需修改控制流**，但前提是 `AMPPPO.process_env_step()` 必须在内部先把 `infos["amp_obs"]` 转到 `self.device`。当前 `on_policy_runner.learn()` 只会搬运 `obs / critic_obs / rewards / dones`，不会搬运 `infos`。同理，`_update_discriminator()` 中从 env 取出的 `ref_amp_obs` 也必须在喂给 discriminator 前转到 RL device。

### 第8步：配置文件

新建 `legged_gym/envs/r2/r2_amp_config.py`

motion_file 路径使用与 `asset.file` 相同的 `{LEGGED_GYM_ROOT_DIR}` 模板，在 `_init_buffers()` 中用 `.format()` 展开（见第3步代码）。运动数据存放在 `legged_gym/motions/` 目录下（需创建该目录）。

```python
from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg, R2InterruptCfgPPO

class R2AmpCfg(R2InterruptCfg):
    class amp:
        enable = True
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/r2_walk.npz"
        amp_obs_dim = 81  # 28+28+1+6+3+3+12
        num_amp_obs_steps = 2
        key_body_names = ["left_hand_roll_link", "right_hand_roll_link",
                          "left_ankle_roll_link", "right_ankle_roll_link"]
        reference_body_name = "base_link"

class R2AmpCfgPPO(R2InterruptCfgPPO):
    class runner(R2InterruptCfgPPO.runner):
        experiment_name = 'r2_amp'

    class amp:
        amp_obs_dim = 81
        num_amp_obs_steps = 2
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/r2_walk.npz"
        task_reward_weight = 0.3
        style_reward_weight = 0.7
        disc_hidden_dims = [1024, 512]
        disc_learning_rate = 5e-5
        disc_grad_penalty = 5.0
        disc_logit_reg = 0.05
        disc_weight_decay = 1e-4
        disc_reward_scale = 2.0
        disc_batch_size = 4096
        replay_buffer_size = 1000000
```

### 第9步：任务注册

修改 `legged_gym/envs/__init__.py`，增加：

```python
from legged_gym.envs.r2.r2_amp_config import R2AmpCfg, R2AmpCfgPPO
task_registry.register("r2amp", R2InterruptRobot, R2AmpCfg(), R2AmpCfgPPO())
```

---

## 运动数据准备：G1→R2 重定向

### 重定向策略：按名字匹配，不硬编码索引

G1 npz 中的 `dof_names` 顺序由 `data_convert.py:179-209` 定义，与 URDF 解析顺序一致。
R2 的 `dof_names` 顺序由 IsaacGym 加载 URDF 时决定（`r2.py:1317`），从 URDF 确认为：

```
R2 dof_names (28个，URDF 顺序):
[0]  left_hip_pitch_joint      [14] left_shoulder_pitch_joint
[1]  left_hip_roll_joint       [15] left_shoulder_roll_joint
[2]  left_hip_yaw_joint        [16] left_shoulder_yaw_joint
[3]  left_knee_joint           [17] left_arm_pitch_joint
[4]  left_ankle_pitch_joint    [18] left_arm_yaw_joint
[5]  left_ankle_roll_joint     [19] left_hand_pitch_joint
[6]  right_hip_pitch_joint     [20] left_hand_roll_joint
[7]  right_hip_roll_joint      [21] right_shoulder_pitch_joint
[8]  right_hip_yaw_joint       [22] right_shoulder_roll_joint
[9]  right_knee_joint          [23] right_shoulder_yaw_joint
[10] right_ankle_pitch_joint   [24] right_arm_pitch_joint
[11] right_ankle_roll_joint    [25] right_arm_yaw_joint
[12] waist_yaw_joint           [26] right_hand_pitch_joint
[13] waist_pitch_joint         [27] right_hand_roll_joint
```

G1 npz 中的 `dof_names`（29个，`data_convert.py:179-209` 定义）：
```
[0]  left_hip_pitch_joint      [15] left_shoulder_pitch_joint
[1]  left_hip_roll_joint       [16] left_shoulder_roll_joint
[2]  left_hip_yaw_joint        [17] left_shoulder_yaw_joint
[3]  left_knee_joint           [18] left_elbow_joint
[4]  left_ankle_pitch_joint    [19] left_wrist_roll_joint
[5]  left_ankle_roll_joint     [20] left_wrist_pitch_joint
[6]  right_hip_pitch_joint     [21] left_wrist_yaw_joint
[7]  right_hip_roll_joint      [22] right_shoulder_pitch_joint
[8]  right_hip_yaw_joint       [23] right_shoulder_roll_joint
[9]  right_knee_joint          [24] right_shoulder_yaw_joint
[10] right_ankle_pitch_joint   [25] right_elbow_joint
[11] right_ankle_roll_joint    [26] right_wrist_roll_joint
[12] waist_yaw_joint           [27] right_wrist_pitch_joint
[13] waist_roll_joint ← R2无   [28] right_wrist_yaw_joint
[14] waist_pitch_joint
```

### 重定向脚本 `retarget_motion.py` 的核心逻辑

**不硬编码索引**，而是用名字映射字典 + `MotionLoader.get_dof_index()` 按名查找：

```python
# G1 关节名 → R2 关节名 的语义映射
G1_TO_R2_DOF_MAP = {
    # 腿部 12 DOF：名字完全相同，直接映射
    "left_hip_pitch_joint":    "left_hip_pitch_joint",
    "left_hip_roll_joint":     "left_hip_roll_joint",
    "left_hip_yaw_joint":      "left_hip_yaw_joint",
    "left_knee_joint":         "left_knee_joint",
    "left_ankle_pitch_joint":  "left_ankle_pitch_joint",
    "left_ankle_roll_joint":   "left_ankle_roll_joint",
    "right_hip_pitch_joint":   "right_hip_pitch_joint",
    "right_hip_roll_joint":    "right_hip_roll_joint",
    "right_hip_yaw_joint":     "right_hip_yaw_joint",
    "right_knee_joint":        "right_knee_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
    "right_ankle_roll_joint":  "right_ankle_roll_joint",
    # 腰部 2 DOF（丢弃 waist_roll_joint）
    "waist_yaw_joint":         "waist_yaw_joint",
    "waist_pitch_joint":       "waist_pitch_joint",
    # 左臂 7→7 DOF
    "left_shoulder_pitch_joint":  "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint":   "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint":    "left_shoulder_yaw_joint",
    "left_elbow_joint":           "left_arm_pitch_joint",
    "left_wrist_roll_joint":      "left_arm_yaw_joint",
    "left_wrist_pitch_joint":     "left_hand_pitch_joint",
    "left_wrist_yaw_joint":       "left_hand_roll_joint",
    # 右臂 7→7 DOF
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint":  "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint":   "right_shoulder_yaw_joint",
    "right_elbow_joint":          "right_arm_pitch_joint",
    "right_wrist_roll_joint":     "right_arm_yaw_joint",
    "right_wrist_pitch_joint":    "right_hand_pitch_joint",
    "right_wrist_yaw_joint":      "right_hand_roll_joint",
}
# waist_roll_joint 不在映射中 → 自动丢弃
```

脚本流程：
1. 加载 G1 npz，读取 `dof_names` 数组
2. 遍历 R2 的 28 个 dof_names，在 `G1_TO_R2_DOF_MAP` 反查对应的 G1 名字
3. 用 G1 名字在 npz 的 `dof_names` 中找到源索引（按名查找，不依赖顺序）
4. 按 R2 顺序重排 `dof_positions` 和 `dof_velocities`
5. body 数据：保留 reference body + key bodies，按名查找索引
6. 对 root 高度做缩放补偿（G1 pelvis 高度 → R2 base_link 高度 0.92m）
7. 输出到 `legged_gym/motions/r2_walk.npz`（需先 `mkdir -p legged_gym/motions`）

### Body 映射（AMP key bodies）

```
G1 body (npz body_names)     R2 body (URDF link name)       用途
──────────────────────────────────────────────────────────────────
pelvis                    → base_link                    reference body
left_rubber_hand          → left_hand_roll_link          key body
right_rubber_hand         → right_hand_roll_link         key body
left_ankle_roll_link      → left_ankle_roll_link         key body (名字相同)
right_ankle_roll_link     → right_ankle_roll_link        key body (名字相同)
```

---

## 数据流总结

```
每个训练步:
  env.step(actions)
    → post_physics_step()
      → refresh_*_tensor()
      → if amp_reset_pending:
          _bootstrap_amp_buffer(pending_env_ids)   ← 对刚 reset 的 env，用当前权威 live frame
                                                     填满整个 history，消除跨 episode 污染
      → check_termination / compute_reward
      → compute_amp_observations()          ← 在 reset 之前记录本步真实状态
      → extras["amp_obs"] = buffer.view(-1, 162)
      → reset_idx(env_ids)
        → _reset_dofs / _reset_root_states  ← R2 原有随机 reset（不改）
        → amp_reset_pending[env_ids] = True ← 下个权威 refresh 再初始化 AMP history
      → compute_observations(env_ids)
    → return partial_obs, obs, rewards, dones, extras

  AMPPPO.process_env_step(rewards, dones, infos):
    → amp_obs = infos["amp_obs"]
    → if amp_obs.device != self.device: amp_obs = amp_obs.to(self.device)
    → amp_obs_collector.append(amp_obs)
    → disc_logit = discriminator(amp_obs)
    → style_reward = -log(1 - sigmoid(disc_logit)) * reward_scale
    → mixed_reward = 0.3 * task_reward + 0.7 * style_reward
    → style_reward_collector / mixed_reward_collector 记录日志统计
    → super().process_env_step(mixed_reward, ...)

  AMPPPO.update():
    → PPO.update()  (更新 actor-critic)
    → amp_replay_buffer.insert(collected amp_obs)
    → agent_batch = replay_buffer.sample(half_batch)
    → ref_batch = env.collect_reference_motions(half_batch)  ← 内部做 wxyz→xyzw 转换
    → if ref_batch.device != self.device: ref_batch = ref_batch.to(self.device)
    → disc_loss = LS-GAN + grad_penalty + logit_reg
    → metrics["style_reward"] / metrics["mixed_reward"]     ← runner 记录到 Loss/*
    → 更新 discriminator

四元数约定:
  live sim (IsaacGym):  xyzw  → compute_amp_obs 直接使用
  motion_loader 返回:   wxyz  → collect_reference_motions 内部转换为 xyzw
  compute_amp_obs:      统一接收 xyzw，使用 isaacgym.torch_utils.quat_apply
```

---

## 实施顺序

1. **运动数据重定向** — `retarget_motion.py` + 创建 `legged_gym/motions/` 目录 + 生成 `r2_walk.npz`
2. **Motion Loader** — `motion_loader.py`（从 humanoid_amp 复制，无修改）
3. **Discriminator 网络** — `discriminator.py`
4. **AMP Replay Buffer** — `amp_storage.py`
5. **环境侧 AMP obs + history + 延迟 bootstrap** — 修改 `r2.py`
6. **AMP PPO** — `amp_ppo.py`
7. **Runner 扩展（init/save/load）** — 修改 `on_policy_runner.py`
8. **配置和注册** — `r2_amp_config.py` + `__init__.py`
9. **联调测试**

---

## 验证方案

1. 运动数据验证：加载重定向后的 `r2_walk.npz`，检查 dof_names 与 R2 URDF 一致，dof_positions shape 为 (T, 28)
2. 四元数验证：对同一组 reference 四元数先做 `wxyz → xyzw` 转换，再与等价的 live sim `xyzw` 输入 `compute_amp_obs`，确认 tangent/normal 输出一致（验证转换正确，而不是把 `wxyz` 直接喂给 `compute_amp_obs`）
3. Terminal 时序验证：对 done env 打印 `infos["amp_obs"]` 对应的 root/key-body 状态，确认它来自 reset 之前的 terminal frame，而不是 reset 后初始态
4. Reset bootstrap 验证：对刚 reset 的 env，在下一次 `refresh_*_tensor()` 之后检查 `amp_observation_buffer`，确认所有 history 槽相同且不含上一 episode 残留
5. Device 验证：在 env device 与 RL device 不同的配置下，确认 `infos["amp_obs"]` 进入 discriminator / replay buffer 前已转到 RL device，训练过程中无 device mismatch
6. Discriminator 验证：初始阶段 ref_logit 应接近 1，agent_logit 应接近 0；训练后两者趋近
7. 集成验证：tensorboard 中观察 `Loss/disc_loss` 收敛，`Loss/style_reward` 上升；`Train/mean_reward` 仍然是环境原始 reward，不作为 AMP 奖励验证指标
8. 路径验证：确认 `{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/r2_walk.npz` 文件存在且 `MotionLoader` 能正常加载
