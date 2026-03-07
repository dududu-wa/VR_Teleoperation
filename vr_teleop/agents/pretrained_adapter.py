"""
Pretrained model adapter for knowledge distillation from Unitree's 12-DOF locomotion model.

Components:
- UnitreeTeacher: Wraps Unitree's TorchScript JIT LSTM model, maps observations,
  produces 12-DOF leg actions as teacher signal.
- UpperBodyDefaultPolicy: Provides default upper body joint targets with optional
  noise for robustness training.
- DistillationLoss: MSE between student's 12 leg actions and teacher's 12 actions,
  with exponential coefficient decay.
"""

import torch
import torch.nn.functional as F


class UnitreeTeacher:
    """Wrapper for Unitree's pretrained 12-DOF TorchScript JIT locomotion model.

    The Unitree model is an LSTM(47->64) + MLP(64->32->12) that controls 12 leg DOFs.
    This class handles observation mapping from our 67-dim obs format to the
    Unitree 47-dim format and maintains per-env LSTM hidden/cell state.

    Unitree obs (47-dim):
        base_ang_vel(3), projected_gravity(3), dof_pos(12), dof_vel(12),
        last_actions(12), commands(3), gait_phase(2)

    Our actor obs (67-dim per step):
        base_ang_vel(3), projected_gravity(3), dof_pos_loco(13), dof_vel_loco(13),
        last_actions(13), upper_body_pos(16), commands_vel(2), command_yaw(1),
        gait_id(1), clock_input(2)
    """

    # Observation scales used by Unitree's pretrained model
    OBS_SCALES = {
        'ang_vel': 0.25,
        'dof_vel': 0.05,
        'commands_lin': 2.0,
        'commands_ang': 0.25,
    }

    def __init__(self, model_path: str, num_envs: int, device: str = 'cuda:0'):
        self.device = device
        self.num_envs = num_envs

        # Load TorchScript JIT model
        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        # LSTM hidden/cell state: (1, num_envs, 64)
        self.hidden_state = torch.zeros(1, num_envs, 64, device=device)
        self.cell_state = torch.zeros(1, num_envs, 64, device=device)
        self._has_internal_state = (
            hasattr(self.model, "hidden_state") and hasattr(self.model, "cell_state")
        )

    def reset(self, env_ids: torch.Tensor = None):
        """Reset LSTM state for specified environments (or all if None)."""
        if env_ids is None:
            self.hidden_state.zero_()
            self.cell_state.zero_()
        else:
            self.hidden_state[:, env_ids] = 0.0
            self.cell_state[:, env_ids] = 0.0

        # Some TorchScript teachers keep hidden state inside model buffers.
        if self._has_internal_state:
            self.model.hidden_state.zero_()
            self.model.cell_state.zero_()

    def _ensure_external_state(self, batch_size: int):
        """Match external LSTM state shape to current batch size."""
        if self.hidden_state.shape[1] != batch_size:
            self.hidden_state = torch.zeros(
                1, batch_size, 64, device=self.device)
            self.cell_state = torch.zeros(
                1, batch_size, 64, device=self.device)

    def map_obs(self, our_obs: torch.Tensor) -> torch.Tensor:
        """Map our 67-dim single-step obs to Unitree's 47-dim format.

        Our obs layout (67-dim):
            [0:3]   base_ang_vel        -> Unitree [0:3] (already scaled by 0.25 in our obs)
            [3:6]   projected_gravity   -> Unitree [3:6]
            [6:19]  dof_pos_loco(13)    -> Unitree [6:18] (take first 12 = legs only)
            [19:32] dof_vel_loco(13)    -> Unitree [18:30] (take first 12 = legs only)
            [32:45] last_actions(13)    -> Unitree [30:42] (take first 12 = legs only)
            [45:61] upper_body_pos(16)  -> not used by Unitree
            [61:63] commands_vel(2)     -> Unitree [42:44] (need rescale: our=1.0, unitree=2.0)
            [63:64] command_yaw(1)      -> Unitree [44:45] (need rescale: our=1.0, unitree=0.25)
            [64:65] gait_id(1)          -> not used by Unitree
            [65:67] clock_input(2)      -> Unitree [45:47]

        Note: Our obs already applies scales (ang_vel*0.25, dof_vel*0.05).
        Unitree expects the same scales, so we can pass them through directly
        for ang_vel and dof_vel. For commands, our obs uses scale=1.0 while
        Unitree uses [2.0, 2.0, 0.25], so we need to rescale.
        """
        # Handle batched obs: extract the current step (last step if 3D)
        if our_obs.dim() == 3:
            # (B, H, 67) -> take last step
            obs = our_obs[:, -1, :]
        elif our_obs.dim() == 2 and our_obs.shape[-1] > 67:
            # Flat (B, 372) = [current_obs(67), history(305)] -> take first 67
            obs = our_obs[:, :67]
        else:
            obs = our_obs

        B = obs.shape[0]
        unitree_obs = torch.zeros(B, 47, device=obs.device)

        # base_ang_vel (already scaled by 0.25 in our pipeline)
        unitree_obs[:, 0:3] = obs[:, 0:3]
        # projected_gravity
        unitree_obs[:, 3:6] = obs[:, 3:6]
        # dof_pos (12 leg joints only, skip waist_pitch at index 12 in loco)
        unitree_obs[:, 6:18] = obs[:, 6:18]
        # dof_vel (12 leg joints only, skip waist_pitch)
        unitree_obs[:, 18:30] = obs[:, 19:31]
        # last_actions (12 leg joints only, skip waist_pitch)
        unitree_obs[:, 30:42] = obs[:, 32:44]
        # commands: lin_vel_x, lin_vel_y (rescale from our 1.0 to Unitree 2.0)
        unitree_obs[:, 42:44] = obs[:, 61:63] * 2.0
        # command_yaw (rescale from our 1.0 to Unitree 0.25)
        unitree_obs[:, 44:45] = obs[:, 63:64] * 0.25
        # gait_phase / clock input
        unitree_obs[:, 45:47] = obs[:, 65:67]

        return unitree_obs

    @torch.no_grad()
    def get_action(self, our_obs: torch.Tensor) -> torch.Tensor:
        """Get 12-DOF leg actions from the teacher model.

        Args:
            our_obs: (N, 67) or (N, 372) or (N, H, 67) actor observations

        Returns:
            (N, 12) teacher leg joint actions
        """
        unitree_obs = self.map_obs(our_obs)
        batch_size = unitree_obs.shape[0]

        # 1) Try explicit-state recurrent signature: model(seq, (h, c))
        try:
            self._ensure_external_state(batch_size)
            actions, (self.hidden_state, self.cell_state) = self.model(
                unitree_obs.unsqueeze(0),  # (1, N, 47)
                (self.hidden_state, self.cell_state),
            )
            return actions.squeeze(0)  # (N, 12)
        except Exception:
            pass

        # 2) Try direct batched signature: model(obs)
        try:
            # For internal-state models, distillation uses shuffled mini-batches.
            # Reset teacher state to keep targets stateless and batch-safe.
            if self._has_internal_state:
                self.model.hidden_state.zero_()
                self.model.cell_state.zero_()
            actions = self.model(unitree_obs)
            if actions.dim() == 1:
                actions = actions.unsqueeze(0)
            return actions
        except Exception:
            pass

        # 3) Fallback: per-sample inference for models with fixed internal state
        # (e.g., hidden_state shape locked to [1, 1, 64]).
        actions_list = []
        for i in range(batch_size):
            if self._has_internal_state:
                self.model.hidden_state.zero_()
                self.model.cell_state.zero_()
            sample_actions = self.model(unitree_obs[i:i+1])
            if sample_actions.dim() == 1:
                sample_actions = sample_actions.unsqueeze(0)
            actions_list.append(sample_actions)
        return torch.cat(actions_list, dim=0)


class UpperBodyDefaultPolicy:
    """Provides default upper body joint targets during training.

    When VR is not connected, the upper body should maintain natural default
    poses. This class generates 16-DOF targets (3 waist + 7 left arm + 7 right arm,
    but since waist_pitch is in loco, this outputs for the remaining 16 upper DOFs
    that are NOT part of the loco policy).

    During training, optional noise is added for robustness.
    """

    def __init__(self, robot_cfg, num_envs: int, device: str = 'cuda:0',
                 noise_std: float = 0.02):
        self.device = device
        self.num_envs = num_envs
        self.noise_std = noise_std

        # Extract default positions for upper body DOFs (indices 15-28)
        # Plus waist_yaw (12) and waist_roll (13) which are NOT in 13-DOF loco
        # The 13-DOF loco controls: 12 legs + waist_pitch(14)
        # Upper body policy controls: waist_yaw(12), waist_roll(13), arms(15-28) = 16 DOFs
        default_pos = robot_cfg.get_default_dof_pos()

        # Indices of DOFs controlled by upper body policy
        # waist_yaw(12), waist_roll(13), left_arm(15-21), right_arm(22-28)
        self.upper_indices = [12, 13] + list(range(15, 29))
        self.num_dofs = len(self.upper_indices)  # 16

        self.default_targets = default_pos[self.upper_indices].to(device)
        self.default_targets = self.default_targets.unsqueeze(0).expand(
            num_envs, -1).clone()

    def get_targets(self, add_noise: bool = True) -> torch.Tensor:
        """Get upper body joint targets.

        Args:
            add_noise: If True, add small Gaussian noise for robustness.

        Returns:
            (num_envs, 16) upper body joint position targets
        """
        targets = self.default_targets.clone()
        if add_noise and self.noise_std > 0:
            targets += torch.randn_like(targets) * self.noise_std
        return targets


class DistillationLoss:
    """Knowledge distillation loss from Unitree teacher to student policy.

    Computes MSE between the student's first 12 action dimensions (leg joints)
    and the teacher's 12-DOF output. The coefficient decays exponentially
    so the student gradually relies on its own reward signal.
    """

    teacher_action_dim = 12  # Unitree model produces 12-DOF leg actions

    def __init__(self, teacher: UnitreeTeacher, coef: float = 1.0,
                 decay_rate: float = 0.9995):
        self.teacher = teacher
        self.coef = coef
        self.decay_rate = decay_rate

    def compute(self, student_actions: torch.Tensor,
                obs: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss (calls teacher inference inline).

        Prefer compute_precomputed() for training to avoid slow per-batch
        teacher inference and LSTM hidden state contamination.
        """
        teacher_actions = self.teacher.get_action(obs)
        loss = F.mse_loss(student_actions[:, :self.teacher_action_dim], teacher_actions)
        return self.coef * loss

    def compute_precomputed(self, student_actions: torch.Tensor,
                            teacher_actions: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss using pre-computed teacher actions.

        Args:
            student_actions: (N, 13+) student policy action means
            teacher_actions: (N, 12) pre-computed teacher actions from rollout

        Returns:
            Scalar distillation loss (coef * MSE)
        """
        loss = F.mse_loss(student_actions[:, :self.teacher_action_dim], teacher_actions)
        return self.coef * loss

    def step(self):
        """Decay the distillation coefficient (call once per PPO update)."""
        self.coef *= self.decay_rate
