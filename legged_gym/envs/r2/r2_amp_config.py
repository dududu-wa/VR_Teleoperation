from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg, R2InterruptCfgPPO


class R2AmpCfg(R2InterruptCfg):
    class amp:
        enable = True
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
        # Lu et al. 2026 use state-dependent AMP routing to separate motion
        # priors. Keep one policy, but split discriminators by command semantics.
        motion_experts = {
            "walk": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/walk",
            "run": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/run",
            "jump": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/jump",
        }
        default_motion_expert = "walk"
        expert_run_velocity_threshold = 1.0
        expert_run_frequency_threshold = 2.0
        expert_jump_swing_height_threshold = 0.18
        expert_jump_body_height_threshold = 0.02
        expert_style_enabled = {"walk": True, "run": True, "jump": True}
        amp_obs_dim = 77  # 26+26+1+6+3+3+12 (26 DOF)
        num_amp_obs_steps = 2
        key_body_names = [
            "left_arm_yaw_link",    # last surviving arm link after collapse_fixed_joints=True
            "right_arm_yaw_link",   # last surviving arm link after collapse_fixed_joints=True
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        reference_body_name = "base_link"


class R2AmpCfgPPO(R2InterruptCfgPPO):
    class runner(R2InterruptCfgPPO.runner):
        experiment_name = "r2_amp"
        save_best_task_checkpoint = True
        save_top_task_checkpoints = 3
        save_best_after = 0

    class amp:
        amp_obs_dim = 77  # 26+26+1+6+3+3+12 (26 DOF)
        num_amp_obs_steps = 2
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
        # Lu et al. 2026 use state-dependent AMP routing to separate motion
        # priors. Keep one policy, but split discriminators by command semantics.
        motion_experts = {
            "walk": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/walk",
            "run": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/run",
            "jump": "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions/jump",
        }
        default_motion_expert = "walk"
        expert_run_velocity_threshold = 1.0
        expert_run_frequency_threshold = 2.0
        expert_jump_swing_height_threshold = 0.18
        expert_jump_body_height_threshold = 0.02
        expert_style_enabled = {"walk": True, "run": True, "jump": True}
        disc_hidden_dims = [1024, 512]
        disc_learning_rate = 5e-5
        disc_grad_penalty = 5.0
        disc_logit_reg = 0.05
        disc_weight_decay = 1e-4
        disc_reward_scale = 15.0
        style_reward_min = 0.0
        style_reward_max = 15.0
        normalize_style_reward = True
        task_reward_weight = 1.0
        style_reward_weight = 1.0
        # Peng et al. 2021 AMP treats style as an auxiliary RL reward; keep it
        # on the same time-integrated scale as task rewards in R2Robot.
        scale_style_reward_by_dt = True
        # Delay and cap style reward so the task policy can first learn stable
        # locomotion; AMP remains an auxiliary prior, not the dominant objective.
        style_reward_start_after = 1000
        style_reward_warmup_iterations = 2000
        style_reward_min_task_reward = None
        style_reward_max_task_ratio = 0.25
        disc_batch_size = 4096
        replay_buffer_size = 1000000
