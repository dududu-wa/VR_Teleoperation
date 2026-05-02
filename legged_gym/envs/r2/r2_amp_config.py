from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg, R2InterruptCfgPPO


class R2AmpCfg(R2InterruptCfg):
    class amp:
        enable = True
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
        amp_obs_dim = 73  # 24+24+1+6+3+3+12 (24 DOF)
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
        save_best_after = 0

    class amp:
        amp_obs_dim = 73  # 24+24+1+6+3+3+12 (24 DOF)
        num_amp_obs_steps = 2
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
        disc_hidden_dims = [1024, 512]
        disc_learning_rate = 5e-5
        disc_grad_penalty = 5.0
        disc_logit_reg = 0.05
        disc_weight_decay = 1e-4
        disc_reward_scale = 2.0
        disc_batch_size = 4096
        replay_buffer_size = 1000000

    class stage2:
        enable = False
        base_policy_path = ""
        residual_hidden_dims = [256, 128]
        residual_scale = 0.2
        residual_min_scale = 0.02
        residual_action_clip = 1.0
        residual_warmup_iters = 2000
        style_reward_weight = 3.0
        gait_reward_weight = 1.0
        # Explicit arm pose recovery: defaults to cfg.init_state.default_joint_angles.
        arm_recovery_reward_weight = 1.0
        arm_recovery_sigma = 0.35
        stage2_reward_dt_scale = True
        residual_action_penalty_weight = 0.001
        gait_reward_terms = {"no_fly": 1.0}
        safety_min_base_height = 0.55
        safety_max_roll = 0.7
        safety_max_pitch = 0.8
        safety_contact_force = 1.0
        safety_dof_limit_margin = 0.02
        # 只对 hip/knee DOF 做限位检查，防止跪地
        # 排除 ankle(4,5,10,11)、waist(12,13)、手臂(14-23)
        safety_dof_check_indices = [0, 1, 2, 3, 6, 7, 8, 9]
        # 手臂 DOF 软惩罚：接近限位时按比例衰减 style reward
        # indices 14-23 对应左右手臂 5+5 个关节
        arm_dof_indices = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        arm_dof_limit_margin = 0.1          # 距限位多近开始衰减 (rad)
        arm_style_penalty_scale = 5.0       # 衰减灵敏度，越大惩罚越陡
        arm_limit_penalty_weight = 1.0
        arm_residual_multiplier = 3.0
