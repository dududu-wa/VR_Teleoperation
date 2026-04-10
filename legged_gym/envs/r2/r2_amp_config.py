from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg, R2InterruptCfgPPO


class R2AmpCfg(R2InterruptCfg):
    class amp:
        enable = True
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
        amp_obs_dim = 73  # 24+24+1+6+3+3+12 (24 DOF)
        num_amp_obs_steps = 2
        key_body_names = [
            "left_hand_roll_link",
            "right_hand_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
        reference_body_name = "base_link"


class R2AmpCfgPPO(R2InterruptCfgPPO):
    class runner(R2InterruptCfgPPO.runner):
        experiment_name = "r2_amp"

    class amp:
        amp_obs_dim = 73  # 24+24+1+6+3+3+12 (24 DOF)
        num_amp_obs_steps = 2
        motion_file = "{LEGGED_GYM_ROOT_DIR}/legged_gym/motions"
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
