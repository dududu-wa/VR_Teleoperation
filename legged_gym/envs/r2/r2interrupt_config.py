from legged_gym.envs.r2.r2_config import (
    R2Cfg,
    R2CfgPPO,
    NUM_ACTIONS,
    PROPRIOCEPTION_DIM as BASE_PROPRIOCEPTION_DIM,
    CMD_DIM as BASE_CMD_DIM,
    TERRAIN_DIM,
    PRIVILEGED_DIM as BASE_PRIVILEGED_DIM,
    CLOCK_INPUT,
    R2_MIRROR_ACTION_INDICES,
    R2_MIRROR_ACTION_SIGNS,
)
from legged_gym import LEGGED_GYM_ROOT_DIR

PROPRIOCEPTION_DIM = BASE_PROPRIOCEPTION_DIM
INTERRUPT_IN_CMD = True
NOISE_IN_PRIVILEGE = False
EXECUTE_IN_PRIVILEGE = False
DISTURB_DIM = 10
CMD_DIM = BASE_CMD_DIM + int(INTERRUPT_IN_CMD)
PRIVILEGED_DIM = BASE_PRIVILEGED_DIM + DISTURB_DIM * NOISE_IN_PRIVILEGE + NUM_ACTIONS * EXECUTE_IN_PRIVILEGE
R2_INTERRUPT_MIRROR_COMMAND_INDICES = list(range(PROPRIOCEPTION_DIM, PROPRIOCEPTION_DIM + CMD_DIM))
R2_INTERRUPT_MIRROR_COMMAND_SIGNS = [1, -1, -1, 1, 1, 1, 1, 1, 1, 1]
R2_INTERRUPT_MIRROR_OBS_INDICES = (
    [0, 1, 2, 3, 4, 5]
    + [6 + i for i in R2_MIRROR_ACTION_INDICES]
    + [6 + NUM_ACTIONS + i for i in R2_MIRROR_ACTION_INDICES]
    + [6 + 2 * NUM_ACTIONS + i for i in R2_MIRROR_ACTION_INDICES]
    + R2_INTERRUPT_MIRROR_COMMAND_INDICES
    + [PROPRIOCEPTION_DIM + CMD_DIM + 1, PROPRIOCEPTION_DIM + CMD_DIM]
)
R2_INTERRUPT_MIRROR_OBS_SIGNS = (
    [-1, 1, -1, 1, -1, 1]
    + R2_MIRROR_ACTION_SIGNS
    + R2_MIRROR_ACTION_SIGNS
    + R2_MIRROR_ACTION_SIGNS
    + R2_INTERRUPT_MIRROR_COMMAND_SIGNS
    + [1, 1]
)


class R2InterruptCfg(R2Cfg):
    class env(R2Cfg.env):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
    
    class rewards(R2Cfg.rewards):
        reward_curriculum_list = ['action_rate_upper', 'action_rate_lower',
                                  'feet_stumble',
                                  'joint_power_distribution', 'feet_contact_forces',
                                  'dof_acc', 'torques',  
                                  'base_height', 'collision', 'stand_still',
                                  'lin_vel_z', 'base_height_min', 'dof_vel_limits', 
                                  'ang_vel_xy', 
                                  'hopping_symmetry',
                                  'orientation_control',
                                  'standing_air',
                                  ]
        class scales(R2Cfg.rewards.scales):
            action_rate = 0
            action_rate_lower = -0.01
            action_rate_upper = -0.01
            base_height = -40.0
            stand_still = -10.0
            standing = 2.0
            orientation_control = -10
            standing_joint_deviation = 0

            # penalize standing
            standing_air = -2
    
    class commands(R2Cfg.commands):
        num_commands = CMD_DIM
    
    class disturb:
        max_curriculum = 1.0
        use_disturb = True
        disturb_dim = DISTURB_DIM
        # Use the R2 26-DoF contract from R2Cfg.init_state.default_joint_angles:
        # legs(12), waist(2), head(2), left arm(5), right arm(5). Keep the
        # interrupt target on the full bilateral arm block and exclude head DOFs.
        disturb_action_indices = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
        disturb_scale = 2
        # Bilateral full-arm interrupt target range:
        # left shoulder pitch/roll/yaw, left arm pitch/yaw, then mirrored right arm.
        noise_scale = [
            5.2,
            3.3,
            5.5,
            3.7,
            3.7,
            5.2,
            3.3,
            5.5,
            3.7,
            3.7,
        ]
        noise_lowerbound = [
            -2.6,
            -0.3,
            -1.2,
            -1.2,
            -1.2,
            -2.6,
            -3.0,
            -4.3,
            -1.2,
            -1.2,
        ]
        uniform_scale = 1 
        uniform_noise = True 
        noise_ratio = 1 
        interrupt_action_buffer = None
        # Curriculum-style release follows Bengio et al. 2009 and Rudin et al.
        # 2021 by default; ablations may disable this to isolate disturb timing.
        start_by_curriculum = True 
        # Optional staged release caps the existing disturb_rad_curriculum at
        # discrete levels and only raises the cap after recent episodes are
        # stable. This keeps the curriculum gradual instead of jumping directly
        # from no interrupt to full interrupt.
        staged_release = False
        stage_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        # Resume-only option: keep False by default so fresh staged curricula
        # still start easy; later-stage continuation runs can initialize the
        # curriculum at the current cap. This follows the same easy-to-hard
        # curriculum idea as Bengio et al. 2009 without silently changing old runs.
        stage_init_curriculum_to_level = False
        stage_min_episodes = 512
        # Scalars keep the same gate at every stage; ablation JSONs may pass
        # per-stage lists matching stage_levels. This follows curriculum
        # learning's easy-to-hard continuation idea (Bengio et al., ICML 2009)
        # and the game-inspired legged-locomotion curriculum in Rudin et al.,
        # CoRL 2022.
        stage_min_task_return = 20.0
        stage_max_fall_rate = 0.10
        stage_monitor_noise_only = True
        # Optional expert filter for staged release gates. None keeps the
        # original all-command window; "run" makes stage advancement wait for
        # run-routed episodes, matching the July21 run-failure diagnosis.
        stage_monitor_expert = None
        # Optional profile filter for commands.profile_mixture ablations. It
        # lets staged release monitor named eval-like profiles instead of only
        # broad AMP experts, so weak stand/jump profiles cannot be hidden by
        # stronger run windows.
        stage_monitor_profiles = None
        # Adaptive staged curricula can back off when the current difficulty no
        # longer meets the gate, consistent with automatic curriculum methods
        # that adjust task difficulty to demonstrated competence.
        stage_regress_on_failure = False
        stage_regress_patience = 2
        replace_action = True 
        disturb_rad = 0.2 
        disturb_rad_curriculum = True 
        disturb_curriculum_method = 2 
        
        noise_update_step = 30 
        switch_prob = 0.005 
        interrupt_in_cmd = INTERRUPT_IN_CMD
        stand_interrupt_only = False 
        noise_curriculum_ratio = 0.5 
        disturb_in_last_action = False
        obs_target_interrupt_in_privilege = NOISE_IN_PRIVILEGE
        obs_executed_actions_in_privilege = EXECUTE_IN_PRIVILEGE
        disturb_terminate_assets = []

    
    class curriculum_thresholds(R2Cfg.curriculum_thresholds):
        class disturb:
            tracking_lin_vel = 0.6

class R2InterruptCfgPPO(R2CfgPPO):
    class runner(R2CfgPPO.runner):
        experiment_name = "r2_interrupt"
        resume = False
        resume_path = None
        max_iterations = 30000
        save_interval = 2000
    
    class policy(R2CfgPPO.policy):
        model_name = "MlpAdaptModel"
        class NetModel:
            class MlpAdaptModel:               
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim = CMD_DIM + CLOCK_INPUT
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM
                latent_dim = 32
                privileged_recon_dim = 3
                max_length = R2InterruptCfg.env.include_history_steps
                actor_hidden_dims = [256, 128, 32]
                mlp_hidden_dims = [256, 128] 
            
        critic_hidden_dims = [512, 256, 128]
        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM

    class algorithm(R2CfgPPO.algorithm):
        symmetry_obs_indices = R2_INTERRUPT_MIRROR_OBS_INDICES
        symmetry_obs_signs = R2_INTERRUPT_MIRROR_OBS_SIGNS
