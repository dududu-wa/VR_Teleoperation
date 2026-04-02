from legged_gym.envs.r2.r2_config import (
    R2Cfg,
    R2CfgPPO,
    NUM_ACTIONS,
    PROPRIOCEPTION_DIM as BASE_PROPRIOCEPTION_DIM,
    CMD_DIM as BASE_CMD_DIM,
    TERRAIN_DIM,
    PRIVILEGED_DIM as BASE_PRIVILEGED_DIM,
    CLOCK_INPUT,
)
from legged_gym import LEGGED_GYM_ROOT_DIR

PROPRIOCEPTION_DIM = BASE_PROPRIOCEPTION_DIM
INTERRUPT_IN_CMD = False
NOISE_IN_PRIVILEGE = False
EXECUTE_IN_PRIVILEGE = False
DISTURB_DIM = 0
CMD_DIM = BASE_CMD_DIM + int(INTERRUPT_IN_CMD)
PRIVILEGED_DIM = BASE_PRIVILEGED_DIM + DISTURB_DIM * NOISE_IN_PRIVILEGE + NUM_ACTIONS * EXECUTE_IN_PRIVILEGE


class R2InterruptCfg(R2Cfg):
    class env(R2Cfg.env):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
    
    class rewards(R2Cfg.rewards):
        reward_curriculum_list = ['action_rate',
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
            action_rate = -0.01
            action_rate_lower = 0
            action_rate_upper = 0
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
        use_disturb = False
        disturb_dim = DISTURB_DIM
        disturb_scale = 2
        noise_scale = []
        noise_lowerbound = []
        uniform_scale = 1 
        uniform_noise = True 
        noise_ratio = 1 
        interrupt_action_buffer = None
        start_by_curriculum = True 
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
        max_iterations = 40000
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
