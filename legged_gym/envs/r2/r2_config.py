from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

NUM_ACTIONS = 24
PROPRIOCEPTION_DIM = 6 + 3 * NUM_ACTIONS
CMD_DIM = 3 + 4 + 1 + 1
TERRAIN_DIM = 221
PRIVILEGED_DIM = 3 + 1 + 2 + 1 + 6
CLOCK_INPUT = 2

class R2Cfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_observations = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM
        num_partial_obs = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT
        num_privileged_obs = None
        num_actions = NUM_ACTIONS
        observe_body_height = True
        observe_body_pitch = True
        observe_waist_roll = False
        stack_history_obs = True
        include_history_steps = 5
        obs_interval = 1
        has_privileged_info = True
        observe_gait_commands = True
        observe_body_height = True
        observe_body_pitch = True
        observe_waist_roll = False
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.92] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_pitch_joint' : -0.25,
           'left_hip_roll_joint' : 0.02,
           'left_hip_yaw_joint' : 0.0,
           'left_knee_joint' : 0.5,
           'left_ankle_pitch_joint' : -0.35,
           'left_ankle_roll_joint' : 0.0,
           'right_hip_pitch_joint' : -0.25,
           'right_hip_roll_joint' : -0.02,
           'right_hip_yaw_joint' : 0.0,
           'right_knee_joint' : 0.5,
           'right_ankle_pitch_joint' : -0.35,
           'right_ankle_roll_joint' : 0.0,
           'waist_yaw_joint' : 0.0,
           'waist_pitch_joint' : 0.0,
           'left_shoulder_pitch_joint' : 0.0,
           'left_shoulder_roll_joint' : 0.0,
           'left_shoulder_yaw_joint' : 0.0,
           'left_arm_pitch_joint' : 0.0,
           'left_arm_yaw_joint' : 0.0,
           'right_shoulder_pitch_joint' : 0.0,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.0,
           'right_arm_pitch_joint' : 0.0,
           'right_arm_yaw_joint' : 0.0,
        }
    
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle_pitch': 40,
                     'ankle_roll': 40,
                     'waist_yaw': 200,
                     'waist_pitch': 200,
                     'shoulder_pitch': 80,
                     'shoulder_roll': 80,
                     'shoulder_yaw': 80,
                     'arm_pitch': 40,
                     'arm_yaw': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 5,
                     'hip_roll': 5,
                     'hip_pitch': 5,
                     'knee': 6,
                     'ankle_pitch': 2,
                     'ankle_roll': 2,
                     'waist_yaw': 5,
                     'waist_pitch': 5,
                     'shoulder_pitch': 2,
                     'shoulder_roll': 2,
                     'shoulder_yaw': 2,
                     'arm_pitch': 1,
                     'arm_yaw': 1,
                     }  # [N*m/rad]  # [N*m*s/rad]
        torque_limits = {
                     'hip_yaw': 130,
                     'hip_roll': 130,
                     'hip_pitch': 150,
                     'knee': 150,
                     'ankle_pitch': 75,
                     'ankle_roll': 75,
                     'waist_yaw': 130,
                     'waist_pitch': 130,
                     'shoulder_pitch': 75,
                     'shoulder_roll': 75,
                     'shoulder_yaw': 75,
                     'arm_pitch': 36,
                     'arm_yaw': 36,
                     }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'trimesh'  
        measure_foot_scan = True
        terrain_proportions = [1.0]
        border_size = 10 # [m]
        max_init_terrain_level = 3  
        print_all_levels = True
        print_all_types = False

        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        curriculum = True

        measured_points_x = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        measured_points_y = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6]
        height_radius = 0.07
        scan_radius = 0.05
        # trimesh only:
        slope_treshold = 0.8  # slopes above this threshold will be corrected to vertical surfaces
        global_difficulty = [1]
        flat_std_limit = 0.04 # m
        init_difficulty = 0
        class level_property:
            random_max_height = 0.02
            max_stairs_height = 0.30
            max_obstacles_height = 0.11  # (-h~h)
            max_large_step_height = 0.25
            max_slope_angle = 0.6 # np.tan(45/180*np.pi)
        
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = CMD_DIM
        resampling_time = 10. 
        heading_command = True 
        selected_terrain_type = None
        terrain_kwargs = None
        min_vel = 0.15 
        num_bins_vel_x = 12
        num_bins_vel_yaw = 10 
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            lin_vel_y = [-0.6, 0.6]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            gait_frequency = [1.5, 3.5]
            foot_swing_height = [0.1, 0.35]
            body_height = [-0.3, 0.0]
            body_pitch = [0.0, 0.4]
            waist_roll = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            abs_vel = [0.35, 0.6]

            limit_vel_x = [-0.6, 2.0]
            limit_vel_yaw = [-1.0, 1.0] 

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False
        max_collision_force = 2
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        termination_height = True
        upper_curriculum = False
        max_contact_force = 500
        base_height_target = 0.82
        penalize_curriculum = True
        curriculum_init = 0.2
        kappa_gait_probs = 0.05
        gait_force_sigma = 50  
        gait_vel_sigma = 5  
        penalize_curriculum_sigma = 0.8
        reward_curriculum_list = ['action_rate', 'feet_stumble',
                                  'joint_power_distribution', 'feet_contact_forces',
                                  'dof_acc', 'torques',  
                                  'base_height', 'collision', 'stand_still',
                                  'lin_vel_z', 'base_height_min', 'dof_vel_limits', 
                                  'ang_vel_xy', 
                                  'hopping_symmetry',
                                  'orientation_control',
                                  'standing_air',
                                  ]
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 3.0
            dof_pos_limits = -10.0
            lin_vel_z = -0.1
            ang_vel_xy = -0.5
            hip_deviation = 0
            shoulder_deviation = 0
            hip_yaw_deviation = -0
            hip_roll_deviation = -0
            shoulder_yaw_deviation = -0
            shoulder_roll_deviation = -0
            shoulder_pitch_deviation = -0
            elbow_deviation = -0
            joint_power_distribution = -0.5
            no_fly = 0.25
            termination = -200

            dof_vel_limits = -2
            action_rate = -0.01
            feet_contact_forces = -0.2  
            feet_slip = -0.2
            feet_stumble = -0.2
            dof_acc = -2.5e-7 
            torques = -5e-6
            orientation_control = -20.0
            waist_control = -2.0
            base_height = -40
            collision = -0.0
            stand_still = -5
            tracking_contacts_shaped_force = 2
            tracking_contacts_shaped_vel = 4
            feet_clearance_cmd_linear = -30 
            feet_clearance_cmd_polynomial = -0
            hopping_symmetry = -5
            standing = 1.0
            standing_air = -1.0
            standing_joint_deviation = 0
            alive = 0.2

    class curriculum_thresholds:
        class upper_motion:
            tracking_lin_vel = 0.8  # closer to 1 is tighter
        class terrains_level:
            tracking_lin_vel = 0.8  # closer to 1 is tighter
            dof_vel_limits = 0.1 
        class terrains_type:  
            tracking_lin_vel = 0.8  # closer to 1 is tighter
        class commands:
            tracking_lin_vel = 0.8
            tracking_ang_vel = 0.4  # too tight 0.7
            tracking_contacts_shaped_force = 0.8
            tracking_contacts_shaped_vel = 0.8

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/r2_v2_with_shell_no_hand/r2v2_with_shell.urdf'
        name = "r2"
        base_name = "base_link"
        foot_name = "ankle_roll"
        auxiliary_foot_link = ["left_ankle_roll_link", "right_ankle_roll_link"]
        penalize_contacts_on = ["base_link", "waist", "hip", "knee"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        override_inertia = False
        override_com = False
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.1, 2.75]  # [0.5, 1.25]
        push_interval_s = 5
        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]
        motor_strength_range = [0.8, 1.2]
        randomize_control_latency = True
        control_latency_range = [0.0, 0.02]  # s
        randomize_link_props = True
        inertia_ratio = [0.8, 1.2]
        mass_ratio = [0.8, 1.2]
        link_com_offset = 0.01
        randomize_base_mass = True
        added_mass_range = [-3, 9]  # [-1, 3]
        base_com_x_offset_range = [-0.03, 0.03]
        base_com_y_offset_range = [-0.03, 0.03]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        max_push_vel_xy = 0.6
        max_push_ang_vel_xy = 0.6
        push_f_v_scale = 15
        push_f_steps = 30
    
class R2CfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        init_noise_std = 1.0
        max_std = 1.2
        min_std = 0.1
        model_name = "MlpAdaptModel"
        class NetModel:
            class MlpAdaptModel:               
                proprioception_dim = PROPRIOCEPTION_DIM
                cmd_dim = CMD_DIM + CLOCK_INPUT
                privileged_dim = PRIVILEGED_DIM
                terrain_dim = TERRAIN_DIM
                latent_dim = 32
                privileged_recon_dim = 3
                max_length = R2Cfg.env.include_history_steps
                actor_hidden_dims = [256, 128, 32]
                mlp_hidden_dims = [256, 128] 
            
        critic_hidden_dims = [512, 256, 128]
        critic_obs_dim = PROPRIOCEPTION_DIM + CMD_DIM + CLOCK_INPUT + PRIVILEGED_DIM + TERRAIN_DIM

        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_activation = None

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'r2_teacher'
        resume = False
        resume_path = None
        save_interval = 2000

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        use_wbc_sym_loss = False
        symmetry_loss_coef = 0.5
        sync_update = True

    



  
