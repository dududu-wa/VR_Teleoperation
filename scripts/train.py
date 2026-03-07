#!/usr/bin/env python3
"""
Training entry point for G1 multi-gait intervention-robust locomotion.

Usage:
    python scripts/train.py
    python scripts/train.py --curriculum-system phase --use-intervention
    python scripts/train.py --curriculum-system lp_teacher --use-intervention
"""

import os
import sys
import argparse
import time

# Isaac Gym MUST be imported before torch
try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DEFAULT_ARGS = {
    "num_envs": 256,
    "sim_backend": "isaacgym",
    "curriculum_system": "phase",
    "max_iterations": 10000,
    "num_steps": 24,
    "save_interval": 100,
    "experiment_name": "g1_multigait",
    "initial_phase": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train G1 multi-gait policy')
    parser.add_argument('--num-envs', type=int, default=DEFAULT_ARGS["num_envs"],
                        help='Number of parallel environments')
    parser.add_argument('--sim-backend', type=str, default=DEFAULT_ARGS["sim_backend"],
                        choices=['isaacgym'],
                        help='Simulation backend entrypoint')
    parser.add_argument('--curriculum-system', type=str, default=DEFAULT_ARGS["curriculum_system"],
                        choices=['phase', 'lp_teacher'],
                        help='Training curriculum system')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_ARGS["max_iterations"],
                        help='Maximum training iterations')
    parser.add_argument('--num-steps', type=int, default=None,
                        help='Rollout steps per env per iteration')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help='Path to pretrained model checkpoint (loads weights only, fresh optimizer/curriculum)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save-interval', type=int, default=None,
                        help='Checkpoint save interval (iterations)')
    parser.add_argument('--experiment-name', type=str, default=DEFAULT_ARGS["experiment_name"],
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-intervention', action='store_true',
                        help='Enable upper-body intervention during training')
    parser.add_argument('--initial-phase', type=int, default=DEFAULT_ARGS["initial_phase"],
                        help='Starting curriculum phase (phase system only)')
    parser.add_argument('--experiment-config', type=str, default=None,
                        help='Path to configs/experiment/*.yaml')
    parser.add_argument('--teacher-model', type=str, default=None,
                        help='Path to Unitree pretrained JIT model for knowledge distillation')
    return parser.parse_args()


def main():
    args = parse_args()
    from vr_teleop.robot.g1_config import G1Config
    from vr_teleop.envs.g1_multigait_env import G1MultigaitEnv
    from vr_teleop.envs.observation import ObsConfig
    from vr_teleop.envs.reward import RewardConfig
    from vr_teleop.envs.termination import TerminationConfig
    from vr_teleop.envs.domain_rand import DomainRandConfig
    from vr_teleop.agents.runner import OnPolicyRunner
    from vr_teleop.curriculum.phase_curriculum import PhaseCurriculum, PhaseConfig
    from vr_teleop.curriculum.lp_teacher_curriculum import (
        LPTeacherCurriculum,
        LPTeacherCurriculumConfig,
    )
    from vr_teleop.intervention.intervention_generator import InterventionGenerator, InterventionConfig
    from vr_teleop.intervention.feasibility_filter import FeasibilityFilter, FeasibilityConfig
    from vr_teleop.utils.config_utils import load_experiment_config, load_yaml, get_config_path, deep_merge_dict

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ---- Optional experiment config override ----
    experiment_cfg = {}
    if args.experiment_config:
        experiment_cfg = load_experiment_config(args.experiment_config)

    def _use_cfg(arg_name: str, cfg_value):
        if cfg_value is None:
            return
        current = getattr(args, arg_name)
        if current is None or current == DEFAULT_ARGS.get(arg_name):
            setattr(args, arg_name, cfg_value)

    _use_cfg("sim_backend", experiment_cfg.get("sim_backend"))
    _use_cfg("num_envs", experiment_cfg.get("num_envs"))

    exp_cfg = experiment_cfg.get("experiment", {}) if isinstance(experiment_cfg, dict) else {}
    _use_cfg("experiment_name", exp_cfg.get("name"))

    algo_cfg = experiment_cfg.get("algo", {}) if isinstance(experiment_cfg, dict) else {}
    _use_cfg("max_iterations", algo_cfg.get("max_iterations"))
    _use_cfg("save_interval", algo_cfg.get("save_interval"))
    _use_cfg("num_steps", algo_cfg.get("num_steps_per_env"))

    cur_cfg = experiment_cfg.get("curriculum", {}) if isinstance(experiment_cfg, dict) else {}
    if isinstance(cur_cfg, dict):
        _use_cfg("curriculum_system", cur_cfg.get("system"))
    _use_cfg("initial_phase", cur_cfg.get("initial_phase"))

    int_cfg = experiment_cfg.get("intervention", {}) if isinstance(experiment_cfg, dict) else {}
    if isinstance(int_cfg, dict) and "use_disturb" in int_cfg and not args.use_intervention:
        args.use_intervention = bool(int_cfg["use_disturb"])

    env_cfg_blob = experiment_cfg.get("env", {}) if isinstance(experiment_cfg, dict) else {}
    # Support both styles:
    # 1) env: {num_envs, command_ranges...}
    # 2) env: {env: {episode_length, commands...}}
    env_cfg = env_cfg_blob.get("env", env_cfg_blob) if isinstance(env_cfg_blob, dict) else {}

    command_ranges = env_cfg.get("command_ranges")
    if command_ranges is None and isinstance(env_cfg.get("commands"), dict):
        commands_cfg = env_cfg["commands"]
        command_ranges = {
            gait: commands_cfg[gait]
            for gait in ("stand", "walk", "run")
            if gait in commands_cfg
        } or None
    gait_probs = env_cfg.get("gait_probs")
    if gait_probs is None and isinstance(env_cfg.get("commands"), dict):
        gait_probs = env_cfg["commands"].get("gait_probs")

    resampling_time = env_cfg.get("resampling_time")
    if resampling_time is None and isinstance(env_cfg.get("commands"), dict):
        resampling_time = env_cfg["commands"].get("resampling_time")

    gait_cfg = env_cfg.get("gait", {}) if isinstance(env_cfg.get("gait"), dict) else {}
    walk_freq = gait_cfg.get("walk_freq", 2.0)
    run_freq = gait_cfg.get("run_freq", 3.0)
    phase_offset = gait_cfg.get("phase_offset", 0.5)

    sim_dt = env_cfg.get("sim_dt", 0.005)
    decimation = env_cfg.get("decimation", 4)
    max_episode_length = env_cfg.get("max_episode_length")
    if max_episode_length is None:
        max_episode_length = env_cfg.get("episode_length")
    if max_episode_length is None:
        ep_s = env_cfg.get("episode_length_s")
        if ep_s is not None:
            max_episode_length = int(float(ep_s) / (float(sim_dt) * int(decimation)))
    if max_episode_length is None:
        max_episode_length = 1000

    if args.sim_backend != "isaacgym":
        raise ValueError(
            "Training backend must be 'isaacgym'. MuJoCo is only for post-training visualization."
        )

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print("Training G1 multi-gait policy")
    print(f"  Envs: {args.num_envs}")
    print(f"  Backend: {args.sim_backend}")
    print(f"  Curriculum: {args.curriculum_system}")
    print(f"  Device: {device}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Intervention: {args.use_intervention}")
    if args.curriculum_system == "phase":
        print(f"  Initial phase: {args.initial_phase}")

    # ---- Load component YAML configs ----
    config_dir = get_config_path()

    # Rewards config: YAML → RewardConfig
    rewards_yaml_path = os.path.join(config_dir, "rewards", "g1_rewards.yaml")
    reward_cfg = RewardConfig()
    if os.path.isfile(rewards_yaml_path):
        rewards_yaml = load_yaml(rewards_yaml_path).get("rewards", {})
        for reward_name, reward_def in rewards_yaml.items():
            if isinstance(reward_def, dict) and "weight" in reward_def:
                reward_cfg.weights[reward_name] = float(reward_def["weight"])
                # Also load non-weight parameters
                if reward_name == "tracking_lin_vel" and "sigma" in reward_def:
                    reward_cfg.tracking_sigma = float(reward_def["sigma"])
                if reward_name == "base_height":
                    for tgt_key, attr_name in [
                        ("target_stand", "base_height_target_stand"),
                        ("target_walk", "base_height_target_walk"),
                        ("target_run", "base_height_target_run"),
                    ]:
                        if tgt_key in reward_def:
                            setattr(reward_cfg, attr_name, float(reward_def[tgt_key]))
                if reward_name == "feet_contact_forces" and "max_force" in reward_def:
                    reward_cfg.max_contact_force = float(reward_def["max_force"])
                if reward_name == "transition_stability" and "window" in reward_def:
                    reward_cfg.transition_window = float(reward_def["window"])
                if reward_name == "feet_air_time" and "target_air_time" in reward_def:
                    reward_cfg.target_air_time = float(reward_def["target_air_time"])

    # Algo config: YAML → ppo_cfg dict
    algo_yaml_path = os.path.join(config_dir, "algo", "ppo.yaml")
    algo_yaml = {}
    if os.path.isfile(algo_yaml_path):
        algo_yaml = load_yaml(algo_yaml_path).get("algo", {})
    # Merge: experiment config overrides YAML defaults
    if algo_cfg:
        algo_yaml = deep_merge_dict(algo_yaml, algo_cfg)

    # ---- Create configs ----
    robot_cfg = G1Config()
    obs_cfg = ObsConfig()
    term_cfg_blob = experiment_cfg.get("termination", {}) if isinstance(experiment_cfg, dict) else {}
    grace_period = term_cfg_blob.get("grace_period", 0) if isinstance(term_cfg_blob, dict) else 0
    term_cfg = TerminationConfig(episode_length=max_episode_length, grace_period=grace_period)
    rand_cfg = DomainRandConfig()

    # ---- Create environment ----
    env = G1MultigaitEnv(
        num_envs=args.num_envs,
        robot_cfg=robot_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        term_cfg=term_cfg,
        rand_cfg=rand_cfg,
        sim_dt=sim_dt,
        decimation=decimation,
        max_episode_length=max_episode_length,
        device=device,
        sim_backend=args.sim_backend,
        command_ranges=command_ranges,
        gait_probs=gait_probs,
        resampling_time=resampling_time if resampling_time is not None else 10.0,
        walk_freq=walk_freq,
        run_freq=run_freq,
        phase_offset=phase_offset,
    )
    resolved_backend = getattr(env.vec_env, "backend_name", args.sim_backend)
    print(f"  Resolved backend: {resolved_backend}")

    # ---- Config dicts for runner ----
    # ActorCritic config from algo YAML
    network_cfg = algo_yaml.get('network', {})
    actor_net = network_cfg.get('actor', {})
    critic_net = network_cfg.get('critic', {})
    actor_critic_cfg = {}
    if actor_net.get('history_encoder_dims'):
        actor_critic_cfg['history_encoder_hidden'] = actor_net['history_encoder_dims']
    if actor_net.get('state_estimator_dims'):
        actor_critic_cfg['state_estimator_hidden'] = actor_net['state_estimator_dims']
    if actor_net.get('controller_dims'):
        actor_critic_cfg['controller_hidden'] = actor_net['controller_dims']
    if critic_net.get('hidden_dims'):
        actor_critic_cfg['critic_hidden_dims'] = critic_net['hidden_dims']
    if 'init_noise_std' in algo_yaml:
        actor_critic_cfg['init_noise_std'] = float(algo_yaml['init_noise_std'])
    if 'min_noise_std' in algo_yaml:
        actor_critic_cfg['min_std'] = float(algo_yaml['min_noise_std'])
    if 'max_noise_std' in algo_yaml:
        actor_critic_cfg['max_std'] = float(algo_yaml['max_noise_std'])

    # PPO config: consume all relevant keys from algo YAML
    ppo_keys = [
        'num_learning_epochs', 'num_mini_batches', 'clip_param',
        'gamma', 'lam', 'value_loss_coef', 'entropy_coef',
        'learning_rate', 'max_grad_norm', 'use_clipped_value_loss',
        'use_symmetry_loss', 'symmetry_loss_coef',
        'sync_update', 'adaptation_loss_coef',
        'schedule', 'desired_kl',
    ]
    ppo_cfg = {}
    for key in ppo_keys:
        if key in algo_yaml:
            ppo_cfg[key] = algo_yaml[key]

    runner_cfg = {
        'num_steps_per_env': int(args.num_steps if args.num_steps is not None else algo_yaml.get('num_steps_per_env', DEFAULT_ARGS['num_steps'])),
        'save_interval': int(args.save_interval if args.save_interval is not None else algo_yaml.get('save_interval', DEFAULT_ARGS['save_interval'])),
    }

    log_dir = os.path.join(args.log_dir, args.experiment_name)

    # ---- Create runner (creates actor-critic + PPO internally) ----
    runner = OnPolicyRunner(
        env=env,
        actor_critic_cfg=actor_critic_cfg,
        ppo_cfg=ppo_cfg,
        runner_cfg=runner_cfg,
        log_dir=log_dir,
        device=device,
    )

    # ---- Teacher model for knowledge distillation ----
    if args.teacher_model:
        from vr_teleop.agents.pretrained_adapter import UnitreeTeacher, DistillationLoss
        print(f"  Loading teacher model from {args.teacher_model}")
        teacher = UnitreeTeacher(
            model_path=args.teacher_model,
            num_envs=args.num_envs,
            device=device,
        )
        distill_coef = float(algo_yaml.get('distillation_coef', 1.0))
        distill_decay = float(algo_yaml.get('distillation_decay', 0.9995))
        distill_loss = DistillationLoss(
            teacher=teacher, coef=distill_coef, decay_rate=distill_decay)
        runner.alg.distillation_loss = distill_loss
        print(f"  Distillation enabled (coef={distill_coef}, decay={distill_decay})")

    # ---- Create curriculum ----
    if args.curriculum_system == "phase":
        phase_cfg = PhaseConfig(initial_phase=args.initial_phase)
        if isinstance(cur_cfg, dict):
            if "max_phase" in cur_cfg:
                phase_cfg.max_phase = int(cur_cfg["max_phase"])
            if "thresholds" in cur_cfg and isinstance(cur_cfg["thresholds"], dict):
                phase_cfg.thresholds = cur_cfg["thresholds"]
            if "velocity_ranges" in cur_cfg and isinstance(cur_cfg["velocity_ranges"], dict):
                phase_cfg.velocity_ranges = cur_cfg["velocity_ranges"]
        curriculum = PhaseCurriculum(cfg=phase_cfg)
    elif args.curriculum_system == "lp_teacher":
        lp_cfg_raw = {}
        if isinstance(cur_cfg, dict):
            lp_cfg_raw = cur_cfg.get("lp_teacher", {})
            if not isinstance(lp_cfg_raw, dict):
                lp_cfg_raw = {}
        lp_cfg = LPTeacherCurriculumConfig(
            num_bins=int(lp_cfg_raw.get("num_bins", 12)),
            ema_alpha=float(lp_cfg_raw.get("ema_alpha", 0.1)),
            exploration_prob=float(lp_cfg_raw.get("exploration_prob", 0.15)),
            min_samples_per_bin=int(lp_cfg_raw.get("min_samples_per_bin", 5)),
            temperature=float(lp_cfg_raw.get("temperature", 1.0)),
            reward_scale=float(lp_cfg_raw.get("reward_scale", 1.0)),
            fall_penalty=float(lp_cfg_raw.get("fall_penalty", 1.0)),
            transition_penalty=float(lp_cfg_raw.get("transition_penalty", 0.5)),
        )
        curriculum = LPTeacherCurriculum(cfg=lp_cfg, seed=args.seed)
    else:
        raise ValueError(f"Unsupported curriculum system: {args.curriculum_system}")

    # Wire curriculum state into periodic checkpoints
    runner.checkpoint_infos_fn = lambda: {'curriculum_state': curriculum.state_dict()}

    # ---- Create intervention generator ----
    intervention = None
    feasibility_filter = None
    if args.use_intervention:
        int_cfg_obj = InterventionConfig()
        intervention = InterventionGenerator(
            num_envs=args.num_envs,
            cfg=int_cfg_obj,
            robot_cfg=robot_cfg,
            device=torch.device(device),
        )
        feasibility_filter = FeasibilityFilter(
            num_envs=args.num_envs,
            cfg=FeasibilityConfig(),
            robot_cfg=robot_cfg,
            device=torch.device(device),
        )
        if args.curriculum_system == "phase":
            intervention.set_phase(args.initial_phase)
        else:
            intervention.set_phase(0)
        env.attach_intervention(
            generator=intervention,
            feasibility_filter=feasibility_filter,
            auto_mode=True,
        )

    # ---- Resume from checkpoint ----
    if args.resume:
        print(f"Resuming from {args.resume}")
        loaded_infos = runner.load(args.resume)
        # Restore curriculum state if saved
        if loaded_infos and 'curriculum_state' in loaded_infos:
            curriculum.load_state_dict(loaded_infos['curriculum_state'])
            print(f"  Restored curriculum state (phase={getattr(curriculum, 'phase', 'N/A')})")
    elif args.pretrained_model:
        runner.load_pretrained(args.pretrained_model)

    # ---- Training ----
    print(f"\nStarting training for {args.max_iterations} iterations...")

    # Use runner.learn() which handles rollout + update + logging
    # But we also need curriculum updates per iteration.
    # So we run learn() in chunks, updating curriculum between.

    start_time = time.time()
    iters_done = 0
    chunk_size = 10  # update curriculum every 10 iterations

    while iters_done < args.max_iterations:
        # Apply curriculum settings before each chunk
        if args.curriculum_system == "phase":
            cmd_ranges = curriculum.get_command_ranges()
            env.update_command_ranges(cmd_ranges)
            env.update_gait_probs(curriculum.get_gait_probs())
            if intervention is not None:
                intervention.set_phase(curriculum.phase)
        else:
            curriculum.sample_tasks()
            cmd_ranges = curriculum.get_command_ranges()
            env.update_command_ranges(cmd_ranges)
            env.update_gait_probs(curriculum.get_gait_probs())
            if intervention is not None:
                intervention.set_phase(curriculum.get_intervention_phase())

        # How many iterations in this chunk
        remaining = args.max_iterations - iters_done
        this_chunk = min(chunk_size, remaining)

        # Run training chunk
        runner.learn(num_learning_iterations=this_chunk)
        iters_done += this_chunk

        # Update curriculum metrics from env
        extras = env.extras
        ep_info = extras.get('episode', {})
        tracking_lin = ep_info.get('reward_tracking_lin_vel', None)
        tracking_ang = ep_info.get('reward_tracking_ang_vel', None)
        if tracking_lin is not None and tracking_ang is not None:
            tracking_reward = 0.5 * (tracking_lin + tracking_ang)
        else:
            tracking_reward = ep_info.get('reward_mean', 0.0)
        fall_rate = ep_info.get('contact_fall_rate', 0.0)
        orientation_fall = ep_info.get('orientation_fall_rate', 0.0)
        total_fall_rate = fall_rate + orientation_fall
        transition_failure = ep_info.get('transition_failure_rate', 0.0)
        ep_len = ep_info.get('episode_length_mean', 0.0)

        curriculum.update_metrics(
            tracking_reward=tracking_reward,
            fall_rate=total_fall_rate,
            transition_failure=transition_failure,
            mean_episode_length=ep_len,
            num_iterations=this_chunk,
        )

        elapsed = time.time() - start_time
        fps = (iters_done * runner_cfg['num_steps_per_env'] * args.num_envs) / max(elapsed, 1)

        if args.curriculum_system == "phase":
            promoted = curriculum.check_promotion()
            if promoted:
                print(f"\n*** Phase promoted to {curriculum.phase} "
                      f"at iteration {iters_done} ***\n")

            print(f"Iter {iters_done:5d}/{args.max_iterations} | "
                  f"Phase {curriculum.phase} | "
                  f"Reward {tracking_reward:6.3f} | "
                  f"Fall {total_fall_rate:5.3f} | "
                  f"TransFail {transition_failure:5.3f} | "
                  f"FPS {fps:7.0f}")
        else:
            summary = curriculum.get_summary()
            print(f"Iter {iters_done:5d}/{args.max_iterations} | "
                  f"LP bin {summary['bin']:2d} | "
                  f"Diff {summary['difficulty']:.3f} | "
                  f"Signal {summary['signal']:+.3f} | "
                  f"Reward {tracking_reward:6.3f} | "
                  f"Fall {total_fall_rate:5.3f} | "
                  f"FPS {fps:7.0f}")

    total_time = time.time() - start_time
    if runner.log_dir is not None:
        runner.save(
            os.path.join(
                runner.log_dir,
                f"model_{runner.current_learning_iteration}.pt"
            ),
            infos={'curriculum_state': curriculum.state_dict()},
        )
    print(f"\nTraining complete in {total_time:.0f}s")
    if args.curriculum_system == "phase":
        print(f"Final phase: {curriculum.phase}")
    else:
        summary = curriculum.get_summary()
        print(f"Final LP difficulty: {summary['difficulty']:.3f} (bin={summary['bin']})")
    runner.close()


if __name__ == '__main__':
    main()
