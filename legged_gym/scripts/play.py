import os
import sys
sys.path.append(os.getcwd())
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch
import tqdm
from isaacgym import gymapi


DEMO_SEQUENCE = (
    {
        "name": "walk",
        "duration_s": 6.0,
        "commands": [0.28, 0.0, 0.0, 1.55, 0.5, 0.5, 0.10, 0.00, 0.02, 0.0],
    },
    {
        "name": "fast_walk",
        "duration_s": 6.0,
        "commands": [0.55, 0.0, 0.0, 2.1, 0.5, 0.5, 0.13, -0.06, 0.08, 0.0],
    },
)

CAMERA_OFFSET = np.array([-2.5, -0.4, 1.15], dtype=np.float64)
LOOK_AT_OFFSET = np.array([0.6, 0.0, 0.85], dtype=np.float64)


def _build_command_tensor(env, command_values):
    command_tensor = torch.zeros(env.commands.shape[1], device=env.device, dtype=env.commands.dtype)
    values = torch.tensor(command_values, device=env.device, dtype=env.commands.dtype)
    command_tensor[: min(command_tensor.shape[0], values.shape[0])] = values[: command_tensor.shape[0]]
    return command_tensor


def _get_demo_phase(timestep, dt):
    elapsed_s = timestep * dt
    cycle_s = sum(phase["duration_s"] for phase in DEMO_SEQUENCE)
    cycle_time = elapsed_s % cycle_s

    phase_start = 0.0
    for phase in DEMO_SEQUENCE:
        phase_end = phase_start + phase["duration_s"]
        if cycle_time < phase_end:
            return phase
        phase_start = phase_end
    return DEMO_SEQUENCE[-1]


def _update_camera(env, track_index):
    base_pos = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
    planar_focus = np.array([base_pos[0], base_pos[1], 0.0], dtype=np.float64)
    env.set_camera(planar_focus + CAMERA_OFFSET, planar_focus + LOOK_AT_OFFSET, track_index)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    resume_path = train_cfg.runner.resume_path
    print(resume_path)
    
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.env.episode_length_s = 100000

    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_load = False
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_link_props = False
    env_cfg.domain_rand.randomize_base_mass = False

    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s
    env_cfg.rewards.penalize_curriculum = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = False
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    env_cfg.terrain.terrain_kwargs = {}

    # prepare     # planeenvironment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    for i in range(env.num_bodies):
        env.gym.set_rigid_body_color(env.envs[0], env.actor_handles[0], i, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    track_index = 0
    
    _, _ = env.reset()

    # keep demo mode deterministic and command-driven
    if hasattr(env, "use_disturb"):
        env.use_disturb = False
    if hasattr(env, "standing_envs_mask"):
        env.standing_envs_mask[:] = False

    initial_phase = _get_demo_phase(0, env.dt)
    env.commands[:] = _build_command_tensor(env, initial_phase["commands"])
    current_phase_name = initial_phase["name"]
    print(f"[demo] switch to {current_phase_name}: {initial_phase['commands']}")

    obs, critic_obs, _, _, _ = env.step(torch.zeros(
            env.num_envs, env.num_actions, dtype=torch.float, device=env.device))
    _update_camera(env, track_index)

    timesteps = int(env_cfg.env.episode_length_s / env.dt) + 1
    for timestep in tqdm.tqdm(range(1, timesteps)):
        with torch.inference_mode():
            phase = _get_demo_phase(timestep, env.dt)
            if phase["name"] != current_phase_name:
                current_phase_name = phase["name"]
                print(f"[demo] switch to {current_phase_name}: {phase['commands']}")

            env.commands[:] = _build_command_tensor(env, phase["commands"])

            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)

            obs, critic_obs, _, _, _ = env.step(actions)
            _update_camera(env, track_index)

            if timestep % 200 == 0:
                base_lin = env.base_lin_vel[0].detach().cpu().numpy()
                cmd = env.commands[0, :3].detach().cpu().numpy()
                act_norm = torch.norm(actions[0]).item()
                print(
                    f"[demo] step={timestep} phase={current_phase_name} "
                    f"cmd(vx,vy,wz)={cmd} base_lin={base_lin} |a|={act_norm:.3f}"
                )

if __name__ == '__main__':
    args = get_args()
    play(args)
