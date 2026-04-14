import os
import sys
sys.path.append(os.getcwd())
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, update_class_from_dict
from isaacgym import gymapi
import numpy as np
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter
import yaml
from isaacgym import gymapi

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

    env_cfg.commands.resampling_time = 100
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

    waist_action_ids = [i for i, n in enumerate(env.dof_names) if "waist" in n.lower()]
    if len(waist_action_ids) > 0:
        print(f"[demo] lock waist joints: {[env.dof_names[i] for i in waist_action_ids]}")

    cfg_eval = {
        "timesteps": (env_cfg.env.episode_length_s) * 500 + 1,
        'cameraTrack': True, 
        'trackIndex': 0,
        "cameraInit": np.pi*8/10,  
        "cameraVel": 1*np.pi/10,
    }
    camera_rot = np.pi * 8 / 10
    camera_rot_per_sec = 1 * np.pi / 10
    camera_relative_position = np.array([1, 0, 0.8])
    track_index = 0

    look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
    env.set_camera(look_at + camera_relative_position, look_at, track_index)
    
    _, _ = env.reset()

    # keep demo mode deterministic and command-driven
    if hasattr(env, "use_disturb"):
        env.use_disturb = False
    if hasattr(env, "standing_envs_mask"):
        env.standing_envs_mask[:] = False
    obs, critic_obs, _, _, _ = env.step(torch.zeros(
            env.num_envs, env.num_actions, dtype=torch.float, device=env.device))

    timesteps = env_cfg.env.episode_length_s * 500 + 1
    for timestep in tqdm.tqdm(range(timesteps)):
        with torch.inference_mode():
            cmd_values = torch.tensor(
                [0.6, 0.0, 0.0, 2.2, 0.5, 0.5, 0.2, -0.1, 0.05, 0.0],
                device=env.device,
                dtype=env.commands.dtype,
            )
            n_cmd = min(env.commands.shape[1], cmd_values.shape[0])
            env.commands[:, :n_cmd] = cmd_values[:n_cmd]

            actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
            if len(waist_action_ids) > 0:
                actions[:, waist_action_ids] = 0.0

            obs, critic_obs, _, _, _ = env.step(actions)
            look_at = np.array(env.root_states[track_index, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
            h_scale = 1
            v_scale = 0.8
            camera_relative_position = 2 * \
                np.array([np.cos(camera_rot) * h_scale,
                         np.sin(camera_rot) * h_scale, 0.5 * v_scale])
            env.set_camera(look_at + camera_relative_position, look_at, track_index)

            if timestep % 200 == 0:
                base_lin = env.base_lin_vel[0].detach().cpu().numpy()
                cmd = env.commands[0, :3].detach().cpu().numpy()
                act_norm = torch.norm(actions[0]).item()
                print(f"[demo] step={timestep} cmd(vx,vy,wz)={cmd} base_lin={base_lin} |a|={act_norm:.3f}")

if __name__ == '__main__':
    args = get_args()
    play(args)