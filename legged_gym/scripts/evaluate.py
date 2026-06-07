import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(os.getcwd())

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import apply_cfg_override_json, get_args, task_registry


PRESETS = {
    "stand": [0.0, 0.0, 0.0, 1.60, 0.5, 0.5, 0.08, 0.00, 0.00, 0.0],
    "walk_slow": [0.5, 0.0, 0.0, 2.20, 0.5, 0.5, 0.12, 0.00, 0.00, 0.0],
    "walk_fast": [1.2, 0.0, 0.0, 2.80, 0.5, 0.5, 0.17, 0.00, 0.03, 0.0],
    "turn_left": [0.4, 0.0, 0.6, 2.20, 0.5, 0.5, 0.12, 0.00, 0.00, 0.0],
    "strafe_right": [0.0, 0.3, 0.0, 2.20, 0.5, 0.5, 0.12, 0.00, 0.00, 0.0],
}


METRIC_FIELDS = [
    "run_id",
    "task_name",
    "method_name",
    "ablation_name",
    "seed",
    "checkpoint",
    "preset_name",
    "num_episodes",
    "episode_seconds",
    "lin_vel_rmse",
    "yaw_vel_rmse",
    "task_return_mean",
    "fall_rate",
    "episode_length_mean_steps",
    "base_height_violation_rate",
    "roll_pitch_violation_rate",
    "amp_style_reward_mean",
    "amp_style_reward_raw_mean",
    "disc_ref_logit_mean",
    "disc_policy_logit_mean",
    "disc_gap_mean",
    "joint_pose_error_dtw_m",
    "key_body_error_dtw_m",
    "torque_l2_mean",
    "action_rate_l2_mean",
    "dof_acc_l2_mean",
    "wall_clock_seconds",
    "notes",
]


def _build_command_tensor(env, command_values):
    command_tensor = torch.zeros(env.commands.shape[1], device=env.device, dtype=env.commands.dtype)
    values = torch.tensor(command_values, device=env.device, dtype=env.commands.dtype)
    command_tensor[: min(command_tensor.shape[0], values.shape[0])] = values[: command_tensor.shape[0]]
    return command_tensor


def _apply_preset(env, preset_name, env_ids=None):
    command_tensor = _build_command_tensor(env, PRESETS[preset_name])
    if env_ids is None:
        env.commands[:] = command_tensor
    elif len(env_ids) > 0:
        env.commands[env_ids] = command_tensor


def _configure_eval_cfg(env_cfg, args):
    """Keep evaluation deterministic and command-driven, matching play.py behavior."""
    if args.num_envs is None:
        env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = float(args.episode_seconds)
    env_cfg.terrain.curriculum = False
    if hasattr(env_cfg.noise, "add_noise"):
        env_cfg.noise.add_noise = False
    for name in (
        "randomize_friction",
        "randomize_load",
        "randomize_gains",
        "randomize_link_props",
        "randomize_base_mass",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)
    env_cfg.commands.curriculum = False
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s
    if hasattr(env_cfg.rewards, "penalize_curriculum"):
        env_cfg.rewards.penalize_curriculum = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 1
    env_cfg.terrain.selected = False
    env_cfg.terrain.selected_terrain_type = "random_uniform"
    env_cfg.terrain.terrain_kwargs = {}


def _selected_presets(args):
    if not args.preset:
        return list(PRESETS.keys())
    names = []
    for preset in args.preset:
        if preset == "all":
            return list(PRESETS.keys())
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Expected one of {sorted(PRESETS)} or 'all'.")
        names.append(preset)
    return names


def _mean_or_none(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def _safe_rate(counts, steps):
    if steps <= 0:
        return None
    return float(counts) / float(steps)


def _init_accumulators(num_envs, device):
    zeros = torch.zeros(num_envs, dtype=torch.float, device=device)
    return {
        "lin_sq": zeros.clone(),
        "yaw_sq": zeros.clone(),
        "task_return": zeros.clone(),
        "style_return": zeros.clone(),
        "style_raw_return": zeros.clone(),
        "torque_l2": zeros.clone(),
        "action_rate_l2": zeros.clone(),
        "dof_acc_l2": zeros.clone(),
        "base_height_violations": zeros.clone(),
        "roll_pitch_violations": zeros.clone(),
        "steps": zeros.clone(),
    }


def _reset_accumulators(acc, env_ids):
    for value in acc.values():
        value[env_ids] = 0.0


def _collect_step_metrics(env, rewards, infos, actions, last_actions, prev_dof_vel, acc, amp_eval_rewards=None):
    cmd = env.commands[:, :3]
    lin_error = env.base_lin_vel[:, :2] - cmd[:, :2]
    yaw_error = env.base_ang_vel[:, 2] - cmd[:, 2]
    task_reward = infos.get("amp_task_reward", rewards)
    if amp_eval_rewards is not None:
        style_raw, style_reward = amp_eval_rewards
    else:
        style_reward = infos.get("amp_style_reward_contrib", torch.zeros_like(rewards))
        style_raw = infos.get("amp_style_reward_raw", torch.zeros_like(rewards))

    acc["lin_sq"] += torch.sum(torch.square(lin_error), dim=1)
    acc["yaw_sq"] += torch.square(yaw_error)
    acc["task_return"] += task_reward.view(-1)
    acc["style_return"] += style_reward.view(-1)
    acc["style_raw_return"] += style_raw.view(-1)
    acc["torque_l2"] += torch.sum(torch.square(env.torques), dim=1)
    acc["action_rate_l2"] += torch.sum(torch.square(actions - last_actions), dim=1)
    acc["dof_acc_l2"] += torch.sum(torch.square((env.dof_vel - prev_dof_vel) / env.dt), dim=1)

    base_height_target = getattr(env.cfg.rewards, "base_height_target", 0.78)
    acc["base_height_violations"] += (env.root_states[:, 2] < base_height_target - 0.20).float()
    acc["roll_pitch_violations"] += ((torch.abs(env.rpy[:, 0]) > 0.8) | (torch.abs(env.rpy[:, 1]) > 1.0)).float()
    acc["steps"] += 1.0


def _collect_step_trajectories(env, traj_bufs):
    """Append current dof_pos and key_body_pos to per-env trajectory buffers."""
    dof_np = env.dof_pos.detach().cpu().numpy()  # (num_envs, num_dof)
    if hasattr(env, "amp_key_body_indices"):
        kb_np = env.rigid_body_states[:, env.amp_key_body_indices, :3].detach().cpu().numpy()  # (num_envs, K, 3)
    else:
        kb_np = None

    for i in range(env.num_envs):
        traj_bufs["dof"][i].append(dof_np[i])
        if kb_np is not None:
            traj_bufs["key_body"][i].append(kb_np[i])


def _reset_traj_bufs(traj_bufs, env_ids):
    for idx in env_ids:
        i = int(idx.item())
        traj_bufs["dof"][i] = []
        traj_bufs["key_body"][i] = []


def _init_traj_bufs(num_envs):
    return {
        "dof": [[] for _ in range(num_envs)],
        "key_body": [[] for _ in range(num_envs)],
    }


# ---------------------------------------------------------------------------
# DTW helpers
# ---------------------------------------------------------------------------

def _dtw_distance_fast(seq_a, seq_b):
    """Vectorised DTW using numpy; O(N*M) time, O(min(N,M)) space.

    seq_a: (T_a, D), seq_b: (T_b, D)
    Returns the path-normalised cost  sum / (T_a + T_b).
    """
    n, d = seq_a.shape
    m = seq_b.shape[0]
    # Use two-row rolling array to save memory
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[:] = np.inf
        curr[0] = np.inf  # no free first-col insertion
        diff = np.abs(seq_a[i - 1] - seq_b)  # (m, d)
        costs = diff.mean(axis=1)  # (m,)
        for j in range(1, m + 1):
            curr[j] = costs[j - 1] + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return float(prev[m]) / (n + m)


def _compute_dtw_for_episode(env, traj_dof_list, traj_kb_list):
    """Compute best-clip DTW for one completed episode trajectory.

    traj_dof_list: list of np arrays, each shape (num_dof,)
    traj_kb_list:  list of np arrays, each shape (K, 3)  — may be empty list if no key bodies

    Returns (joint_dtw, key_body_dtw) or (None, None).
    """
    if not hasattr(env, "_motion_loader") or len(traj_dof_list) == 0:
        return None, None

    ml = env._motion_loader
    traj_dof = np.stack(traj_dof_list, axis=0)          # (T, num_dof)
    has_kb = len(traj_kb_list) > 0
    if has_kb:
        traj_kb = np.stack(traj_kb_list, axis=0)         # (T, K, 3)
    T = len(traj_dof)

    motion_dof_indices = getattr(env, "motion_dof_indices", None)
    motion_key_body_indices = getattr(env, "motion_key_body_indices", None)

    # Select DOF columns in policy trajectory that correspond to reference DOFs
    if motion_dof_indices is not None:
        pol_dof_idx = motion_dof_indices.cpu().numpy()  # indices into env.dof_pos
        pol_dof = traj_dof[:, pol_dof_idx]
    else:
        pol_dof = traj_dof

    best_joint_dtw = None
    best_kb_dtw = None

    for clip in ml._clips:
        num_frames = clip["num_frames"]
        if num_frames < 2:
            continue

        # Sample T evenly-spaced frames from reference clip
        sample_idx = np.linspace(0, num_frames - 1, T).astype(int)
        ref_dof_all = clip["dof_positions"][sample_idx].cpu().numpy()  # (T, ref_dofs)

        if motion_dof_indices is not None:
            ref_dof = ref_dof_all[:, motion_dof_indices.cpu().numpy()]
        else:
            min_d = min(pol_dof.shape[1], ref_dof_all.shape[1])
            ref_dof = ref_dof_all[:, :min_d]
            pol_dof_clip = pol_dof[:, :min_d]
        if motion_dof_indices is not None:
            pol_dof_clip = pol_dof

        jd = _dtw_distance_fast(pol_dof_clip, ref_dof)
        if best_joint_dtw is None or jd < best_joint_dtw:
            best_joint_dtw = jd

        if has_kb:
            ref_body_all = clip["body_positions"][sample_idx].cpu().numpy()  # (T, all_bodies, 3)
            if motion_key_body_indices is not None:
                kidx = motion_key_body_indices.cpu().numpy()
                ref_kb = ref_body_all[:, kidx, :]       # (T, K, 3)
                pol_kb = traj_kb
            else:
                K = min(traj_kb.shape[1], ref_body_all.shape[1])
                ref_kb = ref_body_all[:, :K, :]
                pol_kb = traj_kb[:, :K, :]

            # Compute root-relative positions (remove global translation)
            ref_root = ref_kb[:, 0:1, :]  # (T, 1, 3) — first key body as pseudo-root
            pol_root = pol_kb[:, 0:1, :]
            ref_kb_rel = (ref_kb - ref_root).reshape(T, -1)
            pol_kb_rel = (pol_kb - pol_root).reshape(T, -1)

            kd = _dtw_distance_fast(pol_kb_rel, ref_kb_rel)
            if best_kb_dtw is None or kd < best_kb_dtw:
                best_kb_dtw = kd

    return best_joint_dtw, best_kb_dtw


def _compute_amp_eval_rewards(env, runner, infos):
    if getattr(runner, "discriminator", None) is None or "amp_obs" not in infos:
        return None
    amp_cfg = getattr(runner.alg, "__dict__", {})
    with torch.no_grad():
        amp_obs = infos["amp_obs"].to(runner.device)
        disc_score = runner.discriminator(amp_obs)
        # Same AMP reward transform as AMPPPO.process_env_step.
        style_base = torch.clamp(
            1.0 - 0.25 * torch.square(disc_score - 1.0),
            min=0.0,
            max=1.0,
        ).squeeze(-1)
        style_raw = style_base * float(amp_cfg.get("disc_reward_scale", 15.0))
        style_min = float(amp_cfg.get("style_reward_min", 0.0))
        style_max = float(amp_cfg.get("style_reward_max", 15.0))
        style_raw = torch.clamp(style_raw, min=style_min, max=style_max)
        style_for_mix = style_raw
        if bool(amp_cfg.get("normalize_style_reward", False)):
            style_for_mix = (style_for_mix - style_min) / (style_max - style_min)
            style_for_mix = torch.clamp(style_for_mix, min=0.0, max=1.0)
        style_contrib = (
            float(amp_cfg.get("style_reward_weight", 1.0))
            * float(amp_cfg.get("style_reward_time_scale", 1.0))
            * style_for_mix
        )
    return style_raw.detach(), style_contrib.detach()


def _collect_disc_metrics(env, runner, infos, disc_agent_values, disc_ref_values):
    if getattr(runner, "discriminator", None) is None or "amp_obs" not in infos:
        return
    with torch.no_grad():
        amp_obs = infos["amp_obs"].to(runner.device)
        agent_logit = runner.discriminator(amp_obs).view(-1)
        disc_agent_values.extend(agent_logit.detach().cpu().numpy().tolist())
        if hasattr(env, "collect_reference_motions"):
            ref_obs = env.collect_reference_motions(amp_obs.shape[0]).view(amp_obs.shape[0], -1)
            ref_logit = runner.discriminator(ref_obs.to(runner.device)).view(-1)
            disc_ref_values.extend(ref_logit.detach().cpu().numpy().tolist())


def _finalize_done_envs(env, dones, infos, acc, episode_rows, traj_bufs):
    done_ids = dones.nonzero(as_tuple=False).flatten()
    if len(done_ids) == 0:
        return done_ids
    time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
    for env_id in done_ids:
        idx = int(env_id.item())
        steps = max(float(acc["steps"][idx].item()), 1.0)

        # DTW computation for this episode
        joint_dtw, kb_dtw = _compute_dtw_for_episode(
            env,
            traj_bufs["dof"][idx],
            traj_bufs["key_body"][idx],
        )

        episode_rows.append(
            {
                "lin_vel_rmse": float(np.sqrt(acc["lin_sq"][idx].item() / steps)),
                "yaw_vel_rmse": float(np.sqrt(acc["yaw_sq"][idx].item() / steps)),
                "task_return": float(acc["task_return"][idx].item()),
                "style_reward": float(acc["style_return"][idx].item() / steps),
                "style_reward_raw": float(acc["style_raw_return"][idx].item() / steps),
                "torque_l2": float(acc["torque_l2"][idx].item() / steps),
                "action_rate_l2": float(acc["action_rate_l2"][idx].item() / steps),
                "dof_acc_l2": float(acc["dof_acc_l2"][idx].item() / steps),
                "base_height_violation_rate": float(acc["base_height_violations"][idx].item() / steps),
                "roll_pitch_violation_rate": float(acc["roll_pitch_violations"][idx].item() / steps),
                "episode_length_steps": steps,
                "fall": 0.0 if bool(time_outs[idx].item()) else 1.0,
                "joint_dtw": joint_dtw,
                "key_body_dtw": kb_dtw,
            }
        )
    _reset_accumulators(acc, done_ids)
    _reset_traj_bufs(traj_bufs, done_ids)
    return done_ids


def _summarize_preset(args, train_cfg, preset_name, episode_rows, disc_agent_values, disc_ref_values, elapsed_s):
    agent_logit = _mean_or_none(disc_agent_values)
    ref_logit = _mean_or_none(disc_ref_values)
    override_name = "none"
    if args.cfg_override_json:
        override_name = Path(args.cfg_override_json).stem

    # DTW — filter out None episodes
    jdtw_vals = [r["joint_dtw"] for r in episode_rows if r.get("joint_dtw") is not None]
    kbdtw_vals = [r["key_body_dtw"] for r in episode_rows if r.get("key_body_dtw") is not None]

    return {
        "run_id": f"{args.task}_{override_name}_{preset_name}",
        "task_name": args.task,
        "method_name": args.task,
        "ablation_name": override_name,
        "seed": getattr(train_cfg, "seed", None),
        "checkpoint": args.checkpoint,
        "preset_name": preset_name,
        "num_episodes": len(episode_rows),
        "episode_seconds": args.episode_seconds,
        "lin_vel_rmse": _mean_or_none([row["lin_vel_rmse"] for row in episode_rows]),
        "yaw_vel_rmse": _mean_or_none([row["yaw_vel_rmse"] for row in episode_rows]),
        "task_return_mean": _mean_or_none([row["task_return"] for row in episode_rows]),
        "fall_rate": _mean_or_none([row["fall"] for row in episode_rows]),
        "episode_length_mean_steps": _mean_or_none([row["episode_length_steps"] for row in episode_rows]),
        "base_height_violation_rate": _mean_or_none([row["base_height_violation_rate"] for row in episode_rows]),
        "roll_pitch_violation_rate": _mean_or_none([row["roll_pitch_violation_rate"] for row in episode_rows]),
        "amp_style_reward_mean": _mean_or_none([row["style_reward"] for row in episode_rows]),
        "amp_style_reward_raw_mean": _mean_or_none([row["style_reward_raw"] for row in episode_rows]),
        "disc_ref_logit_mean": ref_logit,
        "disc_policy_logit_mean": agent_logit,
        "disc_gap_mean": None if ref_logit is None or agent_logit is None else ref_logit - agent_logit,
        "joint_pose_error_dtw_m": _mean_or_none(jdtw_vals) if jdtw_vals else None,
        "key_body_error_dtw_m": _mean_or_none(kbdtw_vals) if kbdtw_vals else None,
        "torque_l2_mean": _mean_or_none([row["torque_l2"] for row in episode_rows]),
        "action_rate_l2_mean": _mean_or_none([row["action_rate_l2"] for row in episode_rows]),
        "dof_acc_l2_mean": _mean_or_none([row["dof_acc_l2"] for row in episode_rows]),
        "wall_clock_seconds": elapsed_s,
        "notes": "",
    }


def _write_outputs(rows, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate(args):
    if args.cfg_override_json is not None and args.load_run is None:
        raise ValueError(
            "evaluate.py requires --load_run when --cfg_override_json is used, "
            "so the checkpoint is tied to the intended ablation run."
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_eval_cfg(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    rows = []
    for preset_name in _selected_presets(args):
        start_time = time.time()
        with torch.inference_mode():
            obs, critic_obs = env.reset()
        _apply_preset(env, preset_name)
        if hasattr(env, "use_disturb"):
            env.use_disturb = False
        if hasattr(env, "standing_envs_mask"):
            env.standing_envs_mask[:] = False

        acc = _init_accumulators(env.num_envs, env.device)
        traj_bufs = _init_traj_bufs(env.num_envs)
        episode_rows = []
        disc_agent_values = []
        disc_ref_values = []
        last_actions = torch.zeros(env.num_envs, env.num_actions, dtype=torch.float, device=env.device)
        max_steps = int(np.ceil(args.episode_seconds / env.dt)) * max(args.num_episodes, 1) + env.num_envs

        for _ in range(max_steps):
            if len(episode_rows) >= args.num_episodes:
                break
            with torch.inference_mode():
                prev_dof_vel = env.dof_vel.clone()
                actions, _ = policy.act_inference(obs, privileged_obs=critic_obs)
                obs, critic_obs, rewards, dones, infos = env.step(actions)
                amp_eval_rewards = _compute_amp_eval_rewards(env, runner, infos)
                _collect_step_metrics(env, rewards, infos, actions, last_actions, prev_dof_vel, acc, amp_eval_rewards)
                _collect_step_trajectories(env, traj_bufs)
                _collect_disc_metrics(env, runner, infos, disc_agent_values, disc_ref_values)
                done_ids = _finalize_done_envs(env, dones, infos, acc, episode_rows, traj_bufs)
                _apply_preset(env, preset_name, done_ids)
                if len(done_ids) > 0:
                    env.compute_observations(done_ids)
                    obs = env.get_observations()
                    critic_obs = env.get_privileged_observations()
                last_actions = actions.clone()
                if len(done_ids) > 0:
                    last_actions[done_ids] = 0.0

        rows.append(
            _summarize_preset(
                args,
                train_cfg,
                preset_name,
                episode_rows[: args.num_episodes],
                disc_agent_values,
                disc_ref_values,
                time.time() - start_time,
            )
        )

    _write_outputs(rows, args.output_dir)
    print(f"Saved evaluation metrics to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    evaluate(get_args())
