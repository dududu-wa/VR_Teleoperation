import csv
import json
import os
import sys
import time
from collections import deque
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
    # Match the repository's run-category AMP prior switch
    # (configs/ablation/motion_run.json) with a forward high-speed command.
    "run": [1.6, 0.0, 0.0, 3.00, 0.5, 0.5, 0.20, 0.00, 0.03, 0.0],
    # Reuse the jump demo command from play.py so DTW evaluation can target
    # clips under legged_gym/motions/jump without inventing a second command.
    "jump": [0.0, 0.0, 0.0, 2.35, 0.0, 0.5, 0.20, 0.03, 0.00, 0.0],
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
    "survival_time_mean_s",
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


REWARD_TERM_FIELDS = [
    "run_id",
    "task_name",
    "ablation_name",
    "seed",
    "checkpoint",
    "preset_name",
    "reward_term",
    "num_episodes",
    "episode_seconds",
    "reward_return_mean",
    "reward_per_step_mean",
    "reward_per_second_mean",
    "notes",
]


TERMINATION_REASON_FIELDS = [
    "run_id",
    "task_name",
    "ablation_name",
    "seed",
    "checkpoint",
    "preset_name",
    "termination_reason",
    "termination_detail",
    "num_episodes",
    "episode_seconds",
    "count",
    "rate",
    "mean_survival_time_s",
    "notes",
]


STATE_TRACE_FIELDS = [
    "run_id",
    "task_name",
    "ablation_name",
    "seed",
    "checkpoint",
    "preset_name",
    "episode_index",
    "env_id",
    "episode_step",
    "steps_until_done",
    "time_s",
    "termination_reason",
    "termination_detail",
    "base_z",
    "roll",
    "pitch",
    "yaw",
    "base_lin_x",
    "base_lin_y",
    "base_lin_z",
    "base_ang_yaw",
    "cmd_lin_x",
    "cmd_lin_y",
    "cmd_yaw",
    "lin_vel_error",
    "yaw_vel_error",
    "contact_force_max",
    "contact_body",
    "base_height_target",
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


def _validate_eval_disturb_ratio(args):
    if args.eval_disturb_ratio is None:
        return
    if args.eval_disturb_ratio < 0.0 or args.eval_disturb_ratio > 1.0:
        raise ValueError("--eval_disturb_ratio must be between 0.0 and 1.0")


def _disable_eval_disturbance(env, env_ids):
    """Clear applied interrupts while preserving the training reward contract."""
    cfg_disturb = getattr(getattr(env, "cfg", None), "disturb", None)
    env.use_disturb = bool(
        getattr(cfg_disturb, "use_disturb", getattr(env, "use_disturb", False))
    )
    # env.reset() runs under inference_mode in evaluate.py, so interrupt
    # buffers touched during reset may be inference tensors. Mutate them under
    # the same mode while keeping the rollout disturbance-free.
    with torch.inference_mode():
        if hasattr(env, "noise_disturb_mode"):
            env.noise_disturb_mode[env_ids] = False
        if hasattr(env, "disturb_isnoise"):
            env.disturb_isnoise[env_ids] = False
        if hasattr(env, "disturb_rad_curriculum"):
            env.disturb_rad_curriculum[env_ids] = 0.0
        if hasattr(env, "disturb_masks"):
            env.disturb_masks[env_ids] = False
        if hasattr(env, "interrupt_mask"):
            env.interrupt_mask[env_ids] = False
        if hasattr(env, "disturb_actions"):
            env.disturb_actions[env_ids] = 0.0


def _apply_eval_disturbance(env, args, env_ids=None):
    """Disable applied interrupts by default or force a fixed robustness ratio."""
    if not hasattr(env, "use_disturb"):
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    if args.eval_disturb_ratio is None:
        # Default evaluation should remove the applied perturbation, not switch
        # R2InterruptRobot to R2Robot reward semantics. The interrupt reward
        # masks are part of the training objective, so task returns stay
        # comparable to train.log while disturb_masks/interrupt_mask stay off.
        _disable_eval_disturbance(env, env_ids)
        return

    env.use_disturb = True
    with torch.inference_mode():
        if hasattr(env, "noise_disturb_mode"):
            env.noise_disturb_mode[env_ids] = True
        if hasattr(env, "disturb_isnoise"):
            env.disturb_isnoise[env_ids] = True
        if hasattr(env, "disturb_rad_curriculum"):
            env.disturb_rad_curriculum[env_ids] = float(args.eval_disturb_ratio)
        if hasattr(env, "disturb_masks"):
            env.disturb_masks[env_ids] = float(args.eval_disturb_ratio) > 0.0
        if hasattr(env, "interrupt_mask"):
            if getattr(env, "disturb_replace_action", False):
                env.interrupt_mask[env_ids] = env.disturb_masks[env_ids]
            else:
                env.interrupt_mask[env_ids] = (
                    env.disturb_masks[env_ids] * (~env.disturb_isnoise[env_ids])
                )
        if float(args.eval_disturb_ratio) > 0.0 and hasattr(env, "disturb_actions"):
            if getattr(env, "disturb_uniform", False) and hasattr(
                env, "Uniform_disturb_resample"
            ):
                sampled_disturb = env.Uniform_disturb_resample()
            elif hasattr(env, "Gaussian_disturb_resample"):
                sampled_disturb = env.Gaussian_disturb_resample()
            else:
                sampled_disturb = None
            if sampled_disturb is not None:
                # Only refresh reset environments so a robustness sweep does
                # not change active episodes' disturbance targets.
                env.disturb_actions[env_ids] = (
                    sampled_disturb[env_ids] / env.cfg.control.action_scale
                )


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


def _init_reward_term_accumulators(env):
    if not getattr(env, "record_reward_terms", False):
        return None
    zeros = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    reward_names = sorted(getattr(env, "reward_scales", {}).keys())
    return {name: zeros.clone() for name in reward_names}


def _collect_reward_terms(env, reward_acc):
    if reward_acc is None:
        return
    for name, values in getattr(env, "last_reward_terms", {}).items():
        values = values.view(-1)
        if name not in reward_acc:
            reward_acc[name] = torch.zeros_like(values)
        reward_acc[name] += values


def _reset_reward_term_accumulators(reward_acc, env_ids):
    if reward_acc is None:
        return
    for value in reward_acc.values():
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


def _init_state_trace_buffers(num_envs, window_steps):
    window_steps = max(int(window_steps), 1)
    return {
        "buffers": [deque(maxlen=window_steps) for _ in range(num_envs)],
        "steps": [0 for _ in range(num_envs)],
    }


def _contact_trace_detail(env, env_index, pre_reset_state=None):
    if not hasattr(env, "termination_contact_indices") or len(env.termination_contact_indices) == 0:
        return 0.0, ""
    contact_forces = (
        pre_reset_state.get("contact_forces", env.contact_forces)
        if pre_reset_state is not None
        else env.contact_forces
    )
    contact_norm = torch.norm(
        contact_forces[env_index, env.termination_contact_indices, :],
        dim=-1,
    )
    max_contact = int(torch.argmax(contact_norm).item())
    force = float(contact_norm[max_contact].item())
    body_index = int(env.termination_contact_indices[max_contact].item())
    body_names = getattr(env, "body_names", [])
    body_name = body_names[body_index] if body_index < len(body_names) else str(body_index)
    return force, body_name


def _detect_termination_reason(env, infos, env_id):
    """Classify a completed episode using the same buffers as check_termination()."""
    idx = int(env_id.item()) if torch.is_tensor(env_id) else int(env_id)
    pre_reset_state = getattr(env, "eval_pre_reset_state", None)
    time_outs = infos.get(
        "time_outs",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    if pre_reset_state is not None and "time_out_buf" in pre_reset_state:
        time_outs = pre_reset_state["time_out_buf"]
    if bool(time_outs[idx].item()):
        return "timeout", ""

    contact_force, contact_body = _contact_trace_detail(env, idx, pre_reset_state)
    if contact_force > 1.0:
        return "contact", contact_body

    rpy = (
        pre_reset_state.get("rpy", env.rpy)
        if pre_reset_state is not None
        else env.rpy
    )
    large_orientation = (torch.abs(rpy[idx, 1]) > 1.0) | (torch.abs(rpy[idx, 0]) > 0.8)
    if bool(large_orientation.item()):
        return "orientation", "roll_pitch"

    base_height_target = getattr(getattr(env.cfg, "rewards", None), "base_height_target", None)
    if base_height_target is not None:
        # Height is currently diagnostic only: check_termination() does not
        # reset on this threshold, but it helps explain near-fall rollouts.
        root_states = (
            pre_reset_state.get("root_states", env.root_states)
            if pre_reset_state is not None
            else env.root_states
        )
        if float(root_states[idx, 2].item()) < float(base_height_target) - 0.20:
            return "base_height", ""

    return "unknown", ""


def _append_state_trace(env, state_trace, preset_name, dones=None):
    if state_trace is None:
        return
    base_height_target = float(getattr(env.cfg.rewards, "base_height_target", 0.0))
    pre_reset_state = getattr(env, "eval_pre_reset_state", None)
    for env_index in range(env.num_envs):
        use_pre_reset = (
            dones is not None
            and bool(dones[env_index].item())
            and pre_reset_state is not None
        )
        root_states = pre_reset_state["root_states"] if use_pre_reset else env.root_states
        rpy = pre_reset_state["rpy"] if use_pre_reset else env.rpy
        base_lin_vel = pre_reset_state["base_lin_vel"] if use_pre_reset else env.base_lin_vel
        base_ang_vel = pre_reset_state["base_ang_vel"] if use_pre_reset else env.base_ang_vel
        commands = pre_reset_state["commands"] if use_pre_reset else env.commands
        state_trace["steps"][env_index] += 1
        cmd = commands[env_index, :3]
        base_lin = base_lin_vel[env_index]
        lin_error = torch.norm(base_lin[:2] - cmd[:2])
        yaw_error = base_ang_vel[env_index, 2] - cmd[2]
        contact_force, contact_body = _contact_trace_detail(
            env,
            env_index,
            pre_reset_state if use_pre_reset else None,
        )
        state_trace["buffers"][env_index].append(
            {
                "preset_name": preset_name,
                "env_id": env_index,
                "episode_step": int(state_trace["steps"][env_index]),
                "time_s": float(state_trace["steps"][env_index] * env.dt),
                "base_z": float(root_states[env_index, 2].item()),
                "roll": float(rpy[env_index, 0].item()),
                "pitch": float(rpy[env_index, 1].item()),
                "yaw": float(rpy[env_index, 2].item()),
                "base_lin_x": float(base_lin[0].item()),
                "base_lin_y": float(base_lin[1].item()),
                "base_lin_z": float(base_lin[2].item()),
                "base_ang_yaw": float(base_ang_vel[env_index, 2].item()),
                "cmd_lin_x": float(cmd[0].item()),
                "cmd_lin_y": float(cmd[1].item()),
                "cmd_yaw": float(cmd[2].item()),
                "lin_vel_error": float(lin_error.item()),
                "yaw_vel_error": float(yaw_error.item()),
                "contact_force_max": contact_force,
                "contact_body": contact_body,
                "base_height_target": base_height_target,
                "notes": "",
            }
        )


def _flush_state_trace_episode(
    args,
    train_cfg,
    preset_name,
    episode_index,
    env_id,
    termination_reason,
    termination_detail,
    state_trace,
    state_trace_rows,
):
    if state_trace is None or state_trace_rows is None:
        return
    idx = int(env_id.item()) if torch.is_tensor(env_id) else int(env_id)
    samples = list(state_trace["buffers"][idx])
    if not samples:
        return
    done_step = samples[-1]["episode_step"]
    override_name = "none"
    if args.cfg_override_json:
        override_name = Path(args.cfg_override_json).stem
    for sample in samples:
        row = {
            "run_id": f"{args.task}_{override_name}_{preset_name}_state_trace",
            "task_name": args.task,
            "ablation_name": override_name,
            "seed": getattr(train_cfg, "seed", None),
            "checkpoint": args.checkpoint,
            "episode_index": episode_index,
            "steps_until_done": int(done_step - sample["episode_step"]),
            "termination_reason": termination_reason,
            "termination_detail": termination_detail or "",
        }
        row.update(sample)
        state_trace_rows.append(row)


def _reset_state_trace_buffers(state_trace, env_ids):
    if state_trace is None:
        return
    for idx in env_ids:
        env_index = int(idx.item()) if torch.is_tensor(idx) else int(idx)
        state_trace["buffers"][env_index].clear()
        state_trace["steps"][env_index] = 0


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


def _has_amp_discriminator(runner):
    return (
        getattr(runner, "discriminator", None) is not None
        or getattr(runner, "discriminators", None) is not None
    )


def _routed_discriminator_score(runner, amp_obs, expert_ids=None):
    discriminators = getattr(runner, "discriminators", None)
    if discriminators is None:
        return runner.discriminator(amp_obs)

    expert_names = list(discriminators.keys())
    if expert_ids is None:
        # Single-expert evaluation keeps the historical discriminator contract.
        expert_ids = torch.zeros(amp_obs.shape[0], dtype=torch.long, device=amp_obs.device)
    elif not torch.is_tensor(expert_ids):
        expert_ids = torch.as_tensor(expert_ids, dtype=torch.long, device=amp_obs.device)
    else:
        expert_ids = expert_ids.to(device=amp_obs.device, dtype=torch.long)
    expert_ids = expert_ids.view(-1)

    if expert_ids.shape[0] != amp_obs.shape[0]:
        raise ValueError(
            f"AMP expert id batch size {expert_ids.shape[0]} does not match AMP obs batch size {amp_obs.shape[0]}"
        )
    if expert_ids.numel() > 0:
        invalid = (expert_ids < 0) | (expert_ids >= len(expert_names))
        if torch.any(invalid):
            bad_ids = torch.unique(expert_ids[invalid]).detach().cpu().tolist()
            raise ValueError(f"Invalid AMP expert ids: {bad_ids}")

    # Mirrors AMPPPO multi-expert routing: score each sample with its expert's
    # discriminator before applying the shared AMP reward transform.
    scores = torch.empty(amp_obs.shape[0], 1, device=amp_obs.device)
    for expert_idx, expert_name in enumerate(expert_names):
        mask = expert_ids == expert_idx
        if torch.any(mask):
            scores[mask] = discriminators[expert_name](amp_obs[mask])
    return scores


def _apply_expert_style_enabled(runner, style_contrib, expert_ids=None):
    """Mirror AMPPPO selective expert style masking during evaluation."""
    expert_style_enabled = getattr(runner.alg, "expert_style_enabled", None)
    expert_names = getattr(runner.alg, "expert_names", None)
    if not expert_style_enabled or not expert_names:
        return style_contrib

    if expert_ids is None:
        expert_ids = torch.zeros(
            style_contrib.shape[0], dtype=torch.long, device=style_contrib.device
        )
    elif not torch.is_tensor(expert_ids):
        expert_ids = torch.as_tensor(expert_ids, dtype=torch.long, device=style_contrib.device)
    else:
        expert_ids = expert_ids.to(device=style_contrib.device, dtype=torch.long)
    expert_ids = expert_ids.view(-1)
    if expert_ids.shape[0] != style_contrib.shape[0]:
        raise ValueError(
            f"AMP expert id batch size {expert_ids.shape[0]} does not match style reward batch size {style_contrib.shape[0]}"
        )

    for expert_idx, expert_name in enumerate(expert_names):
        if not expert_style_enabled.get(expert_name, True):
            style_contrib = style_contrib.masked_fill(expert_ids == expert_idx, 0.0)
    return style_contrib


def _compute_amp_eval_rewards(env, runner, infos):
    if not _has_amp_discriminator(runner) or "amp_obs" not in infos:
        return None
    amp_cfg = getattr(runner.alg, "__dict__", {})
    with torch.no_grad():
        amp_obs = infos["amp_obs"].to(runner.device)
        disc_score = _routed_discriminator_score(runner, amp_obs, infos.get("amp_expert_id"))
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
        style_contrib = _apply_expert_style_enabled(
            runner, style_contrib, infos.get("amp_expert_id")
        )
    return style_raw.detach(), style_contrib.detach()


def _collect_disc_metrics(env, runner, infos, disc_agent_values, disc_ref_values):
    if not _has_amp_discriminator(runner) or "amp_obs" not in infos:
        return
    with torch.no_grad():
        amp_obs = infos["amp_obs"].to(runner.device)
        expert_ids = infos.get("amp_expert_id")
        agent_logit = _routed_discriminator_score(runner, amp_obs, expert_ids).view(-1)
        disc_agent_values.extend(agent_logit.detach().cpu().numpy().tolist())
        if hasattr(env, "collect_reference_motions"):
            if expert_ids is None:
                ref_obs = env.collect_reference_motions(amp_obs.shape[0]).view(amp_obs.shape[0], -1)
            else:
                ref_obs = env.collect_reference_motions(
                    amp_obs.shape[0],
                    expert_ids=expert_ids,
                ).view(amp_obs.shape[0], -1)
            ref_logit = _routed_discriminator_score(
                runner,
                ref_obs.to(runner.device),
                expert_ids,
            ).view(-1)
            disc_ref_values.extend(ref_logit.detach().cpu().numpy().tolist())


def _finalize_done_envs(
    env,
    dones,
    infos,
    acc,
    episode_rows,
    traj_bufs,
    compute_dtw,
    reward_acc=None,
    record_termination_reasons=False,
    state_trace=None,
    state_trace_rows=None,
    args=None,
    train_cfg=None,
    preset_name=None,
    max_episodes=None,
):
    done_ids = dones.nonzero(as_tuple=False).flatten()
    if len(done_ids) == 0:
        return done_ids
    time_outs = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
    for env_id in done_ids:
        should_record_episode = max_episodes is None or len(episode_rows) < max_episodes
        if not should_record_episode:
            continue
        idx = int(env_id.item())
        steps = max(float(acc["steps"][idx].item()), 1.0)

        # DTW is O(T * motion_frames) per completed episode, so keep it opt-in
        # for targeted imitation checks rather than every fixed-preset eval.
        if compute_dtw:
            joint_dtw, kb_dtw = _compute_dtw_for_episode(
                env,
                traj_bufs["dof"][idx],
                traj_bufs["key_body"][idx],
            )
        else:
            joint_dtw, kb_dtw = None, None
        reward_terms = {}
        if reward_acc is not None:
            reward_terms = {
                name: float(values[idx].item())
                for name, values in reward_acc.items()
            }
        should_record_reason = record_termination_reasons or state_trace is not None
        termination_reason, termination_detail = (
            _detect_termination_reason(env, infos, env_id)
            if should_record_reason
            else (None, None)
        )
        episode_index = len(episode_rows)
        _flush_state_trace_episode(
            args,
            train_cfg,
            preset_name,
            episode_index,
            env_id,
            termination_reason,
            termination_detail,
            state_trace,
            state_trace_rows,
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
                "survival_time_s": steps * float(env.dt),
                "joint_dtw": joint_dtw,
                "key_body_dtw": kb_dtw,
                "reward_terms": reward_terms,
                "termination_reason": termination_reason,
                "termination_detail": termination_detail,
            }
        )
    _reset_accumulators(acc, done_ids)
    _reset_reward_term_accumulators(reward_acc, done_ids)
    _reset_traj_bufs(traj_bufs, done_ids)
    _reset_state_trace_buffers(state_trace, done_ids)
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
        "survival_time_mean_s": _mean_or_none([row["survival_time_s"] for row in episode_rows]),
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


def _summarize_reward_terms(args, train_cfg, preset_name, episode_rows):
    override_name = "none"
    if args.cfg_override_json:
        override_name = Path(args.cfg_override_json).stem
    reward_terms = sorted(
        {
            name
            for row in episode_rows
            for name in row.get("reward_terms", {}).keys()
        }
    )
    rows = []
    for reward_term in reward_terms:
        returns = [
            row.get("reward_terms", {}).get(reward_term, 0.0)
            for row in episode_rows
        ]
        per_step = [
            value / max(float(row["episode_length_steps"]), 1.0)
            for value, row in zip(returns, episode_rows)
        ]
        per_second = [
            value / max(float(args.episode_seconds), 1e-6)
            for value in returns
        ]
        rows.append(
            {
                "run_id": f"{args.task}_{override_name}_{preset_name}_{reward_term}",
                "task_name": args.task,
                "ablation_name": override_name,
                "seed": getattr(train_cfg, "seed", None),
                "checkpoint": args.checkpoint,
                "preset_name": preset_name,
                "reward_term": reward_term,
                "num_episodes": len(episode_rows),
                "episode_seconds": args.episode_seconds,
                "reward_return_mean": _mean_or_none(returns),
                "reward_per_step_mean": _mean_or_none(per_step),
                "reward_per_second_mean": _mean_or_none(per_second),
                "notes": "",
            }
        )
    return rows


def _summarize_termination_reasons(args, train_cfg, preset_name, episode_rows):
    override_name = "none"
    if args.cfg_override_json:
        override_name = Path(args.cfg_override_json).stem
    total = max(len(episode_rows), 1)
    reasons = sorted(
        {
            (
                row.get("termination_reason", "unknown") or "unknown",
                row.get("termination_detail", "") or "",
            )
            for row in episode_rows
        }
    )
    rows = []
    for reason, detail in reasons:
        matching = [
            row
            for row in episode_rows
            if (row.get("termination_reason", "unknown") or "unknown") == reason
            and (row.get("termination_detail", "") or "") == detail
        ]
        rows.append(
            {
                "run_id": f"{args.task}_{override_name}_{preset_name}_{reason}_{detail or 'none'}",
                "task_name": args.task,
                "ablation_name": override_name,
                "seed": getattr(train_cfg, "seed", None),
                "checkpoint": args.checkpoint,
                "preset_name": preset_name,
                "termination_reason": reason,
                "termination_detail": detail,
                "num_episodes": len(episode_rows),
                "episode_seconds": args.episode_seconds,
                "count": len(matching),
                "rate": float(len(matching)) / float(total),
                "mean_survival_time_s": _mean_or_none(
                    [row["survival_time_s"] for row in matching]
                ),
                "notes": "",
            }
        )
    return rows


def _write_outputs(
    rows,
    output_dir,
    reward_term_rows=None,
    termination_reason_rows=None,
    state_trace_rows=None,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    if reward_term_rows is not None:
        with open(out_dir / "reward_terms.json", "w", encoding="utf-8") as f:
            json.dump(reward_term_rows, f, indent=2)
        with open(out_dir / "reward_terms.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REWARD_TERM_FIELDS)
            writer.writeheader()
            for row in reward_term_rows:
                writer.writerow(row)
    if termination_reason_rows is not None:
        with open(out_dir / "termination_reasons.json", "w", encoding="utf-8") as f:
            json.dump(termination_reason_rows, f, indent=2)
        with open(out_dir / "termination_reasons.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TERMINATION_REASON_FIELDS)
            writer.writeheader()
            for row in termination_reason_rows:
                writer.writerow(row)
    if state_trace_rows is not None:
        with open(out_dir / "state_trace.json", "w", encoding="utf-8") as f:
            json.dump(state_trace_rows, f, indent=2)
        with open(out_dir / "state_trace.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STATE_TRACE_FIELDS)
            writer.writeheader()
            for row in state_trace_rows:
                writer.writerow(row)


def evaluate(args):
    _validate_eval_disturb_ratio(args)
    if args.cfg_override_json is not None and args.load_run is None:
        raise ValueError(
            "evaluate.py requires --load_run when --cfg_override_json is used, "
            "so the checkpoint is tied to the intended ablation run."
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _configure_eval_cfg(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.record_reward_terms = bool(args.record_reward_terms)
    env.record_eval_pre_reset_state = bool(args.record_state_trace)
    train_cfg.runner.resume = True
    # Evaluation artifacts belong under args.output_dir; disable runner log_dir so
    # checkpoint loading does not create train-style directories in logs/r2_amp.
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    policy = runner.get_inference_policy(device=env.device)

    rows = []
    reward_term_rows = []
    termination_reason_rows = []
    state_trace_rows = []
    for preset_name in _selected_presets(args):
        start_time = time.time()
        with torch.inference_mode():
            obs, critic_obs = env.reset()
        _apply_preset(env, preset_name)
        _apply_eval_disturbance(env, args)
        if hasattr(env, "standing_envs_mask"):
            env.standing_envs_mask[:] = False

        acc = _init_accumulators(env.num_envs, env.device)
        reward_acc = _init_reward_term_accumulators(env)
        traj_bufs = _init_traj_bufs(env.num_envs)
        state_trace = (
            _init_state_trace_buffers(env.num_envs, args.state_trace_window_steps)
            if args.record_state_trace
            else None
        )
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
                _collect_reward_terms(env, reward_acc)
                _collect_step_trajectories(env, traj_bufs)
                _append_state_trace(env, state_trace, preset_name, dones)
                _collect_disc_metrics(env, runner, infos, disc_agent_values, disc_ref_values)
                done_ids = _finalize_done_envs(
                    env,
                    dones,
                    infos,
                    acc,
                    episode_rows,
                    traj_bufs,
                    args.compute_dtw,
                    reward_acc,
                    args.record_termination_reasons,
                    state_trace,
                    state_trace_rows,
                    args,
                    train_cfg,
                    preset_name,
                    args.num_episodes,
                )
                _apply_preset(env, preset_name, done_ids)
                _apply_eval_disturbance(env, args, done_ids)
                if len(done_ids) > 0:
                    env.compute_observations(done_ids)
                    obs = env.get_observations()
                    critic_obs = env.get_privileged_observations()
                last_actions = actions.clone()
                if len(done_ids) > 0:
                    last_actions[done_ids] = 0.0

        preset_episode_rows = episode_rows[: args.num_episodes]
        rows.append(
            _summarize_preset(
                args,
                train_cfg,
                preset_name,
                preset_episode_rows,
                disc_agent_values,
                disc_ref_values,
                time.time() - start_time,
            )
        )
        if args.record_reward_terms:
            reward_term_rows.extend(
                _summarize_reward_terms(args, train_cfg, preset_name, preset_episode_rows)
            )
        if args.record_termination_reasons:
            termination_reason_rows.extend(
                _summarize_termination_reasons(
                    args,
                    train_cfg,
                    preset_name,
                    preset_episode_rows,
                )
            )

    _write_outputs(
        rows,
        args.output_dir,
        reward_term_rows if args.record_reward_terms else None,
        termination_reason_rows if args.record_termination_reasons else None,
        state_trace_rows if args.record_state_trace else None,
    )
    print(f"Saved evaluation metrics to {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    evaluate(get_args())
