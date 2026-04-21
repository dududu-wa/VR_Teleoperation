import os
import sys
import shutil
import subprocess
import tempfile
from datetime import datetime
sys.path.append(os.getcwd())
import isaacgym
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, get_load_path, task_registry
import numpy as np
import torch
import tqdm
from isaacgym import gymapi


DEMO_PRESETS = {
    "stand": {
        "name": "stand",
        "duration_s": 3.0,
        "commands": [0.0, 0.0, 0.0, 1.60, 0.5, 0.5, 0.08, 0.00, 0.00, 0.0],
    },
    "jump": {
        "name": "jump",
        "duration_s": 3.0,
        "commands": [0.0, 0.0, 0.0, 2.35, 0.0, 0.5, 0.20, 0.04, 0.00, 0.0],
    },
    "walk": {
        "name": "walk",
        "duration_s": 6.0,
        "commands": [0.70, 0.0, 0.0, 2.20, 0.5, 0.5, 0.12, 0.00, 0.00, 0.0],
    },
    "fast_walk": {
        "name": "fast_walk",
        "duration_s": 6.0,
        "commands": [1.05, 0.0, 0.0, 2.80, 0.5, 0.5, 0.17, 0.00, 0.03, 0.0],
    },
}
INITIAL_GAIT_NAME = os.environ.get("R2_PLAY_INITIAL_GAIT", "stand").strip().lower()
if INITIAL_GAIT_NAME not in DEMO_PRESETS:
    raise ValueError(
        f"Unsupported R2_PLAY_INITIAL_GAIT='{INITIAL_GAIT_NAME}'. "
        f"Expected one of: {', '.join(sorted(DEMO_PRESETS))}"
    )

DEMO_SEQUENCE_NAMES = []
for phase_name in (INITIAL_GAIT_NAME, "walk", "fast_walk"):
    if phase_name not in DEMO_SEQUENCE_NAMES:
        DEMO_SEQUENCE_NAMES.append(phase_name)
DEMO_SEQUENCE = tuple(DEMO_PRESETS[phase_name] for phase_name in DEMO_SEQUENCE_NAMES)

CAMERA_OFFSET = np.array([-2.5, -0.4, 1.15], dtype=np.float64)
LOOK_AT_OFFSET = np.array([0.6, 0.0, 0.85], dtype=np.float64)
RECORD_DURATION_S = 30.0
VIDEO_OUTPUT_DIR = os.path.join("logs", "play_videos")


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


def _sanitize_name(value):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(value))


def _resolve_record_output_root(train_cfg):
    if train_cfg.runner.resume:
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
        if train_cfg.runner.resume_path is None:
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
        else:
            resume_path = train_cfg.runner.resume_path
        return os.path.dirname(resume_path)
    return os.path.join(os.getcwd(), VIDEO_OUTPUT_DIR)


def _init_recording(args, env, output_root):
    if env.viewer is None:
        print("[record] viewer unavailable, skip mp4 recording")
        return None

    if not hasattr(env.gym, "write_viewer_image_to_file"):
        print("[record] gym.write_viewer_image_to_file unavailable, skip mp4 recording")
        return None

    os.makedirs(output_root, exist_ok=True)

    load_run = getattr(args, "load_run", None) or "latest"
    checkpoint = getattr(args, "checkpoint", None)
    checkpoint_name = "last" if checkpoint is None else str(checkpoint)
    clip_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{_sanitize_name(args.task)}_"
        f"{_sanitize_name(load_run)}_"
        f"ckpt_{_sanitize_name(checkpoint_name)}"
    )
    frame_dir = tempfile.mkdtemp(prefix=f"{clip_name}_frames_", dir=output_root)
    record_state = {
        "frame_dir": frame_dir,
        "video_path": os.path.abspath(os.path.join(output_root, f"{clip_name}.mp4")),
        "fps": max(int(round(1.0 / env.dt)), 1),
        "record_steps": max(int(np.ceil(RECORD_DURATION_S / env.dt)), 1),
        "next_frame_idx": 0,
        "captured_frames": 0,
        "disabled": False,
    }

    print(
        f"[record] capture first {RECORD_DURATION_S:.1f}s "
        f"({record_state['record_steps']} frames @ {record_state['fps']} fps) "
        f"to {record_state['video_path']}"
    )
    return record_state


def _get_ffmpeg_executable():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _capture_record_frame(env, record_state):
    if record_state is None or record_state["disabled"]:
        return
    if record_state["next_frame_idx"] >= record_state["record_steps"]:
        return

    frame_path = os.path.join(
        record_state["frame_dir"], f"frame_{record_state['next_frame_idx']:06d}.png"
    )

    try:
        if env.device != "cpu":
            env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.draw_viewer(env.viewer, env.sim, True)
        env.gym.write_viewer_image_to_file(env.viewer, frame_path)
        record_state["next_frame_idx"] += 1
        record_state["captured_frames"] += 1
    except Exception as exc:
        record_state["disabled"] = True
        print(f"[record] frame capture disabled: {exc}")


def _finalize_recording(record_state):
    if record_state is None:
        return
    if record_state["captured_frames"] == 0:
        shutil.rmtree(record_state["frame_dir"], ignore_errors=True)
        return

    ffmpeg_path = _get_ffmpeg_executable()
    if ffmpeg_path is None:
        print(
            "[record] no ffmpeg encoder found. "
            f"Captured frames kept at {record_state['frame_dir']}. "
            "Install ffmpeg or imageio-ffmpeg to auto-export mp4."
        )
        return

    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-framerate",
                str(record_state["fps"]),
                "-i",
                os.path.join(record_state["frame_dir"], "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                record_state["video_path"],
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"[record] saved mp4 to {record_state['video_path']}")
        shutil.rmtree(record_state["frame_dir"], ignore_errors=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        print(
            f"[record] failed to encode mp4 with ffmpeg ({stderr or exc}). "
            f"Captured frames kept at {record_state['frame_dir']}"
        )
    except Exception as exc:
        print(
            f"[record] failed to encode mp4 ({exc}). "
            f"Captured frames kept at {record_state['frame_dir']}"
        )


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
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
    record_output_root = _resolve_record_output_root(train_cfg)
    record_state = _init_recording(args, env, record_output_root)
    print(f"[demo] initial gait preset: {INITIAL_GAIT_NAME}")

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
    _capture_record_frame(env, record_state)

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
            _capture_record_frame(env, record_state)

            if record_state is not None and record_state["next_frame_idx"] >= record_state["record_steps"]:
                _finalize_recording(record_state)
                record_state = None

            if timestep % 200 == 0:
                base_lin = env.base_lin_vel[0].detach().cpu().numpy()
                cmd = env.commands[0, :3].detach().cpu().numpy()
                act_norm = torch.norm(actions[0]).item()
                print(
                    f"[demo] step={timestep} phase={current_phase_name} "
                    f"cmd(vx,vy,wz)={cmd} base_lin={base_lin} |a|={act_norm:.3f}"
                )

    _finalize_recording(record_state)

if __name__ == '__main__':
    args = get_args()
    play(args)
